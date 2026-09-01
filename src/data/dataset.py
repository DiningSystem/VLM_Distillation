import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn.functional import pad
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.arguments import DataArguments, ModelArguments


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IMAGE_TOKEN = "<image>"
SRE_MAX_SPANS = 1024
MAX_MULTIMODAL_TURNS = 3  # tối đa số cặp Q&A được tách từ 1 sample có ảnh


def _get_arg(args: Any, name: str, default=None):
    return getattr(args, name, default) if args is not None else default


def _longest_common_subsequence_offsets(a: torch.Tensor, b: torch.Tensor, s_i: int = 0, s_j: int = 0) -> List[tuple]:
    a_list = a.cpu().tolist()
    b_list = b.cpu().tolist()
    i, j = s_i, s_j
    result = []

    while i < len(a_list) and j < len(b_list):
        if a_list[i][1] == 0:
            i += 1
            continue
        if b_list[j][1] == 0:
            j += 1
            continue

        if a_list[i][1] == b_list[j][1]:
            result.append((i + 1, j + 1))
            i += 1
            j += 1
        elif a_list[i][1] < b_list[j][1]:
            i += 1
        else:
            j += 1

        if i + 1 < len(a_list) and j + 1 < len(b_list):
            if (a_list[i][1] == 0 and a_list[i + 1][1] > 0) or (b_list[j][1] == 0 and b_list[j + 1][1] > 0):
                while j < len(b_list) and b_list[j][1] > 0:
                    j += 1
                while i < len(a_list) and a_list[i][1] > 0:
                    i += 1
                i += 1
                j += 1
                result.append((i, j))

    if len(result) > SRE_MAX_SPANS:
        step = len(result) / SRE_MAX_SPANS
        return [result[int((idx + 1) * step) - 1] for idx in range(SRE_MAX_SPANS)]
    return result


def _get_pooler_tensor(segments_idxs: List[tuple]) -> Dict[str, torch.Tensor]:
    padded_idx_batch = []
    max_seg, max_len_all = 1, 1

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))
        if seg_idx:
            padded = torch.stack([pad(x, (0, max_len - len(x)), value=-1) for x in seg_idx])
        else:
            padded = torch.full((1, 1), -1, dtype=torch.long)
        padded_idx_batch.append(padded)

    def pad2d(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return pad(tensor, (0, width - tensor.size(1), 0, height - tensor.size(0)), value=-1)

    padded_idx_batch = torch.stack([pad2d(t, max_seg, max_len_all) for t in padded_idx_batch])
    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)
    return {"safe_idx": safe_idx.long(), "mask": mask.bool()}


def _first_supervised_token(labels: torch.Tensor, fallback_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    starts = []
    for row_idx in range(labels.shape[0]):
        valid = labels[row_idx].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if valid.numel() > 0:
            starts.append(valid[0])
            continue
        if fallback_mask is not None:
            fallback = fallback_mask[row_idx].bool().nonzero(as_tuple=False).flatten()
            if fallback.numel() > 0:
                starts.append(fallback[0])
                continue
        starts.append(torch.tensor(0, dtype=torch.long, device=labels.device))
    return torch.stack(starts).cpu()


def _prepare_sre_pooler(
    student_starts: torch.Tensor,
    student_offset_mapping: torch.Tensor,
    teacher_starts: torch.Tensor,
    teacher_offset_mapping: torch.Tensor,
) -> tuple:
    student_seg_idxs, teacher_seg_idxs = [], []
    for student_start, student_offset, teacher_start, teacher_offset in zip(
        student_starts,
        student_offset_mapping,
        teacher_starts,
        teacher_offset_mapping,
    ):
        student_start = int(student_start.item())
        teacher_start = int(teacher_start.item())
        common_offsets = [(student_start, teacher_start)] + _longest_common_subsequence_offsets(
            student_offset,
            teacher_offset,
            student_start,
            teacher_start,
        )

        student_seg_idx, teacher_seg_idx = [], []
        student_max_len, teacher_max_len = 1, 1
        for idx in range(1, len(common_offsets)):
            s_prev, t_prev = common_offsets[idx - 1]
            s_cur, t_cur = common_offsets[idx]
            if s_cur <= s_prev or t_cur <= t_prev:
                continue
            student_seg_idx.append(torch.arange(s_prev, s_cur, dtype=torch.long))
            teacher_seg_idx.append(torch.arange(t_prev, t_cur, dtype=torch.long))
            student_max_len = max(student_max_len, student_seg_idx[-1].numel())
            teacher_max_len = max(teacher_max_len, teacher_seg_idx[-1].numel())

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return _get_pooler_tensor(student_seg_idxs), _get_pooler_tensor(teacher_seg_idxs)


def _resolve_data_path(data_args: DataArguments, data_path: Optional[str] = None) -> str:
    if data_path:
        return data_path

    for name in ("data_path", "dataset_name", "dataset_config"):
        value = _get_arg(data_args, name)
        if value:
            return value

    raise ValueError(
        "A training json path is required. Pass data_path explicitly, or set one "
        "of data_args.data_path, data_args.dataset_name, or data_args.dataset_config."
    )


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported json root type: {type(data)!r}")
    except json.JSONDecodeError:
        samples = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"Expected a JSON object on line {line_no} in {path}")
            samples.append(sample)
        return samples


def _normalize_base_dir(data_path: str, data_args: Optional[DataArguments] = None) -> Path:
    image_dir = _get_arg(data_args, "image_dir")
    if image_dir:
        return Path(image_dir).expanduser()
    return Path(data_path).expanduser().resolve().parent


def _resolve_image_path(base_dir: Path, image_value: Any) -> Optional[str]:
    if image_value is None or image_value == "":
        return None
    if isinstance(image_value, (list, tuple)):
        if not image_value:
            return None
        image_value = image_value[0]
    # Guard: chỉ chấp nhận str/Path — int, float, bool đều là path rác
    if not isinstance(image_value, (str, Path)):
        return None
    image_path = Path(str(image_value)).expanduser()
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    return str(image_path)


def _iter_image_values(sample: Dict[str, Any]) -> List[Any]:
    """Return all explicitly declared image values from a JSON sample."""
    for key in ("image", "image_path", "images"):
        if key not in sample:
            continue
        value = sample.get(key)
        if value is None or value == "":
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return []


def _resolve_sample_image_paths(base_dir: Path, sample: Dict[str, Any]) -> List[str]:
    """Resolve every image path declared by a sample.

    The dataset currently attaches only one image to each conversation, but some
    JSON records can still contain a list of image paths.  We validate the whole
    declared list so a sample with any missing image is filtered before lazy
    loading reaches ``__getitem__``.
    """
    resolved_paths = []
    for image_value in _iter_image_values(sample):
        image_path = _resolve_image_path(base_dir, image_value)
        if image_path is not None:
            resolved_paths.append(image_path)
    return resolved_paths


_RESOLUTION_MAX_DIM: Dict[str, int] = {
    "high": 1344,
    "mid":   672,
    "low":   448,
}


def _resize_image_bounded(image: Image.Image, max_dim: int) -> Image.Image:
    """
    Resize *image* so its longest side is at most *max_dim*, preserving
    aspect ratio with LANCZOS resampling.  Returns the original object
    unchanged if it already fits.
    """
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), resample=Image.LANCZOS)


def _load_and_prepare_image(
    image_path: str,
    resolution: Optional[str] = None,
    min_side: int = 16,
) -> Image.Image:
    """
    Open *image_path*, convert to RGB, pad tiny images to *min_side* × *min_side*,
    then optionally down-scale to the cap defined by *resolution*.

    Args:
        image_path:  Absolute path to the image file.
        resolution:  One of ``"high"`` / ``"mid"`` / ``"low"`` (or ``None`` to
                     skip resizing).  Unknown strings fall back to no resize so
                     that new presets added to ``_RESOLUTION_MAX_DIM`` do not
                     silently break old code.
        min_side:    Minimum allowed width/height (default 16).  Images smaller
                     than this are centre-padded with black to avoid errors in
                     vision encoders that require a minimum patch size.

    Returns:
        A PIL ``Image`` in RGB mode, ready to pass to a processor.
    """
    image = Image.open(image_path).convert("RGB")

    # --- pad images that are too small for the vision encoder ---
    w, h = image.size
    if w < min_side or h < min_side:
        new_w = max(w, min_side)
        new_h = max(h, min_side)
        padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        padded.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
        image = padded

    # --- cap resolution to prevent uncontrolled VRAM usage ---
    if resolution is not None:
        max_dim = _RESOLUTION_MAX_DIM.get(resolution)
        if max_dim is not None:
            image = _resize_image_bounded(image, max_dim)
        # unknown resolution string → no resize (fail-safe, not fail-fast)

    return image


def _strip_image_token(value: str) -> str:
    return value.replace(IMAGE_TOKEN, "").strip()


def _message_content(role: str, text: str, image: Optional[Image.Image] = None):
    if role != "user" or image is None:
        return [{"type": "text", "text": text}]

    content = [{"type": "image", "image": image}]
    if text:
        content.append({"type": "text", "text": text})
    return content


def _normalize_conversations(conversations: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized = []
    for turn in conversations:
        speaker = turn.get("from") or turn.get("role")
        value = str(turn.get("value") if "value" in turn else turn.get("content", ""))
        if speaker in {"human", "user"}:
            role = "user"
            value = _strip_image_token(value)
        elif speaker in {"gpt", "assistant"}:
            role = "assistant"
            value = value.strip()
        elif speaker == "system":
            role = "system"
            value = value.strip()
        else:
            raise ValueError(f"Unsupported conversation speaker: {speaker!r}")
        normalized.append({"role": role, "value": value})
    return normalized

def _split_multimodal_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = sample["conversations"]
    system_turns = [t for t in conversations if t["role"] == "system"]
    qa_turns = [t for t in conversations if t["role"] != "system"]

    pairs: List[tuple] = []
    i = 0
    while i + 1 < len(qa_turns):
        if qa_turns[i]["role"] == "user" and qa_turns[i + 1]["role"] == "assistant":
            pairs.append((qa_turns[i], qa_turns[i + 1]))
            i += 2
        else:
            i += 1  

    if len(pairs) <= 1:
        return [sample]

    pairs = pairs[:MAX_MULTIMODAL_TURNS]

    sub_samples = []
    for idx, (user_turn, assistant_turn) in enumerate(pairs):
        sub_samples.append(
            {
                "id": f"{sample['id']}_t{idx}",
                "image_path": sample["image_path"],  
                "conversations": system_turns + [user_turn, assistant_turn],
            }
        )
    return sub_samples

class LazyVlmDistillDataset(Dataset):
    """
    Lazy dataset for VLM distillation.

    Reads LLaVA-style JSON/JSONL samples, skips samples whose image file is
    missing on disk, and keeps per-model processing out of __getitem__ so the
    collator can apply separate student and teacher processors.

    Multi-turn multimodal samples are split into independent sub-samples of at
    most MAX_MULTIMODAL_TURNS pairs each (see _split_multimodal_sample). Every
    sub-sample carries the image so it can stand alone without context from
    prior turns. Text-only samples and single-turn multimodal samples are kept
    as-is.

    Notes on `lengths` / `modality_lengths`:
        These are used only by LengthGroupedSampler for approximate batch
        bucketing — they do not need to be exact.  Image token counts vary
        wildly per architecture (e.g. dynamic tiling in Qwen2-VL vs fixed
        patch grids in LLaVA), so we deliberately exclude any image-token
        estimate here and leave that detail to the processor/collator layer.
        `modality_lengths` uses the sign convention expected by
        LengthGroupedSampler: positive = has image, negative = text-only.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        data_args: Optional[DataArguments] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        super().__init__()
        self.data_args = data_args
        self.data_path = _resolve_data_path(data_args, data_path)
        self.base_dir = _normalize_base_dir(self.data_path, data_args)
        self.system_prompt = system_prompt

        samples = _read_json_or_jsonl(self.data_path)
        percent_data = _get_arg(data_args, "percent_data", 1.0)
        if percent_data is not None and 0 < percent_data < 1:
            samples = samples[: max(1, int(len(samples) * percent_data))]

        self.list_data_dict = []
        self.skipped_missing_images = 0
        for sample_index, sample in enumerate(samples):
            image_paths = _resolve_sample_image_paths(self.base_dir, sample)
            # Skip multimodal samples as soon as any declared image file is
            # absent.  This prevents lazy PIL loading from crashing later and
            # also catches list-valued image fields where a non-first image is
            # missing.  Samples without an image field remain valid text-only
            # samples.
            if any(not os.path.isfile(image_path) for image_path in image_paths):
                self.skipped_missing_images += 1
                continue
            image_path = image_paths[0] if image_paths else None
            conversations = sample.get("conversations")
            if not conversations:
                continue

            base_sample = {
                # A stable fallback is required by candidate-cache based
                # distillation methods when source JSON does not provide ids.
                "id": sample.get("id", sample_index),
                "image_path": image_path,
                "conversations": _normalize_conversations(conversations),
            }
            for sub in _split_multimodal_sample(base_sample):
                self.list_data_dict.append(sub)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        """Approximate sequence length in whitespace-split tokens (text only)."""
        return [
            sum(len(turn["value"].split()) for turn in sample["conversations"])
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self) -> List[int]:
        """
        Positive → multimodal sample, negative → text-only sample.
        Magnitude is the approximate text-token count.
        """
        result = []
        for sample in self.list_data_dict:
            text_tokens = sum(len(turn["value"].split()) for turn in sample["conversations"])
            result.append(text_tokens if sample.get("image_path") else -text_tokens)
        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.list_data_dict[index]
        image = None
        if sample["image_path"]:
            resolution = _get_arg(self.data_args, "image_resolution")
            image = _load_and_prepare_image(sample["image_path"], resolution=resolution)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

        image_attached = False
        for turn in sample["conversations"]:
            role = turn["role"]
            turn_image = None
            if role == "user" and image is not None and not image_attached:
                turn_image = image
                image_attached = True
            messages.append(
                {
                    "role": role,
                    "content": _message_content(role, turn["value"], turn_image),
                }
            )

        return {
            "id": sample["id"],
            "messages": messages,
            "image_path": sample["image_path"],
        }


IGNORE_INDEX = -100


def _find_subsequence(row: List[int], pattern: List[int], start: int, end: int) -> int:
    if not pattern:
        return -1
    stop = end - len(pattern)
    i = start
    while i <= stop:
        if row[i] == pattern[0] and row[i : i + len(pattern)] == pattern:
            return i
        i += 1
    return -1


def _unique_token_patterns(tokenizer, patterns: Sequence[str]) -> List[List[int]]:
    token_patterns = []
    seen = set()
    for pattern in patterns:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        token_patterns.append(ids)
    return token_patterns


def _valid_token_bounds(attention_mask: Optional[torch.Tensor], row_idx: int, seq_len: int) -> tuple:
    if attention_mask is None:
        return 0, seq_len

    valid = attention_mask[row_idx].bool().nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0, 0
    return int(valid[0].item()), int(valid[-1].item()) + 1


def _make_labels_chatml(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build assistant-only labels for ChatML-style VLM chat templates.

    FastVLM, Qwen2/2.5/3-VL, and LLaVA-OV templates used in this repo all mark
    turns with ``<|im_start|>{role}`` and close them with ``<|im_end|>``.  The
    separator after ``assistant`` differs by template, so we search several
    tokenized header variants instead of assuming the Qwen newline form only.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    assistant_header_patterns = _unique_token_patterns(
        tok,
        (
            "<|im_start|>assistant\n",
            "<|im_start|>assistant ",
            "<|im_start|>assistant",
        ),
    )
    im_end_patterns = _unique_token_patterns(tok, ("<|im_end|>",))
    if not assistant_header_patterns or not im_end_patterns:
        raise RuntimeError("Tokenizer cannot encode ChatML assistant/end markers.")

    batch_size, seq_len = input_ids.shape
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    ids_list = input_ids.tolist()

    for b, row in enumerate(ids_list):
        valid_start, valid_end = _valid_token_bounds(attention_mask, b, seq_len)
        i = valid_start
        while i < valid_end:
            matched_header = None
            for header_ids in assistant_header_patterns:
                header_end = i + len(header_ids)
                if header_end <= valid_end and row[i:header_end] == header_ids:
                    matched_header = header_ids
                    break
            if matched_header is None:
                i += 1
                continue

            content_start = i + len(matched_header)
            end_pos, end_len = -1, 0
            for im_end_ids in im_end_patterns:
                found = _find_subsequence(row, im_end_ids, content_start, valid_end)
                if found != -1 and (end_pos == -1 or found < end_pos):
                    end_pos = found
                    end_len = len(im_end_ids)

            if end_pos == -1:
                labels[b, content_start:valid_end] = input_ids[b, content_start:valid_end]
                i = valid_end
            else:
                labels[b, content_start : end_pos + end_len] = input_ids[b, content_start : end_pos + end_len]
                i = end_pos + end_len

    return labels


def _make_labels_qwen(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _make_labels_chatml(input_ids, tokenizer, attention_mask)


@dataclass
class VlmDistillDataCollator:
    student_processor: Any
    data_args: Optional[DataArguments] = None
    model_args: Optional[ModelArguments] = None
    teacher_processor: Any = None
    add_generation_prompt: bool = False

    def __post_init__(self):
        self.max_length = _get_arg(self.data_args, "max_len")
        has_teacher_arg = bool(_get_arg(self.model_args, "teacher_model_name"))
        self.use_teacher = self.teacher_processor is not None and has_teacher_arg
        # kd_loss_type lives on TrainingArguments, which this collator does not
        # receive.  train.py sets this attribute on data_args before building
        # the data module.
        kd_loss_type = _get_arg(self.data_args, "kd_loss_type")
        self.use_sre_pooler = bool(
            kd_loss_type in {"sre", "joint", "unit_aligned", "unit_aligned_distillation"}
            and self.use_teacher
        )

    def _apply_processor(
        self,
        processor,
        conversations: List[List[Dict[str, Any]]],
        return_offsets_mapping: bool = False,
    ) -> Dict[str, Any]:
        if not hasattr(processor, "apply_chat_template"):
            raise RuntimeError(
                "The processor must support apply_chat_template for VLM_Distill chat data."
            )

        kwargs = {
            "tokenize": True,
            "add_generation_prompt": self.add_generation_prompt,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if return_offsets_mapping:
            kwargs["return_offsets_mapping"] = True
        if self.max_length is not None:
            kwargs.update({"max_length": self.max_length, "truncation": True})

        try:
            inputs = processor.apply_chat_template(conversations, **kwargs)
        except (TypeError, ValueError):
            if not return_offsets_mapping:
                raise
            kwargs.pop("return_offsets_mapping", None)
            inputs = processor.apply_chat_template(conversations, **kwargs)
        for key in ("input_ids", "attention_mask"):
            if key in inputs and torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].long()
        return inputs

    def _add_labels(self, inputs: Dict[str, Any], processor: Any = None) -> None:
        """
        Attach ``labels`` to *inputs* in-place.

        Positions that are not part of an assistant response are masked with
        IGNORE_INDEX so the LM cross-entropy loss is computed only on the
        tokens the model is supposed to generate.

        ``processor`` defaults to the student processor; pass the teacher
        processor when labelling teacher_inputs (their tokenizers can split
        the assistant region differently and we need the indices to live in
        each side's own coordinate system).
        """
        processor = processor or self.student_processor
        inputs["labels"] = _make_labels_chatml(
            inputs["input_ids"], processor, inputs.get("attention_mask")
        )

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        conversations = [instance["messages"] for instance in instances]
        student_inputs = self._apply_processor(
            self.student_processor,
            conversations,
            return_offsets_mapping=self.use_sre_pooler,
        )
        student_offsets = student_inputs.pop("offset_mapping", None)
        self._add_labels(student_inputs)

        batch = {
            "ids": [instance["id"] for instance in instances],
            "image_paths": [instance["image_path"] for instance in instances],
            "student_inputs": student_inputs,
        }

        # Keep the original multimodal conversations as criterion metadata.
        # UGRIMD rebuilds candidate-specific teacher-forcing inputs with each
        # model's own processor; these must never be inferred from token IDs.
        if _get_arg(self.data_args, "kd_loss_type") == "ugrimd":
            batch["candidate_messages"] = conversations

        if self.use_teacher:
            teacher_inputs = self._apply_processor(
                self.teacher_processor,
                conversations,
                return_offsets_mapping=self.use_sre_pooler,
            )
            teacher_offsets = teacher_inputs.pop("offset_mapping", None)

            # Always attach labels to teacher_inputs. SCVA, CGKD, and any future
            # criterion that wants to gate by "what was the teacher generating
            # at this position" needs them. Previously only the SRE pooler
            # branch computed them, leaving SCVA's response gate empty for
            # `scva` / `cgkd` / `scva_cgkd` kd_loss_types and silently zeroing
            # the SCVA term.
            self._add_labels(teacher_inputs, processor=self.teacher_processor)

            if self.use_sre_pooler and student_offsets is not None and teacher_offsets is not None:
                student_starts = _first_supervised_token(student_inputs["labels"], student_inputs.get("attention_mask"))
                teacher_starts = _first_supervised_token(teacher_inputs["labels"], teacher_inputs.get("attention_mask"))
                student_pooler, teacher_pooler = _prepare_sre_pooler(
                    student_starts,
                    student_offsets,
                    teacher_starts,
                    teacher_offsets,
                )
                student_inputs["pooler_safe_idx"] = student_pooler["safe_idx"]
                student_inputs["pooler_mask"] = student_pooler["mask"]
                teacher_inputs["pooler_safe_idx"] = teacher_pooler["safe_idx"]
                teacher_inputs["pooler_mask"] = teacher_pooler["mask"]

            batch["teacher_inputs"] = teacher_inputs

        return batch


def make_vlm_distill_data_module(
    student_processor,
    data_args: DataArguments,
    model_args: Optional[ModelArguments] = None,
    teacher_processor=None,
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    train_dataset = LazyVlmDistillDataset(data_path=data_path, data_args=data_args)
    data_collator = VlmDistillDataCollator(
        student_processor=student_processor,
        teacher_processor=teacher_processor,
        data_args=data_args,
        model_args=model_args,
    )
    return {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }


# Aliases for backward compatibility
VlmDistillDataset = LazyVlmDistillDataset
DataCollatorForVlmDistill = VlmDistillDataCollator
make_supervised_data_module = make_vlm_distill_data_module
