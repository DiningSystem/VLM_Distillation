"""Uncertainty-Gated Ranking + Multimodal Interaction Distillation (UGRIMD).

Teacher and student only meet through answer *strings*.  Each model scores the
same strings with its own tokenizer and LM head, so this criterion never aligns
vocabularies, logits, token positions, or hidden-state dimensions.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import _make_labels_chatml


IGNORE_INDEX = -100


class UGRIMDCriterion(nn.Module):
    """Stage-2 CE plus hard-sample ranking and visual-shift matching.

    Stage 1 is the repository's normal SFT path.  Stage 2 keeps CE active,
    mines examples from student candidate entropy, then uses detached teacher
    uncertainty to route selected examples to ranking or interaction loss.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distill_lambda = float(getattr(args, "distill_lambda", 1.0))
        self.interaction_lambda = float(getattr(args, "interaction_lambda", 1.0))
        self.num_candidates = max(1, int(getattr(args, "num_candidates", 8)))
        self.regenerate_steps = max(1, int(getattr(args, "candidate_regenerate_steps", 500)))
        self.pair_top_percent = min(100.0, max(0.0, float(getattr(args, "uncertainty_top_percent", 30.0))))
        self.hard_sample_top_percent = min(100.0, max(0.0, float(getattr(args, "hard_sample_top_percent", 30.0))))
        self.hard_sample_refresh_steps = max(1, int(getattr(args, "hard_sample_refresh_steps", 1000)))
        self.exploration_percent = min(100.0, max(0.0, float(getattr(args, "exploration_percent", 10.0))))
        self.rank_temperature = max(float(getattr(args, "rank_temperature", 0.1)), 1e-6)
        self.rank_epsilon = max(float(getattr(args, "rank_epsilon", 1e-8)), 1e-12)
        self.visual_attention_layers = tuple(int(layer) for layer in getattr(args, "visual_attention_layers", [1, 2, 3]))
        self.visual_weight_temperature = max(float(getattr(args, "visual_weight_temperature", 0.1)), 1e-6)
        self.visual_weight_floor = min(1.0, max(0.0, float(getattr(args, "visual_weight_floor", 0.25))))
        self.candidate_temperature = max(float(getattr(args, "candidate_temperature", 1.0)), 1e-6)
        self.interaction_temperature = max(float(getattr(args, "interaction_temperature", 1.0)), 1e-6)
        self.strength_temperature = max(float(getattr(args, "interaction_strength_temperature", 0.1)), 1e-6)
        self.loss_epsilon = max(float(getattr(args, "loss_epsilon", 1e-8)), 1e-12)
        self.global_step = 0
        self._last_regeneration_step = None
        self.candidate_cache: Dict[str, List[str]] = {}
        self.teacher_candidate_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.hard_sample_cache: Dict[str, bool] = {}
        self._last_hard_refresh_step = None

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    @staticmethod
    def _answer(messages: Sequence[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return "".join(
                    part.get("text", "") for part in message.get("content", []) if part.get("type") == "text"
                ).strip()
        raise ValueError("UGRIMD requires an assistant answer in every training conversation.")

    @staticmethod
    def _with_candidate(messages: Sequence[Dict[str, Any]], candidate: str, language_only: bool) -> List[Dict[str, Any]]:
        result = copy.deepcopy(list(messages))
        assistant_index = next((i for i in range(len(result) - 1, -1, -1) if result[i].get("role") == "assistant"), None)
        if assistant_index is None:
            raise ValueError("UGRIMD requires an assistant answer in every training conversation.")
        result[assistant_index]["content"] = [{"type": "text", "text": candidate}]
        if language_only:
            for message in result:
                message["content"] = [part for part in message.get("content", []) if part.get("type") != "image"]
        return result

    @staticmethod
    def _prompt(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = copy.deepcopy(list(messages))
        assistant_index = next((i for i in range(len(prompt) - 1, -1, -1) if prompt[i].get("role") == "assistant"), None)
        if assistant_index is None:
            raise ValueError("UGRIMD requires an assistant answer in every training conversation.")
        return prompt[:assistant_index]

    @staticmethod
    def _processor_inputs(processor, conversations: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
        inputs = processor.apply_chat_template(
            list(conversations), tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", padding=True,
        )
        for key in ("input_ids", "attention_mask"):
            if key in inputs and torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].long()
        tokenizer = getattr(processor, "tokenizer", processor)
        inputs["labels"] = _make_labels_chatml(inputs["input_ids"], tokenizer, inputs.get("attention_mask"))
        return inputs

    @staticmethod
    def _to_model_device(inputs: Dict[str, Any], model) -> Dict[str, Any]:
        """Candidate inputs bypass Trainer's recursive device preparation."""
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return inputs
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    @staticmethod
    def _log_likelihood(outputs, labels: torch.Tensor) -> torch.Tensor:
        logits = outputs.logits[:, :-1].float()
        targets = labels[:, 1:]
        valid = targets.ne(IGNORE_INDEX)
        safe_targets = targets.masked_fill(~valid, 0)
        log_probs = F.log_softmax(logits, dim=-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        return (log_probs * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)

    def _generate_candidates(self, distiller: Any, messages: Sequence[Dict[str, Any]], answer: str) -> List[str]:
        if self.num_candidates == 1:
            return [answer]
        processor = distiller.get_student_processor()
        prompt_inputs = processor.apply_chat_template(
            [self._prompt(messages)] * (self.num_candidates - 1), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        prompt_inputs = self._to_model_device(prompt_inputs, distiller.student)
        with torch.no_grad():
            generated = distiller.student.generate(**prompt_inputs, max_new_tokens=64, do_sample=True)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        tokenizer = getattr(processor, "tokenizer", processor)
        decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        # Preserve K candidate slots even when generation repeats or is empty;
        # the answer remains candidate 0 as required.
        return [answer] + [(text.strip() or answer) for text in decoded[: self.num_candidates - 1]] + [answer] * max(0, self.num_candidates - 1 - len(decoded))

    def _candidates(self, distiller: Any, ids: Sequence[Any], messages: Sequence[Sequence[Dict[str, Any]]]) -> List[List[str]]:
        regenerate = (
            self.global_step % self.regenerate_steps == 0
            and self._last_regeneration_step != self.global_step
        )
        candidates = []
        for index, (sample_id, sample_messages) in enumerate(zip(ids, messages)):
            key = str(sample_id if sample_id is not None else f"batch-{index}")
            if regenerate or key not in self.candidate_cache:
                self.candidate_cache[key] = self._generate_candidates(distiller, sample_messages, self._answer(sample_messages))
                self.teacher_candidate_cache.pop(key, None)
            candidates.append(self.candidate_cache[key])
        if regenerate:
            self._last_regeneration_step = self.global_step
        return candidates

    def _hard_sample_mask(self, ids: Sequence[Any], student_vl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select difficult samples from detached student candidate entropy.

        Selection is deliberately discrete and cached by sample id: student
        uncertainty determines *which* examples receive teacher forwards, not
        a differentiable weight inside either teacher-guided loss.  Samples
        first encountered between global refreshes are evaluated once and then
        retain that decision until the next refresh.
        """
        candidate_count = student_vl.shape[-1]
        student_probs = torch.softmax(student_vl.detach() / self.candidate_temperature, dim=-1)
        uncertainty = -(student_probs * (student_probs + self.loss_epsilon).log()).sum(dim=-1)
        uncertainty = uncertainty / math.log(max(candidate_count, 2))
        refresh = (
            self.global_step % self.hard_sample_refresh_steps == 0
            and self._last_hard_refresh_step != self.global_step
        )
        keys = [str(sample_id if sample_id is not None else f"batch-{index}") for index, sample_id in enumerate(ids)]
        update_indices = [index for index, key in enumerate(keys) if refresh or key not in self.hard_sample_cache]
        if update_indices:
            update_tensor = torch.tensor(update_indices, device=uncertainty.device, dtype=torch.long)
            update_uncertainty = uncertainty.index_select(0, update_tensor)
            selected = self._top_mask(update_uncertainty, self.hard_sample_top_percent)
            remaining = (~selected).nonzero(as_tuple=False).flatten()
            explore_count = min(remaining.numel(), math.ceil(remaining.numel() * self.exploration_percent / 100.0))
            if explore_count:
                selected[remaining[torch.randperm(remaining.numel(), device=remaining.device)[:explore_count]]] = True
            for local_index, batch_index in enumerate(update_indices):
                self.hard_sample_cache[keys[batch_index]] = bool(selected[local_index].item())
        if refresh:
            self._last_hard_refresh_step = self.global_step
        mask = torch.tensor([self.hard_sample_cache[key] for key in keys], device=student_vl.device, dtype=torch.bool)
        return mask, uncertainty

    def _scores(self, model, processor, messages, candidates, language_only: bool, requires_grad: bool) -> torch.Tensor:
        flat_messages = [self._with_candidate(sample, candidate, language_only) for sample, sample_candidates in zip(messages, candidates) for candidate in sample_candidates]
        inputs = self._processor_inputs(processor, flat_messages)
        inputs = self._to_model_device(inputs, model)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            scores = self._log_likelihood(model(**inputs), inputs["labels"])
        return scores.reshape(len(candidates), self.num_candidates)

    def _visual_grounding(self, teacher_outputs, labels: torch.Tensor) -> torch.Tensor:
        """Return detached response-to-vision attention ratios per candidate.

        Each requested layer contributes the mean-over-heads attention from
        assistant-response query tokens to vision-key tokens.  Backbones that
        cannot provide a compatible attention/mask safely fall back to zero,
        making the configured floor the only visual reliability contribution.
        """
        vision_mask = getattr(teacher_outputs, "vision_feature_mask", None)
        attentions = getattr(teacher_outputs, "attentions", None)
        result = labels.new_zeros(labels.shape[0], dtype=torch.float32)
        if vision_mask is None or attentions is None:
            return result

        layer_ratios = []
        for layer_number in self.visual_attention_layers:
            layer_index = max(layer_number - 1, 0)
            try:
                attention = attentions[layer_index]
            except (IndexError, TypeError):
                continue
            if attention is None or attention.ndim != 4:
                continue
            sequence_length = min(attention.shape[-2], attention.shape[-1], labels.shape[1], vision_mask.shape[1])
            if sequence_length == 0:
                continue
            attention = attention[:, :, :sequence_length, :sequence_length].float().mean(dim=1)
            response_mask = labels[:, :sequence_length].ne(IGNORE_INDEX)
            vision_keys = vision_mask[:, :sequence_length].to(device=attention.device, dtype=attention.dtype)
            vision_attention = (attention * vision_keys[:, None, :]).sum(dim=-1)
            attention_ratio = vision_attention / (attention.sum(dim=-1) + self.loss_epsilon)
            layer_ratios.append(
                (attention_ratio * response_mask.to(dtype=attention_ratio.dtype)).sum(dim=-1)
                / response_mask.sum(dim=-1).clamp_min(1)
            )
        if not layer_ratios:
            return result
        return torch.stack(layer_ratios, dim=0).mean(dim=0).detach()

    def _teacher_vl_scores_and_grounding(self, model, processor, messages, candidates) -> tuple[torch.Tensor, torch.Tensor]:
        flat_messages = [
            self._with_candidate(sample, candidate, language_only=False)
            for sample, sample_candidates in zip(messages, candidates)
            for candidate in sample_candidates
        ]
        inputs = self._to_model_device(self._processor_inputs(processor, flat_messages), model)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = self._log_likelihood(outputs, inputs["labels"])
            grounding = self._visual_grounding(outputs, inputs["labels"])
        return scores.reshape(len(candidates), self.num_candidates), grounding.reshape(len(candidates), self.num_candidates)

    def _cached_teacher_scores(self, distiller: Any, ids, messages, candidates, student_device: torch.device):
        """Cache detached teacher candidate scores and visual grounding by id."""
        keys = [str(sample_id if sample_id is not None else f"batch-{index}") for index, sample_id in enumerate(ids)]
        missing = [index for index, key in enumerate(keys) if key not in self.teacher_candidate_cache]
        if missing:
            missing_messages = [messages[index] for index in missing]
            missing_candidates = [candidates[index] for index in missing]
            vl_scores, grounding = self._teacher_vl_scores_and_grounding(
                distiller.teacher, distiller.get_teacher_processor(), missing_messages, missing_candidates
            )
            l_scores = self._scores(
                distiller.teacher, distiller.get_teacher_processor(), missing_messages, missing_candidates,
                language_only=True, requires_grad=False,
            )
            for row, index in enumerate(missing):
                self.teacher_candidate_cache[keys[index]] = {
                    "vl_scores": vl_scores[row].detach().cpu(),
                    "l_scores": l_scores[row].detach().cpu(),
                    "grounding": grounding[row].detach().cpu(),
                }
        return tuple(
            torch.stack([self.teacher_candidate_cache[key][name] for key in keys]).to(student_device)
            for name in ("vl_scores", "l_scores", "grounding")
        )

    @staticmethod
    def _binary_entropy(probability: torch.Tensor, eps: float) -> torch.Tensor:
        return -(probability * (probability + eps).log() + (1 - probability) * (1 - probability + eps).log()) / math.log(2)

    @staticmethod
    def _top_mask(values: torch.Tensor, percent: float) -> torch.Tensor:
        mask = torch.zeros_like(values, dtype=torch.bool)
        if values.numel() == 0 or percent <= 0:
            return mask
        count = min(values.numel(), max(1, math.ceil(values.numel() * percent / 100.0)))
        mask[values.topk(count).indices] = True
        return mask

    def compute_score_losses(self, teacher_vl, student_vl, teacher_l, student_l, teacher_grounding):
        """Compute the two independent objectives from [B, K] candidate scores."""
        eps = self.rank_epsilon
        _bsz, candidate_count = teacher_vl.shape
        pair_i, pair_j = torch.triu_indices(candidate_count, candidate_count, offset=1, device=teacher_vl.device)
        teacher_margin = (teacher_vl[:, pair_i] - teacher_vl[:, pair_j]).detach()
        student_margin = student_vl[:, pair_i] - student_vl[:, pair_j]
        q_teacher = torch.sigmoid(teacher_margin / self.rank_temperature)
        q_student = torch.sigmoid(student_margin / self.rank_temperature)
        teacher_probs_vl = torch.softmax(teacher_vl.detach() / self.candidate_temperature, dim=-1)
        teacher_uncertainty = -(teacher_probs_vl * (teacher_probs_vl + self.loss_epsilon).log()).sum(dim=-1)
        teacher_uncertainty = teacher_uncertainty / math.log(max(candidate_count, 2))
        teacher_certainty = 1 - teacher_uncertainty
        visual_margin = (teacher_grounding[:, pair_i] - teacher_grounding[:, pair_j]).detach()
        visual_agreement = torch.sigmoid(
            (teacher_margin * visual_margin) / self.visual_weight_temperature
        ).detach()
        visual_reliability = (
            self.visual_weight_floor
            + (1 - self.visual_weight_floor) * visual_agreement
        ).detach()
        # Pair supervision is teacher-certain + student-uncertain, additionally
        # modulated by sample-level certainty and vision-aware teacher
        # reliability. This is a weight only; it creates no visual loss.
        rank_gate = (
            teacher_certainty[:, None]
            * (1 - self._binary_entropy(q_teacher, eps))
            * self._binary_entropy(q_student, eps)
            * visual_reliability
        )
        selected_pairs = self._top_mask(rank_gate.detach().flatten(), self.pair_top_percent).reshape_as(rank_gate)
        rank_gate = rank_gate * selected_pairs
        rank_bce = -(q_teacher * (q_student + eps).log() + (1 - q_teacher) * (1 - q_student + eps).log())
        rank_per_sample = (rank_gate * rank_bce).sum(dim=-1) / (rank_gate.sum(dim=-1) + self.loss_epsilon)
        rank_loss = (teacher_certainty * rank_per_sample).mean()

        p_teacher_vl = teacher_probs_vl
        p_teacher_l = torch.softmax(teacher_l.detach() / self.candidate_temperature, dim=-1)
        p_student_vl = torch.softmax(student_vl / self.candidate_temperature, dim=-1)
        p_student_l = torch.softmax(student_l / self.candidate_temperature, dim=-1)
        teacher_shift = (p_teacher_vl - p_teacher_l).detach()
        student_shift = p_student_vl - p_student_l
        strength = 0.5 * teacher_shift.abs().sum(dim=-1)
        # Teacher uncertainty chooses interaction; student uncertainty has
        # already selected this sample and is not used as a loss weight.
        interaction_gate = teacher_uncertainty * strength / (strength + self.strength_temperature)
        interaction_per_sample = F.smooth_l1_loss(student_shift, teacher_shift, reduction="none", beta=self.interaction_temperature).mean(dim=-1)
        interaction_loss = (interaction_per_sample * interaction_gate).sum() / (interaction_gate.sum() + self.loss_epsilon)
        return rank_loss, interaction_loss, teacher_uncertainty.mean(), interaction_gate.sum(), visual_reliability.mean()

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_outputs = distiller.student(**batch["student_inputs"])
        ce_loss = student_outputs.loss
        if ce_loss is None:
            raise RuntimeError("Student model did not return supervised CE loss.")
        messages = batch.get("candidate_messages")
        if messages is None:
            raise RuntimeError("UGRIMD requires candidate_messages from VlmDistillDataCollator.")
        candidates = self._candidates(distiller, batch["ids"], messages)
        student_processor = distiller.get_student_processor()
        student_vl = self._scores(distiller.student, student_processor, messages, candidates, False, True)
        hard_mask, student_uncertainty = self._hard_sample_mask(batch["ids"], student_vl)
        hard_indices = hard_mask.nonzero(as_tuple=False).flatten()
        if hard_indices.numel() == 0:
            zero = ce_loss.new_zeros(())
            return {"loss": ce_loss, "supervised_loss": ce_loss.detach(), "rank_loss": zero, "interaction_loss": zero, "visual_ranking_weight": zero, "student_uncertainty": student_uncertainty.mean().detach(), "teacher_uncertainty": zero, "hard_sample_count": zero, "active_interaction_samples": zero}
        selected_indices = hard_indices.tolist()
        selected_messages = [messages[index] for index in selected_indices]
        selected_candidates = [candidates[index] for index in selected_indices]
        selected_ids = [batch["ids"][index] for index in selected_indices]
        student_vl = student_vl.index_select(0, hard_indices)
        student_l = self._scores(distiller.student, student_processor, selected_messages, selected_candidates, True, True)
        teacher_vl, teacher_l, teacher_grounding = self._cached_teacher_scores(
            distiller, selected_ids, selected_messages, selected_candidates, student_vl.device
        )
        rank_loss, interaction_loss, teacher_uncertainty, active_interaction, visual_reliability = self.compute_score_losses(
            teacher_vl, student_vl, teacher_l, student_l, teacher_grounding
        )
        total = ce_loss + self.distill_lambda * (rank_loss + self.interaction_lambda * interaction_loss)
        return {"loss": total, "supervised_loss": ce_loss.detach(), "rank_loss": rank_loss.detach(), "interaction_loss": interaction_loss.detach(), "visual_ranking_weight": visual_reliability.detach(), "student_uncertainty": student_uncertainty.mean().detach(), "teacher_uncertainty": teacher_uncertainty.detach(), "hard_sample_count": hard_mask.sum().detach(), "active_interaction_samples": active_interaction.detach()}
