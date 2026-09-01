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
    """CE plus within-sample ranking and sample-level visual-shift matching."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distill_lambda = float(getattr(args, "distill_lambda", 1.0))
        self.interaction_lambda = float(getattr(args, "interaction_lambda", 1.0))
        self.num_candidates = max(1, int(getattr(args, "num_candidates", 8)))
        self.regenerate_steps = max(1, int(getattr(args, "candidate_regenerate_steps", 500)))
        self.top_percent = min(100.0, max(0.0, float(getattr(args, "uncertainty_top_percent", 30.0))))
        self.exploration_percent = min(100.0, max(0.0, float(getattr(args, "exploration_percent", 10.0))))
        self.rank_temperature = max(float(getattr(args, "rank_temperature", 0.1)), 1e-6)
        self.rank_epsilon = max(float(getattr(args, "rank_epsilon", 1e-8)), 1e-12)
        self.candidate_temperature = max(float(getattr(args, "candidate_temperature", 1.0)), 1e-6)
        self.interaction_temperature = max(float(getattr(args, "interaction_temperature", 1.0)), 1e-6)
        self.strength_temperature = max(float(getattr(args, "interaction_strength_temperature", 0.1)), 1e-6)
        self.loss_epsilon = max(float(getattr(args, "loss_epsilon", 1e-8)), 1e-12)
        self.global_step = 0
        self._last_regeneration_step = None
        self.candidate_cache: Dict[str, List[str]] = {}

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
            candidates.append(self.candidate_cache[key])
        if regenerate:
            self._last_regeneration_step = self.global_step
        return candidates

    def _scores(self, model, processor, messages, candidates, language_only: bool, requires_grad: bool) -> torch.Tensor:
        flat_messages = [self._with_candidate(sample, candidate, language_only) for sample, sample_candidates in zip(messages, candidates) for candidate in sample_candidates]
        inputs = self._processor_inputs(processor, flat_messages)
        inputs = self._to_model_device(inputs, model)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            scores = self._log_likelihood(model(**inputs), inputs["labels"])
        return scores.reshape(len(candidates), self.num_candidates)

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

    def compute_score_losses(self, teacher_vl, student_vl, teacher_l, student_l):
        """Compute the two independent objectives from [B, K] candidate scores."""
        eps = self.rank_epsilon
        bsz, candidate_count = teacher_vl.shape
        pair_i, pair_j = torch.triu_indices(candidate_count, candidate_count, offset=1, device=teacher_vl.device)
        teacher_margin = (teacher_vl[:, pair_i] - teacher_vl[:, pair_j]).detach()
        student_margin = student_vl[:, pair_i] - student_vl[:, pair_j]
        q_teacher = torch.sigmoid(teacher_margin / self.rank_temperature)
        q_student = torch.sigmoid(student_margin / self.rank_temperature)
        rank_gate = (1 - self._binary_entropy(q_teacher, eps)) * self._binary_entropy(q_student, eps)
        selected_pairs = self._top_mask(rank_gate.detach().flatten(), self.top_percent).reshape_as(rank_gate)
        rank_gate = rank_gate * selected_pairs
        rank_bce = -(q_teacher * (q_student + eps).log() + (1 - q_teacher) * (1 - q_student + eps).log())
        rank_loss = ((rank_gate * rank_bce).sum(dim=-1) / (rank_gate.sum(dim=-1) + self.loss_epsilon)).mean()

        p_teacher_vl = torch.softmax(teacher_vl.detach() / self.candidate_temperature, dim=-1)
        p_teacher_l = torch.softmax(teacher_l.detach() / self.candidate_temperature, dim=-1)
        p_student_vl = torch.softmax(student_vl / self.candidate_temperature, dim=-1)
        p_student_l = torch.softmax(student_l / self.candidate_temperature, dim=-1)
        uncertainty = -(p_teacher_vl * (p_teacher_vl + self.loss_epsilon).log()).sum(dim=-1) / math.log(max(candidate_count, 2))
        uncertainty_mask = self._top_mask(uncertainty.detach(), self.top_percent)
        remaining = (~uncertainty_mask).nonzero(as_tuple=False).flatten()
        explore_count = min(remaining.numel(), math.ceil(remaining.numel() * self.exploration_percent / 100.0))
        if explore_count:
            uncertainty_mask[remaining[torch.randperm(remaining.numel(), device=remaining.device)[:explore_count]]] = True
        teacher_shift = (p_teacher_vl - p_teacher_l).detach()
        student_shift = p_student_vl - p_student_l
        strength = 0.5 * teacher_shift.abs().sum(dim=-1)
        interaction_gate = uncertainty_mask.to(strength.dtype) * strength / (strength + self.strength_temperature)
        interaction_per_sample = F.smooth_l1_loss(student_shift, teacher_shift, reduction="none", beta=self.interaction_temperature).mean(dim=-1)
        interaction_loss = (interaction_per_sample * interaction_gate).sum() / (interaction_gate.sum() + self.loss_epsilon)
        return rank_loss, interaction_loss, uncertainty.mean(), interaction_gate.sum()

    def forward(self, distiller: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        student_outputs = distiller.student(**batch["student_inputs"])
        ce_loss = student_outputs.loss
        if ce_loss is None:
            raise RuntimeError("Student model did not return supervised CE loss.")
        messages = batch.get("candidate_messages")
        if messages is None:
            raise RuntimeError("UGRIMD requires candidate_messages from VlmDistillDataCollator.")
        candidates = self._candidates(distiller, batch["ids"], messages)
        student_processor, teacher_processor = distiller.get_student_processor(), distiller.get_teacher_processor()
        student_vl = self._scores(distiller.student, student_processor, messages, candidates, False, True)
        student_l = self._scores(distiller.student, student_processor, messages, candidates, True, True)
        teacher_vl = self._scores(distiller.teacher, teacher_processor, messages, candidates, False, False)
        teacher_l = self._scores(distiller.teacher, teacher_processor, messages, candidates, True, False)
        rank_loss, interaction_loss, teacher_uncertainty, active_interaction = self.compute_score_losses(teacher_vl, student_vl, teacher_l, student_l)
        total = ce_loss + self.distill_lambda * (rank_loss + self.interaction_lambda * interaction_loss)
        return {"loss": total, "supervised_loss": ce_loss.detach(), "rank_loss": rank_loss.detach(), "interaction_loss": interaction_loss.detach(), "teacher_uncertainty": teacher_uncertainty.detach(), "active_interaction_samples": active_interaction.detach()}
