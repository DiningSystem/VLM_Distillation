import logging
import os
from numbers import Number
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from src.arguments import ModelArguments, TrainingArguments
from src.distiller import Distiller
from src.model.model import VLMModel
from src.utils import print_master


logger = logging.getLogger(__name__)

# Metadata keys that should never be forwarded to the model.
_BATCH_METADATA_KEYS = frozenset(
    {"ids", "image_paths", "texts", "images", "candidate_messages", "global_dataset_name", "dataset_name"}
)


class DistillTrainer(Trainer):
    """
    HF Trainer subclass that handles two training modes:

    SFT mode (no teacher):
        Pass a ``VLMModel`` as ``model``.  The trainer uses the cross-entropy
        loss computed against the ``labels`` tensor supplied by the collator.
        Labels are built in ``VlmDistillDataCollator`` and mask all non-assistant
        tokens (system prompt, user turns, padding) with -100 so only assistant
        responses contribute to the loss.

    Distillation mode (student + teacher):
        Pass a ``Distiller`` as ``model`` **and** supply a ``criterion``
        callable with signature ``criterion(distiller, batch) -> loss``.
        The distiller owns both student and teacher.  ``student_inputs`` always
        carries a ``labels`` tensor (same masking as SFT) which the criterion
        can use to gate token-level KD loss to assistant positions only.
    """

    def __init__(
        self,
        model_args: ModelArguments,
        criterion: Optional[Callable] = None,
        **kwargs,
    ):
        self.model_args = model_args
        self.criterion = criterion

        model = kwargs.get("model")
        self._is_distillation = isinstance(model, Distiller)

        if self._is_distillation and criterion is None:
            raise ValueError("A criterion callable is required when model is a Distiller.")

        # The collator returns nested dicts; Trainer's column-removal
        # introspects forward() signatures and would silently drop everything.
        args = kwargs.get("args")
        if args is not None:
            args.remove_unused_columns = False

        super().__init__(**kwargs)

        self._is_ddp = dist.is_initialized()
        self._loss_metric_sums: Dict[str, float] = {}
        self._loss_metric_counts: Dict[str, int] = {}
        self._latest_loss_metrics: Dict[str, float] = {}

        # Accumulators for criterion-side metrics. compute_loss is called once
        # per micro-batch; we average across grad_accum_steps and emit at the
        # next logging step via our `log()` override below.
        self._kd_metric_sums: Dict[str, float] = {}
        self._kd_metric_count: int = 0

    # ------------------------------------------------------------------
    # Column removal — disabled; our batches are nested dicts
    # ------------------------------------------------------------------

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    # ------------------------------------------------------------------
    # Component-metric logging
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        """Inject accumulated criterion-side metrics into each periodic log.

        HF Trainer's `_maybe_log_save_evaluate` calls `self.log(logs)` at
        `args.logging_steps`. We hook here so SCVA / CGKD / hard_loss /
        validity counters land in WandB at every visible step.
        """
        if self._kd_metric_count > 0:
            for key, total in self._kd_metric_sums.items():
                logs[f"train/{key}"] = total / self._kd_metric_count
            self._kd_metric_sums.clear()
            self._kd_metric_count = 0
        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._is_distillation:
            if hasattr(self.criterion, "set_global_step"):
                # ``state.global_step`` counts optimizer updates (rather than
                # micro-batches), which is exactly the candidate-cache cadence.
                self.criterion.set_global_step(self.state.global_step)
            # The criterion receives the full batch, which includes
            # student_inputs["labels"] for assistant-position gating.
            loss_output = model(self.criterion, inputs)
            if isinstance(loss_output, dict):
                self._record_loss_metrics(loss_output)
                self._update_tqdm_postfix()
                loss = loss_output["loss"]
            else:
                loss = loss_output

            # Accumulate component metrics (everything in the dict except "loss")
            # for the next call to `log()`. This is how SCVA/CGKD/hard_loss/
            # scva_valid_* surface in WandB and verifies the headline method is
            # actually contributing, not silently zeroed.
            if isinstance(loss_output, dict):
                for key, value in loss_output.items():
                    if key == "loss":
                        continue
                    if torch.is_tensor(value):
                        if value.numel() == 0:
                            continue
                        scalar = value.detach().float().mean().item()
                    else:
                        scalar = float(value)
                    self._kd_metric_sums[key] = self._kd_metric_sums.get(key, 0.0) + scalar
                self._kd_metric_count += 1

            return (loss, None) if return_outputs else loss

        # SFT mode: forward through student model only.
        student_inputs = inputs.get("student_inputs", inputs)
        model_inputs = self._build_sft_model_inputs(student_inputs)
        outputs = model(**model_inputs)

        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                "VLMModel did not return a loss. Make sure labels are present in the "
                "batch, or that the backbone computes LM loss internally."
            )
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _build_sft_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip metadata keys and forward the rest to the model.

        ``labels`` is now expected to be present in ``inputs`` — it is built
        by ``VlmDistillDataCollator._build_labels`` at collation time, with
        all non-assistant positions already masked to -100.  We therefore no
        longer synthesise labels here; if they are somehow absent we raise an
        explicit error rather than silently using a wrong label tensor.
        """
        cleaned = {k: v for k, v in inputs.items() if k not in _BATCH_METADATA_KEYS}

        if "labels" not in cleaned:
            raise RuntimeError(
                "No 'labels' key found in student_inputs. "
                "Ensure VlmDistillDataCollator is used and that "
                "_build_labels ran successfully during collation."
            )

        return cleaned

    # ------------------------------------------------------------------
    # Logging — include criterion loss components
    # ------------------------------------------------------------------

    def _record_loss_metrics(self, loss_output: Dict[str, Any]) -> None:
        latest = {}
        for name, value in loss_output.items():
            scalar = self._to_log_scalar(value)
            if scalar is None:
                continue

            latest[name] = scalar
            if name == "loss":
                continue

            self._loss_metric_sums[name] = self._loss_metric_sums.get(name, 0.0) + scalar
            self._loss_metric_counts[name] = self._loss_metric_counts.get(name, 0) + 1
        self._latest_loss_metrics = latest

    def _update_tqdm_postfix(self) -> None:
        if not self._latest_loss_metrics:
            return

        priority_keys = (
            "loss",
            "supervised_loss",
            "hard_loss",
            "kd_loss",
            "t2s_ce_loss",
            "t2s_kd_loss",
            "s2t_kd_loss",
            "dtw_loss",
            "weighted_dtw_loss",
            "mcw_ot_logits_loss",
            "mcw_ot_hidden_loss",
            "scva_loss",
            "cgkd_loss",
            "sre_kd_loss",
            "sre_span_loss",
            "sre_geom_loss",
            "sre_logit_loss",
            "response_kd_loss",
            "vision_kd_loss",
            "vision_logit_loss",
            "affinity_loss",
        )

        loss_keys = [
            key
            for key in self._latest_loss_metrics
            if key == "loss" or key.endswith("_loss") or "loss" in key
        ]
        ordered_keys = [key for key in priority_keys if key in loss_keys]
        ordered_keys.extend(sorted(key for key in loss_keys if key not in set(ordered_keys)))

        postfix = {key: f"{self._latest_loss_metrics[key]:.4g}" for key in ordered_keys}
        if not postfix:
            return

        for callback in getattr(self.callback_handler, "callbacks", []):
            training_bar = getattr(callback, "training_bar", None)
            if training_bar is not None:
                try:
                    training_bar.set_postfix(postfix, refresh=False)
                except TypeError:
                    training_bar.set_postfix(postfix)
                break

    @staticmethod
    def _to_log_scalar(value: Any) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().float()
            if not torch.isfinite(value):
                return None
            return float(value.item())

        if isinstance(value, Number):
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                return None
            return value

        return None

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._loss_metric_sums:
            for name, total in self._loss_metric_sums.items():
                count = self._loss_metric_counts.get(name, 0)
                if count > 0 and name not in logs:
                    logs[name] = total / count
            self._loss_metric_sums.clear()
            self._loss_metric_counts.clear()

        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Optimizer — separate LR for projectors when configured
    # ------------------------------------------------------------------

    def create_optimizer(self):
        """
        Build the optimizer.  When in distillation mode, projector parameters
        are placed in a dedicated param group if ``projector_lr`` differs from
        the base learning rate.
        """
        super().create_optimizer()

        if not (self._is_distillation and self._projector_needs_own_group()):
            return self.optimizer

        projector_ids = {
            id(p)
            for p in self.model.projectors.parameters()
            if p.requires_grad
        }

        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in projector_ids]

        proj_lr = self.model_args.projector_lr or self.args.learning_rate
        self.optimizer.add_param_group(
            {
                "params": [p for p in self.model.projectors.parameters() if p.requires_grad],
                "lr": proj_lr,
            }
        )
        print_master(f"Projector param group added to optimizer (lr={proj_lr})")
        return self.optimizer

    def _projector_needs_own_group(self) -> bool:
        if not (hasattr(self.model, "projectors") and self.model.projectors is not None):
            return False
        if not any(p.requires_grad for p in self.model.projectors.parameters()):
            return False
        proj_lr = getattr(self.model_args, "projector_lr", None)
        return proj_lr is not None and proj_lr != self.args.learning_rate

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        raw_model = self.model
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module

        if self._is_distillation:
            raw_model.student.save(output_dir)
            raw_model.save_projectors(os.path.join(output_dir, "projectors"))
        else:
            raw_model.save(output_dir)

        processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing_class is not None:
            processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        print_master(f"Checkpoint saved to {output_dir}")
