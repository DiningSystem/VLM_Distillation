# VLM Distillation

This repository is used to fine-tune and distill Vision-Language Models (VLMs) with a teacher-student setup. The main entry point is `train.py`: it parses CLI arguments, loads the student and optional teacher, builds the dataset/collator, selects the distillation criterion, and runs `DistillTrainer`.

## Main Training Flow

1. Pick a script from `script_train/<method>/`.
2. The script launches `train.py` through `torchrun`.
3. `train.py` parses `ModelArguments`, `DataArguments`, and `TrainingArguments`.
4. If `--teacher_model_name` is set, training runs in distillation mode through `src/distiller.py`; otherwise it runs student-only SFT.
5. Data is loaded by `src/data/dataset.py`, while processors and models are loaded from `src/model/`.
6. The loss is selected with `--kd_loss_type` from `src/criterions/`.
7. Checkpoints and projectors are written to `outputs/`.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `train.py` | Main training entry point. Initializes model, processor, criterion, dataset, trainer, and resume checkpoint handling. |
| `requirements.txt` | Python dependencies for the repository. |
| `.env.example` | Example environment variables for WandB and Hugging Face Hub. |
| `config/` | Projector configs for specific KD methods, such as `dskd_v2_projectors.json`, `dwa_kd_projectors.json`, and `mcw_kd_projectors.json`. |
| `configs/` | Runtime configs, including DeepSpeed ZeRO-2 and evaluation configs. |
| `docs/` | Technical notes and plans, for example the unit-aligned distillation training plan. |
| `experiments/` | Experiment tracking files, such as `experiment_tracker.xlsx`. |
| `outputs/` | Runtime-generated training/evaluation artifacts: checkpoints, projectors, logs, and results. This is not core source code. |
| `script_train/` | Shell training recipes grouped by method and model pair. This is the easiest place to start when running experiments. |
| `scripts/` | Utility scripts outside the main training path, such as tracker generation, UMI computation, evaluation, and LoRA merge. |
| `src/` | Main source code: arguments, trainer, distiller, model wrapper, data pipeline, and distillation losses. |
| `test_loadmodel/` | Smoke tests for loading VLM backbones such as FastVLM, Qwen2-VL, Qwen2.5-VL, and Qwen3-VL. |
| `tests/` | Small unit tests for KD criterion and collator plumbing. |
| `train_data/` | Training datasets and data preparation helpers. Includes the placeholder `put_train_data_here`. |

## Main Modules In `src/`

| File/folder | Purpose |
| --- | --- |
| `src/arguments.py` | Defines dataclass arguments for model, data, and training. Most `train.py` CLI flags live here. |
| `src/utils.py` | Shared logging, rank-aware printing, and utility helpers. |
| `src/trainer.py` | `DistillTrainer`, a Hugging Face `Trainer` subclass. Handles both SFT and distillation modes, logs loss components, and saves checkpoints/projectors. |
| `src/distiller.py` | Teacher-student wrapper. Loads student/teacher, freezes the teacher, creates/loads/initializes projectors, and forwards batches to the selected criterion. |
| `src/data/dataset.py` | Lazy dataset for LLaVA-style VLM chat JSON/JSONL data. Loads images, resizes/pads images, splits multi-turn samples, builds assistant-only labels, and collates student/teacher inputs. |
| `src/model/model.py` | `VLMModel` wrapper around generation backbones. Loads models, enables hidden states/attentions, supports LoRA/PEFT, and saves checkpoints. |
| `src/model/processor.py` | Loads processors/tokenizers by backbone, normalizes image/video tokens, and applies resize config. |
| `src/model/vlm_backbone/` | Custom backbone/modeling/processing code for `fast_vlm`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `llava_next`, and `llava_onevision`. |
| `src/criterions/` | Distillation losses. `__init__.py` maps `--kd_loss_type` to the corresponding criterion class. |

## Criteria In `src/criterions/`

| File | Purpose |
| --- | --- |
| `ce_only.py` | Uses only the student's supervised cross-entropy loss. Useful for baselines, SFT, or disabled KD. |
| `default_distillation.py` | Default KD: combines CE with hidden-state matching through projectors/layer mappings. |
| `em_kd.py` | EM-KD: distills response logits and vision-language affinity. |
| `sre.py` | SRE: span, geometry, and logit distillation; uses the collator's pooler tensors. |
| `unit_aligned.py` | Joint SRE + EM-KD criterion for unit-aligned distillation. |
| `scva.py` | SCVA: semantic-cluster visual attention loss for vision tokens. |
| `cgkd.py` | CGKD: confidence-gated generative KD. |
| `ugrimd.py` | UGRIMD: uncertainty-gated within-sample candidate ranking plus multimodal interaction distillation. |
| `scva_cgkd.py` | Joint SCVA + CGKD criterion. Also available through the `draft` alias. |
| `dwa_kd.py` | DWA-KD with KL, hidden/logit losses, SoftDTW, and projectors. |
| `dskd_v2.py` | DSKD v2 with teacher-to-student/student-to-teacher projectors and top-k vocabulary config. |
| `mcw_kd.py` | MCW-KD with OT/Sinkhorn losses for logits and hidden context. |
| `cross_entropy_loss.py`, `various_divergence.py`, `soft_dtw_cuda.py`, `etp.py` | Shared loss helpers used by multiple criteria. |

Mapped `--kd_loss_type` values include: `ce_only`, `default`, `default_distillation`, `emkd`, `em_kd`, `sre`, `dwa_kd`, `dwakd`, `dskd_v2`, `dskdv2`, `mcw_kd`, `mcwkd`, `joint`, `unit_aligned`, `unit_aligned_distillation`, `scva`, `cgkd`, `scva_cgkd`, `ugrimd`, and `draft`.

## Training Scripts

`script_train/` is grouped by method:

| Folder | Contents |
| --- | --- |
| `script_train/sft/` | Student-only fine-tuning recipes. |
| `script_train/ce_only/` | CE-only baselines inside the teacher-student pipeline. |
| `script_train/em_kd/` | EM-KD recipes. |
| `script_train/sre/` | SRE recipes. |
| `script_train/unit_aligned/` | Joint/unit-aligned recipes. |
| `script_train/scva/` | SCVA recipes. |
| `script_train/cgkd/` | CGKD recipes. |
| `script_train/ugrimd/` | UGRIMD recipes for all three supported teacher-student pairs. |
| `script_train/scva_cgkd/` | SCVA + CGKD recipes. |
| `script_train/dwa_kd/` | DWA-KD recipes. |
| `script_train/dskd_v2/` | DSKD v2 recipes. The currently open file, `train_qwen3_teacher_4b_fastvlm_student_05b_dskd_v2_with_eta.sh`, lives here. |
| `script_train/mcw_kd/` | MCW-KD recipes. |
| `script_train/_common.sh` | Shared helper for loading `.env`, enabling WandB, and setting Hugging Face Hub push flags when tokens are available. |

For example, `script_train/dskd_v2/train_qwen3_teacher_4b_fastvlm_student_05b_dskd_v2_with_eta.sh` uses `FastVLM-0.5B` as the student and `Qwen3-VL-4B-Instruct` as the teacher. It uses `config/dskd_v2_projectors.json`, trains on `train_data/llava_v1_5_mix665k.json`, and writes outputs to `outputs/qwen3_teacher_4b_fastvlm_student_05b_dskd_v2_with_eta`.

## Training Data

The main training pipeline receives `--data_path` pointing to JSON or JSONL data. A typical sample looks like:

```json
{
  "id": "sample_id",
  "image": "relative/or/absolute/image/path.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

`src/data/dataset.py` will:

- resolve images from `--image_dir` or the directory containing the JSON file;
- skip samples that declare an image path but point to a missing file;
- normalize roles: `human/user`, `gpt/assistant`, and `system`;
- split multi-turn multimodal samples into independent Q&A pairs;
- create `-100` labels for system/user/padding tokens so only assistant responses contribute to CE loss;
- create separate student and teacher inputs when running distillation.

## Evaluation And Tests

| File/folder | Purpose |
| --- | --- |
| `scripts/eval/run_eval.sh` | Runs a VLMEvalKit suite for one checkpoint. It can auto-merge LoRA adapters before evaluation. |
| `scripts/eval/merge_lora.py` | Merges a PEFT/LoRA adapter into the base model to produce a full Hugging Face checkpoint. |
| `scripts/eval/run_all_methods.sh` | Runs evaluation across multiple methods/checkpoints. |
| `configs/eval/*.json` | Dataset/judge definitions for evaluation suites: `fast_signal`, `main_table`, and `full`. |
| `test_load_dataset.py` | Smoke test for dataset + collator with real processors. |
| `test_loadmodel/*.py` | Load tests for individual backbone/model processors. |
| `tests/test_kd_plumbing.py` | Small unit tests for criterion mapping and selected argument/collator behavior. |

Run the quick unit test:

```bash
python -m unittest tests/test_kd_plumbing.py
```

Smoke-test the dataset pipeline:

```bash
python test_load_dataset.py \
  --data_path train_data/llava_v1_5_mix665k.json \
  --image_dir train_data \
  --student_model Qwen/Qwen2-VL-2B-Instruct \
  --batch_size 2 \
  --percent_data 0.01
```

## Quick Notes

- Several training scripts hard-code `PROJECT_DIR=/workspace/ComfyUI/models/instantid/VLM_Distillation`; update that path or export the matching variable if the repo is moved.
- `outputs/`, `.venv/`, large datasets, and checkpoints are runtime artifacts, not core source code.
- The teacher is frozen in `Distiller`; trainable parameters are mainly the student LoRA adapter and projectors when enabled.
