#!/usr/bin/env bash
# Run a VLMEvalKit evaluation suite over a single training checkpoint.
#
# Usage:
#   bash scripts/eval/run_eval.sh <suite> <ckpt_path> [model_name]
#
#   <suite>          : fast_signal | main_table | cvpr_main | full
#                      (matches configs/eval/<suite>.json)
#   <ckpt_path>      : path to a saved student checkpoint dir (DistillTrainer._save layout)
#   <model_name>     : name used inside VLMEvalKit outputs (default: checkpoint basename).
#
# Env overrides:
#   PROJECT_DIR     : repo root (default: cwd)
#   PYTHON_BIN      : Python interpreter for merge/config/VLMEvalKit (default: python3)
#   VLMEVALKIT_DIR  : path to your VLMEvalKit checkout (default: $PROJECT_DIR/VLMEvalKit)
#   VLMEVAL_MODEL_CLASS       : VLMEvalKit model class (default: Qwen2VLChat)
#   VLMEVAL_USE_CUSTOM_PROMPT : true | false | auto (default: auto)
#   VLMEVAL_MIN_PIXELS        : Qwen min_pixels (default: 1280*28*28)
#   VLMEVAL_MAX_PIXELS        : Qwen max_pixels (default: 16384*28*28)
#   MERGE_LORA      : auto | false (default: auto). Auto-merges adapter-only checkpoints.
#   MERGED_ROOT     : where auto-merged checkpoints are written (default: outputs/eval/merged_checkpoints)
#   MERGE_BASE_MODEL: optional base model override for LoRA merge
#   MERGE_DEVICE_MAP: merge-time device_map, e.g. auto | cuda:0 | cpu | none (default: auto)
#   MODE            : all | infer | eval (default: all). Use MODE=infer to skip judging/eval.
#   JUDGE           : override judge_model from the config (e.g. JUDGE=gpt-4o-mini for cheaper)
#   OPENAI_API_KEY  : must be set when judge_model is a GPT-4-class model
set -euo pipefail

SUITE="${1:?usage: run_eval.sh <suite> <ckpt_path> [model_name]}"
CKPT="${2:?usage: run_eval.sh <suite> <ckpt_path> [model_name]}"
MODEL_NAME="${3:-$(basename "${CKPT%/}")}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_DIR}/VLMEvalKit}"
CONFIG="${PROJECT_DIR}/configs/eval/${SUITE}.json"
MODE="${MODE:-all}"

[[ -f "${CONFIG}" ]]      || { echo "Missing config: ${CONFIG}"; exit 1; }
[[ -d "${CKPT}" ]]        || { echo "Missing ckpt dir: ${CKPT}"; exit 1; }
[[ -d "${VLMEVALKIT_DIR}" ]] || { echo "Missing VLMEvalKit checkout at ${VLMEVALKIT_DIR}. Clone https://github.com/open-compass/VLMEvalKit and pin a commit, then set VLMEVALKIT_DIR."; exit 1; }
if [[ ! -f "${CKPT}/config.json" && -f "${CKPT}/adapter_config.json" ]]; then
  MERGE_LORA="${MERGE_LORA:-auto}"
  if [[ "${MERGE_LORA}" == "false" ]]; then
    echo "Checkpoint looks like a PEFT adapter-only directory: ${CKPT}"
    echo "VLMEvalKit Qwen2VLChat expects a merged HF checkpoint with config.json."
    echo "Unset MERGE_LORA=false or merge LoRA first, then pass the merged checkpoint dir."
    exit 1
  fi

  ORIGINAL_CKPT="${CKPT}"
  CKPT_NAME=$(basename "${CKPT%/}")
  MERGED_ROOT="${MERGED_ROOT:-${PROJECT_DIR}/outputs/eval/merged_checkpoints}"
  MERGED_CKPT="${MERGED_ROOT}/${CKPT_NAME}"
  if [[ ! -f "${MERGED_CKPT}/config.json" ]]; then
    echo "[eval] auto-merging LoRA adapter for VLMEvalKit: ${ORIGINAL_CKPT}"
    MERGE_ARGS=(--adapter "${ORIGINAL_CKPT}" --output "${MERGED_CKPT}" --device-map "${MERGE_DEVICE_MAP:-auto}")
    if [[ -n "${MERGE_BASE_MODEL:-}" ]]; then
      MERGE_ARGS+=(--base-model "${MERGE_BASE_MODEL}")
    fi
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/eval/merge_lora.py" "${MERGE_ARGS[@]}"
  fi
  CKPT="${MERGED_CKPT}"
fi
[[ -f "${CKPT}/config.json" ]] || { echo "Checkpoint is not a merged HF model directory with config.json: ${CKPT}"; exit 1; }

CONFIG_JUDGE=$("${PYTHON_BIN}" -c "import json,sys; cfg=json.load(open(sys.argv[1])); print(cfg.get('judge_model') or '')" "${CONFIG}")
DATASETS=$("${PYTHON_BIN}" -c "import json,sys; cfg=json.load(open(sys.argv[1])); print(' '.join(cfg['datasets']))" "${CONFIG}")

JUDGE="${JUDGE-${CONFIG_JUDGE}}"

CKPT_NAME=$(basename "${CKPT%/}")
WORK_DIR="${PROJECT_DIR}/outputs/eval/${SUITE}/${CKPT_NAME}"
mkdir -p "${WORK_DIR}"
GENERATED_CONFIG="${WORK_DIR}/vlmeval_config.json"
JUDGE_ARGS_JSON=$("${PYTHON_BIN}" -c "import json,sys; cfg=json.load(open(sys.argv[1])); kwargs={k:v for k,v in dict(cfg.get('judge_kwargs') or {}).items() if k != 'open_judge_fallback' and v is not None}; print(json.dumps(kwargs) if kwargs else '')" "${CONFIG}")

VLMEVALKIT_DIR="${VLMEVALKIT_DIR}" \
CONFIG="${CONFIG}" \
CKPT="${CKPT}" \
MODEL_NAME="${MODEL_NAME}" \
VLMEVAL_MODEL_CLASS="${VLMEVAL_MODEL_CLASS:-Qwen2VLChat}" \
VLMEVAL_USE_CUSTOM_PROMPT="${VLMEVAL_USE_CUSTOM_PROMPT:-auto}" \
VLMEVAL_MIN_PIXELS="${VLMEVAL_MIN_PIXELS:-1003520}" \
VLMEVAL_MAX_PIXELS="${VLMEVAL_MAX_PIXELS:-12845056}" \
GENERATED_CONFIG="${GENERATED_CONFIG}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

vlmevalkit_dir = os.environ["VLMEVALKIT_DIR"]
sys.path.insert(0, vlmevalkit_dir)

from vlmeval.dataset import DATASET_CLASSES
from vlmeval.dataset.video_dataset_config import supported_video_datasets

with open(os.environ["CONFIG"], encoding="utf-8") as f:
    suite_cfg = json.load(f)

datasets = suite_cfg["datasets"]
data_cfg = {}
for dataset_name in datasets:
    if dataset_name in supported_video_datasets:
        data_cfg[dataset_name] = {}
        continue

    dataset_cls = None
    for cls in DATASET_CLASSES:
        if dataset_name in cls.supported_datasets():
            dataset_cls = cls
            break
    if dataset_cls is None:
        raise SystemExit(f"Dataset {dataset_name} is not supported by this VLMEvalKit checkout.")

    data_cfg[dataset_name] = {
        "class": dataset_cls.__name__,
        "dataset": dataset_name,
    }

use_custom_prompt = os.environ["VLMEVAL_USE_CUSTOM_PROMPT"].lower()
if use_custom_prompt == "auto":
    target = f"{os.environ['MODEL_NAME']} {os.environ['CKPT']}".lower()
    use_custom_prompt = "false" if "qwen2.5" in target or "qwen3" in target else "true"
if use_custom_prompt not in {"true", "false"}:
    raise SystemExit("VLMEVAL_USE_CUSTOM_PROMPT must be true, false, or auto.")

model_name = os.environ["MODEL_NAME"].replace("/", "__")
model_cfg = {
    model_name: {
        "class": os.environ["VLMEVAL_MODEL_CLASS"],
        "model_path": os.environ["CKPT"],
        "min_pixels": int(os.environ["VLMEVAL_MIN_PIXELS"]),
        "max_pixels": int(os.environ["VLMEVAL_MAX_PIXELS"]),
        "use_custom_prompt": use_custom_prompt == "true",
    }
}

generated = {
    "model": model_cfg,
    "data": data_cfg,
}
with open(os.environ["GENERATED_CONFIG"], "w", encoding="utf-8") as f:
    json.dump(generated, f, indent=2)
PY

echo "[eval] suite        = ${SUITE}"
echo "[eval] datasets     = ${DATASETS}"
echo "[eval] model_name   = ${MODEL_NAME}"
echo "[eval] ckpt         = ${CKPT}"
echo "[eval] judge        = ${JUDGE:-<none>}"
echo "[eval] mode         = ${MODE}"
echo "[eval] work_dir     = ${WORK_DIR}"
echo "[eval] config       = ${GENERATED_CONFIG}"

cd "${VLMEVALKIT_DIR}"
RUN_ARGS=(
  --config "${GENERATED_CONFIG}"
  --work-dir "${WORK_DIR}"
  --mode "${MODE}"
  --reuse
)
if [[ -n "${JUDGE}" ]]; then
  RUN_ARGS+=(--judge "${JUDGE}")
fi
if [[ -n "${JUDGE_ARGS_JSON}" ]]; then
  RUN_ARGS+=(--judge-args "${JUDGE_ARGS_JSON}")
fi
"${PYTHON_BIN}" run.py "${RUN_ARGS[@]}"
