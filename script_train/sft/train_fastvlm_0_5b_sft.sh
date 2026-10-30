#!/usr/bin/env bash
# Stage 1: supervised fine-tuning with assistant-only cross-entropy.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
STUDENT_MODEL="${STUDENT_MODEL:-KamilaMila/FastVLM-0.5B}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
SEED="${SEED:-1337}"
RUN_NAME="${RUN_NAME:-fastvlm_0_5b_lora_sft_seed_${SEED}}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"

cd "${PROJECT_DIR}"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"
source "${PROJECT_DIR}/script_train/_common.sh"

"${TORCHRUN}" --nproc_per_node "${NPROC_PER_NODE:-8}" --master_port "${MASTER_PORT:-29501}" "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" \
  --data_path "${DATA_PATH}" --image_dir "${IMAGE_DIR}" --output_dir "${OUTPUT_DIR}" \
  --percent_data "${PERCENT_DATA:-1.0}" \
  --lora true --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_BS:-1}" --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --bf16 true --save_strategy epoch --save_total_limit 2 --logging_steps 100 \
  --dataloader_num_workers "${DATALOADER_WORKERS:-2}" --max_len 2048 --image_resolution low \
  --resume_from none --report_to "${REPORT_TO}" --seed "${SEED}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
