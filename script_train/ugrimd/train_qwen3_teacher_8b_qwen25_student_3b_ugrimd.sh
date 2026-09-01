#!/usr/bin/env bash
# Uncertainty-gated ranking + multimodal interaction distillation (UGRIMD).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
RUN_NAME="${RUN_NAME:-qwen3_teacher_8b_qwen25_student_3b_ugrimd}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"

cd "${PROJECT_DIR}"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"
source "${PROJECT_DIR}/script_train/_common.sh"

"${TORCHRUN}" --nproc_per_node "${NPROC_PER_NODE:-8}" --master_port "${MASTER_PORT:-29501}" "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" --teacher_model_name "${TEACHER_MODEL}" \
  --data_path "${DATA_PATH}" --image_dir "${IMAGE_DIR}" --output_dir "${OUTPUT_DIR}" \
  --lora true --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
  --per_device_train_batch_size "${PER_DEVICE_BS:-2}" --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --num_train_epochs 1 --learning_rate 1e-5 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --bf16 true --save_strategy steps --save_steps "${SAVE_STEPS:-1000}" --save_total_limit 2 \
  --logging_steps 50 --dataloader_num_workers "${DATALOADER_WORKERS:-2}" --max_len 2048 --image_resolution low \
  --resume_from none --report_to "${REPORT_TO}" --seed 1337 \
  --kd_loss_type ugrimd --distill_lambda 1.0 --interaction_lambda 1.0 \
  --num_candidates 8 --candidate_regenerate_steps 500 \
  --uncertainty_top_percent 30 --exploration_percent 10 \
  --rank_temperature 0.1 --rank_epsilon 1e-8 --candidate_temperature 1.0 \
  --interaction_temperature 1.0 --interaction_strength_temperature 0.1 --loss_epsilon 1e-8 \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
