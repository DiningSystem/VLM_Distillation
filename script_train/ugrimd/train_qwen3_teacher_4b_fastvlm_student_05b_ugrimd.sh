#!/usr/bin/env bash
# Stage 2: UGRIMD after the matching Stage-1 SFT recipe in script_train/sft/.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TRAIN_PY="${PROJECT_DIR}/train.py"
TORCHRUN="${PROJECT_DIR}/.venv/bin/torchrun"
STUDENT_MODEL="${STUDENT_MODEL:-KamilaMila/FastVLM-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/train_data/llava_v1_5_mix665k.json}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_DIR}/train_data}"
SEED="${SEED:-1337}"
RUN_NAME="${RUN_NAME:-qwen3_teacher_4b_fastvlm_student_05b_ugrimd_seed_${SEED}}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${RUN_NAME}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${PROJECT_DIR}/outputs/fastvlm_0_5b_lora_sft_seed_${SEED}}"

cd "${PROJECT_DIR}"
[[ -x "${TORCHRUN}" ]] || TORCHRUN="torchrun"
source "${PROJECT_DIR}/script_train/_common.sh"

[[ -d "${SFT_CHECKPOINT}" ]] || { echo "Missing Stage-1 SFT checkpoint: ${SFT_CHECKPOINT}" >&2; exit 1; }

"${TORCHRUN}" --nproc_per_node "${NPROC_PER_NODE:-8}" --master_port "${MASTER_PORT:-29501}" "${TRAIN_PY}" \
  --model_name "${STUDENT_MODEL}" --teacher_model_name "${TEACHER_MODEL}" \
  --checkpoint_path "${SFT_CHECKPOINT}" --load_pretrained_lora true \
  --data_path "${DATA_PATH}" --image_dir "${IMAGE_DIR}" --output_dir "${OUTPUT_DIR}" \
  --lora true --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
  --percent_data "${PERCENT_DATA:-1.0}" \
  --per_device_train_batch_size "${PER_DEVICE_BS:-2}" --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --max_steps "${MAX_STEPS:-20000}" --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --bf16 true --save_strategy steps --save_steps "${SAVE_STEPS:-1000}" --save_total_limit 2 \
  --logging_steps 50 --dataloader_num_workers "${DATALOADER_WORKERS:-2}" --max_len 2048 --image_resolution low \
  --resume_from none --report_to "${REPORT_TO}" --seed "${SEED}" \
  --kd_loss_type ugrimd --distill_lambda "${DISTILL_LAMBDA:-0.5}" --interaction_lambda "${INTERACTION_LAMBDA:-0.7}" \
  --num_candidates "${NUM_CANDIDATES:-8}" --candidate_max_new_tokens "${CANDIDATE_MAX_NEW_TOKENS:-64}" \
  --candidate_regenerate_steps "${CANDIDATE_REGENERATE_STEPS:-500}" \
  --hard_sample_top_percent "${HARD_SAMPLE_TOP_PERCENT:-30}" \
  --hard_sample_refresh_steps "${HARD_SAMPLE_REFRESH_STEPS:-1000}" --exploration_percent "${EXPLORATION_PERCENT:-10}" \
  --rank_temperature "${RANK_TEMPERATURE:-0.1}" --rank_epsilon "${RANK_EPSILON:-1e-8}" \
  --visual_attention_layers ${VISUAL_ATTENTION_LAYERS:-1 2 3} \
  --visual_weight_temperature "${VISUAL_WEIGHT_TEMPERATURE:-0.1}" --visual_weight_floor "${VISUAL_WEIGHT_FLOOR:-0.25}" \
  --candidate_temperature "${CANDIDATE_TEMPERATURE:-1.0}" --interaction_temperature "${INTERACTION_TEMPERATURE:-1.0}" \
  --interaction_strength_temperature "${INTERACTION_STRENGTH_TEMPERATURE:-0.1}" --loss_epsilon "${LOSS_EPSILON:-1e-8}" \
  ${HUB_FLAGS[@]+"${HUB_FLAGS[@]}"}
