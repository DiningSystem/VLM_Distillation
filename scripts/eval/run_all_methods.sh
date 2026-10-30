#!/usr/bin/env bash
# Batch wrapper: run `scripts/eval/run_eval.sh <suite> <ckpt>` for every
# method checkpoint we just trained. Walks `outputs/` for directories matching
# our sprint RUN_NAME pattern, optionally merging LoRA adapters first.
#
# Usage:
#   bash scripts/eval/run_all_methods.sh <suite> [pattern]
#
#   <suite>    : fast_signal | main_table | cvpr_main | full
#   [pattern]  : glob under outputs/ (default "qwen25_teacher_7b_qwen2_student_2b_*")
#
# Env overrides:
#   PROJECT_DIR        : repo root (default: cwd)
#   MERGE_LORA         : "true" to run scripts/eval/merge_lora.py before eval (default: "auto" — merge if adapter_config.json present and config.json absent)
#   MODE               : passed to run_eval.sh (all | infer | eval; default: all)
#   JUDGE              : passed to run_eval.sh; "" disables paid judges
#   SKIP_DONE          : "true" to skip ckpts that already have a results file (default: true)
set -euo pipefail

SUITE="${1:?usage: run_all_methods.sh <suite> [pattern]}"
PATTERN="${2:-qwen25_teacher_7b_qwen2_student_2b_*}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MERGE_LORA="${MERGE_LORA:-auto}"
MODE="${MODE:-all}"
SKIP_DONE="${SKIP_DONE:-true}"

CONFIG="${PROJECT_DIR}/configs/eval/${SUITE}.json"
[[ -f "${CONFIG}" ]] || { echo "missing suite config: ${CONFIG}"; exit 1; }

shopt -s nullglob
CANDIDATES=("${PROJECT_DIR}"/outputs/${PATTERN})
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "no checkpoints under ${PROJECT_DIR}/outputs/${PATTERN}"
  exit 1
fi

echo "Found ${#CANDIDATES[@]} candidate run dir(s):"
for c in "${CANDIDATES[@]}"; do echo "  ${c}"; done
echo

passed=0
failed=0
skipped=0
for run_dir in "${CANDIDATES[@]}"; do
  run_name=$(basename "${run_dir}")
  # Prefer the final epoch checkpoint if save_strategy=epoch produced one;
  # otherwise use the run dir itself.
  ckpt="${run_dir}"
  latest=$(ls -1d "${run_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  [[ -n "${latest}" ]] && ckpt="${latest}"

  out_dir="${PROJECT_DIR}/outputs/eval/${SUITE}/$(basename "${ckpt%/}")"
  if [[ "${SKIP_DONE}" == "true" && -d "${out_dir}" && -n "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
    echo "[skip] ${run_name}  (results present at ${out_dir})"
    skipped=$((skipped + 1))
    continue
  fi

  # Optionally merge LoRA adapter into a deployable checkpoint.
  if [[ "${MERGE_LORA}" == "true" || "${MERGE_LORA}" == "auto" ]]; then
    if [[ ! -f "${ckpt}/config.json" && -f "${ckpt}/adapter_config.json" ]]; then
      merged="${ckpt}_merged"
      if [[ ! -f "${merged}/config.json" ]]; then
        echo "[merge] ${ckpt} -> ${merged}"
        python3 "${PROJECT_DIR}/scripts/eval/merge_lora.py" \
          --adapter "${ckpt}" \
          --output "${merged}" || { echo "[fail] merge_lora.py on ${ckpt}"; failed=$((failed + 1)); continue; }
      fi
      ckpt="${merged}"
    fi
  fi

  echo "[eval] ${run_name}  suite=${SUITE}  mode=${MODE}"
  if MODE="${MODE}" JUDGE="${JUDGE-}" bash "${PROJECT_DIR}/scripts/eval/run_eval.sh" "${SUITE}" "${ckpt}"; then
    passed=$((passed + 1))
  else
    echo "[fail] ${run_name}"
    failed=$((failed + 1))
  fi
done

echo
echo "[summary] passed=${passed} failed=${failed} skipped=${skipped}"
[[ ${failed} -eq 0 ]] || exit 1
