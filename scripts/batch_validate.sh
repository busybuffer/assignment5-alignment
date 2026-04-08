#!/usr/bin/env bash
# Batch leaderboard validation for all saved checkpoints.
# Runs validate_leaderboard.py on each checkpoint_step50 and final checkpoint.
#
# Usage:
#   bash scripts/batch_validate.sh
#   VLLM_DEVICE=cuda:0 bash scripts/batch_validate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VLLM_DEVICE="${VLLM_DEVICE:-cuda:1}"
OUTPUTS_DIR="outputs"
RESULTS_DIR="outputs/leaderboard_results"
mkdir -p "${RESULTS_DIR}"

RUNS=(
    "grpo_lr_sweep_20260407_lr_3e-5"
    "grpo_no_std_norm"
    "grpo_clip_offp"
    "grpo_lengnorm_mean"
    "grpo_lengnorm_normalize"
    "grpo_no_bl"
    "grpo_no_clip_offp"
)

for run in "${RUNS[@]}"; do
    run_dir="${OUTPUTS_DIR}/${run}"

    for ckpt_name in "checkpoint_step50" "checkpoint"; do
        ckpt="${run_dir}/${ckpt_name}"
        if [[ ! -d "${ckpt}" ]]; then
            echo "  [skip] ${ckpt} not found"
            continue
        fi

        out_file="${RESULTS_DIR}/${run}_${ckpt_name}.json"
        if [[ -f "${out_file}" ]]; then
            echo "  [skip] ${out_file} already exists"
            continue
        fi

        echo "========== ${run} / ${ckpt_name} =========="
        python scripts/validate_leaderboard.py \
            --checkpoint "${ckpt}" \
            --vllm-device "${VLLM_DEVICE}" \
            --output-file "${out_file}"
    done
done

echo ""
echo "All done. Results in ${RESULTS_DIR}/"
echo ""

# Print summary table
echo "===== Summary ====="
printf "%-50s  %s\n" "Run/Checkpoint" "Accuracy"
echo "-------------------------------------------------------------------"
for f in "${RESULTS_DIR}"/*.json; do
    name="$(basename "${f}" .json)"
    acc="$(python -c "import json; d=json.load(open('${f}')); print(f\"{d['accuracy']:.4f}\")")"
    printf "%-50s  %s\n" "${name}" "${acc}"
done
