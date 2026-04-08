#!/usr/bin/env bash
# Batch leaderboard validation for all saved checkpoints.
# Splits checkpoints across two GPUs and runs them in parallel.
#
# Usage:
#   bash scripts/batch_validate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

# Collect all (run, ckpt_name) pairs that need to be evaluated
declare -a JOBS=()
for run in "${RUNS[@]}"; do
    for ckpt_name in "checkpoint_step50" "checkpoint"; do
        ckpt="${OUTPUTS_DIR}/${run}/${ckpt_name}"
        out_file="${RESULTS_DIR}/${run}_${ckpt_name}.json"
        if [[ ! -d "${ckpt}" ]]; then
            echo "[skip] ${ckpt} not found"
            continue
        fi
        if [[ -f "${out_file}" ]]; then
            echo "[skip] ${out_file} already exists"
            continue
        fi
        JOBS+=("${run}:${ckpt_name}")
    done
done

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "Nothing to do."
else
    # Split jobs across cuda:0 and cuda:1, run in parallel
    GPU0_PIDS=(); GPU1_PIDS=()
    GPU0_LOGS=(); GPU1_LOGS=()

    for i in "${!JOBS[@]}"; do
        job="${JOBS[$i]}"
        run="${job%%:*}"
        ckpt_name="${job##*:}"
        ckpt="${OUTPUTS_DIR}/${run}/${ckpt_name}"
        out_file="${RESULTS_DIR}/${run}_${ckpt_name}.json"

        if (( i % 2 == 0 )); then
            device="cuda:0"
        else
            device="cuda:1"
        fi

        echo "========== [${device}] ${run} / ${ckpt_name} =========="
        python scripts/validate_leaderboard.py \
            --checkpoint "${ckpt}" \
            --vllm-device "${device}" \
            --output-file "${out_file}" \
            > "${RESULTS_DIR}/${run}_${ckpt_name}.log" 2>&1 &

        if (( i % 2 == 0 )); then
            GPU0_PIDS+=($!)
            GPU0_LOGS+=("${RESULTS_DIR}/${run}_${ckpt_name}.log")
        else
            GPU1_PIDS+=($!)
            GPU1_LOGS+=("${RESULTS_DIR}/${run}_${ckpt_name}.log")
            # Wait for both GPUs to finish before launching the next pair
            for pid in "${GPU0_PIDS[@]}" "${GPU1_PIDS[@]}"; do
                wait "${pid}"
            done
            # Print last line (accuracy) from each completed log
            for log in "${GPU0_LOGS[@]}" "${GPU1_LOGS[@]}"; do
                [[ -f "${log}" ]] && grep -E "Accuracy|N correct" "${log}" | tail -2 || true
            done
            GPU0_PIDS=(); GPU1_PIDS=()
            GPU0_LOGS=(); GPU1_LOGS=()
        fi
    done

    # Wait for any remaining job on cuda:0 (odd total number of jobs)
    for pid in "${GPU0_PIDS[@]}"; do
        wait "${pid}"
    done
    for log in "${GPU0_LOGS[@]}"; do
        [[ -f "${log}" ]] && grep -E "Accuracy|N correct" "${log}" | tail -2 || true
    done
fi

echo ""
echo "All done. Results in ${RESULTS_DIR}/"
echo ""

# Print summary table
echo "===== Summary ====="
printf "%-52s  %-8s  %-8s  %s\n" "Run/Checkpoint" "Accuracy" "Ans_Rew" "N_correct/N_total"
echo "-------------------------------------------------------------------------------------"
shopt -s nullglob
json_files=("${RESULTS_DIR}"/*.json)
if [[ ${#json_files[@]} -eq 0 ]]; then
    echo "  (no results found)"
else
    for f in "${json_files[@]}"; do
        name="$(basename "${f}" .json)"
        python -c "
import json
d = json.load(open('${f}'))
print(f\"{name:<52}  {d['accuracy']:.4f}    {d['avg_answer_reward']:.4f}    {d['n_correct']}/{d['n_total']}\")
"
    done
fi
