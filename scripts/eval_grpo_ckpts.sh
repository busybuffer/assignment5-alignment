#!/usr/bin/env bash
# Evaluate GRPO checkpoints on the full validation set (5000 examples).
# Leaderboard constraints: temperature=1.0, max_tokens=1024, r1_zero prompt.
# Usage: bash scripts/eval_grpo_ckpts.sh
#
# Uses two GPUs in parallel by assigning different checkpoints to each GPU.

set -euo pipefail

DATA_PATH="data/MATH/validation.jsonl"
OUT_DIR="outputs/eval"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
mkdir -p "$OUT_DIR"

CKPT_DIRS=(
    # "outputs/grpo_leaderboard/checkpoint_step_150"
    "outputs/grpo_lr_sweep_20260407_lr_3e-5/checkpoint_step50"
    "outputs/grpo_no_std_norm/checkpoint_step50"
    # "outputs/grpo_clip_offp/checkpoint_step50"
    # "outputs/grpo_lengnorm_mean/checkpoint_step50"
    # "outputs/grpo_lengnorm_normalize/checkpoint_step50"
    # "outputs/grpo_no_bl/checkpoint_step50"
    # "outputs/grpo_no_clip_offp/checkpoint"
)

GPU0_CKPTS=()
GPU1_CKPTS=()

for i in "${!CKPT_DIRS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_CKPTS+=("${CKPT_DIRS[$i]}")
    else
        GPU1_CKPTS+=("${CKPT_DIRS[$i]}")
    fi
done

run_eval_queue() {
    local gpu_id="$1"
    shift
    local ckpt_dir

    for ckpt_dir in "$@"; do
        if [ ! -d "$ckpt_dir" ]; then
            echo "Skipping $ckpt_dir (not found)"
            continue
        fi

        local run_name
        local out_file
        run_name="$(basename "$(dirname "$ckpt_dir")")"
        out_file="${OUT_DIR}/grpo_${run_name}.jsonl"

        if [ -f "$out_file" ]; then
            echo "Skipping $run_name (already evaluated)"
            continue
        fi

        echo "=============================="
        echo "[GPU ${gpu_id}] Evaluating: $run_name"
        echo "=============================="

        CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/math_baseline.py \
            --model "$ckpt_dir" \
            --data-path "$DATA_PATH" \
            --output-path "$out_file" \
            --max-tokens 1024 \
            --temperature 1.0
    done
}

run_eval_queue "$GPU0" "${GPU0_CKPTS[@]}" &
PID0=$!

run_eval_queue "$GPU1" "${GPU1_CKPTS[@]}" &
PID1=$!

wait "$PID0"
wait "$PID1"

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
python3 -c "
import glob
import json

rows = []
for f in glob.glob('outputs/eval/grpo_*.jsonl'):
    lines = [json.loads(l) for l in open(f)]
    correct = sum(l['answer_reward'] == 1.0 for l in lines)
    total = len(lines)
    name = f.split('/')[-1].replace('.jsonl', '')
    rows.append((correct / total, name, correct, total))

for acc, name, correct, total in sorted(rows, reverse=True):
    print(f'  {name:<45} {acc:.4f}  ({correct}/{total})')
"
