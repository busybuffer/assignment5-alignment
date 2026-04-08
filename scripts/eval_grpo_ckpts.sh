#!/usr/bin/env bash
# Evaluate GRPO checkpoints on the full validation set (5000 examples).
# Leaderboard constraints: temperature=1.0, max_tokens=1024, r1_zero prompt.
# Usage: bash scripts/eval_grpo_ckpts.sh
#
# Uses two GPUs together via vLLM tensor parallelism.

set -euo pipefail

DATA_PATH="data/MATH/validation.jsonl"
OUT_DIR="outputs/eval"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
mkdir -p "$OUT_DIR"

CKPT_DIRS=(
    "outputs/outputs/grpo_leaderboard/checkpoint_step_150"
    "outputs/grpo_lr_sweep_20260407_lr_3e-5/checkpoint_step50"
    # "outputs/grpo_no_std_norm/checkpoint_step50"
    # "outputs/grpo_clip_offp/checkpoint_step50"
    # "outputs/grpo_lengnorm_mean/checkpoint_step50"
    # "outputs/grpo_lengnorm_normalize/checkpoint_step50"
    # "outputs/grpo_no_bl/checkpoint_step50"
    # "outputs/grpo_no_clip_offp/checkpoint"
)

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    if [ ! -d "$ckpt_dir" ]; then
        echo "Skipping $ckpt_dir (not found)"
        continue
    fi

    run_name="$(basename "$(dirname "$ckpt_dir")")"
    out_file="${OUT_DIR}/grpo_${run_name}.jsonl"

    if [ -f "$out_file" ]; then
        echo "Skipping $run_name (already evaluated)"
        continue
    fi

    echo "=============================="
    echo "Evaluating: $run_name"
    echo "GPUs: $GPU_DEVICES | tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "=============================="

    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" python scripts/math_baseline.py \
        --model "$ckpt_dir" \
        --data-path "$DATA_PATH" \
        --output-path "$out_file" \
        --max-tokens 1024 \
        --temperature 1.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
done

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
