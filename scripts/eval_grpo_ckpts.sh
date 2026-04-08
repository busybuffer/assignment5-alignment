#!/usr/bin/env bash
# Evaluate GRPO checkpoints on the full validation set (5000 examples).
# Leaderboard constraints: temperature=1.0, max_tokens=1024, r1_zero prompt.
# Usage: bash scripts/eval_grpo_ckpts.sh

set -e

DATA_PATH="data/MATH/validation.jsonl"
mkdir -p outputs/eval

for CKPT_DIR in \
    outputs/grpo_lr_sweep_20260407_lr_3e-5/checkpoint_step50 \
    outputs/grpo_no_std_norm/checkpoint_step50 \
    outputs/grpo_clip_offp/checkpoint_step50 \
    outputs/grpo_lengnorm_mean/checkpoint_step50 \
    outputs/grpo_lengnorm_normalize/checkpoint_step50 \
    outputs/grpo_no_bl/checkpoint_step50 \
    outputs/grpo_no_clip_offp/checkpoint; do

    if [ ! -d "$CKPT_DIR" ]; then
        echo "Skipping $CKPT_DIR (not found)"
        continue
    fi

    RUN_NAME="$(basename "$(dirname "$CKPT_DIR")")"
    OUT_FILE="outputs/eval/grpo_${RUN_NAME}.jsonl"

    if [ -f "$OUT_FILE" ]; then
        echo "Skipping $RUN_NAME (already evaluated)"
        continue
    fi

    echo "=============================="
    echo "Evaluating: $RUN_NAME"
    echo "=============================="

    CUDA_VISIBLE_DEVICES=0 python scripts/math_baseline.py \
        --model "$CKPT_DIR" \
        --data-path "$DATA_PATH" \
        --output-path "$OUT_FILE" \
        --max-tokens 1024 \
        --temperature 1.0
done

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
python3 -c "
import json, glob
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
