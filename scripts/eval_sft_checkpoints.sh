#!/usr/bin/env bash
# Evaluate all SFT checkpoints on the full validation set (5000 examples).
# Usage: bash scripts/eval_sft_checkpoints.sh
#
# Results are saved to outputs/eval/{run_name}.jsonl

set -e

DATA_PATH="data/MATH/validation.jsonl"
DEVICE="cuda:1"

for CKPT_DIR in outputs/sft_full outputs/sft_128 outputs/sft_256 outputs/sft_512 outputs/sft_1024 outputs/sft_filtered_correct; do
    if [ ! -d "$CKPT_DIR" ]; then
        echo "Skipping $CKPT_DIR (not found)"
        continue
    fi

    RUN_NAME=$(basename "$CKPT_DIR")
    OUTPUT_PATH="outputs/eval/${RUN_NAME}.jsonl"

    echo "=============================="
    echo "Evaluating: $RUN_NAME"
    echo "=============================="

    CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python scripts/math_baseline.py \
        --model "$CKPT_DIR" \
        --data-path "$DATA_PATH" \
        --output-path "$OUTPUT_PATH" \
        --max-tokens 1024 \
        --temperature 1.0 \
        --top-p 1.0
done

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
for f in outputs/eval/*.jsonl; do
    [ -f "$f" ] || continue
    NAME=$(basename "$f" .jsonl)
    python3 -c "
import json
lines = [json.loads(l) for l in open('$f')]
correct = sum(l['answer_reward'] == 1.0 for l in lines)
total = len(lines)
print(f'  {\"$NAME\":<30} {correct/total:.4f}  ({correct}/{total})')
"
done
