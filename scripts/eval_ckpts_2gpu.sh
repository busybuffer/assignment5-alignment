#!/usr/bin/env bash
# Evaluate all SFT checkpoints on the full validation set (5000 examples).
# Usage: bash scripts/eval_sft_checkpoints.sh
#
# Results are saved to outputs/eval/{run_name}.jsonl
# Uses two GPUs in parallel: cuda:0 and cuda:1

set -e

DATA_PATH="data/MATH/validation.jsonl"
mkdir -p outputs/eval

(
  # for CKPT_DIR in outputs/sft_full outputs/sft_256 outputs/sft_1024; do
  for CKPT_DIR in outputs/ei_G4_e1_db512 outputs/ei_G4_e2_db512 outputs/ei_G4_e1_db1024; do
    if [ ! -d "$CKPT_DIR" ]; then
      echo "Skipping $CKPT_DIR (not found)"
      continue
    fi
    RUN_NAME=$(basename "$CKPT_DIR")
    echo "=============================="
    echo "[GPU 0] Evaluating: $RUN_NAME"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=0 python scripts/math_baseline.py \
        --model "$CKPT_DIR" \
        --data-path "$DATA_PATH" \
        --output-path "outputs/eval/${RUN_NAME}.jsonl" \
        --max-tokens 1024 \
        --temperature 0.0
  done
) &

(
  # for CKPT_DIR in outputs/sft_128 outputs/sft_512 outputs/sft_filtered_correct; do
  for CKPT_DIR in outputs/ei_G8_e1_db512 outputs/ei_G16_e1_db512; do
    if [ ! -d "$CKPT_DIR" ]; then
      echo "Skipping $CKPT_DIR (not found)"
      continue
    fi
    RUN_NAME=$(basename "$CKPT_DIR")
    echo "=============================="
    echo "[GPU 1] Evaluating: $RUN_NAME"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=1 python scripts/math_baseline.py \
        --model "$CKPT_DIR" \
        --data-path "$DATA_PATH" \
        --output-path "outputs/eval/${RUN_NAME}.jsonl" \
        --max-tokens 1024 \
        --temperature 0.0
  done
) &

wait

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
python3 -c "
import json, glob
rows = []
for f in glob.glob('outputs/eval/*.jsonl'):
    lines = [json.loads(l) for l in open(f)]
    correct = sum(l['answer_reward'] == 1.0 for l in lines)
    total = len(lines)
    name = f.split('/')[-1].replace('.jsonl', '')
    rows.append((correct / total, name, correct, total))
for acc, name, correct, total in sorted(rows, reverse=True):
    print(f'  {name:<30} {acc:.4f}  ({correct}/{total})')
"
