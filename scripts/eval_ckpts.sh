#!/usr/bin/env bash
# Unified checkpoint evaluation script.
# 
# Features:
# - auto-discovers model checkpoints under outputs/
# - or evaluates checkpoint dirs passed as arguments
# - uses two GPUs in parallel by splitting checkpoints across workers
# - saves results to outputs/eval/<name>.jsonl
# - reports both answer_reward and format_reward summaries
#
# Usage:
#   bash scripts/eval_ckpts.sh
#   bash scripts/eval_ckpts.sh outputs/sft_full outputs/ei_G4_e1_db512
#   bash scripts/eval_ckpts.sh outputs/sft_eval1*/
#   bash scripts/eval_ckpts.sh outputs/ei*/
#   TEMPERATURE=1.0 bash scripts/eval_ckpts.sh outputs/grpo_run/checkpoint_step50

set -euo pipefail

DATA_PATH="${DATA_PATH:-data/MATH/validation.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/eval}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$OUT_DIR"

make_run_name() {
    local ckpt_dir="$1"
    local base
    local parent
    base="$(basename "$ckpt_dir")"
    parent="$(basename "$(dirname "$ckpt_dir")")"

    if [[ "$base" == checkpoint* ]]; then
        printf '%s__%s' "$parent" "$base"
    else
        printf '%s' "$base"
    fi
}

collect_ckpts() {
    if [ "$#" -gt 0 ]; then
        printf '%s\n' "$@"
        return
    fi

    find outputs \
        -path 'outputs/eval' -prune -o \
        -type f -name 'config.json' -print | \
        sed 's#/config.json$##' | \
        sort -u
}

mapfile -t CKPT_DIRS < <(collect_ckpts "$@")

if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
    echo "No checkpoints found."
    exit 1
fi

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
        run_name="$(make_run_name "$ckpt_dir")"
        out_file="${OUT_DIR}/${run_name}.jsonl"

        if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_file" ]; then
            echo "Skipping $run_name (already evaluated)"
            continue
        fi

        echo "=============================="
        echo "[GPU ${gpu_id}] Evaluating: $run_name"
        echo "  ckpt: $ckpt_dir"
        echo "  temp: $TEMPERATURE"
        echo "=============================="

        CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/math_baseline.py \
            --model "$ckpt_dir" \
            --data-path "$DATA_PATH" \
            --output-path "$out_file" \
            --max-tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE"
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
from pathlib import Path

rows = []
for f in sorted(glob.glob('${OUT_DIR}/*.jsonl')):
    path = Path(f)
    lines = [json.loads(l) for l in open(path) if l.strip()]
    if not lines:
        continue
    format_correct = sum(l['format_reward'] == 1.0 for l in lines)
    answer_correct = sum(l['answer_reward'] == 1.0 for l in lines)
    total = len(lines)
    rows.append((
        answer_correct / total,
        format_correct / total,
        path.stem,
        answer_correct,
        format_correct,
        total,
    ))

for answer_acc, format_acc, name, answer_correct, format_correct, total in sorted(rows, reverse=True):
    print(
        f'  {name:<45} '
        f'answer={answer_acc:.4f} ({answer_correct}/{total})  '
        f'format={format_acc:.4f} ({format_correct}/{total})'
    )
"
