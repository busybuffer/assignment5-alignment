#!/usr/bin/env bash
# Sequential GRPO learning-rate sweep for the assignment deliverable (grpo_learning_rate).
# Budget: plan ~1 H100-hour per run × number of LRs (adjust list to fit your 6 H100-hr cap).
#
# Prerequisites: wandb login, data/MATH/*.jsonl, two GPUs (policy + vLLM) unless you override devices.
#
# After runs finish, in Weights & Biases:
#   - Open your project → group by "group" (SWEEP_GROUP) or filter runs by tag "lr_sweep".
#   - Custom chart: X = grpo_step, Y = eval/avg_answer_reward (or eval/accuracy), split/compare by run.
#   - For the write-up: record final-step eval/accuracy (need ≥0.25 on ≥1024 val examples) and note any
#     divergent runs (NaN loss, exploding train/loss or train/grad_norm).
#
# Example discussion template (replace after you inspect logs):
#   "Higher learning rates improved validation answer reward up to ~1e-5, then train/grad_norm grew and
#    accuracy plateaued or degraded; format_reward and response_length tracked answer accuracy, with
#    longer generations as the policy became more confident."

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SWEEP_GROUP="${SWEEP_GROUP:-grpo_lr_sweep_$(date +%Y%m%d)}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:1}"

# Edit this list to match your compute budget (fewer points = fewer GPU-hours).
LEARNING_RATES=(
  # "3e-6"
  # "5e-6"
  # "1e-5"
  # "3e-5"
  # "1e-4"
  # "5e-5"
  # "1.5e-5"
  "4e-5"
  "5e-5"
  # "2e-5"
  # "3e-5"
)

echo "Sweep group: ${SWEEP_GROUP}"
echo "Learning rates: ${LEARNING_RATES[*]}"

for lr in "${LEARNING_RATES[@]}"; do
  # Safe run name for outputs/ and W&B (avoid slashes / odd path chars)
  run_name="${SWEEP_GROUP}_temp1_lr_${lr}"
  echo "========== LR=${lr}  run_name=${run_name} =========="
  python scripts/train_grpo.py \
    --run-name "${run_name}" \
    --wandb-group "${SWEEP_GROUP}" \
    --wandb-tags "lr_sweep,lr_${lr}" \
    --learning-rate "${lr}" \
    --val-examples 1024 \
    --val-every 5 \
    --policy-device "${POLICY_DEVICE}" \
    --vllm-device "${VLLM_DEVICE}" \
    "$@"
done

echo "Done. Compare runs in W&B under group: ${SWEEP_GROUP}"
