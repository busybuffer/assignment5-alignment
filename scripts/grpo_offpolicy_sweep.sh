#!/usr/bin/env bash
# Off-policy GRPO hyperparameter sweep (grpo_off_policy_sweep).
# Budget: ~12 H100-hours total.
#
# Fixed: rollout_batch_size=256, loss_type=grpo_clip, lr=3e-5
# Sweep: epochs_per_rollout_batch × train_batch_size
#
# Memory constraint: keep micro_train_batch_size = train_batch_size / gradient_accumulation_steps = 4
#   train_batch_size=256 → gradient_accumulation_steps=64
#   train_batch_size=128 → gradient_accumulation_steps=32
#   train_batch_size=64  → gradient_accumulation_steps=16
#
# Optimizer steps per rollout batch = epochs_per_rollout_batch × (rollout_batch_size / train_batch_size)
#   epochs=1, train=256: 1 step  (on-policy baseline)
#   epochs=2, train=128: 4 steps
#   epochs=2, train=64:  8 steps
#   epochs=4, train=128: 8 steps
#   epochs=4, train=64:  16 steps
#
# PHASE 1 (broad, <50 steps): run all configs, early-stop bad ones.
#   bash scripts/grpo_offpolicy_sweep.sh --n-grpo-steps 50
#
# PHASE 2 (focused, 200 steps): run best 2-3 configs from phase 1.
#   CONFIGS="2x128 4x128" bash scripts/grpo_offpolicy_sweep.sh --n-grpo-steps 200
#
# W&B comparison:
#   - Group by SWEEP_GROUP, X = grpo_step, Y = eval/avg_answer_reward
#   - Also plot vs wall-clock time (W&B supports this natively)
#   - Watch: train/grad_norm (stability), eval/avg_token_entropy (collapse?),
#            eval/avg_response_length (reward hacking?), train/clip_fraction (clipping active?)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SWEEP_GROUP="${SWEEP_GROUP:-grpo_offpolicy_sweep_$(date +%Y%m%d)}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:0}"
VLLM_DEVICE="${VLLM_DEVICE:-cuda:1}"
LR="${LR:-3e-5}"

# Set of configs to run: "epochs:train_batch_size:grad_accum"
# CONFIGS env var can override, e.g. CONFIGS="2x128 4x128"
if [[ -n "${CONFIGS:-}" ]]; then
  # Parse user-specified "EPOCHSxTRAIN_BS" pairs; derive grad_accum assuming microbatch=4
  declare -a RUNS=()
  for cfg in $CONFIGS; do
    epochs="${cfg%%x*}"
    train_bs="${cfg##*x}"
    grad_accum=$(( train_bs / 4 ))
    RUNS+=("${epochs}:${train_bs}:${grad_accum}")
  done
else
  # Default: full broad sweep
  RUNS=(
    "1:256:64"    # on-policy baseline (epochs=1, train=256, microbatch=4)
    "2:128:32"    # epochs=2, 4 optimizer steps per rollout
    "2:64:16"     # epochs=2, 8 optimizer steps per rollout
    "4:128:32"    # epochs=4, 8 optimizer steps per rollout
    "4:64:16"     # epochs=4, 16 optimizer steps per rollout
  )
fi

echo "Sweep group:   ${SWEEP_GROUP}"
echo "Learning rate: ${LR}"
echo "Configs (epochs:train_bs:grad_accum):"
for cfg in "${RUNS[@]}"; do echo "  ${cfg}"; done

for cfg in "${RUNS[@]}"; do
  IFS=: read -r epochs train_bs grad_accum <<< "${cfg}"
  loss_type="grpo_clip"
  # on-policy baseline uses reinforce_with_baseline (no old_log_probs needed)
  if [[ "${epochs}" == "1" && "${train_bs}" == "256" ]]; then
    loss_type="reinforce_with_baseline"
  fi

  run_name="${SWEEP_GROUP}_ep${epochs}_tb${train_bs}"
  echo ""
  echo "========== epochs=${epochs}  train_batch=${train_bs}  grad_accum=${grad_accum}  loss=${loss_type} =========="
  echo "  run_name: ${run_name}"

  python scripts/train_grpo.py \
    --run-name "${run_name}" \
    --wandb-group "${SWEEP_GROUP}" \
    --wandb-tags "offpolicy_sweep,ep${epochs},tb${train_bs}" \
    --loss-type "${loss_type}" \
    --learning-rate "${LR}" \
    --rollout-batch-size 256 \
    --train-batch-size "${train_bs}" \
    --gradient-accumulation-steps "${grad_accum}" \
    --epochs-per-rollout-batch "${epochs}" \
    --val-examples 1024 \
    --val-every 5 \
    --policy-device "${POLICY_DEVICE}" \
    --vllm-device "${VLLM_DEVICE}" \
    "$@"
done

echo ""
echo "Done. Compare runs in W&B under group: ${SWEEP_GROUP}"
