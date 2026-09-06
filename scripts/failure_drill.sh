#!/usr/bin/env bash
# Kill a rank mid-run and get the training back.
#
# Three curves on one plot: an uninterrupted reference, the killed-and-resumed run, and a
# no-checkpoint control where the same kill sends the loss back to its starting value. The
# control is what makes the other two mean anything.
set -uo pipefail
RANKS="${RANKS:-4}"
CONFIG="${CONFIG:-configs/qwen3-1.7b.yaml}"
STEPS="${STEPS:-500}"
KILL_AT="${KILL_AT:-300}"

launch () {
  python -m torch.distributed.run --standalone --nproc_per_node="$RANKS" --max-restarts=0 \
    -m shardkit train --config "$CONFIG" "train.steps=$STEPS" "$@"
}

# A and A': identical seeds, uninterrupted. Their difference is the noise envelope, and the
# resume tolerance is derived from it rather than chosen.
launch run_name=drill-ref-a  train.checkpoint_every=50
launch run_name=drill-ref-b  train.checkpoint_every=0

# B: killed at KILL_AT, then resumed from the last checkpoint before it.
launch run_name=drill-killed train.checkpoint_every=50 "train.inject_fail_at_step=$KILL_AT" || true
LATEST=$(ls -d runs/drill-killed/checkpoints/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$LATEST" ] && python -m torch.distributed.run --standalone --nproc_per_node="$RANKS" \
  -m shardkit train --config "$CONFIG" "train.steps=$STEPS" \
  run_name=drill-resumed --resume "$LATEST"

# The control: same kill, no checkpoint to come back to.
launch run_name=drill-control train.checkpoint_every=0 "train.inject_fail_at_step=$KILL_AT" || true

# The reshard: saved on RANKS, resumed on half of them.
HALF=$(( RANKS / 2 ))
[ -n "$LATEST" ] && [ "$HALF" -ge 1 ] && python -m torch.distributed.run --standalone \
  --nproc_per_node="$HALF" -m shardkit train --config "$CONFIG" "train.steps=$STEPS" \
  run_name=drill-resharded --resume "$LATEST"
