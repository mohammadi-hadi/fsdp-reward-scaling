#!/usr/bin/env bash
# Two ranks, gloo, a tiny model, a few steps. Seconds.
#
# --launcher env exists because torchrun on macOS is not something I have verified: it prints
# "Redirects are currently not supported in MacOs" on import, and the plain two-process
# launcher below is the one that was actually checked on this laptop. CI uses torchrun.
set -euo pipefail
LAUNCHER="${LAUNCHER:-torchrun}"
RANKS="${RANKS:-2}"
[ "${1:-}" = "--launcher" ] && LAUNCHER="$2"

if [ "$LAUNCHER" = "torchrun" ]; then
  python -m torch.distributed.run --standalone --nproc_per_node="$RANKS" -m shardkit smoke "$@"
else
  export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29517}" WORLD_SIZE="$RANKS"
  pids=()
  for rank in $(seq 0 $((RANKS - 1))); do
    RANK="$rank" LOCAL_RANK="$rank" python -m shardkit smoke &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi
