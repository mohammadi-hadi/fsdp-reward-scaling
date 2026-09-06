#!/usr/bin/env bash
set -euo pipefail
python -m torch.distributed.run --standalone --nproc_per_node=2 -m shardkit train \
  --config "${CONFIG:-configs/qwen3-1.7b.yaml}" "$@"
