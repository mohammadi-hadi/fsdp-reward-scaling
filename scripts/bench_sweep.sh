#!/usr/bin/env bash
# The sweep. An OOM is a result, not a crash: the row is recorded and the sweep continues.
#
# Cut order if the box is expensive: drop the 4B rows except 4-GPU full_shard and the 1-GPU
# OOM (the OOM is free and carries the argument); then the HSDP row; then the AC-off row.
# Never cut the failure drill or the resharded resume.
set -uo pipefail
CONFIG="${CONFIG:-configs/qwen3-1.7b.yaml}"

run () {
  local ranks="$1" strategy="$2" name="$3"; shift 3
  echo "=== $name (${ranks} ranks, ${strategy})"
  if ! python -m torch.distributed.run --standalone --nproc_per_node="$ranks" \
        -m shardkit train --config "$CONFIG" \
        "run_name=$name" "parallel.strategy=$strategy" "$@" \
        > "runs/${name}.log" 2>&1; then
    if grep -qi "out of memory" "runs/${name}.log"; then
      echo "    OOM, which is the result for this row"
      mkdir -p "runs/$name"
      grep -i -A5 "out of memory" "runs/${name}.log" | head -40 > "runs/$name/oom.txt"
    else
      echo "    FAILED, see runs/${name}.log"
    fi
  fi
}

for ranks in 1 2 4; do
  run "$ranks" no_shard   "1.7b-${ranks}gpu-ddp"
  run "$ranks" full_shard "1.7b-${ranks}gpu-full-shard"
  run "$ranks" grad_op    "1.7b-${ranks}gpu-grad-op"
done
run 4 hsdp "1.7b-4gpu-hsdp" parallel.replicate=2
run 4 full_shard "1.7b-4gpu-full-shard-no-ac" parallel.activation_checkpointing=false
run 4 full_shard "1.7b-4gpu-full-shard-bf16-reduce" parallel.reduce_dtype=bfloat16

for ranks in 1 2 4; do
  run "$ranks" full_shard "4b-${ranks}gpu-full-shard" model.name=Qwen/Qwen3-4B-Base
done
run 1 no_shard "4b-1gpu-ddp" model.name=Qwen/Qwen3-4B-Base   # expected OOM, and that is the point
