<div align="center">

# fsdp-reward-scaling

[![CI](https://github.com/mohammadi-hadi/fsdp-reward-scaling/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/fsdp-reward-scaling/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.13%E2%80%932.14-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Full-parameter reward-model training sharded across GPUs, measured end to end: throughput, memory, scaling efficiency, and what happens when a rank dies.*

</div>

Training a reward model on preference pairs, with every parameter updated, no LoRA adapter,
because that is where sharding stops being a preference and becomes arithmetic. AdamW keeps an fp32 master copy, an fp32 gradient and two fp32 moments per
parameter: **16 bytes per parameter of static state**, replicated on every rank unless
something shards it.

For Qwen3-1.7B that is 25.6 GiB per rank before activations. For Qwen3-4B it is **59.9 GiB,
which does not fit on a 48 GB card at any batch size, including one**.

## What this is, and what it is not

Four L40S GPUs on one rented node, models at 1.7B and 4B parameters, about eight hours of GPU
time. No cluster, no 100B-parameter run, no multi-node interconnect, and no claim that any of
this is production scale. The scope is deliberately the part that does not change with scale:
sharding parameters, gradients and optimizer state across ranks; measuring what that costs in
throughput and what it buys in memory; writing and reloading a sharded checkpoint; and killing
a rank in the middle of a run to get the training back.

The numbers come from `runs/`, which holds per-step JSONL and a resolved config for every
configuration. `make table` regenerates the table below from those logs, and CI fails if the
regenerated table differs from the committed one, so a number here cannot drift from the log
that produced it. Where a column is an upper bound and not a measurement, the header says so. `docs/FAILURES.md` keeps what went wrong on the box, including the parts that were my
fault.

The L40S has no NVLink, so the collectives run over PCIe and the scaling efficiency below is
worse than the same code would show on an NVLink node. That is a property of the hardware I
rented and not of the implementation, which is why the communication-volume column is there.

## Results

<!-- results:begin -->
| run | hardware | ranks | strategy | step p50 (ms) | step p90 (ms) | tokens/s | tokens/s/rank | peak alloc / resvd (GiB) | MFU | comm GB/step |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `smoke-cpu` | smoke (CPU) | 2 | `full_shard` | 12.9 | 13.1 | 39,820 | 19,910 | n/a | n/a | 0.001 |
| `smoke-full_shard` | smoke (CPU) | 2 | `full_shard` | 12.5 | 12.5 | 41,413 | 20,706 | n/a | n/a | 0.001 |
| `smoke-grad_op` | smoke (CPU) | 2 | `grad_op` | 11.3 | 11.3 | 45,565 | 22,782 | n/a | n/a | 0.001 |
| `smoke-no_shard` | smoke (CPU) | 2 | `no_shard` | 11.3 | 11.4 | 45,271 | 22,636 | n/a | n/a | 0.001 |

**Every row above is a CPU smoke run on a four-layer model.** They are here to show the pipeline produces the table, and they say nothing about throughput on real hardware. The GPU sweep replaces them.
<!-- results:end -->

## The strategies, and what they actually do

FSDP2 replaced FSDP1's `ShardingStrategy` enum with a device mesh and one boolean, so the names
people say in interviews map onto mesh shape:

| this repo | ZeRO stage | FSDP2 expression | collectives per step, L units |
|---|---|---|---|
| `full_shard` | 3 | 1D mesh, `reshard_after_forward=True` | 2L all-gather + L reduce-scatter |
| `grad_op` | 2 | 1D mesh, `reshard_after_forward=False` | L all-gather + L reduce-scatter |
| `no_shard` | 0 | 2D mesh `(world, 1)` | L all-reduce |
| `hsdp` | 3 within a replica | 2D mesh `(replicate, shard)` | shard collectives + all-reduce |

**Those counts are asserted in the test suite, on CPU, with no GPU involved.** A sharding
strategy is a claim about which collectives run and how often, and that claim holds or fails
identically on gloo and on NCCL. `tests/test_parallel_cpu.py` counts them with `CommDebugMode`
and fails if a strategy stops doing what its name says. Getting this wrong is otherwise
invisible: a mis-wrapped model still trains and still converges, it just moves far more data
than it should.

DeepSpeed is not here. ZeRO-3 is the same idea through a different implementation, and running
both would double the sweep cost to produce a second set of throughput numbers and no second
insight. The mapping above is the answer to the interview question.

## Resuming, and proving it

A checkpoint carrying only weights produces a run that looks resumed and is quietly training on
a different sample of the data. So `AppState` carries the model, the optimizer, the LR
schedule, the step counter, the token count, per-rank RNG, and the position in the data stream.
The sampler is index-based and derives its permutation from `(seed, epoch)`, because DataLoader
worker RNG cannot be snapshotted and a sampler with hidden state cannot be resumed exactly.

On CPU the standard is **bitwise**: a run that stops at step 3 and resumes must produce exactly
the losses at steps 4 to 6 that an uninterrupted run produced. That is asserted in
`tests/test_checkpoint_cpu.py` and it passes.

On GPU that standard is unavailable. Non-associative bf16 accumulation and non-deterministic
kernels mean two *identical uninterrupted* runs already differ, so the protocol measures the
noise envelope first: run A and A' with the same seed, take `e = max|loss_A - loss_A'|`,
publish `e`, and require the resumed run to sit inside `max(2e, 1e-3)`. Separately, and with
zero tolerance, the step count, the cumulative token count, the set of dataset indices seen and
the LR at every step must match exactly. A loose loss tolerance can hide a resume that replayed
a batch; those four cannot.

Checkpoints are written in DCP's sharded format, so **a run saved on four ranks can be resumed
on two**. That reshard is the part that matters in production, where the cluster you resume on
is rarely the one you crashed on.

## Running it

```bash
pip install -e ".[dev]"
make test          # 21 tests, 7 of them multi-rank on CPU. Under 20 seconds.
make smoke         # 2 ranks, gloo, a four-layer model, a few steps
make table         # regenerate the table above from runs/
```

The smoke path exercises process-group setup, mesh construction, `fully_shard`, the mixed
precision policy, the Bradley-Terry loss, the resumable sampler, and a sharded checkpoint
round-trip. Only NCCL itself and the CUDA memory probes go unexercised until there is a GPU.

On a rented box:

```bash
bash scripts/smoke_cpu.sh                          # cheapest failures first
bash scripts/run_2gpu.sh --steps 5                 # then the CUDA path
bash scripts/bench_sweep.sh                        # the full sweep
bash scripts/failure_drill.sh                      # kill a rank, resume, compare
make table && git diff                             # real numbers replace the smoke rows
```

## Notes on the measurements

- **Step time is the rank maximum**, because a step ends when the slowest rank ends. Reporting
  rank 0 measures one rank's luck.
- **Warmup steps are discarded** and the summary reports **median and p90**, so one
  garbage-collection pause cannot become a throughput claim.
- **Only non-padding tokens are counted**, with the padding share reported separately. A
  tokens/s number that counts padding is the commonest way one of these tables lies.
- `num_alloc_retries` is recorded. Anything above zero means the allocator hit fragmentation,
  and that run's throughput is not clean.
- **MFU is `6ND` over the hardware peak; HFU is `8ND`**, counting the extra forward that
  activation checkpointing pays for. Both are reported, because the difference between them is
  the checkpointing overhead.
- Communication volume is analytic, derived from parameter count, world size and strategy. The
  profiler's NCCL kernel time is an **upper bound** on exposed communication, since collectives
  overlap compute.

## License

MIT.
