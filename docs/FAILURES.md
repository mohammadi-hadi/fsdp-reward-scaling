# What went wrong

Kept because the throughput table is the part of this repository anyone could reproduce, and
this is the part they cannot. Entries are in the order they happened. The ones that were my own
fault say so.

## Before any GPU: building it on a laptop

**`CommDebugMode` is not where the tutorials put it.**
`torch.distributed._tools.comm_mode` does not exist in torch 2.13. The public path is
`torch.distributed.tensor.debug.CommDebugMode`. The private path appears in a lot of
third-party writing about FSDP2 and is presumably older. Cost: one confused minute. Worth
noting because the whole collective-counting test rests on it.

**Both ranks made their own temporary directory.**
The resume scenario called `tempfile.TemporaryDirectory()` inside the function that every rank
runs, so rank 0 and rank 1 each created a different directory and each wrote half a checkpoint
into it. `dcp.load` then failed on a missing `.metadata`. The error pointed at the checkpoint
format, and the bug was that the two ranks never agreed where the checkpoint was.

Mine. The general shape is worth remembering: in SPMD code, anything that generates a name
generates a *different* name on every rank unless it is derived from something shared.

**Seeding per rank before building the model.**
`_seed_everything` ended with `torch.manual_seed(seed + rank * 1000)`, which ran before
`build()`. The intent was decorrelated dropout; the effect was that every rank initialised a
different random model. `fully_shard` chunks the local tensor and never broadcasts, so rank 0
kept chunk 0 of model A and rank 1 kept chunk 1 of model B. Nothing crashes and the loss still
falls, because after the first all-gather the ranks agree on whatever they happen to hold.

Mine, and the kind that survives a green test suite. Worse: the evidence was already committed.
`runs/` held four smoke runs, and `no_shard` finished at 0.39 where `full_shard` and `grad_op`
both gave 0.26. The three strategies are the same arithmetic and must agree. Nobody had diffed
the rows, including me, because each one on its own looks like a loss curve going down.

Under `no_shard` every rank keeps a whole model, so divergent initialisation shows up
immediately; under sharding the ranks hold complementary chunks and the curve looks fine. The
per-rank offset now runs after `shard()`, and `test_strategies_agree_on_the_loss` asserts the
invariant so the rows cannot drift apart unnoticed again. Loading pretrained weights hides the
bug entirely, which is why it would have shipped: `from_pretrained` gives every rank the same
tensors whatever the seed is.

**Then the fix broke the resume.** With the ranks deliberately seeded apart after sharding, the
per-rank RNG became load-bearing, and `AppState` was storing `torch.get_rng_state()` as a plain
tensor. DCP treats a plain tensor as replicated and keeps one copy, so every rank came back on
the same stream: two ranks holding different streams both drew 0.030121922 after a restore.
Silent, and invisible to the bitwise resume test, which seeds every rank the same.

The states are now gathered and stored per rank, `restore_rng_states` hands each rank its own
back, and `test_a_resume_gives_each_rank_its_own_rng_stream` checks the exact draw. The general
lesson is the one above turned around: a checkpoint replicates whatever is not explicitly
sharded, so anything that is meant to differ per rank has to say so.

**`fully_shard(**kwargs)` type-checks as nothing.**
Passing the arguments as a dict made mypy fall back to "no overload matches", which was correct
and unhelpful. Passing them as explicit keywords restored the check. Left as explicit keywords
with a comment, because the dict version silently accepted a misspelled argument name.

## What the CPU tests cannot cover

Written down so the gap is visible rather than implied.

* NCCL. The gloo path exercises the same collectives with the same shapes and counts, and that
  is what the sharding tests assert, but it is a different transport.
* The CUDA memory probes. `max_memory_allocated`, `max_memory_reserved` and
  `num_alloc_retries` return zeros on CPU, so every memory column is `n/a` until there is a GPU.
* Anything about overlap. Whether a collective actually hides behind compute is a property of
  the hardware and the stream schedule.
* bf16 numerics at scale. The smoke config runs fp32 for readability; bf16 works on gloo and is
  exercised by the mixed-precision policy, but a laptop cannot show what accumulating gradients
  in bf16 across four ranks does to a loss curve over a thousand steps.

## On the rented box

_Nothing yet. This section fills in during the sweep, including the parts that are my fault._
