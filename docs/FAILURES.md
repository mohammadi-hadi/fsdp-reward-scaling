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
