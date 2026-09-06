"""Timing, memory and throughput, measured so the numbers survive an interview.

Four choices that decide whether a throughput table means anything:

* **Rank-max step time, not rank 0's.** A step ends when the slowest rank ends. Reporting rank
  0 measures one rank's luck.
* **Warmup discarded.** The first steps pay for allocator growth, autotuning and the first
  collective, and folding them in flatters or ruins a row depending on how many you ran.
* **Median and p90, not mean.** One garbage-collection pause moves a mean and leaves a median
  alone, and a mean quietly turns a stall into a throughput claim.
* **Non-padding tokens only**, with the padding share reported separately. A tokens/s number
  that counts padding is the commonest way one of these tables lies.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class StepTimer:
    """CUDA events where they exist, wall clock otherwise."""

    device_type: str
    _start: Any = None
    _end: Any = None
    _t0: float = 0.0

    def __post_init__(self) -> None:
        if self.device_type == "cuda":
            # torch.cuda.Event carries no annotations in torch's stubs.
            self._start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

    def start(self) -> None:
        if self.device_type == "cuda":
            self._start.record()
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self.device_type == "cuda":
            self._end.record()
            torch.cuda.synchronize()
            return float(self._start.elapsed_time(self._end)) / 1000.0
        return time.perf_counter() - self._t0


def rank_max(value: float, device: torch.device) -> float:
    """The slowest rank's value, which is the one that describes the step."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def rank_sum(value: float, device: torch.device) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def memory_snapshot(device_type: str) -> dict[str, float]:
    """Allocated and reserved peaks, plus the retry counter.

    ``num_alloc_retries`` above zero means the allocator hit fragmentation and had to go back
    to the driver, so the throughput number from that run is not clean and should be rerun
    rather than published.
    """
    if device_type != "cuda":
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0, "alloc_retries": 0.0}
    stats = torch.cuda.memory_stats()
    gib = 1024**3
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
        "alloc_retries": float(stats.get("num_alloc_retries", 0)),
    }


def reset_memory(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def mfu(params: int, tokens_per_s: float, world_size: int, peak_flops: float) -> float:
    """Model FLOPs utilisation: 6ND over the hardware peak. Excludes recompute."""
    if peak_flops <= 0 or world_size <= 0:
        return 0.0
    return 6.0 * params * tokens_per_s / (world_size * peak_flops)


def hfu(params: int, tokens_per_s: float, world_size: int, peak_flops: float) -> float:
    """Hardware FLOPs utilisation: 8ND, counting the extra forward that checkpointing pays."""
    if peak_flops <= 0 or world_size <= 0:
        return 0.0
    return 8.0 * params * tokens_per_s / (world_size * peak_flops)


def comm_bytes_per_step(
    params: int, world_size: int, strategy: str, param_bytes: int = 2, reduce_bytes: int = 4
) -> float:
    """Analytic communication volume per rank per step.

    Full-shard gathers parameters twice (forward and backward) and reduce-scatters gradients
    once; grad-op keeps them gathered through the backward and pays one gather. Both move
    ``(N-1)/N`` of each tensor off-rank.
    """
    if world_size <= 1:
        return 0.0
    fraction = (world_size - 1) / world_size
    if strategy == "full_shard":
        return (2 * param_bytes + reduce_bytes) * params * fraction
    if strategy == "grad_op":
        return (param_bytes + reduce_bytes) * params * fraction
    if strategy in {"no_shard", "hsdp"}:
        return 2 * reduce_bytes * params * fraction
    return 0.0


@dataclass
class RunStats:
    """Collects per-step numbers and summarises them the way the table wants."""

    warmup_steps: int
    step_times: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    padding_tokens: list[int] = field(default_factory=list)

    def record(self, *, step_time: float, loss: float, tokens: int, padding: int) -> None:
        self.step_times.append(step_time)
        self.losses.append(loss)
        self.tokens.append(tokens)
        self.padding_tokens.append(padding)

    @property
    def timed(self) -> list[float]:
        return self.step_times[self.warmup_steps :]

    def summary(self) -> dict[str, float]:
        timed = self.timed
        if not timed:
            return {}
        ordered = sorted(timed)
        p90 = ordered[max(0, int(0.9 * len(ordered)) - 1)]
        tokens = self.tokens[self.warmup_steps :]
        total_tokens = sum(tokens)
        total_time = sum(timed)
        padding = sum(self.padding_tokens[self.warmup_steps :])
        return {
            "steps_timed": float(len(timed)),
            "step_time_median_s": statistics.median(timed),
            "step_time_p90_s": p90,
            "tokens_per_s": total_tokens / total_time if total_time else 0.0,
            "padding_share": padding / (total_tokens + padding) if total_tokens + padding else 0.0,
            "final_loss": self.losses[-1] if self.losses else float("nan"),
        }
