"""Sharded checkpoints that resume exactly, including where the data got to.

What goes in is the whole point. Model and optimizer are the obvious half; the half that
actually breaks resumes is the rest: the learning-rate schedule, the step counter, per-rank RNG,
and the position in the data stream. A checkpoint carrying only weights produces a run that
looks like it resumed and is quietly training on a different sample of the data.

Saved through ``torch.distributed.checkpoint`` in its sharded form, so a run saved on four
ranks can be loaded on two. That reshard is the part that makes this useful in production,
where the cluster you resume on is rarely the cluster you crashed on.
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as _dcp_load
from torch.distributed.checkpoint.state_dict_saver import async_save as _dcp_async_save
from torch.distributed.checkpoint.state_dict_saver import save as _dcp_save
from torch.distributed.checkpoint.stateful import Stateful

log = logging.getLogger("shardkit")

SHARDED = StateDictOptions(full_state_dict=False, cpu_offload=False)


def rng_state() -> dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_rng_states() -> list[bytes]:
    """Every rank's RNG state, serialised, so a resume gives each rank its own stream back.

    Storing ``rng_state()`` directly does not survive a checkpoint. DCP treats a plain tensor as
    replicated and keeps one copy, so every rank reloads rank 0's stream and the per-rank
    decorrelation that ``train._seed_per_rank`` sets up is silently gone. Measured before the
    fix: two ranks holding different streams both drew 0.030121922 after a restore.

    One ``all_gather_object`` per checkpoint, of about 5 KB per rank. Checkpoints are rare.
    """
    buffer = io.BytesIO()
    torch.save(rng_state(), buffer)
    blob = buffer.getvalue()
    if not _distributed():
        return [blob]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, blob)
    return cast("list[bytes]", gathered)


def restore_rng_states(blobs: Sequence[bytes]) -> None:
    """Give each rank its own stream back, reusing one when the world got wider.

    ``rank % len(blobs)`` is the resharding case: resuming a 2-rank checkpoint on 4 ranks has no
    stored stream for ranks 2 and 3, and sharing one is better than leaving them wherever the
    process happened to be. Narrowing drops the extra streams, which is exact.
    """
    rank = dist.get_rank() if _distributed() else 0
    blob = blobs[rank % len(blobs)]
    set_rng_state(torch.load(io.BytesIO(blob), weights_only=False))


class AppState(Stateful):
    """Everything needed to carry on as though nothing happened."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step = 0
        self.epoch = 0
        self.tokens_seen = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(self.model, options=SHARDED),
            "optimizer": get_optimizer_state_dict(self.model, self.optimizer, options=SHARDED),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "progress": {
                "step": self.step,
                "epoch": self.epoch,
                "tokens_seen": self.tokens_seen,
            },
            "rng": all_rng_states(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, state_dict["model"], options=SHARDED)
        set_optimizer_state_dict(
            self.model, self.optimizer, optim_state_dict=state_dict["optimizer"], options=SHARDED
        )
        if self.scheduler is not None and state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        progress = state_dict["progress"]
        self.step = progress["step"]
        self.epoch = progress["epoch"]
        self.tokens_seen = progress["tokens_seen"]
        restore_rng_states(state_dict["rng"])


def save(state: AppState, path: str | Path, *, async_save: bool = False) -> Any:
    """Write a sharded checkpoint. Returns the future when asynchronous."""
    target = str(path)
    if async_save:
        return _dcp_async_save({"app": state}, checkpoint_id=target)
    _dcp_save({"app": state}, checkpoint_id=target)
    log.info("checkpoint written to %s at step %d", target, state.step)
    return None


def load(state: AppState, path: str | Path) -> None:
    """Load in place. The mesh may differ from the one that saved it."""
    _dcp_load({"app": state}, checkpoint_id=str(path))
    log.info("resumed from %s at step %d", path, state.step)


def latest(root: str | Path) -> Path | None:
    """Newest ``step-N`` directory under ``root``, by step number rather than mtime."""
    base = Path(root)
    if not base.is_dir():
        return None
    candidates = [p for p in base.glob("step-*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))
