"""The training loop, and the deliberate crash.

``--inject-fail-at-step`` sends SIGKILL rather than raising. An exception would unwind through
cleanup handlers and flush whatever was pending, which is the opposite of what a real crash
does. SIGKILL leaves the checkpoint exactly as complete as it was a moment before, which is the
state a resume actually has to cope with.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from shardkit import checkpoint as ckpt
from shardkit.config import Config
from shardkit.data import DeterministicSampler, collate, count_real_tokens, synthetic_pairs
from shardkit.dist import build_mesh, cleanup, resolve_device, setup
from shardkit.loss import bradley_terry_loss, pair_accuracy
from shardkit.metrics import (
    RunStats,
    StepTimer,
    comm_bytes_per_step,
    hfu,
    memory_snapshot,
    mfu,
    rank_max,
    reset_memory,
)
from shardkit.model import build, count_parameters
from shardkit.parallel import apply_activation_checkpointing, shard

log = logging.getLogger("shardkit")


def _seed_everything(seed: int, rank: int) -> None:
    """Same data order everywhere, different dropout noise per rank."""
    import random

    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed + rank * 1000)


def run(cfg: Config, resume: str | None = None) -> dict[str, Any]:
    device_type = resolve_device(cfg.device_type)
    ranks, device = setup(device_type)
    _seed_everything(cfg.train.seed, ranks.rank)

    model = build(cfg.model).to(device)
    params = count_parameters(model)
    if cfg.parallel.activation_checkpointing and not cfg.model.tiny:
        apply_activation_checkpointing(model)
    mesh = build_mesh(device_type, ranks.world_size, cfg.parallel.strategy, cfg.parallel.replicate)
    model = shard(model, mesh, cfg.parallel)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(1, cfg.train.steps)
    )
    state = ckpt.AppState(model, optimizer, scheduler)

    out_dir = Path(cfg.out_dir) / cfg.run_name
    if ranks.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        ckpt.load(state, resume)

    pairs = synthetic_pairs(
        max(cfg.data.synthetic_pairs, cfg.data.micro_batch_pairs * ranks.world_size * 8),
        vocab=cfg.model.tiny_vocab if cfg.model.tiny else 32000,
        length=cfg.model.max_length if not cfg.model.tiny else 32,
        seed=cfg.train.seed,
    )
    sampler = DeterministicSampler(
        len(pairs),
        rank=ranks.rank,
        world_size=ranks.world_size,
        micro_batch=cfg.data.micro_batch_pairs,
        seed=cfg.train.seed,
        epoch=state.epoch,
        start_step=state.step,
    )

    stats = RunStats(warmup_steps=cfg.train.warmup_steps)
    timer = StepTimer(device_type)
    reset_memory(device_type)
    indices = iter(sampler)
    started = time.perf_counter()

    for step in range(state.step, cfg.train.steps):
        try:
            batch_indices = next(indices)
        except StopIteration:
            state.epoch += 1
            sampler = DeterministicSampler(
                len(pairs),
                rank=ranks.rank,
                world_size=ranks.world_size,
                micro_batch=cfg.data.micro_batch_pairs,
                seed=cfg.train.seed,
                epoch=state.epoch,
            )
            indices = iter(sampler)
            batch_indices = next(indices)

        batch = collate([pairs[i] for i in batch_indices])
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        n_pairs = int(batch["n_pairs"].item())

        if dist.is_initialized():
            dist.barrier()
        timer.start()

        scores = model(input_ids=input_ids, attention_mask=mask)
        if hasattr(scores, "logits"):
            scores = scores.logits.squeeze(-1)
        chosen, rejected = scores[:n_pairs], scores[n_pairs:]
        loss = bradley_terry_loss(chosen, rejected)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        elapsed = rank_max(timer.stop(), device)
        real_tokens = count_real_tokens(batch)
        state.step = step + 1
        state.tokens_seen += real_tokens * ranks.world_size
        stats.record(
            step_time=elapsed,
            loss=float(loss.detach().item()),
            tokens=real_tokens * ranks.world_size,
            padding=int(mask.numel() - real_tokens),
        )

        if ranks.is_main and (step < 3 or step % 10 == 0):
            log.info(
                "step %d loss %.6f acc %.3f %.3fs",
                step,
                stats.losses[-1],
                float(pair_accuracy(chosen, rejected).item()),
                elapsed,
            )

        if cfg.train.checkpoint_every and state.step % cfg.train.checkpoint_every == 0:
            ckpt.save(state, out_dir / "checkpoints" / f"step-{state.step}")

        if step == cfg.train.inject_fail_at_step and ranks.rank == cfg.train.inject_fail_rank:
            log.warning("injecting a hard failure at step %d on rank %d", step, ranks.rank)
            os.kill(os.getpid(), signal.SIGKILL)

    summary = stats.summary()
    summary.update(memory_snapshot(device_type))
    tokens_per_s = summary.get("tokens_per_s", 0.0)
    summary.update(
        {
            "params": float(params),
            "world_size": float(ranks.world_size),
            "strategy": cfg.parallel.strategy,  # type: ignore[dict-item]
            "wall_s": time.perf_counter() - started,
            "tokens_per_s_per_gpu": tokens_per_s / max(ranks.world_size, 1),
            "mfu": mfu(params, tokens_per_s, ranks.world_size, cfg.peak_flops_per_gpu),
            "hfu": hfu(params, tokens_per_s, ranks.world_size, cfg.peak_flops_per_gpu),
            "comm_gb_per_step": comm_bytes_per_step(params, ranks.world_size, cfg.parallel.strategy)
            / 1e9,
        }
    )

    if ranks.is_main:
        (out_dir / "summary.json").write_text(
            json.dumps({"config": asdict(cfg), "summary": summary}, indent=2, default=str) + "\n"
        )
        with (out_dir / "steps.jsonl").open("w") as handle:
            for i, (t, loss_value) in enumerate(zip(stats.step_times, stats.losses, strict=True)):
                handle.write(json.dumps({"step": i, "step_time_s": t, "loss": loss_value}) + "\n")
        log.info("wrote %s", out_dir / "summary.json")

    cleanup()
    return summary
