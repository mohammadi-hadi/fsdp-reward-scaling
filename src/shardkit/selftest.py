"""Multi-rank scenarios the test suite runs in a subprocess.

Each scenario prints one JSON line to stdout from rank 0. The tests launch them with
``torch.distributed.run`` and parse that line. Using this rather than
``torch.testing._internal.common_distributed`` is deliberate: that is a private API and it
changes between releases.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from shardkit import checkpoint as ckpt
from shardkit.config import ModelConfig, ParallelConfig
from shardkit.dist import build_mesh, cleanup, setup
from shardkit.loss import bradley_terry_loss
from shardkit.model import build
from shardkit.parallel import shard

TINY = ModelConfig(tiny=True, tiny_layers=3, tiny_hidden=32, tiny_vocab=64)


def _model_and_optimizer(strategy: str, world_size: int, replicate: int = 1):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    model = build(TINY)
    mesh = build_mesh("cpu", world_size, strategy, replicate)
    cfg = ParallelConfig(
        strategy=strategy,
        replicate=replicate,
        param_dtype="float32",
        reduce_dtype="float32",
        activation_checkpointing=False,
    )
    model = shard(model, mesh, cfg)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    n, length = 2, 8
    ids = torch.randint(0, TINY.tiny_vocab, (2 * n, length), generator=generator)
    mask = torch.ones_like(ids)
    return ids, mask, n


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, seed: int) -> float:
    ids, mask, n = _batch(seed)
    scores = model(input_ids=ids, attention_mask=mask)
    loss = bradley_terry_loss(scores[:n], scores[n:])
    loss.backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().item())


def scenario_sharding() -> dict[str, Any]:
    """Parameters really are DTensors sharded on dim 0, and a step completes."""
    world = dist.get_world_size()
    model, optimizer = _model_and_optimizer("full_shard", world)
    weight = model.blocks[0].linear1.weight
    result = {
        "is_dtensor": isinstance(weight, DTensor),
        "placements": [str(p) for p in weight.placements] if isinstance(weight, DTensor) else [],
        "loss": _step(model, optimizer, seed=1),
    }
    return result


def scenario_collectives(strategy: str, replicate: int = 1) -> dict[str, Any]:
    """Count the collectives one step issues, which is what a strategy name means."""
    from torch.distributed.tensor.debug import CommDebugMode

    world = dist.get_world_size()
    model, optimizer = _model_and_optimizer(strategy, world, replicate)
    with CommDebugMode() as mode:  # type: ignore[no-untyped-call]
        _step(model, optimizer, seed=2)
    counts = {str(k).split(".")[-1]: int(v) for k, v in mode.get_comm_counts().items() if v}
    return {"strategy": strategy, "counts": counts, "units": len(model.blocks) + 1}


def scenario_strategy_equivalence(steps: int = 4) -> dict[str, Any]:
    """The three strategies are the same arithmetic, so they must produce the same losses.

    Sharding changes which rank holds which parameter and which collective moves it. It does not
    change the gradient. Any disagreement here is a bug in the wrapping, not a numerical effect,
    and this is the scenario that catches a model initialised differently on each rank: under
    ``no_shard`` every rank keeps a whole model, so divergent initialisation shows up immediately,
    while under ``full_shard`` the ranks hold complementary chunks and the loss looks plausible.

    Two different assertions, deliberately. The **first step is bitwise identical** or the ranks
    did not start from the same weights: it is a forward pass on the initial parameters, with no
    optimizer arithmetic in front of it, so nothing can excuse a difference. **Later steps drift**
    by a few ulps, because ``clip_grad_norm_`` reduces a global norm over shards in one case and
    over whole tensors in the other, and fp32 addition does not reassociate. Measured at 3e-8 on
    step 4 at world size 2, against the 0.13 the initialisation bug produced.
    """
    world = dist.get_world_size()
    losses = {}
    for strategy in ("full_shard", "grad_op", "no_shard"):
        model, optimizer = _model_and_optimizer(strategy, world)
        losses[strategy] = [_step(model, optimizer, seed=20 + i) for i in range(steps)]
    reference = losses["full_shard"]
    return {
        "losses": losses,
        "first_step_bitwise_equal": all(v[0] == reference[0] for v in losses.values()),
        "max_abs_diff": max(
            abs(a - b) for v in losses.values() for a, b in zip(v, reference, strict=True)
        ),
        "world_size": world,
    }


def scenario_resume() -> dict[str, Any]:
    """Six steps uninterrupted against three, snapshot, restore, three more."""
    world = dist.get_world_size()
    # One path every rank agrees on. A per-rank temp directory looks fine and silently gives
    # each rank its own checkpoint, which DCP then cannot reassemble.
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    shared = Path(tempfile.gettempdir()) / f"shardkit-selftest-{run_id}"
    try:
        path = shared / "ckpt"

        model, optimizer = _model_and_optimizer("full_shard", world)
        straight = [_step(model, optimizer, seed=10 + i) for i in range(6)]

        model_b, optimizer_b = _model_and_optimizer("full_shard", world)
        first = [_step(model_b, optimizer_b, seed=10 + i) for i in range(3)]
        state = ckpt.AppState(model_b, optimizer_b)
        state.step = 3
        ckpt.save(state, path)

        model_c, optimizer_c = _model_and_optimizer("full_shard", world)
        restored = ckpt.AppState(model_c, optimizer_c)
        ckpt.load(restored, path)
        second = [_step(model_c, optimizer_c, seed=13 + i) for i in range(3)]
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(shared, ignore_errors=True)

    return {
        "uninterrupted": straight,
        "stitched": first + second,
        "bitwise_equal": straight == first + second,
        "resumed_step": restored.step,
    }


SCENARIOS = {
    "sharding": scenario_sharding,
    "collectives_full_shard": lambda: scenario_collectives("full_shard"),
    "collectives_grad_op": lambda: scenario_collectives("grad_op"),
    "collectives_no_shard": lambda: scenario_collectives("no_shard"),
    "strategy_equivalence": scenario_strategy_equivalence,
    "resume": scenario_resume,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        raise SystemExit(f"usage: python -m shardkit.selftest <{'|'.join(SCENARIOS)}>")
    setup("cpu")
    try:
        result = SCENARIOS[sys.argv[1]]()
        if dist.get_rank() == 0:
            print("RESULT " + json.dumps(result, default=str), flush=True)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
