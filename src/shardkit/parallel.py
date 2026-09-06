"""Applying a sharding strategy, and the vocabulary for talking about it.

FSDP2 exposes ``fully_shard`` and a device mesh instead of FSDP1's ``ShardingStrategy`` enum,
so the names people say in interviews map onto mesh shape and one boolean:

===============  ==========================================  ==============================
this repo        FSDP2 expression                            collectives per step, L units
===============  ==========================================  ==============================
``full_shard``   1D mesh, ``reshard_after_forward=True``     2L all-gather + L reduce-scatter
``grad_op``      1D mesh, ``reshard_after_forward=False``    L all-gather + L reduce-scatter
``no_shard``     2D mesh ``(world, 1)``                      L all-reduce
``hsdp``         2D mesh ``(replicate, shard)``              shard collectives + all-reduce
===============  ==========================================  ==============================

Those counts are asserted in ``tests/test_parallel_cpu.py`` with no GPU involved, which is the
cheapest way to know a strategy is doing what its name says.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from shardkit.config import ParallelConfig
from shardkit.model import transformer_layers

log = logging.getLogger("shardkit")

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def mixed_precision(cfg: ParallelConfig) -> MixedPrecisionPolicy:
    """bf16 for the gathered parameters, fp32 for the gradient reduction.

    Reducing in fp32 is the setting that matters: bf16 has eight mantissa bits, and summing
    gradients across ranks in it loses small updates in a way that shows up as a training curve
    that quietly stops improving.
    """
    return MixedPrecisionPolicy(
        param_dtype=DTYPES[cfg.param_dtype],
        reduce_dtype=DTYPES[cfg.reduce_dtype],
    )


def apply_activation_checkpointing(model: nn.Module) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    for block in transformer_layers(model):
        for name, child in list(block.named_children()):
            setattr(block, name, checkpoint_wrapper(child))


def shard(model: nn.Module, mesh: DeviceMesh, cfg: ParallelConfig) -> nn.Module:
    """Shard every transformer block, then the root. Returns the same module, modified.

    Per-block first and root last is not stylistic. ``fully_shard`` treats each call as one
    communication unit, so wrapping only the root would all-gather the entire model in one
    collective and defeat the overlap the strategy exists for.
    """
    reshard = cfg.strategy != "grad_op"
    policy = mixed_precision(cfg)

    # Passing these as explicit keywords rather than **kwargs: fully_shard is overloaded, and a
    # dict argument silently drops the type checker's ability to see which overload applies.
    blocks = transformer_layers(model)
    for block in blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
    log.info(
        "sharded %d blocks + root: strategy=%s reshard_after_forward=%s mesh=%s",
        len(blocks),
        cfg.strategy,
        reshard,
        tuple(mesh.shape),
    )
    return model
