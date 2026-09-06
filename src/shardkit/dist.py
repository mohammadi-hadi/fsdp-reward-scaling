"""Process group, device and mesh setup.

One rule worth stating: the device type is chosen explicitly and never inferred from whatever
accelerator happens to be present. On Apple Silicon ``torch.accelerator`` reports MPS, gloo
cannot move an MPS tensor, and the failure arrives deep inside a collective rather than at
startup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

log = logging.getLogger("shardkit")


@dataclass(frozen=True)
class Ranks:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def resolve_device(requested: str) -> str:
    """``auto`` means CUDA when it is really there, and CPU otherwise. Never MPS."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"device_type must be auto, cuda or cpu; got {requested!r}")
    return requested


def env_ranks() -> Ranks:
    return Ranks(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def setup(device_type: str) -> tuple[Ranks, torch.device]:
    """Initialise the process group and return the ranks plus this rank's device."""
    ranks = env_ranks()
    backend = "nccl" if device_type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if device_type == "cuda":
        torch.cuda.set_device(ranks.local_rank)
        device = torch.device("cuda", ranks.local_rank)
    else:
        device = torch.device("cpu")
    configure_logging(ranks)
    log.info("process group up: backend=%s world=%d device=%s", backend, ranks.world_size, device)
    return ranks, device


def build_mesh(device_type: str, world_size: int, strategy: str, replicate: int) -> DeviceMesh:
    """The mesh is where a sharding strategy actually lives.

    FSDP2 has no ``ShardingStrategy`` enum. Full-shard and grad-op are one-dimensional meshes
    that differ only in ``reshard_after_forward``; no-shard is a 2D mesh whose shard dimension
    is 1, so every collective becomes an all-reduce; HSDP shards inside a replica group and
    all-reduces between them.
    """
    if strategy in {"full_shard", "grad_op"}:
        return init_device_mesh(device_type, (world_size,), mesh_dim_names=("shard",))
    if strategy == "no_shard":
        return init_device_mesh(device_type, (world_size, 1), mesh_dim_names=("replicate", "shard"))
    if strategy == "hsdp":
        if world_size % replicate:
            raise ValueError(f"world size {world_size} is not divisible by replicate={replicate}")
        return init_device_mesh(
            device_type,
            (replicate, world_size // replicate),
            mesh_dim_names=("replicate", "shard"),
        )
    raise ValueError(f"unknown strategy {strategy!r}")


def configure_logging(ranks: Ranks) -> None:
    """Rank 0 speaks; the others stay quiet unless something is wrong."""
    logging.basicConfig(
        level=logging.INFO if ranks.is_main else logging.WARNING,
        format=f"[rank{ranks.rank}] %(levelname)s %(message)s",
        force=True,
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
