"""Preference pairs, and a sampler you can resume into the middle of.

The sampler is index-based and stateless by construction: given (seed, epoch) it derives the
same permutation every time, and a resume is an offset into it. That matters more than it
looks. DataLoader worker RNG cannot be snapshotted and restored, so any sampler that carries
hidden state cannot be resumed exactly, and a resume that silently replays or skips a few
hundred examples is invisible in the loss curve.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Pair:
    """One preference pair, already tokenised."""

    chosen_ids: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_ids: torch.Tensor
    rejected_mask: torch.Tensor


class DeterministicSampler:
    """A rank-disjoint, resumable index stream.

    ``start_step`` skips whole global batches, so a resumed run consumes exactly the examples
    an uninterrupted run would have consumed next.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        micro_batch: int,
        seed: int = 7,
        epoch: int = 0,
        start_step: int = 0,
    ) -> None:
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.micro_batch = micro_batch
        self.seed = seed
        self.epoch = epoch
        self.start_step = start_step
        self.global_batch = micro_batch * world_size
        if self.global_batch > dataset_size:
            raise ValueError(
                f"global batch {self.global_batch} exceeds dataset size {dataset_size}"
            )

    def permutation(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.dataset_size, generator=generator)

    def steps_per_epoch(self) -> int:
        return self.dataset_size // self.global_batch

    def __iter__(self) -> Iterator[list[int]]:
        order = self.permutation()
        for step in range(self.start_step, self.steps_per_epoch()):
            begin = step * self.global_batch + self.rank * self.micro_batch
            yield order[begin : begin + self.micro_batch].tolist()


def synthetic_pairs(n: int, *, vocab: int, length: int, seed: int = 0) -> list[Pair]:
    """Pairs with a learnable signal and no network access.

    The chosen sequence draws from the upper half of the vocabulary and the rejected one from
    the lower half, so the loss has something real to descend and a smoke test that trains on
    noise cannot pass by accident.
    """
    generator = torch.Generator().manual_seed(seed)
    half = vocab // 2
    pairs: list[Pair] = []
    for _ in range(n):
        chosen = torch.randint(half, vocab, (length,), generator=generator)
        rejected = torch.randint(0, half, (length,), generator=generator)
        ones = torch.ones(length, dtype=torch.long)
        pairs.append(Pair(chosen, ones.clone(), rejected, ones.clone()))
    return pairs


def collate(pairs: list[Pair]) -> dict[str, torch.Tensor]:
    """Stack a micro-batch. Chosen and rejected go through the model as one batch of 2N."""
    return {
        "input_ids": torch.cat(
            [
                torch.stack([p.chosen_ids for p in pairs]),
                torch.stack([p.rejected_ids for p in pairs]),
            ]
        ),
        "attention_mask": torch.cat(
            [
                torch.stack([p.chosen_mask for p in pairs]),
                torch.stack([p.rejected_mask for p in pairs]),
            ]
        ),
        "n_pairs": torch.tensor(len(pairs)),
    }


def count_real_tokens(batch: dict[str, torch.Tensor]) -> int:
    """Non-padding tokens only. Counting padding is the commonest way a tokens/s number lies."""
    return int(batch["attention_mask"].sum().item())
