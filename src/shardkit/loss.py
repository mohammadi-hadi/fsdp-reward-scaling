"""Bradley-Terry preference loss, and the metric that goes with it."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bradley_terry_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """``-log sigmoid(r_chosen - r_rejected)``, averaged over the batch.

    Computed through ``logsigmoid`` rather than ``log(sigmoid(x))`` so a confidently wrong pair
    gives a finite gradient instead of an inf.
    """
    if chosen.shape != rejected.shape:
        raise ValueError(f"shape mismatch: {tuple(chosen.shape)} vs {tuple(rejected.shape)}")
    return -F.logsigmoid(chosen.float() - rejected.float()).mean()


def pair_accuracy(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Share of pairs the model orders correctly. Ties count as wrong, which is the strict read."""
    return (chosen.float() > rejected.float()).float().mean()
