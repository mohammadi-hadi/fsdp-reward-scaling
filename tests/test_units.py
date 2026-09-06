"""Single-process tests: config, loss, sampler, metrics."""

from __future__ import annotations

import math

import pytest
import torch

from shardkit.config import load
from shardkit.data import DeterministicSampler, collate, count_real_tokens, synthetic_pairs
from shardkit.loss import bradley_terry_loss, pair_accuracy
from shardkit.metrics import RunStats, comm_bytes_per_step, mfu


def test_overrides_reach_nested_sections(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "c.yaml"
    path.write_text("run_name: base\ntrain:\n  steps: 5\n")
    cfg = load(path, ["train.steps=9", "parallel.strategy=grad_op", "run_name=over"])
    assert (cfg.train.steps, cfg.parallel.strategy, cfg.run_name) == (9, "grad_op", "over")


def test_an_unknown_strategy_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "c.yaml"
    path.write_text("parallel:\n  strategy: nonsense\n")
    with pytest.raises(ValueError, match="strategy must be one of"):
        load(path)


def test_the_loss_is_finite_when_the_model_is_confidently_wrong() -> None:
    """logsigmoid rather than log(sigmoid(x)): the naive form returns inf here."""
    loss = bradley_terry_loss(torch.tensor([-100.0]), torch.tensor([100.0]))
    assert math.isfinite(float(loss))
    assert float(loss) > 100


def test_the_loss_falls_as_the_margin_grows() -> None:
    wide = bradley_terry_loss(torch.tensor([3.0]), torch.tensor([-3.0]))
    narrow = bradley_terry_loss(torch.tensor([0.1]), torch.tensor([-0.1]))
    assert float(wide) < float(narrow)


def test_pair_accuracy_counts_a_tie_as_wrong() -> None:
    assert float(pair_accuracy(torch.tensor([1.0]), torch.tensor([1.0]))) == 0.0


def test_ranks_receive_disjoint_indices() -> None:
    batches = [
        list(DeterministicSampler(64, rank=r, world_size=4, micro_batch=4, seed=7))
        for r in range(4)
    ]
    first_step = [set(b[0]) for b in batches]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not first_step[i] & first_step[j], f"ranks {i} and {j} share indices"


def test_no_index_repeats_within_an_epoch() -> None:
    seen: list[int] = []
    for r in range(2):
        for batch in DeterministicSampler(64, rank=r, world_size=2, micro_batch=8, seed=7):
            seen.extend(batch)
    assert len(seen) == len(set(seen))


def test_resuming_skips_exactly_the_batches_already_consumed() -> None:
    """The property a loss curve cannot check: a resume must not replay or skip examples."""
    whole = list(DeterministicSampler(64, rank=0, world_size=2, micro_batch=4, seed=7))
    resumed = list(
        DeterministicSampler(64, rank=0, world_size=2, micro_batch=4, seed=7, start_step=3)
    )
    assert resumed == whole[3:]


def test_a_different_epoch_reshuffles() -> None:
    first = list(DeterministicSampler(64, rank=0, world_size=2, micro_batch=4, seed=7, epoch=0))
    second = list(DeterministicSampler(64, rank=0, world_size=2, micro_batch=4, seed=7, epoch=1))
    assert first != second


def test_padding_is_excluded_from_the_token_count() -> None:
    pairs = synthetic_pairs(2, vocab=64, length=8, seed=0)
    batch = collate(pairs)
    batch["attention_mask"][:, -3:] = 0
    assert count_real_tokens(batch) == batch["attention_mask"].numel() - 4 * 3


def test_full_shard_moves_more_bytes_than_grad_op() -> None:
    args = dict(params=1_000_000, world_size=4)
    assert comm_bytes_per_step(strategy="full_shard", **args) > comm_bytes_per_step(
        strategy="grad_op", **args
    )


def test_a_single_rank_communicates_nothing() -> None:
    assert comm_bytes_per_step(params=1_000_000, world_size=1, strategy="full_shard") == 0.0


def test_warmup_steps_are_excluded_from_the_summary() -> None:
    stats = RunStats(warmup_steps=2)
    for value in (10.0, 10.0, 1.0, 1.0, 1.0):
        stats.record(step_time=value, loss=0.5, tokens=100, padding=0)
    summary = stats.summary()
    assert summary["steps_timed"] == 3
    assert summary["step_time_median_s"] == 1.0


def test_mfu_is_zero_without_a_hardware_peak() -> None:
    assert mfu(1_000_000, 1000.0, 1, 0.0) == 0.0
