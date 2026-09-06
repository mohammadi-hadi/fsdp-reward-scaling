"""The sharding semantics, checked on CPU.

This is the cheapest useful test in the repository. A sharding strategy is a claim about which
collectives run and how often, and that claim holds or fails identically on gloo and NCCL, so
it can be checked on a laptop for the price of two CPU processes. Getting it wrong is otherwise
invisible: a mis-wrapped model still trains, still converges, and simply moves far more data
than it should.

Counts are per optimizer step with L communication units (3 blocks plus the root):

    full_shard   2L all-gather + L reduce-scatter
    grad_op       L all-gather + L reduce-scatter
    no_shard      L all-reduce
"""

from __future__ import annotations

import pytest
from conftest import run_scenario  # type: ignore[import-not-found]

pytestmark = pytest.mark.distributed

UNITS = 4  # 3 transformer blocks + the root


def test_parameters_are_really_sharded() -> None:
    result = run_scenario("sharding")
    assert result["is_dtensor"] is True, "fully_shard did not produce DTensor parameters"
    assert result["placements"] == ["S(0)"], result["placements"]
    assert result["loss"] > 0


def test_full_shard_gathers_twice_and_reduce_scatters_once() -> None:
    result = run_scenario("collectives_full_shard")
    counts = result["counts"]
    assert result["units"] == UNITS
    assert counts["_allgather_base_"] == 2 * UNITS
    assert counts["_reduce_scatter_base_"] == UNITS


def test_grad_op_keeps_parameters_gathered_through_the_backward() -> None:
    """The whole difference from full_shard: one all-gather per unit instead of two."""
    result = run_scenario("collectives_grad_op")
    counts = result["counts"]
    assert counts["_allgather_base_"] == UNITS
    assert counts["_reduce_scatter_base_"] == UNITS


def test_no_shard_degenerates_to_all_reduce() -> None:
    """A shard dimension of 1 is DDP, reached through the same API."""
    result = run_scenario("collectives_no_shard")
    counts = result["counts"]
    assert counts.get("allreduce_", 0) == UNITS
    assert "_reduce_scatter_base_" not in counts


def test_full_shard_moves_more_than_grad_op() -> None:
    """The trade the two strategies exist to express, stated as a comparison."""
    full = run_scenario("collectives_full_shard")["counts"]
    grad_op = run_scenario("collectives_grad_op")["counts"]
    assert full["_allgather_base_"] > grad_op["_allgather_base_"]
    assert full["_reduce_scatter_base_"] == grad_op["_reduce_scatter_base_"]
