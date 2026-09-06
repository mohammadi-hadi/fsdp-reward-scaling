"""Resuming from a sharded checkpoint, checked exactly.

On CPU the standard is bitwise: a run that stops at step 3 and resumes must produce the same
losses at steps 4 to 6 as one that never stopped. On GPU that standard is unavailable, because
non-associative bf16 accumulation and non-deterministic kernels make two identical
uninterrupted runs differ; there the protocol measures the noise envelope first and asserts
the resume sits inside it. The CPU test is the one that can be exact, so it is exact.
"""

from __future__ import annotations

import pytest
from conftest import run_scenario  # type: ignore[import-not-found]

pytestmark = pytest.mark.distributed


def test_a_resumed_run_reproduces_an_uninterrupted_one_bitwise() -> None:
    result = run_scenario("resume")
    assert result["uninterrupted"] == result["stitched"], (
        "resume drifted:\n"
        f"  uninterrupted {result['uninterrupted']}\n"
        f"  stitched      {result['stitched']}"
    )
    assert result["bitwise_equal"] is True


def test_the_step_counter_survives_the_checkpoint() -> None:
    """Loss agreement alone can hide a resume that replayed or skipped a batch."""
    assert run_scenario("resume")["resumed_step"] == 3
