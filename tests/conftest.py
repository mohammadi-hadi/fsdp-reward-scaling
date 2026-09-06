"""Helpers for launching multi-rank scenarios from a single-process test."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest


def run_scenario(name: str, nproc: int = 2, timeout: int = 300) -> dict[str, Any]:
    """Launch ``shardkit.selftest`` under torchrun and parse rank 0's JSON line."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "-m",
            "shardkit.selftest",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in done.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    pytest.fail(
        f"scenario {name!r} produced no result (exit {done.returncode})\n"
        f"stdout tail:\n{done.stdout[-1500:]}\nstderr tail:\n{done.stderr[-2000:]}"
    )
