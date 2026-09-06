"""Turn committed run logs into the README's table.

The rule this enforces: no number in the README is typed by hand. Every run writes a
``summary.json`` next to its own config, this reads them, and CI regenerates the table and
fails on any difference. A number in the README that nothing in ``runs/`` produces cannot
survive a commit.

The table also states plainly which rows are smoke runs, so a laptop's CPU numbers can never
be mistaken for the GPU sweep they are standing in for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

START = "<!-- results:begin -->"
END = "<!-- results:end -->"


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        payload["run_name"] = summary_path.parent.name
        runs.append(payload)
    return runs


def _row(run: dict[str, Any]) -> str:
    s, cfg = run["summary"], run["config"]
    world = int(s.get("world_size", 1))
    strategy = s.get("strategy", cfg["parallel"]["strategy"])
    device = cfg.get("device_type", "?")
    baseline = "smoke (CPU)" if device == "cpu" else f"{world}x GPU"
    mem = (
        f"{s['peak_allocated_gib']:.1f} / {s['peak_reserved_gib']:.1f}"
        if s.get("peak_allocated_gib")
        else "n/a"
    )
    mfu = f"{s['mfu']:.1%}" if s.get("mfu") else "n/a"
    return (
        f"| `{run['run_name']}` | {baseline} | {world} | `{strategy}` | "
        f"{s['step_time_median_s'] * 1000:.1f} | {s['step_time_p90_s'] * 1000:.1f} | "
        f"{s['tokens_per_s']:,.0f} | {s['tokens_per_s_per_gpu']:,.0f} | {mem} | {mfu} | "
        f"{s['comm_gb_per_step']:.3f} |"
    )


def render(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "_No runs committed yet. `make smoke` writes one._"
    lines = [
        "| run | hardware | ranks | strategy | step p50 (ms) | step p90 (ms) | tokens/s |"
        " tokens/s/rank | peak alloc / resvd (GiB) | MFU | comm GB/step |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [_row(r) for r in runs]
    if all(r["config"].get("device_type") == "cpu" for r in runs):
        lines += [
            "",
            "**Every row above is a CPU smoke run on a four-layer model.** They are here to show "
            "the pipeline produces the table, and they say nothing about throughput on real "
            "hardware. The GPU sweep replaces them.",
        ]
    return "\n".join(lines)


def inject(readme: Path, block: str) -> bool:
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme} has no {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    updated = f"{head}{START}\n{block}\n{END}{tail}"
    if updated == text:
        return False
    readme.write_text(updated)
    return True


def main(runs_dir: Path, *, check: bool = False, readme: Path | None = None) -> int:
    target = readme or Path("README.md")
    block = render(load_runs(runs_dir))
    if check:
        before = target.read_text()
        if inject(target, block):
            target.write_text(before)
            print(f"{target} is stale; run `make table`", file=__import__("sys").stderr)
            return 1
        print(f"{target} matches the committed runs")
        return 0
    changed = inject(target, block)
    print(f"{'updated' if changed else 'unchanged'} {target}")
    return 0
