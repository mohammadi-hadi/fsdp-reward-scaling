"""Command line: train | smoke | report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shardkit.config import load


def main() -> None:
    parser = argparse.ArgumentParser(prog="shardkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("train", "run a training job under torchrun"),
        ("smoke", "a few steps on the tiny model, for checking the plumbing"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=None)
        p.add_argument("--resume", default=None)
        p.add_argument("overrides", nargs="*", help="section.field=value")

    report = sub.add_parser("report", help="rebuild the results table from committed logs")
    report.add_argument("--runs", default="runs")
    report.add_argument("--check", action="store_true", help="fail if the table would change")

    args = parser.parse_args()

    if args.command == "report":
        from shardkit.report import main as report_main

        sys.exit(report_main(Path(args.runs), check=args.check))

    from shardkit.train import run

    default = "configs/smoke-cpu.yaml" if args.command == "smoke" else None
    cfg = load(args.config or default, args.overrides)
    run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
