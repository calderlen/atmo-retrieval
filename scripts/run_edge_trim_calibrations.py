#!/usr/bin/env python3
"""Run diagnostic-only edge-trim calibrations without Jupyter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.hrs_diagnostic_products import (
    DEFAULT_INPUT_ROOT,
    discover_raw_calibration_inventory,
    planet_slug,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics" / "edge_trim_calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--planet", help="Configured planet name.")
    target.add_argument("--all", action="store_true", help="Run every discovered case.")
    parser.add_argument("--mode", choices=("transmission", "emission"))
    parser.add_argument("--epoch", action="append", help="Restrict to an epoch; repeat as needed.")
    parser.add_argument("--arm", action="append", choices=("blue", "red"))
    parser.add_argument("--list", action="store_true", help="List matching cases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected datasets without running calibration.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _selected_cases(args: argparse.Namespace) -> list[dict]:
    inventory = discover_raw_calibration_inventory(args.input_root)
    requested_planet = planet_slug(args.planet) if args.planet else None
    selected: list[dict] = []
    for (mode, slug), entry in inventory.items():
        if requested_planet is not None and planet_slug(slug) != requested_planet:
            continue
        if args.mode is not None and mode != args.mode:
            continue
        datasets = tuple(
            (epoch, arm)
            for epoch, arm in entry["datasets"]
            if (args.epoch is None or epoch in args.epoch)
            and (args.arm is None or arm in args.arm)
        )
        if datasets:
            selected.append({**entry, "mode": mode, "slug": slug, "datasets": datasets})
    if not selected:
        raise SystemExit("No matching raw edge-trim calibration datasets were found.")
    return selected


def _print_cases(cases: list[dict]) -> None:
    for case in cases:
        datasets = ", ".join(f"{epoch} {arm}" for epoch, arm in case["datasets"])
        print(
            f"{case['mode']}\t{case['planet_display']}\t{case['ephemeris']}\t{datasets}"
        )


def main() -> int:
    args = build_parser().parse_args()
    if not args.list and not args.all and args.planet is None:
        raise SystemExit("Pass --planet, --all, or --list.")
    cases = _selected_cases(args)
    if args.list or args.dry_run:
        _print_cases(cases)
        print("Prepared arrays written: False")
        return 0

    from pipeline.edge_trim_calibration import (
        CalibrationSettings,
        print_summary,
        run_edge_trim_calibration,
    )

    failed = False
    for case in cases:
        result = run_edge_trim_calibration(
            planet=case["planet_display"],
            ephemeris=case["ephemeris"],
            mode=case["mode"],
            datasets=case["datasets"],
            settings=CalibrationSettings(),
            output_root=args.output_dir,
        )
        print_summary(result)
        failed = failed or result["manifest"]["overall_status"] != "accepted_post_sysrem"
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
