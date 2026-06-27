#!/usr/bin/env python3
"""Summarize saved MCMC diagnostics across retrieval output directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


SUMMARY_NAME = "mcmc_diagnostics_summary.json"
MOVEMENT_NAME = "mcmc_parameter_movement.csv"
DIAGNOSTICS_DIR = "diagnostics"

OUTPUT_FIELDS = [
    "label",
    "run_dir",
    "completed",
    "divergences",
    "frac_max_tree",
    "median_num_steps",
    "median_step_size",
    "median_accept_prob",
    "chains_frozen",
    "worst_parameter",
    "notes",
]


def iter_summary_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates = [root] if root.name == SUMMARY_NAME else []
        else:
            candidates = sorted(root.glob(f"**/{DIAGNOSTICS_DIR}/{SUMMARY_NAME}"))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return payload


def run_dir_from_summary(summary_path: Path) -> Path:
    if summary_path.parent.name == DIAGNOSTICS_DIR:
        return summary_path.parent.parent
    return summary_path.parent


def label_from_run(run_dir: Path, summary: dict[str, Any]) -> str:
    label = summary.get("diagnostic_label")
    if isinstance(label, str) and label.strip():
        return label.strip()

    run_config_label = label_from_run_config(run_dir / "run_config.log")
    if run_config_label:
        return run_config_label

    parent = run_dir.parent.name
    if parent in {"transmission", "emission"}:
        return ""
    return parent


def label_from_run_config(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Diagnostic label:"):
                value = line.split(":", 1)[1].strip()
                return value or None
    except OSError:
        return None
    return None


def summarize_run(summary_path: Path) -> dict[str, Any]:
    run_dir = run_dir_from_summary(summary_path)
    try:
        summary = load_json(summary_path)
        flags = summary.get("warning_flags") or {}
        if not isinstance(flags, dict):
            flags = {}
        completed = True
        notes = make_notes(summary, flags)
        worst_parameter = find_worst_parameter(run_dir / DIAGNOSTICS_DIR / MOVEMENT_NAME)
        return {
            "label": label_from_run(run_dir, summary),
            "run_dir": str(run_dir),
            "completed": completed,
            "divergences": summary.get("divergence_count"),
            "frac_max_tree": summary.get("fraction_at_max_tree"),
            "median_num_steps": summary.get("median_num_steps"),
            "median_step_size": summary.get("median_step_size"),
            "median_accept_prob": summary.get("median_accept_prob"),
            "chains_frozen": flags.get("chains_frozen"),
            "worst_parameter": worst_parameter,
            "notes": notes,
        }
    except Exception as exc:
        return {
            "label": "",
            "run_dir": str(run_dir),
            "completed": False,
            "divergences": "",
            "frac_max_tree": "",
            "median_num_steps": "",
            "median_step_size": "",
            "median_accept_prob": "",
            "chains_frozen": "",
            "worst_parameter": "",
            "notes": f"failed to read diagnostics: {exc}",
        }


def make_notes(summary: dict[str, Any], flags: dict[str, Any]) -> str:
    notes: list[str] = []
    for name in sorted(flags):
        if flags.get(name):
            notes.append(name)

    missing = summary.get("missing_extra_fields") or []
    if missing:
        notes.append("missing_extra_fields=" + ",".join(str(item) for item in missing))

    warnings = summary.get("warnings") or []
    for warning in warnings[:3]:
        notes.append(str(warning))
    if len(warnings) > 3:
        notes.append(f"{len(warnings) - 3} more warnings")
    return "; ".join(notes)


def find_worst_parameter(path: Path) -> str:
    if not path.exists():
        return ""
    worst: dict[str, str] | None = None
    worst_key: tuple[float, int] = (-1.0, 0)
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                zero_diff = parse_float(row.get("zero_diff_fraction"))
                unique_count = parse_int(row.get("unique_rounded_1e10"))
                if zero_diff is None:
                    continue
                key = (zero_diff, -unique_count if unique_count is not None else 0)
                if key > worst_key:
                    worst = row
                    worst_key = key
    except OSError:
        return ""
    if worst is None:
        return ""

    parameter = worst.get("parameter", "")
    chain = worst.get("chain", "")
    zero_diff = worst.get("zero_diff_fraction", "")
    unique_count = worst.get("unique_rounded_1e10", "")
    return f"{parameter}[chain={chain}, zero_diff={zero_diff}, unique={unique_count}]"


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def write_csv(rows: list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        stream = sys.stdout
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(normalize_rows(rows))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(normalize_rows(rows))
    print(f"Wrote {len(rows)} diagnostic rows to {output}")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append({field: csv_value(row.get(field)) for field in OUTPUT_FIELDS})
    return normalized


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine saved retrieval MCMC diagnostic summaries into one CSV."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Output root directories, run directories, or summary JSON files to scan.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the combined CSV to this path. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_paths = iter_summary_paths(args.roots)
    rows = [summarize_run(path) for path in summary_paths]
    write_csv(rows, args.output)
    if not summary_paths:
        print("No mcmc_diagnostics_summary.json files found.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
