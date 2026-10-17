#!/usr/bin/env python3
"""Compare prepared HRS spectra across epochs, arms, and product types."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.hrs_diagnostic_products import (
    DEFAULT_INPUT_ROOT,
    bundle_summary,
    discover_prepared_inventory,
    load_product_records,
    resolve_product_specs,
)
from plotting.spectral_diagnostics import (
    plot_bundle_overview,
    plot_column_quality,
    plot_observed_spectrum,
    plot_stacked_spectra,
)
from plotting.style import save_figure_pdf
from spectroscopy.spectral_diagnostics import (
    build_stacked_spectrum_bundle,
    contrast_summary,
    region_metrics,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics" / "spectral_processing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--planet")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--mode", choices=("transmission", "emission"))
    parser.add_argument("--epoch", action="append")
    parser.add_argument("--arm", action="append", choices=("blue", "red"))
    parser.add_argument("--variant", choices=("untrimmed", "edge-trimmed"), default="untrimmed")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--edge-trim-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _save(figure, path: Path) -> str | None:
    if figure is None:
        return None
    saved = save_figure_pdf(figure, path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(saved)


def render_case(
    *,
    planet: str,
    mode: str,
    epochs: tuple[str, ...] | None,
    arms: tuple[str, ...] | None,
    variant: str,
    bundle_root: Path,
    manifest_path: Path | None,
    output_root: Path,
) -> Path:
    specs = []
    for product in ("timeseries", "collapse_source"):
        specs.extend(
            resolve_product_specs(
                planet=planet,
                mode=mode,
                product=product,
                epochs=epochs,
                arms=arms,
                input_root=bundle_root,
            )
        )
    bundles = load_product_records(specs)
    selected_manifest = manifest_path.resolve() if variant == "edge-trimmed" else None

    slug = specs[0]["planet_slug"]
    run_dir = output_root / mode / slug / variant / _stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries = [bundle_summary(bundle) for bundle in bundles]
    regions: list[dict[str, Any]] = []
    figures: list[str] = []
    for bundle in bundles:
        prefix = f"{bundle['epoch']}_{bundle['arm']}_{bundle['product']}"
        bundle_dir = run_dir / prefix
        bundle_dir.mkdir()
        regions.extend(region_metrics(bundle))
        for filename, figure in (
            ("01_processing_matrix.pdf", plot_bundle_overview(bundle)),
            ("02_observed_spectrum.pdf", plot_observed_spectrum(bundle)),
            ("03_column_quality.pdf", plot_column_quality(bundle)),
        ):
            saved = _save(figure, bundle_dir / filename)
            if saved:
                figures.append(saved)

    stacked = build_stacked_spectrum_bundle(bundles, group_by="epoch_arm")
    stacked_path = _save(plot_stacked_spectra(stacked), run_dir / "cross_product_stacks.pdf")
    if stacked_path:
        figures.append(stacked_path)
    contrasts = contrast_summary(stacked)
    _write_csv(run_dir / "bundle_summary.csv", summaries)
    _write_csv(run_dir / "region_metrics.csv", regions)
    _write_csv(run_dir / "stack_contrasts.csv", contrasts)
    report = {
        "kind": "spectral_processing_diagnostics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "planet": planet,
        "mode": mode,
        "variant": variant,
        "bundle_root": str(bundle_root),
        "edge_trim_manifest": str(selected_manifest) if selected_manifest else None,
        "bundles": summaries,
        "figures": figures,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    if args.all:
        for (mode, _slug), entry in discover_prepared_inventory(args.bundle_root).items():
            if args.mode and mode != args.mode:
                continue
            render_case(
                planet=entry["planet_display"],
                mode=mode,
                epochs=tuple(args.epoch) if args.epoch else None,
                arms=tuple(args.arm) if args.arm else None,
                variant=args.variant,
                bundle_root=args.bundle_root,
                manifest_path=args.edge_trim_manifest,
                output_root=args.output_dir,
            )
        return
    render_case(
        planet=args.planet,
        mode=args.mode,
        epochs=tuple(args.epoch) if args.epoch else None,
        arms=tuple(args.arm) if args.arm else None,
        variant=args.variant,
        bundle_root=args.bundle_root,
        manifest_path=args.edge_trim_manifest,
        output_root=args.output_dir,
    )


if __name__ == "__main__":
    main()
