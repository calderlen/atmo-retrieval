#!/usr/bin/env python3
"""Render prepared HRS product diagnostics without Jupyter."""

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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_utils
from dataio.hrs_diagnostic_products import (
    DEFAULT_INPUT_ROOT,
    bundle_summary,
    discover_prepared_inventory,
    load_product_records,
    resolve_product_specs,
)
from dataio.orbital_velocity import planet_radial_velocity_kms
from plotting.spectral_diagnostics import (
    plot_bundle_overview,
    plot_bundle_profiles,
    plot_collapsed_product,
    plot_column_quality,
    plot_line_diagnostic,
    plot_observed_spectrum,
    plot_stacked_spectra,
)
from plotting.style import save_figure_pdf
from spectroscopy.spectral_diagnostics import (
    build_stacked_spectrum_bundle,
    contrast_summary,
    region_metrics,
    stack_line_window,
    wave_1d,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics" / "hrs_products"
PRODUCT_CHOICES = ("timeseries", "collapse_source", "both")
VACUUM_DIAGNOSTIC_LINES = (
    ("H beta", 4862.68),
    ("Mg I b2", 5174.13),
    ("H alpha", 6564.61),
    ("Li I", 6709.66),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--planet", help="Configured planet name")
    target.add_argument("--all", action="store_true", help="Render every discovered target/mode")
    parser.add_argument("--mode", choices=("transmission", "emission"))
    parser.add_argument("--product", choices=PRODUCT_CHOICES, default="timeseries")
    parser.add_argument("--epoch", action="append", help="Epoch to include; repeat as needed")
    parser.add_argument("--arm", action="append", choices=("blue", "red"))
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-lines", type=int, default=4)
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _line_rows_and_figures(bundle: dict[str, Any], output_dir: Path, max_lines: int):
    wavelength = wave_1d(bundle["wavelength"])
    params = config_utils.get_params(
        bundle["planet"],
        bundle.get("metadata", {}).get("ephemeris") or "Recommended",
    )
    kp = float(params.get("Kp", np.nan))
    velocity = None
    if np.isfinite(kp) and kp > 0:
        eccentricity = float(params.get("eccentricity", 0.0) or 0.0)
        omega = params.get("omega")
        omega = float(omega) if omega is not None and np.isfinite(float(omega)) else None
        velocity = planet_radial_velocity_kms(
            np.asarray(bundle["phase"], dtype=float),
            kp_kms=kp,
            eccentricity=eccentricity,
            omega_deg=omega,
        )
    rows: list[dict[str, Any]] = []
    figures: list[str] = []
    selected = [
        (label, rest) for label, rest in VACUUM_DIAGNOSTIC_LINES
        if float(np.nanmin(wavelength)) <= rest <= float(np.nanmax(wavelength))
    ][: max(0, max_lines)]
    for index, (label, rest) in enumerate(selected, start=1):
        result = stack_line_window(
            bundle,
            rest_vacuum_A=rest,
            velocity_kms=velocity,
        )
        finite = np.asarray(result["coadd"])[np.isfinite(result["coadd"])]
        rows.append(
            {
                "epoch": bundle["epoch"],
                "arm": bundle["arm"],
                "product": bundle["product"],
                "line": label,
                "rest_vacuum_A": rest,
                "coadd_rms": float(np.sqrt(np.mean(np.square(finite)))) if finite.size else np.nan,
                "minimum_coverage": int(np.min(result["coverage"])),
                "maximum_coverage": int(np.max(result["coverage"])),
            }
        )
        figure = plot_line_diagnostic(
            result,
            title=f"{bundle['planet']} {bundle['epoch']} {bundle['arm']} {label}",
        )
        saved = _save(figure, output_dir / f"line_{index:02d}_{label.lower().replace(' ', '_')}.pdf")
        if saved:
            figures.append(saved)
    return rows, figures


def render_case(
    *,
    planet: str,
    mode: str,
    product: str,
    epochs: tuple[str, ...] | None,
    arms: tuple[str, ...] | None,
    input_root: Path,
    output_root: Path,
    max_lines: int,
) -> Path:
    specs = resolve_product_specs(
        planet=planet,
        mode=mode,
        product=product,
        epochs=epochs,
        arms=arms,
        input_root=input_root,
    )
    if not specs:
        raise FileNotFoundError(f"No {planet} {mode} {product} bundles found under {input_root}.")
    bundles = load_product_records(specs)
    slug = specs[0]["planet_slug"]
    run_dir = output_root / mode / slug / product / _stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries = [bundle_summary(bundle) for bundle in bundles]
    regions: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    for bundle in bundles:
        prefix = f"{bundle['epoch']}_{bundle['arm']}"
        bundle_dir = run_dir / prefix
        bundle_dir.mkdir()
        figures = (
            ("01_matrix.pdf", plot_bundle_overview(bundle)),
            ("02_profiles.pdf", plot_bundle_profiles(bundle)),
            ("03_observed_spectrum.pdf", plot_observed_spectrum(bundle)),
            ("04_column_quality.pdf", plot_column_quality(bundle)),
            ("05_collapsed_product.pdf", plot_collapsed_product(bundle)),
        )
        for filename, figure in figures:
            saved = _save(figure, bundle_dir / filename)
            if saved:
                figure_paths.append(saved)
        regions.extend(region_metrics(bundle))
        current_rows, current_figures = _line_rows_and_figures(bundle, bundle_dir, max_lines)
        line_rows.extend(current_rows)
        figure_paths.extend(current_figures)

    stacked = build_stacked_spectrum_bundle(bundles, group_by="epoch_arm")
    stacked_path = _save(plot_stacked_spectra(stacked), run_dir / "stacked_spectra.pdf")
    if stacked_path:
        figure_paths.append(stacked_path)
    contrasts = contrast_summary(stacked)
    _write_csv(run_dir / "bundle_summary.csv", summaries)
    _write_csv(run_dir / "region_metrics.csv", regions)
    _write_csv(run_dir / "line_metrics.csv", line_rows)
    _write_csv(run_dir / "stack_contrasts.csv", contrasts)
    report = {
        "kind": "hrs_product_diagnostics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "planet": planet,
        "mode": mode,
        "product": product,
        "input_root": str(input_root),
        "epochs": list(epochs or []),
        "arms": list(arms or []),
        "bundles": summaries,
        "figures": figure_paths,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    products = ("timeseries", "collapse_source") if args.product == "both" else (args.product,)
    if args.all:
        inventory = discover_prepared_inventory(args.input_root)
        for (mode, _slug), entry in inventory.items():
            if args.mode and mode != args.mode:
                continue
            for product in products:
                render_case(
                    planet=entry["planet_display"],
                    mode=mode,
                    product=product,
                    epochs=tuple(args.epoch) if args.epoch else None,
                    arms=tuple(args.arm) if args.arm else None,
                    input_root=args.input_root,
                    output_root=args.output_dir,
                    max_lines=args.max_lines,
                )
        return
    if args.mode is None:
        raise SystemExit("--mode is required with --planet.")
    for product in products:
        render_case(
            planet=args.planet,
            mode=args.mode,
            product=product,
            epochs=tuple(args.epoch) if args.epoch else None,
            arms=tuple(args.arm) if args.arm else None,
            input_root=args.input_root,
            output_root=args.output_dir,
            max_lines=args.max_lines,
        )


if __name__ == "__main__":
    main()
