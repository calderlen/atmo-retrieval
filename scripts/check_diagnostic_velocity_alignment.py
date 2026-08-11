#!/usr/bin/env python3
"""Regression check for velocity sampling in spectral diagnostic plots.

This is intentionally a standalone check rather than a formal test module.  It
plants a moving line whose observed-frame trail exceeds the standard diagnostic
window, then verifies that direct planet-frame sampling recovers the injected
offset with complete contribution coverage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "atmo_retrieval_matplotlib"),
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
from plotting.style import configure_matplotlib, save_figure_pdf

matplotlib.use("Agg")
configure_matplotlib()
import matplotlib.pyplot as plt
import numpy as np

from plotting.transmission_diagnostics import (  # noqa: E402
    C_KMS,
    plot_species_2d_atlas,
    species_frame_stack_results,
)


LINE = {
    "label": "synthetic 5000",
    "species": "synthetic",
    "rest_vacuum_A": 5000.0,
    "rest_wavelength_medium": "vacuum",
}


def doppler_factor(velocity_kms):
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def make_bundle(injected_offset_kms, *, kp_kms=240.0):
    phase = np.linspace(0.36, 0.64, 31)
    trail = kp_kms * np.sin(2.0 * np.pi * phase)
    native_velocity = np.linspace(-450.0, 450.0, 7201)
    wavelength = LINE["rest_vacuum_A"] * doppler_factor(native_velocity)
    sigma = np.full((phase.size, wavelength.size), 2.0e-4)
    data = np.empty_like(sigma)
    for row, trail_velocity in enumerate(trail):
        line_center = (
            LINE["rest_vacuum_A"]
            * doppler_factor(trail_velocity)
            * doppler_factor(injected_offset_kms)
        )
        relative_velocity = C_KMS * np.tanh(np.log(wavelength / line_center))
        data[row] = 0.004 * np.exp(-0.5 * np.square(relative_velocity / 3.0))
    return {
        "epoch": "synthetic",
        "arm": "validation",
        "phase": phase,
        "wavelength": wavelength,
        "data": data,
        "sigma": sigma,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-atlas",
        type=Path,
        help="Optional output path for the synthetic observed-frame atlas (saved as PDF).",
    )
    args = parser.parse_args()

    kp_kms = 240.0
    failures = []
    for injected_offset in (-25.0, 0.0, 17.0):
        bundle = make_bundle(injected_offset, kp_kms=kp_kms)
        results = species_frame_stack_results(
            bundle,
            line_catalog=(LINE,),
            species=("synthetic",),
            window_kms=150.0,
            bin_kms=0.25,
            sigma_threshold=0.5,
            kp_kms=kp_kms,
            vsys_kms=0.0,
        )
        if len(results) != 1:
            failures.append(f"offset {injected_offset:+.1f}: no synthetic result")
            continue
        result = results[0]
        recovered = float(result["velocity"][np.nanargmax(result["mean"])])
        error = recovered - injected_offset
        print(
            f"injected={injected_offset:+6.2f} km/s  "
            f"recovered={recovered:+6.2f} km/s  error={error:+.3f} km/s  "
            f"coverage@0={result['center_coverage_fraction']:.1%}  "
            f"min_coverage={result['minimum_coverage_fraction']:.1%}"
        )
        if abs(error) > 0.30:
            failures.append(
                f"offset {injected_offset:+.1f}: recovery error {error:+.3f} km/s"
            )
        if result["minimum_coverage_fraction"] < 1.0:
            failures.append(
                f"offset {injected_offset:+.1f}: incomplete output-grid coverage"
            )
        if result["coverage_warning"]:
            failures.append(f"offset {injected_offset:+.1f}: unexpected coverage warning")

    atlas_bundle = make_bundle(0.0, kp_kms=kp_kms)
    atlas = plot_species_2d_atlas(
        atlas_bundle,
        line_catalog=(LINE,),
        species=("synthetic",),
        window_kms=150.0,
        bin_kms=2.0,
        sigma_threshold=0.5,
        kp_kms=kp_kms,
        vsys_kms=0.0,
        ncols=1,
    )
    if atlas is None:
        failures.append("observed-frame atlas produced no result")
    else:
        figure, axes, _ = atlas
        expected_trail = kp_kms * np.sin(2.0 * np.pi * atlas_bundle["phase"])
        x_lo, x_hi = axes.ravel()[0].get_xlim()
        if x_lo > float(np.nanmin(expected_trail)) or x_hi < float(np.nanmax(expected_trail)):
            failures.append("observed-frame atlas clipped the expected planet trail")
        if args.save_atlas is not None:
            output_path = save_figure_pdf(figure, args.save_atlas, dpi=150)
            print(f"saved atlas: {output_path}")
        plt.close(figure)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS: direct relativistic diagnostic sampling preserves injected offsets.")


if __name__ == "__main__":
    main()
