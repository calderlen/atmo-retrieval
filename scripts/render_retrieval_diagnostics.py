#!/usr/bin/env python3
"""Render saved-run retrieval diagnostics without a notebook."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", help="Required only when the saved run did not log an epoch.")
    parser.add_argument("--component", help="Spectroscopic component name; defaults to the primary component.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature-draws", type=int, default=100)
    parser.add_argument("--trace-parameter", action="append")
    parser.add_argument("--max-trace-parameters", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.temperature_draws < 1 or args.max_trace_parameters < 1:
        raise SystemExit("--temperature-draws and --max-trace-parameters must be positive.")
    import matplotlib.pyplot as plt
    import numpy as np

    from pipeline.diagnostics import (
        default_corner_variables,
        load_saved_run_diagnostics,
        planet_rest_frame_products,
        plot_contribution_functions,
        plot_planet_rest_frame_coadd,
        plot_posterior_traces,
        plot_temperature_profiles,
        posterior_summary_rows,
    )
    from plotting.plot import plot_transmission_spectrum, save_retrieval_corner_plots
    from plotting.style import save_figure_pdf

    products = load_saved_run_diagnostics(
        args.run_dir,
        epoch=args.epoch,
        component_name=args.component,
    )
    context = products["context"]
    component = products["component"]
    state = products["atmospheric_state"]
    output_dir = args.output_dir or args.run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _slug(products["component_name"])

    posterior_rows = posterior_summary_rows(products["posterior"])
    if not posterior_rows:
        raise ValueError("The saved posterior contains no finite numeric samples.")
    posterior_table = output_dir / "posterior_summary.csv"
    with posterior_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(posterior_rows[0]))
        writer.writeheader()
        writer.writerows(posterior_rows)
    corner_variables = default_corner_variables(products["posterior"])
    save_retrieval_corner_plots(
        str(output_dir),
        hmc_samples=products["posterior"],
        variables=corner_variables,
    )
    trace_figure = plot_posterior_traces(
        products["posterior"],
        variables=args.trace_parameter,
        maximum=args.max_trace_parameters,
    )
    if trace_figure is not None:
        save_figure_pdf(trace_figure, output_dir / "posterior_traces.pdf")
        plt.close(trace_figure)

    temperature_figure = plot_temperature_profiles(
        products["posterior"],
        context.shared_region_config,
        sample_prefix=context.shared_region_sample_prefix,
        draw_count=args.temperature_draws,
    )
    save_figure_pdf(temperature_figure, output_dir / f"{stem}_temperature_profile.pdf")
    plt.close(temperature_figure)

    contribution_figure = plot_contribution_functions(
        component.nu_grid,
        state["pressure"],
        state["dtau"],
        dtau_per_species=state.get("dtau_per_species"),
    )
    save_figure_pdf(contribution_figure, output_dir / f"{stem}_contribution_functions.pdf")
    plt.close(contribution_figure)

    rest_frame = planet_rest_frame_products(
        component.wav_obs,
        component.data,
        component.sigma,
        products["model_timeseries"],
        component.phase,
        products["posterior"],
    )
    rest_frame_figure = plot_planet_rest_frame_coadd(rest_frame)
    save_figure_pdf(rest_frame_figure, output_dir / f"{stem}_planet_rest_frame_coadd.pdf")
    plt.close(rest_frame_figure)

    if context.mode == "transmission":
        plot_transmission_spectrum(
            wavelength_nm=np.asarray(component.wav_obs) / 10.0,
            rp_obs=products["observed_mean"],
            rp_err=products["observed_error"],
            rp_hmc=products["model_timeseries"],
            rp_svi=np.mean(products["model_timeseries"], axis=0),
            save_path=str(output_dir / f"{stem}_transmission_spectrum.pdf"),
        )

    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "component": products["component_name"],
        "mode": context.mode,
        "Kp_kms": rest_frame["Kp"],
        "v_sys_kms": rest_frame["v_sys"],
        "posterior_parameters": [row["parameter"] for row in posterior_rows],
        "corner_variables": corner_variables,
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
