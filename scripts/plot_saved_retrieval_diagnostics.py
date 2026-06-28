#!/usr/bin/env python3
"""Generate post-hoc plots from saved retrieval outputs.

This script does not rerun SVI or MCMC.  Quick plots only read saved arrays.
The optional ``--model-plots`` path reloads the prepared data/opacities and
reconstructs the median processed model spectrum from the saved posterior.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CORNER_VARS = (
    "Kp",
    "spectroscopy/dRV",
    "logVMR_Fe I",
    "Tirr",
    "log_metallicity",
    "C_O_ratio",
    "Rp",
    "Rstar",
    "gamma",
    "kappa_ir_cgs",
)
DEFAULT_TRACE_VARS = (
    "Kp",
    "spectroscopy/dRV",
    "logVMR_Fe I",
    "Tirr",
    "log_metallicity",
    "C_O_ratio",
)


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser()
        candidates: list[Path]
        if (path / "run_config.log").exists():
            candidates = [path]
        else:
            candidates = [p.parent for p in sorted(path.glob("**/run_config.log"))]

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            if (candidate / "posterior_sample.npz").exists() or (
                candidate / "posterior_sample_by_chain.npz"
            ).exists():
                runs.append(candidate)
                seen.add(candidate)
    return runs


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def parse_run_config(run_dir: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    path = run_dir / "run_config.log"
    if not path.exists():
        return config
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def run_label(run_dir: Path) -> str:
    parsed = parse_run_config(run_dir)
    return parsed.get("Diagnostic label") or run_dir.name


def sigma_scale_from_config(run_dir: Path) -> float:
    parsed = parse_run_config(run_dir)
    value = parsed.get("Spectroscopic sigma scale", "1.0")
    try:
        scale = float(value.replace(",", ""))
    except ValueError:
        return 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def output_dir_for(run_dir: Path, subdir: str) -> Path:
    out = run_dir / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def flatten_sample(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def finite_dynamic(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return False
    return bool(np.nanmax(arr) > np.nanmin(arr))


def plot_svi_loss(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "svi_losses.npy"
    if not path.exists():
        return
    losses = np.asarray(np.load(path), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(losses.size), losses, lw=1.2)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_label(run_dir)}: SVI loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "svi_loss.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def plot_corner_hmc(run_dir: Path, out_dir: Path, variables: tuple[str, ...]) -> None:
    posterior_path = run_dir / "posterior_sample.npz"
    if not posterior_path.exists():
        return
    try:
        import corner
    except ImportError:
        print("  corner is not installed; skipping corner_plot_hmc.png", file=sys.stderr)
        return

    posterior = load_npz(posterior_path)
    columns: list[np.ndarray] = []
    labels: list[str] = []
    for name in variables:
        if name not in posterior:
            continue
        arr = flatten_sample(posterior[name]).astype(float)
        if not finite_dynamic(arr):
            continue
        columns.append(arr)
        labels.append(name.split("/")[-1])

    if len(columns) < 2:
        print("  fewer than two dynamic posterior columns; skipping corner plot")
        return

    n = min(col.size for col in columns)
    data = np.column_stack([col[:n] for col in columns])
    fig = corner.corner(data, labels=labels, bins=40, smooth=1.0, show_titles=True)
    save_path = out_dir / "corner_plot_hmc.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def plot_chain_traces(run_dir: Path, out_dir: Path, variables: tuple[str, ...]) -> None:
    path = run_dir / "posterior_sample_by_chain.npz"
    if not path.exists():
        return
    samples = load_npz(path)
    present = [name for name in variables if name in samples and np.asarray(samples[name]).ndim == 2]
    if not present:
        return

    ncols = 2
    nrows = int(math.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, present):
        arr = np.asarray(samples[name], dtype=float)
        for chain in range(arr.shape[0]):
            ax.plot(arr[chain], lw=0.9, alpha=0.85, label=f"chain {chain}")
        ax.set_title(name)
        ax.set_xlabel("draw")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(present):]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    fig.suptitle(f"{run_label(run_dir)}: chain traces", y=1.01)
    fig.tight_layout()
    save_path = out_dir / "chain_traces_key_params.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def plot_kp_drv_posterior(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "posterior_sample_by_chain.npz"
    if not path.exists():
        return
    samples = load_npz(path)
    drv_key = "spectroscopy/dRV" if "spectroscopy/dRV" in samples else "dRV"
    if "Kp" not in samples or drv_key not in samples:
        return

    kp = np.asarray(samples["Kp"], dtype=float)
    drv = np.asarray(samples[drv_key], dtype=float)
    if kp.ndim != 2 or drv.ndim != 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for chain in range(kp.shape[0]):
        ax.scatter(drv[chain], kp[chain], s=12, alpha=0.65, label=f"chain {chain}")
    ax.set_xlabel("dRV [km/s]")
    ax.set_ylabel("Kp [km/s]")
    ax.set_title(f"{run_label(run_dir)}: Kp-dRV posterior")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = out_dir / "kp_drv_posterior.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def plot_saved_temperature_profile(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "atmospheric_state.npz"
    if not path.exists():
        return
    state = load_npz(path)
    if "Tarr" not in state or "pressure" not in state:
        return

    temperature = np.asarray(state["Tarr"], dtype=float)
    pressure = np.asarray(state["pressure"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(temperature, pressure, marker="o", lw=1.5)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [bar]")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_title(f"{run_label(run_dir)}: saved median T-P")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = out_dir / "temperature_profile_saved_state.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def matched_filter_scale(data: np.ndarray, model: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_clip = np.clip(np.asarray(sigma, dtype=float), 1.0e-30, None)
    weights = 1.0 / np.square(sigma_clip)
    numerator = np.sum(weights * data * model, axis=1)
    denominator = np.sum(weights * np.square(model), axis=1) + 1.0e-30
    return numerator / denominator


def plot_processed_model_residual_panel(
    *,
    wavelength: np.ndarray,
    phase: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    model: np.ndarray,
    save_path: Path,
    title: str,
    wavelength_stride: int,
) -> None:
    wavelength = np.asarray(wavelength, dtype=float)
    phase = np.asarray(phase, dtype=float)
    observed = np.asarray(observed, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    alpha = matched_filter_scale(observed, model, sigma)
    residual = observed - alpha[:, None] * model

    stride = max(1, int(wavelength_stride))
    wav = wavelength[::stride]
    extent = [float(wav[0]), float(wav[-1]), float(np.min(phase)), float(np.max(phase))]

    panels = [
        ("Observed processed residuals", observed),
        ("Processed model", model),
        ("Observed - alpha*model", residual),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, (panel_title, arr) in zip(axes, panels):
        arr_plot = np.asarray(arr[:, ::stride], dtype=float)
        limit = float(np.nanpercentile(np.abs(arr_plot), 98.0))
        if not math.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        im = ax.imshow(
            arr_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Wavelength [A]")
        fig.colorbar(im, ax=ax, shrink=0.82)
    axes[0].set_ylabel("Orbital phase")
    fig.suptitle(title)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def component_filename(base: str, component_name: str, n_components: int) -> str:
    if n_components <= 1:
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    return f"{stem}_{component_name}{suffix}"


def write_model_plots(
    run_dir: Path,
    out_dir: Path,
    *,
    components: tuple[str, ...],
    apply_sysrem: bool | None,
    wavelength_stride: int,
    save_model_arrays: bool,
) -> None:
    from pipeline import retrieval as _retrieval
    from pipeline.diagnostics import (
        build_diag_config_from_run_dir,
        build_diagnostic_context,
        synthesize_processed_model_timeseries,
    )
    from plotting.plot import plot_transmission_spectrum

    config = build_diag_config_from_run_dir(run_dir)
    context = build_diagnostic_context(**config, apply_sysrem=apply_sysrem)
    if context.mode != "transmission":
        print(f"  model plots currently target transmission runs; got mode={context.mode!r}")
        return

    posterior = load_npz(run_dir / "posterior_sample.npz")
    selected_components = components or context.spectroscopic_component_names
    sigma_scale = sigma_scale_from_config(run_dir)

    for component_name in selected_components:
        component = context.spectroscopic_components[component_name]
        print(f"  rebuilding median processed model for component {component_name!r}")
        model_ts, _ = synthesize_processed_model_timeseries(
            context,
            posterior,
            component_name=component_name,
        )
        model_ts = np.asarray(model_ts, dtype=float)

        sigma = np.asarray(component.sigma, dtype=float) * sigma_scale
        obs_mean, obs_err = _retrieval._summarize_observed_spectrum(component.data, sigma)

        pre_mean = None
        pre_err = None
        if component.pre_sysrem_data is not None and component.pre_sysrem_sigma is not None:
            pre_sigma = np.asarray(component.pre_sysrem_sigma, dtype=float) * sigma_scale
            pre_mean, pre_err = _retrieval._summarize_observed_spectrum(
                component.pre_sysrem_data,
                pre_sigma,
            )

        spectrum_path = out_dir / component_filename(
            "transmission_spectrum.png",
            component_name,
            len(selected_components),
        )
        plot_transmission_spectrum(
            wavelength_nm=np.asarray(component.wav_obs, dtype=float) / 10.0,
            rp_obs=obs_mean,
            rp_err=obs_err,
            rp_hmc=model_ts,
            rp_svi=None,
            rp_pre_sysrem=pre_mean,
            rp_pre_sysrem_err=pre_err,
            save_path=str(spectrum_path),
        )

        panel_path = out_dir / component_filename(
            "processed_data_model_residuals.png",
            component_name,
            len(selected_components),
        )
        plot_processed_model_residual_panel(
            wavelength=np.asarray(component.wav_obs, dtype=float),
            phase=np.asarray(component.phase, dtype=float),
            observed=np.asarray(component.data, dtype=float),
            sigma=sigma,
            model=model_ts,
            save_path=panel_path,
            title=f"{run_label(run_dir)}: processed data/model/residuals",
            wavelength_stride=wavelength_stride,
        )

        if save_model_arrays:
            arrays_path = out_dir / component_filename(
                "posthoc_model_timeseries.npz",
                component_name,
                len(selected_components),
            )
            np.savez_compressed(
                arrays_path,
                wavelength=np.asarray(component.wav_obs),
                phase=np.asarray(component.phase),
                observed=np.asarray(component.data),
                sigma=sigma,
                model=model_ts,
            )
            print(f"  wrote {arrays_path}")


def write_quick_plots(
    run_dir: Path,
    out_dir: Path,
    *,
    corner_variables: tuple[str, ...],
    trace_variables: tuple[str, ...],
) -> None:
    plot_svi_loss(run_dir, out_dir)
    plot_corner_hmc(run_dir, out_dir, corner_variables)
    plot_chain_traces(run_dir, out_dir, trace_variables)
    plot_kp_drv_posterior(run_dir, out_dir)
    plot_saved_temperature_profile(run_dir, out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate post-hoc plots from saved retrieval output directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories or roots containing saved run_config.log files.",
    )
    parser.add_argument(
        "--output-subdir",
        default="posthoc_plots",
        help="Subdirectory created under each run directory for generated plots.",
    )
    parser.add_argument(
        "--model-plots",
        action="store_true",
        help="Also reconstruct median processed model spectra and residual panels.",
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Spectroscopic component to model-plot. Repeatable. Defaults to all components.",
    )
    parser.add_argument(
        "--apply-sysrem",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override SYSREM use for reconstructed model plots. Default uses config default.",
    )
    parser.add_argument(
        "--wavelength-stride",
        type=int,
        default=8,
        help="Stride for 2D processed data/model panels.",
    )
    parser.add_argument(
        "--no-save-model-arrays",
        action="store_true",
        help="Do not save reconstructed model timeseries arrays.",
    )
    parser.add_argument(
        "--corner-vars",
        nargs="*",
        default=list(DEFAULT_CORNER_VARS),
        help="Posterior variables to include in HMC corner plots.",
    )
    parser.add_argument(
        "--trace-vars",
        nargs="*",
        default=list(DEFAULT_TRACE_VARS),
        help="Chain variables to include in trace plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = discover_run_dirs(args.paths)
    if not run_dirs:
        print("No saved retrieval run directories found.", file=sys.stderr)
        return 1

    for run_dir in run_dirs:
        label = run_label(run_dir)
        print(f"\n=== {label} ===")
        print(f"run_dir: {run_dir}")
        out_dir = output_dir_for(run_dir, args.output_subdir)
        write_quick_plots(
            run_dir,
            out_dir,
            corner_variables=tuple(args.corner_vars),
            trace_variables=tuple(args.trace_vars),
        )
        if args.model_plots:
            try:
                write_model_plots(
                    run_dir,
                    out_dir,
                    components=tuple(args.component),
                    apply_sysrem=args.apply_sysrem,
                    wavelength_stride=args.wavelength_stride,
                    save_model_arrays=not args.no_save_model_arrays,
                )
            except Exception as exc:
                print(f"  model plot reconstruction failed for {run_dir}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
