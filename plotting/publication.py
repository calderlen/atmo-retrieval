"""Versioned publication-figure bundles for atmospheric retrieval runs.

This module is deliberately independent of JAX and the retrieval runtime.  It
owns figure layout, plotted-data sidecars, and the completeness manifest.  The
retrieval pipeline remains responsible for supplying model arrays that have
already been evaluated in the correct likelihood space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import (
    COMPUTER_MODERN_BRIGHT_FONTS,
    COMPUTER_MODERN_BRIGHT_RCPARAMS,
    configure_matplotlib,
    save_figure_pdf,
)
from plotting.wavelength import (
    fill_between_wavelength_segments,
    pcolormesh_wavelength_segments,
    plot_wavelength_segments,
    wavelength_segment_slices,
)


configure_matplotlib()


PUBLICATION_FIGURE_SCHEMA_VERSION = 1
PUBLICATION_MODEL_DRAW_COUNT = 32
PUBLICATION_TEMPERATURE_DRAW_COUNT = 256


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _git_snapshot(repository_root: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return snapshot
    snapshot["commit"] = commit or None
    snapshot["dirty"] = bool(status.strip())
    return snapshot


def _safe_identifier(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in str(value))
    return cleaned.strip("_").lower() or "figure"


@dataclass
class PublicationBundle:
    """Write a structured publication bundle and its completeness manifest."""

    run_dir: Path
    metadata: dict[str, Any]
    repository_root: Path | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.repository_root = (
            Path(self.repository_root)
            if self.repository_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.figure_root = self.run_dir / "figures"
        self.data_root = self.run_dir / "figure_data"
        for tier in ("paper", "supplement", "qc"):
            (self.figure_root / tier).mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def figure_path(self, figure_id: str, tier: str) -> Path:
        if tier not in {"paper", "supplement", "qc"}:
            raise ValueError(f"Unsupported publication tier: {tier!r}")
        return self.figure_root / tier / f"{_safe_identifier(figure_id)}.pdf"

    def data_path(self, figure_id: str) -> Path:
        return self.data_root / f"{_safe_identifier(figure_id)}.npz"

    def save_figure(
        self,
        figure: plt.Figure,
        *,
        figure_id: str,
        tier: str,
        required: bool,
        plotted_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        warnings: Iterable[str] = (),
        bbox_inches: str | None = "tight",
    ) -> Path:
        output_path = self.figure_path(figure_id, tier)
        kwargs: dict[str, Any] = {"dpi": 200}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        output_path = save_figure_pdf(figure, output_path, **kwargs)
        plt.close(figure)

        sidecar_path: Path | None = None
        if plotted_data:
            sidecar_path = self.data_path(figure_id)
            arrays = {str(key): np.asarray(value) for key, value in plotted_data.items()}
            np.savez_compressed(sidecar_path, **arrays)

        self.entries.append(
            {
                "figure_id": figure_id,
                "tier": tier,
                "required": bool(required),
                "status": "ok",
                "path": str(output_path.relative_to(self.run_dir)),
                "data_path": (
                    None if sidecar_path is None else str(sidecar_path.relative_to(self.run_dir))
                ),
                "metadata": _json_safe(metadata or {}),
                "warnings": [str(item) for item in warnings],
            }
        )
        return output_path

    def register_existing(
        self,
        *,
        figure_id: str,
        tier: str,
        path: str | Path,
        required: bool,
        metadata: dict[str, Any] | None = None,
        warnings: Iterable[str] = (),
    ) -> None:
        path = Path(path)
        if not path.exists():
            self.record_failure(
                figure_id=figure_id,
                tier=tier,
                required=required,
                error=f"Expected figure was not written: {path}",
                metadata=metadata,
            )
            return
        self.entries.append(
            {
                "figure_id": figure_id,
                "tier": tier,
                "required": bool(required),
                "status": "ok",
                "path": str(path.relative_to(self.run_dir)),
                "data_path": None,
                "metadata": _json_safe(metadata or {}),
                "warnings": [str(item) for item in warnings],
            }
        )

    def record_failure(
        self,
        *,
        figure_id: str,
        tier: str,
        required: bool,
        error: Exception | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            {
                "figure_id": figure_id,
                "tier": tier,
                "required": bool(required),
                "status": "failed",
                "path": None,
                "data_path": None,
                "metadata": _json_safe(metadata or {}),
                "warnings": [],
                "error": str(error),
            }
        )

    def finalize(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        required_failures = [
            entry["figure_id"]
            for entry in self.entries
            if entry["required"] and entry["status"] != "ok"
        ]
        required_successes = [
            entry["figure_id"]
            for entry in self.entries
            if entry["required"] and entry["status"] == "ok"
        ]
        manifest = {
            "schema_version": PUBLICATION_FIGURE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "publication_bundle_complete": bool(required_successes) and not required_failures,
            "required_figure_count": len(required_successes) + len(required_failures),
            "required_failures": required_failures,
            "typography": {
                "font_family": list(COMPUTER_MODERN_BRIGHT_FONTS),
                "mathtext_fontset": COMPUTER_MODERN_BRIGHT_RCPARAMS["mathtext.fontset"],
                "mathtext_fallback": COMPUTER_MODERN_BRIGHT_RCPARAMS["mathtext.fallback"],
                "pdf_fonttype": COMPUTER_MODERN_BRIGHT_RCPARAMS["pdf.fonttype"],
            },
            "metadata": _json_safe(self.metadata),
            "git": _git_snapshot(self.repository_root),
            "figures": self.entries,
        }
        if extra:
            manifest.update(_json_safe(extra))
        manifest_path = self.run_dir / "figures_manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        return manifest


def deterministic_draw_indices(sample_count: int, maximum: int) -> np.ndarray:
    sample_count = int(sample_count)
    maximum = int(maximum)
    if sample_count <= 0 or maximum <= 0:
        return np.asarray([], dtype=int)
    count = min(sample_count, maximum)
    return np.unique(np.linspace(0, sample_count - 1, count).round().astype(int))


def _phase_edges(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    if phase.size == 1:
        return np.asarray([phase[0] - 0.5, phase[0] + 0.5])
    edges = np.empty(phase.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (phase[:-1] + phase[1:])
    edges[0] = phase[0] - 0.5 * (phase[1] - phase[0])
    edges[-1] = phase[-1] + 0.5 * (phase[-1] - phase[-2])
    if np.any(np.diff(edges) <= 0.0):
        return np.arange(phase.size + 1, dtype=float) - 0.5
    return edges


def _robust_symmetric_limit(values: np.ndarray, *, percentile: float = 99.2) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0
    return limit


def plot_likelihood_space_triptych(
    *,
    wavelength_A: np.ndarray,
    phase: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    title: str,
    max_wavelength_points: int = 2400,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """Plot processed data, model, and residuals without collapsing exposures."""
    wavelength = np.asarray(wavelength_A, dtype=float)
    phase = np.asarray(phase, dtype=float)
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if data.ndim != 2 or data.shape != model.shape or data.shape != sigma.shape:
        raise ValueError(
            "Likelihood triptych requires matching exposure x wavelength arrays; "
            f"got data={data.shape}, model={model.shape}, sigma={sigma.shape}."
        )
    if wavelength.shape != (data.shape[1],) or phase.shape != (data.shape[0],):
        raise ValueError("Wavelength/phase coordinates do not match the likelihood arrays.")

    order = np.argsort(phase)
    wavelength_order = np.argsort(wavelength)
    wavelength = wavelength[wavelength_order]
    data = data[:, wavelength_order]
    model = model[:, wavelength_order]
    sigma = sigma[:, wavelength_order]
    phase_sorted = phase[order]
    if wavelength.size > int(max_wavelength_points):
        index = deterministic_draw_indices(wavelength.size, int(max_wavelength_points))
    else:
        index = np.arange(wavelength.size, dtype=int)
    wave_plot = wavelength[index]
    data_plot = data[order][:, index]
    model_plot = model[order][:, index]
    sigma_plot = sigma[order][:, index]
    residual_plot = data_plot - model_plot

    shared_limit = _robust_symmetric_limit(np.concatenate((data_plot.ravel(), model_plot.ravel())))
    residual_limit = _robust_symmetric_limit(residual_plot)
    phase_edges = _phase_edges(phase_sorted)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.25),
        sharey=True,
        constrained_layout=True,
    )
    specifications = (
        (data_plot, "Processed data", shared_limit),
        (model_plot, "Posterior-median processed model", shared_limit),
        (residual_plot, "Data - model", residual_limit),
    )
    for panel, (array, panel_title, limit) in zip(axes, specifications):
        meshes = pcolormesh_wavelength_segments(
            panel,
            wave_plot,
            array,
            y_edges=phase_edges,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        for mesh in meshes:
            mesh.set_rasterized(True)
        panel.set_title(panel_title, fontsize=8.5)
        panel.set_xlabel(r"Vacuum wavelength [$\AA$]")
        fig.colorbar(meshes[0], ax=panel, pad=0.012, fraction=0.045)
    axes[0].set_ylabel("Orbital phase")
    fig.suptitle(title, fontsize=9.5)
    plotted = {
        "wavelength_A": wave_plot,
        "phase": phase_sorted,
        "data": data_plot,
        "model": model_plot,
        "residual": residual_plot,
        "sigma": sigma_plot,
    }
    return fig, plotted


def prepare_planet_frame_operator(
    *,
    wavelength_A: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    kp_kms: float,
    v_sys_kms: float,
    bin_size: int = 1,
) -> dict[str, Any]:
    """Build the same fixed, gap-safe planet-frame operator for data and draws."""
    from dataio.collapse_emission_timeseries_to_1d import build_emission_collapse_operator

    wavelength = np.asarray(wavelength_A, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    phase = np.asarray(phase, dtype=float)
    if sigma.ndim != 2 or sigma.shape != (phase.size, wavelength.size):
        raise ValueError("Planet-frame operator coordinates do not match sigma.")
    order = np.argsort(wavelength)
    operator = build_emission_collapse_operator(
        wavelength[order],
        sigma[:, order],
        phase,
        kp_kms=float(kp_kms),
        velocity_offset_kms=float(v_sys_kms),
        eccentricity=0.0,
        omega_deg=None,
        bin_size=int(bin_size),
    )
    return {
        "wavelength_order": order,
        "operator": operator,
        "orbital_velocity_model": "circular_sine_plus_v_sys",
        "kp_kms": float(kp_kms),
        "v_sys_kms": float(v_sys_kms),
    }


def apply_planet_frame_operator(
    values: np.ndarray,
    prepared_operator: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from dataio.collapse_emission_timeseries_to_1d import apply_emission_collapse_operator

    values = np.asarray(values, dtype=float)
    order = np.asarray(prepared_operator["wavelength_order"], dtype=int)
    return apply_emission_collapse_operator(
        values[:, order],
        prepared_operator["operator"],
    )


def _inverse_variance_bins(
    wavelength: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    *,
    max_bins: int = 450,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float)
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    output_wave: list[float] = []
    output_value: list[float] = []
    output_error: list[float] = []
    segments = wavelength_segment_slices(wavelength)
    total = sum(segment.stop - segment.start for segment in segments)
    for segment in segments:
        indices = np.arange(segment.start, segment.stop)
        segment_bins = max(1, round(max_bins * indices.size / max(total, 1)))
        for group in np.array_split(indices, min(segment_bins, indices.size)):
            valid = (
                np.isfinite(wavelength[group])
                & np.isfinite(values[group])
                & np.isfinite(errors[group])
                & (errors[group] > 0.0)
            )
            if not np.any(valid):
                continue
            group = group[valid]
            weight = 1.0 / np.square(errors[group])
            output_wave.append(float(np.average(wavelength[group], weights=weight)))
            output_value.append(float(np.average(values[group], weights=weight)))
            output_error.append(float(np.sqrt(1.0 / np.sum(weight))))
    return np.asarray(output_wave), np.asarray(output_value), np.asarray(output_error)


def plot_planet_frame_posterior_predictive(
    *,
    wavelength_A: np.ndarray,
    observed: np.ndarray,
    observed_error: np.ndarray,
    model_draws: np.ndarray,
    title: str,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """Plot a fixed-frame posterior predictive spectrum from genuine draws."""
    wavelength = np.asarray(wavelength_A, dtype=float)
    observed = np.asarray(observed, dtype=float)
    observed_error = np.asarray(observed_error, dtype=float)
    draws = np.asarray(model_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != wavelength.size:
        raise ValueError(
            "model_draws must be posterior_draw x wavelength; got "
            f"{draws.shape} for {wavelength.size} wavelengths."
        )
    if draws.shape[0] < 4:
        raise ValueError("At least four successful posterior model draws are required.")
    if observed.shape != wavelength.shape or observed_error.shape != wavelength.shape:
        raise ValueError("Observed planet-frame arrays do not match wavelength.")

    q025, q16, q50, q84, q975 = np.nanpercentile(
        draws,
        [2.5, 16.0, 50.0, 84.0, 97.5],
        axis=0,
    )
    residual = observed - q50
    bin_wave, bin_observed, bin_error = _inverse_variance_bins(
        wavelength,
        observed,
        observed_error,
    )
    median_at_bins = np.interp(bin_wave, wavelength, q50)
    bin_residual = bin_observed - median_at_bins

    fig, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        constrained_layout=True,
    )
    axis.plot(wavelength, observed, ".", ms=0.45, color="0.15", alpha=0.12, rasterized=True)
    fill_between_wavelength_segments(
        axis,
        wavelength,
        q025,
        q975,
        color="tab:blue",
        alpha=0.12,
        label="95% posterior predictive interval",
    )
    fill_between_wavelength_segments(
        axis,
        wavelength,
        q16,
        q84,
        color="tab:blue",
        alpha=0.25,
        label="68% posterior predictive interval",
    )
    plot_wavelength_segments(
        axis,
        wavelength,
        q50,
        color="tab:blue",
        lw=0.9,
        label="Posterior predictive median",
    )
    axis.errorbar(
        bin_wave,
        bin_observed,
        yerr=bin_error,
        fmt="o",
        ms=1.7,
        lw=0.45,
        color="black",
        ecolor="0.3",
        label="Inverse-variance bins",
        zorder=5,
    )
    axis.set_ylabel("Processed residual flux")
    axis.legend(loc="best", fontsize=6.7, ncol=2)
    axis.set_title(title, fontsize=9.5)

    residual_axis.errorbar(
        bin_wave,
        bin_residual,
        yerr=bin_error,
        fmt="o",
        ms=1.7,
        lw=0.45,
        color="black",
        ecolor="0.3",
    )
    residual_axis.axhline(0.0, color="0.35", lw=0.7, ls="--")
    residual_axis.set_xlabel(r"Planet-frame vacuum wavelength [$\AA$]")
    residual_axis.set_ylabel("Data - model")
    plotted = {
        "wavelength_A": wavelength,
        "observed": observed,
        "observed_error": observed_error,
        "model_draws": draws,
        "model_q025": q025,
        "model_q16": q16,
        "model_q50": q50,
        "model_q84": q84,
        "model_q975": q975,
        "residual": residual,
        "binned_wavelength_A": bin_wave,
        "binned_observed": bin_observed,
        "binned_error": bin_error,
        "binned_residual": bin_residual,
    }
    return fig, plotted


def plot_temperature_pressure_posterior(
    *,
    pressure_bar: np.ndarray,
    temperature_draws_K: np.ndarray,
    profile_label: str,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    pressure = np.asarray(pressure_bar, dtype=float)
    draws = np.asarray(temperature_draws_K, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != pressure.size:
        raise ValueError("Temperature draws must have shape draw x pressure.")
    if draws.shape[0] < 4:
        raise ValueError("At least four temperature-profile draws are required.")
    q025, q16, q50, q84, q975 = np.nanpercentile(
        draws,
        [2.5, 16.0, 50.0, 84.0, 97.5],
        axis=0,
    )
    fig, axis = plt.subplots(figsize=(3.35, 4.5), constrained_layout=True)
    axis.fill_betweenx(pressure, q025, q975, color="tab:purple", alpha=0.12, label="95% interval")
    axis.fill_betweenx(pressure, q16, q84, color="tab:purple", alpha=0.28, label="68% interval")
    axis.plot(q50, pressure, color="tab:purple", lw=1.25, label="Posterior median")
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.set_xlabel("Temperature [K]")
    axis.set_ylabel("Pressure [bar]")
    axis.set_title(f"Temperature-pressure profile ({profile_label})", fontsize=9)
    axis.legend(loc="best", fontsize=7)
    plotted = {
        "pressure_bar": pressure,
        "temperature_draws_K": draws,
        "temperature_q025_K": q025,
        "temperature_q16_K": q16,
        "temperature_q50_K": q50,
        "temperature_q84_K": q84,
        "temperature_q975_K": q975,
    }
    return fig, plotted


def plot_bandpass_posterior_predictive(
    *,
    observed: float,
    observed_error: float,
    model_draws: np.ndarray,
    component_name: str,
    observable: str,
) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """Plot one scalar bandpass constraint against its predictive posterior."""
    draws = np.asarray(model_draws, dtype=float).reshape(-1)
    draws = draws[np.isfinite(draws)]
    if draws.size < 4:
        raise ValueError("At least four finite bandpass model draws are required.")
    observed = float(observed)
    observed_error = float(observed_error)
    if not np.isfinite(observed) or not np.isfinite(observed_error) or observed_error <= 0.0:
        raise ValueError("Bandpass observation and uncertainty must be finite and positive.")

    q025, q16, q50, q84, q975 = np.percentile(
        draws,
        [2.5, 16.0, 50.0, 84.0, 97.5],
    )
    lower = min(q025, observed - 3.5 * observed_error)
    upper = max(q975, observed + 3.5 * observed_error)
    padding = max(0.08 * (upper - lower), np.finfo(float).eps)

    fig, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.hist(
        draws,
        bins=min(40, max(12, int(np.sqrt(draws.size) * 2))),
        density=True,
        histtype="stepfilled",
        color="tab:blue",
        alpha=0.22,
        edgecolor="tab:blue",
        linewidth=0.85,
        label="Posterior predictive",
    )
    axis.axvspan(q025, q975, color="tab:blue", alpha=0.08, label="95% interval")
    axis.axvspan(q16, q84, color="tab:blue", alpha=0.16, label="68% interval")
    axis.axvline(q50, color="tab:blue", lw=1.15, label="Posterior median")
    axis.axvspan(
        observed - observed_error,
        observed + observed_error,
        color="0.25",
        alpha=0.12,
        label=r"Observed $\pm 1\sigma$",
    )
    axis.axvline(observed, color="0.15", lw=1.0, ls="--")
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_xlabel("Observable value")
    axis.set_ylabel("Posterior density")
    observable_label = observable.replace("_", " ")
    axis.set_title(f"{component_name}: {observable_label}", fontsize=9)
    axis.legend(loc="best", fontsize=6.4)
    plotted = {
        "observed": np.asarray(observed),
        "observed_error": np.asarray(observed_error),
        "model_draws": draws,
        "model_quantiles_2p5_16_50_84_97p5": np.asarray(
            [q025, q16, q50, q84, q975]
        ),
        "component_name": np.asarray(component_name, dtype="U"),
        "observable": np.asarray(observable, dtype="U"),
    }
    return fig, plotted


def _posterior_key(samples: dict[str, np.ndarray], basename: str) -> str | None:
    exact = [key for key in samples if key == basename]
    if exact:
        return exact[0]
    scoped = [key for key in samples if key.rsplit("/", 1)[-1] == basename]
    return sorted(scoped)[0] if scoped else None


def plot_kp_vsys_posterior(
    samples: dict[str, np.ndarray],
) -> tuple[plt.Figure, dict[str, np.ndarray]] | None:
    kp_key = _posterior_key(samples, "Kp")
    vsys_key = _posterior_key(samples, "v_sys")
    if kp_key is None or vsys_key is None:
        return None
    kp = np.asarray(samples[kp_key], dtype=float).reshape(-1)
    vsys = np.asarray(samples[vsys_key], dtype=float).reshape(-1)
    valid = np.isfinite(kp) & np.isfinite(vsys)
    kp = kp[valid]
    vsys = vsys[valid]
    if kp.size < 20:
        return None

    density, x_edges, y_edges = np.histogram2d(vsys, kp, bins=48)
    positive = np.sort(density[density > 0.0].ravel())[::-1]
    levels: list[float] = []
    if positive.size:
        cumulative = np.cumsum(positive) / np.sum(positive)
        for probability in (0.95, 0.68):
            levels.append(float(positive[min(np.searchsorted(cumulative, probability), positive.size - 1)]))
    levels = sorted(set(levels))
    x_center = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_center = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, axis = plt.subplots(figsize=(3.6, 3.35), constrained_layout=True)
    image = axis.pcolormesh(x_edges, y_edges, density.T, cmap="Blues", shading="auto", rasterized=True)
    if levels:
        axis.contour(x_center, y_center, density.T, levels=levels, colors="0.15", linewidths=0.8)
    median_vsys = float(np.median(vsys))
    median_kp = float(np.median(kp))
    axis.plot(median_vsys, median_kp, marker="+", color="tab:red", ms=8, mew=1.2)
    axis.set_xlabel(r"$v_{\rm sys}$ [km s$^{-1}$]")
    axis.set_ylabel(r"$K_p$ [km s$^{-1}$]")
    axis.set_title(r"Orbital-velocity posterior", fontsize=9)
    fig.colorbar(image, ax=axis, label="Posterior samples per bin")
    return fig, {
        "kp_kms": kp,
        "v_sys_kms": vsys,
        "density": density,
        "v_sys_edges_kms": x_edges,
        "kp_edges_kms": y_edges,
    }


def plot_abundance_constraints(
    samples: dict[str, np.ndarray],
) -> tuple[plt.Figure, dict[str, np.ndarray]] | None:
    rows: list[tuple[str, np.ndarray]] = []
    for key in sorted(samples):
        basename = key.rsplit("/", 1)[-1]
        if not (
            basename.startswith("logVMR_")
            or basename in {"log_metallicity", "C_O_ratio"}
        ):
            continue
        array = np.asarray(samples[key], dtype=float)
        if array.ndim == 1:
            rows.append((key, array))
        elif array.ndim > 1 and int(np.prod(array.shape[1:])) <= 6:
            flat = array.reshape(array.shape[0], -1)
            rows.extend((f"{key}[{index}]", flat[:, index]) for index in range(flat.shape[1]))
    if not rows:
        return None

    labels: list[str] = []
    quantiles: list[np.ndarray] = []
    raw: dict[str, np.ndarray] = {}
    for label, values in rows:
        values = values[np.isfinite(values)]
        if values.size < 4:
            continue
        labels.append(label)
        quantiles.append(np.percentile(values, [2.5, 16.0, 50.0, 84.0, 97.5]))
        raw[label] = values
    if not quantiles:
        return None
    quantile_array = np.asarray(quantiles)
    y = np.arange(len(labels))
    fig_height = max(2.6, 0.34 * len(labels) + 1.25)
    fig, axis = plt.subplots(figsize=(5.0, fig_height), constrained_layout=True)
    axis.hlines(y, quantile_array[:, 0], quantile_array[:, 4], color="tab:blue", lw=1.0, alpha=0.45)
    axis.hlines(y, quantile_array[:, 1], quantile_array[:, 3], color="tab:blue", lw=3.0)
    axis.plot(quantile_array[:, 2], y, "o", color="black", ms=3.5)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("Posterior parameter value")
    axis.set_title("Atmospheric abundance constraints", fontsize=9)
    plotted: dict[str, np.ndarray] = {
        "labels": np.asarray(labels, dtype="U"),
        "quantiles_2p5_16_50_84_97p5": quantile_array,
    }
    for index, label in enumerate(labels):
        plotted[f"samples_{index}"] = raw[label]
    return fig, plotted


def plot_mcmc_chain_traces(
    posterior_by_chain: dict[str, np.ndarray],
    *,
    max_parameters: int = 8,
) -> tuple[plt.Figure, dict[str, np.ndarray]] | None:
    candidates: list[tuple[str, np.ndarray]] = []
    priority = ("Kp", "v_sys", "Tirr", "log_metallicity", "C_O_ratio", "Rp", "Mp")
    for basename in priority:
        key = _posterior_key(posterior_by_chain, basename)
        if key is None:
            continue
        array = np.asarray(posterior_by_chain[key], dtype=float)
        if array.ndim == 2:
            candidates.append((key, array))
    for key in sorted(posterior_by_chain):
        if any(existing == key for existing, _ in candidates):
            continue
        array = np.asarray(posterior_by_chain[key], dtype=float)
        if array.ndim == 2:
            candidates.append((key, array))
        if len(candidates) >= int(max_parameters):
            break
    candidates = candidates[: int(max_parameters)]
    if not candidates:
        return None

    fig, axes = plt.subplots(
        len(candidates),
        2,
        figsize=(7.2, max(2.4, 1.35 * len(candidates))),
        squeeze=False,
        constrained_layout=True,
    )
    plotted: dict[str, np.ndarray] = {}
    colors = plt.get_cmap("tab10")
    for row, (key, values) in enumerate(candidates):
        for chain in range(values.shape[0]):
            axes[row, 0].plot(values[chain], lw=0.55, alpha=0.8, color=colors(chain % 10))
            axes[row, 1].hist(
                values[chain],
                bins=36,
                density=True,
                histtype="step",
                lw=0.8,
                color=colors(chain % 10),
            )
        axes[row, 0].set_ylabel(key.rsplit("/", 1)[-1], fontsize=7)
        axes[row, 0].tick_params(labelsize=6.5)
        axes[row, 1].tick_params(labelsize=6.5)
        plotted[f"parameter_{row}"] = values
    axes[-1, 0].set_xlabel("Post-warmup draw")
    axes[-1, 1].set_xlabel("Marginal density")
    axes[0, 0].set_title("Chain traces", fontsize=9)
    axes[0, 1].set_title("Per-chain marginals", fontsize=9)
    plotted["parameter_names"] = np.asarray([key for key, _ in candidates], dtype="U")
    return fig, plotted
