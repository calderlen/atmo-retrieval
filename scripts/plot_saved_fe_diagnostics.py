#!/usr/bin/env python3
"""Plot Fe-template diagnostics from a saved publication spectrum sidecar."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import configure_matplotlib, save_figure_pdf
from plotting.wavelength import wavelength_segment_slices


SPEED_OF_LIGHT_KMS = 299_792.458


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--component", default="spectroscopy")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-windows", type=int, default=6)
    parser.add_argument("--window-width-A", type=float, default=3.0)
    parser.add_argument("--window-separation-A", type=float, default=1.0)
    parser.add_argument("--velocity-min-kms", type=float, default=-50.0)
    parser.add_argument("--velocity-max-kms", type=float, default=50.0)
    parser.add_argument("--velocity-step-kms", type=float, default=0.25)
    return parser


def _load_sidecar(path: Path) -> dict[str, np.ndarray]:
    required = {
        "wavelength_A",
        "observed",
        "observed_error",
        "model_draws",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Saved spectrum sidecar not found: {path}")
    with np.load(path) as saved:
        missing = sorted(required - set(saved.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        products = {name: np.asarray(saved[name]) for name in saved.files}

    wavelength = np.asarray(products["wavelength_A"], dtype=float)
    observed = np.asarray(products["observed"], dtype=float)
    error = np.asarray(products["observed_error"], dtype=float)
    draws = np.asarray(products["model_draws"], dtype=float)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength_A must be a one-dimensional array with at least two pixels.")
    if observed.shape != wavelength.shape or error.shape != wavelength.shape:
        raise ValueError("Observed spectrum and uncertainty must match wavelength_A.")
    if draws.ndim != 2 or draws.shape[1] != wavelength.size or draws.shape[0] < 4:
        raise ValueError(
            "model_draws must have shape (at least 4 posterior draws, wavelength)."
        )
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("wavelength_A must be strictly increasing.")
    valid = np.isfinite(wavelength) & np.isfinite(observed) & np.isfinite(error) & (error > 0.0)
    valid &= np.all(np.isfinite(draws), axis=0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("No common finite observed/model pixels with positive uncertainty.")

    products["wavelength_A"] = wavelength
    products["observed"] = observed
    products["observed_error"] = error
    products["model_draws"] = draws
    products["common_valid"] = valid
    return products


def _run_title(run_dir: Path, component: str) -> str:
    values: dict[str, str] = {}
    config_path = run_dir / "run_config.log"
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in {"Planet", "Epoch", "Collapsed emission selection"}:
                values[key] = value
    fields = [values.get("Planet"), values.get("Epoch"), values.get("Collapsed emission selection")]
    fields = [None if field is None else field.replace("_", "-") for field in fields]
    prefix = " ".join(field for field in fields if field)
    return f"{prefix} - {component}" if prefix else f"{run_dir.name} - {component}"


def _bin_groups(wavelength: np.ndarray, *, max_bins: int) -> list[np.ndarray]:
    segments = wavelength_segment_slices(wavelength)
    total = sum(segment.stop - segment.start for segment in segments)
    groups: list[np.ndarray] = []
    for segment in segments:
        indices = np.arange(segment.start, segment.stop)
        count = max(1, round(max_bins * indices.size / max(total, 1)))
        groups.extend(np.array_split(indices, min(count, indices.size)))
    return [group for group in groups if group.size]


def _bin_products(
    wavelength: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    draws: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_wave: list[float] = []
    bin_observed: list[float] = []
    bin_error: list[float] = []
    bin_draws: list[np.ndarray] = []
    for group in groups:
        valid = (
            np.isfinite(wavelength[group])
            & np.isfinite(observed[group])
            & np.isfinite(error[group])
            & (error[group] > 0.0)
            & np.all(np.isfinite(draws[:, group]), axis=0)
        )
        if not np.any(valid):
            continue
        index = group[valid]
        weight = 1.0 / np.square(error[index])
        weight_sum = float(np.sum(weight))
        bin_wave.append(float(np.average(wavelength[index], weights=weight)))
        bin_observed.append(float(np.average(observed[index], weights=weight)))
        bin_error.append(float(np.sqrt(1.0 / weight_sum)))
        bin_draws.append(np.sum(draws[:, index] * weight[None, :], axis=1) / weight_sum)
    return (
        np.asarray(bin_wave),
        np.asarray(bin_observed),
        np.asarray(bin_error),
        np.asarray(bin_draws).T,
    )


def _rank_model_windows(
    wavelength: np.ndarray,
    model: np.ndarray,
    error: np.ndarray,
    *,
    count: int,
    width_A: float,
    separation_A: float,
) -> list[dict[str, float]]:
    power = np.where(
        np.isfinite(model) & np.isfinite(error) & (error > 0.0),
        np.square(model / error),
        0.0,
    )
    candidates: list[dict[str, float]] = []
    for segment in wavelength_segment_slices(wavelength):
        segment_step = float(np.median(np.diff(wavelength[segment])))
        stride = max(1, int(round(0.05 / segment_step)))
        cumulative = np.concatenate(([0.0], np.cumsum(power[segment])))
        for local_start in range(0, segment.stop - segment.start, stride):
            start = segment.start + local_start
            stop = int(
                np.searchsorted(
                    wavelength[segment],
                    wavelength[start] + width_A,
                    side="right",
                )
            )
            if stop <= local_start:
                continue
            local_stop = min(stop, segment.stop - segment.start)
            end = segment.start + local_stop
            if wavelength[end - 1] - wavelength[start] < 0.9 * width_A:
                continue
            candidates.append(
                {
                    "score": float(cumulative[local_stop] - cumulative[local_start]),
                    "lo_A": float(wavelength[start]),
                    "hi_A": float(wavelength[end - 1]),
                }
            )

    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        separated = all(
            candidate["hi_A"] < existing["lo_A"] - separation_A
            or candidate["lo_A"] > existing["hi_A"] + separation_A
            for existing in selected
        )
        if separated:
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} non-overlapping model-power windows were found.")
    return sorted(selected, key=lambda row: row["lo_A"])


def _posterior_quantiles(draws: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.nanpercentile(draws, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0))


def _plot_zoom_panels(
    *,
    wavelength: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    draws: np.ndarray,
    windows: list[dict[str, float]],
    title: str,
    output_path: Path,
) -> None:
    q025, q16, q50, q84, q975 = _posterior_quantiles(draws)
    columns = 2
    rows = int(np.ceil(len(windows) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(8.2, 2.35 * rows), squeeze=False)
    for index, (axis, window) in enumerate(zip(axes.flat, windows)):
        selected = (wavelength >= window["lo_A"]) & (wavelength <= window["hi_A"])
        indices = np.flatnonzero(selected)
        local_groups = [group for group in np.array_split(indices, min(18, indices.size)) if group.size]
        bin_wave, bin_observed, bin_error, _ = _bin_products(
            wavelength, observed, error, draws, local_groups
        )
        wave_nm = wavelength[selected] / 10.0
        axis.plot(
            wave_nm,
            observed[selected],
            ".",
            ms=1.5,
            color="0.35",
            alpha=0.25,
            rasterized=True,
            label="Native observed" if index == 0 else None,
        )
        axis.fill_between(
            wave_nm,
            q025[selected],
            q975[selected],
            color="tab:blue",
            alpha=0.12,
            lw=0.0,
            label="95% HMC interval" if index == 0 else None,
        )
        axis.fill_between(
            wave_nm,
            q16[selected],
            q84[selected],
            color="tab:blue",
            alpha=0.28,
            lw=0.0,
            label="68% HMC interval" if index == 0 else None,
        )
        axis.plot(
            wave_nm,
            q50[selected],
            color="tab:blue",
            lw=1.0,
            label="HMC median" if index == 0 else None,
        )
        axis.errorbar(
            bin_wave / 10.0,
            bin_observed,
            yerr=bin_error,
            fmt="o",
            ms=2.2,
            lw=0.55,
            color="black",
            ecolor="0.2",
            zorder=5,
            label="Local inverse-variance bins" if index == 0 else None,
        )
        axis.axhline(0.0, color="0.45", lw=0.6, ls="--")
        finite = np.concatenate(
            (
                observed[selected][np.isfinite(observed[selected])],
                q025[selected][np.isfinite(q025[selected])],
                q975[selected][np.isfinite(q975[selected])],
                bin_observed - bin_error,
                bin_observed + bin_error,
            )
        )
        low, high = np.nanpercentile(finite, [0.5, 99.5])
        padding = 0.12 * max(high - low, np.finfo(float).eps)
        axis.set_ylim(low - padding, high + padding)
        axis.set_title(
            f"{window['lo_A']:.1f}-{window['hi_A']:.1f} A; "
            rf"$\sum(m/\sigma)^2={window['score']:.2f}$",
            fontsize=8.5,
        )
        axis.set_xlabel("Wavelength [nm]")
        axis.set_ylabel("Processed residual flux")
        axis.grid(alpha=0.18)
    for axis in axes.flat[len(windows) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=4,
        fontsize=7.3,
        frameon=False,
    )
    fig.suptitle(f"{title}: strongest Fe-template windows", fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.006,
        "Windows are selected by posterior-median model power, not by observed excursions.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.91))
    save_figure_pdf(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_posterior_residuals(
    *,
    wavelength: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    draws: np.ndarray,
    title: str,
    output_path: Path,
) -> dict[str, np.ndarray]:
    native_q025, native_q16, native_q50, native_q84, native_q975 = _posterior_quantiles(draws)
    groups = _bin_groups(wavelength, max_bins=450)
    bin_wave, bin_observed, bin_error, bin_draws = _bin_products(
        wavelength, observed, error, draws, groups
    )
    bin_q025, bin_q16, bin_q50, bin_q84, bin_q975 = _posterior_quantiles(bin_draws)
    bin_residual = bin_observed - bin_q50

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.5, 1.45]},
        constrained_layout=True,
    )
    axis, model_axis, residual_axis = axes
    x_bin = bin_wave / 10.0
    x_native = wavelength / 10.0
    axis.fill_between(x_bin, bin_q025, bin_q975, color="tab:blue", alpha=0.12, lw=0.0)
    axis.fill_between(x_bin, bin_q16, bin_q84, color="tab:blue", alpha=0.28, lw=0.0)
    axis.plot(x_bin, bin_q50, color="tab:blue", lw=0.9, label="Consistently binned HMC model")
    axis.errorbar(
        x_bin,
        bin_observed,
        yerr=bin_error,
        fmt="o",
        ms=1.8,
        lw=0.45,
        color="black",
        ecolor="0.3",
        label="Inverse-variance binned observed",
    )
    axis.axhline(0.0, color="0.45", lw=0.6, ls="--")
    axis.set_ylabel("Binned residual flux")
    axis.set_title(f"{title}: posterior predictive spectrum and residuals", fontsize=10.5)
    axis.legend(fontsize=7.3, ncol=2, loc="upper right")
    axis.grid(alpha=0.18)

    model_axis.fill_between(
        x_native, native_q025, native_q975, color="tab:blue", alpha=0.12, lw=0.0
    )
    model_axis.fill_between(
        x_native, native_q16, native_q84, color="tab:blue", alpha=0.28, lw=0.0
    )
    model_axis.plot(x_native, native_q50, color="tab:blue", lw=0.7)
    model_axis.axhline(0.0, color="0.45", lw=0.6, ls="--")
    model_axis.set_ylabel("HMC model\n(native grid)")
    model_axis.grid(alpha=0.18)

    residual_axis.errorbar(
        x_bin,
        bin_residual,
        yerr=bin_error,
        fmt="o",
        ms=1.8,
        lw=0.45,
        color="black",
        ecolor="0.3",
    )
    residual_axis.axhline(0.0, color="0.35", lw=0.7, ls="--")
    residual_axis.set_xlabel("Wavelength [nm]")
    residual_axis.set_ylabel("Data - model")
    residual_axis.grid(alpha=0.18)
    save_figure_pdf(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "binned_wavelength_A": bin_wave,
        "binned_observed": bin_observed,
        "binned_error": bin_error,
        "binned_model_draws": bin_draws,
        "binned_residual": bin_residual,
    }


def _shift_template_segmented(
    wavelength: np.ndarray,
    template: np.ndarray,
    velocity_kms: float,
    weight: np.ndarray,
) -> np.ndarray:
    beta = float(velocity_kms) / SPEED_OF_LIGHT_KMS
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    shifted = np.full(template.shape, np.nan, dtype=float)
    for segment in wavelength_segment_slices(wavelength):
        segment_wave = wavelength[segment]
        shifted[segment] = np.interp(
            segment_wave / doppler_factor,
            segment_wave,
            template[segment],
            left=np.nan,
            right=np.nan,
        )
    valid = np.isfinite(shifted) & np.isfinite(weight) & (weight > 0.0)
    if np.any(valid):
        shifted[valid] -= np.average(shifted[valid], weights=weight[valid])
    return shifted


def _velocity_scan(
    wavelength: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    draws: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, np.ndarray]:
    base_valid = np.isfinite(observed) & np.isfinite(error) & (error > 0.0)
    weight = np.zeros(error.shape, dtype=float)
    weight[base_valid] = 1.0 / np.square(error[base_valid])
    alpha = np.full((draws.shape[0], velocities.size), np.nan)
    alpha_error = np.full_like(alpha, np.nan)
    signed_snr = np.full_like(alpha, np.nan)
    delta_chi2 = np.full_like(alpha, np.nan)
    for draw_index, template in enumerate(draws):
        for velocity_index, velocity in enumerate(velocities):
            shifted = _shift_template_segmented(wavelength, template, velocity, weight)
            valid = base_valid & np.isfinite(shifted)
            denominator = float(np.sum(weight[valid] * np.square(shifted[valid])))
            if denominator <= 0.0:
                continue
            numerator = float(np.sum(weight[valid] * observed[valid] * shifted[valid]))
            alpha[draw_index, velocity_index] = numerator / denominator
            alpha_error[draw_index, velocity_index] = denominator ** -0.5
            signed_snr[draw_index, velocity_index] = numerator / np.sqrt(denominator)
            delta_chi2[draw_index, velocity_index] = numerator**2 / denominator
    return {
        "velocity_kms": velocities,
        "alpha": alpha,
        "alpha_error": alpha_error,
        "signed_snr": signed_snr,
        "delta_chi2": delta_chi2,
    }


def _plot_velocity_scan(
    scan: dict[str, np.ndarray],
    *,
    title: str,
    output_path: Path,
) -> dict[str, float]:
    velocities = scan["velocity_kms"]
    snr_q = np.nanpercentile(scan["signed_snr"], [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    alpha_q = np.nanpercentile(scan["alpha"], [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    peak_index = int(np.nanargmax(snr_q[2]))
    zero_index = int(np.argmin(np.abs(velocities)))

    fig, (snr_axis, alpha_axis) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.1),
        sharex=True,
    )
    snr_axis.fill_between(velocities, snr_q[0], snr_q[4], color="tab:orange", alpha=0.12)
    snr_axis.fill_between(velocities, snr_q[1], snr_q[3], color="tab:orange", alpha=0.28)
    snr_axis.plot(velocities, snr_q[2], color="tab:orange", lw=1.2)
    snr_axis.axhline(0.0, color="0.4", lw=0.7, ls="--")
    snr_axis.axvline(0.0, color="0.25", lw=0.8, ls=":", label="Adopted planet frame")
    snr_axis.axvline(
        velocities[peak_index], color="tab:red", lw=0.8, ls="--", label="Template peak"
    )
    snr_axis.set_ylabel("Signed template statistic")
    snr_axis.set_title(f"{title}: velocity-shifted posterior template", fontsize=10.5)
    snr_axis.legend(fontsize=7.5, loc="best")
    snr_axis.grid(alpha=0.18)
    snr_axis.text(
        0.015,
        0.04,
        f"Peak: {velocities[peak_index]:+.2f} km/s, statistic={snr_q[2, peak_index]:.2f}",
        transform=snr_axis.transAxes,
        fontsize=8,
    )

    alpha_axis.fill_between(
        velocities, alpha_q[0], alpha_q[4], color="tab:blue", alpha=0.12
    )
    alpha_axis.fill_between(
        velocities, alpha_q[1], alpha_q[3], color="tab:blue", alpha=0.28
    )
    alpha_axis.plot(velocities, alpha_q[2], color="tab:blue", lw=1.2)
    alpha_axis.axhline(0.0, color="0.4", lw=0.7, ls="--")
    alpha_axis.axhline(1.0, color="0.55", lw=0.7, ls=":")
    alpha_axis.axvline(0.0, color="0.25", lw=0.8, ls=":")
    alpha_axis.axvline(velocities[peak_index], color="tab:red", lw=0.8, ls="--")
    alpha_axis.set_xlabel("Template velocity offset [km/s]")
    alpha_axis.set_ylabel(r"Fitted amplitude $\alpha$")
    alpha_axis.grid(alpha=0.18)
    fig.text(
        0.5,
        0.025,
        "In-sample alignment diagnostic only: the posterior template was inferred from these data.",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.92, hspace=0.08)
    save_figure_pdf(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "peak_velocity_kms": float(velocities[peak_index]),
        "peak_signed_statistic": float(snr_q[2, peak_index]),
        "zero_velocity_signed_statistic": float(snr_q[2, zero_index]),
        "zero_velocity_alpha": float(alpha_q[2, zero_index]),
    }


def _likelihood_products(
    observed: np.ndarray,
    error: np.ndarray,
    draws: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    valid = np.isfinite(observed) & np.isfinite(error) & (error > 0.0)
    valid &= np.all(np.isfinite(draws), axis=0)
    data = observed[valid]
    sigma = error[valid]
    models = draws[:, valid]
    weight = 1.0 / np.square(sigma)
    chi2_zero = float(np.sum(weight * np.square(data)))
    chi2_model = np.sum(weight[None, :] * np.square(data[None, :] - models), axis=1)
    denominator = np.sum(weight[None, :] * np.square(models), axis=1)
    numerator = np.sum(weight[None, :] * data[None, :] * models, axis=1)
    alpha = numerator / denominator
    chi2_profiled = np.sum(
        weight[None, :] * np.square(data[None, :] - alpha[:, None] * models), axis=1
    )
    median_model = np.nanpercentile(models, 50.0, axis=0)
    delta_pixel = weight * (np.square(data) - np.square(data - median_model))
    return {
        "valid": valid,
        "n_valid": int(data.size),
        "chi2_zero": chi2_zero,
        "chi2_model": chi2_model,
        "chi2_profiled": chi2_profiled,
        "delta_chi2_fixed": chi2_zero - chi2_model,
        "delta_chi2_profiled": chi2_zero - chi2_profiled,
        "alpha": alpha,
        "delta_chi2_pixel_median_model": delta_pixel,
    }


def _plot_likelihood_improvement(
    *,
    wavelength: np.ndarray,
    likelihood: dict[str, np.ndarray | float | int],
    title: str,
    output_path: Path,
) -> dict[str, float | int | list[float]]:
    valid = np.asarray(likelihood["valid"], dtype=bool)
    n_valid = int(likelihood["n_valid"])
    chi2_zero = float(likelihood["chi2_zero"])
    chi2_model = np.asarray(likelihood["chi2_model"], dtype=float)
    chi2_profiled = np.asarray(likelihood["chi2_profiled"], dtype=float)
    delta_fixed = np.asarray(likelihood["delta_chi2_fixed"], dtype=float)
    delta_profiled = np.asarray(likelihood["delta_chi2_profiled"], dtype=float)
    delta_pixel = np.asarray(likelihood["delta_chi2_pixel_median_model"], dtype=float)
    alpha = np.asarray(likelihood["alpha"], dtype=float)
    valid_wave = wavelength[valid]

    block_groups = [group for group in np.array_split(np.arange(valid_wave.size), 100) if group.size]
    block_wave = np.asarray([np.mean(valid_wave[group]) for group in block_groups])
    block_delta = np.asarray([np.sum(delta_pixel[group]) for group in block_groups])
    cumulative = np.cumsum(delta_pixel)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 7.8),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.55]},
    )
    chi_axis, delta_axis, contribution_axis = axes
    chi_axis.scatter([0.0], [chi2_zero / n_valid], color="0.2", s=30, zorder=4)
    chi_axis.boxplot(
        [chi2_model / n_valid, chi2_profiled / n_valid],
        positions=[1.0, 2.0],
        widths=0.45,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "tab:blue", "alpha": 0.25},
        medianprops={"color": "tab:blue", "linewidth": 1.2},
    )
    chi_axis.set_xticks([0.0, 1.0, 2.0])
    chi_axis.set_xticklabels(["Zero model", "HMC draws", r"HMC draws $\times\alpha$"])
    chi_axis.set_ylabel(r"$\chi^2/N$")
    chi_axis.set_title(f"{title}: in-sample likelihood improvement", fontsize=10.5)
    chi_axis.grid(axis="y", alpha=0.18)

    positions = np.arange(delta_fixed.size)
    delta_axis.plot(positions, delta_fixed, "o", ms=3.0, color="tab:blue", label="Fixed amplitude")
    delta_axis.plot(
        positions,
        delta_profiled,
        "o",
        ms=3.0,
        color="tab:orange",
        label=r"Profiled amplitude $\alpha$",
    )
    delta_axis.axhline(0.0, color="0.4", lw=0.7, ls="--")
    delta_axis.set_xlabel("Saved posterior model draw")
    delta_axis.set_ylabel(r"$\Delta\chi^2$ versus zero")
    delta_axis.legend(fontsize=7.5, ncol=2)
    delta_axis.grid(alpha=0.18)

    colors = np.where(block_delta >= 0.0, "tab:blue", "tab:red")
    width = 0.8 * float(np.median(np.diff(block_wave)))
    contribution_axis.bar(block_wave / 10.0, block_delta, width=width / 10.0, color=colors, alpha=0.55)
    contribution_axis.axhline(0.0, color="0.35", lw=0.7)
    contribution_axis.set_xlabel("Wavelength [nm]")
    contribution_axis.set_ylabel(r"Block $\Delta\chi^2$")
    cumulative_axis = contribution_axis.twinx()
    cumulative_axis.plot(valid_wave / 10.0, cumulative, color="0.15", lw=0.8)
    cumulative_axis.set_ylabel(r"Cumulative $\Delta\chi^2$")
    contribution_axis.grid(axis="y", alpha=0.18)
    fig.text(
        0.5,
        0.022,
        "Descriptive comparison to a zero processed spectrum; not a Bayes factor or independent detection significance.",
        ha="center",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.13, top=0.94, hspace=0.46)
    save_figure_pdf(fig, output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    quantiles = [2.5, 16.0, 50.0, 84.0, 97.5]
    return {
        "n_valid_pixels": n_valid,
        "chi2_zero": chi2_zero,
        "chi2_zero_per_pixel": chi2_zero / n_valid,
        "chi2_model_per_pixel_quantiles": np.percentile(
            chi2_model / n_valid, quantiles
        ).tolist(),
        "delta_chi2_fixed_quantiles": np.percentile(delta_fixed, quantiles).tolist(),
        "delta_chi2_profiled_quantiles": np.percentile(delta_profiled, quantiles).tolist(),
        "alpha_quantiles": np.percentile(alpha, quantiles).tolist(),
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.num_windows < 1:
        raise SystemExit("--num-windows must be positive.")
    if args.window_width_A <= 0.0 or args.window_separation_A < 0.0:
        raise SystemExit("Window width must be positive and separation must be nonnegative.")
    if args.velocity_step_kms <= 0.0 or args.velocity_max_kms <= args.velocity_min_kms:
        raise SystemExit("Velocity bounds and step do not define a positive grid.")

    configure_matplotlib()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "posthoc_fe_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    component_id = _slug(args.component)
    sidecar_path = run_dir / "figure_data" / f"planet_frame_spectrum_{component_id}.npz"
    saved = _load_sidecar(sidecar_path)
    wavelength = saved["wavelength_A"]
    observed = saved["observed"]
    error = saved["observed_error"]
    draws = saved["model_draws"]
    median_model = np.nanpercentile(draws, 50.0, axis=0)
    title = _run_title(run_dir, args.component)

    windows = _rank_model_windows(
        wavelength,
        median_model,
        error,
        count=args.num_windows,
        width_A=args.window_width_A,
        separation_A=args.window_separation_A,
    )
    _plot_zoom_panels(
        wavelength=wavelength,
        observed=observed,
        error=error,
        draws=draws,
        windows=windows,
        title=title,
        output_path=output_dir / "fe_rich_zoom_panels.pdf",
    )
    binned = _plot_posterior_residuals(
        wavelength=wavelength,
        observed=observed,
        error=error,
        draws=draws,
        title=title,
        output_path=output_dir / "posterior_predictive_residuals.pdf",
    )

    velocities = np.arange(
        args.velocity_min_kms,
        args.velocity_max_kms + 0.5 * args.velocity_step_kms,
        args.velocity_step_kms,
    )
    velocity_scan = _velocity_scan(wavelength, observed, error, draws, velocities)
    velocity_summary = _plot_velocity_scan(
        velocity_scan,
        title=title,
        output_path=output_dir / "velocity_template_ccf.pdf",
    )

    likelihood = _likelihood_products(observed, error, draws)
    likelihood_summary = _plot_likelihood_improvement(
        wavelength=wavelength,
        likelihood=likelihood,
        title=title,
        output_path=output_dir / "likelihood_improvement.pdf",
    )

    np.savez_compressed(
        output_dir / "saved_fe_diagnostics_data.npz",
        source_wavelength_A=wavelength,
        source_observed=observed,
        source_observed_error=error,
        source_model_draws=draws,
        selected_window_lo_A=np.asarray([window["lo_A"] for window in windows]),
        selected_window_hi_A=np.asarray([window["hi_A"] for window in windows]),
        selected_window_power=np.asarray([window["score"] for window in windows]),
        **binned,
        velocity_kms=velocity_scan["velocity_kms"],
        velocity_alpha_draws=velocity_scan["alpha"],
        velocity_alpha_error_draws=velocity_scan["alpha_error"],
        velocity_signed_statistic_draws=velocity_scan["signed_snr"],
        velocity_delta_chi2_draws=velocity_scan["delta_chi2"],
        likelihood_delta_chi2_fixed=np.asarray(likelihood["delta_chi2_fixed"]),
        likelihood_delta_chi2_profiled=np.asarray(likelihood["delta_chi2_profiled"]),
        likelihood_alpha=np.asarray(likelihood["alpha"]),
    )
    summary = {
        "run_dir": str(run_dir),
        "source_sidecar": str(sidecar_path),
        "component": args.component,
        "posterior_model_draw_count": int(draws.shape[0]),
        "wavelength_pixel_count": int(wavelength.size),
        "selected_windows": windows,
        "velocity_diagnostic": velocity_summary,
        "likelihood_improvement": likelihood_summary,
        "interpretation_warnings": [
            "The velocity/template statistic is in-sample because the posterior template was inferred from the same data.",
            "The zero-model likelihood comparison is descriptive and is not a Bayes factor.",
            "Correlated wavelength pixels invalidate a naive independent-pixel sigma interpretation.",
        ],
    }
    (output_dir / "saved_fe_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
