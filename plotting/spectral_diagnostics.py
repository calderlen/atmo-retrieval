"""Shared non-interactive figures for prepared HRS diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from plotting.style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

from spectroscopy.spectral_diagnostics import (
    binned_series,
    column_metrics,
    observed_spectrum,
    phase_order,
    phase_values,
    robust_limit,
    wave_1d,
)


def plot_bundle_overview(bundle: Mapping[str, Any]):
    """Plot final data, uncertainty, and optional pre-SYSREM data matrices."""

    order = phase_order(bundle)
    phase = phase_values(bundle)[order]
    data = np.asarray(bundle["data"], dtype=float)[order]
    sigma = np.asarray(bundle["sigma"], dtype=float)[order]
    panels = [("Final residuals", data, True), ("Uncertainty", sigma, False)]
    if "pre_sysrem_data" in bundle:
        panels.insert(0, ("Pre-SYSREM residuals", np.asarray(bundle["pre_sysrem_data"])[order], True))
    figure, axes = plt.subplots(len(panels), 1, figsize=(12, 3.4 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (title, values, symmetric) in zip(axes, panels):
        if symmetric:
            limit = robust_limit(values)
            image = axis.imshow(values, aspect="auto", origin="upper", cmap="RdBu_r", vmin=-limit, vmax=limit)
        else:
            finite = values[np.isfinite(values)]
            maximum = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            image = axis.imshow(values, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=maximum)
        axis.set_title(title)
        axis.set_ylabel("Phase-ordered exposure")
        figure.colorbar(image, ax=axis, pad=0.01)
    axes[-1].set_xlabel("Wavelength column")
    figure.suptitle(f"{bundle['planet']} {bundle['mode']} {bundle['epoch']} {bundle['arm']} {bundle['product']}\nphase={np.nanmin(phase):.4f}..{np.nanmax(phase):.4f}")
    figure.tight_layout()
    return figure


def plot_bundle_profiles(bundle: Mapping[str, Any]):
    """Plot exposure-level residual and uncertainty summaries."""

    order = phase_order(bundle)
    phase = phase_values(bundle)[order]
    data = np.asarray(bundle["data"], dtype=float)[order]
    sigma = np.asarray(bundle["sigma"], dtype=float)[order]
    with np.errstate(invalid="ignore"):
        residual_rms = np.sqrt(np.nanmean(np.square(data), axis=1))
        median_sigma = np.nanmedian(sigma, axis=1)
        finite_fraction = np.mean(np.isfinite(data) & np.isfinite(sigma) & (sigma > 0), axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(phase, residual_rms, marker="o", lw=1)
    axes[0].set_ylabel("Residual RMS")
    axes[1].plot(phase, median_sigma, marker="o", lw=1)
    axes[1].set_ylabel("Median sigma")
    axes[2].plot(phase, finite_fraction, marker="o", lw=1)
    axes[2].set_ylabel("Finite fraction")
    axes[2].set_xlabel("Orbital phase" + (" (mod 1)" if bundle.get("mode") == "emission" else ""))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"{bundle['epoch']} {bundle['arm']} exposure profiles")
    figure.tight_layout()
    return figure


def plot_observed_spectrum(bundle: Mapping[str, Any], *, max_bins: int = 900):
    """Plot the weighted residual spectrum and uncertainty."""

    wavelength = wave_1d(bundle["wavelength"])
    mean, error, coverage = observed_spectrum(bundle["data"], bundle["sigma"])
    binned_wave, binned_mean, binned_error = binned_series(
        wavelength,
        mean,
        error,
        max_bins=max_bins,
    )
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(binned_wave, binned_mean, color="black", lw=0.9)
    if binned_error is not None:
        axes[0].fill_between(binned_wave, binned_mean - binned_error, binned_mean + binned_error, color="0.7", alpha=0.4)
    axes[0].axhline(0.0, color="0.5", ls="--", lw=0.8)
    axes[0].set_ylabel("Weighted residual")
    binned_wave_cov, binned_coverage, _ = binned_series(wavelength, coverage, max_bins=max_bins)
    axes[1].plot(binned_wave_cov, binned_coverage, color="tab:blue", lw=0.9)
    axes[1].set_ylabel("Exposure coverage")
    axes[1].set_xlabel("Vacuum wavelength [Angstrom]")
    figure.suptitle(f"{bundle['epoch']} {bundle['arm']} observed residual summary")
    figure.tight_layout()
    return figure


def plot_column_quality(bundle: Mapping[str, Any], *, max_bins: int = 900):
    """Plot per-column finite coverage and uncertainty metrics."""

    wavelength = wave_1d(bundle["wavelength"])
    metrics = column_metrics(bundle)
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for axis, (name, label) in zip(
        axes,
        (
            ("finite_fraction", "Finite fraction"),
            ("median_sigma", "Median sigma"),
            ("median_abs_data", "Median |residual|"),
        ),
    ):
        out_wave, out_value, _ = binned_series(wavelength, metrics[name], max_bins=max_bins)
        axis.plot(out_wave, out_value, lw=0.9)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Vacuum wavelength [Angstrom]")
    figure.suptitle(f"{bundle['epoch']} {bundle['arm']} wavelength-column quality")
    figure.tight_layout()
    return figure


def plot_collapsed_product(bundle: Mapping[str, Any], *, max_bins: int = 900):
    """Plot the persisted one-dimensional collapsed product, if available."""

    collapsed = bundle.get("collapsed")
    if collapsed is None:
        return None
    wave, spectrum, uncertainty = binned_series(
        collapsed["wavelength"],
        collapsed["spectrum"],
        collapsed["uncertainty"],
        max_bins=max_bins,
    )
    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.plot(wave, spectrum, color="black", lw=0.9)
    if uncertainty is not None:
        axis.fill_between(wave, spectrum - uncertainty, spectrum + uncertainty, color="0.7", alpha=0.4)
    axis.axhline(0.0, color="0.5", ls="--", lw=0.8)
    axis.set_xlabel("Vacuum wavelength [Angstrom]")
    axis.set_ylabel("Collapsed spectrum")
    axis.set_title(f"{bundle['epoch']} {bundle['arm']} persisted collapsed product")
    figure.tight_layout()
    return figure


def plot_stacked_spectra(stacked: Mapping[str, Any]):
    """Plot weighted spectra from multiple epoch/arm rows."""

    rows = list(stacked.get("rows", []))
    if not rows:
        return None
    figure, axes = plt.subplots(len(rows), 1, figsize=(13, max(3.0, 2.6 * len(rows))), squeeze=False)
    for axis, row in zip(axes[:, 0], rows):
        wave, spectrum, uncertainty = binned_series(
            row["wavelength"],
            row["spectrum"],
            row["uncertainty"],
            max_bins=900,
        )
        axis.plot(wave, spectrum, color="black", lw=0.8)
        if uncertainty is not None:
            axis.fill_between(wave, spectrum - uncertainty, spectrum + uncertainty, color="0.8", alpha=0.4)
        axis.axhline(0.0, color="0.5", ls="--", lw=0.7)
        axis.set_ylabel(str(row["label"]))
    axes[-1, 0].set_xlabel("Vacuum wavelength [Angstrom]")
    figure.suptitle(f"Stacked spectra grouped by {stacked['group_by']}")
    figure.tight_layout()
    return figure


def plot_line_diagnostic(
    line_result: Mapping[str, np.ndarray],
    *,
    title: str,
):
    """Plot a sampled line matrix and its weighted coadd."""

    velocity = np.asarray(line_result["velocity_kms"])
    data = np.asarray(line_result["data"])
    limit = robust_limit(data, percentile=99.0)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": (2, 1)}, sharex=True)
    image = axes[0].imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=(velocity[0], velocity[-1], data.shape[0], 0),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    figure.colorbar(image, ax=axes[0], pad=0.01, label="Residual")
    axes[0].set_ylabel("Exposure")
    axes[1].plot(velocity, line_result["coadd"], color="black")
    axes[1].fill_between(
        velocity,
        line_result["coadd"] - line_result["coadd_error"],
        line_result["coadd"] + line_result["coadd_error"],
        color="0.7",
        alpha=0.4,
    )
    axes[1].axvline(0.0, color="tab:red", ls="--", lw=0.8)
    axes[1].set_xlabel("Velocity [km/s]")
    axes[1].set_ylabel("Weighted coadd")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def close_figures(figures: Sequence[Any]) -> None:
    """Close every non-null figure returned by a diagnostic renderer."""

    for figure in figures:
        if figure is not None:
            plt.close(figure)
