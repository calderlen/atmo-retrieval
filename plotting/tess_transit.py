"""Saved figures for the headless TESS transit fitting workflow."""

from __future__ import annotations

from typing import Any, Mapping

from plotting.style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np


def plot_lightcurve_overview(
    dataset: Mapping[str, Any],
    *,
    period_d: float,
    t0_btjd: float,
    target: str,
):
    """Plot the prepared multi-sector light curve folded at the input ephemeris."""

    time = np.asarray(dataset["time"], dtype=float)
    flux = np.asarray(dataset["flux"], dtype=float)
    uncertainty = np.asarray(dataset["flux_err"], dtype=float)
    sector_index = np.asarray(dataset["sector_idx"], dtype=int)
    phase = ((time - t0_btjd + 0.5 * period_d) % period_d) - 0.5 * period_d
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].errorbar(time, flux, yerr=uncertainty, fmt=".", ms=2, alpha=0.35, color="0.25")
    axes[0].set_xlabel("BTJD")
    axes[0].set_ylabel("Normalized flux - 1")
    axes[0].set_title(f"{target} prepared TESS cadences")
    for sector in np.unique(sector_index):
        selected = sector_index == sector
        label = str(dataset["sector_labels"][int(sector)])
        axes[1].plot(phase[selected], flux[selected], ".", ms=2.5, alpha=0.4, label=label)
    axes[1].set_xlabel("Time from mid-transit [days]")
    axes[1].set_ylabel("Normalized flux - 1")
    axes[1].set_title("Folded sector overlay")
    axes[1].legend(fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_transit_fit(result):
    """Plot the best-fit transit, GP trend, and residuals."""

    dataset = result.dataset
    best = result.best_fit
    phase = np.asarray(best["phase_days"], dtype=float)
    order = np.argsort(phase)
    flux = np.asarray(dataset["flux"], dtype=float)
    uncertainty = np.asarray(dataset["flux_err"], dtype=float)
    model = np.asarray(best["mean_model"], dtype=float)
    detrended = np.asarray(best["detrended_flux"], dtype=float)
    residual = np.asarray(best["residual_flux"], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].errorbar(phase, flux, yerr=uncertainty, fmt=".", ms=2, alpha=0.25, color="0.3")
    axes[0].plot(phase[order], model[order], color="tab:red", lw=1.2)
    axes[0].set_ylabel("Flux - 1")
    axes[0].set_title("Joint transit and per-sector GP fit")
    axes[1].errorbar(phase, detrended, yerr=uncertainty, fmt=".", ms=2, alpha=0.3, color="0.3")
    axes[1].plot(best["phase_grid"], best["phase_model_grid"], color="tab:red", lw=1.4)
    axes[1].set_ylabel("GP-detrended flux")
    axes[2].errorbar(phase, residual, yerr=uncertainty, fmt=".", ms=2, alpha=0.3, color="0.3")
    axes[2].axhline(0.0, color="tab:red", ls="--", lw=0.8)
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Time from mid-transit [days]")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_posterior_corner(result):
    """Plot the seven global sampled parameters when corner is installed."""

    try:
        import corner
    except ModuleNotFoundError:
        return None
    samples = np.asarray(result.best_fit["posterior_theta"], dtype=float)
    labels = (
        r"$\log P$",
        r"$t_0$",
        r"$\log \rho_\star$",
        r"$\log R_p/R_\star$",
        r"$b/(1+R_p/R_\star)$",
        r"$q_1$",
        r"$q_2$",
    )
    return corner.corner(
        samples[:, : len(labels)],
        labels=labels,
        show_titles=True,
        title_fmt=".4g",
    )
