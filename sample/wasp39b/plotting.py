"""
Plotting Module
===============

Visualization functions for SVI and MCMC diagnostics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner


def plot_svi_loss(loss_values, save_path):
    """
    Plot SVI loss trajectory.

    Parameters
    ----------
    loss_values : np.ndarray
        Loss values over iterations
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(loss_values))
    ax.plot(x, np.asarray(loss_values), lw=1.5)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("Loss")
    ax.set_title("SVI loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"SVI loss plot saved to {save_path}")


def plot_spectrum_overlay(wavelength_nm, rp_obs, rp_err, rp_hmc, rp_svi, save_path):
    """
    Plot observed vs SVI vs HMC spectra.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength array [nm]
    rp_obs : np.ndarray
        Observed R_p/R_s
    rp_err : np.ndarray
        Observed uncertainty
    rp_hmc : np.ndarray
        HMC predictions (samples x wavelength)
    rp_svi : np.ndarray
        SVI prediction
    save_path : str
        Path to save figure
    """
    rp_hmc_np = np.asarray(rp_hmc)
    mean = rp_hmc_np.mean(axis=0)
    std = rp_hmc_np.std(axis=0)
    rp_svi_np = np.asarray(rp_svi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        wavelength_nm,
        rp_obs,
        yerr=rp_err,
        fmt=".",
        ms=1,
        color="k",
        ecolor="0.3",
        elinewidth=0.5,
        alpha=0.5,
        label="Observed",
    )
    ax.fill_between(
        wavelength_nm,
        mean - std,
        mean + std,
        color="C0",
        alpha=0.25,
        label="HMC ±1$\sigma$",
    )
    ax.plot(wavelength_nm, mean, color="C0", lw=1.4, label="HMC mean")
    ax.plot(wavelength_nm, rp_svi_np, color="C3", lw=1.4, label="SVI median model")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$R_p/R_s$")
    ax.set_title("Observed vs SVI vs HMC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Spectrum overlay plot saved to {save_path}")


def _corner_data(sample_dict, variables):
    """
    Extract corner plot data from sample dictionary.

    Parameters
    ----------
    sample_dict : dict
        Dictionary of samples
    variables : list
        List of variable names to include

    Returns
    -------
    data : np.ndarray or None
        Column-stacked data array
    labels : list or None
        Variable labels
    """
    cols = []
    labels = []
    available = [v for v in variables if v in sample_dict]
    for var in available:
        arr = np.asarray(sample_dict[var])
        arr = arr.reshape(arr.shape[0], -1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            labels.append(var if arr.shape[1] == 1 else f"{var}[{j}]")
    if not cols:
        return None, None
    return np.column_stack(cols), labels


def plot_corner(hmc_samples=None, svi_samples=None, variables=None, save_path=None):
    """
    Create corner plot for parameter distributions.

    Parameters
    ----------
    hmc_samples : dict, optional
        HMC posterior samples
    svi_samples : dict, optional
        SVI samples
    variables : list
        Variable names to include
    save_path : str
        Path to save figure
    """
    datasets = []
    labels = None

    if hmc_samples is not None:
        hmc_data, labels = _corner_data(hmc_samples, variables)
        if hmc_data is not None:
            datasets.append((hmc_data, "C0", {}))

    if svi_samples is not None:
        svi_data, labels_svi = _corner_data(svi_samples, variables)
        if labels is None:
            labels = labels_svi
        if svi_data is not None:
            datasets.append((svi_data, "C3", {"hist_kwargs": {"linestyle": "--"}}))

    if not datasets or labels is None:
        print("No data for corner plot; skipping.")
        return

    fig = None
    for data, color, extra_kwargs in datasets:
        fig = corner.corner(
            data,
            labels=labels,
            color=color,
            bins=40,
            smooth=1.0,
            fig=fig,
            show_titles=True,
            **extra_kwargs,
        )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Corner plot saved to {save_path}")


def create_all_plots(
    losses,
    wav_obs,
    rp_mean,
    rp_std,
    predictions,
    svi_mu,
    posterior_sample,
    svi_samples,
    opa_mols,
    output_dir,
):
    """
    Generate all diagnostic plots.

    Parameters
    ----------
    losses : np.ndarray
        SVI loss values
    wav_obs : np.ndarray
        Observed wavelength [nm]
    rp_mean : np.ndarray
        Observed spectrum
    rp_std : np.ndarray
        Observed uncertainty
    predictions : dict
        HMC predictions
    svi_mu : np.ndarray
        SVI prediction
    posterior_sample : dict
        HMC posterior samples
    svi_samples : dict
        SVI samples
    opa_mols : dict
        Molecular opacities (for variable names)
    output_dir : str
        Output directory
    """
    print("Generating diagnostic plots...")

    # SVI loss
    loss_plot_path = os.path.join(output_dir, "svi_loss.png")
    plot_svi_loss(losses, loss_plot_path)

    # Spectrum overlay
    overlay_plot_path = os.path.join(output_dir, "spectrum_overlay.png")
    plot_spectrum_overlay(
        wav_obs, rp_mean, rp_std, predictions["rp"], svi_mu, overlay_plot_path
    )

    # Corner plots
    corner_vars = ["Radius_btm", "T0", "logP_cloud", "RV"]
    corner_vars += [f"logVMR_{mol}" for mol in list(opa_mols.keys())]

    corner_plot_path = os.path.join(output_dir, "corner_plot_svi.png")
    plot_corner(svi_samples=svi_samples, variables=corner_vars, save_path=corner_plot_path)

    hmc_corner_plot_path = os.path.join(output_dir, "corner_plot.png")
    plot_corner(
        hmc_samples=posterior_sample, variables=corner_vars, save_path=hmc_corner_plot_path
    )

    hmc_svi_corner_overlay_path = os.path.join(
        output_dir, "corner_plot_hmc_svi_overlay.png"
    )
    plot_corner(
        hmc_samples=posterior_sample,
        svi_samples=svi_samples,
        variables=corner_vars,
        save_path=hmc_svi_corner_overlay_path,
    )

    print("All plots generated successfully.")
