"""
Plotting Module
===============

Visualization functions for retrieval diagnostics.
Supports both transmission and emission spectra.
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


def plot_transmission_spectrum(wavelength_nm, rp_obs, rp_err, rp_hmc, rp_svi, save_path):
    """
    Plot transmission spectrum: observed vs model.

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        wavelength_nm,
        rp_obs,
        yerr=rp_err,
        fmt=".",
        ms=2,
        color="k",
        ecolor="0.3",
        elinewidth=0.5,
        alpha=0.6,
        label="Observed",
    )
    ax.fill_between(
        wavelength_nm,
        mean - std,
        mean + std,
        color="C0",
        alpha=0.25,
        label="HMC ±1σ",
    )
    ax.plot(wavelength_nm, mean, color="C0", lw=1.5, label="HMC mean")
    ax.plot(wavelength_nm, rp_svi_np, color="C3", lw=1.5, ls="--", label="SVI median")
    ax.set_xlabel("Wavelength [nm]", fontsize=12)
    ax.set_ylabel(r"$R_p/R_\star$", fontsize=12)
    ax.set_title("Transmission Spectrum", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Transmission spectrum plot saved to {save_path}")


def plot_emission_spectrum(wavelength_nm, fp_obs, fp_err, fp_hmc, fp_svi, save_path):
    """
    Plot emission spectrum: observed vs model.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength array [nm]
    fp_obs : np.ndarray
        Observed F_p/F_s
    fp_err : np.ndarray
        Observed uncertainty
    fp_hmc : np.ndarray
        HMC predictions (samples x wavelength)
    fp_svi : np.ndarray
        SVI prediction
    save_path : str
        Path to save figure
    """
    fp_hmc_np = np.asarray(fp_hmc)
    mean = fp_hmc_np.mean(axis=0)
    std = fp_hmc_np.std(axis=0)
    fp_svi_np = np.asarray(fp_svi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        wavelength_nm,
        fp_obs,
        yerr=fp_err,
        fmt=".",
        ms=2,
        color="k",
        ecolor="0.3",
        elinewidth=0.5,
        alpha=0.6,
        label="Observed",
    )
    ax.fill_between(
        wavelength_nm,
        mean - std,
        mean + std,
        color="C1",
        alpha=0.25,
        label="HMC ±1σ",
    )
    ax.plot(wavelength_nm, mean, color="C1", lw=1.5, label="HMC mean")
    ax.plot(wavelength_nm, fp_svi_np, color="C3", lw=1.5, ls="--", label="SVI median")
    ax.set_xlabel("Wavelength [nm]", fontsize=12)
    ax.set_ylabel(r"$F_p/F_\star$", fontsize=12)
    ax.set_title("Emission Spectrum", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Emission spectrum plot saved to {save_path}")


def plot_temperature_profile(posterior_samples, art, save_path, Ncurve=100):
    """
    Plot retrieved temperature-pressure profile.

    Parameters
    ----------
    posterior_samples : dict
        Posterior samples from MCMC
    art : ArtEmisPure or ArtTransPure
        Atmospheric RT object
    save_path : str
        Path to save figure
    Ncurve : int
        Number of random curves to plot
    """
    import jax.numpy as jnp

    # Extract temperature samples
    # This depends on temperature parametrization
    # For isothermal: single T0
    # For free: multiple T_nodes

    fig, ax = plt.subplots(figsize=(6, 7))

    if "T0" in posterior_samples:
        # Isothermal
        T0_samples = np.asarray(posterior_samples["T0"])
        for i in np.random.choice(len(T0_samples), min(Ncurve, len(T0_samples)), replace=False):
            Tarr = T0_samples[i] * np.ones_like(art.pressure)
            ax.plot(Tarr, art.pressure, "C0-", alpha=0.05)

        # Median
        T0_median = np.median(T0_samples)
        Tarr_median = T0_median * np.ones_like(art.pressure)
        ax.plot(Tarr_median, art.pressure, "C0-", lw=2, label="Median (isothermal)")

    elif "T_btm" in posterior_samples:
        # Gradient profile
        T_btm_samples = np.asarray(posterior_samples["T_btm"])
        T_top_samples = np.asarray(posterior_samples["T_top"])

        for i in np.random.choice(len(T_btm_samples), min(Ncurve, len(T_btm_samples)), replace=False):
            log_p = np.log10(art.pressure)
            log_p_btm = np.log10(art.pressure[-1])
            log_p_top = np.log10(art.pressure[0])
            Tarr = T_top_samples[i] + (T_btm_samples[i] - T_top_samples[i]) * \
                   (log_p - log_p_top) / (log_p_btm - log_p_top)
            ax.plot(Tarr, art.pressure, "C0-", alpha=0.05)

        # Median
        T_btm_med = np.median(T_btm_samples)
        T_top_med = np.median(T_top_samples)
        log_p = np.log10(art.pressure)
        log_p_btm = np.log10(art.pressure[-1])
        log_p_top = np.log10(art.pressure[0])
        Tarr_median = T_top_med + (T_btm_med - T_top_med) * (log_p - log_p_top) / (log_p_btm - log_p_top)
        ax.plot(Tarr_median, art.pressure, "C0-", lw=2, label="Median (gradient)")

    ax.set_xlabel("Temperature [K]", fontsize=12)
    ax.set_ylabel("Pressure [bar]", fontsize=12)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_title("Temperature-Pressure Profile", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"T-P profile plot saved to {save_path}")


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


def create_transmission_plots(
    losses,
    wav_obs,
    rp_mean,
    rp_std,
    predictions,
    svi_mu,
    posterior_sample,
    svi_samples,
    opa_mols,
    art,
    output_dir,
):
    """
    Generate all transmission retrieval diagnostic plots.

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
    art : ArtTransPure
        Atmospheric RT object
    output_dir : str
        Output directory
    """
    print("Generating diagnostic plots...")

    # SVI loss
    plot_svi_loss(losses, os.path.join(output_dir, "svi_loss.png"))

    # Transmission spectrum
    plot_transmission_spectrum(
        wav_obs, rp_mean, rp_std, predictions["rp"], svi_mu,
        os.path.join(output_dir, "transmission_spectrum.png")
    )

    # Temperature profile
    plot_temperature_profile(
        posterior_sample, art, os.path.join(output_dir, "temperature_profile.png")
    )

    # Corner plots
    corner_vars = ["Radius_btm", "T0", "logP_cloud", "RV"]
    corner_vars += [f"logVMR_{mol}" for mol in list(opa_mols.keys())]

    plot_corner(
        svi_samples=svi_samples,
        variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_svi.png")
    )

    plot_corner(
        hmc_samples=posterior_sample,
        variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_hmc.png")
    )

    plot_corner(
        hmc_samples=posterior_sample,
        svi_samples=svi_samples,
        variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_overlay.png")
    )

    print("All plots generated successfully.")
