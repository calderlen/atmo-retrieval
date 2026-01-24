"""Visualization functions for retrieval diagnostics."""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner


def plot_svi_loss(loss_values: np.ndarray, save_path: str) -> None:
    """Plot SVI loss trajectory."""
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


def plot_transmission_spectrum(
    wavelength_nm: np.ndarray,
    rp_obs: np.ndarray,
    rp_err: np.ndarray,
    rp_hmc: np.ndarray,
    rp_svi: np.ndarray,
    save_path: str,
) -> None:
    """Plot transmission spectrum: observed vs model."""
    rp_hmc_np = np.asarray(rp_hmc)
    mean = rp_hmc_np.mean(axis=0)
    std = rp_hmc_np.std(axis=0)
    rp_svi_np = np.asarray(rp_svi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        wavelength_nm, rp_obs, yerr=rp_err,
        fmt=".", ms=2, color="k", ecolor="0.3", elinewidth=0.5, alpha=0.6, label="Observed",
    )
    ax.fill_between(wavelength_nm, mean - std, mean + std, color="C0", alpha=0.25, label="HMC ±1σ")
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


def plot_emission_spectrum(
    wavelength_nm: np.ndarray,
    fp_obs: np.ndarray,
    fp_err: np.ndarray,
    fp_hmc: np.ndarray,
    fp_svi: np.ndarray,
    save_path: str,
) -> None:
    """Plot emission spectrum: observed vs model."""
    fp_hmc_np = np.asarray(fp_hmc)
    mean = fp_hmc_np.mean(axis=0)
    std = fp_hmc_np.std(axis=0)
    fp_svi_np = np.asarray(fp_svi)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        wavelength_nm, fp_obs, yerr=fp_err,
        fmt=".", ms=2, color="k", ecolor="0.3", elinewidth=0.5, alpha=0.6, label="Observed",
    )
    ax.fill_between(wavelength_nm, mean - std, mean + std, color="C1", alpha=0.25, label="HMC ±1σ")
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


def plot_temperature_profile(
    posterior_samples: dict,
    art: object,
    save_path: str,
    Ncurve: int = 100,
) -> None:
    """Plot retrieved temperature-pressure profile."""
    fig, ax = plt.subplots(figsize=(6, 7))

    if "T0" in posterior_samples:
        T0_samples = np.asarray(posterior_samples["T0"])
        for i in np.random.choice(len(T0_samples), min(Ncurve, len(T0_samples)), replace=False):
            Tarr = T0_samples[i] * np.ones_like(art.pressure)
            ax.plot(Tarr, art.pressure, "C0-", alpha=0.05)

        T0_median = np.median(T0_samples)
        Tarr_median = T0_median * np.ones_like(art.pressure)
        ax.plot(Tarr_median, art.pressure, "C0-", lw=2, label="Median (isothermal)")

    elif "T_btm" in posterior_samples:
        T_btm_samples = np.asarray(posterior_samples["T_btm"])
        T_top_samples = np.asarray(posterior_samples["T_top"])

        for i in np.random.choice(len(T_btm_samples), min(Ncurve, len(T_btm_samples)), replace=False):
            log_p = np.log10(art.pressure)
            log_p_btm = np.log10(art.pressure[-1])
            log_p_top = np.log10(art.pressure[0])
            Tarr = T_top_samples[i] + (T_btm_samples[i] - T_top_samples[i]) * \
                   (log_p - log_p_top) / (log_p_btm - log_p_top)
            ax.plot(Tarr, art.pressure, "C0-", alpha=0.05)

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


def _corner_data(
    sample_dict: dict,
    variables: list[str],
) -> tuple[np.ndarray | None, list[str] | None]:
    """Extract corner plot data from sample dictionary."""
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


def plot_corner(
    hmc_samples: dict | None = None,
    svi_samples: dict | None = None,
    variables: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Create corner plot for parameter distributions."""
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
            data, labels=labels, color=color, bins=40, smooth=1.0,
            fig=fig, show_titles=True, **extra_kwargs,
        )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Corner plot saved to {save_path}")


def create_transmission_plots(
    losses: np.ndarray,
    wav_obs: np.ndarray,
    rp_mean: np.ndarray,
    rp_std: np.ndarray,
    predictions: dict,
    svi_mu: np.ndarray | None,
    posterior_sample: dict,
    svi_samples: dict | None,
    opa_mols: dict,
    art: object,
    output_dir: str,
) -> None:
    """Generate all transmission retrieval diagnostic plots."""
    print("Generating diagnostic plots...")

    plot_svi_loss(losses, os.path.join(output_dir, "svi_loss.png"))

    plot_transmission_spectrum(
        wav_obs, rp_mean, rp_std, predictions["rp"], svi_mu,
        os.path.join(output_dir, "transmission_spectrum.png")
    )

    plot_temperature_profile(
        posterior_sample, art, os.path.join(output_dir, "temperature_profile.png")
    )

    corner_vars = ["Radius_btm", "T0", "logP_cloud", "RV"]
    corner_vars += [f"logVMR_{mol}" for mol in list(opa_mols.keys())]

    plot_corner(
        svi_samples=svi_samples, variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_svi.png")
    )

    plot_corner(
        hmc_samples=posterior_sample, variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_hmc.png")
    )

    plot_corner(
        hmc_samples=posterior_sample, svi_samples=svi_samples, variables=corner_vars,
        save_path=os.path.join(output_dir, "corner_plot_overlay.png")
    )

    print("All plots generated successfully.")
