import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

from physics.model import reconstruct_temperature_profile


_CORNER_LOG10_BASES = frozenset({"kappa_ir_cgs", "gamma"})
_HC_OVER_K_CM = 1.438776877


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _replace_basename(name: str, new_base: str) -> str:
    if "/" not in name:
        return new_base
    prefix, _ = name.rsplit("/", 1)
    return f"{prefix}/{new_base}"


def _augment_corner_samples(sample_dict: dict | None) -> dict | None:
    if sample_dict is None:
        return None

    augmented = dict(sample_dict)
    for name, values in sample_dict.items():
        base = _basename(name)
        if base not in _CORNER_LOG10_BASES:
            continue

        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            continue

        augmented[_replace_basename(name, f"log10_{base}")] = np.log10(arr)

    return augmented


def _corner_label(name: str) -> str:
    base = _basename(name)
    if base.startswith("log10_"):
        label = f"log10({base[6:]})"
    else:
        label = base

    if "/" not in name:
        return label

    prefix, _ = name.rsplit("/", 1)
    return f"{prefix}/{label}"


def plot_svi_loss(loss_values: np.ndarray, save_path: str) -> None:
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


def _bin_observed_spectrum(
    wavelength: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    max_bins: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    errors = np.asarray(errors, dtype=float).reshape(-1)

    if not (wavelength.size == values.size == errors.size):
        raise ValueError(
            "wavelength, values, and errors must have the same flattened length "
            f"(got {wavelength.size}, {values.size}, {errors.size})."
        )

    valid = np.isfinite(wavelength) & np.isfinite(values) & np.isfinite(errors)
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])

    wavelength = wavelength[valid]
    values = values[valid]
    errors = errors[valid]

    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    values = values[sort_idx]
    errors = errors[sort_idx]

    n_bins = min(max_bins, wavelength.size)
    index_bins = np.array_split(np.arange(wavelength.size), n_bins)

    binned_wavelength = []
    binned_values = []
    binned_errors = []

    for indices in index_bins:
        if indices.size == 0:
            continue

        wave_bin = wavelength[indices]
        value_bin = values[indices]
        error_bin = errors[indices]
        positive_error = error_bin > 0.0

        if np.any(positive_error):
            weights = 1.0 / np.square(error_bin[positive_error])
            binned_wavelength.append(float(np.average(wave_bin[positive_error], weights=weights)))
            binned_values.append(float(np.average(value_bin[positive_error], weights=weights)))
            binned_errors.append(float(np.sqrt(1.0 / np.sum(weights))))
        else:
            binned_wavelength.append(float(np.mean(wave_bin)))
            binned_values.append(float(np.mean(value_bin)))
            if value_bin.size > 1:
                binned_errors.append(float(np.std(value_bin) / np.sqrt(value_bin.size)))
            else:
                binned_errors.append(np.nan)

    return (
        np.asarray(binned_wavelength),
        np.asarray(binned_values),
        np.asarray(binned_errors),
    )


def plot_transmission_spectrum(
    wavelength_nm: np.ndarray,
    rp_obs: np.ndarray,
    rp_err: np.ndarray,
    rp_hmc: np.ndarray,
    rp_svi: np.ndarray,
    save_path: str,
    rp_pre_sysrem: np.ndarray | None = None,
    rp_pre_sysrem_err: np.ndarray | None = None,
) -> None:
    rp_hmc_np = np.asarray(rp_hmc)
    mean = rp_hmc_np.mean(axis=0)
    std = rp_hmc_np.std(axis=0)
    rp_svi_np = np.asarray(rp_svi)

    wavelength_np = np.asarray(wavelength_nm, dtype=float)
    sort_idx = np.argsort(wavelength_np)
    wavelength_sorted = wavelength_np[sort_idx]
    obs_sorted = np.asarray(rp_obs, dtype=float)[sort_idx]
    err_sorted = np.asarray(rp_err, dtype=float)[sort_idx]
    mean_sorted = np.asarray(mean, dtype=float)[sort_idx]
    std_sorted = np.asarray(std, dtype=float)[sort_idx]
    svi_sorted = rp_svi_np[sort_idx]

    bin_wavelength, bin_obs, bin_err = _bin_observed_spectrum(
        wavelength_sorted,
        obs_sorted,
        err_sorted,
    )
    if bin_wavelength.size:
        model_valid = np.isfinite(wavelength_sorted) & np.isfinite(mean_sorted)
        if np.count_nonzero(model_valid) >= 2:
            bin_model = np.interp(bin_wavelength, wavelength_sorted[model_valid], mean_sorted[model_valid])
        else:
            bin_model = np.full_like(bin_obs, np.nan)
        bin_residual = bin_obs - bin_model
    else:
        bin_residual = np.array([])

    has_pre_sysrem = rp_pre_sysrem is not None and rp_pre_sysrem_err is not None
    if has_pre_sysrem:
        pre_sorted = np.asarray(rp_pre_sysrem, dtype=float)[sort_idx]
        pre_err_sorted = np.asarray(rp_pre_sysrem_err, dtype=float)[sort_idx]
        pre_bin_wavelength, pre_bin_obs, pre_bin_err = _bin_observed_spectrum(
            wavelength_sorted,
            pre_sorted,
            pre_err_sorted,
        )
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 3, 1]},
        )
        ax_pre, ax, ax_resid = axes

        ax_pre.plot(
            wavelength_sorted,
            pre_sorted,
            ".",
            ms=1.0,
            color="k",
            alpha=0.08,
            label="Pre-SYSREM raw",
            zorder=1,
        )
        if pre_bin_wavelength.size:
            ax_pre.errorbar(
                pre_bin_wavelength,
                pre_bin_obs,
                yerr=pre_bin_err,
                fmt="o",
                ms=2.5,
                color="k",
                ecolor="0.2",
                elinewidth=0.6,
                alpha=0.85,
                label="Pre-SYSREM binned",
                zorder=3,
            )
        ax_pre.set_ylabel(r"$R_p/R_\star$", fontsize=12)
        ax_pre.set_title("Before SYSREM / Systematics Correction", fontsize=12)
        ax_pre.legend(fontsize=9, loc="upper right")
        ax_pre.grid(True, alpha=0.3)
    else:
        fig, (ax, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

    ax.plot(
        wavelength_sorted,
        obs_sorted,
        ".",
        ms=1.0,
        color="k",
        alpha=0.08,
        label="Observed raw",
        zorder=1,
    )
    if bin_wavelength.size:
        ax.errorbar(
            bin_wavelength,
            bin_obs,
            yerr=bin_err,
            fmt="o",
            ms=2.5,
            color="k",
            ecolor="0.2",
            elinewidth=0.6,
            alpha=0.85,
            label="Observed binned",
            zorder=4,
        )
    ax.fill_between(
        wavelength_sorted,
        mean_sorted - std_sorted,
        mean_sorted + std_sorted,
        color="C0",
        alpha=0.25,
        label="HMC ±1σ",
        zorder=2,
    )
    ax.plot(wavelength_sorted, mean_sorted, color="C0", lw=1.7, label="HMC mean", zorder=5)
    ax.plot(wavelength_sorted, svi_sorted, color="C3", lw=1.5, ls="--", label="SVI median", zorder=6)
    ax.set_ylabel(r"$R_p/R_\star$", fontsize=12)
    if has_pre_sysrem:
        ax.set_title("After SYSREM / Model-Space Comparison", fontsize=12)
    else:
        ax.set_title("Transmission Spectrum", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    if bin_wavelength.size:
        ax_resid.errorbar(
            bin_wavelength,
            bin_residual,
            yerr=bin_err,
            fmt="o",
            ms=2.5,
            color="k",
            ecolor="0.2",
            elinewidth=0.6,
            alpha=0.85,
            zorder=3,
        )
    ax_resid.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=2)
    ax_resid.set_xlabel("Wavelength [nm]", fontsize=12)
    ax_resid.set_ylabel("Obs - HMC", fontsize=11)
    ax_resid.set_title("Residual: Corrected Observed - HMC Mean", fontsize=11)
    ax_resid.grid(True, alpha=0.3)
    fig.subplots_adjust(hspace=0.32 if has_pre_sysrem else 0.22)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    pt_profile: str = "guillot",
    sample_prefix: str | None = None,
    Tint_fixed: float = 100.0,
    Ncurve: int = 100,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 7))
    sample_sizes = []
    for values in posterior_samples.values():
        arr = np.asarray(values)
        if arr.ndim > 0:
            sample_sizes.append(arr.shape[0])
    if not sample_sizes:
        raise ValueError("posterior_samples does not contain any sample arrays.")

    n_samples = min(sample_sizes)
    draw_count = min(Ncurve, n_samples)
    draw_indices = np.random.choice(n_samples, draw_count, replace=False)

    for idx in draw_indices:
        sample_params = {}
        for key, values in posterior_samples.items():
            arr = np.asarray(values)
            sample_params[key] = arr if arr.ndim == 0 else arr[idx]
        Tarr = reconstruct_temperature_profile(
            sample_params,
            art,
            pt_profile=pt_profile,
            Tint_fixed=Tint_fixed,
            sample_prefix=sample_prefix,
        )
        ax.plot(np.asarray(Tarr), art.pressure, "C0-", alpha=0.05)

    median_params = {}
    for key, values in posterior_samples.items():
        arr = np.asarray(values)
        median_params[key] = arr if arr.ndim == 0 else np.median(arr, axis=0)
    Tarr_median = reconstruct_temperature_profile(
        median_params,
        art,
        pt_profile=pt_profile,
        Tint_fixed=Tint_fixed,
        sample_prefix=sample_prefix,
    )
    ax.plot(np.asarray(Tarr_median), art.pressure, "C0-", lw=2, label=f"Median ({pt_profile})")

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
    variables: list[str] | None,
) -> tuple[np.ndarray | None, list[str] | None]:
    if variables is None:
        variables = list(sample_dict.keys())

    cols = []
    labels = []
    available = []
    for var in variables:
        if var in sample_dict:
            available.append(var)
    for var in available:
        arr = np.asarray(sample_dict[var])
        arr = arr.reshape(arr.shape[0], -1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            base_label = _corner_label(var)
            labels.append(base_label if arr.shape[1] == 1 else f"{base_label}[{j}]")
    if not cols:
        return None, None
    return np.column_stack(cols), labels


def _is_corner_friendly(arr: np.ndarray, max_components: int = 6) -> bool:
    if arr.ndim == 0:
        return False
    if arr.shape[0] < 2:
        return False
    n_components = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
    return n_components <= max_components


def _default_corner_variables(sample_dict: dict) -> list[str]:
    if not sample_dict:
        return []

    priority = [
        "Kp", "dRV", "dRV_0", "dRV_slope",
        "Rp", "Mp", "Rstar",
        "T0", "T_bottom", "T_top", "Tirr", "log10_kappa_ir_cgs", "log10_gamma",
        "T_deep", "log_P_trans", "delta_P",
        "log_metallicity", "C_O_ratio",
    ]
    skip_names = {
        "dRV_mean", "dRV_std", "dRV_at_ingress", "dRV_at_egress",
    }

    corner_ready = {}
    for name, values in sample_dict.items():
        arr = np.asarray(values)
        if _is_corner_friendly(arr):
            corner_ready[name] = arr

    selected: list[str] = []

    for name in priority:
        for key in sorted(corner_ready):
            if _basename(key) == name and key not in selected:
                selected.append(key)

    for name in sorted(corner_ready):
        if _basename(name).startswith("logVMR_") and name not in selected:
            selected.append(name)

    for name in sorted(corner_ready):
        base = _basename(name)
        if base in _CORNER_LOG10_BASES:
            log_name = _replace_basename(name, f"log10_{base}")
            if log_name in corner_ready:
                continue
        if name in selected or base in skip_names or base.endswith("_kms"):
            continue
        selected.append(name)

    return selected[:16]


def _filter_corner_datasets(
    datasets: list[tuple[np.ndarray, str, dict]],
    labels: list[str],
) -> tuple[list[tuple[np.ndarray, str, dict]], list[str]]:
    keep = np.ones(len(labels), dtype=bool)
    dropped = []

    for i, label in enumerate(labels):
        for data, _color, _extra_kwargs in datasets:
            col = np.asarray(data[:, i], dtype=float)
            if col.size < 2 or np.any(~np.isfinite(col)) or np.min(col) == np.max(col):
                keep[i] = False
                dropped.append(label)
                break

    if dropped:
        print(
            "Skipping corner column(s) with non-finite values or no dynamic range: "
            + ", ".join(dropped)
        )

    if not np.any(keep):
        return [], []

    filtered = []
    for data, color, extra_kwargs in datasets:
        filtered.append((data[:, keep], color, extra_kwargs))

    return filtered, [label for label, use_col in zip(labels, keep) if use_col]


def plot_corner(
    hmc_samples: dict | None = None,
    svi_samples: dict | None = None,
    variables: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    datasets = []
    labels = None

    if hmc_samples is not None:
        hmc_data, labels_hmc = _corner_data(hmc_samples, variables)
        if hmc_data is not None:
            labels = labels_hmc
            datasets.append((hmc_data, "C0", {}))

    if svi_samples is not None:
        svi_data, labels_svi = _corner_data(svi_samples, variables)
        if labels is None:
            labels = labels_svi
        if svi_data is not None and labels_svi == labels:
            datasets.append((svi_data, "C3", {"hist_kwargs": {"linestyle": "--"}}))
        elif svi_data is not None:
            print("SVI/HMC corner labels do not match; skipping overlay dataset.")

    if not datasets or labels is None:
        print("No data for corner plot; skipping.")
        return

    datasets, labels = _filter_corner_datasets(datasets, labels)
    if not datasets or not labels:
        print("No corner columns with finite dynamic range; skipping.")
        return

    fig = None
    for data, color, extra_kwargs in datasets:
        fig = corner.corner(
            data, labels=labels, color=color, bins=40, smooth=1.0,
            fig=fig, show_titles=True, **extra_kwargs,
        )

    if fig is None:
        print("No corner figure was generated; skipping save.")
        return

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Corner plot saved to {save_path}")


def save_retrieval_corner_plots(
    output_dir: str,
    hmc_samples: dict | None = None,
    svi_samples: dict | None = None,
    variables: list[str] | None = None,
) -> None:
    hmc_corner_samples = _augment_corner_samples(hmc_samples)
    svi_corner_samples = _augment_corner_samples(svi_samples)

    if hmc_corner_samples is None and svi_corner_samples is None:
        print("No posterior samples available for corner plots; skipping.")
        return

    if variables is None:
        merged_samples = {}
        if hmc_corner_samples is not None:
            merged_samples.update(hmc_corner_samples)
        if svi_corner_samples is not None:
            for key, value in svi_corner_samples.items():
                merged_samples.setdefault(key, value)
        variables = _default_corner_variables(merged_samples)

    if not variables:
        print("No suitable variables for corner plots; skipping.")
        return

    if svi_corner_samples is not None:
        svi_vars = []
        for var in variables:
            if var in svi_corner_samples:
                svi_vars.append(var)
        if svi_vars:
            plot_corner(
                svi_samples=svi_corner_samples,
                variables=svi_vars,
                save_path=os.path.join(output_dir, "corner_plot_svi.png"),
            )
        else:
            print("No SVI variables available for corner_plot_svi.png; skipping.")

    if hmc_corner_samples is not None:
        hmc_vars = []
        for var in variables:
            if var in hmc_corner_samples:
                hmc_vars.append(var)
        if hmc_vars:
            plot_corner(
                hmc_samples=hmc_corner_samples,
                variables=hmc_vars,
                save_path=os.path.join(output_dir, "corner_plot_hmc.png"),
            )
        else:
            print("No HMC variables available for corner_plot_hmc.png; skipping.")

    if hmc_corner_samples is not None and svi_corner_samples is not None:
        overlay_vars = []
        for var in variables:
            if var in hmc_corner_samples and var in svi_corner_samples:
                overlay_vars.append(var)
        if overlay_vars:
            plot_corner(
                hmc_samples=hmc_corner_samples,
                svi_samples=svi_corner_samples,
                variables=overlay_vars,
                save_path=os.path.join(output_dir, "corner_plot_overlay.png"),
            )
        else:
            print("No overlapping HMC/SVI variables for corner_plot_overlay.png; skipping.")


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
    print("Generating diagnostic plots...")

    plot_svi_loss(losses, os.path.join(output_dir, "svi_loss.png"))

    plot_transmission_spectrum(
        wav_obs, rp_mean, rp_std, predictions["rp"], svi_mu,
        os.path.join(output_dir, "transmission_spectrum.png")
    )

    plot_temperature_profile(
        posterior_sample, art, os.path.join(output_dir, "temperature_profile.png")
    )

    save_retrieval_corner_plots(
        output_dir=output_dir,
        hmc_samples=posterior_sample,
        svi_samples=svi_samples,
    )


# ==============================================================================
# PHASE-RESOLVED ANALYSIS PLOTS
# ==============================================================================

def plot_phase_trace(
    posteriors: dict,
    phase: np.ndarray,
    param: str = "dRV",
    save_path: str | None = None,
    show_exposures: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    samples = posteriors.get("samples", posteriors)
    
    if param in samples:
        param_samples = np.asarray(samples[param])
        
        if param_samples.ndim == 1:
            mean = np.mean(param_samples)
            std = np.std(param_samples)
            q16, q84 = np.percentile(param_samples, [16, 84])
            
            ax.axhline(mean, color='C0', lw=2, label=f'{param} = {mean:.2f} km/s')
            ax.axhspan(q16, q84, alpha=0.3, color='C0', label='68% CI')
            
        elif param_samples.ndim == 2:
            n_samples, n_exp = param_samples.shape
            
            means = np.mean(param_samples, axis=0)
            stds = np.std(param_samples, axis=0)
            q16 = np.percentile(param_samples, 16, axis=0)
            q84 = np.percentile(param_samples, 84, axis=0)
            
            sort_idx = np.argsort(phase)
            phase_sorted = phase[sort_idx]
            means_sorted = means[sort_idx]
            q16_sorted = q16[sort_idx]
            q84_sorted = q84[sort_idx]
            
            ax.fill_between(phase_sorted, q16_sorted, q84_sorted, 
                          alpha=0.3, color='C0', label='68% CI')
            ax.plot(phase_sorted, means_sorted, 'C0-', lw=2, label='Mean')
            
            if show_exposures:
                ax.errorbar(phase, means, yerr=stds, fmt='o', color='C0', 
                          alpha=0.6, markersize=6, capsize=2)
    
    if f"{param}_0" in samples:
        phase_fine = np.linspace(phase.min(), phase.max(), 200)
        
        p0 = np.asarray(samples[f"{param}_0"])
        slope = np.asarray(samples[f"{param}_slope"])
        curves = p0[:, None] + slope[:, None] * phase_fine[None, :]
        mean_curve = np.mean(curves, axis=0)
        q16_curve = np.percentile(curves, 16, axis=0)
        q84_curve = np.percentile(curves, 84, axis=0)
        ax.fill_between(phase_fine, q16_curve, q84_curve,
                      alpha=0.3, color='C0', label='68% CI')
        ax.plot(phase_fine, mean_curve, 'C0-', lw=2, label='Linear trend')
    
    ax.axhline(0, color='k', ls='--', alpha=0.5, lw=1)
    ax.axvline(0, color='k', ls=':', alpha=0.5, lw=1, label='Mid-transit')
    
    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel(f'{param} [km/s]', fontsize=12)
    ax.set_title(f'{param} vs Orbital Phase', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Phase trace plot saved to {save_path}")
    else:
        plt.show()


def plot_phase_comparison(
    bin_posteriors: dict[str, dict],
    params: list[str],
    save_path: str | None = None,
) -> None:
    colors = {'T12': 'C0', 'T23': 'C1', 'T34': 'C2'}
    
    fig = None
    handles = []
    labels_legend = []
    
    for bin_name, posteriors in bin_posteriors.items():
        if posteriors is None or "error" in posteriors:
            continue
        
        samples = posteriors.get("samples", posteriors)
        data, labels = _corner_data(samples, params)
        
        if data is None:
            continue
        
        color = colors.get(bin_name, 'C3')
        
        fig = corner.corner(
            data, labels=labels, color=color, bins=30, smooth=1.0,
            fig=fig, show_titles=(bin_name == list(bin_posteriors.keys())[0]),
            plot_contours=True, fill_contours=False,
            hist_kwargs={"alpha": 0.5},
        )
        
        # For legend
        handles.append(Line2D([0], [0], color=color, lw=2))
        labels_legend.append(bin_name)
    
    if fig is not None:
        fig.legend(handles, labels_legend, loc='upper right', fontsize=12)
        
        if save_path:
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f"Phase comparison corner plot saved to {save_path}")
        else:
            plt.show()


def plot_aliasing_matrix(
    matrix: np.ndarray,
    species_names: list[str],
    save_path: str | None = None,
    threshold: float = 0.3,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Peak Correlation')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(species_names)))
    ax.set_yticks(np.arange(len(species_names)))
    ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(species_names, fontsize=9)
    
    # Add grid
    ax.set_xticks(np.arange(len(species_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(species_names) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Annotate cells with values above threshold
    for i in range(len(species_names)):
        for j in range(len(species_names)):
            if i != j and np.abs(matrix[i, j]) >= threshold:
                text_color = 'white' if np.abs(matrix[i, j]) > 0.7 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=8, color=text_color, fontweight='bold')
    
    ax.set_title(f'Species Aliasing Matrix\n(values shown where |r| >= {threshold})', fontsize=13)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Aliasing matrix plot saved to {save_path}")
    else:
        plt.show()


def plot_ccf_pair(
    ccf: np.ndarray,
    velocities: np.ndarray,
    species_a: str,
    species_b: str,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(velocities, ccf, 'C0-', lw=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.5, lw=1)
    ax.axvline(0, color='r', ls=':', alpha=0.7, lw=1, label='v = 0')
    
    peak_idx = np.argmax(np.abs(ccf))
    ax.scatter(velocities[peak_idx], ccf[peak_idx], s=100, c='r', marker='*',
              zorder=5, label=f'Peak: r = {ccf[peak_idx]:.3f}')
    
    ax.set_xlabel('Velocity [km/s]', fontsize=12)
    ax.set_ylabel('Cross-Correlation', fontsize=12)
    ax.set_title(f'{species_a} × {species_b} Template CCF', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"CCF pair plot saved to {save_path}")
    else:
        plt.show()


# ==============================================================================
# CONTRIBUTION FUNCTION PLOTS
# ==============================================================================

def _normalize_contribution(raw_contribution: np.ndarray) -> np.ndarray:
    contribution = np.nan_to_num(raw_contribution, nan=0.0, posinf=0.0, neginf=0.0)
    contribution = np.clip(contribution, 0.0, None)
    contribution_sum = np.sum(contribution, axis=0, keepdims=True)
    contribution_sum = np.where(contribution_sum > 0.0, contribution_sum, 1.0)
    return contribution / contribution_sum


def compute_contribution_function(
    nu_grid: np.ndarray,
    dtau: np.ndarray,
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Compute contribution function from optical depth.
    
    The contribution function shows which atmospheric layers contribute
    to the observed spectrum at each wavelength. For emission spectroscopy,
    it represents the source function weighted by the optical depth gradient.
    
    Args:
        nu_grid: Wavenumber grid (cm^-1)
        dtau: Optical depth matrix, shape (n_layers, n_wavelength)
        Tarr: Temperature array per layer
        pressure: Pressure grid
        dParr: Pressure differential per layer
        mode: Retrieval mode, either "transmission" or "emission"
        
    Returns:
        cf: Contribution function, shape (n_layers, n_wavelength)
    """
    mode = str(mode).lower().strip()
    if mode not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported contribution-function mode: {mode!r}")

    dtau = np.asarray(dtau, dtype=float)
    Tarr = np.asarray(Tarr, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    dParr = np.asarray(dParr, dtype=float)
    nu_grid = np.asarray(nu_grid, dtype=float)

    if dtau.ndim != 2:
        raise ValueError(f"dtau must be 2D, got shape {dtau.shape}.")
    if dtau.shape != (pressure.size, nu_grid.size):
        raise ValueError(
            "dtau shape must match (n_pressure, n_wavenumber); "
            f"got {dtau.shape}, pressure={pressure.size}, nu_grid={nu_grid.size}."
        )

    tau_cumsum = np.cumsum(dtau, axis=0)
    escape = np.exp(-np.clip(tau_cumsum, 0.0, 700.0))

    if mode == "transmission":
        return _normalize_contribution(dtau * escape)

    pressure_weight = pressure[:, None] / np.clip(dParr[:, None], 1.0e-300, None)
    temperature = np.clip(Tarr[:, None], 1.0e-300, None)
    nu = np.clip(nu_grid[None, :], 1.0e-300, None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        source_weight = np.power(nu, 3) / np.expm1(_HC_OVER_K_CM * nu / temperature)
    return _normalize_contribution(escape * dtau * pressure_weight * source_weight)


def plot_contribution_function(
    nu_grid: np.ndarray,
    dtau: np.ndarray,
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
    mode: str,
    save_path: str | None = None,
    wavelength_unit: str = "AA",
    cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> np.ndarray:
    """Plot contribution function showing pressure vs wavelength.
    
    The contribution function shows which atmospheric pressure levels 
    contribute to the observed spectrum at each wavelength.
    
    Args:
        nu_grid: Wavenumber grid (cm^-1)
        dtau: Optical depth matrix, shape (n_layers, n_wavelength)
        Tarr: Temperature array per layer
        pressure: Pressure grid (bar)
        dParr: Pressure differential per layer
        mode: Retrieval mode, either "transmission" or "emission"
        save_path: Path to save figure (if None, displays)
        wavelength_unit: "AA" for Angstroms, "nm" for nanometers, "um" for microns
        cmap: Colormap name
        title: Plot title (auto-generated if None)
        figsize: Figure size
        
    Returns:
        cf: Computed contribution function array
    """
    # Compute contribution function
    cf = compute_contribution_function(nu_grid, dtau, Tarr, pressure, dParr, mode=mode)
    
    # Convert wavenumber to desired wavelength unit
    if wavelength_unit == "AA":
        wave = 1e8 / nu_grid  # cm^-1 to Angstroms
        wave_label = "Wavelength (Angstroms)"
    elif wavelength_unit == "nm":
        wave = 1e7 / nu_grid  # cm^-1 to nm
        wave_label = "Wavelength (nm)"
    elif wavelength_unit == "um":
        wave = 1e4 / nu_grid  # cm^-1 to microns
        wave_label = "Wavelength (um)"
    else:
        wave = nu_grid
        wave_label = "Wavenumber (cm$^{-1}$)"
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use pcolormesh for 2D plot
    # Note: wave may be in descending order (from wavenumber conversion)
    # So we need to handle this properly
    if wave[0] > wave[-1]:
        wave = wave[::-1]
        cf = cf[:, ::-1]
    
    im = ax.pcolormesh(wave, pressure, cf, cmap=cmap, shading='auto')
    
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel(wave_label, fontsize=12)
    ax.set_ylabel('Pressure (bar)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, label='Contribution')
    
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Atmospheric Contribution Function', fontsize=13)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Contribution function plot saved to {save_path}")
    else:
        plt.show()
    
    return cf


def plot_contribution_per_species(
    nu_grid: np.ndarray,
    dtau_per_species: dict[str, np.ndarray],
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
    mode: str,
    save_path: str | None = None,
    wavelength_unit: str = "AA",
    cmap: str = "viridis",
    figsize_per_panel: tuple[float, float] = (8, 4),
    ncols: int = 2,
) -> dict[str, np.ndarray]:
    """Plot contribution function for each species separately.
    
    Creates a multi-panel figure showing the contribution function
    for each species, useful for understanding which species dominate
    at different wavelengths and pressure levels.
    
    Args:
        nu_grid: Wavenumber grid (cm^-1)
        dtau_per_species: Dict mapping species name -> dtau array
        Tarr: Temperature array per layer
        pressure: Pressure grid (bar)
        dParr: Pressure differential per layer
        mode: Retrieval mode, either "transmission" or "emission"
        save_path: Path to save figure
        wavelength_unit: "AA", "nm", "um", or "cm-1"
        cmap: Colormap name
        figsize_per_panel: Size of each subplot
        ncols: Number of columns in subplot grid
        
    Returns:
        Dict mapping species name -> contribution function array
    """
    n_species = len(dtau_per_species)
    if n_species == 0:
        print("No species to plot")
        return {}
    
    nrows = (n_species + ncols - 1) // ncols
    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if n_species == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Convert wavelength
    if wavelength_unit == "AA":
        wave = 1e8 / nu_grid
        wave_label = "Wavelength (Angstroms)"
    elif wavelength_unit == "nm":
        wave = 1e7 / nu_grid
        wave_label = "Wavelength (nm)"
    elif wavelength_unit == "um":
        wave = 1e4 / nu_grid
        wave_label = "Wavelength (um)"
    else:
        wave = nu_grid
        wave_label = "Wavenumber (cm$^{-1}$)"
    
    # Handle wavelength ordering
    if wave[0] > wave[-1]:
        wave = wave[::-1]
        flip_wave = True
    else:
        flip_wave = False
    
    cf_dict = {}
    
    for idx, (species, dtau) in enumerate(dtau_per_species.items()):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Compute contribution function for this species
        cf = compute_contribution_function(nu_grid, dtau, Tarr, pressure, dParr, mode=mode)
        if flip_wave:
            cf = cf[:, ::-1]
        cf_dict[species] = cf
        
        im = ax.pcolormesh(wave, pressure, cf, cmap=cmap, shading='auto')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlabel(wave_label, fontsize=10)
        ax.set_ylabel('Pressure (bar)', fontsize=10)
        ax.set_title(species, fontsize=11)
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for idx in range(n_species, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    fig.suptitle('Contribution Function by Species', fontsize=13, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Per-species contribution plot saved to {save_path}")
    else:
        plt.show()
    
    return cf_dict


def plot_contribution_combined(
    nu_grid: np.ndarray,
    dtau: np.ndarray,
    dtau_per_species: dict[str, np.ndarray],
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
    mode: str,
    save_path: str | None = None,
    wavelength_unit: str = "AA",
    species_to_show: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Create combined contribution function plot with total and per-species panels.
    
    Shows the total contribution function alongside individual species
    contributions.
    
    Args:
        nu_grid: Wavenumber grid
        dtau: Total optical depth matrix
        dtau_per_species: Dict of per-species dtau
        Tarr: Temperature profile
        pressure: Pressure grid
        dParr: Pressure differentials
        mode: Retrieval mode, either "transmission" or "emission"
        save_path: Path to save figure
        wavelength_unit: Wavelength unit for x-axis
        species_to_show: List of species to show (None = all)
        figsize: Figure size
    """
    # Convert wavelength
    if wavelength_unit == "AA":
        wave = 1e8 / nu_grid
        wave_label = "Wavelength"
    else:
        wave = nu_grid
        wave_label = "Wavenumber"
    
    if wave[0] > wave[-1]:
        wave = wave[::-1]
        flip = True
    else:
        flip = False
    
    # Filter species if requested
    if species_to_show is not None:
        filtered_dtau_per_species = {}
        for k, v in dtau_per_species.items():
            if k in species_to_show:
                filtered_dtau_per_species[k] = v
        dtau_per_species = filtered_dtau_per_species
    
    n_species = len(dtau_per_species)
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    
    # Main panel for combined + species subpanels
    gs = fig.add_gridspec(2, max(2, (n_species + 1) // 2 + 1), height_ratios=[2, 1])
    
    # Top panel: Total contribution function
    ax_total = fig.add_subplot(gs[0, :])
    cf_total = compute_contribution_function(nu_grid, dtau, Tarr, pressure, dParr, mode=mode)
    if flip:
        cf_total = cf_total[:, ::-1]
    
    im = ax_total.pcolormesh(wave, pressure, cf_total, cmap='viridis', shading='auto')
    ax_total.set_yscale('log')
    ax_total.invert_yaxis()
    ax_total.set_xlabel(wave_label, fontsize=11)
    ax_total.set_ylabel('Pressure (bar)', fontsize=11)
    ax_total.set_title('Combined Contribution Function', fontsize=12)
    plt.colorbar(im, ax=ax_total, label='Contribution', shrink=0.8)
    
    # Bottom panels: Per-species
    for idx, (species, sp_dtau) in enumerate(dtau_per_species.items()):
        if idx >= gs.ncols:
            break
        ax = fig.add_subplot(gs[1, idx])
        
        cf_sp = compute_contribution_function(nu_grid, sp_dtau, Tarr, pressure, dParr, mode=mode)
        if flip:
            cf_sp = cf_sp[:, ::-1]
        
        im_sp = ax.pcolormesh(wave, pressure, cf_sp, cmap='viridis', shading='auto')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlabel(wave_label, fontsize=9)
        if idx == 0:
            ax.set_ylabel('P (bar)', fontsize=9)
        ax.set_title(species, fontsize=10)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Combined contribution plot saved to {save_path}")
    else:
        plt.show()
