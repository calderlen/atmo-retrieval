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
    """Plot parameter evolution across orbital phase with uncertainty bands.
    
    Works for per-exposure and smooth phase models. Shows how the parameter
    (typically dRV) varies through the transit.
    
    Args:
        posteriors: Dict with 'samples' containing posterior samples
        phase: Orbital phase array (0 = mid-transit)
        param: Parameter to plot (default: 'dRV')
        save_path: Path to save figure (if None, displays)
        show_exposures: Show individual exposure estimates as points
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    samples = posteriors.get("samples", posteriors)
    
    # Check if we have per-exposure samples
    if param in samples:
        param_samples = np.asarray(samples[param])
        
        if param_samples.ndim == 1:
            # Single value (shared mode) - plot as horizontal band
            mean = np.mean(param_samples)
            std = np.std(param_samples)
            q16, q84 = np.percentile(param_samples, [16, 84])
            
            ax.axhline(mean, color='C0', lw=2, label=f'{param} = {mean:.2f} km/s')
            ax.axhspan(q16, q84, alpha=0.3, color='C0', label='68% CI')
            
        elif param_samples.ndim == 2:
            # Per-exposure or interpolated values
            n_samples, n_exp = param_samples.shape
            
            # Compute statistics per exposure
            means = np.mean(param_samples, axis=0)
            stds = np.std(param_samples, axis=0)
            q16 = np.percentile(param_samples, 16, axis=0)
            q84 = np.percentile(param_samples, 84, axis=0)
            
            # Sort by phase for plotting
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
    
    # Check for linear/quadratic model parameters
    if f"{param}_0" in samples or f"{param}_a" in samples:
        # Generate smooth curve
        phase_fine = np.linspace(phase.min(), phase.max(), 200)
        
        if f"{param}_0" in samples and f"{param}_slope" in samples:
            # Linear model
            p0 = np.asarray(samples[f"{param}_0"])
            slope = np.asarray(samples[f"{param}_slope"])
            
            curves = p0[:, None] + slope[:, None] * phase_fine[None, :]
            mean_curve = np.mean(curves, axis=0)
            q16_curve = np.percentile(curves, 16, axis=0)
            q84_curve = np.percentile(curves, 84, axis=0)
            
            ax.fill_between(phase_fine, q16_curve, q84_curve,
                          alpha=0.3, color='C0', label='68% CI')
            ax.plot(phase_fine, mean_curve, 'C0-', lw=2, label='Linear trend')
            
        elif f"{param}_a" in samples:
            # Quadratic model
            pa = np.asarray(samples[f"{param}_a"])
            pb = np.asarray(samples[f"{param}_b"])
            pc = np.asarray(samples[f"{param}_c"])
            
            curves = pa[:, None] + pb[:, None] * phase_fine[None, :] + pc[:, None] * phase_fine[None, :]**2
            mean_curve = np.mean(curves, axis=0)
            q16_curve = np.percentile(curves, 16, axis=0)
            q84_curve = np.percentile(curves, 84, axis=0)
            
            ax.fill_between(phase_fine, q16_curve, q84_curve,
                          alpha=0.3, color='C0', label='68% CI')
            ax.plot(phase_fine, mean_curve, 'C0-', lw=2, label='Quadratic trend')
    
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
    """Create overlaid corner plot comparing posteriors across phase bins.
    
    Args:
        bin_posteriors: Dict mapping bin_name -> posterior samples dict
        params: List of parameters to include in corner plot
        save_path: Path to save figure
    """
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
        from matplotlib.lines import Line2D
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
    """Plot heatmap of peak cross-correlations between all species pairs.
    
    Args:
        matrix: Square matrix of peak correlations
        species_names: List of species names (same order as matrix)
        save_path: Path to save figure
        threshold: Correlation threshold to highlight (default 0.3)
    """
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
    """Plot cross-correlation function between two species templates.
    
    Args:
        ccf: CCF array
        velocities: Velocity array (km/s)
        species_a: First species name
        species_b: Second species name
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(velocities, ccf, 'C0-', lw=1.5)
    ax.axhline(0, color='k', ls='--', alpha=0.5, lw=1)
    ax.axvline(0, color='r', ls=':', alpha=0.7, lw=1, label='v = 0')
    
    # Mark peak
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

def compute_contribution_function(
    dtau: np.ndarray,
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
) -> np.ndarray:
    """Compute contribution function from optical depth.
    
    The contribution function shows which atmospheric layers contribute
    to the observed spectrum at each wavelength. For emission spectroscopy,
    it represents the source function weighted by the optical depth gradient.
    
    Args:
        dtau: Optical depth matrix, shape (n_layers, n_wavelength)
        Tarr: Temperature array per layer
        pressure: Pressure grid
        dParr: Pressure differential per layer
        
    Returns:
        cf: Contribution function, shape (n_layers, n_wavelength)
    """
    # Cumulative optical depth from top of atmosphere
    tau_cumsum = np.cumsum(dtau, axis=0)
    
    # Transmission: exp(-tau)
    transmission = np.exp(-tau_cumsum)
    
    # Contribution function: dtau/dP * exp(-tau)
    # This represents the fractional contribution of each layer
    cf = dtau * transmission
    
    # Normalize per wavelength to show fractional contribution
    cf_sum = np.sum(cf, axis=0, keepdims=True)
    cf_sum = np.where(cf_sum > 0, cf_sum, 1.0)  # Avoid division by zero
    cf_normalized = cf / cf_sum
    
    return cf_normalized


def plot_contribution_function(
    nu_grid: np.ndarray,
    dtau: np.ndarray,
    Tarr: np.ndarray,
    pressure: np.ndarray,
    dParr: np.ndarray,
    save_path: str | None = None,
    wavelength_unit: str = "AA",
    cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    use_exojax_plotcf: bool = False,
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
        save_path: Path to save figure (if None, displays)
        wavelength_unit: "AA" for Angstroms, "nm" for nanometers, "um" for microns
        cmap: Colormap name
        title: Plot title (auto-generated if None)
        figsize: Figure size
        use_exojax_plotcf: If True, use ExoJAX's built-in plotcf
        
    Returns:
        cf: Computed contribution function array
    """
    # Try to use ExoJAX's built-in contribution function if requested
    if use_exojax_plotcf:
        try:
            from exojax.plot.atmplot import plotcf
            cf = plotcf(nu_grid, dtau, Tarr, pressure, dParr)
            print("Used ExoJAX plotcf for contribution function")
            return cf
        except ImportError:
            print("ExoJAX plotcf not available, using custom implementation")
    
    # Compute contribution function
    cf = compute_contribution_function(dtau, Tarr, pressure, dParr)
    
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
        cf = compute_contribution_function(dtau, Tarr, pressure, dParr)
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
    save_path: str | None = None,
    wavelength_unit: str = "AA",
    species_to_show: list[str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Create combined contribution function plot with total and per-species panels.
    
    Similar to Figure 4 in Bonidie et al. 2026, showing the total contribution
    function alongside individual species contributions.
    
    Args:
        nu_grid: Wavenumber grid
        dtau: Total optical depth matrix
        dtau_per_species: Dict of per-species dtau
        Tarr: Temperature profile
        pressure: Pressure grid
        dParr: Pressure differentials
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
        dtau_per_species = {k: v for k, v in dtau_per_species.items() if k in species_to_show}
    
    n_species = len(dtau_per_species)
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    
    # Main panel for combined + species subpanels
    gs = fig.add_gridspec(2, max(2, (n_species + 1) // 2 + 1), height_ratios=[2, 1])
    
    # Top panel: Total contribution function
    ax_total = fig.add_subplot(gs[0, :])
    cf_total = compute_contribution_function(dtau, Tarr, pressure, dParr)
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
        
        cf_sp = compute_contribution_function(sp_dtau, Tarr, pressure, dParr)
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