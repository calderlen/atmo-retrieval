"""Orchestrates high-resolution spectroscopic retrieval pipeline."""

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from exojax.rt import ArtTransPure, ArtEmisPure

import config
from load import load_observed_spectrum, ResolutionInterpolator
from grid_setup import setup_wavenumber_grid, setup_spectral_operators
from opacity_setup import setup_cia_opacities, load_molecular_opacities, load_atomic_opacities
from model import create_retrieval_model, PhaseMode, compute_atmospheric_state_from_posterior
from inference import run_svi, run_mcmc, generate_predictions
from plot import (
    create_transmission_plots,
    plot_contribution_function,
    plot_contribution_per_species,
    plot_contribution_combined,
)


def load_timeseries_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time-series spectroscopic data.

    Expected files in data_dir:
        - wavelength.npy: (n_wavelengths,) wavelength array in Angstroms
        - data.npy: (n_exposures, n_wavelengths) flux time-series
        - sigma.npy: (n_exposures, n_wavelengths) uncertainties
        - phase.npy: (n_exposures,) orbital phase for each exposure
        
    Returns:
        (wavelength, data, sigma, phase)
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    
    wavelength = np.load(data_dir / "wavelength.npy")
    data = np.load(data_dir / "data.npy")
    sigma = np.load(data_dir / "sigma.npy")
    phase = np.load(data_dir / "phase.npy")
    
    return wavelength, data, sigma, phase


def run_retrieval(
    mode: str = "transmission",
    skip_svi: bool = False,
    svi_only: bool = False,
    no_plots: bool = False,
    temperature_profile: str = "guillot",
    phase_mode: PhaseMode = "shared",
    check_aliasing: bool = False,
    compute_contribution: bool = True,
    seed: int = 42,
) -> None:
    """Run atmospheric retrieval.
    
    Args:
        mode: "transmission" or "emission"
        skip_svi: Skip SVI warm-up, go straight to MCMC
        svi_only: Run only SVI, skip MCMC
        no_plots: Skip diagnostic plots
        temperature_profile: T-P profile type (guillot, isothermal, madhu_seager, free)
        phase_mode: How to model phase-dependent velocity offset:
            - "shared": Single dRV for all exposures (default)
            - "per_exposure": Independent dRV for each exposure
            - "hierarchical": dRV[i] ~ Normal(dRV_mean, dRV_scatter)
            - "linear": dRV = dRV_0 + dRV_slope * phase
            - "quadratic": dRV = a + b*phase + c*phase^2
        check_aliasing: Run species aliasing diagnostics before retrieval
        compute_contribution: Compute and plot contribution function after MCMC
        seed: Random seed
    """
    # Create timestamped output directory
    base_dir = config.DIR_SAVE or config.get_output_dir()
    output_dir = config.create_timestamped_dir(base_dir)
    print(f"\nOutput directory: {output_dir}")

    # Save run configuration
    config.save_run_config(
        output_dir=output_dir,
        mode=mode,
        temperature_profile=temperature_profile,
        skip_svi=skip_svi,
        svi_only=svi_only,
        seed=seed,
    )

    # Get planet parameters
    params = config.get_params()
    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")
    
    print("\n[1/7] Loading time-series data...")
    data_paths = config.TRANSMISSION_DATA if mode == "transmission" else config.EMISSION_DATA

    try:
        data_dir = config.get_data_dir()
        wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
        print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
        print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
    except FileNotFoundError:
        print("  Warning: time-series not found, loading single spectrum...")
        wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
            data_paths["wavelength"],
            data_paths["spectrum"],
            data_paths["uncertainty"],
        )
        # Create mock time-series (single exposure at phase=0.5)
        data = spectrum[np.newaxis, :]
        sigma = uncertainty[np.newaxis, :]
        phase = np.array([0.5])
        print(f"  Loaded single spectrum with {len(wav_obs)} points")
    
    print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} Angstroms")

    # Convert to wavenumber
    from exojax.utils.grids import wav2nu
    inst_nus = wav2nu(wav_obs, "AA")

    # Setup instrumental resolution
    print("\n[2/7] Setting up instrumental resolution...")
    Rinst = config.get_resolution()
    print(f"  Instrument resolving power: R = {Rinst:.0f}")

    # Setup wavenumber grid
    print("\n[3/7] Building wavenumber grid...")
    wav_min, wav_max = config.get_wavelength_range()
    nu_grid, wav_grid, res_high = setup_wavenumber_grid(
        wav_min - config.WAV_MIN_OFFSET,
        wav_max + config.WAV_MAX_OFFSET,
        config.N_SPECTRAL_POINTS,
        unit="AA",
    )

    sop_rot, sop_inst, _ = setup_spectral_operators(
        nu_grid, Rinst, vsini_max=150.0, vrmax=500.0
    )
    print("  Spectral operators initialized")

    # Setup atmospheric RT
    print("\n[4/7] Initializing atmospheric RT...")
    if mode == "transmission":
        art = ArtTransPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=config.NLAYER,
        )
    else:
        art = ArtEmisPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=config.NLAYER,
        )
    art.change_temperature_range(config.TLOW, config.THIGH)
    print(f"  {config.NLAYER} atmospheric layers")
    print(f"  Pressure range: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature range: {config.TLOW:.0f} - {config.THIGH:.0f} K")

    # Load opacities
    print("\n[5/7] Loading opacities...")

    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    print(f"  Loaded {len(opa_cias)} CIA sources")

    opa_mols, molmass_arr = load_molecular_opacities(
        config.MOLPATH_HITEMP,
        config.MOLPATH_EXOMOL,
        nu_grid,
        config.OPA_LOAD,
        config.NDIV,
        config.DIFFMODE,
        config.TLOW,
        config.THIGH,
    )
    print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

    # Load atomic opacities (optional)
    opa_atoms, atommass_arr = load_atomic_opacities(
        config.ATOMIC_SPECIES,
        nu_grid,
        config.OPA_LOAD,
        config.NDIV,
        config.DIFFMODE,
        config.TLOW,
        config.THIGH,
    )
    if opa_atoms:
        print(f"  Loaded {len(opa_atoms)} atomic species: {list(opa_atoms.keys())}")

    # Run aliasing diagnostics if requested
    if check_aliasing:
        print("\n  Running species aliasing diagnostics...")
        from aliasing import check_aliasing_with_fe, generate_aliasing_report
        import os
        
        # Build templates from opacity objects
        # For now, just print a warning - full implementation would generate 
        # model spectra for each species
        aliasing_dir = os.path.join(output_dir, "aliasing")
        os.makedirs(aliasing_dir, exist_ok=True)
        
        all_species = list(opa_mols.keys()) + list(opa_atoms.keys())
        print(f"  Species to check: {', '.join(all_species)}")
        print(f"  (Full aliasing analysis requires template generation - see aliasing.py)")
        print(f"  Aliasing directory: {aliasing_dir}")

    print(f"\n[6/7] Building {mode} forward model ({temperature_profile} T-P)...")
    
    # Convert params to format expected by create_retrieval_model
    model_params = {
        "Kp": params.get("Kp", 150.0),
        "Kp_err": params.get("Kp_err", 20.0),
        "RV_abs": params.get("RV_abs", 0.0),
        "RV_abs_err": params.get("RV_abs_err", 5.0),
        "R_p": params["R_p"].nominal_value if hasattr(params["R_p"], "nominal_value") else params["R_p"],
        "R_p_err": params["R_p"].std_dev if hasattr(params["R_p"], "std_dev") else 0.1,
        "M_p": params["M_p"].nominal_value if hasattr(params["M_p"], "nominal_value") else params["M_p"],
        "M_p_err": params["M_p"].std_dev if hasattr(params["M_p"], "std_dev") else 0.1,
        "R_star": params["R_star"].nominal_value if hasattr(params["R_star"], "nominal_value") else params["R_star"],
        "R_star_err": params["R_star"].std_dev if hasattr(params["R_star"], "std_dev") else 0.1,
        "T_star": params.get("T_star", 6000.0),
        "T_eq": params.get("T_eq"),
        "period": params["period"].nominal_value if hasattr(params["period"], "nominal_value") else params["period"],
    }
    
    model_c = create_retrieval_model(
        mode=mode,
        params=model_params,
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,  # Pass {} if empty
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=Rinst,
        inst_nus=inst_nus,
        temperature_profile=temperature_profile,
        Tlow=config.TLOW,
        Thigh=config.THIGH,
        phase_mode=phase_mode,
    )
    print(f"  Model created (phase_mode={phase_mode})")

    # Convert data to JAX arrays
    data_jnp = jnp.array(data)
    sigma_jnp = jnp.array(sigma)
    phase_jnp = jnp.array(phase)

    # Run inference
    print("\n[7/7] Running Bayesian inference...")
    rng_key = random.PRNGKey(seed)

    if not skip_svi:
        print(f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}")
        print("  (SVI not yet implemented, skipping to MCMC)")

    if svi_only:
        print("\n  SVI-only mode not yet implemented")
        print(f"  Configuration saved to: {output_dir}/")
        return

    # Run MCMC directly
    print(f"\n  Running HMC-NUTS sampling...")
    print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
    print(f"  Chains: {config.MCMC_NUM_CHAINS}")
    
    from numpyro.infer import MCMC, NUTS
    
    kernel = NUTS(model_c, max_tree_depth=config.MCMC_MAX_TREE_DEPTH)
    mcmc = MCMC(
        kernel, 
        num_warmup=config.MCMC_NUM_WARMUP, 
        num_samples=config.MCMC_NUM_SAMPLES, 
        num_chains=config.MCMC_NUM_CHAINS
    )
    
    rng_key, rng_key_ = random.split(rng_key)
    mcmc.run(rng_key_, data=data_jnp, sigma=sigma_jnp, phase=phase_jnp)
    
    mcmc.print_summary()
    
    # Save results
    import os
    from contextlib import redirect_stdout
    
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()
    
    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)
    
    # Compute contribution function from posterior
    if compute_contribution:
        print("\n  Computing contribution function from posterior...")
        
        try:
            # Convert posterior samples from JAX to numpy
            posterior_np = {k: np.array(v) for k, v in posterior_sample.items()}
            
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_np,
                art=art,
                opa_mols=opa_mols,
                opa_atoms=opa_atoms,
                opa_cias=opa_cias,
                nu_grid=nu_grid,
                temperature_profile=temperature_profile,
                use_median=True,
            )
            
            # Save atmospheric state
            np.savez(
                os.path.join(output_dir, "atmospheric_state.npz"),
                dtau=np.array(atmo_state['dtau']),
                Tarr=np.array(atmo_state['Tarr']),
                pressure=np.array(atmo_state['pressure']),
                dParr=np.array(atmo_state['dParr']),
                mmw=np.array(atmo_state['mmw']),
                vmrH2=np.array(atmo_state['vmrH2']),
                vmrHe=np.array(atmo_state['vmrHe']),
            )
            
            # Plot contribution function
            if not no_plots:
                print("  Plotting contribution function...")
                
                # Total contribution function
                cf = plot_contribution_function(
                    nu_grid=np.array(nu_grid),
                    dtau=np.array(atmo_state['dtau']),
                    Tarr=np.array(atmo_state['Tarr']),
                    pressure=np.array(atmo_state['pressure']),
                    dParr=np.array(atmo_state['dParr']),
                    save_path=os.path.join(output_dir, "contribution_function.pdf"),
                    wavelength_unit="AA",
                    title=f"{config.PLANET} Contribution Function ({mode})",
                )
                
                # Per-species contribution functions (if available)
                if atmo_state['dtau_per_species']:
                    dtau_per_species_np = {
                        k: np.array(v) for k, v in atmo_state['dtau_per_species'].items()
                    }
                    
                    plot_contribution_per_species(
                        nu_grid=np.array(nu_grid),
                        dtau_per_species=dtau_per_species_np,
                        Tarr=np.array(atmo_state['Tarr']),
                        pressure=np.array(atmo_state['pressure']),
                        dParr=np.array(atmo_state['dParr']),
                        save_path=os.path.join(output_dir, "contribution_per_species.pdf"),
                        wavelength_unit="AA",
                    )
                    
                    # Combined plot
                    plot_contribution_combined(
                        nu_grid=np.array(nu_grid),
                        dtau=np.array(atmo_state['dtau']),
                        dtau_per_species=dtau_per_species_np,
                        Tarr=np.array(atmo_state['Tarr']),
                        pressure=np.array(atmo_state['pressure']),
                        dParr=np.array(atmo_state['dParr']),
                        save_path=os.path.join(output_dir, "contribution_combined.pdf"),
                        wavelength_unit="AA",
                    )
                
                print(f"  Contribution function plots saved to {output_dir}/")
            
        except Exception as e:
            print(f"  Warning: Could not compute contribution function: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    print("Running with default settings.")
    print("For more options, use: python __main__.py --help\n")

    if config.RETRIEVAL_MODE in ("transmission", "emission"):
        run_retrieval(mode=config.RETRIEVAL_MODE)
    else:
        raise ValueError(f"Unknown retrieval mode: {config.RETRIEVAL_MODE}")
