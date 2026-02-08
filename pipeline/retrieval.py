import jax
from pathlib import Path
from jax import random
import jax.numpy as jnp
import numpy as np

from exojax.rt import ArtTransPure, ArtEmisPure

import config
from dataio.load import load_observed_spectrum, ResolutionInterpolator
from physics.grid_setup import setup_wavenumber_grid, setup_spectral_operators
from databases.opacity import setup_cia_opacities, load_molecular_opacities, load_atomic_opacities
from physics.model import create_retrieval_model, PhaseMode, compute_atmospheric_state_from_posterior
from pipeline.inference import run_svi, run_mcmc, generate_predictions
from plotting.plot import (
    create_transmission_plots,
    plot_contribution_function,
    plot_contribution_per_species,
    plot_contribution_combined,
)


def load_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pathlib import Path
    data_dir = Path(data_dir)

    required = ["wavelength.npy", "data.npy", "sigma.npy", "phase.npy"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        missing_fmt = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing time-series files in {data_dir}: {missing_fmt}"
        )

    wavelength = np.load(data_dir / "wavelength.npy")
    data = np.load(data_dir / "data.npy")
    sigma = np.load(data_dir / "sigma.npy")
    phase = np.load(data_dir / "phase.npy")
    
    return wavelength, data, sigma, phase


def _normalize_phase(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase)
    if phase.size == 0:
        return phase

    phase_min = float(np.nanmin(phase))
    phase_max = float(np.nanmax(phase))
    median = float(np.nanmedian(phase))

    if 0.0 <= phase_min and phase_max <= 1.0 and abs(median - 0.5) < 0.2:
        print("  Phase appears centered on 0.5; shifting to mid-transit at 0.0")
        phase = phase - 0.5

    if phase_min < -0.5 or phase_max > 0.5:
        phase = (phase + 0.5) % 1.0 - 0.5

    phase_min = float(np.nanmin(phase))
    phase_max = float(np.nanmax(phase))
    if phase_min < -0.5 or phase_max > 0.5:
        raise ValueError(
            "Phase values must fall in [-0.5, 0.5] after normalization. "
            f"Got range {phase_min:.4f} .. {phase_max:.4f}."
        )

    return phase


def _format_range(arr: np.ndarray) -> str:
    if arr is None:
        return "None"
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    return f"{np.nanmin(arr):.4g} .. {np.nanmax(arr):.4g}"


def _preflight_spectrum_checks(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    inst_nus: np.ndarray,
) -> None:
    errors = []

    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)
    phase = np.asarray(phase)
    inst_nus = np.asarray(inst_nus)

    if wav_obs.size == 0:
        errors.append("wavelength array is empty")
    if inst_nus.size == 0:
        errors.append("instrument wavenumber array is empty")
    if data.ndim not in (1, 2):
        errors.append(f"data has invalid ndim={data.ndim} (expected 1 or 2)")
    if sigma.shape != data.shape:
        errors.append(f"sigma shape {sigma.shape} does not match data shape {data.shape}")

    if data.ndim == 1:
        if data.size != wav_obs.size:
            errors.append(f"data length {data.size} != wavelength length {wav_obs.size}")
        expected_exposures = 1
    elif data.ndim == 2:
        if data.shape[1] != wav_obs.size:
            errors.append(
                f"data spectral axis {data.shape[1]} != wavelength length {wav_obs.size}"
            )
        expected_exposures = data.shape[0]
    else:
        expected_exposures = None

    if phase.ndim != 1:
        errors.append(f"phase has invalid ndim={phase.ndim} (expected 1)")
    elif expected_exposures is not None and phase.size != expected_exposures:
        errors.append(
            f"phase length {phase.size} != number of exposures {expected_exposures}"
        )

    if not np.all(np.isfinite(wav_obs)):
        errors.append("wavelength array contains non-finite values")
    if not np.all(np.isfinite(inst_nus)):
        errors.append("instrument wavenumber array contains non-finite values")
    if not np.all(np.isfinite(data)):
        errors.append("data array contains non-finite values")
    if not np.all(np.isfinite(sigma)):
        errors.append("sigma array contains non-finite values")

    if errors:
        print("\nPreflight check failed:")
        for err in errors:
            print(f"  - {err}")
        print("Diagnostics:")
        print(f"  wav_obs: shape={wav_obs.shape}, range={_format_range(wav_obs)}")
        print(f"  inst_nus: shape={inst_nus.shape}, range={_format_range(inst_nus)}")
        print(f"  data: shape={data.shape}")
        print(f"  sigma: shape={sigma.shape}")
        print(f"  phase: shape={phase.shape}, range={_format_range(phase)}")
        raise ValueError("Preflight check failed; see diagnostics above.")


def _preflight_grid_checks(inst_nus: np.ndarray, nu_grid: np.ndarray) -> None:
    inst_nus = np.asarray(inst_nus)
    nu_grid = np.asarray(nu_grid)

    if nu_grid.size == 0:
        raise ValueError("Preflight check failed: nu_grid is empty.")

    inst_min = np.nanmin(inst_nus) if inst_nus.size > 0 else np.nan
    inst_max = np.nanmax(inst_nus) if inst_nus.size > 0 else np.nan
    nu_min = np.nanmin(nu_grid)
    nu_max = np.nanmax(nu_grid)

    if inst_nus.size == 0:
        raise ValueError("Preflight check failed: inst_nus is empty.")

    if inst_min < nu_min or inst_max > nu_max:
        msg = (
            "Preflight check failed: instrument wavenumber grid is outside model grid.\n"
            f"  inst_nus range: {inst_min:.4g} .. {inst_max:.4g}\n"
            f"  nu_grid range: {nu_min:.4g} .. {nu_max:.4g}\n"
            "  Check wavelength range / offsets or data wavelength files."
        )
        raise ValueError(msg)


def _cia_header_range(path: str) -> tuple[float, float]:
    with open(path, "r") as f:
        header = f.readline().strip().split()
    if len(header) < 3:
        raise ValueError(f"CIA header in {path} does not contain min/max wavenumber fields.")
    nu_min = float(header[1])
    nu_max = float(header[2])
    return nu_min, nu_max


def _format_um(nu_cm: float) -> float:
    if nu_cm <= 0:
        return float("nan")
    return 1.0e4 / nu_cm


def _report_cia_coverage(cia_paths: dict[str, str], nu_grid: np.ndarray) -> None:
    nu_grid = np.asarray(nu_grid)
    if nu_grid.size == 0:
        return
    grid_min = float(np.nanmin(nu_grid))
    grid_max = float(np.nanmax(nu_grid))
    grid_um_min = _format_um(grid_max)
    grid_um_max = _format_um(grid_min)
    print(
        f"  CIA coverage check: model nu_grid {grid_min:.0f}-{grid_max:.0f} cm^-1 "
        f"({grid_um_min:.3f}-{grid_um_max:.3f} um)"
    )
    for name, path in cia_paths.items():
        rng = _cia_header_range(str(path))
        cia_min, cia_max = rng
        cia_um_min = _format_um(cia_max)
        cia_um_max = _format_um(cia_min)
        ov_min = max(cia_min, grid_min)
        ov_max = min(cia_max, grid_max)
        if ov_min >= ov_max:
            overlap = "none"
        else:
            ov_um_min = _format_um(ov_max)
            ov_um_max = _format_um(ov_min)
            overlap = (
                f"{ov_min:.0f}-{ov_max:.0f} cm^-1 "
                f"({ov_um_min:.3f}-{ov_um_max:.3f} um)"
            )
        print(
            f"    {name}: {cia_min:.0f}-{cia_max:.0f} cm^-1 "
            f"({cia_um_min:.3f}-{cia_um_max:.3f} um), overlap: {overlap}"
        )


def run_retrieval(
    mode: str = "transmission",
    epoch: str | None = None,
    data_dir: str | Path | None = None,
    data_format: str = "auto",
    skip_svi: bool = False,
    svi_only: bool = False,
    no_plots: bool = False,
    pt_profile: str = "guillot",
    phase_mode: PhaseMode = "global",
    check_aliasing: bool = False,
    compute_contribution: bool = True,
    seed: int = 42,
    wav_obs: np.ndarray | None = None,
    data: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    phase: np.ndarray | None = None,
) -> None:
    # Create timestamped output directory
    base_dir = config.DIR_SAVE or config.get_output_dir()
    output_dir = config.create_timestamped_dir(base_dir)
    print(f"\nOutput directory: {output_dir}")

    # Save run configuration
    config.save_run_config(
        output_dir=output_dir,
        mode=mode,
        pt_profile=pt_profile,
        skip_svi=skip_svi,
        svi_only=svi_only,
        seed=seed,
    )

    # Get planet parameters
    params = config.get_params()
    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")
    
    print("\n[1/7] Loading time-series data...")
    if epoch:
        print(f"  Using epoch: {epoch}")
    if any(val is not None for val in (wav_obs, data, sigma, phase)):
        if any(val is None for val in (wav_obs, data, sigma, phase)):
            raise ValueError("Must provide wav_obs, data, sigma, and phase together.")
        phase = _normalize_phase(phase)
        print(f"  Using provided data: {data.shape[0]} exposures x {data.shape[1]} wavelengths")
        print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
    else:
        if data_format not in {"auto", "timeseries", "spectrum"}:
            raise ValueError(f"Unknown data_format: {data_format}")

        resolved_data_dir = Path(data_dir) if data_dir is not None else config.get_data_dir(epoch=epoch)

        if data_dir is not None:
            suffix = "transmission" if mode == "transmission" else "emission"
            data_paths = {
                "wavelength": Path(data_dir) / f"wavelength_{suffix}.npy",
                "spectrum": Path(data_dir) / f"spectrum_{suffix}.npy",
                "uncertainty": Path(data_dir) / f"uncertainty_{suffix}.npy",
            }
        else:
            data_paths = (
                config.get_transmission_paths(epoch=epoch) if mode == "transmission"
                else config.get_emission_paths(epoch=epoch)
            )

        if data_format == "timeseries":
            wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
            phase = _normalize_phase(phase)
            print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
        elif data_format == "spectrum":
            wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
                str(data_paths["wavelength"]),
                str(data_paths["spectrum"]),
                str(data_paths["uncertainty"]),
            )
            data = spectrum[np.newaxis, :]
            sigma = uncertainty[np.newaxis, :]
            phase = np.array([0.0])
            print(f"  Loaded single spectrum with {len(wav_obs)} points")
        else:
            wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
            phase = _normalize_phase(phase)
            print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
    
    print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} Angstroms")

    # Convert to wavenumber
    from exojax.utils.grids import wav2nu
    inst_nus = wav2nu(wav_obs, "AA")
    # Ensure wavenumber grid and data are in ascending order
    if inst_nus.size > 1 and np.any(np.diff(inst_nus) <= 0):
        sort_idx = np.argsort(inst_nus)
        inst_nus = inst_nus[sort_idx]
        wav_obs = wav_obs[sort_idx]
        if data.ndim == 2:
            data = data[:, sort_idx]
            sigma = sigma[:, sort_idx]
        else:
            data = data[sort_idx]
            sigma = sigma[sort_idx]

    _preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

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
    _preflight_grid_checks(inst_nus, nu_grid)

    sop_rot, sop_inst, _ = setup_spectral_operators(
        nu_grid, Rinst, vsini_max=config.STITCH_VSINI_MAX, vrmax=config.STITCH_VRMAX
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
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    print(f"  {config.NLAYER} atmospheric layers")
    print(f"  Pressure range: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature range: {config.T_LOW:.0f} - {config.T_HIGH:.0f} K")

    # Load opacities
    print("\n[5/7] Loading opacities...")
    _report_cia_coverage(config.CIA_PATHS, nu_grid)

    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    n_cia = sum(1 for cia in opa_cias.values() if not getattr(cia, "_is_dummy", False))
    if n_cia == 0:
        print("  Loaded 0 CIA sources (no overlap with nu_grid)")
    else:
        print(f"  Loaded {n_cia} CIA sources")

    opa_mols, molmass_arr = load_molecular_opacities(
        config.MOLPATH_HITEMP,
        config.MOLPATH_EXOMOL,
        nu_grid,
        config.OPA_LOAD,
        config.NDIV,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

    # Load atomic opacities (optional, uses Kurucz with auto-download)
    opa_atoms, atommass_arr = load_atomic_opacities(
        config.ATOMIC_SPECIES,
        nu_grid,
        config.OPA_LOAD,
        config.NDIV,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    if opa_atoms:
        print(f"  Loaded {len(opa_atoms)} atomic species: {list(opa_atoms.keys())}")

    # Run aliasing diagnostics if requested
    if check_aliasing:
        print("\n  Running species aliasing diagnostics...")
        from plotting.aliasing import check_aliasing_with_fe, generate_aliasing_report
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

    print(f"\n[6/7] Building {mode} forward model ({pt_profile} P-T)...")
    
    # Convert params to format expected by create_retrieval_model
    model_params = {
        "Kp": params.get("Kp", config.DEFAULT_KP),
        "Kp_err": params.get("Kp_err", config.DEFAULT_KP_ERR),
        "RV_abs": params.get("RV_abs", config.DEFAULT_RV_ABS),
        "RV_abs_err": params.get("RV_abs_err", config.DEFAULT_RV_ABS_ERR),
        "R_p": params["R_p"].nominal_value if hasattr(params["R_p"], "nominal_value") else params["R_p"],
        "R_p_err": params["R_p"].std_dev if hasattr(params["R_p"], "std_dev") else config.DEFAULT_RP_ERR,
        "M_p": params["M_p"].nominal_value if hasattr(params["M_p"], "nominal_value") else params["M_p"],
        "M_p_err": params["M_p"].std_dev if hasattr(params["M_p"], "std_dev") else config.DEFAULT_MP_ERR,
        "R_star": params["R_star"].nominal_value if hasattr(params["R_star"], "nominal_value") else params["R_star"],
        "R_star_err": params["R_star"].std_dev if hasattr(params["R_star"], "std_dev") else config.DEFAULT_RSTAR_ERR,
        "T_star": params.get("T_star", config.DEFAULT_TSTAR),
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
        pt_profile=pt_profile,
        T_low=config.T_LOW,
        T_high=config.T_HIGH,
        phase_mode=phase_mode,
        stitch_inference=config.ENABLE_INFERENCE_STITCHING,
        stitch_chunk_points=config.INFERENCE_STITCH_CHUNK_POINTS,
        stitch_n_chunks=config.INFERENCE_STITCH_NCHUNKS,
        stitch_guard_kms=config.INFERENCE_STITCH_GUARD_KMS,
        stitch_guard_points=config.INFERENCE_STITCH_GUARD_POINTS,
        stitch_min_guard_points=config.INFERENCE_STITCH_MIN_GUARD_POINTS,
    )
    print(f"  Model created (phase_mode={phase_mode})")

    # Convert data to JAX arrays
    data_jnp = jnp.array(data)
    sigma_jnp = jnp.array(sigma)
    phase_jnp = jnp.array(phase)

    # Run inference
    print("\n[7/7] Running Bayesian inference...")
    rng_key = random.PRNGKey(seed)

    if svi_only and skip_svi:
        raise ValueError("Cannot set svi_only=True when skip_svi=True")

    from numpyro.infer import MCMC, NUTS, init_to_median

    init_strategy = init_to_median(num_samples=config.INIT_TO_MEDIAN_SAMPLES)
    if not skip_svi:
        print(f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}")
        rng_key, rng_key_ = random.split(rng_key)
        _, _, init_strategy, _, _ = run_svi(
            model_c,
            rng_key_,
            data=data_jnp,
            sigma=sigma_jnp,
            phase=phase_jnp,
            Mp_mean=model_params["M_p"],
            Mp_std=model_params["M_p_err"],
            Rstar_mean=model_params["R_star"],
            Rstar_std=model_params["R_star_err"],
            output_dir=str(output_dir),
            num_steps=config.SVI_NUM_STEPS,
            lr=config.SVI_LEARNING_RATE,
        )

        if svi_only:
            print("  SVI complete (svi_only=True); skipping MCMC.")
            return

    # Run MCMC directly
    print(f"\n  Running HMC-NUTS sampling...")
    print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
    print(f"  Chains: {config.MCMC_NUM_CHAINS}")

    # Use median initialization or SVI-derived init values
    kernel = NUTS(
        model_c,
        max_tree_depth=config.MCMC_MAX_TREE_DEPTH,
        init_strategy=init_strategy,
    )
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

        # Convert posterior samples from JAX to numpy
        posterior_np = {k: np.array(v) for k, v in posterior_sample.items()}

        atmo_state = compute_atmospheric_state_from_posterior(
            posterior_samples=posterior_np,
            art=art,
            opa_mols=opa_mols,
            opa_atoms=opa_atoms,
            opa_cias=opa_cias,
            nu_grid=nu_grid,
            pt_profile=pt_profile,
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
            plot_contribution_function(
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
    
    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    print("Running with default settings.")
    print("For more options, use: python -m atmo_retrieval --help\n")

    if config.RETRIEVAL_MODE in ("transmission", "emission"):
        run_retrieval(mode=config.RETRIEVAL_MODE)
    else:
        raise ValueError(f"Unknown retrieval mode: {config.RETRIEVAL_MODE}")
