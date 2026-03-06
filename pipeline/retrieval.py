import os
import importlib
from contextlib import redirect_stdout
from pathlib import Path

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_median

from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.utils.grids import wav2nu
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import RJ, Rs

import config
from dataio.load import load_observed_spectrum, ResolutionInterpolator
from plotting.aliasing import check_aliasing_with_fe, generate_aliasing_report
from physics.chemistry import ConstantVMR, FastChemHybridChemistry
from physics.grid_setup import setup_wavenumber_grid, setup_spectral_operators
from databases.opacity import setup_cia_opacities, load_molecular_opacities, load_atomic_opacities
from physics.model import (
    create_retrieval_model,
    PhaseMode,
    compute_atmospheric_state_from_posterior,
    planet_rv_kms,
    sysrem_filter_model,
)
from pipeline.inference import run_svi
from plotting.plot import (
    plot_svi_loss,
    plot_transmission_spectrum,
    plot_emission_spectrum,
    plot_temperature_profile,
    plot_contribution_function,
    plot_contribution_per_species,
    plot_contribution_combined,
    save_retrieval_corner_plots,
)


def load_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def _load_sysrem_inputs(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)

    u_candidates = [
        data_dir / "U.npy",
        data_dir / "U_sysrem.npy",
        data_dir / "U_sysrem.npz",
    ]
    invvar_candidates = [
        data_dir / "invvar_spec.npy",
        data_dir / "invvar.npy",
    ]

    u_path = next((p for p in u_candidates if p.exists()), None)
    invvar_path = next((p for p in invvar_candidates if p.exists()), None)

    if u_path is None or invvar_path is None:
        raise FileNotFoundError(
            "apply_sysrem=True requires SYSREM auxiliaries in data directory. "
            "Expected one of {U.npy, U_sysrem.npy, U_sysrem.npz} and one of "
            "{invvar_spec.npy, invvar.npy} in "
            f"{data_dir}."
        )

    if u_path.suffix == ".npz":
        with np.load(u_path) as u_data:
            if "U" in u_data:
                U = u_data["U"]
            elif "U_sysrem" in u_data:
                U = u_data["U_sysrem"]
            else:
                raise KeyError(
                    f"{u_path} must contain key 'U' or 'U_sysrem'."
                )
    else:
        U = np.load(u_path)

    invvar_spec = np.load(invvar_path)
    return U, invvar_spec


def _validate_sysrem_inputs(
    U: np.ndarray,
    invvar_spec: np.ndarray,
    n_exp: int,
) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U)
    invvar_spec = np.asarray(invvar_spec)

    if U.ndim == 3:
        if U.shape[2] != 1:
            raise ValueError(
                "SYSREM basis U has multiple chunks; retrieval currently supports "
                f"only a single chunk. Got U.shape={U.shape}."
            )
        U = U[:, :, 0]

    if U.ndim != 2:
        raise ValueError(f"SYSREM basis U must be 2D; got shape {U.shape}.")
    if invvar_spec.ndim != 1:
        raise ValueError(
            f"invvar_spec must be 1D over exposures; got shape {invvar_spec.shape}."
        )

    if U.shape[0] != n_exp:
        raise ValueError(
            f"U exposure axis mismatch: U.shape[0]={U.shape[0]} but n_exp={n_exp}."
        )
    if invvar_spec.size != n_exp:
        raise ValueError(
            "invvar_spec exposure axis mismatch: "
            f"invvar_spec.size={invvar_spec.size} but n_exp={n_exp}."
        )
    if not np.all(np.isfinite(U)):
        raise ValueError("SYSREM basis U contains non-finite values.")
    if not np.all(np.isfinite(invvar_spec)):
        raise ValueError("invvar_spec contains non-finite values.")
    if np.any(invvar_spec <= 0):
        raise ValueError("invvar_spec must be strictly positive.")

    return U, invvar_spec


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


def _build_composition_solver(
    chemistry_model: str,
    fastchem_parameter_file: str | None,
):
    model = chemistry_model.lower().strip()
    if model == "constant":
        return ConstantVMR()

    if model == "fastchem_hybrid_grid":
        parameter_file = fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
        if parameter_file is None:
            raise ValueError(
                "chemistry_model='fastchem_hybrid_grid' requires a FastChem "
                "parameters.dat path. Pass --fastchem-parameter-file or set "
                "FASTCHEM_PARAMETER_FILE in config."
            )

        return FastChemHybridChemistry(
            fastchem_parameter_file=parameter_file,
            continuum_species=tuple(config.FASTCHEM_HYBRID_CONTINUUM_SPECIES),
            metallicity_range=tuple(config.FASTCHEM_HYBRID_METALLICITY_RANGE),
            co_ratio_range=tuple(config.FASTCHEM_HYBRID_CO_RATIO_RANGE),
            n_metallicity=int(config.FASTCHEM_HYBRID_N_METALLICITY),
            n_co_ratio=int(config.FASTCHEM_HYBRID_N_CO_RATIO),
            log_vmr_min=float(config.LOG_VMR_MIN),
            log_vmr_max=float(config.LOG_VMR_MAX),
            h2_he_ratio=float(config.H2_HE_RATIO),
            n_temp=int(config.FASTCHEM_N_TEMP),
            n_pressure=int(config.FASTCHEM_N_PRESSURE),
            t_min=float(config.FASTCHEM_T_MIN),
            t_max=float(config.FASTCHEM_T_MAX),
            cache_dir=config.FASTCHEM_CACHE_DIR,
        )

    raise ValueError(
        f"Unknown chemistry_model: {chemistry_model}. "
        "Choose from {'constant', 'fastchem_hybrid_grid'}."
    )


def _preflight_spectrum_checks(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    inst_nus: np.ndarray,
) -> None:
    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)
    phase = np.asarray(phase)
    inst_nus = np.asarray(inst_nus)

    if wav_obs.size == 0:
        raise ValueError("wavelength array is empty")
    if inst_nus.size == 0:
        raise ValueError("instrument wavenumber array is empty")
    if data.ndim not in (1, 2):
        raise ValueError(f"data has invalid ndim={data.ndim} (expected 1 or 2)")
    if sigma.shape != data.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match data shape {data.shape}")

    if data.ndim == 1:
        if data.size != wav_obs.size:
            raise ValueError(f"data length {data.size} != wavelength length {wav_obs.size}")
        expected_exposures = 1
    else:
        if data.shape[1] != wav_obs.size:
            raise ValueError(
                f"data spectral axis {data.shape[1]} != wavelength length {wav_obs.size}"
            )
        expected_exposures = data.shape[0]

    if phase.ndim != 1:
        raise ValueError(f"phase has invalid ndim={phase.ndim} (expected 1)")
    if phase.size != expected_exposures:
        raise ValueError(
            f"phase length {phase.size} != number of exposures {expected_exposures}"
        )

    for name, arr in (
        ("wavelength", wav_obs),
        ("instrument wavenumber", inst_nus),
        ("data", data),
        ("sigma", sigma),
        ("phase", phase),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} array contains non-finite values")

    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")


def _preflight_grid_checks(inst_nus: np.ndarray, nu_grid: np.ndarray) -> None:
    inst_nus = np.asarray(inst_nus)
    nu_grid = np.asarray(nu_grid)

    if nu_grid.size == 0 or inst_nus.size == 0:
        raise ValueError("nu_grid and inst_nus must be non-empty")

    inst_min = np.nanmin(inst_nus)
    inst_max = np.nanmax(inst_nus)
    nu_min = np.nanmin(nu_grid)
    nu_max = np.nanmax(nu_grid)

    if inst_min < nu_min or inst_max > nu_max:
        raise ValueError(
            "instrument wavenumber grid is outside model grid: "
            f"inst_nus={inst_min:.4g}..{inst_max:.4g}, "
            f"nu_grid={nu_min:.4g}..{nu_max:.4g}"
        )


def _sample_svi_posterior(
    guide: object | None,
    params: dict | None,
    rng_key: jax.Array,
    num_samples: int,
) -> dict[str, np.ndarray] | None:
    if guide is None or params is None or num_samples <= 0:
        return None

    try:
        svi_draws = guide.sample_posterior(
            rng_key,
            params,
            sample_shape=(num_samples,),
        )
    except Exception as exc:
        print(f"  Warning: failed to sample SVI posterior for corner plots: {exc}")
        return None

    return {
        name: np.asarray(jax.device_get(values))
        for name, values in svi_draws.items()
    }


def _get_piBarr():
    mod = importlib.import_module("exojax.spec.planck")
    return mod.piBarr


def _summarize_observed_spectrum(
    data: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data_arr = np.asarray(data)
    sigma_arr = np.asarray(sigma)

    if data_arr.ndim == 1:
        return data_arr, sigma_arr

    obs_mean = np.mean(data_arr, axis=0)
    obs_err = np.sqrt(np.mean(np.square(sigma_arr), axis=0))
    return obs_mean, obs_err


def _phase_dependent_drv(params: dict[str, float], phase: np.ndarray) -> jnp.ndarray:
    if "dRV_0" in params and "dRV_slope" in params:
        return jnp.asarray(params["dRV_0"] + params["dRV_slope"] * np.asarray(phase))
    return jnp.asarray(params.get("dRV", 0.0))


def _synthesize_timeseries_from_atmospheric_state(
    atmo_state: dict,
    mode: str,
    model_params: dict,
    art: object,
    sop_rot: object,
    sop_inst: object,
    inst_nus: np.ndarray,
    nu_grid: np.ndarray,
    phase: np.ndarray,
    instrument_resolution: float,
    apply_sysrem: bool,
    U_sysrem: np.ndarray | None,
    invvar_spec: np.ndarray | None,
) -> np.ndarray:
    params = atmo_state["params"]

    Rp_rj = float(params.get("Rp", config.DEFAULT_POSTERIOR_RP))
    Mp_mj = float(params.get("Mp", config.DEFAULT_POSTERIOR_MP))
    Rstar_rs = float(params.get("Rstar", model_params["R_star"]))
    Kp_kms = float(params.get("Kp", model_params["Kp"]))
    Vsys_kms = float(params.get("Vsys", model_params["RV_abs"]))

    Rp_cm = Rp_rj * RJ
    Rstar_cm = Rstar_rs * Rs
    g_ref = gravity_surface(Rp_rj, Mp_mj)

    dtau = jnp.asarray(atmo_state["dtau"])
    Tarr = jnp.asarray(atmo_state["Tarr"])
    mmw_profile = jnp.asarray(atmo_state["mmw"])

    rt = art.run(dtau, Tarr, mmw_profile, Rp_cm, g_ref)

    vsini = 2.0 * np.pi * Rp_cm / (float(model_params["period"]) * 86400.0) / 1.0e5
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)
    rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
    rt = sop_inst.ipgauss(rt, beta_inst)

    dRV = _phase_dependent_drv(params, phase)
    rv = planet_rv_kms(jnp.asarray(phase), Kp_kms, Vsys_kms, dRV)
    planet_ts = jax.vmap(lambda v: sop_inst.sampling(rt, v, inst_nus))(rv)

    if mode == "transmission":
        model_ts = jnp.sqrt(planet_ts) * (Rp_cm / Rstar_cm)
    else:
        Tstar = model_params.get("T_star")
        if Tstar is None:
            raise ValueError("T_star is required for emission spectrum plotting.")
        piBarr = _get_piBarr()
        Fs = piBarr(nu_grid, Tstar)
        Fs_ts = jax.vmap(lambda v: sop_inst.sampling(Fs, v, inst_nus))(rv)
        model_ts = planet_ts / jnp.clip(Fs_ts, 1.0e-30, None) * (Rp_cm / Rstar_cm) ** 2

    if config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT:
        model_ts = model_ts - jnp.mean(model_ts, axis=1, keepdims=True)

    if apply_sysrem and U_sysrem is not None and invvar_spec is not None:
        model_ts = sysrem_filter_model(
            model_ts,
            jnp.asarray(U_sysrem),
            jnp.asarray(invvar_spec),
        )

    return np.asarray(jax.device_get(model_ts))


def _compute_model_timeseries_for_plot(
    posterior_samples: dict[str, np.ndarray],
    mode: str,
    model_params: dict,
    art: object,
    opa_mols: dict,
    opa_atoms: dict,
    opa_cias: dict,
    nu_grid: np.ndarray,
    pt_profile: str,
    sop_rot: object,
    sop_inst: object,
    inst_nus: np.ndarray,
    phase: np.ndarray,
    instrument_resolution: float,
    apply_sysrem: bool,
    U_sysrem: np.ndarray | None,
    invvar_spec: np.ndarray | None,
    atmo_state: dict | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    try:
        if atmo_state is None:
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_samples,
                art=art,
                opa_mols=opa_mols,
                opa_atoms=opa_atoms,
                opa_cias=opa_cias,
                nu_grid=nu_grid,
                pt_profile=pt_profile,
                use_median=True,
            )

        model_ts = _synthesize_timeseries_from_atmospheric_state(
            atmo_state=atmo_state,
            mode=mode,
            model_params=model_params,
            art=art,
            sop_rot=sop_rot,
            sop_inst=sop_inst,
            inst_nus=inst_nus,
            nu_grid=nu_grid,
            phase=phase,
            instrument_resolution=instrument_resolution,
            apply_sysrem=apply_sysrem,
            U_sysrem=U_sysrem,
            invvar_spec=invvar_spec,
        )
        return model_ts, atmo_state
    except Exception as exc:
        print(f"  Warning: failed to build diagnostic spectrum plot data: {exc}")
        return None, atmo_state


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
    chemistry_model: str = config.CHEMISTRY_MODEL_DEFAULT,
    fastchem_parameter_file: str | None = None,
    check_aliasing: bool = False,
    compute_contribution: bool = True,
    seed: int = 42,
    wav_obs: np.ndarray | None = None,
    data: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    U: np.ndarray | None = None,
    invvar_spec: np.ndarray | None = None,
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

    apply_sysrem = bool(config.APPLY_SYSREM_DEFAULT)

    U_sysrem: np.ndarray | None = None
    invvar_spec: np.ndarray | None = None
    
    print("\n[1/7] Loading time-series data...")
    if epoch:
        print(f"  Using epoch: {epoch}")
    if any(val is not None for val in (wav_obs, data, sigma, phase)):
        if any(val is None for val in (wav_obs, data, sigma, phase)):
            raise ValueError("Must provide wav_obs, data, sigma, and phase together.")
        phase = _normalize_phase(phase)
        print(f"  Using provided data: {data.shape[0]} exposures x {data.shape[1]} wavelengths")
        print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
        if apply_sysrem:
            if U is None or invvar_spec is None:
                raise ValueError(
                    "apply_sysrem=True requires U and invvar_spec when providing "
                    "wav_obs/data/sigma/phase directly."
                )
            U_sysrem, invvar_spec = _validate_sysrem_inputs(
                U, invvar_spec, n_exp=data.shape[0]
            )
            print(
                f"  Using provided SYSREM auxiliaries: U shape={U_sysrem.shape}, "
                f"invvar_spec shape={invvar_spec.shape}"
            )
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
            if apply_sysrem:
                U_raw, invvar_raw = _load_sysrem_inputs(resolved_data_dir)
                U_sysrem, invvar_spec = _validate_sysrem_inputs(
                    U_raw, invvar_raw, n_exp=data.shape[0]
                )
                print(
                    f"  Loaded SYSREM auxiliaries: U shape={U_sysrem.shape}, "
                    f"invvar_spec shape={invvar_spec.shape}"
                )
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
            if apply_sysrem:
                raise ValueError(
                    "apply_sysrem=True with data_format='spectrum' requires SYSREM "
                    "auxiliaries tied to time-series exposures, which are unavailable "
                    "for single-spectrum input. Use data_format='timeseries' or set "
                    "APPLY_SYSREM_DEFAULT=False."
                )
        else:
            wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
            phase = _normalize_phase(phase)
            print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
            if apply_sysrem:
                U_raw, invvar_raw = _load_sysrem_inputs(resolved_data_dir)
                U_sysrem, invvar_spec = _validate_sysrem_inputs(
                    U_raw, invvar_raw, n_exp=data.shape[0]
                )
                print(
                    f"  Loaded SYSREM auxiliaries: U shape={U_sysrem.shape}, "
                    f"invvar_spec shape={invvar_spec.shape}"
                )
    
    print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} Angstroms")

    # Convert to wavenumber
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

    sop_rot, sop_inst, _ = setup_spectral_operators(nu_grid, Rinst)
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
    print(f"  Chemistry model: {chemistry_model}")

    composition_solver = _build_composition_solver(
        chemistry_model=chemistry_model,
        fastchem_parameter_file=fastchem_parameter_file,
    )
    
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
        apply_sysrem=apply_sysrem,
        composition_solver=composition_solver,
    )
    print(f"  Model created (phase_mode={phase_mode})")

    # Convert data to JAX arrays
    data_jnp = jnp.array(data)
    sigma_jnp = jnp.array(sigma)
    phase_jnp = jnp.array(phase)
    U_jnp = None if U_sysrem is None else jnp.array(U_sysrem)
    invvar_spec_jnp = None if invvar_spec is None else jnp.array(invvar_spec)

    # Run inference
    print("\n[7/7] Running Bayesian inference...")
    rng_key = random.PRNGKey(seed)

    if svi_only and skip_svi:
        raise ValueError("Cannot set svi_only=True when skip_svi=True")

    init_strategy = init_to_median(num_samples=config.INIT_TO_MEDIAN_SAMPLES)
    svi_params: dict | None = None
    svi_guide: object | None = None
    svi_losses: np.ndarray | None = None

    if not skip_svi:
        print(f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}")
        rng_key, rng_key_ = random.split(rng_key)
        svi_params, svi_losses, init_strategy, _, svi_guide = run_svi(
            model_c,
            rng_key_,
            data=data_jnp,
            sigma=sigma_jnp,
            phase=phase_jnp,
            U=U_jnp,
            invvar_spec=invvar_spec_jnp,
            Mp_mean=model_params["M_p"],
            Mp_std=model_params["M_p_err"],
            Rstar_mean=model_params["R_star"],
            Rstar_std=model_params["R_star_err"],
            output_dir=str(output_dir),
            num_steps=config.SVI_NUM_STEPS,
            lr=config.SVI_LEARNING_RATE,
        )

        if svi_only:
            if not no_plots:
                print("  Generating corner plots from SVI posterior...")
                rng_key, rng_key_plot = random.split(rng_key)
                svi_samples_for_plots = _sample_svi_posterior(
                    guide=svi_guide,
                    params=svi_params,
                    rng_key=rng_key_plot,
                    num_samples=max(100, int(config.MCMC_NUM_SAMPLES)),
                )
                save_retrieval_corner_plots(
                    output_dir=str(output_dir),
                    svi_samples=svi_samples_for_plots,
                )

                if svi_losses is not None:
                    plot_svi_loss(
                        np.asarray(jax.device_get(svi_losses)),
                        os.path.join(output_dir, "svi_loss.png"),
                    )

                if svi_samples_for_plots is not None:
                    if (
                        "T0" in svi_samples_for_plots
                        or ("T_btm" in svi_samples_for_plots and "T_top" in svi_samples_for_plots)
                    ):
                        plot_temperature_profile(
                            posterior_samples=svi_samples_for_plots,
                            art=art,
                            save_path=os.path.join(output_dir, "temperature_profile.png"),
                        )
                    else:
                        print(
                            "  Skipping temperature profile plot: no supported temperature "
                            "parameterization in SVI samples."
                        )

                    obs_mean, obs_err = _summarize_observed_spectrum(data, sigma)
                    wav_obs_nm = np.asarray(wav_obs) / 10.0

                    svi_model_ts, _ = _compute_model_timeseries_for_plot(
                        posterior_samples=svi_samples_for_plots,
                        mode=mode,
                        model_params=model_params,
                        art=art,
                        opa_mols=opa_mols,
                        opa_atoms=opa_atoms,
                        opa_cias=opa_cias,
                        nu_grid=np.asarray(nu_grid),
                        pt_profile=pt_profile,
                        sop_rot=sop_rot,
                        sop_inst=sop_inst,
                        inst_nus=np.asarray(inst_nus),
                        phase=np.asarray(phase),
                        instrument_resolution=Rinst,
                        apply_sysrem=apply_sysrem,
                        U_sysrem=U_sysrem,
                        invvar_spec=invvar_spec,
                    )

                    if svi_model_ts is not None:
                        svi_line = np.mean(np.asarray(svi_model_ts), axis=0)
                        if mode == "transmission":
                            plot_transmission_spectrum(
                                wavelength_nm=wav_obs_nm,
                                rp_obs=obs_mean,
                                rp_err=obs_err,
                                rp_hmc=np.atleast_2d(svi_line),
                                rp_svi=svi_line,
                                save_path=os.path.join(output_dir, "transmission_spectrum.png"),
                            )
                        else:
                            plot_emission_spectrum(
                                wavelength_nm=wav_obs_nm,
                                fp_obs=obs_mean,
                                fp_err=obs_err,
                                fp_hmc=np.atleast_2d(svi_line),
                                fp_svi=svi_line,
                                save_path=os.path.join(output_dir, "emission_spectrum.png"),
                            )
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
    mcmc.run(
        rng_key_,
        data=data_jnp,
        sigma=sigma_jnp,
        phase=phase_jnp,
        U=U_jnp,
        invvar_spec=invvar_spec_jnp,
    )
    
    mcmc.print_summary()
    
    # Save results
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()
    
    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)

    posterior_np: dict[str, np.ndarray] | None = None
    svi_samples_for_plots: dict[str, np.ndarray] | None = None

    if not no_plots:
        print("\n  Generating corner plots...")
        posterior_np = {
            name: np.asarray(jax.device_get(values))
            for name, values in posterior_sample.items()
        }

        if svi_params is not None and svi_guide is not None:
            n_hmc_samples = max(100, int(config.MCMC_NUM_SAMPLES))
            if posterior_np:
                first_site = next(iter(posterior_np))
                n_hmc_samples = int(np.asarray(posterior_np[first_site]).shape[0])

            rng_key, rng_key_plot = random.split(rng_key)
            svi_samples_for_plots = _sample_svi_posterior(
                guide=svi_guide,
                params=svi_params,
                rng_key=rng_key_plot,
                num_samples=n_hmc_samples,
            )

        save_retrieval_corner_plots(
            output_dir=str(output_dir),
            hmc_samples=posterior_np,
            svi_samples=svi_samples_for_plots,
        )

        if svi_losses is not None:
            plot_svi_loss(
                np.asarray(jax.device_get(svi_losses)),
                os.path.join(output_dir, "svi_loss.png"),
            )

        if "T0" in posterior_np or ("T_btm" in posterior_np and "T_top" in posterior_np):
            plot_temperature_profile(
                posterior_samples=posterior_np,
                art=art,
                save_path=os.path.join(output_dir, "temperature_profile.png"),
            )
        else:
            print(
                "  Skipping temperature profile plot: no supported temperature "
                "parameterization in HMC samples."
            )

    atmo_state = None
    if compute_contribution or not no_plots:
        print("\n  Computing atmospheric state from posterior...")

        if posterior_np is None:
            posterior_np = {
                name: np.asarray(jax.device_get(values))
                for name, values in posterior_sample.items()
            }

        try:
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
        except Exception:
            if compute_contribution:
                raise
            print("  Warning: unable to compute atmospheric state; skipping spectrum diagnostics.")

    if not no_plots and atmo_state is not None:
        print("  Plotting fitted spectrum diagnostics...")
        wav_obs_nm = np.asarray(wav_obs) / 10.0
        obs_mean, obs_err = _summarize_observed_spectrum(data, sigma)

        hmc_model_ts, atmo_state = _compute_model_timeseries_for_plot(
            posterior_samples=posterior_np,
            mode=mode,
            model_params=model_params,
            art=art,
            opa_mols=opa_mols,
            opa_atoms=opa_atoms,
            opa_cias=opa_cias,
            nu_grid=np.asarray(nu_grid),
            pt_profile=pt_profile,
            sop_rot=sop_rot,
            sop_inst=sop_inst,
            inst_nus=np.asarray(inst_nus),
            phase=np.asarray(phase),
            instrument_resolution=Rinst,
            apply_sysrem=apply_sysrem,
            U_sysrem=U_sysrem,
            invvar_spec=invvar_spec,
            atmo_state=atmo_state,
        )

        svi_model_ts = None
        if svi_samples_for_plots is not None:
            svi_model_ts, _ = _compute_model_timeseries_for_plot(
                posterior_samples=svi_samples_for_plots,
                mode=mode,
                model_params=model_params,
                art=art,
                opa_mols=opa_mols,
                opa_atoms=opa_atoms,
                opa_cias=opa_cias,
                nu_grid=np.asarray(nu_grid),
                pt_profile=pt_profile,
                sop_rot=sop_rot,
                sop_inst=sop_inst,
                inst_nus=np.asarray(inst_nus),
                phase=np.asarray(phase),
                instrument_resolution=Rinst,
                apply_sysrem=apply_sysrem,
                U_sysrem=U_sysrem,
                invvar_spec=invvar_spec,
            )

        if hmc_model_ts is not None or svi_model_ts is not None:
            hmc_plot = hmc_model_ts
            if hmc_plot is None and svi_model_ts is not None:
                hmc_plot = np.atleast_2d(np.mean(np.asarray(svi_model_ts), axis=0))

            if hmc_plot is not None:
                svi_line = np.mean(np.asarray(hmc_plot), axis=0)
                if svi_model_ts is not None:
                    svi_line = np.mean(np.asarray(svi_model_ts), axis=0)

                if mode == "transmission":
                    plot_transmission_spectrum(
                        wavelength_nm=wav_obs_nm,
                        rp_obs=obs_mean,
                        rp_err=obs_err,
                        rp_hmc=np.asarray(hmc_plot),
                        rp_svi=np.asarray(svi_line),
                        save_path=os.path.join(output_dir, "transmission_spectrum.png"),
                    )
                else:
                    plot_emission_spectrum(
                        wavelength_nm=wav_obs_nm,
                        fp_obs=obs_mean,
                        fp_err=obs_err,
                        fp_hmc=np.asarray(hmc_plot),
                        fp_svi=np.asarray(svi_line),
                        save_path=os.path.join(output_dir, "emission_spectrum.png"),
                    )

    if compute_contribution and atmo_state is not None:
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
