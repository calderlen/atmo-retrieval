from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import jax
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import SVI, Trace_ELBO

import config
from physics.chemistry import ConstantVMR, FastChemHybridChemistry, FreeVMR
from physics.model import compute_atmospheric_state_from_posterior
from pipeline import retrieval as _retrieval
from pipeline.inference import build_guide, build_svi_optimizer
from exojax.utils.grids import wav2nu
from physics.grid_setup import setup_spectral_operators, setup_wavenumber_grid


_FULL_ATOMIC_SPECIES = deepcopy(config.ATOMIC_SPECIES)
_FULL_MOLPATH_HITEMP = deepcopy(config.MOLPATH_HITEMP)
_FULL_MOLPATH_EXOMOL = deepcopy(config.MOLPATH_EXOMOL)

_EPS = 1.0e-30


@dataclass(frozen=True)
class PrimaryDiagnosticContext:
    planet: str
    ephemeris: str
    epoch: str | None
    mode: str
    chemistry_model: str
    pt_profile: str
    model_params: dict[str, float | None]
    primary_region_config: object
    primary_component: _retrieval.SpectroscopicComponentBundle
    primary_region_sample_prefix: str | None
    primary_component_sample_prefix: str | None
    shared_system: object
    atmosphere_region_configs: tuple[object, ...]
    observation_configs: tuple[object, ...]
    model_c: Any
    model_inputs: dict[str, object]


def _deepcopy_or_value(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


@contextmanager
def temporary_runtime_config(overrides: dict[str, Any]):
    previous = {
        name: _deepcopy_or_value(getattr(config, name))
        for name in overrides
        if hasattr(config, name)
    }
    try:
        for name, value in overrides.items():
            config.set_runtime_config(name, _deepcopy_or_value(value))
        yield
    finally:
        for name, value in previous.items():
            config.set_runtime_config(name, _deepcopy_or_value(value))


def _filter_species_map(
    base: dict[str, Any],
    wanted: Iterable[str] | None,
) -> dict[str, Any]:
    if wanted is None:
        return deepcopy(base)
    wanted_set = {str(name).strip() for name in wanted}
    return {key: deepcopy(value) for key, value in base.items() if key in wanted_set}


def _build_runtime_overrides(
    *,
    planet: str,
    ephemeris: str,
    mode: str,
    observing_mode: str,
    resolution_mode: str,
    nlayer: int,
    n_spectral_points: int,
    load_opacities: bool,
    atoms: Iterable[str] | None,
    molecules: Iterable[str] | None,
) -> dict[str, Any]:
    mol_h = _filter_species_map(_FULL_MOLPATH_HITEMP, molecules)
    mol_e = _filter_species_map(_FULL_MOLPATH_EXOMOL, molecules)
    atom_map = _filter_species_map(_FULL_ATOMIC_SPECIES, atoms)

    return {
        "PLANET": planet,
        "EPHEMERIS": ephemeris,
        "RETRIEVAL_MODE": mode,
        "OBSERVING_MODE": observing_mode,
        "RESOLUTION_MODE": resolution_mode,
        "NLAYER": nlayer,
        "N_SPECTRAL_POINTS": n_spectral_points,
        "OPA_LOAD": load_opacities,
        "MOLPATH_HITEMP": mol_h,
        "MOLPATH_EXOMOL": mol_e,
        "ATOMIC_SPECIES": atom_map,
    }


def build_primary_diagnostic_context(
    *,
    planet: str,
    ephemeris: str,
    epoch: str | None,
    mode: str,
    pt_profile: str,
    chemistry_model: str,
    observing_mode: str,
    resolution_mode: str,
    nlayer: int,
    n_spectral_points: int,
    atoms: Iterable[str] | None = None,
    molecules: Iterable[str] | None = None,
    load_opacities: bool = True,
    data_format: str = "timeseries",
    phase_mode: str = "global",
    atmosphere_regions: list[dict[str, Any]] | None = None,
    apply_sysrem: bool | None = None,
    phoenix_spectrum_path: str | Path | None = None,
    phoenix_cache_dir: str | Path | None = None,
) -> PrimaryDiagnosticContext:
    if data_format != "timeseries":
        raise ValueError("Diagnostics currently only support data_format='timeseries'.")

    overrides = _build_runtime_overrides(
        planet=planet,
        ephemeris=ephemeris,
        mode=mode,
        observing_mode=observing_mode,
        resolution_mode=resolution_mode,
        nlayer=nlayer,
        n_spectral_points=n_spectral_points,
        load_opacities=load_opacities,
        atoms=atoms,
        molecules=molecules,
    )

    with temporary_runtime_config(overrides):
        params = config.get_params(planet, ephemeris)
        model_params = _retrieval._coerce_model_params(params)
        resolved_data_dir = config.get_data_dir(planet=planet, arm=observing_mode, epoch=epoch)

        wav_obs, data, sigma, phase = _retrieval.load_timeseries_data(resolved_data_dir)
        phase = _retrieval._normalize_phase(phase)
        primary_sysrem = _retrieval._validate_sysrem_inputs(
            _retrieval._load_sysrem_inputs(resolved_data_dir),
            n_exp=data.shape[0],
        )

        inst_nus = wav2nu(wav_obs, "AA")
        if inst_nus.size > 1 and np.any(np.diff(inst_nus) <= 0):
            sort_idx = np.argsort(inst_nus)
            inst_nus = inst_nus[sort_idx]
            wav_obs = wav_obs[sort_idx]
            data = data[:, sort_idx]
            sigma = sigma[:, sort_idx]

        _retrieval._preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

        wav_min, wav_max = config.get_wavelength_range(mode=observing_mode)
        nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
            wav_min - config.WAV_MIN_OFFSET,
            wav_max + config.WAV_MAX_OFFSET,
            config.N_SPECTRAL_POINTS,
            unit="AA",
        )
        _retrieval._preflight_grid_checks(inst_nus, nu_grid)

        instrument_resolution = config.get_resolution(resolution_mode=resolution_mode)
        sop_rot, sop_inst, _ = setup_spectral_operators(nu_grid, instrument_resolution)

        primary_art = _retrieval._build_art_for_mode(mode)
        primary_region_name = _retrieval._default_region_name_for_mode(mode)
        shared_system = _retrieval.build_shared_system_config(params=model_params)

        opa_cias = _retrieval.setup_cia_opacities(config.CIA_PATHS, nu_grid)
        opa_mols, _ = _retrieval.load_molecular_opacities(
            config.MOLPATH_HITEMP,
            config.MOLPATH_EXOMOL,
            nu_grid,
            config.OPA_LOAD,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
        )
        opa_atoms, _ = _retrieval.load_atomic_opacities(
            config.ATOMIC_SPECIES,
            nu_grid,
            config.OPA_LOAD,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
        )

        primary_component = _retrieval._build_primary_spectroscopic_component(
            name="spectroscopy",
            mode=mode,
            wav_obs=wav_obs,
            data=data,
            sigma=sigma,
            phase=phase,
            sysrem=primary_sysrem,
            instrument_resolution=instrument_resolution,
            nu_grid=nu_grid,
            inst_nus=inst_nus,
            sop_rot=sop_rot,
            sop_inst=sop_inst,
            opa_cias=opa_cias,
            opa_mols=opa_mols,
            opa_atoms=opa_atoms,
            region_name=primary_region_name,
            Tstar=model_params["T_star"],
            logg_star=model_params["logg_star"],
            metallicity=model_params["Fe_H"],
            Mstar=model_params["M_star"],
            Rstar=model_params["R_star"],
            phoenix_spectrum_path=phoenix_spectrum_path,
            phoenix_cache_dir=phoenix_cache_dir,
            phase_mode=phase_mode,
            apply_sysrem=bool(config.APPLY_SYSREM_DEFAULT) if apply_sysrem is None else bool(apply_sysrem),
            radial_velocity_mode="orbital",
            likelihood_kind="matched_filter",
            subtract_per_exposure_mean=config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT,
            sample_prefix="spectroscopy",
        )

        observation_configs: list[object] = [primary_component.observation_config]
        observations_payload = {
            primary_component.name: primary_component.observation_inputs,
        }
        atmosphere_region_configs, atmosphere_region_lookup = _retrieval._build_atmosphere_regions(
            model_params=model_params,
            primary_mode=mode,
            primary_region_name=primary_region_name,
            primary_art=primary_art,
            observation_configs=observation_configs,
            default_pt_profile=pt_profile,
            default_chemistry_model=chemistry_model,
            default_fastchem_parameter_file=None,
            atmosphere_regions=atmosphere_regions,
        )
        model_c = _retrieval.create_joint_retrieval_model(
            shared_system=shared_system,
            atmosphere_regions=atmosphere_region_configs,
            observations=tuple(observation_configs),
        )

        primary_region_config = atmosphere_region_lookup[primary_region_name]

    return PrimaryDiagnosticContext(
        planet=planet,
        ephemeris=ephemeris,
        epoch=epoch,
        mode=mode,
        chemistry_model=chemistry_model,
        pt_profile=pt_profile,
        model_params=model_params,
        primary_region_config=primary_region_config,
        primary_component=primary_component,
        primary_region_sample_prefix=primary_region_config.sample_prefix,
        primary_component_sample_prefix=primary_component.observation_config.sample_prefix,
        shared_system=shared_system,
        atmosphere_region_configs=tuple(atmosphere_region_configs),
        observation_configs=tuple(observation_configs),
        model_c=model_c,
        model_inputs={"observations": observations_payload},
    )


def default_named_params_for_context(context: PrimaryDiagnosticContext) -> dict[str, float]:
    region = context.primary_region_config
    params: dict[str, float] = {
        "Kp": float(context.model_params["Kp"]),
        "Mp": float(context.model_params["M_p"]),
        "Rstar": float(context.model_params["R_star"]),
        "Rp": float(context.model_params["R_p"]),
        "dRV": 0.0,
    }

    if region.pt_profile == "guillot":
        if region.Tirr_mean is not None:
            params["Tirr"] = float(region.Tirr_mean)
        else:
            params["Tirr"] = 0.5 * (float(region.T_low) + float(region.T_high))
        params["kappa_ir_cgs"] = float(
            np.sqrt(region.kappa_ir_cgs_bounds[0] * region.kappa_ir_cgs_bounds[1])
        )
        params["gamma"] = float(
            np.sqrt(region.gamma_bounds[0] * region.gamma_bounds[1])
        )

    composition_solver = region.composition_solver
    if isinstance(composition_solver, ConstantVMR):
        log_center = 0.5 * (
            float(composition_solver.log_vmr_min) + float(composition_solver.log_vmr_max)
        )
        for atom_name in region.atom_names:
            params[f"logVMR_{atom_name}"] = log_center
        for mol_name in region.mol_names:
            params[f"logVMR_{mol_name}"] = log_center
    elif isinstance(composition_solver, FastChemHybridChemistry):
        log_center = 0.5 * (
            float(composition_solver.log_vmr_min) + float(composition_solver.log_vmr_max)
        )
        for atom_name in region.atom_names:
            if not composition_solver.is_hybrid_managed_species(atom_name):
                params[f"logVMR_{atom_name}"] = log_center
        for mol_name in region.mol_names:
            if not composition_solver.is_hybrid_managed_species(mol_name):
                params[f"logVMR_{mol_name}"] = log_center
        if composition_solver.requires_hybrid_parameters():
            params["log_metallicity"] = float(
                0.5
                * (
                    float(composition_solver.metallicity_range[0])
                    + float(composition_solver.metallicity_range[1])
                )
            )
            params["C_O_ratio"] = float(
                0.5
                * (
                    float(composition_solver.co_ratio_range[0])
                    + float(composition_solver.co_ratio_range[1])
                )
            )
    elif isinstance(composition_solver, FreeVMR):
        log_center = 0.5 * (
            float(composition_solver.log_vmr_min) + float(composition_solver.log_vmr_max)
        )
        for atom_name in region.atom_names:
            for i in range(int(composition_solver.n_nodes)):
                params[f"logVMR_{atom_name}_node{i}"] = log_center
        for mol_name in region.mol_names:
            for i in range(int(composition_solver.n_nodes)):
                params[f"logVMR_{mol_name}_node{i}"] = log_center

    return params


def merge_named_params(
    context: PrimaryDiagnosticContext,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = default_named_params_for_context(context)
    if overrides:
        params.update(overrides)
    return params


def load_named_init_values(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    with np.load(run_dir / "svi_init_values.npz") as init_values:
        return {
            key: np.asarray(init_values[key]).item()
            if np.asarray(init_values[key]).shape == ()
            else np.asarray(init_values[key])
            for key in init_values.files
        }


def load_saved_run_config(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_config_text = (run_dir / "run_config.log").read_text()

    parsed: dict[str, Any] = {
        "run_config_text": run_config_text,
        "molecules_hitemp": [],
        "molecules_exomol": [],
        "atoms": [],
    }
    list_headers = {
        "Molecules (HITEMP):": "molecules_hitemp",
        "Molecules (ExoMol):": "molecules_exomol",
        "Atomic species:": "atoms",
    }
    current_list_key: str | None = None

    for line in run_config_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) in ({"="}, {"-"}):
            continue
        if stripped in list_headers:
            current_list_key = list_headers[stripped]
            continue
        if current_list_key is not None:
            if stripped.startswith("- "):
                parsed[current_list_key].append(stripped[2:].strip())
                continue
            current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = value.strip()

    return parsed


def _parse_bool_token(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "disabled"}:
        return False
    raise ValueError(f"Could not parse boolean value from {value!r}.")


def _infer_chemistry_model_from_run_dir(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    init_path = run_dir / "svi_init_values.npz"
    if not init_path.exists():
        return "constant"

    with np.load(init_path) as init_values:
        keys = set(init_values.files)

    if {"log_metallicity", "C_O_ratio"}.issubset(keys):
        return "fastchem_hybrid_grid"
    if any("_node" in key for key in keys if key.startswith("logVMR_")):
        return "free"
    if any(key.startswith("logVMR_") for key in keys):
        return "constant"
    return "constant"


def build_diag_config_from_run_dir(
    run_dir: str | Path,
    *,
    epoch: str | None = None,
    chemistry_model: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    saved = load_saved_run_config(run_dir)

    resolved_epoch = epoch or saved.get("Epoch")
    if resolved_epoch is None:
        raise ValueError(
            "epoch must be provided for saved runs that do not log it in run_config.log."
        )

    resolved_chemistry_model = (
        chemistry_model
        or saved.get("Chemistry model")
        or _infer_chemistry_model_from_run_dir(run_dir)
    )
    molecules = [
        *saved.get("molecules_hitemp", []),
        *saved.get("molecules_exomol", []),
    ]

    return {
        "planet": str(saved["Planet"]),
        "ephemeris": str(saved["Ephemeris"]),
        "epoch": str(resolved_epoch),
        "mode": str(saved["Mode"]),
        "pt_profile": str(saved["P-T profile"]),
        "chemistry_model": str(resolved_chemistry_model),
        "observing_mode": str(saved["Observing mode"]),
        "resolution_mode": str(saved["Resolution mode"]),
        "nlayer": int(str(saved["Layers"]).replace(",", "")),
        "n_spectral_points": int(str(saved["Spectral points"]).replace(",", "")),
        "atoms": list(saved.get("atoms", [])),
        "molecules": molecules,
        "load_opacities": _parse_bool_token(saved.get("Opacity loading", "True")),
    }


def default_kp_drv_grids(
    context: PrimaryDiagnosticContext,
    *,
    num_kp: int = 11,
    num_drv: int = 17,
    drv_bounds: tuple[float, float] = (-20.0, 20.0),
) -> tuple[np.ndarray, np.ndarray]:
    kp_low = context.model_params.get("Kp_low")
    kp_high = context.model_params.get("Kp_high")
    if kp_low is not None and kp_high is not None:
        kp_grid = np.linspace(float(kp_low), float(kp_high), int(num_kp))
    else:
        kp_grid = np.linspace(0.0, 220.0, int(num_kp))
    drv_grid = np.linspace(float(drv_bounds[0]), float(drv_bounds[1]), int(num_drv))
    return kp_grid, drv_grid


def _normal_logpdf(value: float, mean: float, std: float) -> float:
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    z = (value - mean) / std
    return float(-0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi))


def _truncated_normal_logpdf(
    value: float,
    mean: float,
    std: float,
    *,
    low: float = 0.0,
) -> float:
    if value < low:
        return -np.inf
    alpha = (low - mean) / std
    norm = 1.0 - 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
    if norm <= 0.0:
        return -np.inf
    return _normal_logpdf(value, mean, std) - math.log(norm)


def _shared_system_log_prior(
    context: PrimaryDiagnosticContext,
    *,
    kp: float,
    drv: float,
) -> float:
    kp_low = context.model_params.get("Kp_low")
    kp_high = context.model_params.get("Kp_high")
    kp_err = context.model_params.get("Kp_err")
    if (kp_low is not None) and (kp_high is not None):
        if not (float(kp_low) <= kp <= float(kp_high)):
            return -np.inf
        kp_log_prior = -math.log(float(kp_high) - float(kp_low))
    elif kp_err is None or math.isnan(float(kp_err)) or float(kp_err) <= 0:
        kp_log_prior = 0.0
    else:
        kp_log_prior = _truncated_normal_logpdf(
            kp,
            float(context.model_params["Kp"]),
            float(kp_err),
            low=0.0,
        )

    phase_mode = getattr(context.primary_component.observation_config, "phase_mode", "global")
    drv_log_prior = _normal_logpdf(drv, 0.0, 10.0) if phase_mode == "global" else 0.0
    return float(kp_log_prior + drv_log_prior)


def synthesize_processed_model_timeseries(
    context: PrimaryDiagnosticContext,
    named_params: dict[str, Any],
    *,
    atmo_state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = merge_named_params(context, named_params)

    if atmo_state is None:
        atmo_state = compute_atmospheric_state_from_posterior(
            posterior_samples=params,
            region_config=context.primary_region_config,
            opa_mols=context.primary_component.opa_mols,
            opa_atoms=context.primary_component.opa_atoms,
            opa_cias=context.primary_component.opa_cias,
            nu_grid=context.primary_component.nu_grid,
            use_median=True,
            sample_prefix=context.primary_region_sample_prefix,
        )
    else:
        atmo_state = dict(atmo_state)
        atmo_state["params"] = dict(atmo_state["params"])
        atmo_state["params"].update(params)

    model_ts = _retrieval._synthesize_timeseries_from_atmospheric_state(
        atmo_state=atmo_state,
        model_params=context.model_params,
        region_config=context.primary_region_config,
        component=context.primary_component,
        component_sample_prefix=context.primary_component_sample_prefix,
    )
    return np.asarray(model_ts), atmo_state


def spectroscopic_log_likelihood(
    data: np.ndarray,
    model_ts: np.ndarray,
    sigma: np.ndarray,
    *,
    likelihood_kind: str = "matched_filter",
) -> float:
    data = np.asarray(data, dtype=float)
    model_ts = np.asarray(model_ts, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if likelihood_kind == "gaussian":
        resid = data - model_ts
        return float(
            -0.5
            * np.sum(
                np.square(resid / np.clip(sigma, _EPS, None))
                + np.log(2.0 * np.pi * np.square(np.clip(sigma, _EPS, None)))
            )
        )

    if likelihood_kind != "matched_filter":
        raise ValueError(f"Unsupported likelihood kind: {likelihood_kind}")

    w_ij = 1.0 / np.clip(sigma, _EPS, None) ** 2
    numerator = np.sum(w_ij * data * model_ts, axis=1)
    denominator = np.sum(w_ij * np.square(model_ts), axis=1) + _EPS
    alpha = numerator / denominator
    resid = data - alpha[:, None] * model_ts
    chi2 = np.sum(w_ij * np.square(resid), axis=1)
    norm = np.sum(np.log((2.0 * np.pi) / w_ij), axis=1)
    return float(np.sum(-0.5 * (chi2 + norm)))


def scan_kp_drv_surface(
    context: PrimaryDiagnosticContext,
    *,
    base_params: dict[str, Any],
    kp_grid: np.ndarray,
    drv_grid: np.ndarray,
    data_override: np.ndarray | None = None,
    include_log_prior: bool = False,
) -> dict[str, Any]:
    kp_grid = np.asarray(kp_grid, dtype=float)
    drv_grid = np.asarray(drv_grid, dtype=float)
    log_likelihood = np.empty((kp_grid.size, drv_grid.size), dtype=float)
    log_prior = np.zeros((kp_grid.size, drv_grid.size), dtype=float)
    surface = np.empty((kp_grid.size, drv_grid.size), dtype=float)

    base_params = merge_named_params(context, base_params)
    _, atmo_state = synthesize_processed_model_timeseries(context, base_params)

    observed = (
        np.asarray(data_override, dtype=float)
        if data_override is not None
        else np.asarray(context.primary_component.data, dtype=float)
    )
    sigma = np.asarray(context.primary_component.sigma, dtype=float)
    likelihood_kind = str(context.primary_component.observation_config.likelihood_kind)

    best_surface_value = -np.inf
    best_indices = (0, 0)
    best_params: dict[str, float] = {"Kp": float(kp_grid[0]), "dRV": float(drv_grid[0])}
    best_raw_log_likelihood = -np.inf
    best_raw_indices = (0, 0)
    best_raw_params: dict[str, float] = {"Kp": float(kp_grid[0]), "dRV": float(drv_grid[0])}

    for i, kp in enumerate(kp_grid):
        for j, drv in enumerate(drv_grid):
            trial_params = dict(base_params)
            trial_params["Kp"] = float(kp)
            trial_params["dRV"] = float(drv)
            model_ts, _ = synthesize_processed_model_timeseries(
                context,
                trial_params,
                atmo_state=atmo_state,
            )
            logl = spectroscopic_log_likelihood(
                observed,
                model_ts,
                sigma,
                likelihood_kind=likelihood_kind,
            )
            prior = (
                _shared_system_log_prior(context, kp=float(kp), drv=float(drv))
                if include_log_prior
                else 0.0
            )
            score = logl + prior
            log_likelihood[i, j] = logl
            log_prior[i, j] = prior
            surface[i, j] = score
            if logl > best_raw_log_likelihood:
                best_raw_log_likelihood = logl
                best_raw_indices = (i, j)
                best_raw_params = {"Kp": float(kp), "dRV": float(drv)}
            if score > best_surface_value:
                best_surface_value = score
                best_indices = (i, j)
                best_params = {"Kp": float(kp), "dRV": float(drv)}

    return {
        "kp_grid": kp_grid,
        "drv_grid": drv_grid,
        "log_likelihood": log_likelihood,
        "log_prior": log_prior if include_log_prior else None,
        "surface": surface,
        "surface_label": "log posterior score" if include_log_prior else "log L",
        "include_log_prior": bool(include_log_prior),
        "best_surface_value": best_surface_value,
        "best_indices": best_indices,
        "best_params": best_params,
        "best_log_likelihood": best_raw_log_likelihood,
        "best_log_likelihood_indices": best_raw_indices,
        "best_log_likelihood_params": best_raw_params,
        "base_params": base_params,
    }


def plot_kp_drv_surface(
    scan_result: dict[str, Any],
    *,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    kp_grid = np.asarray(scan_result["kp_grid"])
    drv_grid = np.asarray(scan_result["drv_grid"])
    surface = np.asarray(scan_result.get("surface", scan_result["log_likelihood"]), dtype=float)
    surface = np.where(np.isfinite(surface), surface, np.nan)
    surface_label = str(scan_result.get("surface_label", "log L"))

    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[drv_grid.min(), drv_grid.max(), kp_grid.min(), kp_grid.max()],
        cmap=cmap,
    )
    best = scan_result["best_params"]
    ax.scatter(
        [best["dRV"]],
        [best["Kp"]],
        c="tab:red",
        s=40,
        marker="x",
        label=f"best: Kp={best['Kp']:.1f}, dRV={best['dRV']:.1f}",
    )
    ax.set_xlabel("dRV [km/s]")
    ax.set_ylabel("Kp [km/s]")
    ax.set_title(
        "Kp-dRV Prior-Weighted Surface"
        if scan_result.get("include_log_prior")
        else "Kp-dRV Log-Likelihood Surface"
    )
    ax.legend(loc="best")
    plt.colorbar(im, ax=ax, label=surface_label)
    return ax


def _matched_filter_scale(
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute per-exposure matched-filter scaling alpha = sum(w*d*m)/sum(w*m^2)."""
    w = 1.0 / np.clip(sigma, _EPS, None) ** 2
    num = np.sum(w * data * model, axis=1)
    den = np.sum(w * np.square(model), axis=1) + _EPS
    return num / den


def plot_processed_timeseries_comparison(
    context: PrimaryDiagnosticContext,
    *,
    model_sets: dict[str, dict[str, Any]],
    observed: np.ndarray | None = None,
    wavelength_stride: int = 8,
    show_difference: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    observed = (
        np.asarray(observed, dtype=float)
        if observed is not None
        else np.asarray(context.primary_component.data, dtype=float)
    )
    sigma = np.asarray(context.primary_component.sigma, dtype=float)
    phase = np.asarray(context.primary_component.phase, dtype=float)
    wavelength = np.asarray(context.primary_component.wav_obs, dtype=float)

    model_arrays: dict[str, np.ndarray] = {}
    for label, params in model_sets.items():
        model_ts, _ = synthesize_processed_model_timeseries(context, params)
        model_arrays[label] = np.asarray(model_ts, dtype=float)

    arrays = [observed]
    labels = ["Observed"]
    for label, arr in model_arrays.items():
        arrays.append(arr)
        labels.append(label)

    # ── Row 1: raw panels (observed + models) ──
    ncols = len(arrays)
    has_diff = show_difference and len(model_arrays) == 2
    nrows = 3 if has_diff else 1

    fig, all_axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    # Share y-axis only within each row (not across rows)
    if ncols > 1:
        for row in range(nrows):
            for col in range(1, ncols):
                all_axes[row, col].sharey(all_axes[row, 0])

    wav_strided = wavelength[::wavelength_stride]
    extent = [
        float(wav_strided[0]),
        float(wav_strided[-1]),
        float(phase.min()),
        float(phase.max()),
    ]

    for ax, arr, label in zip(all_axes[0], arrays, labels):
        arr_vmin = float(np.nanpercentile(arr, 2.0))
        arr_vmax = float(np.nanpercentile(arr, 98.0))
        im = ax.imshow(
            arr[:, ::wavelength_stride],
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=arr_vmin,
            vmax=arr_vmax,
            cmap="RdBu_r",
        )
        ax.set_title(label)
        ax.set_xlabel("Wavelength [A]")
        fig.colorbar(im, ax=ax, label="Processed flux", shrink=0.85)
    all_axes[0, 0].set_ylabel("Orbital phase")

    if has_diff:
        model_labels = list(model_arrays.keys())
        m1 = model_arrays[model_labels[0]]
        m2 = model_arrays[model_labels[1]]

        # ── Row 2: matched-filter-scaled residuals ──
        resid_labels = []
        resid_arrays = []
        for mlabel, marr in model_arrays.items():
            alpha = _matched_filter_scale(observed, marr, sigma)
            resid = observed - alpha[:, None] * marr
            resid_arrays.append(resid)
            resid_labels.append(f"Obs − α·{mlabel}")

        # Shared color scale across residuals
        all_resid = np.concatenate(resid_arrays, axis=0)
        resid_vlim = float(np.nanpercentile(np.abs(all_resid), 98.0))

        # First column of row 2: model difference
        diff = m1 - m2
        diff_vlim = float(np.nanpercentile(np.abs(diff), 98.0))
        im_diff = all_axes[1, 0].imshow(
            diff[:, ::wavelength_stride],
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=-diff_vlim,
            vmax=diff_vlim,
            cmap="RdBu_r",
        )
        all_axes[1, 0].set_title(f"{model_labels[0]} − {model_labels[1]}")
        all_axes[1, 0].set_xlabel("Wavelength [A]")
        all_axes[1, 0].set_ylabel("Orbital phase")
        fig.colorbar(im_diff, ax=all_axes[1, 0], label="Δ model flux", shrink=0.85)

        # Residual panels in remaining columns of row 2
        for idx, (rlabel, rarr) in enumerate(zip(resid_labels, resid_arrays)):
            col = idx + 1
            if col >= ncols:
                break
            im_r = all_axes[1, col].imshow(
                rarr[:, ::wavelength_stride],
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=-resid_vlim,
                vmax=resid_vlim,
                cmap="RdBu_r",
            )
            all_axes[1, col].set_title(rlabel)
            all_axes[1, col].set_xlabel("Wavelength [A]")
            fig.colorbar(im_r, ax=all_axes[1, col], label="Residual flux", shrink=0.85)

        # ── Row 3: cross-correlation-style 1D summaries ──
        # Collapse wavelength dimension to show phase-dependent signal strength
        for idx, (mlabel, marr) in enumerate(model_arrays.items()):
            alpha = _matched_filter_scale(observed, marr, sigma)
            # Per-exposure cross-correlation SNR
            w = 1.0 / np.clip(sigma, _EPS, None) ** 2
            ccf = np.sum(w * observed * marr, axis=1) / np.sqrt(
                np.sum(w * np.square(marr), axis=1) + _EPS
            )
            all_axes[2, idx].plot(phase, ccf, lw=1.2, label=mlabel)
            all_axes[2, idx].axhline(0, color="gray", ls="--", lw=0.8)
            all_axes[2, idx].set_title(f"CCF signal: {mlabel}")
            all_axes[2, idx].set_xlabel("Orbital phase")
            all_axes[2, idx].set_ylabel("CCF (weighted)")
            all_axes[2, idx].legend(fontsize=8)

        # Last column of row 3: overlay comparison
        if ncols > 2:
            for mlabel, marr in model_arrays.items():
                w = 1.0 / np.clip(sigma, _EPS, None) ** 2
                ccf = np.sum(w * observed * marr, axis=1) / np.sqrt(
                    np.sum(w * np.square(marr), axis=1) + _EPS
                )
                all_axes[2, 2].plot(phase, ccf, lw=1.2, label=mlabel)
            all_axes[2, 2].axhline(0, color="gray", ls="--", lw=0.8)
            all_axes[2, 2].set_title("CCF comparison")
            all_axes[2, 2].set_xlabel("Orbital phase")
            all_axes[2, 2].set_ylabel("CCF (weighted)")
            all_axes[2, 2].legend(fontsize=8)

    return fig, all_axes


def run_post_sysrem_injection_recovery(
    context: PrimaryDiagnosticContext,
    *,
    inject_params: dict[str, Any],
    kp_grid: np.ndarray,
    drv_grid: np.ndarray,
    injection_scale: float = 1.0,
    include_log_prior: bool = False,
) -> dict[str, Any]:
    inject_params = merge_named_params(context, inject_params)
    injected_model, _ = synthesize_processed_model_timeseries(context, inject_params)
    injected_data = np.asarray(context.primary_component.data, dtype=float) + injection_scale * np.asarray(
        injected_model,
        dtype=float,
    )
    scan = scan_kp_drv_surface(
        context,
        base_params=inject_params,
        kp_grid=kp_grid,
        drv_grid=drv_grid,
        data_override=injected_data,
        include_log_prior=include_log_prior,
    )
    scan["injected_data"] = injected_data
    scan["injected_model"] = injected_model
    scan["injection_scale"] = float(injection_scale)
    scan["inject_params"] = inject_params
    return scan


def run_multiseed_svi(
    context: PrimaryDiagnosticContext,
    *,
    seeds: Iterable[int],
    num_steps: int,
    lr: float,
    lr_decay_steps: int | None = None,
    lr_decay_rate: float | None = None,
    elbo_num_particles: int = 1,
    tracked_params: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    tracked = tuple(
        tracked_params
        or ("Kp", "Tirr", "dRV", "gamma", "kappa_ir_cgs")
    )
    results: list[dict[str, Any]] = []

    for seed in seeds:
        guide = build_guide(
            context.model_c,
            context.model_params["M_p"],
            context.model_params["M_p_err"],
            context.model_params["R_star"],
            context.model_params["R_star_err"],
        )
        optimizer = build_svi_optimizer(
            lr,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
        )
        svi = SVI(
            context.model_c,
            guide,
            optimizer,
            loss=Trace_ELBO(num_particles=elbo_num_particles),
        )
        rng_key = random.PRNGKey(int(seed))
        svi_result = svi.run(
            rng_key,
            int(num_steps),
            **context.model_inputs,
        )
        losses = np.asarray(jax.device_get(svi_result.losses))
        median = guide[-1].median(svi_result.params)
        median = {key: np.asarray(jax.device_get(value)) for key, value in median.items()}

        summary: dict[str, Any] = {
            "seed": int(seed),
            "best_loss": float(np.nanmin(losses)),
            "best_step": int(np.nanargmin(losses)),
            "final_loss": float(losses[-1]),
            "loss_tail_std": float(np.nanstd(losses[-25:])),
        }
        for name in tracked:
            if name in median:
                arr = np.asarray(median[name])
                summary[name] = float(arr.item()) if arr.shape == () else arr
        results.append(summary)

    return results


def plot_multiseed_summary(
    results: list[dict[str, Any]],
    *,
    parameters: Iterable[str] = ("Kp", "Tirr", "dRV"),
) -> tuple[plt.Figure, np.ndarray]:
    params = tuple(parameters)
    fig, axes = plt.subplots(1, len(params), figsize=(4.5 * len(params), 3.8))
    axes = np.atleast_1d(axes)
    seeds = [item["seed"] for item in results]

    for ax, param in zip(axes, params):
        values = [item.get(param, np.nan) for item in results]
        ax.plot(seeds, values, marker="o")
        ax.set_title(param)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Value")
    fig.tight_layout()
    return fig, axes
