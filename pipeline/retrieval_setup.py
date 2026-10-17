"""Construct retrieval components, atmosphere regions, and the joint model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import jax.numpy as jnp
import numpy as np
import config
import config_utils
from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.utils.grids import wav2nu
from dataio.load import (
    load_nasa_archive_spectrum,
    load_observed_spectrum,
    parse_nasa_archive_tbl,
)
from physics.chemistry import (
    ConstantVMR,
    FastChemHybridChemistry,
    FastChemMetallicityEquilibriumChemistry,
    FreeVMR,
)
from physics.grid_setup import setup_wavenumber_grid, setup_spectral_operators
from opacities import (
    load_atomic_opacities,
    load_molecular_opacities,
    premodit_cache_signature,
    setup_cia_opacities,
)
from physics.model import (
    _default_region_name_for_mode,
    BandpassObservationInputs,
    SpectroscopicObservationInputs,
    build_atmosphere_region_config,
    build_bandpass_observation_config,
    build_shared_system_config,
    build_spectroscopic_observation_config,
    create_joint_retrieval_model,
)
from physics.observation_operators import (
    CollapsedEmissionInputs,
    CollapsedTransmissionInputs,
    FrozenTimeseriesInputs,
)
from dataio.bandpass import load_tess_bandpass
from pipeline.retrieval_inputs import (
    FrozenTimeseriesOperatorSpec,
    SysremInputBundle,
    _apply_spectral_thinning,
    _build_model_chunked_sysrem,
    _build_model_frozen_timeseries,
    _build_sysrem_inputs,
    _describe_sysrem_inputs,
    _load_collapsed_emission_operator,
    _load_collapsed_transmission_operator,
    _load_frozen_timeseries_operator_spec,
    _load_sysrem_inputs,
    _normalize_phase,
    _prepare_observed_spectrum_arrays,
    _remap_frozen_timeseries_wavelengths,
    _remap_sysrem_wavelength_sort,
    _scale_spectroscopic_sigma,
    _spectral_subset_indices,
    _subtract_inverse_variance_weighted_constant,
    load_pre_sysrem_timeseries_data,
    load_timeseries_data,
)
from dataio.phoenix import (
    _load_phoenix_surface_flux_on_grid,
    _normalize_phoenix_cache_dir,
)


DEFAULT_ROTATION_VSINI_MAX_KMS = 100.0


STELLAR_ROTATION_VSINI_MARGIN = 1.10


def _rotation_operator_vsini_max(
    stellar_vsini: float | None,
) -> float:
    """Return rotation-kernel support for the planet and star."""
    try:
        stellar_vsini_value = float(stellar_vsini)
    except (TypeError, ValueError):
        return DEFAULT_ROTATION_VSINI_MAX_KMS
    if not np.isfinite(stellar_vsini_value) or stellar_vsini_value <= 0.0:
        return DEFAULT_ROTATION_VSINI_MAX_KMS
    return max(
        DEFAULT_ROTATION_VSINI_MAX_KMS,
        STELLAR_ROTATION_VSINI_MARGIN * stellar_vsini_value,
    )


def _build_composition_solver(
    chemistry_model: str,
    fastchem_parameter_file: str | None,
):
    model = chemistry_model.lower().strip()
    if model == "constant":
        return ConstantVMR()

    if model == "free":
        return FreeVMR()

    parameter_file = fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
    parameter_file = Path(parameter_file).expanduser()

    solver_type = {
        "fastchem_hybrid_grid": FastChemHybridChemistry,
        "fastchem_equilibrium_metallicity_grid": FastChemMetallicityEquilibriumChemistry,
    }[model]
    free_vmr_settings = (
        {"log_vmr_min": float(config.LOG_VMR_MIN), "log_vmr_max": float(config.LOG_VMR_MAX)}
        if model == "fastchem_hybrid_grid"
        else {}
    )
    solver = solver_type(
        fastchem_parameter_file=str(parameter_file),
        continuum_species=tuple(config.FASTCHEM_HYBRID_CONTINUUM_SPECIES),
        metallicity_range=tuple(config.FASTCHEM_HYBRID_METALLICITY_RANGE),
        co_ratio_range=tuple(config.FASTCHEM_HYBRID_CO_RATIO_RANGE),
        n_metallicity=int(config.FASTCHEM_HYBRID_N_METALLICITY),
        n_co_ratio=int(config.FASTCHEM_HYBRID_N_CO_RATIO),
        h2_he_ratio=float(config.H2_HE_RATIO),
        n_temp=int(config.FASTCHEM_N_TEMP),
        n_pressure=int(config.FASTCHEM_N_PRESSURE),
        t_min=float(config.FASTCHEM_T_MIN),
        t_max=float(config.FASTCHEM_T_MAX),
        cache_dir=config.FASTCHEM_CACHE_DIR,
        **free_vmr_settings,
    )
    return solver


def _preflight_grid_checks(inst_nus: np.ndarray, nu_grid: np.ndarray) -> None:
    inst_nus = np.asarray(inst_nus)
    nu_grid = np.asarray(nu_grid)

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


@dataclass(frozen=True)
class SpectroscopicComponentBundle:
    name: str
    wav_obs: np.ndarray
    grid_source_wavelength_range: tuple[float, float]
    data: np.ndarray
    sigma: np.ndarray
    pre_sysrem_data: np.ndarray | None
    pre_sysrem_sigma: np.ndarray | None
    phase: np.ndarray
    sysrem: SysremInputBundle | None
    inst_nus: np.ndarray
    nu_grid: np.ndarray
    sop_rot: object
    sop_inst: object
    instrument_resolution: float
    opa_cias: dict
    opa_mols: dict
    opa_atoms: dict
    observation_config: object
    observation_inputs: SpectroscopicObservationInputs


@dataclass(frozen=True)
class BandpassConstraintBundle:
    name: str
    observation_config: object
    observation_inputs: BandpassObservationInputs


def _append_resolved_spectral_grid_config(
    output_dir: str | Path,
    components: Sequence[SpectroscopicComponentBundle],
) -> None:
    """Append the data-resolved model grids after observations are loaded."""
    log_path = Path(output_dir) / "run_config.log"
    with log_path.open("a") as handle:
        handle.write("\nRESOLVED SPECTRAL GRIDS\n")
        handle.write("-" * 70 + "\n")
        for component in components:
            observed_min, observed_max = component.grid_source_wavelength_range
            model_wavelength = 1.0e8 / np.asarray(component.nu_grid, dtype=float)
            model_min = float(np.min(model_wavelength))
            model_max = float(np.max(model_wavelength))
            signature = premodit_cache_signature(
                component.nu_grid,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                config.PREMODIT_CUTWING,
            )
            handle.write(f"Component: {component.name}\n")
            handle.write(
                "  Observed wavelength range before thinning: "
                f"{observed_min:.6f} - {observed_max:.6f} Angstroms\n"
            )
            handle.write(
                f"  Model wavelength range: {model_min:.6f} - "
                f"{model_max:.6f} Angstroms\n"
            )
            handle.write(f"  Model spectral points: {component.nu_grid.size}\n")
            handle.write("  Grid source: complete prepared wavelength array before thinning\n")
            handle.write(f"  Opacity grid signature: {signature}\n")


def _build_spectroscopic_observation_inputs(
    *,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries: FrozenTimeseriesInputs | None = None,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
) -> SpectroscopicObservationInputs:
    return SpectroscopicObservationInputs(
        data=jnp.asarray(data),
        sigma=jnp.asarray(sigma),
        phase=jnp.asarray(phase),
        U=(
            None
            if frozen_timeseries is not None
            or sysrem is None
            or sysrem.U is None
            else jnp.asarray(sysrem.U)
        ),
        chunked_sysrem=(
            None
            if frozen_timeseries is not None
            else _build_model_chunked_sysrem(sysrem)
        ),
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )


def _coerce_model_params(params: dict) -> dict[str, float | None]:
    def _configured_error(parameter_name: str, default: float) -> float:
        error_name = f"{parameter_name}_err"
        error = params.get(error_name)
        if error is None and hasattr(params[parameter_name], "std_dev"):
            error = params[parameter_name].std_dev
        if error is None:
            return float(default)

        error = float(error)
        if not np.isfinite(error) or error <= 0.0:
            return float(default)
        return error

    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Mp_upper_3sigma = params.get("M_p_upper_3sigma")
    if Kp_low is not None and Kp_low != Kp_low:
        Kp_low = None
    if Kp_high is not None and Kp_high != Kp_high:
        Kp_high = None
    if Mp_upper_3sigma is not None and Mp_upper_3sigma != Mp_upper_3sigma:
        Mp_upper_3sigma = None

    return {
        "Kp": params.get("Kp", config.DEFAULT_KP),
        "Kp_err": params.get("Kp_err", config.DEFAULT_KP_ERR),
        "Kp_low": Kp_low,
        "Kp_high": Kp_high,
        # Retained for provenance/reporting only; the stellar-rest model does
        # not add this absolute stellar velocity to the planet Doppler shift.
        "RV_abs": params.get("RV_abs", config.DEFAULT_RV_ABS),
        "RV_abs_err": params.get("RV_abs_err", config.DEFAULT_RV_ABS_ERR),
        "R_p": params["R_p"].nominal_value if hasattr(params["R_p"], "nominal_value") else params["R_p"],
        "R_p_err": _configured_error("R_p", config.DEFAULT_RP_ERR),
        "M_p": params["M_p"].nominal_value if hasattr(params["M_p"], "nominal_value") else params["M_p"],
        "M_p_err": _configured_error("M_p", config.DEFAULT_MP_ERR),
        "M_p_upper_3sigma": Mp_upper_3sigma,
        "M_star": (
            params["M_star"].nominal_value
            if ("M_star" in params and hasattr(params["M_star"], "nominal_value"))
            else params.get("M_star")
        ),
        "R_star": params["R_star"].nominal_value if hasattr(params["R_star"], "nominal_value") else params["R_star"],
        "R_star_err": _configured_error("R_star", config.DEFAULT_RSTAR_ERR),
        "T_star": params.get("T_star", config.DEFAULT_TSTAR),
        "logg_star": params.get("logg_star"),
        "Fe_H": params.get("Fe_H"),
        "v_sini_star": params.get("v_sini_star"),
        "gamma1": params.get("gamma1"),
        "gamma2": params.get("gamma2"),
        "T_eq": params.get("T_eq"),
        "Tirr_mean": params.get("Tirr_mean", params.get("T_eq")),
        "Tirr_std": params.get("Tirr_std"),
        "a": (
            params["a"].nominal_value
            if ("a" in params and hasattr(params["a"], "nominal_value"))
            else params.get("a")
        ),
        "period": params["period"].nominal_value if hasattr(params["period"], "nominal_value") else params["period"],
    }


def _normalize_retrieval_mode(mode: str) -> str:
    return str(mode).lower().strip()


def _build_art_for_mode(mode: str) -> object:
    mode = _normalize_retrieval_mode(mode)
    pressure_top, pressure_btm = config_utils.get_pressure_bounds_for_mode(mode)
    if mode == "transmission":
        art = ArtTransPure(
            pressure_top=pressure_top,
            pressure_btm=pressure_btm,
            nlayer=config.NLAYER,
        )
    else:
        art = ArtEmisPure(
            pressure_top=pressure_top,
            pressure_btm=pressure_btm,
            nlayer=config.NLAYER,
        )
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    return art


def _normalize_region_name(region_name: str | None, mode: str) -> str:
    if region_name is None:
        return _default_region_name_for_mode(mode)
    normalized = str(region_name).strip()
    return normalized


def _resolve_model_wavelength_range(
    wav_obs_before_thinning: np.ndarray,
) -> tuple[float, float, float, float]:
    """Resolve observed and padded model bounds from the complete data grid."""
    wavelengths = np.asarray(wav_obs_before_thinning, dtype=float)
    observed_min = float(np.min(wavelengths))
    observed_max = float(np.max(wavelengths))
    return (
        observed_min,
        observed_max,
        observed_min - float(config.WAV_MIN_OFFSET),
        observed_max + float(config.WAV_MAX_OFFSET),
    )


def _build_component_grid_and_ops(
    wav_obs: np.ndarray,
    instrument_resolution: float,
    *,
    wav_obs_before_thinning: np.ndarray | None = None,
    stellar_vsini: float | None = None,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    inst_nus = wav2nu(wav_obs, "AA")
    grid_source = (
        wav_obs
        if wav_obs_before_thinning is None
        else wav_obs_before_thinning
    )
    _, _, model_min, model_max = _resolve_model_wavelength_range(grid_source)
    nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
        model_min,
        model_max,
        config.N_SPECTRAL_POINTS,
        unit="AA",
    )
    _preflight_grid_checks(inst_nus, nu_grid)
    rotation_vsini_max = _rotation_operator_vsini_max(stellar_vsini)
    sop_rot, sop_inst, _ = setup_spectral_operators(
        nu_grid,
        instrument_resolution,
        vsini_max=rotation_vsini_max,
    )
    if rotation_vsini_max > DEFAULT_ROTATION_VSINI_MAX_KMS:
        print(
            "  Stellar rotation operator support: "
            f"vsini_max={rotation_vsini_max:g} km/s"
        )
    return inst_nus, nu_grid, sop_rot, sop_inst


def _build_in_memory_timeseries_component_spec(
    *,
    name: str,
    mode: str,
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries_operator: FrozenTimeseriesOperatorSpec | None,
    instrument_resolution: float,
    apply_sysrem: bool,
    radial_velocity_mode: str,
    data_format: str = "timeseries",
    subtract_weighted_global_mean: bool = False,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
    region_name: str | None = None,
    pre_sysrem_data: np.ndarray | None = None,
    pre_sysrem_sigma: np.ndarray | None = None,
    wav_obs_before_thinning: np.ndarray | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "data_format": data_format,
        "wav_obs": np.asarray(wav_obs),
        "wav_obs_before_thinning": (
            np.asarray(wav_obs)
            if wav_obs_before_thinning is None
            else np.asarray(wav_obs_before_thinning)
        ),
        "data": np.asarray(data),
        "sigma": np.asarray(sigma),
        "phase": np.asarray(phase),
        "sysrem": sysrem,
        "frozen_timeseries_operator": frozen_timeseries_operator,
        "pre_sysrem_data": None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        "pre_sysrem_sigma": None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        "instrument_resolution": float(instrument_resolution),
        "apply_sysrem": bool(apply_sysrem),
        "radial_velocity_mode": radial_velocity_mode,
        "subtract_weighted_global_mean": bool(subtract_weighted_global_mean),
        "collapsed_emission": collapsed_emission,
        "collapsed_transmission": collapsed_transmission,
    }
    if region_name is not None:
        spec["region_name"] = region_name
    return spec


def _load_opacity_bundle(
    nu_grid: np.ndarray,
    *,
    atomic_species: dict[str, dict] | None = None,
    molpath_hitemp: dict[str, str | Path] | None = None,
    molpath_exomol: dict[str, str | Path] | None = None,
) -> tuple[dict, dict, dict]:
    if atomic_species is None:
        atomic_species = config.ATOMIC_SPECIES
    if molpath_hitemp is None:
        molpath_hitemp = config.MOLPATH_HITEMP
    if molpath_exomol is None:
        molpath_exomol = config.MOLPATH_EXOMOL
    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    opa_mols, _ = load_molecular_opacities(
        molpath_hitemp,
        molpath_exomol,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    opa_atoms, _ = load_atomic_opacities(
        atomic_species,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    return opa_cias, opa_mols, opa_atoms


def _build_primary_spectroscopic_component(
    *,
    name: str,
    mode: str,
    wav_obs: np.ndarray,
    grid_source_wavelength_range: tuple[float, float],
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries: FrozenTimeseriesInputs | None,
    instrument_resolution: float,
    nu_grid: np.ndarray,
    inst_nus: np.ndarray,
    sop_rot: object,
    sop_inst: object,
    opa_cias: dict,
    opa_mols: dict,
    opa_atoms: dict,
    region_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    stellar_vsini: float | None,
    stellar_limb_darkening_u1: float | None,
    stellar_limb_darkening_u2: float | None,
    phoenix_cache_dir: str | Path | None,
    apply_sysrem: bool,
    radial_velocity_mode: str,
    subtract_weighted_global_mean: bool,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
    sample_prefix: str | None = None,
    pre_sysrem_data: np.ndarray | None = None,
    pre_sysrem_sigma: np.ndarray | None = None,
) -> SpectroscopicComponentBundle:
    mode = _normalize_retrieval_mode(mode)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        stellar_vsini=stellar_vsini,
        stellar_limb_darkening_u1=stellar_limb_darkening_u1,
        stellar_limb_darkening_u2=stellar_limb_darkening_u2,
        radial_velocity_mode=radial_velocity_mode,
        subtract_weighted_global_mean=subtract_weighted_global_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=sample_prefix,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        grid_source_wavelength_range=grid_source_wavelength_range,
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        pre_sysrem_data=None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        pre_sysrem_sigma=None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _sanitize_name_for_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _infer_tbl_mode(metadata: dict[str, str]) -> str | None:
    spec_type = metadata.get("SPEC_TYPE", "").strip().lower()
    if "eclipse" in spec_type:
        return "emission"
    if "transit" in spec_type or "transmission" in spec_type:
        return "transmission"
    return None


def _convert_unit_to_micron(values: np.ndarray, unit: str) -> np.ndarray:
    unit_norm = unit.lower().strip()
    if "angstrom" in unit_norm or unit_norm == "aa":
        return values / 10000.0
    if unit_norm in {"nm", "nanometer", "nanometers"}:
        return values / 1000.0
    return values


def _combine_scalar_measurements(values: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
    if np.any(mask):
        weights = 1.0 / np.square(sigma[mask])
        value = float(np.sum(values[mask] * weights) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        return value, uncertainty

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite scalar measurements available to combine.")

    value = float(np.mean(finite_values))
    if finite_values.size == 1:
        return value, float("nan")
    return value, float(np.std(finite_values, ddof=1) / np.sqrt(finite_values.size))


def _make_tophat_bandpass(center_micron: float, bandwidth_micron: float, n_samples: int = 64) -> tuple[np.ndarray, np.ndarray]:
    width = float(bandwidth_micron)
    if not np.isfinite(width) or width <= 0:
        width = max(center_micron * 0.05, 0.02)

    half = width / 2.0
    start = max(center_micron - half, 1.0e-6)
    stop = center_micron + half
    wavelength_m = np.linspace(start, stop, int(n_samples), dtype=float) * 1.0e-6
    response = np.ones_like(wavelength_m)
    return wavelength_m, response


def _resolve_lowres_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "lrs" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def _resolve_bandpass_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "phot" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def make_joint_spectrum_component_from_tbl(tbl_path: str | Path) -> dict[str, Any]:
    tbl_path = _resolve_lowres_tbl_path(tbl_path)
    metadata, _columns, _data_by_col, _units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)

    return {
        "name": f"lrs_{_sanitize_name_for_id(tbl_path.stem)}",
        "mode": mode,
        "tbl_path": str(tbl_path),
        "data_format": "spectrum",
    }


def make_bandpass_constraints_from_tbl(tbl_path: str | Path) -> list[dict[str, Any]]:
    tbl_path = _resolve_bandpass_tbl_path(tbl_path)
    metadata, _columns, data_by_col, units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)

    wav_angstrom, spectrum, sigma, _meta = load_nasa_archive_spectrum(tbl_path, mode=mode)

    bandwidth_values = np.full_like(wav_angstrom, np.nan, dtype=float)
    if "BANDWIDTH" in data_by_col:
        raw_bandwidth_values = []
        for value in data_by_col["BANDWIDTH"]:
            raw_bandwidth_values.append(np.nan if value is None else float(value))
        raw_bandwidth = np.asarray(
            raw_bandwidth_values,
            dtype=float,
        )
        if raw_bandwidth.size == wav_angstrom.size:
            bandwidth_values = _convert_unit_to_micron(
                raw_bandwidth,
                units_by_col.get("BANDWIDTH", ""),
            )

    grouped_indices: dict[float, list[int]] = {}
    for idx, wav in enumerate(np.round(wav_angstrom, 6)):
        grouped_indices.setdefault(float(wav), []).append(idx)

    instrument = (metadata.get("INSTRUMENT") or metadata.get("INSTRUMENT_NAME") or "").strip()
    facility = (metadata.get("FACILITY") or metadata.get("FACILITY_NAME") or "").strip()
    instrument_text = f"{instrument} {facility}".lower()
    base_name = f"lrs_{_sanitize_name_for_id(tbl_path.stem)}"

    constraints: list[dict[str, Any]] = []
    for group_idx, indices in enumerate(grouped_indices.values(), start=1):
        value, value_sigma = _combine_scalar_measurements(
            spectrum[indices],
            sigma[indices],
        )
        if not np.isfinite(value_sigma) or value_sigma <= 0:
            continue

        name = base_name if len(grouped_indices) == 1 else f"{base_name}_{group_idx}"
        observable = "eclipse_depth" if mode == "emission" else "transit_depth"
        constraint: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "observable": observable,
            "value": value,
            "sigma": value_sigma,
        }

        if "tess" not in instrument_text:
            center_micron = float(np.mean(wav_angstrom[indices]) / 10000.0)
            bandwidth = bandwidth_values[indices]
            finite_bandwidth = bandwidth[np.isfinite(bandwidth) & (bandwidth > 0)]
            bandwidth_micron = (
                float(np.mean(finite_bandwidth))
                if finite_bandwidth.size > 0
                else float("nan")
            )
            wavelength_m, response = _make_tophat_bandpass(center_micron, bandwidth_micron)
            constraint["wavelength_m"] = wavelength_m
            constraint["response"] = response

        constraints.append(constraint)

    return constraints


def _build_atmosphere_regions(
    *,
    model_params: dict[str, float | None],
    primary_mode: str,
    primary_region_name: str,
    primary_art: object,
    observation_configs: list[object],
    default_pt_profile: str,
    default_chemistry_model: str,
    default_fastchem_parameter_file: str | None,
    atmosphere_regions: list[dict[str, Any]] | None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    explicit_specs: dict[str, dict[str, Any]] = {}
    if atmosphere_regions:
        for raw_spec in atmosphere_regions:
            spec = dict(raw_spec)
            name = str(spec.get("name", "")).strip()
            explicit_specs[name] = spec

    component_modes: dict[str, str] = {}
    region_mol_names: dict[str, set[str]] = {}
    region_atom_names: dict[str, set[str]] = {}
    for observation_config in observation_configs:
        region_name = str(observation_config.region_name)
        region_mode = _normalize_retrieval_mode(observation_config.mode)
        component_modes[region_name] = region_mode
        region_mol_names.setdefault(region_name, set()).update(observation_config.opa_mols.keys())
        region_atom_names.setdefault(region_name, set()).update(observation_config.opa_atoms.keys())

    region_configs = []
    region_lookup: dict[str, object] = {}
    for region_name in component_modes:
        region_mode = component_modes[region_name]
        spec = explicit_specs.get(region_name, {})
        chemistry_name = str(spec.get("chemistry_model", default_chemistry_model))
        chemistry_param_file = spec.get(
            "fastchem_parameter_file",
            default_fastchem_parameter_file,
        )
        composition_solver = _build_composition_solver(
            chemistry_model=chemistry_name,
            fastchem_parameter_file=chemistry_param_file,
        )
        kappa_bounds = spec.get("kappa_ir_cgs_bounds")
        gamma_bounds = spec.get("gamma_bounds")

        art = (
            primary_art
            if region_name == primary_region_name and region_mode == primary_mode
            else _build_art_for_mode(region_mode)
        )
        region_pt_profile = config_utils.resolve_pt_profile_for_region(
            region_mode,
            primary_pt_profile=default_pt_profile,
            is_primary=(
                region_name == primary_region_name and region_mode == primary_mode
            ),
            pt_profile=spec.get("pt_profile"),
        )
        region_config = build_atmosphere_region_config(
            mode=region_mode,
            art=art,
            mol_names=tuple(sorted(region_mol_names[region_name])),
            atom_names=tuple(sorted(region_atom_names[region_name])),
            pt_profile=region_pt_profile,
            T_low=spec.get("T_low"),
            T_high=spec.get("T_high"),
            Tirr_mean=spec.get("Tirr_mean", model_params.get("Tirr_mean")),
            Tirr_std=spec.get("Tirr_std", model_params.get("Tirr_std")),
            Tint_fixed=spec.get("Tint_fixed"),
            kappa_ir_cgs_bounds=None if kappa_bounds is None else tuple(kappa_bounds),
            gamma_bounds=None if gamma_bounds is None else tuple(gamma_bounds),
            composition_solver=composition_solver,
            name=region_name,
            sample_prefix=spec.get("sample_prefix"),
            velocity_offset_mode=str(
                spec.get("velocity_offset_mode", "shared")
            ),
            velocity_offset_species=tuple(
                spec.get("velocity_offset_species", ())
            ),
            velocity_offset_bounds_kms=tuple(
                spec.get("velocity_offset_bounds_kms", (-20.0, 20.0))
            ),
        )
        region_configs.append(region_config)
        region_lookup[region_name] = region_config

    return tuple(region_configs), region_lookup


def _load_joint_spectroscopic_component(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_stellar_vsini: float | None,
    default_stellar_limb_darkening_u1: float | None,
    default_stellar_limb_darkening_u2: float | None,
    default_phoenix_cache_dir: str | Path | None,
    default_sigma_scale: float = 1.0,
    default_spectral_stride: int = 1,
    default_spectral_offset: int = 0,
) -> SpectroscopicComponentBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_component"))
    data_format = str(spec.get("data_format", "spectrum")).lower().strip()
    instrument_resolution = float(spec.get("instrument_resolution", config_utils.get_resolution()))
    radial_velocity_mode = str(spec.get("radial_velocity_mode", "orbital" if data_format == "timeseries" else "none"))
    apply_sysrem = bool(spec.get("apply_sysrem", data_format == "timeseries" and config.APPLY_SYSREM_DEFAULT))
    subtract_weighted_global_mean = bool(
        spec.get(
            "subtract_weighted_global_mean",
            component_mode == "transmission"
            and data_format == "spectrum"
            and "tbl_path" not in spec,
        )
    )
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    stellar_vsini = spec.get("v_sini_star", default_stellar_vsini)
    stellar_limb_darkening_u1 = spec.get(
        "gamma1",
        spec.get(
            "stellar_limb_darkening_u1",
            default_stellar_limb_darkening_u1,
        ),
    )
    stellar_limb_darkening_u2 = spec.get(
        "gamma2",
        spec.get(
            "stellar_limb_darkening_u2",
            default_stellar_limb_darkening_u2,
        ),
    )
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)
    sigma_scale = float(spec.get("sigma_scale", default_sigma_scale))
    spectral_stride = int(spec.get("spectral_stride", default_spectral_stride))
    spectral_offset = int(spec.get("spectral_offset", default_spectral_offset))
    pre_sysrem_data = None
    pre_sysrem_sigma = None
    collapsed_emission = spec.get("collapsed_emission")
    collapsed_transmission = spec.get("collapsed_transmission")
    frozen_timeseries_operator = spec.get("frozen_timeseries_operator")
    component_data_dir: Path | None = None

    if "tbl_path" in spec:
        wav_obs, spectrum, uncertainty, _meta = load_nasa_archive_spectrum(
            spec["tbl_path"],
            mode=component_mode,
        )
        # The spectroscopy forward model represents transmission as the
        # negative perturbation of normalized stellar flux, whereas archive
        # tables conventionally store a positive absolute transit depth.
        # Convert only at this internal spectroscopic boundary; scalar
        # bandpass constraints continue to use positive transit depths.
        if component_mode == "transmission":
            spectrum = -np.asarray(spectrum)
        data = spectrum[np.newaxis, :]
        sigma = uncertainty[np.newaxis, :]
        phase = np.zeros((1,), dtype=float)
        sysrem = None
    elif all(key in spec for key in ("wav_obs", "data", "sigma")):
        wav_obs = np.asarray(spec["wav_obs"])
        data = np.asarray(spec["data"])
        sigma = np.asarray(spec["sigma"])
        phase = np.asarray(spec.get("phase", np.zeros((1 if data.ndim == 1 else data.shape[0],), dtype=float)))
        if spec.get("pre_sysrem_data") is not None and spec.get("pre_sysrem_sigma") is not None:
            pre_sysrem_data = np.asarray(spec["pre_sysrem_data"])
            pre_sysrem_sigma = np.asarray(spec["pre_sysrem_sigma"])
        if spec.get("sysrem") is not None:
            sysrem_spec = spec["sysrem"]
            if isinstance(sysrem_spec, SysremInputBundle):
                sysrem = sysrem_spec
            else:
                sysrem = _build_sysrem_inputs(sysrem_spec)
        elif spec.get("U") is not None:
            raw_sysrem = {"U": np.asarray(spec["U"])}
            sysrem = _build_sysrem_inputs(raw_sysrem)
        else:
            sysrem = None
    elif "data_dir" in spec:
        data_dir = Path(spec["data_dir"])
        component_data_dir = data_dir
        if data_format == "timeseries":
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            frozen_timeseries_operator = _load_frozen_timeseries_operator_spec(data_dir)
            apply_sysrem = frozen_timeseries_operator.has_sysrem
            pre_sysrem_bundle = load_pre_sysrem_timeseries_data(data_dir)
            if pre_sysrem_bundle is not None:
                pre_sysrem_data, pre_sysrem_sigma = pre_sysrem_bundle
            phase = _normalize_phase(phase)
            if apply_sysrem:
                sysrem = _build_sysrem_inputs(_load_sysrem_inputs(data_dir))
            else:
                sysrem = None
        elif data_format == "spectrum":
            suffix = "transmission" if component_mode == "transmission" else "emission"
            wav_obs, spectrum, uncertainty, _ = load_observed_spectrum(
                str(data_dir / f"wavelength_{suffix}.npy"),
                str(data_dir / f"spectrum_{suffix}.npy"),
                str(data_dir / f"uncertainty_{suffix}.npy"),
            )
            data = spectrum[np.newaxis, :]
            sigma = uncertainty[np.newaxis, :]
            phase = np.zeros((1,), dtype=float)
            sysrem = None
        else:
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            sysrem = None

    wav_obs_before_thinning = np.asarray(
        spec.get("wav_obs_before_thinning", wav_obs),
        dtype=float,
    ).copy()
    observed_min, observed_max, _, _ = _resolve_model_wavelength_range(
        wav_obs_before_thinning
    )

    frozen_subset_indices = (
        None
        if frozen_timeseries_operator is None
        or (spectral_stride == 1 and spectral_offset == 0)
        else _spectral_subset_indices(
            np.asarray(wav_obs).size,
            spectral_stride,
            spectral_offset,
        )
    )
    wav_obs, data, sigma, sysrem, pre_sysrem_data, pre_sysrem_sigma = _apply_spectral_thinning(
        wav_obs=wav_obs,
        data=data,
        sigma=sigma,
        sysrem=sysrem,
        pre_sysrem_data=pre_sysrem_data,
        pre_sysrem_sigma=pre_sysrem_sigma,
        spectral_stride=spectral_stride,
        spectral_offset=spectral_offset,
        component_name=name,
    )
    frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
        frozen_timeseries_operator,
        frozen_subset_indices,
    )

    inst_nus_before_prepare = wav2nu(np.asarray(wav_obs), "AA")
    sort_idx_for_prepare = None
    if inst_nus_before_prepare.size > 1 and np.any(np.diff(inst_nus_before_prepare) <= 0):
        sort_idx_for_prepare = np.argsort(inst_nus_before_prepare)

    wav_obs, data, sigma, inst_nus = _prepare_observed_spectrum_arrays(wav_obs, data, sigma)
    sysrem = _remap_sysrem_wavelength_sort(sysrem, sort_idx_for_prepare)
    frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
        frozen_timeseries_operator,
        sort_idx_for_prepare,
    )
    frozen_timeseries = _build_model_frozen_timeseries(
        frozen_timeseries_operator,
        sysrem,
    )
    if (
        collapsed_emission is None
        and component_data_dir is not None
        and component_mode == "emission"
        and data_format == "spectrum"
        and (component_data_dir / "collapse_metadata.json").exists()
    ):
        collapsed_emission = _load_collapsed_emission_operator(
            component_data_dir,
            wav_obs,
        )
        if "subtract_weighted_global_mean" not in spec:
            subtract_weighted_global_mean = True
    if (
        collapsed_transmission is None
        and component_data_dir is not None
        and component_mode == "transmission"
        and data_format == "spectrum"
        and (component_data_dir / "collapse_metadata.json").exists()
    ):
        collapsed_transmission = _load_collapsed_transmission_operator(
            component_data_dir,
            wav_obs,
        )
    sigma = _scale_spectroscopic_sigma(sigma, sigma_scale)
    if subtract_weighted_global_mean:
        data = _subtract_inverse_variance_weighted_constant(data, sigma)
    if pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        pre_sysrem_data = np.asarray(pre_sysrem_data)
        pre_sysrem_sigma = np.asarray(pre_sysrem_sigma)
        if sort_idx_for_prepare is not None:
            if pre_sysrem_data.ndim == 2:
                pre_sysrem_data = pre_sysrem_data[:, sort_idx_for_prepare]
                pre_sysrem_sigma = pre_sysrem_sigma[:, sort_idx_for_prepare]
            else:
                pre_sysrem_data = pre_sysrem_data[sort_idx_for_prepare]
                pre_sysrem_sigma = pre_sysrem_sigma[sort_idx_for_prepare]
        pre_sysrem_sigma = _scale_spectroscopic_sigma(pre_sysrem_sigma, sigma_scale)
    if phase.ndim == 0:
        phase = np.asarray([float(phase)])
    if data_format == "timeseries":
        phase = _normalize_phase(phase)
    elif phase.size == 0:
        phase = np.zeros((1,), dtype=float)

    inst_nus_component, nu_grid, sop_rot, sop_inst = _build_component_grid_and_ops(
        wav_obs,
        instrument_resolution,
        wav_obs_before_thinning=wav_obs_before_thinning,
        stellar_vsini=(stellar_vsini if component_mode == "emission" else None),
    )
    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(
        nu_grid,
        atomic_species=spec.get("atomic_species"),
        molpath_hitemp=spec.get("molpath_hitemp"),
        molpath_exomol=spec.get("molpath_exomol"),
    )
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )

    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus_component,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        stellar_vsini=stellar_vsini,
        stellar_limb_darkening_u1=stellar_limb_darkening_u1,
        stellar_limb_darkening_u2=stellar_limb_darkening_u2,
        radial_velocity_mode=radial_velocity_mode,
        subtract_weighted_global_mean=subtract_weighted_global_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=name,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        grid_source_wavelength_range=(observed_min, observed_max),
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        pre_sysrem_data=None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        pre_sysrem_sigma=None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus_component),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _load_bandpass_constraint(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_semi_major_axis_au: float | None,
    default_phoenix_cache_dir: str | Path | None,
) -> BandpassConstraintBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_bandpass"))
    observable = str(spec["observable"])
    value = float(spec["value"])
    sigma = float(spec["sigma"])
    photon_weighted = bool(spec.get("photon_weighted", False))
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)
    include_reflection = bool(spec.get("include_reflection", False))
    semi_major_axis_au = spec.get("semi_major_axis_au", default_semi_major_axis_au)
    if semi_major_axis_au is not None:
        semi_major_axis_au = float(semi_major_axis_au)
    geometric_albedo_bounds_raw = spec.get("geometric_albedo_bounds")
    geometric_albedo_bounds = (
        None
        if geometric_albedo_bounds_raw is None
        else tuple(float(x) for x in geometric_albedo_bounds_raw)
    )
    model_sigma_raw = spec.get("model_sigma")
    model_sigma = None if model_sigma_raw is None else float(model_sigma_raw)
    model_sigma_bounds_raw = spec.get("model_sigma_bounds")
    model_sigma_bounds = (
        None
        if model_sigma_bounds_raw is None
        else tuple(float(x) for x in model_sigma_bounds_raw)
    )

    if "wavelength_m" in spec and "response" in spec:
        wavelength_m = np.asarray(spec["wavelength_m"], dtype=float)
        response = np.asarray(spec["response"], dtype=float)
    else:
        bandpass_path = spec.get("bandpass_path")
        wavelength_m, response, _used_path = load_tess_bandpass(
            bandpass_path,
            download_if_missing=bool(spec.get("download_bandpass", True)),
        )

    wavelength_angstrom = np.asarray(wavelength_m, dtype=float) * 1.0e10
    if "nu_grid" in spec:
        nu_grid = np.asarray(spec["nu_grid"], dtype=float)
    else:
        nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
            float(np.min(wavelength_angstrom)) - config.WAV_MIN_OFFSET,
            float(np.max(wavelength_angstrom)) + config.WAV_MAX_OFFSET,
            config.N_SPECTRAL_POINTS,
            unit="AA",
        )

    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(nu_grid)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_bandpass_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        wavelength_m=wavelength_m,
        response=response,
        observable=observable,
        photon_weighted=photon_weighted,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        include_reflection=include_reflection,
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo_bounds=geometric_albedo_bounds,
        model_sigma=model_sigma,
        model_sigma_bounds=model_sigma_bounds,
        sample_prefix=name,
    )
    observation_inputs = BandpassObservationInputs(
        value=jnp.asarray(value),
        sigma=jnp.asarray(sigma),
    )
    return BandpassConstraintBundle(
        name=name,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


class _StepTimer:
    def __init__(self, label: str):
        self.label = label
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.start
        status = "failed after" if exc_type is not None else "completed in"
        print(f"  {self.label} {status} {elapsed:.2f}s")
        return False


@dataclass(frozen=True)
class PreparedRetrieval:
    """Model and resolved components shared by inference and output generation."""

    model_c: object
    model_inputs: dict
    model_params: dict
    atmosphere_region_lookup: dict[str, object]
    shared_region_config: object
    spectroscopic_components: list[SpectroscopicComponentBundle]
    scalar_constraints: list[BandpassConstraintBundle]


def prepare_retrieval(
    *,
    mode: str,
    epochs: Sequence[str],
    data_format: str,
    pt_profile: str | None,
    chemistry_model: str | None,
    fastchem_parameter_file: str | None,
    wav_obs: np.ndarray | None,
    data: np.ndarray | None,
    sigma: np.ndarray | None,
    phase: np.ndarray | None,
    U: np.ndarray | None,
    sysrem_inputs: SysremInputBundle | None,
    joint_spectra: list[dict[str, Any]] | None,
    bandpass_constraints: list[dict[str, Any]] | None,
    atmosphere_regions: list[dict[str, Any]] | None,
    phoenix_cache_dir: str | Path | None,
    sigma_scale: float,
    spectral_stride: int,
    spectral_offset: int,
    apply_sysrem_override: bool | None,
    emission_selection: str | None,
    shared_prior_modes: dict[str, str] | None,
    primary_atomic_species: dict[str, dict] | None,
    primary_molpath_hitemp: dict[str, str | Path] | None,
    primary_molpath_exomol: dict[str, str | Path] | None,
    model_param_overrides: dict[str, float] | None,
    output_dir: str | Path,
    reference_pressure_bar: float | None,
) -> PreparedRetrieval:
    """Load observations and construct the same joint model used by inference."""
    primary_epoch = epochs[0] if epochs else None
    params = dict(config_utils.get_params())
    if model_param_overrides:
        params.update(model_param_overrides)
    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")

    apply_sysrem_default = config.APPLY_SYSREM_DEFAULT if apply_sysrem_override is None else apply_sysrem_override
    apply_sysrem = bool(apply_sysrem_default and data_format == "timeseries")
    primary_subtract_weighted_global_mean = bool(
        data_format == "spectrum"
        and (
            mode == "transmission"
            or (mode == "emission" and emission_selection is not None)
        )
    )

    primary_sysrem: SysremInputBundle | None = sysrem_inputs
    primary_frozen_timeseries_operator: FrozenTimeseriesOperatorSpec | None = None
    primary_pre_sysrem_data: np.ndarray | None = None
    primary_pre_sysrem_sigma: np.ndarray | None = None
    primary_collapsed_emission: CollapsedEmissionInputs | None = None
    primary_collapsed_transmission: CollapsedTransmissionInputs | None = None
    primary_wav_obs_before_thinning: np.ndarray | None = None
    primary_grid_source_wavelength_range: tuple[float, float] | None = None

    full_arm_mode = (
        config.OBSERVING_MODE == "full"
        and all(val is None for val in (wav_obs, data, sigma, phase))
    )
    primary_component_name = "spectroscopy_red" if full_arm_mode else "spectroscopy"

    if full_arm_mode:
        arm_dirs = (
            config_utils.get_full_arm_timeseries_dirs(
                epoch=primary_epoch,
                mode=mode,
            )
            if data_format == "timeseries"
            else config_utils.get_full_arm_data_dirs(
                epoch=primary_epoch,
                mode=mode,
            )
        )
        if mode == "emission" and emission_selection is not None:
            arm_dirs = {
                arm: config_utils.get_collapsed_emission_dir(
                    epoch=primary_epoch,
                    arm=arm,
                    selection=emission_selection,
                )
                for arm in config.FULL_ARM_MEMBERS
            }
        elif mode == "transmission" and data_format == "spectrum":
            arm_dirs = {
                arm: config_utils.get_collapsed_transmission_dir(
                    epoch=primary_epoch,
                    arm=arm,
                )
                for arm in config.FULL_ARM_MEMBERS
            }
        blue_component_spec = {
            "name": "spectroscopy_blue",
            "mode": mode,
            "data_format": data_format,
            "data_dir": str(arm_dirs["blue"]),
            "radial_velocity_mode": "orbital" if data_format == "timeseries" else "none",
            "subtract_weighted_global_mean": bool(
                data_format == "spectrum"
                and (
                    mode == "transmission"
                    or (
                        mode == "emission"
                        and emission_selection is not None
                    )
                )
            ),
            "instrument_resolution": config_utils.get_resolution(),
            "atomic_species": primary_atomic_species,
            "molpath_hitemp": primary_molpath_hitemp,
            "molpath_exomol": primary_molpath_exomol,
        }
        if apply_sysrem_override is not None and data_format == "timeseries":
            blue_component_spec["apply_sysrem"] = bool(apply_sysrem_override)
        joint_spectra = list(joint_spectra or []) + [blue_component_spec]

    print("\n[1/7] Loading time-series data...")
    with _StepTimer("Step 1/7"):
        if epochs:
            if len(epochs) == 1:
                print(f"  Using epoch: {primary_epoch}")
            else:
                print(f"  Using epochs: {', '.join(epochs)}")
        if any(val is not None for val in (wav_obs, data, sigma, phase)):
            phase = _normalize_phase(phase)
            print(f"  Using provided data: {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
            if apply_sysrem:
                if primary_sysrem is None:
                    raw_sysrem = {"U": U}
                    primary_sysrem = _build_sysrem_inputs(raw_sysrem)
                print(f"  Using provided SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
        else:
            if full_arm_mode:
                resolved_data_dir = arm_dirs["red"]
                print(
                    "  OBSERVING_MODE='full': loading red as primary, blue as "
                    "a second spectroscopic component."
                )
                data_paths = (
                    config_utils.get_transmission_paths(
                        epoch=primary_epoch,
                        arm="red",
                        collapsed=data_format == "spectrum",
                    )
                    if mode == "transmission"
                    else config_utils.get_emission_paths(
                        epoch=primary_epoch,
                        arm="red",
                        selection=emission_selection,
                    )
                )
            else:
                if mode == "emission" and emission_selection is not None:
                    resolved_data_dir = config_utils.get_collapsed_emission_dir(
                        epoch=primary_epoch,
                        arm=config.OBSERVING_MODE,
                        selection=emission_selection,
                    )
                else:
                    if mode == "transmission" and data_format == "spectrum":
                        resolved_data_dir = (
                            config_utils.get_collapsed_transmission_dir(
                                epoch=primary_epoch,
                                arm=config.OBSERVING_MODE,
                            )
                        )
                    else:
                        resolved_data_dir = (
                            config_utils.get_timeseries_data_dir(
                                epoch=primary_epoch,
                                mode=mode,
                            )
                            if data_format == "timeseries"
                            else config_utils.get_data_dir(
                                epoch=primary_epoch,
                                mode=mode,
                            )
                        )
                data_paths = (
                    config_utils.get_transmission_paths(
                        epoch=primary_epoch,
                        collapsed=data_format == "spectrum",
                    ) if mode == "transmission"
                    else config_utils.get_emission_paths(
                        epoch=primary_epoch,
                        selection=emission_selection,
                    )
                )

            if data_format == "timeseries":
                wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
                primary_frozen_timeseries_operator = _load_frozen_timeseries_operator_spec(
                    resolved_data_dir
                )
                apply_sysrem = primary_frozen_timeseries_operator.has_sysrem
                pre_sysrem_bundle = load_pre_sysrem_timeseries_data(resolved_data_dir)
                if pre_sysrem_bundle is not None:
                    primary_pre_sysrem_data, primary_pre_sysrem_sigma = pre_sysrem_bundle
                phase = _normalize_phase(phase)
                print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
                print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
                if apply_sysrem:
                    primary_sysrem = _build_sysrem_inputs(_load_sysrem_inputs(resolved_data_dir))
                    print(f"  Loaded SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
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

        primary_wav_obs_before_thinning = np.asarray(wav_obs, dtype=float).copy()
        (
            primary_observed_min,
            primary_observed_max,
            _,
            _,
        ) = _resolve_model_wavelength_range(primary_wav_obs_before_thinning)
        primary_grid_source_wavelength_range = (
            primary_observed_min,
            primary_observed_max,
        )

        if primary_pre_sysrem_data is not None and primary_pre_sysrem_sigma is not None:
            primary_pre_sysrem_data = np.asarray(primary_pre_sysrem_data)
            primary_pre_sysrem_sigma = np.asarray(primary_pre_sysrem_sigma)

        primary_frozen_subset_indices = (
            None
            if primary_frozen_timeseries_operator is None
            or (spectral_stride == 1 and spectral_offset == 0)
            else _spectral_subset_indices(
                np.asarray(wav_obs).size,
                spectral_stride,
                spectral_offset,
            )
        )
        wav_obs, data, sigma, primary_sysrem, primary_pre_sysrem_data, primary_pre_sysrem_sigma = _apply_spectral_thinning(
            wav_obs=wav_obs,
            data=data,
            sigma=sigma,
            sysrem=primary_sysrem,
            pre_sysrem_data=primary_pre_sysrem_data,
            pre_sysrem_sigma=primary_pre_sysrem_sigma,
            spectral_stride=spectral_stride,
            spectral_offset=spectral_offset,
            component_name=primary_component_name,
        )
        primary_frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
            primary_frozen_timeseries_operator,
            primary_frozen_subset_indices,
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
            if primary_pre_sysrem_data is not None and primary_pre_sysrem_sigma is not None:
                if primary_pre_sysrem_data.ndim == 2:
                    primary_pre_sysrem_data = primary_pre_sysrem_data[:, sort_idx]
                    primary_pre_sysrem_sigma = primary_pre_sysrem_sigma[:, sort_idx]
                else:
                    primary_pre_sysrem_data = primary_pre_sysrem_data[sort_idx]
                    primary_pre_sysrem_sigma = primary_pre_sysrem_sigma[sort_idx]
            primary_sysrem = _remap_sysrem_wavelength_sort(primary_sysrem, sort_idx)
            primary_frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
                primary_frozen_timeseries_operator,
                sort_idx,
            )

        primary_frozen_timeseries = _build_model_frozen_timeseries(
            primary_frozen_timeseries_operator,
            primary_sysrem,
        )

        if mode == "emission" and emission_selection is not None:
            primary_collapsed_emission = _load_collapsed_emission_operator(
                resolved_data_dir,
                wav_obs,
            )
        elif mode == "transmission" and data_format == "spectrum":
            primary_collapsed_transmission = (
                _load_collapsed_transmission_operator(
                    resolved_data_dir,
                    wav_obs,
                )
            )

        sigma = _scale_spectroscopic_sigma(sigma, sigma_scale)
        if primary_subtract_weighted_global_mean:
            data = _subtract_inverse_variance_weighted_constant(data, sigma)
            print("  Subtracted the inverse-variance-weighted constant from the 1D spectrum")
        if primary_pre_sysrem_sigma is not None:
            primary_pre_sysrem_sigma = _scale_spectroscopic_sigma(primary_pre_sysrem_sigma, sigma_scale)
        if sigma_scale != 1.0:
            print(f"  Applied spectroscopic sigma scale: x{sigma_scale:g}")

    # Setup instrumental resolution
    print("\n[2/7] Setting up instrumental resolution...")
    with _StepTimer("Step 2/7"):
        Rinst = config_utils.get_resolution()
        print(f"  Instrument resolving power: R = {Rinst:.0f}")

    nu_grid = None
    sop_rot = None
    sop_inst = None
    opa_cias = None
    opa_mols = None
    opa_atoms = None

    # Setup wavenumber grid
    print("\n[3/7] Building wavenumber grid...")
    with _StepTimer("Step 3/7"):
        if full_arm_mode:
            print(
                "  Full-arm mode: skipping global HRS grid; "
                "red and blue components build arm-specific grids in step 6."
            )
        else:
            observed_min, observed_max, model_min, model_max = (
                _resolve_model_wavelength_range(primary_wav_obs_before_thinning)
            )
            nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
                model_min,
                model_max,
                config.N_SPECTRAL_POINTS,
                unit="AA",
            )
            _preflight_grid_checks(inst_nus, nu_grid)
            print(
                "  Complete observed range before thinning: "
                f"{observed_min:.3f} - {observed_max:.3f} Angstroms"
            )
            print(
                f"  Padded model range: {model_min:.3f} - "
                f"{model_max:.3f} Angstroms"
            )

            rotation_vsini_max = _rotation_operator_vsini_max(
                params.get("v_sini_star") if mode == "emission" else None,
            )
            sop_rot, sop_inst, _ = setup_spectral_operators(
                nu_grid,
                Rinst,
                vsini_max=rotation_vsini_max,
            )
            print("  Spectral operators initialized")
            if rotation_vsini_max > DEFAULT_ROTATION_VSINI_MAX_KMS:
                print(
                    "  Stellar rotation operator support: "
                    f"vsini_max={rotation_vsini_max:g} km/s"
                )

    # Setup primary atmospheric RT geometry
    print("\n[4/7] Initializing primary atmospheric RT...")
    with _StepTimer("Step 4/7"):
        primary_art = _build_art_for_mode(mode)
        pressure_top, pressure_btm = config_utils.get_pressure_bounds_for_mode(mode)
        print(f"  {config.NLAYER} atmospheric layers")
        print(f"  Pressure range: {pressure_top:.1e} - {pressure_btm:.1e} bar")
        print(f"  Temperature range: {config.T_LOW:.0f} - {config.T_HIGH:.0f} K")

    # Load opacities
    print("\n[5/7] Loading opacities...")
    with _StepTimer("Step 5/7"):
        if full_arm_mode:
            print(
                "  Full-arm mode: skipping global opacity preload; "
                "each arm loads opacities on its own grid in step 6."
            )
        else:
            opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
            n_cia = sum(1 for cia in opa_cias.values() if not getattr(cia, "_is_dummy", False))
            if n_cia == 0:
                print("  Loaded 0 CIA sources (no overlap with nu_grid)")
            else:
                print(f"  Loaded {n_cia} CIA sources")

            resolved_primary_hitemp = (
                config.MOLPATH_HITEMP
                if primary_molpath_hitemp is None
                else primary_molpath_hitemp
            )
            resolved_primary_exomol = (
                config.MOLPATH_EXOMOL
                if primary_molpath_exomol is None
                else primary_molpath_exomol
            )
            resolved_primary_atoms = (
                config.ATOMIC_SPECIES
                if primary_atomic_species is None
                else primary_atomic_species
            )
            opa_mols, _molmass_arr = load_molecular_opacities(
                resolved_primary_hitemp,
                resolved_primary_exomol,
                nu_grid,
                config.OPA_LOAD,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                cutwing=config.PREMODIT_CUTWING,
            )
            print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

            # Load atomic opacities (optional, uses Kurucz with auto-download)
            opa_atoms, _atommass_arr = load_atomic_opacities(
                resolved_primary_atoms,
                nu_grid,
                config.OPA_LOAD,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                cutwing=config.PREMODIT_CUTWING,
            )
            if opa_atoms:
                print(f"  Loaded {len(opa_atoms)} atomic species: {list(opa_atoms.keys())}")

    print(f"\n[6/7] Building retrieval model (primary={mode}, default P-T={pt_profile})...")
    print(f"  Chemistry model: {chemistry_model}")
    with _StepTimer("Step 6/7"):
        model_params = _coerce_model_params(params)
        # Persist the resolved radius-pressure convention alongside the other
        # runtime model parameters.  Catalog ``P0`` values remain metadata;
        # transmission uses the explicit adopted reference pressure below.
        model_params["reference_pressure_bar"] = reference_pressure_bar
        primary_region_name = _default_region_name_for_mode(mode)
        if mode == "emission":
            print("  PHOENIX stellar spectrum: auto-fetch/cache")
            print(
                "  PHOENIX surface-flux cache: "
                f"{_normalize_phoenix_cache_dir(phoenix_cache_dir)}"
            )
            if model_params["v_sini_star"] is None or not np.isfinite(
                float(model_params["v_sini_star"])
            ):
                print("  Stellar rotation: skipped (no finite v_sini_star)")
            else:
                print(
                    "  Stellar rotation: "
                    f"v sin(i)={float(model_params['v_sini_star']):g} km/s"
                )
            print(f"  Stellar instrumental profile: Gaussian convolution at R={Rinst:g}")
            print("  Stellar denominator velocity: 0 km/s (stellar-rest frame)")

        primary_is_timeseries = (
            np.asarray(phase).size > 1
            or bool(apply_sysrem)
            or (np.asarray(data).ndim == 2 and np.asarray(data).shape[0] > 1)
        )
        primary_radial_velocity_mode = "orbital" if primary_is_timeseries else "none"
        primary_sample_prefix = (
            primary_component_name
            if (joint_spectra or bandpass_constraints)
            else None
        )

        if full_arm_mode:
            primary_component_spec = _build_in_memory_timeseries_component_spec(
                name=primary_component_name,
                mode=mode,
                wav_obs=wav_obs,
                data=data,
                sigma=sigma,
                phase=phase,
                sysrem=primary_sysrem,
                frozen_timeseries_operator=primary_frozen_timeseries_operator,
                instrument_resolution=Rinst,
                apply_sysrem=apply_sysrem,
                radial_velocity_mode=primary_radial_velocity_mode,
                data_format=data_format,
                subtract_weighted_global_mean=primary_subtract_weighted_global_mean,
                collapsed_emission=primary_collapsed_emission,
                collapsed_transmission=primary_collapsed_transmission,
                region_name=primary_region_name,
                pre_sysrem_data=primary_pre_sysrem_data,
                pre_sysrem_sigma=primary_pre_sysrem_sigma,
                wav_obs_before_thinning=primary_wav_obs_before_thinning,
            )
            primary_component_spec.update(
                {
                    "atomic_species": primary_atomic_species,
                    "molpath_hitemp": primary_molpath_hitemp,
                    "molpath_exomol": primary_molpath_exomol,
                }
            )
            primary_component = _load_joint_spectroscopic_component(
                primary_component_spec,
                default_mode=mode,
                default_tstar=model_params["T_star"],
                default_logg_star=model_params["logg_star"],
                default_metallicity=model_params["Fe_H"],
                default_mstar=model_params["M_star"],
                default_rstar=model_params["R_star"],
                default_stellar_vsini=model_params["v_sini_star"],
                default_stellar_limb_darkening_u1=model_params["gamma1"],
                default_stellar_limb_darkening_u2=model_params["gamma2"],
                default_phoenix_cache_dir=phoenix_cache_dir,
                default_sigma_scale=1.0,
                default_spectral_stride=1,
                default_spectral_offset=0,
            )
        else:
            primary_component = _build_primary_spectroscopic_component(
                name=primary_component_name,
                mode=mode,
                wav_obs=wav_obs,
                grid_source_wavelength_range=primary_grid_source_wavelength_range,
                data=data,
                sigma=sigma,
                phase=phase,
                sysrem=primary_sysrem,
                frozen_timeseries=primary_frozen_timeseries,
                instrument_resolution=Rinst,
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
                stellar_vsini=model_params["v_sini_star"],
                stellar_limb_darkening_u1=model_params["gamma1"],
                stellar_limb_darkening_u2=model_params["gamma2"],
                phoenix_cache_dir=phoenix_cache_dir,
                apply_sysrem=apply_sysrem,
                radial_velocity_mode=primary_radial_velocity_mode,
                subtract_weighted_global_mean=primary_subtract_weighted_global_mean,
                collapsed_emission=primary_collapsed_emission,
                collapsed_transmission=primary_collapsed_transmission,
                sample_prefix=primary_sample_prefix,
                pre_sysrem_data=primary_pre_sysrem_data,
                pre_sysrem_sigma=primary_pre_sysrem_sigma,
            )
        observation_configs: list[object] = [primary_component.observation_config]
        observations_payload: dict[str, object] = {primary_component.name: primary_component.observation_inputs}

        auxiliary_components: list[SpectroscopicComponentBundle] = []
        if joint_spectra:
            for spec in joint_spectra:
                component = _load_joint_spectroscopic_component(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_stellar_vsini=model_params["v_sini_star"],
                    default_stellar_limb_darkening_u1=model_params["gamma1"],
                    default_stellar_limb_darkening_u2=model_params["gamma2"],
                    default_phoenix_cache_dir=phoenix_cache_dir,
                    default_sigma_scale=sigma_scale,
                    default_spectral_stride=spectral_stride,
                    default_spectral_offset=spectral_offset,
                )
                auxiliary_components.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        scalar_constraints: list[BandpassConstraintBundle] = []
        if bandpass_constraints:
            for spec in bandpass_constraints:
                component = _load_bandpass_constraint(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_semi_major_axis_au=model_params["a"],
                    default_phoenix_cache_dir=phoenix_cache_dir,
                )
                scalar_constraints.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        if full_arm_mode:
            orbital_component_names = tuple(
                [primary_component.name]
                + [
                    component.name
                    for component in auxiliary_components
                    if component.observation_config.radial_velocity_mode != "none"
                ]
            )
            print(
                f"  Full-arm mode: sharing one global v_sys across "
                f"{len(orbital_component_names)} spectroscopic components: "
                f"{', '.join(orbital_component_names)}."
            )

        shared_system = build_shared_system_config(
            params=model_params,
            reference_pressure_bar=reference_pressure_bar,
            prior_modes=shared_prior_modes,
        )

        atmosphere_region_configs, atmosphere_region_lookup = _build_atmosphere_regions(
            model_params=model_params,
            primary_mode=mode,
            primary_region_name=primary_region_name,
            primary_art=primary_art,
            observation_configs=observation_configs,
            default_pt_profile=pt_profile,
            default_chemistry_model=chemistry_model,
            default_fastchem_parameter_file=fastchem_parameter_file,
            atmosphere_regions=atmosphere_regions,
        )
        joint_model = create_joint_retrieval_model(
            shared_system=shared_system,
            atmosphere_regions=atmosphere_region_configs,
            observations=tuple(observation_configs),
        )
        model_c = joint_model
        model_inputs = {"observations": observations_payload}
        shared_region_config = atmosphere_region_lookup[primary_region_name]
        shared_pt_profile = shared_region_config.pt_profile
        shared_region_sample_prefix = shared_region_config.sample_prefix
        spectroscopic_components: list[SpectroscopicComponentBundle] = [primary_component]
        spectroscopic_components.extend(auxiliary_components)
        spectroscopic_component_count = len(spectroscopic_components)
        component_names = [component.name for component in spectroscopic_components]
        component_names.extend(component.name for component in scalar_constraints)
        print(
            f"  Joint model created with {len(component_names)} component(s): "
            f"{', '.join(component_names)}"
        )
        print(
            "  Atmosphere regions: "
            + ", ".join(
                f"{region_config.name} "
                f"[{next(cfg.mode for cfg in observation_configs if cfg.region_name == region_config.name)}]"
                for region_config in atmosphere_region_configs
            )
        )

    _append_resolved_spectral_grid_config(output_dir, spectroscopic_components)
    return PreparedRetrieval(
        model_c=model_c,
        model_inputs=model_inputs,
        model_params=model_params,
        atmosphere_region_lookup=atmosphere_region_lookup,
        shared_region_config=shared_region_config,
        spectroscopic_components=spectroscopic_components,
        scalar_constraints=scalar_constraints,
    )
