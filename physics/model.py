"""Shared-system sampling, atmospheric forward models, and joint likelihood assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import config
from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.postproc.specop import SopInstProfile, SopRotation
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import MJ, RJ, Rs
from physics.pt import (
    guillot_profile,
    numpyro_free_temperature,
    numpyro_gradient,
    numpyro_madhu_seager,
    numpyro_pspline_knots_on_art_grid,
    numpyro_gp_temperature,
)
from physics.chemistry import CompositionSolver
from physics.observation_operators import (
    ChunkedSysremInputs,
    CollapsedEmissionInputs,
    CollapsedTransmissionInputs,
    FrozenTimeseriesInputs,
    apply_collapsed_emission_operator,
    apply_collapsed_transmission_operator,
    apply_frozen_timeseries_operator,
    apply_model_pipeline_corrections,
)
from physics.opacity import _compute_opacity_terms, _sum_opacity_terms


RetrievalMode = Literal["transmission", "emission"]


PTProfileMode = Literal[
    "guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"
]


AtmosphereRegionName = Literal["terminator", "dayside"]


RVBehavior = Literal["orbital", "none"]


VelocityOffsetMode = Literal["shared", "region", "species", "none"]


SystemPriorMode = Literal["fixed", "normal", "upper_limit"]


BandpassObservable = Literal["flux_ratio", "eclipse_depth", "radius_ratio", "transit_depth"]


def _run_rt_for_mode(
    *,
    mode: RetrievalMode,
    art: object,
    dtau: jnp.ndarray,
    Tarr_rt: jnp.ndarray,
    mmw_rt: jnp.ndarray,
    radius_btm: float | jnp.ndarray,
    gravity_btm: float | jnp.ndarray,
    nu_grid: jnp.ndarray,
) -> jnp.ndarray:
    if mode == "transmission":
        # ExoJAX defines these inputs at the lower pressure boundary.  For
        # transmission, radius_btm is explicitly the adopted R_ref at
        # P_ref == the configured transmission pressure lower boundary.
        return art.run(dtau, Tarr_rt, mmw_rt, radius_btm, gravity_btm)
    if mode == "emission":
        return art.run(dtau, Tarr_rt, nu_grid)


# the below dataclasses are all configuration objects, they describe how to build and sample the model
# system paramaters that are shared across all atmospheric regions and observation types in a joint retrieval
@dataclass(frozen=True)
class SharedSystemConfig:
    Kp_mean: float
    Kp_std: float
    Kp_bounds: tuple[float, float] | None
    v_sys_mean: float
    v_sys_std: float
    R_ref_mean: float
    R_ref_std: float
    Mp_mean: float
    Mp_std: float
    Mp_upper_3sigma: float | None
    Rstar_mean: float
    Rstar_std: float
    period_day: float
    reference_pressure_bar: float | None = None
    Mp_prior_mode: SystemPriorMode = "upper_limit"
    R_ref_prior_mode: SystemPriorMode = "normal"
    Rstar_prior_mode: SystemPriorMode = "normal"


# TODO: ensure these region-specific data-ignorant params in the dataclass shold be region-specific, or if some of these should be shared between atmopsheric regions
# paramaters that are region-specific (so emission vs. transmission), but shared across all observation types in a joint retrieval.
@dataclass(frozen=True)
class AtmosphereRegionConfig:
    name: str
    art: object
    pt_profile: PTProfileMode
    T_low: float
    T_high: float
    Tirr_std: float | None
    Tint_fixed: float
    kappa_ir_cgs_bounds: tuple[float, float]
    gamma_bounds: tuple[float, float]
    composition_solver: CompositionSolver
    mol_names: tuple[str, ...]
    atom_names: tuple[str, ...]
    mol_masses: jnp.ndarray
    atom_masses: jnp.ndarray
    Tirr_mean: float | None
    sample_prefix: str | None = None
    velocity_offset_mode: VelocityOffsetMode = "shared"
    velocity_offset_species: tuple[str, ...] = ()
    velocity_offset_bounds_kms: tuple[float, float] = (-20.0, 20.0)


# TODO: vice versa from above, ensure these observation-specific data-ignorant params in the dataclass should be observation-specific, or if some of these should be shared between observation types
# parameters that are specific to an observation type (e.g. high-res spectroscopy vs. broadband photometry) for a given atmospheric region
@dataclass(frozen=True)
class SpectroscopicObservationConfig:
    name: str
    region_name: str
    mode: RetrievalMode
    opa_mols: dict[str, OpaPremodit]
    opa_atoms: dict[str, OpaPremodit]
    opa_cias: dict[str, OpaCIA]
    nu_grid: jnp.ndarray
    sop_rot: SopRotation
    sop_inst: SopInstProfile
    inst_nus: jnp.ndarray
    beta_inst: float
    radial_velocity_mode: RVBehavior
    subtract_weighted_global_mean: bool
    apply_sysrem: bool
    Tstar: float | None
    stellar_surface_flux: jnp.ndarray | None = None
    stellar_vsini: float | None = None
    stellar_limb_darkening_u1: float = 0.0
    stellar_limb_darkening_u2: float = 0.0
    sample_prefix: str | None = None


# paramaters for bandpass observation
@dataclass(frozen=True)
class BandpassObservationConfig:
    name: str
    region_name: str
    mode: RetrievalMode
    opa_mols: dict[str, OpaPremodit]
    opa_atoms: dict[str, OpaPremodit]
    opa_cias: dict[str, OpaCIA]
    nu_grid: jnp.ndarray
    wavelength_m: jnp.ndarray
    response: jnp.ndarray
    observable: BandpassObservable
    photon_weighted: bool
    Tstar: float | None
    stellar_surface_flux: jnp.ndarray | None = None
    include_reflection: bool = False
    semi_major_axis_au: float | None = None
    geometric_albedo_bounds: tuple[float, float] | None = None
    model_sigma: float | None = None
    model_sigma_bounds: tuple[float, float] | None = None
    sample_prefix: str | None = None


ObservationConfig = SpectroscopicObservationConfig | BandpassObservationConfig


# the full configuration for a joint retrieval, including shared system parameters, region-specific atmospheric parameters, and observation-specific parameters for all observations included in the retrieval
@dataclass(frozen=True)
class JointRetrievalModelConfig:
    shared_system: SharedSystemConfig
    atmosphere_regions: tuple[AtmosphereRegionConfig, ...]
    observations: tuple[ObservationConfig, ...]


# the below observation input clases are state objects, holding the realized values for a retrieval sample after the configs have been used

# the computed atmospheric state for a given atmospheric region, reconstructed from the posterior samples, and the shared system state for a given retrieval sample
class SharedSystemState(NamedTuple):
    Kp: jnp.ndarray
    v_sys: jnp.ndarray
    Mp: jnp.ndarray
    Rstar: jnp.ndarray
    R_ref: jnp.ndarray
    g_btm: jnp.ndarray


class AtmosphereState(NamedTuple):
    art: object
    Tarr: jnp.ndarray
    g_profile: jnp.ndarray
    mmw_profile: jnp.ndarray
    mmr_mols: dict[str, jnp.ndarray]
    mmr_atoms: dict[str, jnp.ndarray]
    mol_masses: dict[str, jnp.ndarray]
    atom_masses: dict[str, jnp.ndarray]
    vmrH2_profile: jnp.ndarray
    vmrHe_profile: jnp.ndarray
    continuum_vmr_profiles: dict[str, jnp.ndarray]
    velocity_offset_mode: VelocityOffsetMode
    region_velocity_offset_kms: jnp.ndarray
    species_velocity_offsets_kms: dict[str, jnp.ndarray]
    is_valid: jnp.ndarray


class SpectroscopicObservationInputs(NamedTuple):
    data: jnp.ndarray
    sigma: jnp.ndarray
    phase: jnp.ndarray | None = None
    U: jnp.ndarray | None = None
    chunked_sysrem: ChunkedSysremInputs | None = None
    frozen_timeseries: FrozenTimeseriesInputs | None = None
    collapsed_emission: CollapsedEmissionInputs | None = None
    collapsed_transmission: CollapsedTransmissionInputs | None = None


class BandpassObservationInputs(NamedTuple):
    value: jnp.ndarray
    sigma: jnp.ndarray


ObservationInputs = SpectroscopicObservationInputs | BandpassObservationInputs


# "Fe I" -> "Fe"
def _element_from_species(species_name: str) -> str:
    return species_name.split()[0]


# converts e.g. "PEPSI/LBT HRS" -> "PEPSI_LBT_HRS"
def _sanitize_site_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned


def _safe_temperature_profile(
    temperature_raw: jnp.ndarray,
    temperature_min: float,
    temperature_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a safe evaluation copy and whether the raw profile is supported."""
    temperature_raw = jnp.asarray(temperature_raw)
    is_valid = jnp.all(jnp.isfinite(temperature_raw)) & jnp.all(
        (temperature_raw >= temperature_min) & (temperature_raw <= temperature_max)
    )
    temperature_safe = jnp.nan_to_num(
        jnp.clip(temperature_raw, temperature_min, temperature_max),
        nan=temperature_min,
        posinf=temperature_max,
        neginf=temperature_min,
    )
    return temperature_safe, is_valid


def _safe_mmw_profile(
    mmw_raw: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the current RT numerical support limits to one shared MMW copy."""
    mmw_raw = jnp.asarray(mmw_raw)
    is_valid = jnp.all(jnp.isfinite(mmw_raw)) & jnp.all(
        (mmw_raw >= config.MMW_RT_MIN) & (mmw_raw <= config.MMW_RT_MAX)
    )
    # This replacement exists only so JAX can finish evaluating a proposal
    # that receives zero probability. Every downstream calculation uses this
    # same safe profile.
    mmw_safe = jnp.nan_to_num(
        jnp.clip(mmw_raw, config.MMW_RT_MIN, config.MMW_RT_MAX),
        nan=config.MMW_RT_MIN,
        posinf=config.MMW_RT_MAX,
        neginf=config.MMW_RT_MIN,
    )
    return mmw_safe, is_valid


def _safe_nonnegative_profile(
    value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    value = jnp.asarray(value)
    is_valid = jnp.all(jnp.isfinite(value)) & jnp.all(value >= 0.0)
    safe = jnp.nan_to_num(
        jnp.maximum(value, 0.0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return safe, is_valid


# given a retrieval mode, return the default atmospheric region name to use if not specified in the configs
def _default_region_name_for_mode(mode: RetrievalMode) -> AtmosphereRegionName:
    if mode == "transmission":
        return "terminator"
    if mode == "emission":
        return "dayside"


# Compute the planet radial velocity in a stellar-rest wavelength frame.
def planet_rv_kms(
    phase: jnp.ndarray,
    Kp_kms: float,
    v_sys_kms: float,
) -> jnp.ndarray:
    return Kp_kms * jnp.sin(2.0 * jnp.pi * phase) + v_sys_kms


def _sample_temperature_profile(
    region_config: AtmosphereRegionConfig,
    gravity_btm: float | jnp.ndarray,
) -> jnp.ndarray:
    art = region_config.art
    pt_profile = region_config.pt_profile

    if pt_profile == "guillot":
        if (region_config.Tirr_mean is not None) and (region_config.Tirr_std is not None):
            Tirr = numpyro.sample(
                "Tirr",
                dist.TruncatedNormal(region_config.Tirr_mean, region_config.Tirr_std, low=0.0),
            )
        else:
            Tirr = numpyro.sample("Tirr", dist.Uniform(region_config.T_low, region_config.T_high))

        kappa_ir_cgs = numpyro.sample(
            "kappa_ir_cgs",
            dist.LogUniform(*region_config.kappa_ir_cgs_bounds),
        )
        gamma = numpyro.sample("gamma", dist.LogUniform(*region_config.gamma_bounds))

        return guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=gravity_btm,
            Tirr=Tirr,
            Tint=region_config.Tint_fixed,
            kappa_ir_cgs=kappa_ir_cgs,
            gamma=gamma,
        )

    if pt_profile == "isothermal":
        T0 = numpyro.sample("T0", dist.Uniform(region_config.T_low, region_config.T_high))
        return T0 * jnp.ones_like(art.pressure)

    if pt_profile == "gradient":
        return numpyro_gradient(art, region_config.T_low, region_config.T_high)

    if pt_profile == "madhu_seager":
        return numpyro_madhu_seager(art, region_config.T_low, region_config.T_high)

    if pt_profile == "free":
        return numpyro_free_temperature(
            art,
            n_layers=5,
            T_low=region_config.T_low,
            T_high=region_config.T_high,
        )

    if pt_profile == "pspline":
        return numpyro_pspline_knots_on_art_grid(
            art,
            T_low=region_config.T_low,
            T_high=region_config.T_high,
        )

    if pt_profile == "gp":
        return numpyro_gp_temperature(
            art,
            T_low=region_config.T_low,
            T_high=region_config.T_high,
        )


def compute_model_timeseries(
    *,
    mode: RetrievalMode,
    art: object,
    dtau: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    radius_btm: float | jnp.ndarray,
    Rstar: float | jnp.ndarray,
    gravity_btm: float | jnp.ndarray,
    phase: jnp.ndarray,
    Kp: float | jnp.ndarray,
    v_sys: float | jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    inst_nus: jnp.ndarray,
    nu_grid: jnp.ndarray,
    beta_inst: float,
    period_day: float,
    Tstar: float | None = None,
    stellar_surface_flux: jnp.ndarray | None = None,
) -> jnp.ndarray:
    # Temperature and MMW have already been validated and replaced with their
    # single safe evaluation copies in _sample_atmosphere_state.
    Tarr_rt = jnp.asarray(Tarr)
    mmw_rt = jnp.asarray(mmw_profile)
    rt = _run_rt_for_mode(
        mode=mode,
        art=art,
        dtau=dtau,
        Tarr_rt=Tarr_rt,
        mmw_rt=mmw_rt,
        radius_btm=radius_btm,
        gravity_btm=gravity_btm,
        nu_grid=nu_grid,
    )

    # Tidal locking: spin period = orbital period.
    vsini = 2.0 * jnp.pi * radius_btm / (period_day * 86400.0) / 1.0e5
    rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
    rt = sop_inst.ipgauss(rt, beta_inst)

    rv = planet_rv_kms(phase, Kp, v_sys)
    planet_ts = jax.vmap(lambda v: sop_inst.sampling(rt, v, inst_nus))(rv)

    if mode == "transmission":
        # ArtTransPure returns (R_lambda / radius_btm)**2. Convert that directly to
        # transit depth and then to the negative perturbation of normalized
        # stellar flux. The prepared transmission data are flux residuals, not
        # radius-ratio spectra.
        transit_depth_ts = planet_ts * (radius_btm / Rstar) ** 2
        return -transit_depth_ts

    Fs = jnp.asarray(stellar_surface_flux)
    # The component builder has already applied the star-specific rotation and
    # instrumental profile to Fs. Keep that prepared denominator fixed in the
    # stellar-rest frame; only the planet follows the orbital velocity.
    stellar_spectrum = sop_inst.sampling(Fs, 0.0, inst_nus)
    return (
        planet_ts
        / stellar_spectrum[None, :]
        * (radius_btm / Rstar) ** 2
    )


def _normalize_chunked_sysrem_inputs(
    chunked_sysrem: ChunkedSysremInputs,
) -> ChunkedSysremInputs:
    return ChunkedSysremInputs(
        chunk_indices=tuple(
            jnp.asarray(indices, dtype=jnp.int32)
            for indices in chunked_sysrem.chunk_indices
        ),
        U_chunks=tuple(jnp.asarray(U) for U in chunked_sysrem.U_chunks),
        sigma_chunks=tuple(jnp.asarray(sigma) for sigma in chunked_sysrem.sigma_chunks),
    )


def _normalize_spectroscopic_observation_inputs(
    inputs: SpectroscopicObservationInputs,
) -> SpectroscopicObservationInputs:
    data = jnp.asarray(inputs.data)
    sigma = jnp.asarray(inputs.sigma)
    phase = None if inputs.phase is None else jnp.asarray(inputs.phase)
    U = None if inputs.U is None else jnp.asarray(inputs.U)
    chunked_sysrem = inputs.chunked_sysrem
    frozen_timeseries = inputs.frozen_timeseries
    collapsed_emission = inputs.collapsed_emission
    collapsed_transmission = inputs.collapsed_transmission

    if data.ndim == 1:
        data = data[None, :]
        sigma = sigma[None, :]

    if chunked_sysrem is not None:
        chunked_sysrem = _normalize_chunked_sysrem_inputs(
            chunked_sysrem,
        )
    if frozen_timeseries is not None:
        source_phase = jnp.asarray(frozen_timeseries.source_phase)
        active_mask = jnp.asarray(frozen_timeseries.active_exposure_mask)
        fixed_source_model = (
            None
            if frozen_timeseries.fixed_source_model is None
            else jnp.asarray(frozen_timeseries.fixed_source_model)
        )
        selected_indices = jnp.asarray(
            frozen_timeseries.selected_exposure_indices,
            dtype=jnp.int32,
        )
        frozen_timeseries = FrozenTimeseriesInputs(
            source_phase=source_phase,
            active_exposure_mask=active_mask,
            selected_exposure_indices=selected_indices,
            subtract_time_median=jnp.asarray(
                frozen_timeseries.subtract_time_median,
                dtype=bool,
            ),
            chunked_sysrem=(
                None
                if frozen_timeseries.chunked_sysrem is None
                else _normalize_chunked_sysrem_inputs(
                    frozen_timeseries.chunked_sysrem,
                )
            ),
            fixed_source_model=fixed_source_model,
        )
    if collapsed_emission is not None:
        source_wavelength = jnp.asarray(collapsed_emission.source_wavelength)
        source_inst_nus = jnp.asarray(collapsed_emission.source_inst_nus)
        source_phase = jnp.asarray(collapsed_emission.source_phase)
        selected_indices = jnp.asarray(
            collapsed_emission.selected_exposure_indices,
            dtype=jnp.int32,
        )
        shift_left_indices = jnp.asarray(
            collapsed_emission.shift_left_indices,
            dtype=jnp.int32,
        )
        shift_fractions = jnp.asarray(collapsed_emission.shift_fractions)
        coadd_weights = jnp.asarray(collapsed_emission.coadd_weights)
        bin_indices = jnp.asarray(collapsed_emission.bin_indices, dtype=jnp.int32)
        bin_weights = jnp.asarray(collapsed_emission.bin_weights)
        output_wavelength = jnp.asarray(collapsed_emission.output_wavelength)
        output_indices = jnp.asarray(
            collapsed_emission.output_indices,
            dtype=jnp.int32,
        )

        collapsed_emission = CollapsedEmissionInputs(
            source_wavelength=source_wavelength,
            source_inst_nus=source_inst_nus,
            source_phase=source_phase,
            selected_exposure_indices=selected_indices,
            shift_left_indices=shift_left_indices,
            shift_fractions=shift_fractions,
            coadd_weights=coadd_weights,
            bin_indices=bin_indices,
            bin_weights=bin_weights,
            output_wavelength=output_wavelength,
            output_indices=output_indices,
            kp_reference_kms=jnp.asarray(
                collapsed_emission.kp_reference_kms
            ),
            velocity_offset_reference_kms=jnp.asarray(
                collapsed_emission.velocity_offset_reference_kms
            ),
            chunked_sysrem=(
                None
                if collapsed_emission.chunked_sysrem is None
                else _normalize_chunked_sysrem_inputs(
                    collapsed_emission.chunked_sysrem,
                )
            ),
        )
    if collapsed_transmission is not None:
        source_wavelength = jnp.asarray(
            collapsed_transmission.source_wavelength
        )
        source_inst_nus = jnp.asarray(
            collapsed_transmission.source_inst_nus
        )
        source_phase = jnp.asarray(collapsed_transmission.source_phase)
        fixed_source_model = jnp.asarray(
            collapsed_transmission.fixed_source_model
        )
        active_mask = jnp.asarray(
            collapsed_transmission.active_exposure_mask
        )
        selected_indices = jnp.asarray(
            collapsed_transmission.selected_exposure_indices,
            dtype=jnp.int32,
        )
        shift_left_indices = jnp.asarray(
            collapsed_transmission.shift_left_indices,
            dtype=jnp.int32,
        )
        shift_fractions = jnp.asarray(
            collapsed_transmission.shift_fractions
        )
        coadd_weights = jnp.asarray(
            collapsed_transmission.coadd_weights
        )
        bin_indices = jnp.asarray(
            collapsed_transmission.bin_indices,
            dtype=jnp.int32,
        )
        bin_weights = jnp.asarray(collapsed_transmission.bin_weights)
        output_wavelength = jnp.asarray(
            collapsed_transmission.output_wavelength
        )
        output_indices = jnp.asarray(
            collapsed_transmission.output_indices,
            dtype=jnp.int32,
        )
        collapsed_transmission = CollapsedTransmissionInputs(
            source_wavelength=source_wavelength,
            source_inst_nus=source_inst_nus,
            source_phase=source_phase,
            fixed_source_model=fixed_source_model,
            active_exposure_mask=active_mask,
            selected_exposure_indices=selected_indices,
            shift_left_indices=shift_left_indices,
            shift_fractions=shift_fractions,
            coadd_weights=coadd_weights,
            bin_indices=bin_indices,
            bin_weights=bin_weights,
            output_wavelength=output_wavelength,
            output_indices=output_indices,
            kp_reference_kms=jnp.asarray(
                collapsed_transmission.kp_reference_kms
            ),
            velocity_offset_reference_kms=jnp.asarray(
                collapsed_transmission.velocity_offset_reference_kms
            ),
            chunked_sysrem=(
                None
                if collapsed_transmission.chunked_sysrem is None
                else _normalize_chunked_sysrem_inputs(
                    collapsed_transmission.chunked_sysrem,
                )
            ),
        )

    return SpectroscopicObservationInputs(
        data=data,
        sigma=sigma,
        phase=phase,
        U=U,
        chunked_sysrem=chunked_sysrem,
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )


def _normalize_bandpass_observation_inputs(
    inputs: BandpassObservationInputs,
) -> BandpassObservationInputs:
    value = jnp.asarray(inputs.value)
    sigma = jnp.asarray(inputs.sigma)
    return BandpassObservationInputs(value=value, sigma=sigma)


def _sample_shared_system_state(
    shared_config: SharedSystemConfig,
    *,
    sample_Kp: bool,
    sample_v_sys: bool,
) -> SharedSystemState:
    def _sample_positive_parameter(
        site_name: str,
        *,
        mean: float,
        std: float,
        low: float,
        mode: SystemPriorMode,
        upper: float | None = None,
        upper_low: float | None = None,
    ) -> jnp.ndarray:
        if mode == "fixed":
            value = jnp.asarray(mean)
            numpyro.deterministic(site_name, value)
            return value
        if mode == "upper_limit":
            uniform_low = low if upper_low is None else float(upper_low)
            return numpyro.sample(site_name, dist.Uniform(uniform_low, float(upper)))
        if mode == "normal":
            return numpyro.sample(
                site_name,
                dist.TruncatedNormal(float(mean), float(std), low=low),
            )

    if not sample_Kp:
        Kp = jnp.asarray(shared_config.Kp_mean)
        numpyro.deterministic("Kp", Kp)
    elif shared_config.Kp_bounds is not None:
        Kp = numpyro.sample("Kp", dist.Uniform(*shared_config.Kp_bounds))
    elif shared_config.Kp_std is None or shared_config.Kp_std == 0:
        Kp = jnp.asarray(shared_config.Kp_mean)
        numpyro.deterministic("Kp", Kp)
    else:
        Kp = numpyro.sample(
            "Kp",
            dist.TruncatedNormal(shared_config.Kp_mean, shared_config.Kp_std, low=0.0),
        )
    if sample_v_sys:
        v_sys = numpyro.sample(
            "v_sys",
            dist.Normal(shared_config.v_sys_mean, shared_config.v_sys_std),
        )
    else:
        v_sys = jnp.asarray(0.0)
        numpyro.deterministic("v_sys", v_sys)
    Mp = _sample_positive_parameter(
        "Mp",
        mean=shared_config.Mp_mean,
        std=shared_config.Mp_std,
        low=0.0,
        mode=shared_config.Mp_prior_mode,
        upper=shared_config.Mp_upper_3sigma,
        upper_low=0.5,
    ) * MJ
    Rstar = _sample_positive_parameter(
        "Rstar",
        mean=shared_config.Rstar_mean,
        std=shared_config.Rstar_std,
        low=0.0,
        mode=shared_config.Rstar_prior_mode,
    ) * Rs
    # Keep the NumPyro site name ``Rp`` for compatibility with existing
    # posterior files, while giving the runtime quantity its explicit
    # reference-radius meaning.
    R_ref = _sample_positive_parameter(
        "Rp",
        mean=shared_config.R_ref_mean,
        std=shared_config.R_ref_std,
        low=0.5,
        mode=shared_config.R_ref_prior_mode,
    ) * RJ
    # Use the gravity at the reference/lower-boundary radius of the RT model.
    g_btm = gravity_surface(R_ref / RJ, Mp / MJ)

    return SharedSystemState(
        Kp=Kp,
        v_sys=v_sys,
        Mp=Mp,
        Rstar=Rstar,
        R_ref=R_ref,
        g_btm=g_btm,
    )


def _sample_atmosphere_state(
    region_config: AtmosphereRegionConfig,
    shared_state: SharedSystemState,
    *,
    scope_prefix: str | None = None,
) -> AtmosphereState:
    if scope_prefix is None:
        Tarr_raw = _sample_temperature_profile(region_config, shared_state.g_btm)
        Tarr, temperature_valid = _safe_temperature_profile(
            Tarr_raw,
            region_config.T_low,
            region_config.T_high,
        )
        comp = region_config.composition_solver.sample(
            region_config.mol_names,
            region_config.mol_masses,
            region_config.atom_names,
            region_config.atom_masses,
            region_config.art,
            Tarr=Tarr,
        )
    else:
        with numpyro.handlers.scope(prefix=scope_prefix):
            Tarr_raw = _sample_temperature_profile(region_config, shared_state.g_btm)
            Tarr, temperature_valid = _safe_temperature_profile(
                Tarr_raw,
                region_config.T_low,
                region_config.T_high,
            )
            comp = region_config.composition_solver.sample(
                region_config.mol_names,
                region_config.mol_masses,
                region_config.atom_names,
                region_config.atom_masses,
                region_config.art,
                Tarr=Tarr,
            )

    def _sample_velocity_offsets() -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        mode = region_config.velocity_offset_mode
        bounds = region_config.velocity_offset_bounds_kms
        if mode == "region":
            return numpyro.sample("delta_v", dist.Uniform(*bounds)), {}
        if mode == "species":
            offsets = {
                species: numpyro.sample(
                    f"delta_v_{_sanitize_site_name(species)}",
                    dist.Uniform(*bounds),
                )
                for species in region_config.velocity_offset_species
            }
            return jnp.asarray(0.0), offsets
        if mode in {"shared", "none"}:
            return jnp.asarray(0.0), {}

    if scope_prefix is None:
        region_velocity_offset_kms, species_velocity_offsets_kms = (
            _sample_velocity_offsets()
        )
    else:
        with numpyro.handlers.scope(prefix=scope_prefix):
            region_velocity_offset_kms, species_velocity_offsets_kms = (
                _sample_velocity_offsets()
            )

    mmw_profile, mmw_valid = _safe_mmw_profile(comp.mmw_profile)
    chemistry_valid = jnp.asarray(True)
    mmr_mols = {}
    for i, mol in enumerate(region_config.mol_names):
        profile, profile_valid = _safe_nonnegative_profile(comp.mmr_mols[i])
        mmr_mols[mol] = profile
        chemistry_valid = chemistry_valid & profile_valid
    mmr_atoms = {}
    for i, atom in enumerate(region_config.atom_names):
        profile, profile_valid = _safe_nonnegative_profile(comp.mmr_atoms[i])
        mmr_atoms[atom] = profile
        chemistry_valid = chemistry_valid & profile_valid

    vmrH2_profile, h2_valid = _safe_nonnegative_profile(comp.vmrH2_profile)
    vmrHe_profile, he_valid = _safe_nonnegative_profile(comp.vmrHe_profile)
    chemistry_valid = chemistry_valid & h2_valid & he_valid
    continuum_vmr_profiles = {}
    for species, raw_profile in comp.continuum_vmr_profiles.items():
        profile, profile_valid = _safe_nonnegative_profile(raw_profile)
        continuum_vmr_profiles[species] = profile
        chemistry_valid = chemistry_valid & profile_valid

    g_profile = region_config.art.gravity_profile(
        Tarr,
        mmw_profile,
        shared_state.R_ref,
        shared_state.g_btm,
    )
    mol_masses = {
        species: region_config.mol_masses[i]
        for i, species in enumerate(region_config.mol_names)
    }
    atom_masses = {
        species: region_config.atom_masses[i]
        for i, species in enumerate(region_config.atom_names)
    }

    return AtmosphereState(
        art=region_config.art,
        Tarr=Tarr,
        g_profile=g_profile,
        mmw_profile=mmw_profile,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        mol_masses=mol_masses,
        atom_masses=atom_masses,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        continuum_vmr_profiles=continuum_vmr_profiles,
        velocity_offset_mode=region_config.velocity_offset_mode,
        region_velocity_offset_kms=region_velocity_offset_kms,
        species_velocity_offsets_kms=species_velocity_offsets_kms,
        is_valid=temperature_valid & chemistry_valid & mmw_valid,
    )


def _compute_component_dtau(
    component_config: ObservationConfig,
    atmosphere_state: AtmosphereState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    opacity_terms, opacity_valid = _compute_opacity_terms(
        art=atmosphere_state.art,
        opa_mols=component_config.opa_mols,
        opa_atoms=component_config.opa_atoms,
        opa_cias=component_config.opa_cias,
        nu_grid=component_config.nu_grid,
        Tarr=atmosphere_state.Tarr,
        mmr_mols=atmosphere_state.mmr_mols,
        mmr_atoms=atmosphere_state.mmr_atoms,
        mol_masses=atmosphere_state.mol_masses,
        atom_masses=atmosphere_state.atom_masses,
        vmrH2_profile=atmosphere_state.vmrH2_profile,
        vmrHe_profile=atmosphere_state.vmrHe_profile,
        mmw_profile=atmosphere_state.mmw_profile,
        g=atmosphere_state.g_profile,
        continuum_vmr_profiles=atmosphere_state.continuum_vmr_profiles,
    )
    if atmosphere_state.species_velocity_offsets_kms:
        for species, delta_v_kms in atmosphere_state.species_velocity_offsets_kms.items():
            opacity_terms[species] = jax.vmap(
                lambda layer_dtau: component_config.sop_inst.sampling(
                    layer_dtau,
                    delta_v_kms,
                    component_config.nu_grid,
                )
            )(opacity_terms[species])
    dtau = _sum_opacity_terms(
        opacity_terms,
        atmosphere_state.art,
        component_config.nu_grid,
    )
    return dtau, atmosphere_state.is_valid & opacity_valid


def _compute_native_observable_spectrum(
    *,
    mode: RetrievalMode,
    art: object,
    dtau: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    radius_btm: float | jnp.ndarray,
    Rstar: float | jnp.ndarray,
    gravity_btm: float | jnp.ndarray,
    nu_grid: jnp.ndarray,
    Tstar: float | None = None,
    stellar_surface_flux: jnp.ndarray | None = None,
) -> jnp.ndarray:
    Tarr_rt = jnp.asarray(Tarr)
    mmw_rt = jnp.asarray(mmw_profile)
    rt = _run_rt_for_mode(
        mode=mode,
        art=art,
        dtau=dtau,
        Tarr_rt=Tarr_rt,
        mmw_rt=mmw_rt,
        radius_btm=radius_btm,
        gravity_btm=gravity_btm,
        nu_grid=nu_grid,
    )

    if mode == "transmission":
        # The native transmission observable is physically non-negative.
        return jnp.sqrt(rt) * (radius_btm / Rstar)

    Fs = jnp.asarray(stellar_surface_flux)
    return rt / Fs * (radius_btm / Rstar) ** 2


def _gaussian_log_likelihood(
    data: jnp.ndarray,
    model: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    var = sigma**2
    return -0.5 * jnp.sum(((data - model) ** 2) / var + jnp.log(2.0 * jnp.pi * var))


def _bandpass_weighted_mean(
    spectrum: jnp.ndarray,
    nu_grid: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    response: jnp.ndarray,
    *,
    photon_weighted: bool,
) -> jnp.ndarray:
    # nu_grid is monotonically ascending by exojax convention (ESLOG in wavenumber),
    # so 1/nu_grid is monotonically descending and a simple reverse [::-1] yields the
    # ascending-wavelength ordering that jnp.interp and jnp.trapezoid require. Using
    # explicit reverse slicing avoids jnp.argsort, whose constant-folded sorted index
    # tensor was being materialized by XLA as a compile-time constant and tripping the
    # 31 MB allocation ceiling on the 50k-element nu_grid.
    # wavelength_m / response are pre-sorted at config construction time (see
    # build_bandpass_observation_config), so no argsort is needed here either.
    model_wavelength_m = 1.0e-2 / nu_grid
    wl_model = model_wavelength_m[::-1]
    spec_sorted = spectrum[::-1]

    rsp_interp = jnp.interp(wl_model, wavelength_m, response, left=0.0, right=0.0)
    weights = rsp_interp * wl_model if photon_weighted else rsp_interp
    norm = jnp.trapezoid(weights, wl_model)
    return jnp.trapezoid(spec_sorted * weights, wl_model) / norm


def _transform_bandpass_observable(
    spectrum: jnp.ndarray,
    observable: BandpassObservable,
) -> jnp.ndarray:
    if observable in {"flux_ratio", "eclipse_depth", "radius_ratio"}:
        return spectrum
    if observable == "transit_depth":
        return spectrum**2


def _bandpass_site_prefix(component_config: BandpassObservationConfig) -> str:
    return _sanitize_site_name(component_config.sample_prefix or component_config.name)


def _sample_geometric_albedo(
    component_config: BandpassObservationConfig,
    *,
    site_prefix: str,
) -> jnp.ndarray:
    if not component_config.include_reflection:
        albedo = jnp.asarray(0.0)
        numpyro.deterministic(f"{site_prefix}_geometric_albedo", albedo)
        return albedo

    albedo_low, albedo_high = component_config.geometric_albedo_bounds
    if albedo_low == albedo_high:
        albedo = jnp.asarray(albedo_low)
        numpyro.deterministic(f"{site_prefix}_geometric_albedo", albedo)
        return albedo

    return numpyro.sample(
        f"{site_prefix}_geometric_albedo",
        dist.Uniform(albedo_low, albedo_high),
    )


def _compute_reflected_bandpass_component(
    geometric_albedo: jnp.ndarray,
    Rp_m: jnp.ndarray,
    semi_major_axis_au: float,
) -> jnp.ndarray:
    semi_major_axis_m = semi_major_axis_au * config.AU_M
    rp_over_a = Rp_m / semi_major_axis_m
    return geometric_albedo * rp_over_a**2


def _sample_bandpass_model_sigma(
    component_config: BandpassObservationConfig,
    *,
    site_prefix: str,
) -> jnp.ndarray | None:
    if component_config.model_sigma is not None:
        model_sigma = jnp.asarray(component_config.model_sigma)
        numpyro.deterministic(f"{site_prefix}_model_sigma", model_sigma)
        return model_sigma

    if component_config.model_sigma_bounds is None:
        return None

    sigma_low, sigma_high = component_config.model_sigma_bounds
    if sigma_low == sigma_high:
        model_sigma = jnp.asarray(sigma_low)
        numpyro.deterministic(f"{site_prefix}_model_sigma", model_sigma)
        return model_sigma

    return numpyro.sample(
        f"{site_prefix}_model_sigma",
        dist.Uniform(sigma_low, sigma_high),
    )


def _sample_effective_bandpass_value(
    site_prefix: str,
    model_value: jnp.ndarray,
    model_sigma: jnp.ndarray | None,
) -> jnp.ndarray:
    if model_sigma is None:
        numpyro.deterministic(f"{site_prefix}_effective_model", model_value)
        return model_value

    effective_value = numpyro.sample(
        f"{site_prefix}_effective_model",
        dist.TruncatedNormal(model_value, model_sigma, low=0.0),
    )
    return effective_value


def _evaluate_spectroscopic_component(
    component_config: SpectroscopicObservationConfig,
    observation_inputs: SpectroscopicObservationInputs,
    shared_config: SharedSystemConfig,
    shared_state: SharedSystemState,
    atmosphere_state: AtmosphereState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtau, model_valid = _compute_component_dtau(component_config, atmosphere_state)

    collapsed_emission = observation_inputs.collapsed_emission
    collapsed_transmission = observation_inputs.collapsed_transmission
    frozen_timeseries = observation_inputs.frozen_timeseries
    collapsed_operator = (
        collapsed_emission
        if collapsed_emission is not None
        else collapsed_transmission
    )

    def _resolved_velocity_offset(
        base_offset: jnp.ndarray,
        *,
        use_shared: bool,
    ) -> jnp.ndarray:
        if atmosphere_state.velocity_offset_mode == "shared":
            return shared_state.v_sys if use_shared else base_offset
        if atmosphere_state.velocity_offset_mode == "region":
            return base_offset + atmosphere_state.region_velocity_offset_kms
        if atmosphere_state.velocity_offset_mode in {"species", "none"}:
            return base_offset

    if collapsed_operator is not None:
        phase = collapsed_operator.source_phase
        Kp = collapsed_operator.kp_reference_kms
        v_sys = _resolved_velocity_offset(
            collapsed_operator.velocity_offset_reference_kms,
            use_shared=False,
        )
        model_inst_nus = collapsed_operator.source_inst_nus
    elif frozen_timeseries is not None:
        phase = frozen_timeseries.source_phase
        Kp = shared_state.Kp
        v_sys = _resolved_velocity_offset(jnp.asarray(0.0), use_shared=True)
        model_inst_nus = component_config.inst_nus
    elif component_config.radial_velocity_mode == "none":
        phase = jnp.zeros(
            (observation_inputs.data.shape[0],), dtype=observation_inputs.data.dtype
        )
        Kp = jnp.asarray(0.0)
        v_sys = _resolved_velocity_offset(jnp.asarray(0.0), use_shared=False)
        model_inst_nus = component_config.inst_nus
    else:
        phase = observation_inputs.phase
        Kp = shared_state.Kp
        v_sys = _resolved_velocity_offset(jnp.asarray(0.0), use_shared=True)
        model_inst_nus = component_config.inst_nus

    model_ts = compute_model_timeseries(
        mode=component_config.mode,
        art=atmosphere_state.art,
        dtau=dtau,
        Tarr=atmosphere_state.Tarr,
        mmw_profile=atmosphere_state.mmw_profile,
        radius_btm=shared_state.R_ref,
        Rstar=shared_state.Rstar,
        gravity_btm=shared_state.g_btm,
        phase=phase,
        Kp=Kp,
        v_sys=v_sys,
        sop_rot=component_config.sop_rot,
        sop_inst=component_config.sop_inst,
        inst_nus=model_inst_nus,
        nu_grid=component_config.nu_grid,
        beta_inst=component_config.beta_inst,
        period_day=shared_config.period_day,
        Tstar=component_config.Tstar,
        stellar_surface_flux=component_config.stellar_surface_flux,
    )
    if collapsed_emission is not None:
        model_ts = apply_collapsed_emission_operator(
            model_ts,
            collapsed_emission,
        )
    elif collapsed_transmission is not None:
        model_ts = apply_collapsed_transmission_operator(
            model_ts,
            collapsed_transmission,
        )
    elif frozen_timeseries is not None:
        model_ts = apply_frozen_timeseries_operator(
            model_ts,
            frozen_timeseries,
        )
    model_ts = apply_model_pipeline_corrections(
        model_ts,
        subtract_weighted_global_mean=component_config.subtract_weighted_global_mean,
        apply_sysrem=(
            component_config.apply_sysrem and frozen_timeseries is None
        ),
        sigma=observation_inputs.sigma,
        U=observation_inputs.U,
        chunked_sysrem=observation_inputs.chunked_sysrem,
    )

    lnL = _gaussian_log_likelihood(
        observation_inputs.data,
        model_ts,
        observation_inputs.sigma,
    )
    return lnL, model_valid


def _evaluate_bandpass_component(
    component_config: BandpassObservationConfig,
    observation_inputs: BandpassObservationInputs,
    shared_state: SharedSystemState,
    atmosphere_state: AtmosphereState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    site_prefix = _bandpass_site_prefix(component_config)
    dtau, model_valid = _compute_component_dtau(component_config, atmosphere_state)
    spectrum = _compute_native_observable_spectrum(
        mode=component_config.mode,
        art=atmosphere_state.art,
        dtau=dtau,
        Tarr=atmosphere_state.Tarr,
        mmw_profile=atmosphere_state.mmw_profile,
        radius_btm=shared_state.R_ref,
        Rstar=shared_state.Rstar,
        gravity_btm=shared_state.g_btm,
        nu_grid=component_config.nu_grid,
        Tstar=component_config.Tstar,
        stellar_surface_flux=component_config.stellar_surface_flux,
    )
    observable_spectrum = _transform_bandpass_observable(
        spectrum,
        component_config.observable,
    )
    model_value = _bandpass_weighted_mean(
        observable_spectrum,
        component_config.nu_grid,
        component_config.wavelength_m,
        component_config.response,
        photon_weighted=component_config.photon_weighted,
    )
    thermal_component = model_value
    reflected_component = jnp.asarray(0.0)

    if component_config.mode == "emission":
        numpyro.deterministic(f"{site_prefix}_thermal_component", thermal_component)
        geometric_albedo = _sample_geometric_albedo(
            component_config,
            site_prefix=site_prefix,
        )
        if component_config.include_reflection:
            reflected_component = _compute_reflected_bandpass_component(
                geometric_albedo,
                shared_state.R_ref,
                component_config.semi_major_axis_au,
            )
        numpyro.deterministic(f"{site_prefix}_reflected_component", reflected_component)
        model_value = thermal_component + reflected_component

    numpyro.deterministic(f"{site_prefix}_model", model_value)
    model_sigma = _sample_bandpass_model_sigma(
        component_config,
        site_prefix=site_prefix,
    )
    effective_value = _sample_effective_bandpass_value(
        site_prefix=site_prefix,
        model_value=model_value,
        model_sigma=model_sigma,
    )
    return (
        _gaussian_log_likelihood(
            observation_inputs.value,
            effective_value,
            observation_inputs.sigma,
        ),
        model_valid,
    )


def joint_retrieval_model(
    model_config: JointRetrievalModelConfig,
    observations: dict[str, ObservationInputs],
) -> None:
    multi_observation = len(model_config.observations) > 1
    orbital_region_names = {
        component.region_name
        for component in model_config.observations
        if isinstance(component, SpectroscopicObservationConfig)
        and component.radial_velocity_mode == "orbital"
    }
    sample_Kp = bool(orbital_region_names)
    sample_shared_v_sys = any(
        region.name in orbital_region_names
        and region.velocity_offset_mode == "shared"
        for region in model_config.atmosphere_regions
    )
    shared_state = _sample_shared_system_state(
        model_config.shared_system,
        sample_Kp=sample_Kp,
        sample_v_sys=sample_shared_v_sys,
    )
    region_states = {}
    for region_config in model_config.atmosphere_regions:
        region_states[region_config.name] = _sample_atmosphere_state(
            region_config,
            shared_state,
            scope_prefix=region_config.sample_prefix,
        )

    total_lnL = 0.0
    model_valid = jnp.asarray(True)
    for component_config in model_config.observations:
        component_input = observations[component_config.name]
        if isinstance(component_config, SpectroscopicObservationConfig):
            component_inputs = _normalize_spectroscopic_observation_inputs(component_input)
            component_lnL, component_valid = _evaluate_spectroscopic_component(
                component_config,
                component_inputs,
                model_config.shared_system,
                shared_state,
                region_states[component_config.region_name],
            )
        elif isinstance(component_config, BandpassObservationConfig):
            component_inputs = _normalize_bandpass_observation_inputs(component_input)
            component_lnL, component_valid = _evaluate_bandpass_component(
                component_config,
                component_inputs,
                shared_state,
                region_states[component_config.region_name],
            )

        total_lnL = total_lnL + component_lnL
        model_valid = model_valid & component_valid

        if multi_observation:
            numpyro.deterministic(
                f"logL_{_sanitize_site_name(component_config.name)}",
                component_lnL,
            )

    numpyro.factor("model_valid", jnp.where(model_valid, 0.0, -jnp.inf))
    numpyro.factor("logL", total_lnL)
    numpyro.deterministic("Kp_kms", shared_state.Kp)
    numpyro.deterministic("v_sys_kms", shared_state.v_sys)
    numpyro.deterministic("vsini_kms", 2.0 * jnp.pi * shared_state.R_ref / (model_config.shared_system.period_day * 86400.0) / 1.0e5,
    )


def create_joint_retrieval_model(
    *,
    shared_system: SharedSystemConfig,
    atmosphere_regions: tuple[AtmosphereRegionConfig, ...],
    observations: tuple[ObservationConfig, ...],
) -> Callable:
    model_config = JointRetrievalModelConfig(
        shared_system=shared_system,
        atmosphere_regions=tuple(atmosphere_regions),
        observations=tuple(observations),
    )
    return partial(joint_retrieval_model, model_config)


def build_shared_system_config(
    *,
    params: dict,
    reference_pressure_bar: float | None = None,
    prior_modes: dict[str, SystemPriorMode] | None = None,
) -> SharedSystemConfig:
    prior_modes = dict(prior_modes or {})
    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Kp_bounds = None
    Mp_upper_3sigma = params.get("M_p_upper_3sigma")
    if Kp_low is not None and Kp_high is not None:
        Kp_bounds = (float(Kp_low), float(Kp_high))

    if reference_pressure_bar is not None:
        reference_pressure_bar = float(reference_pressure_bar)

    return SharedSystemConfig(
        Kp_mean=params["Kp"],
        Kp_std=params["Kp_err"],
        Kp_bounds=Kp_bounds,
        v_sys_mean=0.0,
        v_sys_std=10.0,
        R_ref_mean=params["R_p"],
        R_ref_std=params["R_p_err"],
        Mp_mean=params["M_p"],
        Mp_std=params["M_p_err"],
        Mp_upper_3sigma=Mp_upper_3sigma,
        Rstar_mean=params["R_star"],
        Rstar_std=params["R_star_err"],
        period_day=params["period"],
        reference_pressure_bar=reference_pressure_bar,
        Mp_prior_mode=prior_modes.get(
            "Mp",
            "upper_limit" if Mp_upper_3sigma is not None else "normal",
        ),
        R_ref_prior_mode=prior_modes.get("Rp", "normal"),
        Rstar_prior_mode=prior_modes.get("Rstar", "normal"),
    )


def _build_species_metadata(
    mol_names: tuple[str, ...],
    atom_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], jnp.ndarray, jnp.ndarray]:
    mol_mass_values = []
    for mol in mol_names:
        mol_mass_values.append(molinfo.molmass_isotope(mol, db_HIT=False))
    mol_masses = jnp.array(mol_mass_values)
    if atom_names:
        atom_mass_values = []
        for atom in atom_names:
            atom_mass_values.append(molinfo.molmass_isotope(_element_from_species(atom), db_HIT=False))
        atom_masses = jnp.array(atom_mass_values)
    else:
        atom_masses = jnp.zeros((0,))
    return mol_names, atom_names, mol_masses, atom_masses


def build_atmosphere_region_config(
    *,
    mode: RetrievalMode,
    art: object,
    mol_names: tuple[str, ...],
    atom_names: tuple[str, ...],
    pt_profile: PTProfileMode,
    T_low: float | None = None,
    T_high: float | None = None,
    Tirr_mean: float | None = None,
    Tirr_std: float | None = None,
    Tint_fixed: float | None = None,
    kappa_ir_cgs_bounds: tuple[float, float] | None = None,
    gamma_bounds: tuple[float, float] | None = None,
    composition_solver: CompositionSolver,
    name: str | None = None,
    sample_prefix: str | None = None,
    velocity_offset_mode: VelocityOffsetMode = "shared",
    velocity_offset_species: tuple[str, ...] = (),
    velocity_offset_bounds_kms: tuple[float, float] = (-20.0, 20.0),
) -> AtmosphereRegionConfig:
    if T_low is None:
        T_low = config.T_LOW
    if T_high is None:
        T_high = config.T_HIGH
    if Tint_fixed is None:
        Tint_fixed = config.TINT_FIXED
    if kappa_ir_cgs_bounds is None:
        kappa_ir_cgs_bounds = tuple(float(10.0**bound) for bound in config.LOG_KAPPA_IR_BOUNDS)
    if gamma_bounds is None:
        gamma_bounds = tuple(float(10.0**bound) for bound in config.LOG_GAMMA_BOUNDS)

    mol_names, atom_names, mol_masses, atom_masses = _build_species_metadata(
        mol_names,
        atom_names,
    )
    region_name = name or _default_region_name_for_mode(mode)
    velocity_offset_species = tuple(velocity_offset_species)
    velocity_offset_bounds_kms = tuple(float(v) for v in velocity_offset_bounds_kms)

    return AtmosphereRegionConfig(
        name=region_name,
        art=art,
        pt_profile=pt_profile,
        T_low=T_low,
        T_high=T_high,
        Tirr_std=Tirr_std,
        Tint_fixed=Tint_fixed,
        kappa_ir_cgs_bounds=kappa_ir_cgs_bounds,
        gamma_bounds=gamma_bounds,
        composition_solver=composition_solver,
        mol_names=mol_names,
        atom_names=atom_names,
        mol_masses=mol_masses,
        atom_masses=atom_masses,
        Tirr_mean=Tirr_mean,
        sample_prefix=sample_prefix,
        velocity_offset_mode=velocity_offset_mode,
        velocity_offset_species=velocity_offset_species,
        velocity_offset_bounds_kms=velocity_offset_bounds_kms,
    )


def build_spectroscopic_observation_config(
    *,
    name: str,
    region_name: str,
    mode: RetrievalMode,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    instrument_resolution: float,
    inst_nus: jnp.ndarray,
    Tstar: float | None = None,
    stellar_surface_flux: jnp.ndarray | np.ndarray | None = None,
    stellar_vsini: float | None = None,
    stellar_limb_darkening_u1: float | None = None,
    stellar_limb_darkening_u2: float | None = None,
    radial_velocity_mode: RVBehavior = "orbital",
    subtract_weighted_global_mean: bool = False,
    apply_sysrem: bool | None = None,
    sample_prefix: str | None = None,
) -> SpectroscopicObservationConfig:
    if apply_sysrem is None:
        apply_sysrem = config.APPLY_SYSREM_DEFAULT

    nu_grid = jnp.asarray(nu_grid)
    inst_nus = jnp.asarray(inst_nus)
    stellar_surface_flux_arr = None
    if stellar_surface_flux is not None:
        stellar_surface_flux_arr = jnp.asarray(stellar_surface_flux)

    if stellar_vsini is not None:
        stellar_vsini = float(stellar_vsini)
    stellar_limb_darkening_u1 = (
        0.0 if stellar_limb_darkening_u1 is None else float(stellar_limb_darkening_u1)
    )
    stellar_limb_darkening_u2 = (
        0.0 if stellar_limb_darkening_u2 is None else float(stellar_limb_darkening_u2)
    )
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)
    if mode == "emission":
        # These quantities are fixed for an observation component, so prepare
        # the stellar denominator once rather than repeating its convolutions
        # at every likelihood evaluation.
        if stellar_vsini is not None and stellar_vsini != 0.0:
            stellar_surface_flux_arr = sop_rot.rigid_rotation(
                stellar_surface_flux_arr,
                stellar_vsini,
                stellar_limb_darkening_u1,
                stellar_limb_darkening_u2,
            )
        stellar_surface_flux_arr = sop_inst.ipgauss(
            stellar_surface_flux_arr,
            beta_inst,
        )

    return SpectroscopicObservationConfig(
        name=name,
        region_name=region_name,
        mode=mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        inst_nus=inst_nus,
        beta_inst=beta_inst,
        radial_velocity_mode=radial_velocity_mode,
        subtract_weighted_global_mean=subtract_weighted_global_mean,
        apply_sysrem=apply_sysrem,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux_arr,
        stellar_vsini=stellar_vsini,
        stellar_limb_darkening_u1=stellar_limb_darkening_u1,
        stellar_limb_darkening_u2=stellar_limb_darkening_u2,
        sample_prefix=sample_prefix,
    )


def build_bandpass_observation_config(
    *,
    name: str,
    region_name: str,
    mode: RetrievalMode,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    response: jnp.ndarray,
    observable: BandpassObservable,
    photon_weighted: bool = False,
    Tstar: float | None = None,
    stellar_surface_flux: jnp.ndarray | np.ndarray | None = None,
    include_reflection: bool = False,
    semi_major_axis_au: float | None = None,
    geometric_albedo_bounds: tuple[float, float] | None = None,
    model_sigma: float | None = None,
    model_sigma_bounds: tuple[float, float] | None = None,
    sample_prefix: str | None = None,
) -> BandpassObservationConfig:
    if include_reflection and geometric_albedo_bounds is None:
        geometric_albedo_bounds = (0.0, 1.0)

    nu_grid = jnp.asarray(nu_grid)
    wavelength_m = jnp.asarray(wavelength_m)
    response = jnp.asarray(response)
    stellar_surface_flux_arr = None
    if stellar_surface_flux is not None:
        stellar_surface_flux_arr = jnp.asarray(stellar_surface_flux)

    # Pre-sort the bandpass response curve by wavelength so _bandpass_weighted_mean
    # can feed jnp.interp directly without a traced argsort. See the note in that
    # function for why this matters (XLA constant-folding of large argsort outputs).
    band_sort_idx = jnp.argsort(wavelength_m)
    wavelength_m = wavelength_m[band_sort_idx]
    response = response[band_sort_idx]

    return BandpassObservationConfig(
        name=name,
        region_name=region_name,
        mode=mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        wavelength_m=wavelength_m,
        response=response,
        observable=observable,
        photon_weighted=photon_weighted,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux_arr,
        include_reflection=include_reflection,
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo_bounds=geometric_albedo_bounds,
        model_sigma=model_sigma,
        model_sigma_bounds=model_sigma_bounds,
        sample_prefix=sample_prefix,
    )
