from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist


# string literals for the various model configurations
RetrievalMode = Literal["transmission", "emission"]
PTProfileMode = Literal[
    "guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"
]
AtmosphereRegionName = Literal["terminator", "dayside"]
RVBehavior = Literal["orbital", "none"]
VelocityOffsetMode = Literal["shared", "region", "species", "none"]
SystemPriorMode = Literal["fixed", "normal", "upper_limit"]
BandpassObservable = Literal["flux_ratio", "eclipse_depth", "radius_ratio", "transit_depth"]

import config

from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.postproc.specop import SopInstProfile, SopRotation
try:
    from exojax.rt.layeropacity import layer_optical_depth_Hminus
except ImportError:  # pragma: no cover - depends on exojax version
    layer_optical_depth_Hminus = None
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import MJ, RJ, Rs

# P-T profiles
from physics.pt import (
    gradient_profile,
    guillot_profile,
    madhu_seager_profile,
    numpyro_free_temperature,
    numpyro_gradient,
    numpyro_madhu_seager,
    numpyro_pspline_knots_on_art_grid,
    numpyro_gp_temperature,
    pspline_knots_profile_on_grid,
)

# chemistry models
from physics.chemistry import (
    CompositionSolver,
    ConstantVMR,
    FastChemEquilibriumChemistry,
    FastChemHybridChemistry,
    FreeVMR,
)
CIA_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("H2H2", "H2", "H2"),
    ("H2He", "H2", "He"),
)
CONTINUUM_SPECIES_MASSES: dict[str, float] = {
    "H": float(molinfo.molmass_isotope("H", db_HIT=False)),
    "H-": float(molinfo.molmass_isotope("H", db_HIT=False) + 5.48579909065e-4),
    "e-": 5.48579909065e-4,
}


def _debug_nonfinite_array(label: str, value: jnp.ndarray) -> None:
    """Emit a JAX-side debug print if an array becomes non-finite."""
    arr = jnp.asarray(value)
    has_nonfinite = jnp.any(~jnp.isfinite(arr))

    def _print(_):
        finite_mask = jnp.isfinite(arr)
        safe_arr = jnp.where(finite_mask, arr, jnp.nan)
        jax.debug.print(
            "[numerics] {label}: non-finite array detected "
            "(any_nan={any_nan}, any_inf={any_inf}, finite_min={finite_min}, finite_max={finite_max})",
            label=label,
            any_nan=jnp.any(jnp.isnan(arr)),
            any_inf=jnp.any(jnp.isinf(arr)),
            finite_min=jnp.nanmin(safe_arr),
            finite_max=jnp.nanmax(safe_arr),
        )
        return 0

    jax.lax.cond(has_nonfinite, _print, lambda _: 0, operand=None)


def _debug_nonfinite_scalar(label: str, value: jnp.ndarray) -> None:
    """Emit a JAX-side debug print if a scalar becomes non-finite."""
    scalar = jnp.asarray(value)
    has_nonfinite = ~jnp.isfinite(scalar)

    def _print(_):
        jax.debug.print(
            "[numerics] {label}: non-finite scalar detected (value={value})",
            label=label,
            value=scalar,
        )
        return 0

    jax.lax.cond(has_nonfinite, _print, lambda _: 0, operand=None)


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

# allows for joint configuration of spectroscopic and bandpass observations in the same retrieval
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


class ChunkedSysremInputs(NamedTuple):
    chunk_indices: tuple[jnp.ndarray, ...]
    U_chunks: tuple[jnp.ndarray, ...]
    sigma_chunks: tuple[jnp.ndarray, ...]


class FrozenTimeseriesInputs(NamedTuple):
    """Full-exposure operator for an ordinary phase-selected time series."""

    source_phase: jnp.ndarray
    active_exposure_mask: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    subtract_time_median: bool
    chunked_sysrem: ChunkedSysremInputs | None = None


class CollapsedEmissionInputs(NamedTuple):
    """Fixed operator used to turn an emission time series into one 1D spectrum."""

    source_wavelength: jnp.ndarray
    source_inst_nus: jnp.ndarray
    source_phase: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    shift_left_indices: jnp.ndarray
    shift_fractions: jnp.ndarray
    coadd_weights: jnp.ndarray
    bin_indices: jnp.ndarray
    bin_weights: jnp.ndarray
    output_wavelength: jnp.ndarray
    output_indices: jnp.ndarray
    kp_reference_kms: jnp.ndarray
    velocity_offset_reference_kms: jnp.ndarray
    chunked_sysrem: ChunkedSysremInputs | None = None


class CollapsedTransmissionInputs(NamedTuple):
    """Fixed operator used to turn a transmission time series into one spectrum."""

    source_wavelength: jnp.ndarray
    source_inst_nus: jnp.ndarray
    source_phase: jnp.ndarray
    active_exposure_mask: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    shift_left_indices: jnp.ndarray
    shift_fractions: jnp.ndarray
    coadd_weights: jnp.ndarray
    bin_indices: jnp.ndarray
    bin_weights: jnp.ndarray
    output_wavelength: jnp.ndarray
    output_indices: jnp.ndarray
    kp_reference_kms: jnp.ndarray
    velocity_offset_reference_kms: jnp.ndarray
    chunked_sysrem: ChunkedSysremInputs | None = None


class SpectroscopicObservationInputs(NamedTuple):
    data: jnp.ndarray
    sigma: jnp.ndarray
    phase: jnp.ndarray | None = None
    U: jnp.ndarray | None = None
    V: jnp.ndarray | None = None
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
    return cleaned or "component"


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


# "Applying SYSREM not only removes the static-stellar and telluric signals in the data, but also distorts the underlying planetary spectrum. This effect must be accounted for in order to retrieve accurate parameters from the planetary spectra. We follow the methodology of N. P. Gibson et al. (2022) to apply a corresponding distortion to the model spectra. The corrected model is, from Equation (7) of N. P. Gibson et al."
def sysrem_model_distortion(
    M: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the SYSREM-induced distortion to the model matrix.

    Parameters
    ----------
    M : jnp.ndarray
        Model matrix with shape (n_exposures, n_wavelengths).
    U : jnp.ndarray
        SYSREM basis matrix with shape (n_exposures, n_basis). A constant
        offset term is represented by a column of ones.
    V : jnp.ndarray
        Diagonal whitening matrix with entries 1 / sigma and shape
        (n_exposures, n_exposures).

    The filtered model is computed from the SYSREM projection

        M_fit = U ((V U)^T (V U))^-1 (V U)^T (V M)
        M_filtered = M - M_fit

    so the effective least-squares weights are V^T V = diag(1 / sigma^2).
    """
    if U.shape[1] == 0:
        return M

    weighted_basis = V @ U
    weighted_model = V @ M
    gram = weighted_basis.T @ weighted_basis
    rhs = weighted_basis.T @ weighted_model

    # The common phase-binned SYSREM case uses a single basis vector.
    # Avoid routing that 1x1 solve through cuSolver, which has been unstable
    # on some GPU setups despite the problem being just a scalar division.
    if U.shape[1] == 1:
        coeffs = rhs / jnp.clip(gram[0, 0], config.F32_FLOOR_RECIP, None)
    else:
        coeffs = jnp.linalg.solve(gram, rhs)

    M_fit = U @ coeffs
    return M - M_fit


def sysrem_model_distortion_per_pixel(
    M: jnp.ndarray,
    U: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a frozen SYSREM basis with exposure-by-wavelength uncertainties.

    Unlike the legacy projection above, each wavelength column gets its own
    weighted least-squares solve. This preserves the pixel-level uncertainties
    used by preparation instead of reducing a whole wavelength chunk to one
    exposure weight.
    """
    M = jnp.asarray(M)
    U = jnp.asarray(U)
    sigma = jnp.asarray(sigma)
    if U.shape[1] == 0:
        return M
    if sigma.shape != M.shape:
        raise ValueError(
            f"Per-pixel SYSREM sigma shape {sigma.shape} does not match model "
            f"shape {M.shape}."
        )

    weights = 1.0 / jnp.clip(
        sigma,
        config.F32_FLOOR_RECIP,
        None,
    ) ** 2
    if U.shape[1] == 1:
        basis = U[:, 0][:, None]
        denominator = jnp.sum(weights * basis**2, axis=0)
        numerator = jnp.sum(weights * basis * M, axis=0)
        coeffs = numerator / jnp.clip(
            denominator,
            config.F32_FLOOR_RECIP,
            None,
        )
        return M - basis * coeffs[None, :]

    gram = jnp.einsum("eb,ew,ec->wbc", U, weights, U)
    rhs = jnp.einsum("eb,ew,ew->wb", U, weights, M)
    coeffs = jnp.linalg.solve(gram, rhs[..., None])[..., 0]
    fitted = jnp.einsum("eb,wb->ew", U, coeffs)
    return M - fitted


def sysrem_model_distortion_chunked(
    M: jnp.ndarray,
    chunked_sysrem: ChunkedSysremInputs,
) -> jnp.ndarray:
    corrected = M
    for chunk_indices, U_chunk, sigma_chunk in zip(
        chunked_sysrem.chunk_indices,
        chunked_sysrem.U_chunks,
        chunked_sysrem.sigma_chunks,
    ):
        if chunk_indices.shape[0] == 0 or U_chunk.shape[1] == 0:
            continue
        corrected_chunk = sysrem_model_distortion_per_pixel(
            M[:, chunk_indices],
            U_chunk,
            sigma_chunk,
        )
        corrected = corrected.at[:, chunk_indices].set(corrected_chunk)
    return corrected


def check_grid_resolution(
    nu_grid: jnp.ndarray,
    R: float,
    min_samples: float = 4.0,
) -> None:
    dnu = jnp.abs(jnp.diff(nu_grid))
    nu_mid = nu_grid[:-1]
    R_grid_local = nu_mid / dnu
    R_grid = jnp.median(R_grid_local)
    required_R_grid = R * min_samples

    if R_grid < required_R_grid:
        warnings.warn(
            f"\n[WARNING] Grid Under-sampling Detected!\n"
            f"  Instrument Resolution (R): {R}\n"
            f"  Physics Grid Resolution (R_grid): ~{R_grid:.0f}\n"
            f"  Ratio (Grid/Inst): {R_grid/R:.2f} pixels per FWHM.\n"
            f"  Recommended: > {min_samples} pixels per FWHM.\n"
            f"  Your dtau/opacity calculation might be aliased. Regenerate 'nu_grid' with higher resolution.",
            UserWarning
        )
    else:
        print(f"[INFO] Grid check passed: {R_grid/R:.1f} pixels per FWHM (Target R={R}).")


def _sanitize_opacity_term(
    opacity: jnp.ndarray,
    *,
    negative_rtol: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a safe nonnegative term and a scalar validity decision."""
    opacity = jnp.asarray(opacity)
    finite = jnp.isfinite(opacity)
    finite_opacity = jnp.where(finite, opacity, 0.0)
    positive_scale = jnp.max(jnp.where(finite_opacity > 0.0, finite_opacity, 0.0))
    negative_tolerance = negative_rtol * positive_scale
    is_valid = jnp.all(finite) & jnp.all(finite_opacity >= -negative_tolerance)
    safe_opacity = jnp.maximum(finite_opacity, 0.0)
    return safe_opacity, is_valid


def _compute_cia_opacity_terms(
    art: object,
    opa_cias: dict[str, OpaCIA],
    Tarr: jnp.ndarray,
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Compute CIA optical-depth contributions keyed by CIA source name."""
    vmr_profiles = {
        "H2": vmrH2_profile,
        "He": vmrHe_profile,
    }

    cia_terms: dict[str, jnp.ndarray] = {}
    cia_valid = jnp.asarray(True)
    for cia_key, species_x, species_y in CIA_COLLISION_PAIRS:
        cia = opa_cias.get(cia_key)
        if cia is None:
            continue
        logacia_matrix_raw = cia.logacia_matrix(Tarr)
        # exojax's CIA interpolation can emit NaNs when the runtime nu_grid
        # extends beyond the tabulated CIA wavenumber range. Treat CIA as zero
        # outside the database support instead of letting those NaNs poison dtau.
        cia_nu = jnp.asarray(cia.cdb.nucia)
        cia_nu_min = jnp.min(cia_nu)
        cia_nu_max = jnp.max(cia_nu)
        supported_nu = (jnp.asarray(cia.nu_grid) >= cia_nu_min) & (
            jnp.asarray(cia.nu_grid) <= cia_nu_max
        )
        in_support_finite = jnp.all(
            jnp.where(
                supported_nu[None, :],
                jnp.isfinite(logacia_matrix_raw),
                True,
            )
        )
        # Outside the CIA database support, zero opacity is intentional. An
        # invalid value inside support is replaced only for safe evaluation
        # and still marks the proposal invalid.
        logacia_matrix = jnp.where(
            supported_nu[None, :] & jnp.isfinite(logacia_matrix_raw),
            logacia_matrix_raw,
            -jnp.inf,
        )
        vmr_x, vmr_x_valid = _safe_nonnegative_profile(vmr_profiles[species_x])
        vmr_y, vmr_y_valid = _safe_nonnegative_profile(vmr_profiles[species_y])
        dtau_cia = art.opacity_profile_cia(
            logacia_matrix,
            Tarr,
            vmr_x,
            vmr_y,
            mmw_profile[:, None],
            g,
        )
        dtau_cia_safe, term_valid = _sanitize_opacity_term(dtau_cia)
        cia_terms[f"CIA_{cia_key}"] = dtau_cia_safe
        cia_valid = cia_valid & (
            in_support_finite & vmr_x_valid & vmr_y_valid & term_valid
        )

    return cia_terms, cia_valid


def _compute_xs_opacity_terms(
    art: object,
    opa_by_species: dict[str, OpaPremodit],
    Tarr: jnp.ndarray,
    mmr_profiles: dict[str, jnp.ndarray],
    species_masses: dict[str, jnp.ndarray],
    g: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Compute line-opacity optical-depth contributions keyed by species.

    ``mmr_profiles`` contains mass mixing ratios, so ExoJAX requires the
    absorber's molecular/atomic mass in ``opacity_profile_xs``. Atmospheric
    mean molecular weight is the corresponding mass argument only when the
    supplied profile is a volume mixing ratio.
    """
    xs_terms: dict[str, jnp.ndarray] = {}
    xs_valid = jnp.asarray(True)
    for species, mmr_profile in mmr_profiles.items():
        opa = opa_by_species.get(species)
        if opa is None:
            continue
        xsmatrix_raw = opa.xsmatrix(Tarr, art.pressure)
        xsmatrix_finite = jnp.all(jnp.isfinite(xsmatrix_raw))
        xsmatrix_safe = jnp.where(jnp.isfinite(xsmatrix_raw), xsmatrix_raw, 0.0)
        mmr_safe, mmr_valid = _safe_nonnegative_profile(mmr_profile)
        dtau_raw = art.opacity_profile_xs(
            xsmatrix_safe,
            mmr_safe,
            species_masses[species],
            g,
        )
        dtau_safe, term_valid = _sanitize_opacity_term(
            dtau_raw,
            negative_rtol=config.PREMODIT_NEGATIVE_ROUNDOFF_RTOL,
        )
        xs_terms[species] = dtau_safe
        xs_valid = xs_valid & xsmatrix_finite & mmr_valid & term_valid

    return xs_terms, xs_valid


def _compute_continuum_opacity_terms(
    art: object,
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    continuum_vmr_profiles: dict[str, jnp.ndarray],
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Compute hidden continuum opacity terms such as the H- continuum."""
    vmre = continuum_vmr_profiles.get("e-")
    vmrh = continuum_vmr_profiles.get("H")
    if vmre is None or vmrh is None:
        return {}, jnp.asarray(True)
    if layer_optical_depth_Hminus is None:
        return {}, jnp.asarray(False)

    # exojax.layer_optical_depth_Hminus combines layer-only quantities with
    # a (n_layer, n_nu) continuum matrix using raw broadcasting. Provide
    # column vectors here so layer-wise mmw and gravity broadcast correctly.
    mmw_column = jnp.asarray(mmw_profile)
    if mmw_column.ndim == 1:
        mmw_column = mmw_column[:, None]

    gravity_column = jnp.asarray(g)
    if gravity_column.ndim == 1:
        gravity_column = gravity_column[:, None]

    vmre_safe, vmre_valid = _safe_nonnegative_profile(vmre)
    vmrh_safe, vmrh_valid = _safe_nonnegative_profile(vmrh)
    # Keep the small positive abundance floors required by ExoJAX's logarithms.
    vmre_safe = jnp.maximum(vmre_safe, 1e-15)
    vmrh_safe = jnp.maximum(vmrh_safe, 1e-15)

    dtau_hminus_raw = layer_optical_depth_Hminus(
        nu_grid,
        Tarr,
        art.pressure,
        art.dParr,
        vmre_safe,
        vmrh_safe,
        mmw_column,
        gravity_column,
    )
    dtau_hminus, term_valid = _sanitize_opacity_term(dtau_hminus_raw)

    return {"CONT_Hminus": dtau_hminus}, vmre_valid & vmrh_valid & term_valid


def _compute_opacity_terms(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    mol_masses: dict[str, jnp.ndarray],
    atom_masses: dict[str, jnp.ndarray],
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
    continuum_vmr_profiles: dict[str, jnp.ndarray] | None = None,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Compute all opacity terms using a single canonical implementation."""
    opacity_terms, opacity_valid = _compute_cia_opacity_terms(
        art,
        opa_cias,
        Tarr,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
    )
    mol_terms, mol_valid = _compute_xs_opacity_terms(
            art,
            opa_mols,
            Tarr,
            mmr_mols,
            mol_masses,
            g,
        )
    opacity_terms.update(mol_terms)
    atom_terms, atom_valid = _compute_xs_opacity_terms(
            art,
            opa_atoms,
            Tarr,
            mmr_atoms,
            atom_masses,
            g,
        )
    opacity_terms.update(atom_terms)
    continuum_terms, continuum_valid = _compute_continuum_opacity_terms(
            art,
            nu_grid=nu_grid,
            Tarr=Tarr,
            continuum_vmr_profiles={} if continuum_vmr_profiles is None else continuum_vmr_profiles,
            mmw_profile=mmw_profile,
            g=g,
        )
    opacity_terms.update(continuum_terms)
    opacity_valid = opacity_valid & mol_valid & atom_valid & continuum_valid
    return opacity_terms, opacity_valid


def _sum_opacity_terms(
    opacity_terms: dict[str, jnp.ndarray],
    art: object,
    nu_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Sum all opacity terms into total dtau."""
    dtau = jnp.zeros((art.pressure.size, nu_grid.size))
    for term_name, dtau_term in opacity_terms.items():
        _debug_nonfinite_array(f"opacity_terms[{term_name}]", dtau_term)
        dtau = dtau + dtau_term
        _debug_nonfinite_array(f"opacity_cumulative[{term_name}]", dtau)
    return dtau


def compute_opacity(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    mol_masses: dict[str, jnp.ndarray],
    atom_masses: dict[str, jnp.ndarray],
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
    continuum_vmr_profiles: dict[str, jnp.ndarray] | None = None,
    return_validity: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    opacity_terms, opacity_valid = _compute_opacity_terms(
        art,
        opa_mols,
        opa_atoms,
        opa_cias,
        nu_grid,
        Tarr,
        mmr_mols,
        mmr_atoms,
        mol_masses,
        atom_masses,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
        continuum_vmr_profiles,
    )
    dtau = _sum_opacity_terms(opacity_terms, art, nu_grid)
    if return_validity:
        return dtau, opacity_valid
    return dtau


def compute_opacity_per_species(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    mol_masses: dict[str, jnp.ndarray],
    atom_masses: dict[str, jnp.ndarray],
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
    nu_grid: jnp.ndarray,
    continuum_vmr_profiles: dict[str, jnp.ndarray] | None = None,
) -> dict[str, jnp.ndarray]:
    opacity_terms, _ = _compute_opacity_terms(
        art,
        opa_mols,
        opa_atoms,
        opa_cias,
        nu_grid,
        Tarr,
        mmr_mols,
        mmr_atoms,
        mol_masses,
        atom_masses,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
        continuum_vmr_profiles,
    )
    return opacity_terms


_MISSING = object()


def _extract_scoped_site_params(
    posterior_params: dict,
    sample_prefix: str | None,
) -> dict:
    if sample_prefix is None:
        unscoped = {}
        for key, value in posterior_params.items():
            if "/" not in key:
                unscoped[key] = value
        return unscoped

    prefix = f"{sample_prefix}/"
    scoped = {}
    for key, value in posterior_params.items():
        if key.startswith(prefix):
            scoped[key[len(prefix):]] = value
    if scoped:
        return scoped
    unscoped = {}
    for key, value in posterior_params.items():
        if "/" not in key:
            unscoped[key] = value
    return unscoped


def _posterior_site_value(
    posterior_params: dict,
    site_name: str,
    *,
    local_params: dict | None = None,
    default: object = _MISSING,
):
    if local_params is not None and site_name in local_params:
        return local_params[site_name]
    if site_name in posterior_params:
        return posterior_params[site_name]
    if default is not _MISSING:
        return default
    return posterior_params[site_name]


def _collect_indexed_site_values(
    local_params: dict,
    *,
    prefix: str,
) -> list[object]:
    indexed_values: list[tuple[int, object]] = []
    for key, value in local_params.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        indexed_values.append((int(suffix), value))
    return [value for _, value in sorted(indexed_values)]


def _summarize_posterior_samples(
    posterior_samples: dict,
    *,
    use_median: bool,
) -> dict:
    reducer = np.median if use_median else np.mean
    summary = {}
    for key, values in posterior_samples.items():
        if key.startswith("_"):
            continue
        arr = np.asarray(values)
        if arr.ndim == 0:
            reduced = arr
        else:
            reduced = reducer(arr, axis=0)
        if np.ndim(reduced) == 0:
            summary[key] = float(reduced)
        else:
            summary[key] = np.asarray(reduced)
    return summary


def reconstruct_temperature_profile(
    posterior_params: dict,
    art: object,
    pt_profile: str = "gp",
    Tint_fixed: float = 100.0,
    sample_prefix: str | None = None,
) -> jnp.ndarray:
    local_params = _extract_scoped_site_params(posterior_params, sample_prefix)

    if pt_profile == "guillot":
        Tirr = _posterior_site_value(posterior_params, "Tirr", local_params=local_params)
        kappa_ir_cgs = _posterior_site_value(
            posterior_params,
            "kappa_ir_cgs",
            local_params=local_params,
        )
        gamma = _posterior_site_value(posterior_params, "gamma", local_params=local_params)

        # ``Rp`` is the legacy posterior site name; semantically it is the
        # sampled reference radius used at the RT lower boundary.
        R_ref = _posterior_site_value(
            posterior_params,
            "Rp",
            default=config.DEFAULT_POSTERIOR_RP,
        )
        Mp = _posterior_site_value(
            posterior_params,
            "Mp",
            default=config.DEFAULT_POSTERIOR_MP,
        )
        g_btm = gravity_surface(R_ref, Mp)

        return guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_btm,
            Tirr=Tirr,
            Tint=Tint_fixed,
            kappa_ir_cgs=kappa_ir_cgs,
            gamma=gamma,
        )

    if pt_profile == "isothermal":
        T0 = _posterior_site_value(posterior_params, "T0", local_params=local_params)
        return jnp.asarray(T0) * jnp.ones_like(art.pressure)

    if pt_profile == "gradient":
        T_bottom = _posterior_site_value(
            posterior_params,
            "T_bottom",
            local_params=local_params,
            default=None,
        )
        if T_bottom is None:
            T_bottom = _posterior_site_value(
                posterior_params,
                "T_btm",
                local_params=local_params,
            )
        T_top = _posterior_site_value(posterior_params, "T_top", local_params=local_params)
        return gradient_profile(art, T_bottom, T_top)

    if pt_profile == "madhu_seager":
        T_deep = _posterior_site_value(posterior_params, "T_deep", local_params=local_params)
        T_high = _posterior_site_value(posterior_params, "T_high", local_params=local_params)
        log_P_trans = _posterior_site_value(
            posterior_params,
            "log_P_trans",
            local_params=local_params,
        )
        delta_P = _posterior_site_value(posterior_params, "delta_P", local_params=local_params)
        return madhu_seager_profile(
            art,
            T_deep,
            T_high,
            jnp.power(10.0, log_P_trans),
            delta_P,
        )

    if pt_profile == "free":
        node_values = _collect_indexed_site_values(local_params, prefix="T_node_")
        log_p = jnp.log10(art.pressure)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), len(node_values))
        return jnp.interp(log_p, log_p_nodes, jnp.asarray(node_values))

    if pt_profile == "pspline":
        knot_values = _collect_indexed_site_values(local_params, prefix="T_")
        return pspline_knots_profile_on_grid(
            pressure_bar=art.pressure,
            T_knots=jnp.asarray(knot_values),
            pressure_eval_bar=art.pressure,
        )

    if pt_profile == "gp":
        Tarr = _posterior_site_value(posterior_params, "Tarr", local_params=local_params)
        Tarr = jnp.asarray(Tarr)
        if Tarr.shape != art.pressure.shape:
            raise ValueError(
                f"GP temperature sample shape {Tarr.shape} does not match "
                f"art.pressure shape {art.pressure.shape}."
            )
        return Tarr

    return None


def reconstruct_vmr_scalars(
    posterior_params: dict,
    mol_names: list[str],
    atom_names: list[str],
    sample_prefix: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    local_params = _extract_scoped_site_params(posterior_params, sample_prefix)
    vmr_mols = {}
    for mol in mol_names:
        key = f"logVMR_{mol}"
        if key in local_params:
            logVMR = local_params[key]
            vmr_mols[mol] = float(jnp.power(10.0, logVMR))

    vmr_atoms = {}
    for atom in atom_names:
        key = f"logVMR_{atom}"
        if key in local_params:
            logVMR = local_params[key]
            vmr_atoms[atom] = float(jnp.power(10.0, logVMR))

    return vmr_mols, vmr_atoms


def reconstruct_vmr_profiles(
    posterior_params: dict,
    mol_names: list[str],
    atom_names: list[str],
    art: object,
    *,
    sample_prefix: str | None = None,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    local_params = _extract_scoped_site_params(posterior_params, sample_prefix)
    log_p = jnp.log10(art.pressure)

    def _profile_for_species(name: str) -> jnp.ndarray | None:
        log_vmr_nodes = _collect_indexed_site_values(
            local_params,
            prefix=f"logVMR_{name}_node",
        )
        if not log_vmr_nodes:
            return None

        log_vmr_nodes_arr = jnp.asarray(log_vmr_nodes)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), log_vmr_nodes_arr.size)
        log_vmr_profile = jnp.interp(log_p, log_p_nodes, log_vmr_nodes_arr)
        return jnp.power(10.0, log_vmr_profile)

    vmr_mols = {}
    for mol in mol_names:
        profile = _profile_for_species(mol)
        if profile is not None:
            vmr_mols[mol] = profile
    vmr_atoms = {}
    for atom in atom_names:
        profile = _profile_for_species(atom)
        if profile is not None:
            vmr_atoms[atom] = profile

    all_profiles = [*vmr_mols.values(), *vmr_atoms.values()]
    if all_profiles:
        stacked = jnp.stack(all_profiles, axis=0)
        sum_trace = jnp.sum(stacked, axis=0)
        scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
        scaled = stacked * scale[None, :]

        mol_keys = list(vmr_mols)
        atom_keys = list(vmr_atoms)
        mol_count = len(mol_keys)
        vmr_mols = {}
        for i, mol in enumerate(mol_keys):
            vmr_mols[mol] = scaled[i]
        vmr_atoms = {}
        for i, atom in enumerate(atom_keys):
            vmr_atoms[atom] = scaled[mol_count + i]

    return vmr_mols, vmr_atoms


def reconstruct_fastchem_hybrid_profiles(
    posterior_params: dict,
    composition_solver: FastChemHybridChemistry,
    mol_names: list[str],
    atom_names: list[str],
    art: object,
    Tarr: jnp.ndarray,
    *,
    sample_prefix: str | None = None,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    local_params = _extract_scoped_site_params(posterior_params, sample_prefix)
    has_hybrid_params = composition_solver.requires_hybrid_parameters()
    if has_hybrid_params:
        log_metallicity = _posterior_site_value(
            posterior_params,
            "log_metallicity",
            local_params=local_params,
        )
        co_ratio = _posterior_site_value(
            posterior_params,
            "C_O_ratio",
            local_params=local_params,
        )
    else:
        log_metallicity = None
        co_ratio = None

    vmr_mols_scalar, vmr_atoms_scalar = reconstruct_vmr_scalars(
        posterior_params,
        mol_names,
        atom_names,
        sample_prefix=sample_prefix,
    )
    n_layers = art.pressure.size
    log_P = jnp.log10(art.pressure)

    mol_profile_map = {}
    for name in mol_names:
        mol_profile_map[name] = jnp.full(n_layers, config.TRACE_SPECIES_FLOOR)
    atom_profile_map = {}
    for name in atom_names:
        atom_profile_map[name] = jnp.full(n_layers, config.TRACE_SPECIES_FLOOR)
    continuum_profile_map = {}
    for species in composition_solver.hidden_continuum_species():
        continuum_profile_map[species] = jnp.full(n_layers, config.TRACE_SPECIES_FLOOR)
    for mol, vmr in vmr_mols_scalar.items():
        mol_profile_map[mol] = jnp.full(n_layers, vmr)
    for atom, vmr in vmr_atoms_scalar.items():
        atom_profile_map[atom] = jnp.full(n_layers, vmr)

    needed = list(continuum_profile_map)
    if needed and has_hybrid_params:
        if composition_solver._hybrid_vmr_grids is None or any(species not in composition_solver._hybrid_vmr_grids for species in needed):
            composition_solver._build_hybrid_grid(np.asarray(art.pressure), needed)

        for species in needed:
            if species not in composition_solver._hybrid_vmr_grids:
                continue
            vmr_profile = composition_solver._interp_4d_log_vmr(
                composition_solver._hybrid_vmr_grids[species],
                jnp.asarray(log_metallicity),
                jnp.asarray(co_ratio),
                Tarr,
                log_P,
            )
            continuum_profile_map[species] = vmr_profile
            for name in mol_profile_map:
                if composition_solver._canonical_species_name(name) == species:
                    mol_profile_map[name] = vmr_profile
            for name in atom_profile_map:
                if composition_solver._canonical_species_name(name) == species:
                    atom_profile_map[name] = vmr_profile

    vmr_mols_profiles = [mol_profile_map[name] for name in mol_names]
    vmr_atoms_profiles = [atom_profile_map[name] for name in atom_names]
    continuum_profiles = [continuum_profile_map[name] for name in continuum_profile_map]
    if vmr_mols_profiles or vmr_atoms_profiles or continuum_profiles:
        all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles + continuum_profiles)
        sum_trace = jnp.sum(all_profiles, axis=0)
        scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
        all_profiles = all_profiles * scale[None, :]
        vmr_mols_profiles = [all_profiles[i] for i in range(len(vmr_mols_profiles))]
        vmr_atoms_profiles = [
            all_profiles[len(vmr_mols_profiles) + i]
            for i in range(len(vmr_atoms_profiles))
        ]
        continuum_offset = len(vmr_mols_profiles) + len(vmr_atoms_profiles)
        continuum_profile_map = {
            name: all_profiles[continuum_offset + i]
            for i, name in enumerate(continuum_profile_map)
        }

    vmr_mols_profiles_dict = {}
    for i, mol in enumerate(mol_names):
        vmr_mols_profiles_dict[mol] = vmr_mols_profiles[i]
    vmr_atoms_profiles_dict = {}
    for i, atom in enumerate(atom_names):
        vmr_atoms_profiles_dict[atom] = vmr_atoms_profiles[i]
    return vmr_mols_profiles_dict, vmr_atoms_profiles_dict, continuum_profile_map


def compute_mmw_and_h2he_from_vmr(
    vmr_mols: dict[str, float],
    vmr_atoms: dict[str, float],
    mol_names: list[str],
    atom_names: list[str],
    h2_he_ratio: float = config.H2_HE_RATIO,
) -> tuple[float, float, float]:
    mol_masses = {}
    for mol in mol_names:
        mol_masses[mol] = molinfo.molmass_isotope(mol, db_HIT=False)
    atom_masses = {}
    for atom in atom_names:
        atom_masses[atom] = molinfo.molmass_isotope(_element_from_species(atom), db_HIT=False)

    vmr_trace_tot = sum(vmr_mols.values()) + sum(vmr_atoms.values())
    vmr_trace_tot = min(max(vmr_trace_tot, 0.0), 1.0)

    h2_frac = h2_he_ratio / (h2_he_ratio + 1.0)
    he_frac = 1.0 / (h2_he_ratio + 1.0)
    vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
    vmrHe = (1.0 - vmr_trace_tot) * he_frac

    mass_H2 = molinfo.molmass_isotope("H2")
    mass_He = molinfo.molmass_isotope("He", db_HIT=False)
    mmw = mass_H2 * vmrH2 + mass_He * vmrHe
    mmw += sum(mol_masses[m] * v for m, v in vmr_mols.items())
    mmw += sum(atom_masses[a] * v for a, v in vmr_atoms.items())

    return mmw, vmrH2, vmrHe


def compute_mmw_and_h2he_from_vmr_profiles(
    vmr_mols: dict[str, jnp.ndarray],
    vmr_atoms: dict[str, jnp.ndarray],
    mol_names: list[str],
    atom_names: list[str],
    *,
    n_layers: int,
    h2_he_ratio: float = config.H2_HE_RATIO,
    extra_vmr_profiles: dict[str, jnp.ndarray] | None = None,
    extra_species_masses: dict[str, float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names}

    profile_values = [
        *vmr_mols.values(),
        *vmr_atoms.values(),
        *({} if extra_vmr_profiles is None else extra_vmr_profiles).values(),
    ]
    if profile_values:
        vmr_trace_tot = jnp.sum(jnp.stack(profile_values, axis=0), axis=0)
    else:
        vmr_trace_tot = jnp.zeros((n_layers,))

    h2_frac = h2_he_ratio / (h2_he_ratio + 1.0)
    he_frac = 1.0 / (h2_he_ratio + 1.0)
    vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
    vmrHe = (1.0 - vmr_trace_tot) * he_frac

    mass_H2 = molinfo.molmass_isotope("H2")
    mass_He = molinfo.molmass_isotope("He", db_HIT=False)
    mmw = mass_H2 * vmrH2 + mass_He * vmrHe
    for mol, vmr_profile in vmr_mols.items():
        mmw = mmw + mol_masses[mol] * vmr_profile
    for atom, vmr_profile in vmr_atoms.items():
        mmw = mmw + atom_masses[atom] * vmr_profile
    if extra_vmr_profiles:
        extra_species_masses = {} if extra_species_masses is None else extra_species_masses
        for species, vmr_profile in extra_vmr_profiles.items():
            mass = extra_species_masses.get(species)
            if mass is None:
                continue
            mmw = mmw + mass * vmr_profile

    return mmw, vmrH2, vmrHe


def convert_vmr_to_mmr_profiles(
    vmr_mols: dict[str, float],
    vmr_atoms: dict[str, float],
    mol_names: list[str],
    atom_names: list[str],
    mmw: float,
    art: object,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names}

    mmr_mols = {}
    for mol, vmr in vmr_mols.items():
        mass = mol_masses[mol]
        mmr = vmr * (mass / mmw)
        mmr_mols[mol] = art.constant_mmr_profile(mmr)

    mmr_atoms = {}
    for atom, vmr in vmr_atoms.items():
        mass = atom_masses[atom]
        mmr = vmr * (mass / mmw)
        mmr_atoms[atom] = art.constant_mmr_profile(mmr)

    return mmr_mols, mmr_atoms


def convert_vmr_profiles_to_mmr_profiles(
    vmr_mols: dict[str, jnp.ndarray],
    vmr_atoms: dict[str, jnp.ndarray],
    mol_names: list[str],
    atom_names: list[str],
    mmw: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names}

    mmr_mols = {}
    for mol, vmr_profile in vmr_mols.items():
        mmr_mols[mol] = vmr_profile * (mol_masses[mol] / mmw)

    mmr_atoms = {}
    for atom, vmr_profile in vmr_atoms.items():
        mmr_atoms[atom] = vmr_profile * (atom_masses[atom] / mmw)

    return mmr_mols, mmr_atoms


def compute_atmospheric_state_from_posterior(
    posterior_samples: dict,
    region_config: AtmosphereRegionConfig,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    use_median: bool = True,
    sample_prefix: str | None = None,
) -> dict:
    art = region_config.art
    params = _summarize_posterior_samples(
        posterior_samples,
        use_median=use_median,
    )

    mol_names = list(region_config.mol_names)
    atom_names = list(region_config.atom_names)
    mol_masses = {
        species: region_config.mol_masses[i]
        for i, species in enumerate(region_config.mol_names)
    }
    atom_masses = {
        species: region_config.atom_masses[i]
        for i, species in enumerate(region_config.atom_names)
    }
    Tarr = reconstruct_temperature_profile(
        params,
        art,
        region_config.pt_profile,
        Tint_fixed=region_config.Tint_fixed,
        sample_prefix=sample_prefix,
    )
    Tarr_np = np.asarray(Tarr, dtype=np.float64)
    if (
        np.any(~np.isfinite(Tarr_np))
        or np.any(Tarr_np < region_config.T_low)
        or np.any(Tarr_np > region_config.T_high)
    ):
        raise ValueError(
            "Posterior temperature profile lies outside the supported range "
            f"[{region_config.T_low}, {region_config.T_high}] K."
        )
    composition_solver = region_config.composition_solver
    continuum_vmr_profiles: dict[str, jnp.ndarray] = {}
    if isinstance(composition_solver, ConstantVMR):
        vmr_mols, vmr_atoms = reconstruct_vmr_scalars(
            params,
            mol_names,
            atom_names,
            sample_prefix=sample_prefix,
        )
        mmw, vmrH2, vmrHe = compute_mmw_and_h2he_from_vmr(
            vmr_mols,
            vmr_atoms,
            mol_names,
            atom_names,
            h2_he_ratio=float(composition_solver.h2_he_ratio),
        )
        mmr_mols, mmr_atoms = convert_vmr_to_mmr_profiles(
            vmr_mols, vmr_atoms, mol_names, atom_names, mmw, art
        )

        vmrH2_profile = art.constant_mmr_profile(vmrH2)
        vmrHe_profile = art.constant_mmr_profile(vmrHe)
        mmw_profile = art.constant_mmr_profile(mmw)
    elif isinstance(composition_solver, FreeVMR):
        vmr_mols_profiles, vmr_atoms_profiles = reconstruct_vmr_profiles(
            params,
            mol_names,
            atom_names,
            art,
            sample_prefix=sample_prefix,
        )
        mmw_profile, vmrH2_profile, vmrHe_profile = compute_mmw_and_h2he_from_vmr_profiles(
            vmr_mols_profiles,
            vmr_atoms_profiles,
            mol_names,
            atom_names,
            n_layers=art.pressure.size,
            h2_he_ratio=float(composition_solver.h2_he_ratio),
        )
        mmr_mols, mmr_atoms = convert_vmr_profiles_to_mmr_profiles(
            vmr_mols_profiles,
            vmr_atoms_profiles,
            mol_names,
            atom_names,
            mmw_profile,
        )
        vmr_mols = {}
        for mol, vmr_profile in vmr_mols_profiles.items():
            vmr_mols[mol] = float(jnp.mean(vmr_profile))
        vmr_atoms = {}
        for atom, vmr_profile in vmr_atoms_profiles.items():
            vmr_atoms[atom] = float(jnp.mean(vmr_profile))
    elif isinstance(composition_solver, FastChemHybridChemistry):
        (
            vmr_mols_profiles_dict,
            vmr_atoms_profiles_dict,
            continuum_vmr_profiles,
        ) = reconstruct_fastchem_hybrid_profiles(
            params,
            composition_solver,
            mol_names,
            atom_names,
            art,
            Tarr,
            sample_prefix=sample_prefix,
        )
        mmw_profile, vmrH2_profile, vmrHe_profile = compute_mmw_and_h2he_from_vmr_profiles(
            vmr_mols_profiles_dict,
            vmr_atoms_profiles_dict,
            mol_names,
            atom_names,
            n_layers=art.pressure.size,
            h2_he_ratio=float(composition_solver.h2_he_ratio),
            extra_vmr_profiles=continuum_vmr_profiles,
            extra_species_masses=CONTINUUM_SPECIES_MASSES,
        )
        mmr_mols, mmr_atoms = convert_vmr_profiles_to_mmr_profiles(
            vmr_mols_profiles_dict,
            vmr_atoms_profiles_dict,
            mol_names,
            atom_names,
            mmw_profile,
        )
        vmr_mols = {}
        for mol, vmr_profile in vmr_mols_profiles_dict.items():
            vmr_mols[mol] = float(jnp.mean(vmr_profile))
        vmr_atoms = {}
        for atom, vmr_profile in vmr_atoms_profiles_dict.items():
            vmr_atoms[atom] = float(jnp.mean(vmr_profile))

    mmw_np = np.asarray(mmw_profile, dtype=np.float64)
    if (
        np.any(~np.isfinite(mmw_np))
        or np.any(mmw_np < config.MMW_RT_MIN)
        or np.any(mmw_np > config.MMW_RT_MAX)
    ):
        raise ValueError(
            "Posterior MMW profile lies outside the current RT numerical support "
            f"[{config.MMW_RT_MIN}, {config.MMW_RT_MAX}]."
        )

    chemistry_profiles = {
        **{f"MMR {name}": profile for name, profile in mmr_mols.items()},
        **{f"MMR {name}": profile for name, profile in mmr_atoms.items()},
        "VMR H2": vmrH2_profile,
        "VMR He": vmrHe_profile,
        **{
            f"continuum VMR {name}": profile
            for name, profile in continuum_vmr_profiles.items()
        },
    }
    for label, profile in chemistry_profiles.items():
        profile_np = np.asarray(profile, dtype=np.float64)
        if np.any(~np.isfinite(profile_np)) or np.any(profile_np < 0.0):
            raise ValueError(f"Posterior reconstruction produced an invalid {label} profile.")

    # Compute gravity profile
    # ``Rp`` remains the persisted posterior site for compatibility; its
    # runtime meaning is the adopted reference radius at the RT lower
    # boundary.
    R_ref = params.get("Rp", config.DEFAULT_POSTERIOR_RP) * RJ
    Mp = params.get("Mp", config.DEFAULT_POSTERIOR_MP) * MJ
    g_btm = gravity_surface(R_ref / RJ, Mp / MJ)
    g = art.gravity_profile(Tarr, mmw_profile, R_ref, g_btm)

    # Compute total dtau (pass MMR profiles for molecules/atoms, VMR for CIA)
    dtau, opacity_valid = compute_opacity(
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        Tarr=Tarr,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        mol_masses=mol_masses,
        atom_masses=atom_masses,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        mmw_profile=mmw_profile,
        g=g,
        continuum_vmr_profiles=continuum_vmr_profiles,
        return_validity=True,
    )
    if not bool(np.asarray(opacity_valid)):
        raise ValueError("Posterior reconstruction produced an invalid opacity term.")

    # Compute per-species dtau
    dtau_per_species = compute_opacity_per_species(
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        Tarr=Tarr,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        mol_masses=mol_masses,
        atom_masses=atom_masses,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        mmw_profile=mmw_profile,
        g=g,
        continuum_vmr_profiles=continuum_vmr_profiles,
    )

    return {
        'dtau': dtau,
        'dtau_per_species': dtau_per_species,
        'Tarr': Tarr,
        'pressure': art.pressure,
        'dParr': art.dParr,
        'mmw': mmw_profile,
        'vmrH2': vmrH2_profile,
        'vmrHe': vmrHe_profile,
        'vmr_mols': vmr_mols,  # scalar VMRs for reference
        'vmr_atoms': vmr_atoms,  # scalar VMRs for reference
        'vmr_continuum': continuum_vmr_profiles,
        'mmr_mols': mmr_mols,  # MMR profiles used in opacity
        'mmr_atoms': mmr_atoms,  # MMR profiles used in opacity
        'params': params,
    }


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

    raise ValueError(f"Unknown P-T profile: {pt_profile}")


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
    # Diagnostic probes before art.run pinpoint any unexpected failure after
    # the shared validation/safe-copy boundary.
    _debug_nonfinite_array("spectroscopy.rt_input.dtau", dtau)
    _debug_nonfinite_array("spectroscopy.rt_input.Tarr_rt", Tarr_rt)
    _debug_nonfinite_array("spectroscopy.rt_input.mmw_rt", mmw_rt)
    _debug_nonfinite_scalar("spectroscopy.rt_input.radius_btm", radius_btm)
    _debug_nonfinite_scalar("spectroscopy.rt_input.gravity_btm", gravity_btm)
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
    _debug_nonfinite_array("spectroscopy.rt", rt)

    # Tidal locking: spin period = orbital period.
    vsini = 2.0 * jnp.pi * radius_btm / (period_day * 86400.0) / 1.0e5
    rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
    rt = sop_inst.ipgauss(rt, beta_inst)

    rv = planet_rv_kms(phase, Kp, v_sys)
    planet_ts = jax.vmap(lambda v: sop_inst.sampling(rt, v, inst_nus))(rv)
    _debug_nonfinite_array("spectroscopy.planet_ts", planet_ts)

    if mode == "transmission":
        # ArtTransPure returns (R_lambda / radius_btm)**2. Convert that directly to
        # transit depth and then to the negative perturbation of normalized
        # stellar flux. The prepared transmission data are flux residuals, not
        # radius-ratio spectra.
        transit_depth_ts = jnp.clip(planet_ts, 0.0, None) * (radius_btm / Rstar) ** 2
        return -transit_depth_ts

    Fs = _resolve_emission_stellar_surface_flux(
        nu_grid=nu_grid,
        stellar_surface_flux=stellar_surface_flux,
        context="compute_model_timeseries",
    )
    # The component builder has already applied the star-specific rotation and
    # instrumental profile to Fs. Keep that prepared denominator fixed in the
    # stellar-rest frame; only the planet follows the orbital velocity.
    stellar_spectrum = sop_inst.sampling(Fs, 0.0, inst_nus)
    return (
        planet_ts
        / jnp.clip(stellar_spectrum[None, :], config.F32_FLOOR_RECIP, None)
        * (radius_btm / Rstar) ** 2
    )


def apply_collapsed_emission_operator(
    model_ts: jnp.ndarray,
    operator: CollapsedEmissionInputs,
) -> jnp.ndarray:
    """Apply the data's time-median subtraction and 1D collapse to a model."""
    model_ts = jnp.asarray(model_ts)
    if model_ts.ndim != 2:
        raise ValueError(
            "A collapsed-emission operator requires a 2D model time series; "
            f"got shape {model_ts.shape}."
        )

    # ExoJAX samples on increasing wavenumber, which is decreasing wavelength.
    # The saved collapse operator uses increasing wavelength, matching NumPy's
    # interpolation convention in the data collapser.
    model_wavelength_order = model_ts[:, ::-1]
    residual_ts = model_wavelength_order - jnp.median(
        model_wavelength_order,
        axis=0,
        keepdims=True,
    )
    if operator.chunked_sysrem is not None:
        residual_ts = sysrem_model_distortion_chunked(
            residual_ts,
            operator.chunked_sysrem,
        )
    selected = residual_ts[operator.selected_exposure_indices]
    exposure_indices = jnp.arange(selected.shape[0])[:, None]
    left = selected[exposure_indices, operator.shift_left_indices]
    right = selected[exposure_indices, operator.shift_left_indices + 1]
    shifted = left + operator.shift_fractions * (right - left)
    spectrum_unbinned = jnp.sum(operator.coadd_weights * shifted, axis=0)
    retained = operator.bin_indices.shape[0]
    spectrum_binned = jnp.zeros_like(operator.output_wavelength).at[
        operator.bin_indices
    ].add(operator.bin_weights * spectrum_unbinned[:retained])
    return spectrum_binned[operator.output_indices][None, :]


def apply_collapsed_transmission_operator(
    model_ts: jnp.ndarray,
    operator: CollapsedTransmissionInputs,
) -> jnp.ndarray:
    """Apply the data's frozen SYSREM and 1D transmission collapse."""
    model_ts = jnp.asarray(model_ts)
    if model_ts.ndim != 2:
        raise ValueError(
            "A collapsed-transmission operator requires a 2D model time "
            f"series; got shape {model_ts.shape}."
        )

    model_wavelength_order = model_ts[:, ::-1]
    residual_ts = (
        model_wavelength_order
        * operator.active_exposure_mask[:, None]
    )
    if operator.chunked_sysrem is not None:
        residual_ts = sysrem_model_distortion_chunked(
            residual_ts,
            operator.chunked_sysrem,
        )
    selected = residual_ts[operator.selected_exposure_indices]
    exposure_indices = jnp.arange(selected.shape[0])[:, None]
    left = selected[exposure_indices, operator.shift_left_indices]
    right = selected[exposure_indices, operator.shift_left_indices + 1]
    shifted = left + operator.shift_fractions * (right - left)
    spectrum_unbinned = jnp.sum(operator.coadd_weights * shifted, axis=0)
    retained = operator.bin_indices.shape[0]
    spectrum_binned = jnp.zeros_like(operator.output_wavelength).at[
        operator.bin_indices
    ].add(operator.bin_weights * spectrum_unbinned[:retained])
    return spectrum_binned[operator.output_indices][None, :]


def apply_frozen_timeseries_operator(
    model_ts: jnp.ndarray,
    operator: FrozenTimeseriesInputs,
) -> jnp.ndarray:
    """Replay full-exposure preprocessing before selecting likelihood rows."""
    model_ts = jnp.asarray(model_ts)
    if model_ts.ndim != 2:
        raise ValueError(
            "A frozen time-series operator requires a 2D model time series; "
            f"got shape {model_ts.shape}."
        )
    if model_ts.shape[0] != operator.source_phase.size:
        raise ValueError(
            "Frozen time-series source exposure count does not match the model: "
            f"{operator.source_phase.size} versus {model_ts.shape[0]}."
        )

    processed = model_ts * operator.active_exposure_mask[:, None]
    median_subtracted = processed - jnp.median(
        processed,
        axis=0,
        keepdims=True,
    )
    processed = jnp.where(
        jnp.asarray(operator.subtract_time_median),
        median_subtracted,
        processed,
    )
    if operator.chunked_sysrem is not None:
        processed = sysrem_model_distortion_chunked(
            processed,
            operator.chunked_sysrem,
        )
    return processed[operator.selected_exposure_indices]


def apply_model_pipeline_corrections(
    model_ts: jnp.ndarray,
    *,
    subtract_weighted_global_mean: bool = False,
    apply_sysrem: bool,
    sigma: jnp.ndarray | None = None,
    U: jnp.ndarray | None = None,
    V: jnp.ndarray | None = None,
    chunked_sysrem: ChunkedSysremInputs | None = None,
) -> jnp.ndarray:
    if model_ts.ndim == 1:
        model_ts = model_ts[None, :]

    if subtract_weighted_global_mean:
        if sigma is None:
            raise ValueError(
                "subtract_weighted_global_mean=True requires the observed uncertainties."
            )
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            sigma = sigma[None, :]
        if sigma.shape != model_ts.shape:
            raise ValueError(
                f"sigma shape {sigma.shape} does not match model shape {model_ts.shape}"
            )
        weights = 1.0 / jnp.clip(
            sigma,
            config.F32_FLOOR_RECIPSQ,
            None,
        ) ** 2
        weighted_mean = jnp.sum(weights * model_ts) / jnp.clip(
            jnp.sum(weights),
            config.F32_FLOOR_RECIP,
            None,
        )
        model_ts = model_ts - weighted_mean

    if apply_sysrem:
        if chunked_sysrem is not None:
            model_ts = sysrem_model_distortion_chunked(model_ts, chunked_sysrem)
        else:
            if U is None or sigma is None:
                raise ValueError(
                    "apply_sysrem=True requires either chunked SYSREM inputs "
                    "or U plus per-pixel sigma."
                )
            model_ts = sysrem_model_distortion_per_pixel(
                model_ts,
                U,
                sigma,
            )

    return model_ts


def _normalize_chunked_sysrem_inputs(
    chunked_sysrem: ChunkedSysremInputs,
    *,
    n_exp: int,
    n_wave: int,
) -> ChunkedSysremInputs:
    chunk_indices = tuple(
        jnp.asarray(indices, dtype=jnp.int32)
        for indices in chunked_sysrem.chunk_indices
    )
    U_chunks = tuple(jnp.asarray(U_chunk) for U_chunk in chunked_sysrem.U_chunks)
    sigma_chunks = tuple(
        jnp.asarray(sigma_chunk) for sigma_chunk in chunked_sysrem.sigma_chunks
    )

    if not (len(chunk_indices) == len(U_chunks) == len(sigma_chunks)):
        raise ValueError(
            "chunked_sysrem must provide the same number of chunk indices, "
            "U chunks, and per-pixel sigma chunks."
        )

    normalized_indices: list[jnp.ndarray] = []
    normalized_u_chunks: list[jnp.ndarray] = []
    normalized_sigma_chunks: list[jnp.ndarray] = []
    assigned_column_count = 0

    for chunk_number, (indices, U_chunk, sigma_chunk) in enumerate(
        zip(chunk_indices, U_chunks, sigma_chunks)
    ):
        if indices.ndim != 1:
            raise ValueError(f"chunk_indices[{chunk_number}] must be 1D, got shape {indices.shape}.")
        assigned_column_count += indices.shape[0]

        if U_chunk.ndim != 2 or U_chunk.shape[0] != n_exp:
            raise ValueError(
                f"U_chunks[{chunk_number}] must have shape (n_exp, n_basis), "
                f"got {U_chunk.shape} with n_exp={n_exp}."
            )
        expected_sigma_shape = (n_exp, indices.size)
        if sigma_chunk.shape != expected_sigma_shape:
            raise ValueError(
                f"sigma_chunks[{chunk_number}] must have shape "
                f"{expected_sigma_shape}, got {sigma_chunk.shape}."
            )
        normalized_indices.append(indices)
        normalized_u_chunks.append(U_chunk)
        normalized_sigma_chunks.append(sigma_chunk)

    if assigned_column_count != n_wave:
        raise ValueError(
            "chunked_sysrem chunk sizes do not cover the full wavelength axis; "
            f"assigned {assigned_column_count} columns for n_wave={n_wave}."
        )

    return ChunkedSysremInputs(
        chunk_indices=tuple(normalized_indices),
        U_chunks=tuple(normalized_u_chunks),
        sigma_chunks=tuple(normalized_sigma_chunks),
    )


def _normalize_spectroscopic_observation_inputs(
    inputs: SpectroscopicObservationInputs,
) -> SpectroscopicObservationInputs:
    data = jnp.asarray(inputs.data)
    sigma = jnp.asarray(inputs.sigma)
    phase = None if inputs.phase is None else jnp.asarray(inputs.phase)
    U = None if inputs.U is None else jnp.asarray(inputs.U)
    V = None if inputs.V is None else jnp.asarray(inputs.V)
    chunked_sysrem = inputs.chunked_sysrem
    frozen_timeseries = inputs.frozen_timeseries
    collapsed_emission = inputs.collapsed_emission
    collapsed_transmission = inputs.collapsed_transmission

    if data.ndim == 1:
        data = data[None, :]
        sigma = sigma[None, :]

    if sigma.shape != data.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match data shape {data.shape}")
    if phase is None:
        phase = jnp.zeros((data.shape[0],), dtype=data.dtype)
    if phase.shape[0] != data.shape[0]:
        raise ValueError(f"phase length {phase.shape[0]} does not match number of exposures {data.shape[0]}")
    if chunked_sysrem is not None and (U is not None or V is not None):
        raise ValueError("Provide either global U/V SYSREM inputs or chunked_sysrem, not both.")
    if frozen_timeseries is not None and (
        U is not None or V is not None or chunked_sysrem is not None
    ):
        raise ValueError(
            "A frozen time-series operator owns its SYSREM projection; do not "
            "also provide top-level U/V or chunked_sysrem inputs."
        )
    if V is not None:
        expected_shape = (data.shape[0], data.shape[0])
        if V.shape != expected_shape:
            raise ValueError(f"V shape {V.shape} does not match expected {expected_shape}")
    if chunked_sysrem is not None:
        chunked_sysrem = _normalize_chunked_sysrem_inputs(
            chunked_sysrem,
            n_exp=data.shape[0],
            n_wave=data.shape[1],
        )
    if frozen_timeseries is not None:
        source_phase = jnp.asarray(frozen_timeseries.source_phase)
        active_mask = jnp.asarray(frozen_timeseries.active_exposure_mask)
        selected_indices = jnp.asarray(
            frozen_timeseries.selected_exposure_indices,
            dtype=jnp.int32,
        )
        if source_phase.ndim != 1 or active_mask.shape != source_phase.shape:
            raise ValueError(
                "Frozen time-series source_phase and active_exposure_mask must "
                "be matching 1D arrays."
            )
        if selected_indices.ndim != 1:
            raise ValueError(
                "Frozen time-series selected_exposure_indices must be 1D."
            )
        if selected_indices.size != data.shape[0]:
            raise ValueError(
                "Frozen time-series selected exposure count must match the "
                f"observed data: {selected_indices.size} versus {data.shape[0]}."
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
                    n_exp=source_phase.size,
                    n_wave=data.shape[1],
                )
            ),
        )
    if frozen_timeseries is not None and (
        collapsed_emission is not None or collapsed_transmission is not None
    ):
        raise ValueError(
            "A spectroscopic component cannot combine an ordinary frozen "
            "time-series operator with a collapsed 1D operator."
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

        if source_wavelength.ndim != 1 or source_inst_nus.shape != source_wavelength.shape:
            raise ValueError(
                "Collapsed emission source_wavelength and source_inst_nus "
                "must be matching 1D arrays."
            )
        if source_phase.ndim != 1:
            raise ValueError("Collapsed emission source_phase must be 1D.")
        if shift_left_indices.shape != coadd_weights.shape:
            raise ValueError(
                "Collapsed emission shift indices and coadd weights must "
                "have matching shapes."
            )
        if shift_fractions.shape != shift_left_indices.shape:
            raise ValueError(
                "Collapsed emission shift indices and fractions must have "
                "matching shapes."
            )
        if shift_left_indices.shape[0] != selected_indices.size:
            raise ValueError(
                "Collapsed emission selected exposure count does not match "
                "the collapse operator."
            )
        if bin_indices.ndim != 1 or bin_weights.shape != bin_indices.shape:
            raise ValueError(
                "Collapsed emission bin_indices and bin_weights must be "
                "matching 1D arrays."
            )
        if output_wavelength.ndim != 1:
            raise ValueError("Collapsed emission output_wavelength must be 1D.")
        if output_indices.shape != (data.shape[1],):
            raise ValueError(
                "Collapsed emission output index count must match the "
                f"observed spectrum: {output_indices.shape} versus {data.shape[1]}."
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
                    n_exp=source_phase.size,
                    n_wave=source_wavelength.size,
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
        if source_wavelength.ndim != 1 or source_inst_nus.shape != source_wavelength.shape:
            raise ValueError(
                "Collapsed transmission source_wavelength and "
                "source_inst_nus must be matching 1D arrays."
            )
        if source_phase.ndim != 1 or active_mask.shape != source_phase.shape:
            raise ValueError(
                "Collapsed transmission source_phase and "
                "active_exposure_mask must be matching 1D arrays."
            )
        if shift_left_indices.shape != coadd_weights.shape:
            raise ValueError(
                "Collapsed transmission shift indices and coadd weights "
                "must have matching shapes."
            )
        if shift_fractions.shape != shift_left_indices.shape:
            raise ValueError(
                "Collapsed transmission shift indices and fractions "
                "must have matching shapes."
            )
        if shift_left_indices.shape[0] != selected_indices.size:
            raise ValueError(
                "Collapsed transmission selected exposure count does not "
                "match the collapse operator."
            )
        if output_indices.shape != (data.shape[1],):
            raise ValueError(
                "Collapsed transmission output index count must match the "
                f"observed spectrum: {output_indices.shape} versus "
                f"{data.shape[1]}."
            )
        collapsed_transmission = CollapsedTransmissionInputs(
            source_wavelength=source_wavelength,
            source_inst_nus=source_inst_nus,
            source_phase=source_phase,
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
                    n_exp=source_phase.size,
                    n_wave=source_wavelength.size,
                )
            ),
        )

    return SpectroscopicObservationInputs(
        data=data,
        sigma=sigma,
        phase=phase,
        U=U,
        V=V,
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
    import math

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
            if not math.isfinite(float(mean)) or float(mean) <= low:
                raise ValueError(
                    f"{site_name} fixed prior requires a finite value > {low}."
                )
            value = jnp.asarray(mean)
            numpyro.deterministic(site_name, value)
            return value
        if mode == "upper_limit":
            uniform_low = low if upper_low is None else float(upper_low)
            if upper is None or not math.isfinite(float(upper)) or float(upper) <= uniform_low:
                raise ValueError(
                    f"{site_name} prior mode 'upper_limit' requires a finite upper bound > {uniform_low}."
                )
            return numpyro.sample(site_name, dist.Uniform(uniform_low, float(upper)))
        if mode == "normal":
            if not math.isfinite(float(mean)):
                raise ValueError(f"{site_name} normal prior requires a finite mean.")
            if not math.isfinite(float(std)) or float(std) <= 0.0:
                raise ValueError(f"{site_name} normal prior requires a finite positive std.")
            return numpyro.sample(
                site_name,
                dist.TruncatedNormal(float(mean), float(std), low=low),
            )
        raise ValueError(f"Unsupported {site_name} prior mode: {mode!r}")

    if not sample_Kp:
        Kp = jnp.asarray(shared_config.Kp_mean)
        numpyro.deterministic("Kp", Kp)
    elif shared_config.Kp_bounds is not None:
        Kp = numpyro.sample("Kp", dist.Uniform(*shared_config.Kp_bounds))
    elif shared_config.Kp_std is None or math.isnan(float(shared_config.Kp_std)) or shared_config.Kp_std <= 0:
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
    # P_ref is validated against the transmission RT lower boundary when the
    # ArtTransPure object is built; this is therefore the gravity at the same
    # reference/lower-boundary radius used by the forward model.
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
        raise ValueError(
            f"Unsupported velocity offset mode for region {region_config.name!r}: {mode!r}"
        )

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
        if not isinstance(component_config, SpectroscopicObservationConfig):
            raise ValueError(
                "Species-specific velocity offsets are only supported for "
                "spectroscopic observations."
            )
        for species, delta_v_kms in atmosphere_state.species_velocity_offsets_kms.items():
            if species not in opacity_terms:
                raise ValueError(
                    f"Velocity-offset species {species!r} is not an active opacity "
                    f"source for component {component_config.name!r}."
                )
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


def _validate_unique_sample_prefixes(
    items: tuple[object, ...],
    *,
    label: str,
) -> None:
    if len(items) <= 1:
        return

    missing = []
    for item in items:
        if getattr(item, "sample_prefix", None) is None:
            missing.append(getattr(item, "name", "<unnamed>"))
    if missing:
        raise ValueError(
            f"Multiple {label} require explicit sample_prefix values. Missing sample_prefix for: "
            + ", ".join(missing)
        )

    prefixes = []
    for item in items:
        prefixes.append(str(getattr(item, "sample_prefix")))
    duplicate_prefixes = set()
    for prefix in prefixes:
        if prefixes.count(prefix) > 1:
            duplicate_prefixes.add(prefix)
    duplicates = sorted(duplicate_prefixes)
    if duplicates:
        raise ValueError(
            f"{label.capitalize()} sample_prefix values must be unique. Duplicates: "
            + ", ".join(duplicates)
        )


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
    _debug_nonfinite_array("bandpass.rt_input.dtau", dtau)
    _debug_nonfinite_array("bandpass.rt_input.Tarr_rt", Tarr_rt)
    _debug_nonfinite_array("bandpass.rt_input.mmw_rt", mmw_rt)
    _debug_nonfinite_scalar("bandpass.rt_input.radius_btm", radius_btm)
    _debug_nonfinite_scalar("bandpass.rt_input.gravity_btm", gravity_btm)
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
    _debug_nonfinite_array("bandpass.rt", rt)

    if mode == "transmission":
        # The native transmission observable is physically non-negative.
        return jnp.sqrt(jnp.clip(rt, 0.0, None)) * (radius_btm / Rstar)

    Fs = _resolve_emission_stellar_surface_flux(
        nu_grid=nu_grid,
        stellar_surface_flux=stellar_surface_flux,
        context="_compute_native_observable_spectrum",
    )
    return rt / jnp.clip(Fs, config.F32_FLOOR_RECIP, None) * (radius_btm / Rstar) ** 2


def _resolve_emission_stellar_surface_flux(
    *,
    nu_grid: jnp.ndarray,
    stellar_surface_flux: jnp.ndarray | None,
    context: str,
) -> jnp.ndarray:
    if stellar_surface_flux is None:
        raise ValueError(
            f"{context} requires stellar_surface_flux for emission mode. "
            "Initialize the cached PHOENIX surface spectrum when building the observation config."
        )

    stellar_surface_flux = jnp.asarray(stellar_surface_flux)
    if stellar_surface_flux.shape != nu_grid.shape:
        raise ValueError(
            f"{context} expected stellar_surface_flux.shape={nu_grid.shape}, "
            f"got {stellar_surface_flux.shape}."
        )
    return stellar_surface_flux


def _gaussian_log_likelihood(
    data: jnp.ndarray,
    model: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    var = jnp.clip(sigma, config.F32_FLOOR_RECIPSQ, None) ** 2
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
    model_wavelength_m = 1.0e-2 / jnp.clip(nu_grid, config.F32_FLOOR_RECIP, None)
    wl_model = model_wavelength_m[::-1]
    spec_sorted = spectrum[::-1]

    rsp_interp = jnp.interp(wl_model, wavelength_m, response, left=0.0, right=0.0)
    weights = rsp_interp * wl_model if photon_weighted else rsp_interp
    norm = jnp.trapezoid(weights, wl_model)
    return jnp.trapezoid(spec_sorted * weights, wl_model) / jnp.clip(
        norm,
        config.F32_FLOOR_RECIP,
        None,
    )


def _transform_bandpass_observable(
    spectrum: jnp.ndarray,
    observable: BandpassObservable,
) -> jnp.ndarray:
    if observable in {"flux_ratio", "eclipse_depth", "radius_ratio"}:
        return spectrum
    if observable == "transit_depth":
        return spectrum**2
    raise ValueError(f"Unknown bandpass observable: {observable}")


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
    rp_over_a = Rp_m / jnp.clip(semi_major_axis_m, config.F32_FLOOR_RECIP, None)
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
    _debug_nonfinite_array(f"spectroscopy[{component_config.name}].dtau", dtau)

    collapsed_emission = observation_inputs.collapsed_emission
    collapsed_transmission = observation_inputs.collapsed_transmission
    frozen_timeseries = observation_inputs.frozen_timeseries
    if collapsed_emission is not None and collapsed_transmission is not None:
        raise ValueError(
            "A spectroscopic component cannot use both collapsed emission "
            "and collapsed transmission operators."
        )
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
        raise ValueError(
            f"Unsupported velocity offset mode: {atmosphere_state.velocity_offset_mode!r}"
        )

    if collapsed_operator is not None:
        if component_config.mode != "emission":
            if collapsed_transmission is None:
                raise ValueError(
                    "A collapsed-emission operator can only be used with "
                    "emission mode."
                )
        elif collapsed_emission is None:
            raise ValueError(
                "A collapsed-transmission operator can only be used with "
                "transmission mode."
            )
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
        phase = jnp.zeros_like(observation_inputs.phase)
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
    _debug_nonfinite_array(f"spectroscopy[{component_config.name}].model_ts_raw", model_ts)
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
        V=observation_inputs.V,
        chunked_sysrem=observation_inputs.chunked_sysrem,
    )
    _debug_nonfinite_array(f"spectroscopy[{component_config.name}].model_ts", model_ts)

    lnL = _gaussian_log_likelihood(
        observation_inputs.data,
        model_ts,
        observation_inputs.sigma,
    )
    _debug_nonfinite_scalar(f"spectroscopy[{component_config.name}].logL", lnL)
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
    if any(
        isinstance(component, SpectroscopicObservationConfig)
        and component.radial_velocity_mode == "orbital"
        for component in model_config.observations
    ) != sample_Kp:
        raise RuntimeError("Internal orbital-component activation mismatch.")
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
        if component_config.name not in observations:
            raise KeyError(f"Missing observation inputs for component '{component_config.name}'")
        if component_config.region_name not in region_states:
            raise KeyError(
                f"Observation component '{component_config.name}' references unknown "
                f"atmosphere region '{component_config.region_name}'"
            )

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
    _validate_unique_sample_prefixes(tuple(atmosphere_regions), label="atmosphere regions")
    _validate_unique_sample_prefixes(tuple(observations), label="observations")
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
    allowed_prior_keys = {"Mp", "Rp", "Rstar"}
    unknown_prior_keys = sorted(set(prior_modes).difference(allowed_prior_keys))
    if unknown_prior_keys:
        raise ValueError(
            "Unknown shared-system prior mode keys: " + ", ".join(unknown_prior_keys)
        )
    valid_prior_modes = {"fixed", "normal", "upper_limit"}
    for parameter, mode in prior_modes.items():
        if mode not in valid_prior_modes:
            raise ValueError(
                f"Unsupported prior mode for {parameter}: {mode!r}. "
                f"Choose from {sorted(valid_prior_modes)}."
            )
    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Kp_bounds = None
    Mp_upper_3sigma = params.get("M_p_upper_3sigma")
    if (
        Kp_low is not None
        and Kp_high is not None
        and np.isfinite(float(Kp_low))
        and np.isfinite(float(Kp_high))
        and float(Kp_low) < float(Kp_high)
    ):
        Kp_bounds = (float(Kp_low), float(Kp_high))

    if reference_pressure_bar is not None:
        reference_pressure_bar = float(reference_pressure_bar)
        if not np.isfinite(reference_pressure_bar) or reference_pressure_bar <= 0.0:
            raise ValueError("reference_pressure_bar must be a finite positive pressure")
        if not np.isclose(
            reference_pressure_bar,
            config.TRANSMISSION_PRESSURE_BTM,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "reference_pressure_bar must equal the configured transmission "
                f"lower boundary ({config.TRANSMISSION_PRESSURE_BTM:g} bar)"
            )

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

    if isinstance(
        composition_solver,
        (FastChemEquilibriumChemistry, FastChemHybridChemistry),
    ) and (
        composition_solver.t_min > T_low or composition_solver.t_max < T_high
    ):
        raise ValueError(
            "FastChem temperature grid must contain the full atmospheric range: "
            f"grid=[{composition_solver.t_min}, {composition_solver.t_max}] K, "
            f"atmosphere=[{T_low}, {T_high}] K. Rebuild with a wider grid."
        )

    if Tirr_mean is not None and (Tirr_mean != Tirr_mean):
        Tirr_mean = None

    mol_names, atom_names, mol_masses, atom_masses = _build_species_metadata(
        mol_names,
        atom_names,
    )
    region_name = name or _default_region_name_for_mode(mode)
    if velocity_offset_mode not in {"shared", "region", "species", "none"}:
        raise ValueError(
            f"Unsupported velocity_offset_mode for {region_name!r}: "
            f"{velocity_offset_mode!r}."
        )
    velocity_offset_species = tuple(velocity_offset_species)
    active_species = set(mol_names).union(atom_names)
    missing_velocity_species = sorted(
        set(velocity_offset_species).difference(active_species)
    )
    if missing_velocity_species:
        raise ValueError(
            f"Region {region_name!r} requested velocity offsets for inactive species: "
            + ", ".join(missing_velocity_species)
        )
    if velocity_offset_mode == "species" and not velocity_offset_species:
        raise ValueError(
            f"Region {region_name!r} uses species velocity offsets but lists no species."
        )
    if velocity_offset_mode != "species" and velocity_offset_species:
        raise ValueError(
            f"Region {region_name!r} lists velocity-offset species but mode is "
            f"{velocity_offset_mode!r}."
        )
    velocity_offset_bounds_kms = tuple(float(v) for v in velocity_offset_bounds_kms)
    if (
        len(velocity_offset_bounds_kms) != 2
        or not np.all(np.isfinite(velocity_offset_bounds_kms))
        or velocity_offset_bounds_kms[0] >= velocity_offset_bounds_kms[1]
    ):
        raise ValueError(
            "velocity_offset_bounds_kms must be a finite increasing (low, high) pair."
        )

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
        if stellar_surface_flux_arr.shape != nu_grid.shape:
            raise ValueError(
                f"stellar_surface_flux shape {stellar_surface_flux_arr.shape} does not match "
                f"nu_grid shape {nu_grid.shape}"
            )
    if mode == "emission" and stellar_surface_flux_arr is None:
        raise ValueError(
            "Emission spectroscopic observations require stellar_surface_flux. "
            "Initialize the cached PHOENIX surface spectrum before building the observation config."
        )

    if stellar_vsini is not None:
        stellar_vsini = float(stellar_vsini)
        if not np.isfinite(stellar_vsini):
            stellar_vsini = None
        elif stellar_vsini < 0.0:
            raise ValueError("stellar_vsini must be non-negative when provided.")

    def _finite_limb_darkening(value: float | None) -> float:
        if value is None:
            return 0.0
        value = float(value)
        return value if np.isfinite(value) else 0.0

    stellar_limb_darkening_u1 = _finite_limb_darkening(
        stellar_limb_darkening_u1
    )
    stellar_limb_darkening_u2 = _finite_limb_darkening(
        stellar_limb_darkening_u2
    )
    if mode == "emission" and stellar_vsini is None:
        warnings.warn(
            "Emission stellar denominator has no finite stellar_vsini; "
            "skipping stellar rotational broadening.",
            UserWarning,
        )
    if (
        mode == "emission"
        and stellar_vsini is not None
        and hasattr(sop_rot, "vrmax")
        and stellar_vsini > float(sop_rot.vrmax)
    ):
        raise ValueError(
            f"stellar_vsini={stellar_vsini:g} km/s exceeds the rotation "
            f"operator limit of {float(sop_rot.vrmax):g} km/s. Build the "
            "operator with a larger vsini_max."
        )
    check_grid_resolution(nu_grid, instrument_resolution)
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)
    if mode == "emission":
        # These quantities are fixed for an observation component, so prepare
        # the stellar denominator once rather than repeating its convolutions
        # at every likelihood evaluation.
        if stellar_vsini is not None and stellar_vsini > 0.0:
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


def _validate_bandpass_observable(
    mode: RetrievalMode,
    observable: BandpassObservable,
) -> None:
    if mode == "transmission" and observable not in {"radius_ratio", "transit_depth"}:
        raise ValueError("Transmission bandpass observations must use 'radius_ratio' or 'transit_depth'.")
    if mode == "emission" and observable not in {"flux_ratio", "eclipse_depth"}:
        raise ValueError("Emission bandpass observations must use 'flux_ratio' or 'eclipse_depth'.")


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
    _validate_bandpass_observable(mode, observable)
    if include_reflection and geometric_albedo_bounds is None:
        geometric_albedo_bounds = (0.0, 1.0)

    nu_grid = jnp.asarray(nu_grid)
    wavelength_m = jnp.asarray(wavelength_m)
    response = jnp.asarray(response)
    stellar_surface_flux_arr = None
    if stellar_surface_flux is not None:
        stellar_surface_flux_arr = jnp.asarray(stellar_surface_flux)
        if stellar_surface_flux_arr.shape != nu_grid.shape:
            raise ValueError(
                f"stellar_surface_flux shape {stellar_surface_flux_arr.shape} does not match "
                f"nu_grid shape {nu_grid.shape}"
            )
    if mode == "emission" and stellar_surface_flux_arr is None:
        raise ValueError(
            "Emission bandpass observations require stellar_surface_flux. "
            "Initialize the cached PHOENIX surface spectrum before building the observation config."
        )
    if wavelength_m.shape != response.shape:
        raise ValueError(f"wavelength_m shape {wavelength_m.shape} does not match response shape {response.shape}")

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
