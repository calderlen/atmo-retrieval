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
PhaseMode = Literal["global", "per_exposure", "linear"]
PTProfileMode = Literal[
    "guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"
]
AtmosphereRegionName = Literal["terminator", "dayside"]
RVBehavior = Literal["orbital", "none"]
SpectroscopicLikelihood = Literal["matched_filter", "gaussian"]
BandpassObservable = Literal["flux_ratio", "eclipse_depth", "radius_ratio", "transit_depth"]

import config

from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.postproc.specop import SopInstProfile, SopRotation
from exojax.rt.planck import piBarr
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
from physics.chemistry import BonidieChemistry, CompositionSolver, ConstantVMR, FreeVMR


EPS = 1.0e-30 # prevent divide by zero
CIA_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("H2H2", "H2", "H2"),
    ("H2He", "H2", "He"),
)

# the below dataclasses are all configuration objects, they describe how to build and sample the model
# system paramaters that are shared across all atmospheric regions and observation types in a joint retrieval
@dataclass(frozen=True)
class SharedSystemConfig:
    Kp_mean: float
    Kp_std: float
    Kp_bounds: tuple[float, float] | None
    Vsys_mean: float
    Vsys_std: float
    Rp_mean: float
    Rp_std: float
    Mp_mean: float
    Mp_std: float
    Rstar_mean: float
    Rstar_std: float
    period_day: float

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
    phase_mode: PhaseMode | None
    likelihood_kind: SpectroscopicLikelihood
    subtract_per_exposure_mean: bool
    apply_sysrem: bool
    Tstar: float | None
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
    Vsys: jnp.ndarray
    Mp: jnp.ndarray
    Rstar: jnp.ndarray
    Rp: jnp.ndarray
    g_ref: jnp.ndarray

class AtmosphereState(NamedTuple):
    art: object
    Tarr: jnp.ndarray
    g_profile: jnp.ndarray
    mmw_profile: jnp.ndarray
    mmr_mols: dict[str, jnp.ndarray]
    mmr_atoms: dict[str, jnp.ndarray]
    vmrH2_profile: jnp.ndarray
    vmrHe_profile: jnp.ndarray


class ChunkedSysremInputs(NamedTuple):
    chunk_indices: tuple[jnp.ndarray, ...]
    U_chunks: tuple[jnp.ndarray, ...]
    V_chunks: tuple[jnp.ndarray, ...]


class SpectroscopicObservationInputs(NamedTuple):
    data: jnp.ndarray
    sigma: jnp.ndarray
    phase: jnp.ndarray | None = None
    U: jnp.ndarray | None = None
    V: jnp.ndarray | None = None
    chunked_sysrem: ChunkedSysremInputs | None = None


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

# given a retrieval mode, return the default atmospheric region name to use if not specified in the configs
def _default_region_name_for_mode(mode: RetrievalMode) -> AtmosphereRegionName:
    if mode == "transmission":
        return "terminator"
    if mode == "emission":
        return "dayside"

# compute the planet radial velocity in km/s for a given phase array, Kp, Vsys, and optional additional RV offset
def planet_rv_kms(
    phase: jnp.ndarray,
    Kp_kms: float,
    Vsys_kms: float,
    dRV_kms: float = 0.0,
) -> jnp.ndarray:
    return Kp_kms * jnp.sin(2.0 * jnp.pi * phase) + Vsys_kms + dRV_kms

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
        coeffs = rhs / jnp.clip(gram[0, 0], EPS, None)
    else:
        coeffs = jnp.linalg.solve(gram, rhs)

    M_fit = U @ coeffs
    return M - M_fit


def sysrem_model_distortion_chunked(
    M: jnp.ndarray,
    chunked_sysrem: ChunkedSysremInputs,
) -> jnp.ndarray:
    corrected = M
    for chunk_indices, U_chunk, V_chunk in zip(
        chunked_sysrem.chunk_indices,
        chunked_sysrem.U_chunks,
        chunked_sysrem.V_chunks,
    ):
        if chunk_indices.shape[0] == 0 or U_chunk.shape[1] == 0:
            continue
        corrected_chunk = sysrem_model_distortion(M[:, chunk_indices], U_chunk, V_chunk)
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


def _compute_cia_opacity_terms(
    art: object,
    opa_cias: dict[str, OpaCIA],
    Tarr: jnp.ndarray,
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute CIA optical-depth contributions keyed by CIA source name."""
    vmr_profiles = {
        "H2": vmrH2_profile,
        "He": vmrHe_profile,
    }

    cia_terms: dict[str, jnp.ndarray] = {}
    for cia_key, species_x, species_y in CIA_COLLISION_PAIRS:
        cia = opa_cias.get(cia_key)
        if cia is None:
            continue
        logacia_matrix = cia.logacia_matrix(Tarr)
        cia_terms[f"CIA_{cia_key}"] = art.opacity_profile_cia(
            logacia_matrix,
            Tarr,
            vmr_profiles[species_x],
            vmr_profiles[species_y],
            mmw_profile[:, None],
            g,
        )

    return cia_terms


def _compute_xs_opacity_terms(
    art: object,
    opa_by_species: dict[str, OpaPremodit],
    Tarr: jnp.ndarray,
    mmr_profiles: dict[str, jnp.ndarray],
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute line-opacity optical-depth contributions keyed by species."""
    xs_terms: dict[str, jnp.ndarray] = {}
    for species, mmr_profile in mmr_profiles.items():
        opa = opa_by_species.get(species)
        if opa is None:
            continue
        xsmatrix = opa.xsmatrix(Tarr, art.pressure)
        xs_terms[species] = art.opacity_profile_xs(
            xsmatrix,
            mmr_profile,
            mmw_profile[:, None],
            g,
        )

    return xs_terms


def _compute_opacity_terms(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute all opacity terms using a single canonical implementation."""
    opacity_terms = _compute_cia_opacity_terms(
        art,
        opa_cias,
        Tarr,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
    )
    opacity_terms.update(
        _compute_xs_opacity_terms(
            art,
            opa_mols,
            Tarr,
            mmr_mols,
            mmw_profile,
            g,
        )
    )
    opacity_terms.update(
        _compute_xs_opacity_terms(
            art,
            opa_atoms,
            Tarr,
            mmr_atoms,
            mmw_profile,
            g,
        )
    )
    return opacity_terms


def _sum_opacity_terms(
    opacity_terms: dict[str, jnp.ndarray],
    art: object,
    nu_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Sum all opacity terms into total dtau."""
    dtau = jnp.zeros((art.pressure.size, nu_grid.size))
    for dtau_term in opacity_terms.values():
        dtau = dtau + dtau_term
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
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    opacity_terms = _compute_opacity_terms(
        art,
        opa_mols,
        opa_atoms,
        opa_cias,
        Tarr,
        mmr_mols,
        mmr_atoms,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
    )
    return _sum_opacity_terms(opacity_terms, art, nu_grid)


def compute_opacity_per_species(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    vmrH2_profile: jnp.ndarray,
    vmrHe_profile: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    g: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    return _compute_opacity_terms(
        art,
        opa_mols,
        opa_atoms,
        opa_cias,
        Tarr,
        mmr_mols,
        mmr_atoms,
        vmrH2_profile,
        vmrHe_profile,
        mmw_profile,
        g,
    )


_MISSING = object()


def _extract_scoped_site_params(
    posterior_params: dict,
    sample_prefix: str | None,
) -> dict:
    if sample_prefix is None:
        return {
            key: value
            for key, value in posterior_params.items()
            if "/" not in key
        }

    prefix = f"{sample_prefix}/"
    scoped = {
        key[len(prefix):]: value
        for key, value in posterior_params.items()
        if key.startswith(prefix)
    }
    if scoped:
        return scoped
    return {
        key: value
        for key, value in posterior_params.items()
        if "/" not in key
    }


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

        Rp = _posterior_site_value(
            posterior_params,
            "Rp",
            default=config.DEFAULT_POSTERIOR_RP,
        )
        Mp = _posterior_site_value(
            posterior_params,
            "Mp",
            default=config.DEFAULT_POSTERIOR_MP,
        )
        g_ref = gravity_surface(Rp, Mp)

        return guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_ref,
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

    vmr_mols = {
        mol: profile
        for mol in mol_names
        if (profile := _profile_for_species(mol)) is not None
    }
    vmr_atoms = {
        atom: profile
        for atom in atom_names
        if (profile := _profile_for_species(atom)) is not None
    }

    all_profiles = [*vmr_mols.values(), *vmr_atoms.values()]
    if all_profiles:
        stacked = jnp.stack(all_profiles, axis=0)
        sum_trace = jnp.sum(stacked, axis=0)
        scale = jnp.where(sum_trace > 1.0, (1.0 - 1.0e-12) / sum_trace, 1.0)
        scaled = stacked * scale[None, :]

        mol_keys = list(vmr_mols)
        atom_keys = list(vmr_atoms)
        mol_count = len(mol_keys)
        vmr_mols = {
            mol: scaled[i]
            for i, mol in enumerate(mol_keys)
        }
        vmr_atoms = {
            atom: scaled[mol_count + i]
            for i, atom in enumerate(atom_keys)
        }

    return vmr_mols, vmr_atoms


def compute_mmw_and_h2he_from_vmr(
    vmr_mols: dict[str, float],
    vmr_atoms: dict[str, float],
    mol_names: list[str],
    atom_names: list[str],
    h2_he_ratio: float = config.H2_HE_RATIO,
) -> tuple[float, float, float]:
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

    profile_values = [*vmr_mols.values(), *vmr_atoms.values()]
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
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

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
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

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
    Tarr = reconstruct_temperature_profile(
        params,
        art,
        region_config.pt_profile,
        Tint_fixed=region_config.Tint_fixed,
        sample_prefix=sample_prefix,
    )
    composition_solver = region_config.composition_solver
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
        vmr_mols = {
            mol: float(jnp.mean(vmr_profile))
            for mol, vmr_profile in vmr_mols_profiles.items()
        }
        vmr_atoms = {
            atom: float(jnp.mean(vmr_profile))
            for atom, vmr_profile in vmr_atoms_profiles.items()
        }
    elif isinstance(composition_solver, BonidieChemistry):
        vmr_mols_scalar, vmr_atoms_scalar = reconstruct_vmr_scalars(
            params,
            mol_names,
            atom_names,
            sample_prefix=sample_prefix,
        )
        n_layers = art.pressure.size
        mol_profile_map = {name: jnp.full(n_layers, 1.0e-30) for name in mol_names}
        atom_profile_map = {name: jnp.full(n_layers, 1.0e-30) for name in atom_names}

        for mol, vmr in vmr_mols_scalar.items():
            mol_profile_map[mol] = jnp.full(n_layers, vmr)
        for atom, vmr in vmr_atoms_scalar.items():
            atom_profile_map[atom] = jnp.full(n_layers, vmr)

        needed = [
            species
            for species in composition_solver.continuum_species
            if (species in mol_profile_map) or (species in atom_profile_map)
        ]
        if needed:
            if composition_solver._vmr_grids is None or any(
                species not in composition_solver._vmr_grids for species in needed
            ):
                composition_solver.build_grid(np.asarray(art.pressure), needed)

            for species in needed:
                if species not in composition_solver._vmr_grids:
                    continue
                vmr_profile = composition_solver._interp_2d(
                    composition_solver._vmr_grids[species],
                    Tarr,
                    jnp.log10(art.pressure),
                )
                if species in mol_profile_map:
                    mol_profile_map[species] = vmr_profile
                else:
                    atom_profile_map[species] = vmr_profile

        vmr_mols_profiles = [mol_profile_map[name] for name in mol_names]
        vmr_atoms_profiles = [atom_profile_map[name] for name in atom_names]
        if (len(vmr_mols_profiles) + len(vmr_atoms_profiles)) > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1.0e-12) / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(len(vmr_mols_profiles))]
            vmr_atoms_profiles = [
                all_profiles[len(vmr_mols_profiles) + i]
                for i in range(len(vmr_atoms_profiles))
            ]

        vmr_mols_profiles_dict = {
            mol: vmr_mols_profiles[i]
            for i, mol in enumerate(mol_names)
        }
        vmr_atoms_profiles_dict = {
            atom: vmr_atoms_profiles[i]
            for i, atom in enumerate(atom_names)
        }

        mmw_profile, vmrH2_profile, vmrHe_profile = compute_mmw_and_h2he_from_vmr_profiles(
            vmr_mols_profiles_dict,
            vmr_atoms_profiles_dict,
            mol_names,
            atom_names,
            n_layers=n_layers,
            h2_he_ratio=float(composition_solver.h2_he_ratio),
        )
        mmr_mols, mmr_atoms = convert_vmr_profiles_to_mmr_profiles(
            vmr_mols_profiles_dict,
            vmr_atoms_profiles_dict,
            mol_names,
            atom_names,
            mmw_profile,
        )
        vmr_mols = {
            mol: float(jnp.mean(vmr_profile))
            for mol, vmr_profile in vmr_mols_profiles_dict.items()
        }
        vmr_atoms = {
            atom: float(jnp.mean(vmr_profile))
            for atom, vmr_profile in vmr_atoms_profiles_dict.items()
        }
    else:
        raise NotImplementedError(
            "Atmospheric-state reconstruction is only implemented for "
            "ConstantVMR, FreeVMR, and BonidieChemistry. Skip contribution "
            "diagnostics for other "
            "chemistry solvers."
        )

    # Compute gravity profile
    Rp = params.get("Rp", config.DEFAULT_POSTERIOR_RP) * RJ
    Mp = params.get("Mp", config.DEFAULT_POSTERIOR_MP) * MJ
    g_ref = gravity_surface(Rp / RJ, Mp / MJ)
    g = art.gravity_profile(Tarr, mmw_profile, Rp, g_ref)

    # Compute total dtau (pass MMR profiles for molecules/atoms, VMR for CIA)
    dtau = compute_opacity(
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        Tarr=Tarr,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        mmw_profile=mmw_profile,
        g=g,
    )

    # Compute per-species dtau
    dtau_per_species = compute_opacity_per_species(
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        Tarr=Tarr,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        mmw_profile=mmw_profile,
        g=g,
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
        'mmr_mols': mmr_mols,  # MMR profiles used in opacity
        'mmr_atoms': mmr_atoms,  # MMR profiles used in opacity
        'params': params,
    }


def _sample_phase_dependent_velocity_offset(
    phase_mode: PhaseMode,
    phase: jnp.ndarray,
) -> jnp.ndarray:
    n_exp = phase.shape[0]

    if phase_mode == "global":
        return numpyro.sample("dRV", dist.Normal(0.0, 10.0))

    if phase_mode == "per_exposure":
        with numpyro.plate("exposures", n_exp):
            dRV = numpyro.sample("dRV", dist.Normal(0.0, 10.0))
        numpyro.deterministic("dRV_mean", jnp.mean(dRV))
        numpyro.deterministic("dRV_std", jnp.std(dRV))
        return dRV

    if phase_mode == "linear":
        dRV_0 = numpyro.sample("dRV_0", dist.Normal(0.0, 10.0))
        dRV_slope = numpyro.sample("dRV_slope", dist.Normal(0.0, 50.0))
        dRV = dRV_0 + dRV_slope * phase
        numpyro.deterministic("dRV_at_ingress", dRV_0 + dRV_slope * jnp.min(phase))
        numpyro.deterministic("dRV_at_egress", dRV_0 + dRV_slope * jnp.max(phase))
        return dRV

    raise ValueError(f"Unknown phase_mode: {phase_mode}")


def _sample_temperature_profile(
    region_config: AtmosphereRegionConfig,
    g_ref: float | jnp.ndarray,
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
            g_cgs=g_ref,
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
    Rp: float | jnp.ndarray,
    Rstar: float | jnp.ndarray,
    g_ref: float | jnp.ndarray,
    phase: jnp.ndarray,
    Kp: float | jnp.ndarray,
    Vsys: float | jnp.ndarray,
    dRV: float | jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    inst_nus: jnp.ndarray,
    nu_grid: jnp.ndarray,
    beta_inst: float,
    period_day: float,
    Tstar: float | None = None,
) -> jnp.ndarray:
    rt = art.run(dtau, Tarr, mmw_profile, Rp, g_ref)

    # Tidal locking: spin period = orbital period.
    vsini = 2.0 * jnp.pi * Rp / (period_day * 86400.0) / 1.0e5
    rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
    rt = sop_inst.ipgauss(rt, beta_inst)

    rv = planet_rv_kms(phase, Kp, Vsys, dRV)
    planet_ts = jax.vmap(lambda v: sop_inst.sampling(rt, v, inst_nus))(rv)

    if mode == "transmission":
        return jnp.sqrt(planet_ts) * (Rp / Rstar)

    if Tstar is None:
        raise ValueError("Tstar is required for emission mode.")

    Fs = piBarr(nu_grid, Tstar)
    Fs_ts = jax.vmap(lambda v: sop_inst.sampling(Fs, v, inst_nus))(rv)
    return planet_ts / jnp.clip(Fs_ts, EPS, None) * (Rp / Rstar) ** 2


def apply_model_pipeline_corrections(
    model_ts: jnp.ndarray,
    *,
    subtract_per_exposure_mean: bool,
    apply_sysrem: bool,
    U: jnp.ndarray | None = None,
    V: jnp.ndarray | None = None,
    chunked_sysrem: ChunkedSysremInputs | None = None,
) -> jnp.ndarray:
    if model_ts.ndim == 1:
        model_ts = model_ts[None, :]

    if subtract_per_exposure_mean:
        model_ts = model_ts - jnp.mean(model_ts, axis=1, keepdims=True)

    if apply_sysrem:
        if chunked_sysrem is not None:
            model_ts = sysrem_model_distortion_chunked(model_ts, chunked_sysrem)
        else:
            if (U is None) or (V is None):
                raise ValueError("apply_sysrem=True requires either chunked SYSREM inputs or both U and V.")
            model_ts = sysrem_model_distortion(model_ts, U, V)

    return model_ts


def _lnL_exposure(
    f_i: jnp.ndarray,
    m_i: jnp.ndarray,
    w_i: jnp.ndarray,
) -> jnp.ndarray:
    alpha_i = jnp.sum(w_i * f_i * m_i) / (jnp.sum(w_i * m_i**2) + EPS)
    r_i = f_i - alpha_i * m_i
    chi2_i = jnp.sum(w_i * r_i**2)
    norm_i = jnp.sum(jnp.log((2.0 * jnp.pi) / w_i))
    return -0.5 * (chi2_i + norm_i)


def _normalize_chunked_sysrem_inputs(
    chunked_sysrem: ChunkedSysremInputs,
    *,
    n_exp: int,
    n_wave: int,
) -> ChunkedSysremInputs:
    chunk_indices_np = tuple(np.asarray(indices, dtype=int) for indices in chunked_sysrem.chunk_indices)
    U_chunks_np = tuple(np.asarray(U_chunk) for U_chunk in chunked_sysrem.U_chunks)
    V_chunks_np = tuple(np.asarray(V_chunk) for V_chunk in chunked_sysrem.V_chunks)

    if not (
        len(chunk_indices_np) == len(U_chunks_np) == len(V_chunks_np)
    ):
        raise ValueError(
            "chunked_sysrem must provide the same number of chunk indices, U chunks, and V chunks."
        )

    assigned = np.zeros((n_wave,), dtype=bool)
    normalized_indices: list[jnp.ndarray] = []
    normalized_u_chunks: list[jnp.ndarray] = []
    normalized_v_chunks: list[jnp.ndarray] = []

    for chunk_number, (indices, U_chunk, V_chunk) in enumerate(
        zip(chunk_indices_np, U_chunks_np, V_chunks_np)
    ):
        if indices.ndim != 1:
            raise ValueError(
                f"chunk_indices[{chunk_number}] must be 1D, got shape {indices.shape}."
            )
        if np.any(indices < 0) or np.any(indices >= n_wave):
            raise ValueError(
                f"chunk_indices[{chunk_number}] contains out-of-range wavelength columns."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"chunk_indices[{chunk_number}] contains duplicate wavelength columns."
            )
        if np.any(assigned[indices]):
            raise ValueError(
                f"chunk_indices[{chunk_number}] overlaps another SYSREM chunk."
            )
        assigned[indices] = True

        if U_chunk.ndim != 2 or U_chunk.shape[0] != n_exp:
            raise ValueError(
                f"U_chunks[{chunk_number}] must have shape (n_exp, n_basis), "
                f"got {U_chunk.shape} with n_exp={n_exp}."
            )
        if V_chunk.ndim == 1:
            if V_chunk.size != n_exp:
                raise ValueError(
                    f"V_chunks[{chunk_number}] length {V_chunk.size} does not match n_exp={n_exp}."
                )
            V_chunk = np.diag(V_chunk)
        elif V_chunk.ndim != 2 or V_chunk.shape != (n_exp, n_exp):
            raise ValueError(
                f"V_chunks[{chunk_number}] must have shape {(n_exp, n_exp)}, got {V_chunk.shape}."
            )

        normalized_indices.append(jnp.asarray(indices, dtype=jnp.int32))
        normalized_u_chunks.append(jnp.asarray(U_chunk))
        normalized_v_chunks.append(jnp.asarray(V_chunk))

    if not np.all(assigned):
        missing = int(np.sum(~assigned))
        raise ValueError(
            f"chunked_sysrem does not cover the full wavelength axis; {missing} columns are unassigned."
        )

    return ChunkedSysremInputs(
        chunk_indices=tuple(normalized_indices),
        U_chunks=tuple(normalized_u_chunks),
        V_chunks=tuple(normalized_v_chunks),
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

    if data.ndim == 1:
        data = data[None, :]
        sigma = sigma[None, :]

    if sigma.shape != data.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match data shape {data.shape}")
    if phase is None:
        phase = jnp.zeros((data.shape[0],), dtype=data.dtype)
    if phase.shape[0] != data.shape[0]:
        raise ValueError(
            f"phase length {phase.shape[0]} does not match number of exposures {data.shape[0]}"
        )
    if chunked_sysrem is not None and (U is not None or V is not None):
        raise ValueError("Provide either global U/V SYSREM inputs or chunked_sysrem, not both.")
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

    return SpectroscopicObservationInputs(
        data=data,
        sigma=sigma,
        phase=phase,
        U=U,
        V=V,
        chunked_sysrem=chunked_sysrem,
    )


def _normalize_bandpass_observation_inputs(
    inputs: BandpassObservationInputs,
) -> BandpassObservationInputs:
    value = jnp.asarray(inputs.value)
    sigma = jnp.asarray(inputs.sigma)
    return BandpassObservationInputs(value=value, sigma=sigma)


def _sample_shared_system_state(
    shared_config: SharedSystemConfig,
) -> SharedSystemState:
    import math
    if shared_config.Kp_bounds is not None:
        Kp = numpyro.sample("Kp", dist.Uniform(*shared_config.Kp_bounds))
    elif shared_config.Kp_std is None or math.isnan(float(shared_config.Kp_std)) or shared_config.Kp_std <= 0:
        Kp = jnp.asarray(shared_config.Kp_mean)
        numpyro.deterministic("Kp", Kp)
    else:
        Kp = numpyro.sample(
            "Kp",
            dist.TruncatedNormal(shared_config.Kp_mean, shared_config.Kp_std, low=0.0),
        )
    Vsys = jnp.asarray(shared_config.Vsys_mean)
    Mp = numpyro.sample("Mp", dist.TruncatedNormal(shared_config.Mp_mean, shared_config.Mp_std, low=0.0)) * MJ
    Rstar = numpyro.sample(
        "Rstar",
        dist.TruncatedNormal(shared_config.Rstar_mean, shared_config.Rstar_std, low=0.0),
    ) * Rs
    Rp = numpyro.sample("Rp", dist.TruncatedNormal(shared_config.Rp_mean, shared_config.Rp_std, low=0.0)) * RJ
    g_ref = gravity_surface(Rp / RJ, Mp / MJ)

    return SharedSystemState(
        Kp=Kp,
        Vsys=Vsys,
        Mp=Mp,
        Rstar=Rstar,
        Rp=Rp,
        g_ref=g_ref,
    )


def _sample_atmosphere_state(
    region_config: AtmosphereRegionConfig,
    shared_state: SharedSystemState,
    *,
    scope_prefix: str | None = None,
) -> AtmosphereState:
    if scope_prefix is None:
        Tarr = _sample_temperature_profile(region_config, shared_state.g_ref)
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
            Tarr = _sample_temperature_profile(region_config, shared_state.g_ref)
            comp = region_config.composition_solver.sample(
                region_config.mol_names,
                region_config.mol_masses,
                region_config.atom_names,
                region_config.atom_masses,
                region_config.art,
                Tarr=Tarr,
            )

    g_profile = region_config.art.gravity_profile(
        Tarr,
        comp.mmw_profile,
        shared_state.Rp,
        shared_state.g_ref,
    )
    mmr_mols = {
        mol: comp.mmr_mols[i] for i, mol in enumerate(region_config.mol_names)
    }
    mmr_atoms = {
        atom: comp.mmr_atoms[i] for i, atom in enumerate(region_config.atom_names)
    }

    return AtmosphereState(
        art=region_config.art,
        Tarr=Tarr,
        g_profile=g_profile,
        mmw_profile=comp.mmw_profile,
        mmr_mols=mmr_mols,
        mmr_atoms=mmr_atoms,
        vmrH2_profile=comp.vmrH2_profile,
        vmrHe_profile=comp.vmrHe_profile,
    )


def _compute_component_dtau(
    component_config: ObservationConfig,
    atmosphere_state: AtmosphereState,
) -> jnp.ndarray:
    return compute_opacity(
        art=atmosphere_state.art,
        opa_mols=component_config.opa_mols,
        opa_atoms=component_config.opa_atoms,
        opa_cias=component_config.opa_cias,
        nu_grid=component_config.nu_grid,
        Tarr=atmosphere_state.Tarr,
        mmr_mols=atmosphere_state.mmr_mols,
        mmr_atoms=atmosphere_state.mmr_atoms,
        vmrH2_profile=atmosphere_state.vmrH2_profile,
        vmrHe_profile=atmosphere_state.vmrHe_profile,
        mmw_profile=atmosphere_state.mmw_profile,
        g=atmosphere_state.g_profile,
    )


def _validate_unique_sample_prefixes(
    items: tuple[object, ...],
    *,
    label: str,
) -> None:
    if len(items) <= 1:
        return

    missing = [getattr(item, "name", "<unnamed>") for item in items if getattr(item, "sample_prefix", None) is None]
    if missing:
        raise ValueError(
            f"Multiple {label} require explicit sample_prefix values. Missing sample_prefix for: "
            + ", ".join(missing)
        )

    prefixes = [str(getattr(item, "sample_prefix")) for item in items]
    duplicates = sorted({prefix for prefix in prefixes if prefixes.count(prefix) > 1})
    if duplicates:
        raise ValueError(
            f"{label.capitalize()} sample_prefix values must be unique. Duplicates: "
            + ", ".join(duplicates)
        )


def _sample_component_velocity_offset(
    component_config: SpectroscopicObservationConfig,
    phase: jnp.ndarray,
    *,
    scope_prefix: str | None = None,
) -> jnp.ndarray:
    if component_config.radial_velocity_mode == "none":
        return jnp.zeros_like(phase)
    if component_config.phase_mode is None:
        return jnp.zeros_like(phase)

    if scope_prefix is None:
        dRV = _sample_phase_dependent_velocity_offset(
            component_config.phase_mode,
            phase,
        )
        numpyro.deterministic("dRV_kms", dRV)
        return dRV

    with numpyro.handlers.scope(prefix=scope_prefix):
        dRV = _sample_phase_dependent_velocity_offset(
            component_config.phase_mode,
            phase,
        )
        numpyro.deterministic("dRV_kms", dRV)
        return dRV


def _compute_native_observable_spectrum(
    *,
    mode: RetrievalMode,
    art: object,
    dtau: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmw_profile: jnp.ndarray,
    Rp: float | jnp.ndarray,
    Rstar: float | jnp.ndarray,
    g_ref: float | jnp.ndarray,
    nu_grid: jnp.ndarray,
    Tstar: float | None = None,
) -> jnp.ndarray:
    rt = art.run(dtau, Tarr, mmw_profile, Rp, g_ref)

    if mode == "transmission":
        return jnp.sqrt(rt) * (Rp / Rstar)

    Fs = piBarr(nu_grid, Tstar)
    return rt / jnp.clip(Fs, EPS, None) * (Rp / Rstar) ** 2


def _gaussian_log_likelihood(
    data: jnp.ndarray,
    model: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    var = jnp.clip(sigma, EPS, None) ** 2
    return -0.5 * jnp.sum(((data - model) ** 2) / var + jnp.log(2.0 * jnp.pi * var))


def _bandpass_weighted_mean(
    spectrum: jnp.ndarray,
    nu_grid: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    response: jnp.ndarray,
    *,
    photon_weighted: bool,
) -> jnp.ndarray:
    model_wavelength_m = 1.0e-2 / jnp.clip(nu_grid, EPS, None)

    model_sort_idx = jnp.argsort(model_wavelength_m)
    wl_model = model_wavelength_m[model_sort_idx]
    spec_sorted = spectrum[model_sort_idx]

    band_sort_idx = jnp.argsort(wavelength_m)
    wl_band = wavelength_m[band_sort_idx]
    rsp_band = response[band_sort_idx]

    rsp_interp = jnp.interp(wl_model, wl_band, rsp_band, left=0.0, right=0.0)
    weights = rsp_interp * wl_model if photon_weighted else rsp_interp
    norm = jnp.trapezoid(weights, wl_model)
    return jnp.trapezoid(spec_sorted * weights, wl_model) / jnp.clip(norm, EPS, None)


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
    rp_over_a = Rp_m / jnp.clip(semi_major_axis_m, EPS, None)
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
    *,
    scope_prefix: str | None = None,
) -> jnp.ndarray:
    dtau = _compute_component_dtau(component_config, atmosphere_state)
    dRV = _sample_component_velocity_offset(
        component_config,
        observation_inputs.phase,
        scope_prefix=scope_prefix,
    )

    if component_config.radial_velocity_mode == "none":
        phase = jnp.zeros_like(observation_inputs.phase)
        Kp = jnp.asarray(0.0)
        Vsys = jnp.asarray(0.0)
    else:
        phase = observation_inputs.phase
        Kp = shared_state.Kp
        Vsys = shared_state.Vsys

    model_ts = compute_model_timeseries(
        mode=component_config.mode,
        art=atmosphere_state.art,
        dtau=dtau,
        Tarr=atmosphere_state.Tarr,
        mmw_profile=atmosphere_state.mmw_profile,
        Rp=shared_state.Rp,
        Rstar=shared_state.Rstar,
        g_ref=shared_state.g_ref,
        phase=phase,
        Kp=Kp,
        Vsys=Vsys,
        dRV=dRV,
        sop_rot=component_config.sop_rot,
        sop_inst=component_config.sop_inst,
        inst_nus=component_config.inst_nus,
        nu_grid=component_config.nu_grid,
        beta_inst=component_config.beta_inst,
        period_day=shared_config.period_day,
        Tstar=component_config.Tstar,
    )
    model_ts = apply_model_pipeline_corrections(
        model_ts,
        subtract_per_exposure_mean=component_config.subtract_per_exposure_mean,
        apply_sysrem=component_config.apply_sysrem,
        U=observation_inputs.U,
        V=observation_inputs.V,
        chunked_sysrem=observation_inputs.chunked_sysrem,
    )

    if component_config.likelihood_kind == "gaussian":
        return _gaussian_log_likelihood(
            observation_inputs.data,
            model_ts,
            observation_inputs.sigma,
        )
    if component_config.likelihood_kind != "matched_filter":
        raise ValueError(f"Unknown spectroscopic likelihood kind: {component_config.likelihood_kind}")

    w_ij = 1.0 / jnp.clip(observation_inputs.sigma, EPS, None) ** 2
    return jnp.sum(jax.vmap(_lnL_exposure)(observation_inputs.data, model_ts, w_ij))


def _evaluate_bandpass_component(
    component_config: BandpassObservationConfig,
    observation_inputs: BandpassObservationInputs,
    shared_state: SharedSystemState,
    atmosphere_state: AtmosphereState,
) -> jnp.ndarray:
    site_prefix = _bandpass_site_prefix(component_config)
    dtau = _compute_component_dtau(component_config, atmosphere_state)
    spectrum = _compute_native_observable_spectrum(
        mode=component_config.mode,
        art=atmosphere_state.art,
        dtau=dtau,
        Tarr=atmosphere_state.Tarr,
        mmw_profile=atmosphere_state.mmw_profile,
        Rp=shared_state.Rp,
        Rstar=shared_state.Rstar,
        g_ref=shared_state.g_ref,
        nu_grid=component_config.nu_grid,
        Tstar=component_config.Tstar,
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
                shared_state.Rp,
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
    return _gaussian_log_likelihood(
        observation_inputs.value,
        effective_value,
        observation_inputs.sigma,
    )


def joint_retrieval_model(
    model_config: JointRetrievalModelConfig,
    observations: dict[str, ObservationInputs],
) -> None:
    multi_observation = len(model_config.observations) > 1
    shared_state = _sample_shared_system_state(model_config.shared_system)
    region_states = {
        region_config.name: _sample_atmosphere_state(
            region_config,
            shared_state,
            scope_prefix=region_config.sample_prefix,
        )
        for region_config in model_config.atmosphere_regions
    }

    total_lnL = 0.0
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
            component_lnL = _evaluate_spectroscopic_component(
                component_config,
                component_inputs,
                model_config.shared_system,
                shared_state,
                region_states[component_config.region_name],
                scope_prefix=component_config.sample_prefix,
            )
        elif isinstance(component_config, BandpassObservationConfig):
            component_inputs = _normalize_bandpass_observation_inputs(component_input)
            component_lnL = _evaluate_bandpass_component(
                component_config,
                component_inputs,
                shared_state,
                region_states[component_config.region_name],
            )

        total_lnL = total_lnL + component_lnL

        if multi_observation:
            numpyro.deterministic(
                f"logL_{_sanitize_site_name(component_config.name)}",
                component_lnL,
            )

    numpyro.factor("logL", total_lnL)
    numpyro.deterministic("Kp_kms", shared_state.Kp)
    numpyro.deterministic("Vsys_kms", shared_state.Vsys)
    numpyro.deterministic("vsini_kms", 2.0 * jnp.pi * shared_state.Rp / (model_config.shared_system.period_day * 86400.0) / 1.0e5,
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
) -> SharedSystemConfig:
    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Kp_bounds = None
    if (Kp_low is not None) and (Kp_high is not None):
        Kp_bounds = (Kp_low, Kp_high)

    return SharedSystemConfig(
        Kp_mean=params["Kp"],
        Kp_std=params["Kp_err"],
        Kp_bounds=Kp_bounds,
        Vsys_mean=params["RV_abs"],
        Vsys_std=params["RV_abs_err"],
        Rp_mean=params["R_p"],
        Rp_std=params["R_p_err"],
        Mp_mean=params["M_p"],
        Mp_std=params["M_p_err"],
        Rstar_mean=params["R_star"],
        Rstar_std=params["R_star_err"],
        period_day=params["period"],
    )


def _build_species_metadata(
    mol_names: tuple[str, ...],
    atom_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], jnp.ndarray, jnp.ndarray]:
    mol_masses = jnp.array(
        [molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names]
    )
    if atom_names:
        atom_masses = jnp.array(
            [molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names]
        )
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
) -> AtmosphereRegionConfig:
    if T_low is None:
        T_low = config.T_LOW
    if T_high is None:
        T_high = config.T_HIGH
    if Tint_fixed is None:
        Tint_fixed = config.TINT_FIXED
    if kappa_ir_cgs_bounds is None:
        kappa_ir_cgs_bounds = tuple(
            float(10.0**bound) for bound in config.LOG_KAPPA_IR_BOUNDS
        )
    if gamma_bounds is None:
        gamma_bounds = tuple(float(10.0**bound) for bound in config.LOG_GAMMA_BOUNDS)

    if Tirr_mean is not None and (Tirr_mean != Tirr_mean):
        Tirr_mean = None

    mol_names, atom_names, mol_masses, atom_masses = _build_species_metadata(
        mol_names,
        atom_names,
    )
    region_name = name or _default_region_name_for_mode(mode)

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
    radial_velocity_mode: RVBehavior = "orbital",
    phase_mode: PhaseMode | None = config.DEFAULT_PHASE_MODE,
    likelihood_kind: SpectroscopicLikelihood = "matched_filter",
    subtract_per_exposure_mean: bool | None = None,
    apply_sysrem: bool | None = None,
    sample_prefix: str | None = None,
) -> SpectroscopicObservationConfig:
    if subtract_per_exposure_mean is None:
        subtract_per_exposure_mean = config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT
    if apply_sysrem is None:
        apply_sysrem = config.APPLY_SYSREM_DEFAULT

    nu_grid = jnp.asarray(nu_grid)
    inst_nus = jnp.asarray(inst_nus)
    check_grid_resolution(nu_grid, instrument_resolution)
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)

    if radial_velocity_mode == "none":
        phase_mode = None

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
        phase_mode=phase_mode,
        likelihood_kind=likelihood_kind,
        subtract_per_exposure_mean=subtract_per_exposure_mean,
        apply_sysrem=apply_sysrem,
        Tstar=Tstar,
        sample_prefix=sample_prefix,
    )


def _validate_bandpass_observable(
    mode: RetrievalMode,
    observable: BandpassObservable,
) -> None:
    if mode == "transmission" and observable not in {"radius_ratio", "transit_depth"}:
        raise ValueError(
            "Transmission bandpass observations must use 'radius_ratio' or 'transit_depth'."
        )
    if mode == "emission" and observable not in {"flux_ratio", "eclipse_depth"}:
        raise ValueError(
            "Emission bandpass observations must use 'flux_ratio' or 'eclipse_depth'."
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
    if wavelength_m.shape != response.shape:
        raise ValueError(
            f"wavelength_m shape {wavelength_m.shape} does not match response shape {response.shape}"
        )

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
        include_reflection=include_reflection,
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo_bounds=geometric_albedo_bounds,
        model_sigma=model_sigma,
        model_sigma_bounds=model_sigma_bounds,
        sample_prefix=sample_prefix,
    )
