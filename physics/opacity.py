"""Canonical line, CIA, and continuum optical depths for inference and reconstruction."""

from __future__ import annotations

import jax.numpy as jnp
import config
from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.rt.layeropacity import layer_optical_depth_Hminus


CIA_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("H2H2", "H2", "H2"),
    ("H2He", "H2", "He"),
)

CONTINUUM_SPECIES_MASSES: dict[str, float] = {
    "H": float(molinfo.molmass_isotope("H", db_HIT=False)),
    "H-": float(molinfo.molmass_isotope("H", db_HIT=False) + 5.48579909065e-4),
    "e-": 5.48579909065e-4,
}


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
            jnp.where(supported_nu[None, :], jnp.isfinite(logacia_matrix_raw), True)
        )
        # Outside the CIA database support, zero opacity is intentional. An
        # invalid value inside support is replaced only for safe evaluation
        # and still marks the proposal invalid.
        logacia_matrix = jnp.where(
            supported_nu[None, :] & jnp.isfinite(logacia_matrix_raw),
            logacia_matrix_raw,
            -jnp.inf,
        )
        dtau_cia = art.opacity_profile_cia(
            logacia_matrix,
            Tarr,
            vmr_profiles[species_x],
            vmr_profiles[species_y],
            mmw_profile[:, None],
            g,
        )
        dtau_cia_safe, term_valid = _sanitize_opacity_term(dtau_cia)
        cia_terms[f"CIA_{cia_key}"] = dtau_cia_safe
        cia_valid = cia_valid & in_support_finite & term_valid

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
        dtau_raw = art.opacity_profile_xs(
            xsmatrix_safe,
            mmr_profile,
            species_masses[species],
            g,
        )
        dtau_safe, term_valid = _sanitize_opacity_term(
            dtau_raw,
            negative_rtol=config.PREMODIT_NEGATIVE_ROUNDOFF_RTOL,
        )
        xs_terms[species] = dtau_safe
        xs_valid = xs_valid & xsmatrix_finite & term_valid

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
    # exojax.layer_optical_depth_Hminus combines layer-only quantities with
    # a (n_layer, n_nu) continuum matrix using raw broadcasting. Provide
    # column vectors here so layer-wise mmw and gravity broadcast correctly.
    mmw_column = jnp.asarray(mmw_profile)
    if mmw_column.ndim == 1:
        mmw_column = mmw_column[:, None]

    gravity_column = jnp.asarray(g)
    if gravity_column.ndim == 1:
        gravity_column = gravity_column[:, None]

    # H- opacity is linear in both abundances. Factor them outside ExoJAX's
    # logarithm so zero abundance also has well-defined derivatives.
    dtau_per_pair = layer_optical_depth_Hminus(
        nu_grid,
        Tarr,
        art.pressure,
        art.dParr,
        jnp.ones_like(vmre),
        jnp.ones_like(vmrh),
        mmw_column,
        gravity_column,
    )
    dtau_hminus_raw = dtau_per_pair * vmre[:, None] * vmrh[:, None]
    dtau_hminus, term_valid = _sanitize_opacity_term(dtau_hminus_raw)

    return {"CONT_Hminus": dtau_hminus}, term_valid


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
