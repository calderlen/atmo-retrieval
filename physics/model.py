"""NumPyro atmospheric model for high-resolution spectroscopic retrieval."""

from __future__ import annotations

import warnings
from typing import Callable, Literal
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

# Phase mode type for type hints
PhaseMode = Literal["shared", "per_exposure", "hierarchical", "linear", "quadratic"]

from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.postproc.specop import SopInstProfile, SopRotation
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import MJ, RJ, Rs

# Import your TP profiles (ensure these exist in your local pt.py)
from physics.pt import (
    guillot_profile,
    numpyro_free_temperature,
    numpyro_gradient,
    numpyro_madhu_seager,
    numpyro_pspline_knots_on_art_grid,
    numpyro_gp_temperature,
)
from physics.chemistry import CompositionSolver, ConstantVMR


_EPS = 1.0e-30
_TWO_PI = 2.0 * jnp.pi
_C_KMS = 299792.458  # Speed of light in km/s


def _closest_divisor(n: int, target: int) -> int:
    """Return the divisor of n closest to target (ties -> smaller)."""
    if n <= 0:
        return 1
    target = max(1, int(target))
    best = 1
    best_delta = abs(target - 1)
    limit = int(np.sqrt(n))
    for d in range(1, limit + 1):
        if n % d != 0:
            continue
        for cand in (d, n // d):
            delta = abs(cand - target)
            if delta < best_delta or (delta == best_delta and cand < best):
                best = cand
                best_delta = delta
    return best


def _estimate_guard_points(
    nu_grid: np.ndarray,
    instrument_resolution: float,
    params: dict,
    period_day: float,
    guard_kms: float | None,
    min_guard_points: int,
) -> int:
    """Estimate guard size in grid points from velocity-scale effects."""
    nu_grid = np.asarray(nu_grid)
    if nu_grid.size < 2:
        return int(min_guard_points)
    dnu = float(np.median(np.abs(np.diff(nu_grid))))
    nu_ref = float(np.nanmax(nu_grid))

    # Instrument sigma in km/s (approx)
    sigma_v = _C_KMS / (instrument_resolution * 2.3548200450309493)

    # Planet rotation broadening (km/s) using mean Rp
    Rp_mean = float(params.get("R_p", 1.0))
    vsini = (2.0 * np.pi * (Rp_mean * RJ) / (period_day * 86400.0) / 1.0e5)

    # RV guard from priors (3-sigma) + dRV prior
    kp = float(params.get("Kp", 0.0))
    kp_err = float(params.get("Kp_err", 0.0))
    vsys = float(params.get("RV_abs", 0.0))
    vsys_err = float(params.get("RV_abs_err", 0.0))
    rv_guard = abs(vsys) + 3.0 * vsys_err + kp + 3.0 * kp_err + 30.0

    v_guard = rv_guard + max(vsini, 5.0 * sigma_v)
    if guard_kms is not None:
        v_guard = max(float(guard_kms), v_guard)

    guard_points = int(np.ceil((nu_ref * v_guard / _C_KMS) / dnu))
    return max(int(min_guard_points), guard_points)


def _build_chunk_plan(
    nu_grid: np.ndarray,
    inst_nus: np.ndarray,
    guard_points: int,
    *,
    chunk_points: int | None = None,
    n_chunks: int | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
    """Plan equal-sized chunks with guard bands.

    Returns list of tuples:
      (chunk_start, chunk_end, core_start, core_end, inst_start, inst_end)
    """
    nu_grid = np.asarray(nu_grid)
    inst_nus = np.asarray(inst_nus)

    if guard_points < 0:
        raise ValueError("guard_points must be >= 0.")
    if nu_grid.size <= 2 * guard_points:
        raise ValueError("guard_points too large for nu_grid size.")

    core_len = int(nu_grid.size - 2 * guard_points)
    if core_len <= 0:
        raise ValueError("Core length is non-positive; reduce guard_points.")

    if n_chunks is None:
        target = int(chunk_points) if chunk_points is not None else core_len
        target = max(1, target)
        n_chunks = max(1, int(round(core_len / target)))

    n_chunks = _closest_divisor(core_len, int(n_chunks))
    core_size = core_len // n_chunks
    chunk_size = core_size + 2 * guard_points

    # Ensure inst_nus are inside the core-safe region
    nu_min_safe = nu_grid[guard_points]
    nu_max_safe = nu_grid[-guard_points - 1]
    if inst_nus.size == 0:
        raise ValueError("inst_nus is empty.")
    if inst_nus.min() < nu_min_safe or inst_nus.max() > nu_max_safe:
        raise ValueError(
            "Instrument grid extends into guard region. Increase WAV offsets or reduce guard."
        )

    chunks: list[tuple[int, int, int, int, int, int]] = []
    for i in range(n_chunks):
        core_start = guard_points + i * core_size
        core_end = core_start + core_size
        chunk_start = core_start - guard_points
        chunk_end = core_end + guard_points

        nu_core_min = nu_grid[core_start]
        nu_core_max = nu_grid[core_end - 1]
        inst_start = int(np.searchsorted(inst_nus, nu_core_min, side="left"))
        inst_end = int(np.searchsorted(inst_nus, nu_core_max, side="right"))
        if inst_end <= inst_start:
            raise ValueError("Chunk has no instrument points; adjust chunk/guard.")

        chunks.append((chunk_start, chunk_end, core_start, core_end, inst_start, inst_end))

    # Sanity: chunk sizes are uniform
    for chunk_start, chunk_end, *_ in chunks:
        if (chunk_end - chunk_start) != chunk_size:
            raise ValueError("Non-uniform chunk sizes detected; check chunk plan.")

    return chunks


def _slice_spectral_matrix(
    matrix: jnp.ndarray,
    start: int,
    end: int,
    n_nu: int,
) -> jnp.ndarray:
    """Slice spectral matrices along the wavenumber axis."""
    if matrix.shape[-1] == n_nu:
        return matrix[..., start:end]
    if matrix.shape[0] == n_nu:
        return matrix[start:end, ...]
    raise ValueError("Unexpected spectral matrix shape; cannot slice.")


def _resolve_spectral_arg(fn: Callable) -> str | None:
    """Return parameter name for passing a custom spectral grid, if supported."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return None
    for name in ("nus", "nu_grid"):
        if name in sig.parameters:
            return name
    return None


def _get_piBarr():
    """Locate ExoJAX's piBarr without hard-coding a single module path."""
    import importlib

    candidates = (
        "exojax.spec.planck",
        "exojax.rt.planck",
        "exojax.special.planck",
    )
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "piBarr", None)
            if fn is not None:
                return fn
        except Exception:
            continue

    try:
        spec = importlib.import_module("exojax.spec")
        planck = getattr(spec, "planck", None)
        if planck is not None and hasattr(planck, "piBarr"):
            return planck.piBarr
    except Exception:
        pass

    raise ImportError(
        "Could not import ExoJAX piBarr. Update ExoJAX or adjust the import path."
    )


def _element_from_species(species_name: str) -> str:
    """Extract element name from species notation.

    Examples:
        "Fe I" -> "Fe"
        "Fe II" -> "Fe"
        "Ca II" -> "Ca"
        "Na" -> "Na"
    """
    # Split on space and take first part (the element)
    return species_name.split()[0]


def planet_rv_kms(
    phase: jnp.ndarray,
    Kp_kms: float, 
    Vsys_kms: float, 
    dRV_kms: 
    float = 0.0
) -> jnp.ndarray:
 
    return Kp_kms * jnp.sin(_TWO_PI * phase) + Vsys_kms + dRV_kms


def sysrem_filter_model(
    model_matrix: 
    jnp.ndarray, 
    U: jnp.ndarray, 
    invvar_spec: 
    jnp.ndarray
) -> jnp.ndarray:
 
    """Apply the SYSREM linear filter: (I - U (UᵀΛU)⁻¹ UᵀΛ) M."""
 
    UTW = U.T * invvar_spec[None, :]  # (K, Nspec)
    U_dag = jnp.linalg.solve(UTW @ U, UTW)  # (K, Nspec)  = (UᵀΛU)⁻¹ UᵀΛ
 
    return model_matrix - U @ (U_dag @ model_matrix)


def check_grid_resolution(nu_grid: jnp.ndarray, 
                          R: float, 
                          min_samples: 
                          float = 4.0):
    """
    Validates that the physics grid (nu_grid) is fine enough to resolve 
    the Instrument Profile defined by R.
    
    Rule of thumb: FWHM should be sampled by at least ~4-5 pixels.
    """
    # Estimate grid resolution R_grid = nu / dnu
    # We take the median spacing to avoid edge effects
    dnu = jnp.abs(jnp.diff(nu_grid))
    nu_mid = nu_grid[:-1]
    R_grid_local = nu_mid / dnu
    R_grid = jnp.median(R_grid_local)

    # Resolution required to sample the FWHM 'min_samples' times:
    # FWHM_dnu = nu / R
    # We need dnu < FWHM_dnu / min_samples
    # => nu/R_grid < (nu/R) / min_samples
    # => R_grid > R * min_samples
    
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


# ==============================================================================
# Contribution Function Utilities
# ==============================================================================

def compute_dtau(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    vmrH2: jnp.ndarray,
    vmrHe: jnp.ndarray,
    mmw: jnp.ndarray,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """Compute optical depth matrix dtau for given atmospheric state.

    This is a standalone function that extracts the dtau computation logic
    from the NumPyro model, allowing computation of contribution functions
    from posterior samples.

    Args:
        art: ExoJAX atmospheric radiative transfer object
        opa_mols: Dictionary of molecular opacity objects
        opa_atoms: Dictionary of atomic opacity objects (pass {} if none)
        opa_cias: Dictionary of CIA opacity objects
        nu_grid: Wavenumber grid
        Tarr: Temperature array per layer
        mmr_mols: Dict mapping molecule name -> MMR profile array
        mmr_atoms: Dict mapping atom name -> MMR profile array
        vmrH2: H2 volume mixing ratio profile (for CIA)
        vmrHe: He volume mixing ratio profile (for CIA)
        mmw: Mean molecular weight profile
        g: Gravity profile

    Returns:
        dtau: Optical depth matrix, shape (n_layers, n_wavenumber)
    """
    dtau = jnp.zeros((art.pressure.size, nu_grid.size))

    # CIA contributions (use VMR for collision partners)
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        key = molA + molB
        if key in opa_cias:
            logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau = dtau + art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], g
            )

    # Molecular contributions (use MMR for opacity_profile_xs)
    for mol, mmr in mmr_mols.items():
        if mol in opa_mols:
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, mmr, mmw[:, None], g
            )

    # Atomic contributions (use MMR for opacity_profile_xs)
    for atom, mmr in mmr_atoms.items():
        if atom in opa_atoms:
            xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, mmr, mmw[:, None], g
            )

    return dtau


def compute_dtau_per_species(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    mmr_mols: dict[str, jnp.ndarray],
    mmr_atoms: dict[str, jnp.ndarray],
    vmrH2: jnp.ndarray,
    vmrHe: jnp.ndarray,
    mmw: jnp.ndarray,
    g: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute optical depth matrix dtau for each species separately.

    Useful for understanding the contribution of individual species
    to the total opacity at each wavelength and pressure level.

    Args:
        Same as compute_dtau()

    Returns:
        Dict mapping species name -> dtau array, shape (n_layers, n_wavenumber)
    """
    dtau_dict = {}

    # CIA contributions (use VMR for collision partners)
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        key = molA + molB
        if key in opa_cias:
            logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau_dict[f"CIA_{key}"] = art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], g
            )

    # Molecular contributions (use MMR for opacity_profile_xs)
    for mol, mmr in mmr_mols.items():
        if mol in opa_mols:
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau_dict[mol] = art.opacity_profile_xs(
                xsmatrix, mmr, mmw[:, None], g
            )

    # Atomic contributions (use MMR for opacity_profile_xs)
    for atom, mmr in mmr_atoms.items():
        if atom in opa_atoms:
            xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
            dtau_dict[atom] = art.opacity_profile_xs(
                xsmatrix, mmr, mmw[:, None], g
            )

    return dtau_dict


def reconstruct_temperature_profile(
    posterior_params: dict,
    art: object,
    pt_profile: str = "gp",
    Tint_fixed: float = 100.0,
) -> jnp.ndarray:
    """Reconstruct temperature profile from posterior parameter values.

    Args:
        posterior_params: Dict of parameter values (e.g., median of posterior)
        art: ExoJAX art object (provides pressure grid)
        pt_profile: Type of P-T profile ("guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp")
        Tint_fixed: Fixed internal temperature for Guillot profile

    Returns:
        Tarr: Temperature array per layer
    """
    if pt_profile == "guillot":
        Tirr = posterior_params.get("Tirr", 2500.0)
        log_kappa_ir = posterior_params.get("log_kappa_ir", -2.0)
        log_gamma = posterior_params.get("log_gamma", 0.0)
        
        # Need gravity for Guillot profile
        Rp = posterior_params.get("Rp", 1.5)  # R_J
        Mp = posterior_params.get("Mp", 1.0)  # M_J
        g_btm = gravity_jupiter(Rp, Mp)
        
        Tarr = guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_btm,
            Tirr=Tirr,
            Tint=Tint_fixed,
            kappa_ir_cgs=jnp.power(10.0, log_kappa_ir),
            gamma=jnp.power(10.0, log_gamma),
        )
        
    elif pt_profile == "isothermal":
        T0 = posterior_params.get("T0", 2500.0)
        Tarr = T0 * jnp.ones_like(art.pressure)

    elif pt_profile == "gradient":
        # Linear gradient in log-pressure
        T_btm = posterior_params.get("T_btm", 3000.0)
        T_top = posterior_params.get("T_top", 1500.0)
        log_p = jnp.log10(art.pressure)
        log_p_btm = jnp.log10(art.pressure[-1])
        log_p_top = jnp.log10(art.pressure[0])
        Tarr = T_top + (T_btm - T_top) * (log_p - log_p_top) / (log_p_btm - log_p_top)

    else:
        raise ValueError(f"Unsupported P-T profile for reconstruction: {pt_profile}")
    
    return Tarr


def reconstruct_vmr_scalars(
    posterior_params: dict,
    mol_names: list[str],
    atom_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract VMR scalar values from posterior parameter values.

    Args:
        posterior_params: Dict of parameter values (e.g., median of posterior)
        mol_names: List of molecule names to reconstruct
        atom_names: List of atom names to reconstruct

    Returns:
        Tuple of (vmr_mols, vmr_atoms) where each is a dict mapping
        species name -> scalar VMR value
    """
    vmr_mols = {}
    for mol in mol_names:
        key = f"logVMR_{mol}"
        if key in posterior_params:
            logVMR = posterior_params[key]
            vmr_mols[mol] = float(jnp.power(10.0, logVMR))

    vmr_atoms = {}
    for atom in atom_names:
        key = f"logVMR_{atom}"
        if key in posterior_params:
            logVMR = posterior_params[key]
            vmr_atoms[atom] = float(jnp.power(10.0, logVMR))

    return vmr_mols, vmr_atoms


def compute_mmw_and_h2he_from_vmr(
    vmr_mols: dict[str, float],
    vmr_atoms: dict[str, float],
    mol_names: list[str],
    atom_names: list[str],
) -> tuple[float, float, float]:
    """Compute mean molecular weight and H2/He VMRs from species VMRs.

    Args:
        vmr_mols: Dict mapping molecule name -> scalar VMR
        vmr_atoms: Dict mapping atom name -> scalar VMR
        mol_names: List of molecule names (for mass lookup)
        atom_names: List of atom names (for mass lookup)

    Returns:
        Tuple of (mmw, vmrH2, vmrHe) scalar values
    """
    # Compute molecular masses
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

    # Sum VMRs
    vmr_trace_tot = sum(vmr_mols.values()) + sum(vmr_atoms.values())
    vmr_trace_tot = min(max(vmr_trace_tot, 0.0), 1.0)

    # H2/He fill (solar ratio 6:1)
    vmrH2 = (1.0 - vmr_trace_tot) * (6.0 / 7.0)
    vmrHe = (1.0 - vmr_trace_tot) * (1.0 / 7.0)

    # Mean molecular weight from VMRs
    mass_H2 = molinfo.molmass_isotope("H2")
    mass_He = molinfo.molmass_isotope("He", db_HIT=False)
    mmw = mass_H2 * vmrH2 + mass_He * vmrHe
    mmw += sum(mol_masses[m] * v for m, v in vmr_mols.items())
    mmw += sum(atom_masses[a] * v for a, v in vmr_atoms.items())

    return mmw, vmrH2, vmrHe


def convert_vmr_to_mmr_profiles(
    vmr_mols: dict[str, float],
    vmr_atoms: dict[str, float],
    mol_names: list[str],
    atom_names: list[str],
    mmw: float,
    art: object,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Convert VMR scalars to MMR profiles for opacity calculations.

    MMR_i = VMR_i * (M_i / mmw)

    Args:
        vmr_mols: Dict mapping molecule name -> scalar VMR
        vmr_atoms: Dict mapping atom name -> scalar VMR
        mol_names: List of molecule names (for mass lookup)
        atom_names: List of atom names (for mass lookup)
        mmw: Mean molecular weight
        art: ExoJAX art object (for creating constant profiles)

    Returns:
        Tuple of (mmr_mols, mmr_atoms) where each is a dict mapping
        species name -> constant MMR profile array
    """
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


def compute_atmospheric_state_from_posterior(
    posterior_samples: dict,
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    pt_profile: str = "gp",
    use_median: bool = True,
) -> dict:
    """Compute full atmospheric state from posterior samples.
    
    This is the main entry point for computing contribution functions
    from retrieval results.
    
    Args:
        posterior_samples: Dict of posterior samples from MCMC
        art: ExoJAX art object
        opa_mols: Molecular opacity objects
        opa_atoms: Atomic opacity objects
        opa_cias: CIA opacity objects
        nu_grid: Wavenumber grid
        pt_profile: P-T profile type
        use_median: If True, use median; else use mean
        
    Returns:
        Dict with keys:
            - 'dtau': Total optical depth matrix
            - 'dtau_per_species': Dict of per-species dtau
            - 'Tarr': Temperature profile
            - 'pressure': Pressure grid
            - 'dParr': Pressure differentials
            - 'mmw': Mean molecular weight
            - 'vmrH2': H2 VMR profile
            - 'vmrHe': He VMR profile
            - 'vmr_mols': Molecular VMR profiles
            - 'vmr_atoms': Atomic VMR profiles
            - 'params': Point estimates used
    """
    import numpy as np
    
    # Extract point estimates
    if use_median:
        params = {k: float(np.median(v)) for k, v in posterior_samples.items() 
                  if not k.startswith("_")}
    else:
        params = {k: float(np.mean(v)) for k, v in posterior_samples.items()
                  if not k.startswith("_")}
    
    # Get species lists from opacity objects
    mol_names = list(opa_mols.keys())
    atom_names = list(opa_atoms.keys())

    # Reconstruct temperature profile
    Tarr = reconstruct_temperature_profile(params, art, pt_profile)

    # Reconstruct VMR scalars from posterior
    vmr_mols, vmr_atoms = reconstruct_vmr_scalars(params, mol_names, atom_names)

    # Compute mean molecular weight and H2/He VMRs from VMR scalars
    mmw, vmrH2, vmrHe = compute_mmw_and_h2he_from_vmr(
        vmr_mols, vmr_atoms, mol_names, atom_names
    )

    # Convert VMR to MMR profiles for opacity calculations
    mmr_mols, mmr_atoms = convert_vmr_to_mmr_profiles(
        vmr_mols, vmr_atoms, mol_names, atom_names, mmw, art
    )

    # Create constant profiles for H2/He VMR (for CIA) and mmw
    vmrH2_profile = art.constant_mmr_profile(vmrH2)
    vmrHe_profile = art.constant_mmr_profile(vmrHe)
    mmw_profile = art.constant_mmr_profile(mmw)

    # Compute gravity profile
    Rp = params.get("Rp", 1.5) * RJ
    Mp = params.get("Mp", 1.0) * MJ
    g_btm = gravity_jupiter(Rp / RJ, Mp / MJ)
    g = art.gravity_profile(Tarr, mmw_profile, Rp, g_btm)

    # Compute total dtau (pass MMR profiles for molecules/atoms, VMR for CIA)
    dtau = compute_dtau(
        art, opa_mols, opa_atoms, opa_cias, nu_grid,
        Tarr, mmr_mols, mmr_atoms, vmrH2_profile, vmrHe_profile, mmw_profile, g
    )

    # Compute per-species dtau
    dtau_per_species = compute_dtau_per_species(
        art, opa_mols, opa_atoms, opa_cias, nu_grid,
        Tarr, mmr_mols, mmr_atoms, vmrH2_profile, vmrHe_profile, mmw_profile, g
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


def create_retrieval_model(
    *,
    mode: Literal["transmission", "emission"],
    params: dict,  # from config.get_params()
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],  # Mandatory: pass {} if empty
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    instrument_resolution: float,  # Replaced beta_inst. Provide R (e.g. 130000)
    inst_nus: jnp.ndarray,
    # P-T profile (Default: pspline)
    pt_profile: Literal[
        "guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"
    ] = "guillot",
    T_low: float = 400.0,
    T_high: float = 3000.0,
    Tirr_std: float | None = None,  # If None, uses uniform prior on Tirr
    Tint_fixed: float = 100.0,
    log_kappa_ir_bounds: tuple[float, float] = (-4.0, 0.0),
    log_gamma_bounds: tuple[float, float] = (-2.0, 2.0),
    # Phase-dependent velocity modeling
    phase_mode: PhaseMode = "shared",
    # Pipeline options
    subtract_per_exposure_mean: bool = True,
    apply_sysrem: bool = False,
    # Inference grid stitching (optional)
    stitch_inference: bool = False,
    stitch_chunk_points: int | None = None,
    stitch_n_chunks: int | None = None,
    stitch_guard_kms: float | None = None,
    stitch_guard_points: int | None = None,
    stitch_min_guard_points: int = 128,
    # Chemistry/composition solver
    composition_solver: CompositionSolver | None = None,
) -> Callable:
    """Create NumPyro model for atmospheric retrieval.

    Args:
        mode: "transmission" or "emission"
        params: Planet parameters dict from config.get_params(), containing:
            - Kp, Kp_err: Planet RV semi-amplitude (km/s)
            - RV_abs, RV_abs_err: Systemic velocity (km/s)
            - R_p, R_p_err: Planet radius (R_J)
            - M_p, M_p_err: Planet mass (M_J)
            - R_star, R_star_err: Stellar radius (R_Sun)
            - T_star: Stellar temperature (K)
            - T_eq: Equilibrium temperature (K), optional
            - period: Orbital period (days)
        art: ExoJAX art object
        opa_mols: Molecular opacity dict
        opa_atoms: Atomic opacity dict (pass {} if none)
        opa_cias: CIA opacity dict
        nu_grid: Wavenumber grid
        sop_rot: Rotation operator
        sop_inst: Instrument profile operator
        instrument_resolution: Spectral resolution R
        inst_nus: Instrument wavenumber grid
        pt_profile: P-T profile type (guillot, isothermal, gradient, madhu_seager, free, pspline, gp)
        T_low, T_high: Temperature bounds (K)
        Tirr_std: If provided, use normal prior on Tirr with T_eq as mean
        Tint_fixed: Internal temperature (K)
        log_kappa_ir_bounds: Prior bounds on log10(kappa_IR)
        log_gamma_bounds: Prior bounds on log10(gamma)
        phase_mode: How to model phase-dependent velocity offset dRV:
            - "shared": Single dRV for all exposures (default)
            - "per_exposure": Independent dRV[i] for each exposure
            - "hierarchical": dRV[i] ~ Normal(dRV_mean, dRV_scatter)
            - "linear": dRV = dRV_0 + dRV_slope * phase
            - "quadratic": dRV = dRV_a + dRV_b * phase + dRV_c * phase^2
        subtract_per_exposure_mean: Subtract per-exposure mean
        apply_sysrem: Apply SYSREM filter
        stitch_inference: Enable chunked forward model across the grid
        stitch_chunk_points: Target number of grid points per chunk core
        stitch_n_chunks: Explicit number of chunks (overrides stitch_chunk_points)
        stitch_guard_kms: Guard size in km/s for edge effects
        stitch_guard_points: Explicit guard size in grid points
        stitch_min_guard_points: Minimum guard size in grid points

    Returns:
        NumPyro model function
    """
    # --- 0. Extract parameters from config ---
    Kp_mean, Kp_std = params["Kp"], params["Kp_err"]
    Vsys_mean, Vsys_std = params["RV_abs"], params["RV_abs_err"]
    Rp_mean, Rp_std = params["R_p"], params["R_p_err"]
    Mp_mean, Mp_std = params["M_p"], params["M_p_err"]
    Rstar_mean, Rstar_std = params["R_star"], params["R_star_err"]
    Tstar = params["T_star"]
    Tirr_mean = params.get("T_eq")  # May be None or nan
    period_day = params["period"]

    # Handle nan values in Tirr_mean
    if Tirr_mean is not None and (Tirr_mean != Tirr_mean):  # nan check
        Tirr_mean = None

    # --- 1. Validation & Setup ---
    check_grid_resolution(nu_grid, instrument_resolution)

    # Calculate beta_inst (IP Gaussian Sigma) from Resolution (R)
    # FWHM = nu / R
    # Sigma = FWHM / (2 * sqrt(2 * ln(2))) approx FWHM / 2.35482
    # For ESLOG (log-wavenumber) grids, beta is dimensionless ~ 1/R_eff
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)

    mol_names = tuple(opa_mols.keys())
    atom_names = tuple(opa_atoms.keys())  # Mandatory argument, checking keys is safe

    # Pre-compute masses
    # Avoid HITRAN-only checks for non-HITRAN species (silences warnings)
    mol_masses = jnp.array([molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names])
    # Handle case where opa_atoms is {}
    if len(atom_names) > 0:
        atom_masses = jnp.array(
            [molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names]
        )
    else:
        atom_masses = jnp.zeros((0,))

    # Default composition solver if not provided
    if composition_solver is None:
        composition_solver = ConstantVMR()

    if (mode == "emission") and (Tstar is None):
        raise ValueError("Tstar is required for emission mode.")

    xs_arg = None
    cia_arg = None
    if opa_mols:
        xs_arg = _resolve_spectral_arg(next(iter(opa_mols.values())).xsmatrix)
    elif opa_atoms:
        # Atoms use same OpaPremodit API - check for chunk-friendly xsmatrix
        xs_arg = _resolve_spectral_arg(next(iter(opa_atoms.values())).xsmatrix)
    if opa_cias:
        cia_arg = _resolve_spectral_arg(next(iter(opa_cias.values())).logacia_matrix)

    # --- Optional: build chunk plan for inference stitching ---
    stitch_plan = None
    stitch_sops: list[tuple[SopRotation, SopInstProfile]] | None = None
    if stitch_inference:
        if apply_sysrem:
            raise ValueError("SYSREM is not supported with inference stitching.")

        guard_points = (
            int(stitch_guard_points)
            if stitch_guard_points is not None
            else _estimate_guard_points(
                np.asarray(nu_grid),
                instrument_resolution,
                params,
                period_day,
                stitch_guard_kms,
                stitch_min_guard_points,
            )
        )

        stitch_plan = _build_chunk_plan(
            np.asarray(nu_grid),
            np.asarray(inst_nus),
            guard_points,
            chunk_points=stitch_chunk_points,
            n_chunks=stitch_n_chunks,
        )

        vsini_max = float(getattr(sop_rot, "vsini_max", 150.0))
        vrmax = float(getattr(sop_inst, "vrmax", 500.0))
        stitch_sops = []
        for chunk_start, chunk_end, *_ in stitch_plan:
            nu_chunk = jnp.array(nu_grid[chunk_start:chunk_end])
            stitch_sops.append(
                (
                    SopRotation(nu_chunk, vsini_max=vsini_max),
                    SopInstProfile(nu_chunk, vrmax=vrmax),
                )
            )
        chunk_size = stitch_plan[0][1] - stitch_plan[0][0]
        core_size = chunk_size - 2 * guard_points
        print(
            f"[INFO] Inference stitching enabled: "
            f"{len(stitch_plan)} chunks, core={core_size}, guard={guard_points}."
        )

    # --- The NumPyro Model ---
    def model(
        data: jnp.ndarray,
        sigma: jnp.ndarray,
        phase: jnp.ndarray,
        U: jnp.ndarray | None = None,
        invvar_spec: jnp.ndarray | None = None,
    ) -> None:

        # 1. System Parameters
        Kp = numpyro.sample("Kp", dist.TruncatedNormal(Kp_mean, Kp_std, low=0.0))
        Vsys = numpyro.sample("Vsys", dist.Normal(Vsys_mean, Vsys_std))
        
        # Phase-dependent velocity offset (dRV) sampling based on phase_mode
        n_exp = phase.shape[0]
        
        if phase_mode == "shared":
            # Single dRV for all exposures (original behavior)
            dRV = numpyro.sample("dRV", dist.Normal(0.0, 10.0))
            
        elif phase_mode == "per_exposure":
            # Independent dRV for each exposure
            with numpyro.plate("exposures", n_exp):
                dRV = numpyro.sample("dRV", dist.Normal(0.0, 10.0))
            # Track mean and std as deterministics
            numpyro.deterministic("dRV_mean", jnp.mean(dRV))
            numpyro.deterministic("dRV_std", jnp.std(dRV))
            
        elif phase_mode == "hierarchical":
            # Hierarchical: dRV[i] ~ Normal(dRV_mean, dRV_scatter)
            dRV_mean = numpyro.sample("dRV_mean", dist.Normal(0.0, 10.0))
            dRV_scatter = numpyro.sample("dRV_scatter", dist.HalfNormal(5.0))
            with numpyro.plate("exposures", n_exp):
                dRV = numpyro.sample("dRV", dist.Normal(dRV_mean, dRV_scatter))
            
        elif phase_mode == "linear":
            # Linear trend: dRV = dRV_0 + dRV_slope * phase
            dRV_0 = numpyro.sample("dRV_0", dist.Normal(0.0, 10.0))
            dRV_slope = numpyro.sample("dRV_slope", dist.Normal(0.0, 50.0))  # km/s per phase
            dRV = dRV_0 + dRV_slope * phase
            # Track values at ingress/egress for interpretation
            numpyro.deterministic("dRV_at_ingress", dRV_0 + dRV_slope * jnp.min(phase))
            numpyro.deterministic("dRV_at_egress", dRV_0 + dRV_slope * jnp.max(phase))
            
        elif phase_mode == "quadratic":
            # Quadratic: dRV = a + b*phase + c*phase^2
            dRV_a = numpyro.sample("dRV_a", dist.Normal(0.0, 10.0))
            dRV_b = numpyro.sample("dRV_b", dist.Normal(0.0, 50.0))
            dRV_c = numpyro.sample("dRV_c", dist.Normal(0.0, 100.0))
            dRV = dRV_a + dRV_b * phase + dRV_c * phase**2
            # Track values at key phases
            numpyro.deterministic("dRV_at_midtransit", dRV_a)
            numpyro.deterministic("dRV_at_ingress", dRV_a + dRV_b * jnp.min(phase) + dRV_c * jnp.min(phase)**2)
            numpyro.deterministic("dRV_at_egress", dRV_a + dRV_b * jnp.max(phase) + dRV_c * jnp.max(phase)**2)
        else:
            raise ValueError(f"Unknown phase_mode: {phase_mode}")

        Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0)) * MJ
        Rstar = numpyro.sample("Rstar", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0)) * Rs
        Rp = numpyro.sample("Rp", dist.TruncatedNormal(Rp_mean, Rp_std, low=0.0)) * RJ

        # 2. Composition (sample VMR, convert to MMR for opacity calculations)
        comp = composition_solver.sample(
            mol_names, mol_masses, atom_names, atom_masses, art
        )
        mmr_mols = comp.mmr_mols
        mmr_atoms = comp.mmr_atoms
        vmrH2_prof = comp.vmrH2_prof
        vmrHe_prof = comp.vmrHe_prof
        mmw_prof = comp.mmw_prof

        # 3. Temperature & Gravity
        g_btm = gravity_jupiter(Rp / RJ, Mp / MJ)

        if pt_profile == "guillot":
            if (Tirr_mean is not None) and (Tirr_std is not None):
                Tirr = numpyro.sample(
                    "Tirr", dist.TruncatedNormal(Tirr_mean, Tirr_std, low=0.0)
                )
            else:
                Tirr = numpyro.sample("Tirr", dist.Uniform(T_low, T_high))

            log_kappa_ir = numpyro.sample(
                "log_kappa_ir", dist.Uniform(*log_kappa_ir_bounds)
            )
            log_gamma = numpyro.sample("log_gamma", dist.Uniform(*log_gamma_bounds))

            Tarr = guillot_profile(
                pressure_bar=art.pressure,
                g_cgs=g_btm,
                Tirr=Tirr,
                Tint=Tint_fixed,
                kappa_ir_cgs=jnp.power(10.0, log_kappa_ir),
                gamma=jnp.power(10.0, log_gamma),
            )

        elif pt_profile == "isothermal":
            T0 = numpyro.sample("T0", dist.Uniform(T_low, T_high))
            Tarr = T0 * jnp.ones_like(art.pressure)
        elif pt_profile == "gradient":
            Tarr = numpyro_gradient(art, T_low, T_high)
        elif pt_profile == "madhu_seager":
            Tarr = numpyro_madhu_seager(art, T_low, T_high)
        elif pt_profile == "free":
            Tarr = numpyro_free_temperature(art, n_layers=5, T_low=T_low, T_high=T_high)
        elif pt_profile == "pspline":
            Tarr = numpyro_pspline_knots_on_art_grid(art, T_low=T_low, T_high=T_high)
        elif pt_profile == "gp":
            Tarr = numpyro_gp_temperature(art, T_low=T_low, T_high=T_high)
        else:
            raise ValueError(f"Unknown P-T profile: {pt_profile}")

        g = art.gravity_profile(Tarr, mmw_prof, Rp, g_btm)

        # 4. Opacity Calculation + Forward Model (full grid or stitched)
        # Ensure 2D shapes for downstream operations
        if data.ndim == 1:
            data = data[None, :]
            sigma = sigma[None, :]

        # Tidal locking: Spin period = Orbital period
        vsini = (
            2.0 * jnp.pi * Rp / (period_day * 86400.0) / 1.0e5
        )  # cm/s -> km/s

        if stitch_plan is None:
            dtau = jnp.zeros((art.pressure.size, nu_grid.size))

            # CIA (use VMR profiles for collision partners)
            for molA, molB in [("H2", "H2"), ("H2", "He")]:
                key = molA + molB
                if key not in opa_cias:
                    continue
                logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
                vmrX, vmrY = (vmrH2_prof, vmrH2_prof) if molB == "H2" else (vmrH2_prof, vmrHe_prof)
                dtau = dtau + art.opacity_profile_cia(
                    logacia_matrix, Tarr, vmrX, vmrY, mmw_prof[:, None], g
                )

            # Molecules (use MMR for opacity_profile_xs)
            for i, mol in enumerate(mol_names):
                xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
                dtau = dtau + art.opacity_profile_xs(
                    xsmatrix, mmr_mols[i], mmw_prof[:, None], g
                )

            # Atoms (use MMR for opacity_profile_xs)
            for i, atom in enumerate(atom_names):
                xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
                dtau = dtau + art.opacity_profile_xs(
                    xsmatrix, mmr_atoms[i], mmw_prof[:, None], g
                )

            # 5. Radiative Transfer
            rt = art.run(dtau, Tarr, mmw_prof, Rp, g_btm)

            # 6. Rotation & Instrument Broadening
            rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
            rt = sop_inst.ipgauss(rt, beta_inst)

            # 7. Time-Series Sampling (Doppler shift)
            rv = planet_rv_kms(phase, Kp, Vsys, dRV)
            planet_ts = jax.vmap(lambda v: sop_inst.sampling(rt, v, inst_nus))(rv)

            # 8. Contrast / Flux Ratio
            if mode == "transmission":
                model_ts = jnp.sqrt(planet_ts) * (Rp / Rstar)
            else:
                piBarr = _get_piBarr()
                Fs = piBarr(nu_grid, Tstar)
                Fs_ts = jax.vmap(lambda v: sop_inst.sampling(Fs, v, inst_nus))(rv)
                model_ts = planet_ts / jnp.clip(Fs_ts, _EPS, None) * (Rp / Rstar) ** 2

            # 9. Pipeline Corrections
            if model_ts.ndim == 1:
                model_ts = model_ts[None, :]
            if model_ts.shape[1] == 0:
                raise ValueError("model_ts has zero spectral points; check inst_nus/nu_grid")

            if subtract_per_exposure_mean:
                model_ts = model_ts - jnp.mean(model_ts, axis=1, keepdims=True)

            if apply_sysrem:
                if (U is None) or (invvar_spec is None):
                    # This should be caught by caller, but safe to check
                    pass
                model_ts = sysrem_filter_model(model_ts, U, invvar_spec)

            # 10. Log-Likelihood (CCF-equivalent Matched Filter)
            w = 1.0 / jnp.clip(sigma, _EPS, None) ** 2

            def _lnL_one(f: jnp.ndarray, m: jnp.ndarray, w_: jnp.ndarray) -> jnp.ndarray:
                s_mm = jnp.sum((m * m) * w_)
                s_fm = jnp.sum((f * m) * w_)
                alpha = s_fm / (s_mm + _EPS)
                r = f - alpha * m
                chi2 = jnp.sum((r * r) * w_)
                norm = jnp.sum(jnp.log(_TWO_PI / w_))
                return -0.5 * (chi2 + norm)

            lnL = jnp.sum(jax.vmap(_lnL_one)(data, model_ts, w))
            numpyro.factor("logL", lnL)
        else:
            # Stitched inference: accumulate log-likelihood across chunks
            w = 1.0 / jnp.clip(sigma, _EPS, None) ** 2
            sum_w = jnp.sum(w, axis=1)
            sum_f_w = jnp.sum(data * w, axis=1)
            s_ff = jnp.sum((data * data) * w, axis=1)
            norm = jnp.sum(jnp.log(_TWO_PI / w), axis=1)

            sum_m = jnp.zeros_like(sum_w)
            sum_m_w = jnp.zeros_like(sum_w)
            sum_m2_w = jnp.zeros_like(sum_w)
            sum_fm_w = jnp.zeros_like(sum_w)

            rv = planet_rv_kms(phase, Kp, Vsys, dRV)

            for (chunk_start, chunk_end, _core_start, _core_end, inst_start, inst_end), sops in zip(
                stitch_plan, stitch_sops
            ):
                sop_rot_chunk, sop_inst_chunk = sops
                nu_chunk = nu_grid[chunk_start:chunk_end]

                dtau_chunk = jnp.zeros((art.pressure.size, nu_chunk.size))

                # CIA
                for molA, molB in [("H2", "H2"), ("H2", "He")]:
                    key = molA + molB
                    if key not in opa_cias:
                        continue
                    if cia_arg is not None:
                        logacia_chunk = opa_cias[key].logacia_matrix(
                            Tarr, **{cia_arg: nu_chunk}
                        )
                    else:
                        logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
                        logacia_chunk = _slice_spectral_matrix(
                            logacia_matrix, chunk_start, chunk_end, nu_grid.size
                        )
                    vmrX, vmrY = (vmrH2_prof, vmrH2_prof) if molB == "H2" else (vmrH2_prof, vmrHe_prof)
                    dtau_chunk = dtau_chunk + art.opacity_profile_cia(
                        logacia_chunk, Tarr, vmrX, vmrY, mmw_prof[:, None], g
                    )

                # Molecules (use MMR for opacity_profile_xs)
                for i, mol in enumerate(mol_names):
                    if xs_arg is not None:
                        xs_chunk = opa_mols[mol].xsmatrix(
                            Tarr, art.pressure, **{xs_arg: nu_chunk}
                        )
                    else:
                        xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
                        xs_chunk = _slice_spectral_matrix(
                            xsmatrix, chunk_start, chunk_end, nu_grid.size
                        )
                    dtau_chunk = dtau_chunk + art.opacity_profile_xs(
                        xs_chunk, mmr_mols[i], mmw_prof[:, None], g
                    )

                # Atoms (use MMR for opacity_profile_xs)
                for i, atom in enumerate(atom_names):
                    if xs_arg is not None:
                        xs_chunk = opa_atoms[atom].xsmatrix(
                            Tarr, art.pressure, **{xs_arg: nu_chunk}
                        )
                    else:
                        xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
                        xs_chunk = _slice_spectral_matrix(
                            xsmatrix, chunk_start, chunk_end, nu_grid.size
                        )
                    dtau_chunk = dtau_chunk + art.opacity_profile_xs(
                        xs_chunk, mmr_atoms[i], mmw_prof[:, None], g
                    )

                # Debug: check dtau_chunk shape before RT
                if dtau_chunk.shape[1] != nu_chunk.size:
                    raise ValueError(
                        f"dtau_chunk shape mismatch: dtau_chunk.shape={dtau_chunk.shape}, "
                        f"nu_chunk.size={nu_chunk.size}"
                    )

                rt = art.run(dtau_chunk, Tarr, mmw_prof, Rp, g_btm)
                rt = sop_rot_chunk.rigid_rotation(rt, vsini, 0.0, 0.0)

                # ExoJAX's rigid_rotation can return 1 fewer element due to a bug
                # in convolve_same (irfft called without explicit n). Pad to match.
                expected_size = sop_rot_chunk.nu_grid.shape[0]
                if rt.shape[0] < expected_size:
                    pad_width = expected_size - rt.shape[0]
                    rt = jnp.pad(rt, (0, pad_width), mode="edge")

                rt = sop_inst_chunk.ipgauss(rt, beta_inst)

                # Same boundary handling for ipgauss (ExoJAX's convolve_same has a
                # bug where irfft is called without explicit n, causing 1-element
                # size reduction for odd FFT lengths)
                expected_size = sop_inst_chunk.nu_grid.shape[0]
                if rt.shape[0] < expected_size:
                    pad_width = expected_size - rt.shape[0]
                    rt = jnp.pad(rt, (0, pad_width), mode="edge")

                inst_slice = inst_nus[inst_start:inst_end]

                planet_ts = jax.vmap(lambda v: sop_inst_chunk.sampling(rt, v, inst_slice))(rv)

                if mode == "transmission":
                    model_ts = jnp.sqrt(planet_ts) * (Rp / Rstar)
                else:
                    piBarr = _get_piBarr()
                    Fs = piBarr(nu_chunk, Tstar)
                    Fs_ts = jax.vmap(lambda v: sop_inst_chunk.sampling(Fs, v, inst_slice))(rv)
                    model_ts = planet_ts / jnp.clip(Fs_ts, _EPS, None) * (Rp / Rstar) ** 2

                if model_ts.ndim == 1:
                    model_ts = model_ts[None, :]

                data_slice = data[:, inst_start:inst_end]
                w_slice = w[:, inst_start:inst_end]

                sum_m = sum_m + jnp.sum(model_ts, axis=1)
                sum_m_w = sum_m_w + jnp.sum(model_ts * w_slice, axis=1)
                sum_m2_w = sum_m2_w + jnp.sum((model_ts * model_ts) * w_slice, axis=1)
                sum_fm_w = sum_fm_w + jnp.sum((data_slice * model_ts) * w_slice, axis=1)

            if subtract_per_exposure_mean:
                n_points = data.shape[1]
                mean_m = sum_m / n_points
                s_mm = sum_m2_w - 2.0 * mean_m * sum_m_w + (mean_m * mean_m) * sum_w
                s_fm = sum_fm_w - mean_m * sum_f_w
            else:
                s_mm = sum_m2_w
                s_fm = sum_fm_w

            alpha = s_fm / (s_mm + _EPS)
            chi2 = s_ff - 2.0 * alpha * s_fm + (alpha * alpha) * s_mm
            lnL = jnp.sum(-0.5 * (chi2 + norm))
            numpyro.factor("logL", lnL)

        # Deterministics for tracking
        numpyro.deterministic("Kp_kms", Kp)
        numpyro.deterministic("Vsys_kms", Vsys)
        numpyro.deterministic("dRV_kms", dRV)
        numpyro.deterministic("vsini_kms", vsini)

    return model
