"""NumPyro atmospheric model for high-resolution spectroscopic retrieval."""

from __future__ import annotations

import warnings
from typing import Callable, Literal

import jax
import jax.numpy as jnp
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
)


_EPS = 1.0e-30
_TWO_PI = 2.0 * jnp.pi
_C_KMS = 299792.458  # Speed of light in km/s


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
    vmr_mols: dict[str, jnp.ndarray],
    vmr_atoms: dict[str, jnp.ndarray],
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
        vmr_mols: Dict mapping molecule name -> VMR profile array
        vmr_atoms: Dict mapping atom name -> VMR profile array
        vmrH2: H2 volume mixing ratio profile
        vmrHe: He volume mixing ratio profile
        mmw: Mean molecular weight profile
        g: Gravity profile
        
    Returns:
        dtau: Optical depth matrix, shape (n_layers, n_wavenumber)
    """
    dtau = jnp.zeros((art.pressure.size, nu_grid.size))
    
    # CIA contributions
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        key = molA + molB
        if key in opa_cias:
            logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau = dtau + art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], g
            )
    
    # Molecular contributions
    for mol, vmr in vmr_mols.items():
        if mol in opa_mols:
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, vmr, mmw[:, None], g
            )
    
    # Atomic contributions
    for atom, vmr in vmr_atoms.items():
        if atom in opa_atoms:
            xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, vmr, mmw[:, None], g
            )
    
    return dtau


def compute_dtau_per_species(
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    Tarr: jnp.ndarray,
    vmr_mols: dict[str, jnp.ndarray],
    vmr_atoms: dict[str, jnp.ndarray],
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
    
    # CIA contributions
    for molA, molB in [("H2", "H2"), ("H2", "He")]:
        key = molA + molB
        if key in opa_cias:
            logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau_dict[f"CIA_{key}"] = art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], g
            )
    
    # Molecular contributions
    for mol, vmr in vmr_mols.items():
        if mol in opa_mols:
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau_dict[mol] = art.opacity_profile_xs(
                xsmatrix, vmr, mmw[:, None], g
            )
    
    # Atomic contributions
    for atom, vmr in vmr_atoms.items():
        if atom in opa_atoms:
            xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
            dtau_dict[atom] = art.opacity_profile_xs(
                xsmatrix, vmr, mmw[:, None], g
            )
    
    return dtau_dict


def reconstruct_temperature_profile(
    posterior_params: dict,
    art: object,
    temperature_profile: str = "guillot",
    Tint_fixed: float = 100.0,
) -> jnp.ndarray:
    """Reconstruct temperature profile from posterior parameter values.
    
    Args:
        posterior_params: Dict of parameter values (e.g., median of posterior)
        art: ExoJAX art object (provides pressure grid)
        temperature_profile: Type of T-P profile ("guillot", "isothermal", etc.)
        Tint_fixed: Fixed internal temperature for Guillot profile
        
    Returns:
        Tarr: Temperature array per layer
    """
    if temperature_profile == "guillot":
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
        
    elif temperature_profile == "isothermal":
        T0 = posterior_params.get("T0", 2500.0)
        Tarr = T0 * jnp.ones_like(art.pressure)
        
    elif temperature_profile == "gradient":
        # Linear gradient in log-pressure
        T_btm = posterior_params.get("T_btm", 3000.0)
        T_top = posterior_params.get("T_top", 1500.0)
        log_p = jnp.log10(art.pressure)
        log_p_btm = jnp.log10(art.pressure[-1])
        log_p_top = jnp.log10(art.pressure[0])
        Tarr = T_top + (T_btm - T_top) * (log_p - log_p_top) / (log_p_btm - log_p_top)
        
    else:
        raise ValueError(f"Unsupported temperature profile for reconstruction: {temperature_profile}")
    
    return Tarr


def reconstruct_vmrs(
    posterior_params: dict,
    art: object,
    mol_names: list[str],
    atom_names: list[str],
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Reconstruct VMR profiles from posterior parameter values.
    
    Args:
        posterior_params: Dict of parameter values (e.g., median of posterior)
        art: ExoJAX art object (provides pressure grid for profile shape)
        mol_names: List of molecule names to reconstruct
        atom_names: List of atom names to reconstruct
        
    Returns:
        Tuple of (vmr_mols, vmr_atoms) where each is a dict mapping
        species name -> constant VMR profile array
    """
    vmr_mols = {}
    for mol in mol_names:
        key = f"logVMR_{mol}"
        if key in posterior_params:
            logVMR = posterior_params[key]
            vmr_mols[mol] = art.constant_mmr_profile(jnp.power(10.0, logVMR))
    
    vmr_atoms = {}
    for atom in atom_names:
        key = f"logVMR_{atom}"
        if key in posterior_params:
            logVMR = posterior_params[key]
            vmr_atoms[atom] = art.constant_mmr_profile(jnp.power(10.0, logVMR))
    
    return vmr_mols, vmr_atoms


def reconstruct_mmw_and_h2he(
    vmr_mols: dict[str, jnp.ndarray],
    vmr_atoms: dict[str, jnp.ndarray],
    mol_names: list[str],
    atom_names: list[str],
    art: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute mean molecular weight and H2/He VMRs from species VMRs.
    
    Args:
        vmr_mols: Dict mapping molecule name -> VMR profile
        vmr_atoms: Dict mapping atom name -> VMR profile
        mol_names: List of molecule names (for mass lookup)
        atom_names: List of atom names (for mass lookup)
        art: ExoJAX art object
        
    Returns:
        Tuple of (mmw, vmrH2, vmrHe) arrays
    """
    # Compute molecular masses
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    # For atoms, extract element name from species notation (e.g., "Fe I" -> "Fe")
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }
    
    # Sum VMRs
    sum_mols = sum(vmr_mols.values()) if vmr_mols else 0.0
    sum_atoms = sum(vmr_atoms.values()) if vmr_atoms else 0.0
    vmr_tot = jnp.clip(sum_mols + sum_atoms, 0.0, 1.0)
    
    # H2/He fill (solar ratio 6:1)
    vmrH2 = (1.0 - vmr_tot) * (6.0 / 7.0)
    vmrHe = (1.0 - vmr_tot) * (1.0 / 7.0)
    
    # Mean molecular weight
    dot_mols = sum(mol_masses[m] * vmr for m, vmr in vmr_mols.items()) if vmr_mols else 0.0
    dot_atoms = sum(atom_masses[a] * vmr for a, vmr in vmr_atoms.items()) if vmr_atoms else 0.0
    
    mmw = (
        molinfo.molmass_isotope("H2") * vmrH2
        + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
        + dot_mols
        + dot_atoms
    )
    
    return mmw, vmrH2, vmrHe


def compute_atmospheric_state_from_posterior(
    posterior_samples: dict,
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    temperature_profile: str = "guillot",
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
        temperature_profile: T-P profile type
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
    Tarr = reconstruct_temperature_profile(params, art, temperature_profile)
    
    # Reconstruct VMRs
    vmr_mols, vmr_atoms = reconstruct_vmrs(params, art, mol_names, atom_names)
    
    # Compute mean molecular weight and H2/He
    mmw, vmrH2, vmrHe = reconstruct_mmw_and_h2he(
        vmr_mols, vmr_atoms, mol_names, atom_names, art
    )
    
    # Compute gravity profile
    Rp = params.get("Rp", 1.5) * RJ
    Mp = params.get("Mp", 1.0) * MJ
    g_btm = gravity_jupiter(Rp / RJ, Mp / MJ)
    g = art.gravity_profile(Tarr, mmw, Rp, g_btm)
    
    # Compute total dtau
    dtau = compute_dtau(
        art, opa_mols, opa_atoms, opa_cias, nu_grid,
        Tarr, vmr_mols, vmr_atoms, vmrH2, vmrHe, mmw, g
    )
    
    # Compute per-species dtau
    dtau_per_species = compute_dtau_per_species(
        art, opa_mols, opa_atoms, opa_cias, nu_grid,
        Tarr, vmr_mols, vmr_atoms, vmrH2, vmrHe, mmw, g
    )
    
    return {
        'dtau': dtau,
        'dtau_per_species': dtau_per_species,
        'Tarr': Tarr,
        'pressure': art.pressure,
        'dParr': art.dParr,
        'mmw': mmw,
        'vmrH2': vmrH2,
        'vmrHe': vmrHe,
        'vmr_mols': vmr_mols,
        'vmr_atoms': vmr_atoms,
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
    # temperature (Default: Guillot)
    temperature_profile: Literal[
        "isothermal", "gradient", "madhu_seager", "free", "guillot"
    ] = "guillot",
    Tlow: float = 400.0,
    Thigh: float = 3000.0,
    Tirr_std: float | None = None,  # If None, uses uniform prior on Tirr
    Tint_fixed: float = 100.0,
    log_kappa_ir_bounds: tuple[float, float] = (-4.0, 0.0),
    log_gamma_bounds: tuple[float, float] = (-2.0, 2.0),
    # Phase-dependent velocity modeling
    phase_mode: PhaseMode = "shared",
    # Pipeline options
    subtract_per_exposure_mean: bool = True,
    apply_sysrem: bool = False,
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
        temperature_profile: TP profile type
        Tlow, Thigh: Temperature bounds (K)
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

    if (mode == "emission") and (Tstar is None):
        raise ValueError("Tstar is required for emission mode.")

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

        # 2. Composition
        vmr_mols_list = []
        for mol in mol_names:
            logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(-15.0, 0.0))
            vmr_mols_list.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
        
        # Handle shape if no molecules (edge case)
        if len(mol_names) > 0:
            vmr_mols = jnp.array(vmr_mols_list)
        else:
            vmr_mols = jnp.zeros((0, art.pressure.size))

        vmr_atoms_list = []
        for atom in atom_names:
            logVMR = numpyro.sample(f"logVMR_{atom}", dist.Uniform(-15.0, 0.0))
            vmr_atoms_list.append(art.constant_mmr_profile(jnp.power(10.0, logVMR)))
        
        if len(atom_names) > 0:
            vmr_atoms = jnp.array(vmr_atoms_list)
            sum_atoms = jnp.sum(vmr_atoms, axis=0)
            dot_atoms = jnp.dot(atom_masses, vmr_atoms)
        else:
            vmr_atoms = jnp.zeros((0, art.pressure.size))
            sum_atoms = 0.0
            dot_atoms = 0.0

        if len(mol_names) > 0:
            sum_mols = jnp.sum(vmr_mols, axis=0)
            dot_mols = jnp.dot(mol_masses, vmr_mols)
        else:
            sum_mols = 0.0
            dot_mols = 0.0

        vmr_tot = jnp.clip(sum_mols + sum_atoms, 0.0, 1.0)
        vmrH2 = (1.0 - vmr_tot) * (6.0 / 7.0)
        vmrHe = (1.0 - vmr_tot) * (1.0 / 7.0)

        mmw = (
            molinfo.molmass_isotope("H2") * vmrH2
            + molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
            + dot_mols
            + dot_atoms
        )

        # 3. Temperature & Gravity
        g_btm = gravity_jupiter(Rp / RJ, Mp / MJ)

        if temperature_profile == "guillot":
            if (Tirr_mean is not None) and (Tirr_std is not None):
                Tirr = numpyro.sample(
                    "Tirr", dist.TruncatedNormal(Tirr_mean, Tirr_std, low=0.0)
                )
            else:
                Tirr = numpyro.sample("Tirr", dist.Uniform(Tlow, Thigh))

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

        elif temperature_profile == "isothermal":
            T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
            Tarr = T0 * jnp.ones_like(art.pressure)
        elif temperature_profile == "gradient":
            Tarr = numpyro_gradient(art, Tlow, Thigh)
        elif temperature_profile == "madhu_seager":
            Tarr = numpyro_madhu_seager(art, Tlow, Thigh)
        elif temperature_profile == "free":
            Tarr = numpyro_free_temperature(art, n_layers=5, Tlow=Tlow, Thigh=Thigh)
        else:
            raise ValueError(f"Unknown temperature profile: {temperature_profile}")

        g = art.gravity_profile(Tarr, mmw, Rp, g_btm)

        # 4. Opacity Calculation (CIA + Mols + Atoms)
        dtau = jnp.zeros((art.pressure.size, nu_grid.size))

        # CIA
        for molA, molB in [("H2", "H2"), ("H2", "He")]:
            key = molA + molB
            if key not in opa_cias:
                continue
            logacia_matrix = opa_cias[key].logacia_matrix(Tarr)
            vmrX, vmrY = (vmrH2, vmrH2) if molB == "H2" else (vmrH2, vmrHe)
            dtau = dtau + art.opacity_profile_cia(
                logacia_matrix, Tarr, vmrX, vmrY, mmw[:, None], g
            )

        # Molecules
        for i, mol in enumerate(mol_names):
            xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, vmr_mols[i], mmw[:, None], g
            )

        # Atoms
        for i, atom in enumerate(atom_names):
            xsmatrix = opa_atoms[atom].xsmatrix(Tarr, art.pressure)
            dtau = dtau + art.opacity_profile_xs(
                xsmatrix, vmr_atoms[i], mmw[:, None], g
            )

        # 5. Radiative Transfer
        rt = art.run(dtau, Tarr, mmw, Rp, g_btm)

        # 6. Rotation & Instrument Broadening
        # Tidal locking: Spin period = Orbital period
        vsini = (
            2.0 * jnp.pi * Rp / (period_day * 86400.0) / 1.0e5
        )  # cm/s -> km/s
        
        rt = sop_rot.rigid_rotation(rt, vsini, 0.0, 0.0)
        
        # Apply Instrument Profile (Calculated from R above)
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
        # Ensure 2D shapes before any per-exposure operations
        if model_ts.ndim == 1:
            model_ts = model_ts[None, :]
        if data.ndim == 1:
            data = data[None, :]
            sigma = sigma[None, :]
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

        # Deterministics for tracking
        numpyro.deterministic("Kp_kms", Kp)
        numpyro.deterministic("Vsys_kms", Vsys)
        numpyro.deterministic("dRV_kms", dRV)
        numpyro.deterministic("vsini_kms", vsini)

    return model
