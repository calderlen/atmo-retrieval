from __future__ import annotations

import warnings
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

PhaseMode = Literal["global", "per_exposure", "linear"]

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
from physics.chemistry import CompositionSolver, ConstantVMR


EPS = 1.0e-30
CKMS = 299792.458


def _closest_divisor(n: int, target: int) -> int:
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
    nu_grid = np.asarray(nu_grid)
    if nu_grid.size < 2:
        return int(min_guard_points)
    dnu = float(np.median(np.abs(np.diff(nu_grid))))
    nu_ref = float(np.nanmax(nu_grid))

    # Instrument sigma in km/s (approx)
    sigma_v = CKMS / (instrument_resolution * 2.3548200450309493)

    # Planet rotation broadening (km/s) using mean Rp
    Rp_mean = float(params.get("R_p", config.DEFAULT_RP_MEAN))
    vsini = (2.0 * np.pi * (Rp_mean * RJ) / (period_day * 86400.0) / 1.0e5)

    # RV guard from priors (3-sigma) + dRV prior
    kp = float(params.get("Kp", config.DEFAULT_KP_MEAN))
    kp_err = float(params.get("Kp_err", config.DEFAULT_KP_ERR_MEAN))
    vsys = float(params.get("RV_abs", config.DEFAULT_RV_ABS_MEAN))
    vsys_err = float(params.get("RV_abs_err", config.DEFAULT_RV_ABS_ERR_MEAN))
    rv_guard = abs(vsys) + 3.0 * vsys_err + kp + 3.0 * kp_err + config.DEFAULT_RV_GUARD_EXTRA

    v_guard = rv_guard + max(vsini, config.DEFAULT_SIGMA_V_FACTOR * sigma_v)
    if guard_kms is not None:
        v_guard = max(float(guard_kms), v_guard)

    guard_points = int(np.ceil((nu_ref * v_guard / CKMS) / dnu))
    return max(int(min_guard_points), guard_points)


def _build_chunk_plan(
    nu_grid: np.ndarray,
    inst_nus: np.ndarray,
    guard_points: int,
    *,
    chunk_points: int | None = None,
    n_chunks: int | None = None,
) -> list[tuple[int, int, int, int, int, int]]:
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
    if matrix.shape[-1] == n_nu:
        return matrix[..., start:end]
    if matrix.shape[0] == n_nu:
        return matrix[start:end, ...]
    raise ValueError("Unexpected spectral matrix shape; cannot slice.")


def _get_piBarr():
    import importlib

    mod = importlib.import_module("exojax.spec.planck")
    return mod.piBarr


def _element_from_species(species_name: str) -> str:
    return species_name.split()[0]


def planet_rv_kms(
    phase: jnp.ndarray,
    Kp_kms: float,
    Vsys_kms: float,
    dRV_kms: float = 0.0,
) -> jnp.ndarray:
    return Kp_kms * jnp.sin(2.0 * jnp.pi * phase) + Vsys_kms + dRV_kms


def sysrem_filter_model(
    model_matrix: jnp.ndarray,
    U: jnp.ndarray,
    invvar_spec: jnp.ndarray,
) -> jnp.ndarray:
    UTW = U.T * invvar_spec[None, :]
    U_dag = jnp.linalg.solve(UTW @ U, UTW)
    return model_matrix - U @ (U_dag @ model_matrix)


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

def compute_opacity(
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


def compute_opacity_per_species(
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
    # posterior_params: parameter values
    # art: ExoJAX art object
    # pt_profile: P-T profile name
    # Tint_fixed: internal temperature for Guillot profile
    if pt_profile == "guillot":
        Tirr = posterior_params["Tirr"]
        log_kappa_ir = posterior_params["log_kappa_ir"]
        log_gamma = posterior_params["log_gamma"]
        
        Rp = posterior_params["Rp"]
        Mp = posterior_params["Mp"]
        g_ref = gravity_surface(Rp, Mp)
        
        Tarr = guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_ref,
            Tirr=Tirr,
            Tint=Tint_fixed,
            kappa_ir_cgs=jnp.power(10.0, log_kappa_ir),
            gamma=jnp.power(10.0, log_gamma),
        )
        
    elif pt_profile == "isothermal":
        T0 = posterior_params["T0"]
        Tarr = T0 * jnp.ones_like(art.pressure)

    elif pt_profile == "gradient":
        # Linear gradient in log-pressure
        T_btm = posterior_params["T_btm"]
        T_top = posterior_params["T_top"]
        log_p = jnp.log10(art.pressure)
        log_p_btm = jnp.log10(art.pressure[-1])
        log_p_top = jnp.log10(art.pressure[0])
        Tarr = T_top + (T_btm - T_top) * (log_p - log_p_top) / (log_p_btm - log_p_top)

    else:
        raise ValueError(f"Temperature reconstruction not implemented for profile: {pt_profile}")
    
    return Tarr


def reconstruct_vmr_scalars(
    posterior_params: dict,
    mol_names: list[str],
    atom_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    # posterior_params: parameter values
    # mol_names: molecule names
    # atom_names: atom names
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
    # vmr_mols: scalar VMR per molecule
    # vmr_atoms: scalar VMR per atom
    # mol_names: molecule names
    # atom_names: atom names
    # Compute molecular masses
    mol_masses = {m: molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names}
    atom_masses = {
        a: molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names
    }

    # Sum VMRs
    vmr_trace_tot = sum(vmr_mols.values()) + sum(vmr_atoms.values())
    vmr_trace_tot = min(max(vmr_trace_tot, 0.0), 1.0)

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
    pt_profile: str = "guillot",
    use_median: bool = True,
) -> dict:
    import numpy as np
    
    if use_median:
        params = {k: float(np.median(v)) for k, v in posterior_samples.items() 
                  if not k.startswith("_")}
    else:
        params = {k: float(np.mean(v)) for k, v in posterior_samples.items()
                  if not k.startswith("_")}

    mol_names = list(opa_mols.keys())
    atom_names = list(opa_atoms.keys())
    Tarr = reconstruct_temperature_profile(params, art, pt_profile)
    vmr_mols, vmr_atoms = reconstruct_vmr_scalars(params, mol_names, atom_names)
    mmw, vmrH2, vmrHe = compute_mmw_and_h2he_from_vmr(
        vmr_mols, vmr_atoms, mol_names, atom_names
    )
    mmr_mols, mmr_atoms = convert_vmr_to_mmr_profiles(
        vmr_mols, vmr_atoms, mol_names, atom_names, mmw, art
    )

    # Create constant profiles for H2/He VMR (for CIA) and mmw
    vmrH2_profile = art.constant_mmr_profile(vmrH2)
    vmrHe_profile = art.constant_mmr_profile(vmrHe)
    mmw_profile = art.constant_mmr_profile(mmw)

    # Compute gravity profile
    Rp = params.get("Rp", config.DEFAULT_POSTERIOR_RP) * RJ
    Mp = params.get("Mp", config.DEFAULT_POSTERIOR_MP) * MJ
    g_ref = gravity_surface(Rp / RJ, Mp / MJ)
    g = art.gravity_profile(Tarr, mmw_profile, Rp, g_ref)

    # Compute total dtau (pass MMR profiles for molecules/atoms, VMR for CIA)
    dtau = compute_opacity(
        art, opa_mols, opa_atoms, opa_cias, nu_grid,
        Tarr, mmr_mols, mmr_atoms, vmrH2_profile, vmrHe_profile, mmw_profile, g
    )

    # Compute per-species dtau
    dtau_per_species = compute_opacity_per_species(
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
    ] = config.PT_PROFILE_DEFAULT,
    T_low: float | None = None,
    T_high: float | None = None,
    Tirr_std: float | None = None,  # If None, uses uniform prior on Tirr
    Tint_fixed: float | None = None,
    log_kappa_ir_bounds: tuple[float, float] | None = None,
    log_gamma_bounds: tuple[float, float] | None = None,
    # Phase-dependent velocity modeling
    phase_mode: PhaseMode = config.DEFAULT_PHASE_MODE,
    # Pipeline options
    subtract_per_exposure_mean: bool | None = None,
    apply_sysrem: bool | None = None,
    # Inference grid stitching (optional)
    stitch_inference: bool | None = None,
    stitch_chunk_points: int | None = None,
    stitch_n_chunks: int | None = None,
    stitch_guard_kms: float | None = None,
    stitch_guard_points: int | None = None,
    stitch_min_guard_points: int | None = None,
    # Chemistry/composition solver
    composition_solver: CompositionSolver | None = None,
) -> Callable:
    if T_low is None:
        T_low = config.T_LOW
    if T_high is None:
        T_high = config.T_HIGH
    if Tint_fixed is None:
        Tint_fixed = config.TINT_FIXED
    if log_kappa_ir_bounds is None:
        log_kappa_ir_bounds = config.LOG_KAPPA_IR_BOUNDS
    if log_gamma_bounds is None:
        log_gamma_bounds = config.LOG_GAMMA_BOUNDS
    if subtract_per_exposure_mean is None:
        subtract_per_exposure_mean = config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT
    if apply_sysrem is None:
        apply_sysrem = config.APPLY_SYSREM_DEFAULT
    if stitch_inference is None:
        stitch_inference = config.ENABLE_INFERENCE_STITCHING
    if stitch_min_guard_points is None:
        stitch_min_guard_points = config.STITCH_MIN_GUARD_POINTS
    Kp_mean, Kp_std = params["Kp"], params["Kp_err"]
    Vsys_mean, Vsys_std = params["RV_abs"], params["RV_abs_err"]
    Rp_mean, Rp_std = params["R_p"], params["R_p_err"]
    Mp_mean, Mp_std = params["M_p"], params["M_p_err"]
    Rstar_mean, Rstar_std = params["R_star"], params["R_star_err"]
    Tstar = params["T_star"]
    Tirr_mean = params.get("T_eq")  # May be None or nan
    period_day = params["period"]

    if Tirr_mean is not None and (Tirr_mean != Tirr_mean):  # nan check
        Tirr_mean = None

    check_grid_resolution(nu_grid, instrument_resolution)

    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)

    mol_names = tuple(opa_mols.keys())
    atom_names = tuple(opa_atoms.keys())  # Mandatory argument, checking keys is safe

    mol_masses = jnp.array([molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names])
    if len(atom_names) > 0:
        atom_masses = jnp.array(
            [molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names]
        )
    else:
        atom_masses = jnp.zeros((0,))

    if composition_solver is None:
        composition_solver = ConstantVMR()

    if (mode == "emission") and (Tstar is None):
        raise ValueError("Tstar is required for emission mode.")

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

        vsini_max = float(getattr(sop_rot, "vsini_max", config.STITCH_VSINI_MAX))
        vrmax = float(getattr(sop_inst, "vrmax", config.STITCH_VRMAX))
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
        
        if phase_mode == "global":
            dRV = numpyro.sample("dRV", dist.Normal(0.0, 10.0))
            
        elif phase_mode == "per_exposure":
            with numpyro.plate("exposures", n_exp):
                dRV = numpyro.sample("dRV", dist.Normal(0.0, 10.0))
            numpyro.deterministic("dRV_mean", jnp.mean(dRV))
            numpyro.deterministic("dRV_std", jnp.std(dRV))

        elif phase_mode == "linear":
            dRV_0 = numpyro.sample("dRV_0", dist.Normal(0.0, 10.0))
            dRV_slope = numpyro.sample("dRV_slope", dist.Normal(0.0, 50.0))
            dRV = dRV_0 + dRV_slope * phase
            numpyro.deterministic("dRV_at_ingress", dRV_0 + dRV_slope * jnp.min(phase))
            numpyro.deterministic("dRV_at_egress", dRV_0 + dRV_slope * jnp.max(phase))
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
        g_ref = gravity_surface(Rp / RJ, Mp / MJ)

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
                g_cgs=g_ref,
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

        g = art.gravity_profile(Tarr, mmw_prof, Rp, g_ref)

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
            rt = art.run(dtau, Tarr, mmw_prof, Rp, g_ref)

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
                model_ts = planet_ts / jnp.clip(Fs_ts, EPS, None) * (Rp / Rstar) ** 2

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
            w = 1.0 / jnp.clip(sigma, EPS, None) ** 2

            def _lnL_one(f: jnp.ndarray, m: jnp.ndarray, w_: jnp.ndarray) -> jnp.ndarray:
                s_mm = jnp.sum((m * m) * w_)
                s_fm = jnp.sum((f * m) * w_)
                alpha = s_fm / (s_mm + EPS)
                r = f - alpha * m
                chi2 = jnp.sum((r * r) * w_)
                norm = jnp.sum(jnp.log((2.0 * jnp.pi) / w_))
                return -0.5 * (chi2 + norm)

            lnL = jnp.sum(jax.vmap(_lnL_one)(data, model_ts, w))
            numpyro.factor("logL", lnL)
        else:
            # Stitched inference: accumulate log-likelihood across chunks
            w = 1.0 / jnp.clip(sigma, EPS, None) ** 2
            sum_w = jnp.sum(w, axis=1)
            sum_f_w = jnp.sum(data * w, axis=1)
            s_ff = jnp.sum((data * data) * w, axis=1)
            norm = jnp.sum(jnp.log((2.0 * jnp.pi) / w), axis=1)

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
                    xsmatrix = opa_mols[mol].xsmatrix(Tarr, art.pressure)
                    xs_chunk = _slice_spectral_matrix(
                        xsmatrix, chunk_start, chunk_end, nu_grid.size
                    )
                    dtau_chunk = dtau_chunk + art.opacity_profile_xs(
                        xs_chunk, mmr_mols[i], mmw_prof[:, None], g
                    )

                # Atoms (use MMR for opacity_profile_xs)
                for i, atom in enumerate(atom_names):
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

                rt = art.run(dtau_chunk, Tarr, mmw_prof, Rp, g_ref)
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
                    model_ts = planet_ts / jnp.clip(Fs_ts, EPS, None) * (Rp / Rstar) ** 2

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

            alpha = s_fm / (s_mm + EPS)
            chi2 = s_ff - 2.0 * alpha * s_fm + (alpha * alpha) * s_mm
            lnL = jnp.sum(-0.5 * (chi2 + norm))
            numpyro.factor("logL", lnL)

        # Deterministics for tracking
        numpyro.deterministic("Kp_kms", Kp)
        numpyro.deterministic("Vsys_kms", Vsys)
        numpyro.deterministic("dRV_kms", dRV)
        numpyro.deterministic("vsini_kms", vsini)

    return model
