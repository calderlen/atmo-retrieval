from __future__ import annotations

import importlib
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
CIA_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("H2H2", "H2", "H2"),
    ("H2He", "H2", "He"),
)


def _get_piBarr():
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
        kappa_ir_cgs = posterior_params["kappa_ir_cgs"]
        gamma = posterior_params["gamma"]
        
        Rp = posterior_params["Rp"]
        Mp = posterior_params["Mp"]
        g_ref = gravity_surface(Rp, Mp)
        
        Tarr = guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_ref,
            Tirr=Tirr,
            Tint=Tint_fixed,
            kappa_ir_cgs=kappa_ir_cgs,
            gamma=gamma,
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
    kappa_ir_cgs_bounds: tuple[float, float] | None = None,
    gamma_bounds: tuple[float, float] | None = None,
    # Phase-dependent velocity modeling
    phase_mode: PhaseMode = config.DEFAULT_PHASE_MODE,
    # Pipeline options
    subtract_per_exposure_mean: bool | None = None,
    apply_sysrem: bool | None = None,
    # Chemistry/composition solver
    composition_solver: CompositionSolver | None = None,
) -> Callable:
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
    if subtract_per_exposure_mean is None:
        subtract_per_exposure_mean = config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT
    if apply_sysrem is None:
        apply_sysrem = config.APPLY_SYSREM_DEFAULT
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

        # 2. Gravity & Temperature 
        g_ref = gravity_surface(Rp / RJ, Mp / MJ)

        if pt_profile == "guillot":
            if (Tirr_mean is not None) and (Tirr_std is not None):
                Tirr = numpyro.sample(
                    "Tirr", dist.TruncatedNormal(Tirr_mean, Tirr_std, low=0.0)
                )
            else:
                Tirr = numpyro.sample("Tirr", dist.Uniform(T_low, T_high))

            kappa_ir_cgs = numpyro.sample(
                "kappa_ir_cgs", dist.LogUniform(*kappa_ir_cgs_bounds)
            )
            gamma = numpyro.sample("gamma", dist.LogUniform(*gamma_bounds))

            Tarr = guillot_profile(
                pressure_bar=art.pressure,
                g_cgs=g_ref,
                Tirr=Tirr,
                Tint=Tint_fixed,
                kappa_ir_cgs=kappa_ir_cgs,
                gamma=gamma,
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

        # 3. Composition (sample VMR, convert to MMR for opacity calculations)
        comp = composition_solver.sample(
            mol_names, mol_masses, atom_names, atom_masses, art, Tarr=Tarr
        )
        mmr_mols = comp.mmr_mols
        mmr_atoms = comp.mmr_atoms
        vmrH2_profile = comp.vmrH2_profile
        vmrHe_profile = comp.vmrHe_profile
        mmw_profile = comp.mmw_profile

        g = art.gravity_profile(Tarr, mmw_profile, Rp, g_ref)

        # 4. Opacity Calculation + Forward Model
        # Ensure 2D shapes for downstream operations
        if data.ndim == 1:
            data = data[None, :]
            sigma = sigma[None, :]

        # Tidal locking: Spin period = Orbital period
        vsini = (
            2.0 * jnp.pi * Rp / (period_day * 86400.0) / 1.0e5
        )  # cm/s -> km/s

        mmr_mol_profiles = {mol: mmr_mols[i] for i, mol in enumerate(mol_names)}
        mmr_atom_profiles = {atom: mmr_atoms[i] for i, atom in enumerate(atom_names)}

        dtau = compute_opacity(
            art=art,
            opa_mols=opa_mols,
            opa_atoms=opa_atoms,
            opa_cias=opa_cias,
            nu_grid=nu_grid,
            Tarr=Tarr,
            mmr_mols=mmr_mol_profiles,
            mmr_atoms=mmr_atom_profiles,
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            g=g,
        )

        # 5. Radiative Transfer
        rt = art.run(dtau, Tarr, mmw_profile, Rp, g_ref)

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
                raise ValueError(
                    "apply_sysrem=True requires both U and invvar_spec inputs."
                )
            model_ts = sysrem_filter_model(model_ts, U, invvar_spec)

        # 10. Log-likelihood (CCF-equivalent matched filter)
        # i = spectrum/exposure index, j = pixel index
        w_ij = 1.0 / jnp.clip(sigma, EPS, None) ** 2

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

        # lnL = sum_i lnL_i
        lnL = jnp.sum(jax.vmap(_lnL_exposure)(data, model_ts, w_ij))
        numpyro.factor("logL", lnL)

        # Deterministics for tracking
        numpyro.deterministic("Kp_kms", Kp)
        numpyro.deterministic("Vsys_kms", Vsys)
        numpyro.deterministic("dRV_kms", dRV)
        numpyro.deterministic("vsini_kms", vsini)

    return model
