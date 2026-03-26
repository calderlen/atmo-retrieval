from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

RetrievalMode = Literal["transmission", "emission"]
PhaseMode = Literal["global", "per_exposure", "linear"]
PTProfileMode = Literal[
    "guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"
]

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


@dataclass(frozen=True)
class RetrievalModelConfig:
    mode: RetrievalMode
    art: object
    opa_mols: dict[str, OpaPremodit]
    opa_atoms: dict[str, OpaPremodit]
    opa_cias: dict[str, OpaCIA]
    nu_grid: jnp.ndarray
    sop_rot: SopRotation
    sop_inst: SopInstProfile
    inst_nus: jnp.ndarray
    pt_profile: PTProfileMode
    T_low: float
    T_high: float
    Tirr_std: float | None
    Tint_fixed: float
    kappa_ir_cgs_bounds: tuple[float, float]
    gamma_bounds: tuple[float, float]
    phase_mode: PhaseMode
    subtract_per_exposure_mean: bool
    apply_sysrem: bool
    composition_solver: CompositionSolver
    beta_inst: float
    mol_names: tuple[str, ...]
    atom_names: tuple[str, ...]
    mol_masses: jnp.ndarray
    atom_masses: jnp.ndarray
    Kp_mean: float
    Kp_std: float
    Vsys_mean: float
    Vsys_std: float
    Rp_mean: float
    Rp_std: float
    Mp_mean: float
    Mp_std: float
    Rstar_mean: float
    Rstar_std: float
    Tstar: float | None
    Tirr_mean: float | None
    period_day: float


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
    model_config: RetrievalModelConfig,
    g_ref: float | jnp.ndarray,
) -> jnp.ndarray:
    art = model_config.art
    pt_profile = model_config.pt_profile

    if pt_profile == "guillot":
        if (model_config.Tirr_mean is not None) and (model_config.Tirr_std is not None):
            Tirr = numpyro.sample(
                "Tirr",
                dist.TruncatedNormal(model_config.Tirr_mean, model_config.Tirr_std, low=0.0),
            )
        else:
            Tirr = numpyro.sample("Tirr", dist.Uniform(model_config.T_low, model_config.T_high))

        kappa_ir_cgs = numpyro.sample(
            "kappa_ir_cgs",
            dist.LogUniform(*model_config.kappa_ir_cgs_bounds),
        )
        gamma = numpyro.sample("gamma", dist.LogUniform(*model_config.gamma_bounds))

        return guillot_profile(
            pressure_bar=art.pressure,
            g_cgs=g_ref,
            Tirr=Tirr,
            Tint=model_config.Tint_fixed,
            kappa_ir_cgs=kappa_ir_cgs,
            gamma=gamma,
        )

    if pt_profile == "isothermal":
        T0 = numpyro.sample("T0", dist.Uniform(model_config.T_low, model_config.T_high))
        return T0 * jnp.ones_like(art.pressure)

    if pt_profile == "gradient":
        return numpyro_gradient(art, model_config.T_low, model_config.T_high)

    if pt_profile == "madhu_seager":
        return numpyro_madhu_seager(art, model_config.T_low, model_config.T_high)

    if pt_profile == "free":
        return numpyro_free_temperature(
            art,
            n_layers=5,
            T_low=model_config.T_low,
            T_high=model_config.T_high,
        )

    if pt_profile == "pspline":
        return numpyro_pspline_knots_on_art_grid(
            art,
            T_low=model_config.T_low,
            T_high=model_config.T_high,
        )

    if pt_profile == "gp":
        return numpyro_gp_temperature(
            art,
            T_low=model_config.T_low,
            T_high=model_config.T_high,
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

    piBarr = _get_piBarr()
    Fs = piBarr(nu_grid, Tstar)
    Fs_ts = jax.vmap(lambda v: sop_inst.sampling(Fs, v, inst_nus))(rv)
    return planet_ts / jnp.clip(Fs_ts, EPS, None) * (Rp / Rstar) ** 2


def apply_model_pipeline_corrections(
    model_ts: jnp.ndarray,
    *,
    subtract_per_exposure_mean: bool,
    apply_sysrem: bool,
    U: jnp.ndarray | None = None,
    invvar_spec: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if model_ts.ndim == 1:
        model_ts = model_ts[None, :]
    if model_ts.shape[1] == 0:
        raise ValueError("model_ts has zero spectral points; check inst_nus/nu_grid")

    if subtract_per_exposure_mean:
        model_ts = model_ts - jnp.mean(model_ts, axis=1, keepdims=True)

    if apply_sysrem:
        if (U is None) or (invvar_spec is None):
            raise ValueError("apply_sysrem=True requires both U and invvar_spec inputs.")
        model_ts = sysrem_filter_model(model_ts, U, invvar_spec)

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


def retrieval_model(
    model_config: RetrievalModelConfig,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    phase: jnp.ndarray,
    U: jnp.ndarray | None = None,
    invvar_spec: jnp.ndarray | None = None,
) -> None:
    data = jnp.asarray(data)
    sigma = jnp.asarray(sigma)
    phase = jnp.asarray(phase)
    U = None if U is None else jnp.asarray(U)
    invvar_spec = None if invvar_spec is None else jnp.asarray(invvar_spec)

    # 1. System parameters
    Kp = numpyro.sample("Kp", dist.TruncatedNormal(model_config.Kp_mean, model_config.Kp_std, low=0.0))
    Vsys = numpyro.sample("Vsys", dist.Normal(model_config.Vsys_mean, model_config.Vsys_std))
    dRV = _sample_phase_dependent_velocity_offset(model_config.phase_mode, phase)

    Mp = numpyro.sample("Mp", dist.TruncatedNormal(model_config.Mp_mean, model_config.Mp_std, low=0.0)) * MJ
    Rstar = numpyro.sample(
        "Rstar",
        dist.TruncatedNormal(model_config.Rstar_mean, model_config.Rstar_std, low=0.0),
    ) * Rs
    Rp = numpyro.sample("Rp", dist.TruncatedNormal(model_config.Rp_mean, model_config.Rp_std, low=0.0)) * RJ

    # 2. Gravity and temperature
    g_ref = gravity_surface(Rp / RJ, Mp / MJ)
    Tarr = _sample_temperature_profile(model_config, g_ref)

    # 3. Composition (sample VMR, convert to MMR for opacity calculations)
    comp = model_config.composition_solver.sample(
        model_config.mol_names,
        model_config.mol_masses,
        model_config.atom_names,
        model_config.atom_masses,
        model_config.art,
        Tarr=Tarr,
    )
    mmr_mols = comp.mmr_mols
    mmr_atoms = comp.mmr_atoms
    vmrH2_profile = comp.vmrH2_profile
    vmrHe_profile = comp.vmrHe_profile
    mmw_profile = comp.mmw_profile

    g = model_config.art.gravity_profile(Tarr, mmw_profile, Rp, g_ref)

    # 4. Opacity calculation
    if data.ndim == 1:
        data = data[None, :]
        sigma = sigma[None, :]

    mmr_mol_profiles = {
        mol: mmr_mols[i] for i, mol in enumerate(model_config.mol_names)
    }
    mmr_atom_profiles = {
        atom: mmr_atoms[i] for i, atom in enumerate(model_config.atom_names)
    }

    dtau = compute_opacity(
        art=model_config.art,
        opa_mols=model_config.opa_mols,
        opa_atoms=model_config.opa_atoms,
        opa_cias=model_config.opa_cias,
        nu_grid=model_config.nu_grid,
        Tarr=Tarr,
        mmr_mols=mmr_mol_profiles,
        mmr_atoms=mmr_atom_profiles,
        vmrH2_profile=vmrH2_profile,
        vmrHe_profile=vmrHe_profile,
        mmw_profile=mmw_profile,
        g=g,
    )

    # 5. Generate model observable
    model_ts = compute_model_timeseries(
        mode=model_config.mode,
        art=model_config.art,
        dtau=dtau,
        Tarr=Tarr,
        mmw_profile=mmw_profile,
        Rp=Rp,
        Rstar=Rstar,
        g_ref=g_ref,
        phase=phase,
        Kp=Kp,
        Vsys=Vsys,
        dRV=dRV,
        sop_rot=model_config.sop_rot,
        sop_inst=model_config.sop_inst,
        inst_nus=model_config.inst_nus,
        nu_grid=model_config.nu_grid,
        beta_inst=model_config.beta_inst,
        period_day=model_config.period_day,
        Tstar=model_config.Tstar,
    )
    model_ts = apply_model_pipeline_corrections(
        model_ts,
        subtract_per_exposure_mean=model_config.subtract_per_exposure_mean,
        apply_sysrem=model_config.apply_sysrem,
        U=U,
        invvar_spec=invvar_spec,
    )

    # 6. Log-likelihood (CCF-equivalent matched filter)
    w_ij = 1.0 / jnp.clip(sigma, EPS, None) ** 2
    lnL = jnp.sum(jax.vmap(_lnL_exposure)(data, model_ts, w_ij))
    numpyro.factor("logL", lnL)

    # Deterministics for tracking
    numpyro.deterministic("Kp_kms", Kp)
    numpyro.deterministic("Vsys_kms", Vsys)
    numpyro.deterministic("dRV_kms", dRV)
    numpyro.deterministic(
        "vsini_kms",
        2.0 * jnp.pi * Rp / (model_config.period_day * 86400.0) / 1.0e5,
    )


def build_retrieval_model_config(
    *,
    mode: RetrievalMode,
    params: dict,
    art: object,
    opa_mols: dict[str, OpaPremodit],
    opa_atoms: dict[str, OpaPremodit],
    opa_cias: dict[str, OpaCIA],
    nu_grid: jnp.ndarray,
    sop_rot: SopRotation,
    sop_inst: SopInstProfile,
    instrument_resolution: float,
    inst_nus: jnp.ndarray,
    pt_profile: PTProfileMode = config.PT_PROFILE_DEFAULT,
    T_low: float | None = None,
    T_high: float | None = None,
    Tirr_std: float | None = None,
    Tint_fixed: float | None = None,
    kappa_ir_cgs_bounds: tuple[float, float] | None = None,
    gamma_bounds: tuple[float, float] | None = None,
    phase_mode: PhaseMode = config.DEFAULT_PHASE_MODE,
    subtract_per_exposure_mean: bool | None = None,
    apply_sysrem: bool | None = None,
    composition_solver: CompositionSolver | None = None,
) -> RetrievalModelConfig:
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
    if composition_solver is None:
        composition_solver = ConstantVMR()

    Kp_mean, Kp_std = params["Kp"], params["Kp_err"]
    Vsys_mean, Vsys_std = params["RV_abs"], params["RV_abs_err"]
    Rp_mean, Rp_std = params["R_p"], params["R_p_err"]
    Mp_mean, Mp_std = params["M_p"], params["M_p_err"]
    Rstar_mean, Rstar_std = params["R_star"], params["R_star_err"]
    Tstar = params["T_star"]
    Tirr_mean = params.get("T_eq")
    period_day = params["period"]

    if Tirr_mean is not None and (Tirr_mean != Tirr_mean):
        Tirr_mean = None

    nu_grid = jnp.asarray(nu_grid)
    inst_nus = jnp.asarray(inst_nus)
    check_grid_resolution(nu_grid, instrument_resolution)
    beta_inst = 1.0 / (instrument_resolution * 2.3548200450309493)

    mol_names = tuple(opa_mols.keys())
    atom_names = tuple(opa_atoms.keys())
    mol_masses = jnp.array(
        [molinfo.molmass_isotope(m, db_HIT=False) for m in mol_names]
    )
    if atom_names:
        atom_masses = jnp.array(
            [molinfo.molmass_isotope(_element_from_species(a), db_HIT=False) for a in atom_names]
        )
    else:
        atom_masses = jnp.zeros((0,))

    if (mode == "emission") and (Tstar is None):
        raise ValueError("Tstar is required for emission mode.")

    return RetrievalModelConfig(
        mode=mode,
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        inst_nus=inst_nus,
        pt_profile=pt_profile,
        T_low=T_low,
        T_high=T_high,
        Tirr_std=Tirr_std,
        Tint_fixed=Tint_fixed,
        kappa_ir_cgs_bounds=kappa_ir_cgs_bounds,
        gamma_bounds=gamma_bounds,
        phase_mode=phase_mode,
        subtract_per_exposure_mean=subtract_per_exposure_mean,
        apply_sysrem=apply_sysrem,
        composition_solver=composition_solver,
        beta_inst=beta_inst,
        mol_names=mol_names,
        atom_names=atom_names,
        mol_masses=mol_masses,
        atom_masses=atom_masses,
        Kp_mean=Kp_mean,
        Kp_std=Kp_std,
        Vsys_mean=Vsys_mean,
        Vsys_std=Vsys_std,
        Rp_mean=Rp_mean,
        Rp_std=Rp_std,
        Mp_mean=Mp_mean,
        Mp_std=Mp_std,
        Rstar_mean=Rstar_mean,
        Rstar_std=Rstar_std,
        Tstar=Tstar,
        Tirr_mean=Tirr_mean,
        period_day=period_day,
    )


def create_retrieval_model(
    *,
    mode: RetrievalMode,
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
    pt_profile: PTProfileMode = config.PT_PROFILE_DEFAULT,
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
    model_config = build_retrieval_model_config(
        mode=mode,
        params=params,
        art=art,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus,
        pt_profile=pt_profile,
        T_low=T_low,
        T_high=T_high,
        Tirr_std=Tirr_std,
        Tint_fixed=Tint_fixed,
        kappa_ir_cgs_bounds=kappa_ir_cgs_bounds,
        gamma_bounds=gamma_bounds,
        phase_mode=phase_mode,
        subtract_per_exposure_mean=subtract_per_exposure_mean,
        apply_sysrem=apply_sysrem,
        composition_solver=composition_solver,
    )
    return partial(retrieval_model, model_config)
