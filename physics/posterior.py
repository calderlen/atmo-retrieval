"""Reconstruct atmospheric states from saved posterior parameters."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import config
from exojax.database import molinfo
from exojax.opacity.opacont import OpaCIA
from exojax.opacity.premodit.api import OpaPremodit
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import MJ, RJ
from physics.pt import (
    gradient_profile,
    guillot_profile,
    madhu_seager_profile,
    pspline_knots_profile_on_grid,
)
from physics.chemistry import ConstantVMR, FastChemHybridChemistry, FreeVMR
from physics.model import AtmosphereRegionConfig, _element_from_species
from physics.opacity import (
    CONTINUUM_SPECIES_MASSES,
    compute_opacity,
    compute_opacity_per_species,
)


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
    return scoped


def _posterior_site_value(
    posterior_params: dict,
    site_name: str,
    *,
    local_params: dict | None = None,
):
    if local_params is not None:
        return local_params[site_name]
    return posterior_params[site_name]


def _collect_indexed_site_values(local_params: dict, *, prefix: str) -> list[object]:
    indexed_values: list[tuple[int, object]] = []
    for key, value in local_params.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        indexed_values.append((int(suffix), value))
    return [value for _, value in sorted(indexed_values)]


def _summarize_posterior_samples(posterior_samples: dict, *, use_median: bool) -> dict:
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
        R_ref = _posterior_site_value(posterior_params, "Rp")
        Mp = _posterior_site_value(posterior_params, "Mp")
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
        return Tarr


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
        logVMR = local_params[key]
        vmr_mols[mol] = float(jnp.power(10.0, logVMR))

    vmr_atoms = {}
    for atom in atom_names:
        key = f"logVMR_{atom}"
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

    def _profile_for_species(name: str) -> jnp.ndarray:
        log_vmr_nodes = _collect_indexed_site_values(
            local_params,
            prefix=f"logVMR_{name}_node",
        )
        log_vmr_nodes_arr = jnp.asarray(log_vmr_nodes)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), log_vmr_nodes_arr.size)
        log_vmr_profile = jnp.interp(log_p, log_p_nodes, log_vmr_nodes_arr)
        return jnp.power(10.0, log_vmr_profile)

    vmr_mols = {}
    for mol in mol_names:
        vmr_mols[mol] = _profile_for_species(mol)
    vmr_atoms = {}
    for atom in atom_names:
        vmr_atoms[atom] = _profile_for_species(atom)

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

    vmr_mols_scalar, vmr_atoms_scalar = reconstruct_vmr_scalars(
        posterior_params,
        [name for name in mol_names if not composition_solver.is_hybrid_managed_species(name)],
        [name for name in atom_names if not composition_solver.is_hybrid_managed_species(name)],
        sample_prefix=sample_prefix,
    )
    n_layers = art.pressure.size
    log_P = jnp.log10(art.pressure)

    mol_profile_map = {
        mol: jnp.full(n_layers, vmr) for mol, vmr in vmr_mols_scalar.items()
    }
    atom_profile_map = {
        atom: jnp.full(n_layers, vmr) for atom, vmr in vmr_atoms_scalar.items()
    }
    continuum_profile_map = {}
    needed = list(composition_solver.hidden_continuum_species())
    if needed:
        if composition_solver._hybrid_vmr_grids is None or any(species not in composition_solver._hybrid_vmr_grids for species in needed):
            composition_solver._build_hybrid_grid(np.asarray(art.pressure), needed)

        for species in needed:
            vmr_profile = composition_solver._interp_4d_log_vmr(
                composition_solver._hybrid_vmr_grids[species],
                jnp.asarray(log_metallicity),
                jnp.asarray(co_ratio),
                Tarr,
                log_P,
            )
            continuum_profile_map[species] = vmr_profile
            for name in mol_names:
                if composition_solver._canonical_species_name(name) == species:
                    mol_profile_map[name] = vmr_profile
            for name in atom_names:
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
        for species, vmr_profile in extra_vmr_profiles.items():
            mmw = mmw + extra_species_masses[species] * vmr_profile

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
    params = _summarize_posterior_samples(posterior_samples, use_median=use_median)

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

    # ``Rp`` remains the persisted posterior site for compatibility; its
    # runtime meaning is the adopted reference radius at the RT lower
    # boundary.
    R_ref = params["Rp"] * RJ
    Mp = params["Mp"] * MJ
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
