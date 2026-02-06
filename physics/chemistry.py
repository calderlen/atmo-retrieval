from __future__ import annotations

from functools import lru_cache
import re
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from exojax.database import molinfo

from config import chemistry_config as chem_config


class CompositionState(NamedTuple):

    vmr_mols: list[jnp.ndarray]  # Scalar VMR per molecule
    vmr_atoms: list[jnp.ndarray]  # Scalar VMR per atom
    vmrH2: jnp.ndarray  # H2 VMR (scalar)
    vmrHe: jnp.ndarray  # He VMR (scalar)
    mmw: jnp.ndarray  # Mean molecular weight (scalar)
    mmr_mols: jnp.ndarray  # MMR profiles (n_mols, n_layers)
    mmr_atoms: jnp.ndarray  # MMR profiles (n_atoms, n_layers)
    vmrH2_prof: jnp.ndarray  # H2 VMR profile (n_layers,)
    vmrHe_prof: jnp.ndarray  # He VMR profile (n_layers,)
    mmw_prof: jnp.ndarray  # MMW profile (n_layers,)


class CompositionSolver(Protocol):

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        ...


class ConstantVMR:

    def __init__(
        self,
        log_vmr_min: float = chem_config.LOG_VMR_MIN,
        log_vmr_max: float = chem_config.LOG_VMR_MAX,
        h2_he_ratio: float = chem_config.H2_HE_RATIO,
    ):
        self.log_vmr_min = log_vmr_min
        self.log_vmr_max = log_vmr_max
        self.h2_he_ratio = h2_he_ratio

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        vmr_mols_raw = []
        for mol in mol_names:
            logVMR = numpyro.sample(
                f"logVMR_{mol}", dist.Uniform(self.log_vmr_min, self.log_vmr_max)
            )
            vmr_mols_raw.append(jnp.power(10.0, logVMR))

        vmr_atoms_raw = []
        for atom in atom_names:
            logVMR = numpyro.sample(
                f"logVMR_{atom}", dist.Uniform(self.log_vmr_min, self.log_vmr_max)
            )
            vmr_atoms_raw.append(jnp.power(10.0, logVMR))

        n_mols = len(vmr_mols_raw)
        n_atoms = len(vmr_atoms_raw)

        # Normalize trace species VMRs (handles empty arrays naturally)
        vmr_trace_arr = jnp.array(vmr_mols_raw + vmr_atoms_raw)
        sum_trace = jnp.sum(vmr_trace_arr)
        scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
        vmr_trace_arr = vmr_trace_arr * scale
        vmr_mols_scalar = [vmr_trace_arr[i] for i in range(n_mols)]
        vmr_atoms_scalar = [vmr_trace_arr[n_mols + i] for i in range(n_atoms)]
        vmr_trace_tot = jnp.sum(vmr_trace_arr)

        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe = (1.0 - vmr_trace_tot) * he_frac

        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw = mass_H2 * vmrH2 + mass_He * vmrHe
        mmw = mmw + sum(m * v for m, v in zip(mol_masses, vmr_mols_scalar))
        mmw = mmw + sum(m * v for m, v in zip(atom_masses, vmr_atoms_scalar))

        # MMR_i = VMR_i * (M_i / mmw) - ensure correct 2D shape for empty cases
        mmr_mols = (
            jnp.array([
                art.constant_mmr_profile(vmr * (mass / mmw))
                for vmr, mass in zip(vmr_mols_scalar, mol_masses)
            ]) if n_mols > 0 else jnp.zeros((0, art.pressure.size))
        )

        mmr_atoms = (
            jnp.array([
                art.constant_mmr_profile(vmr * (mass / mmw))
                for vmr, mass in zip(vmr_atoms_scalar, atom_masses)
            ]) if n_atoms > 0 else jnp.zeros((0, art.pressure.size))
        )

        # Step 6: Create constant profiles for CIA inputs and mmw
        vmrH2_prof = art.constant_mmr_profile(vmrH2)
        vmrHe_prof = art.constant_mmr_profile(vmrHe)
        mmw_prof = art.constant_mmr_profile(mmw)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )

# ---------------------------------------------------------------------------
# Solar abundances (Asplund et al. 2021, A&A 653, A141)
# Values are number fractions relative to H (n_X / n_H)
# ---------------------------------------------------------------------------
SOLAR_ABUNDANCES = {
    # Molecules (approximate VMR in solar-composition gas at ~1000K)
    "H2O": 5.4e-4,
    "CO": 6.0e-4,
    "CO2": 1.0e-7,
    "CH4": 3.5e-4,
    "NH3": 8.0e-5,
    "H2S": 1.5e-5,
    "HCN": 1.0e-7,
    "C2H2": 1.0e-8,
    "PH3": 3.0e-7,
    "TiO": 1.0e-7,
    "VO": 1.0e-8,
    "FeH": 1.0e-8,
    "SiO": 3.5e-5,
    # Atoms (solar photospheric, log eps scale converted to n_X/n_H)
    "Fe": 3.16e-5,   # log eps = 7.50
    "Fe I": 3.16e-5,
    "Fe II": 3.16e-6,  # Rough ionization fraction
    "Na": 2.14e-6,   # log eps = 6.33
    "Na I": 2.14e-6,
    "K": 1.35e-7,    # log eps = 5.13
    "K I": 1.35e-7,
    "Ca": 2.19e-6,   # log eps = 6.34
    "Ca I": 2.19e-6,
    "Ca II": 2.19e-6,
    "Mg": 3.98e-5,   # log eps = 7.60
    "Mg I": 3.98e-5,
    "Ti": 8.91e-8,   # log eps = 4.95
    "Ti I": 8.91e-8,
    "Ti II": 8.91e-8,
    "V": 1.00e-8,    # log eps = 3.93
    "V I": 1.00e-8,
    "Cr": 4.68e-7,   # log eps = 5.67
    "Cr I": 4.68e-7,
    "Mn": 3.47e-7,   # log eps = 5.54
    "Mn I": 3.47e-7,
    "Ni": 1.78e-6,   # log eps = 6.25
    "Ni I": 1.78e-6,
    "Si": 3.24e-5,   # log eps = 7.51
    "Si I": 3.24e-5,
    "Al": 2.82e-6,   # log eps = 6.45
    "Al I": 2.82e-6,
    "Li": 1.0e-9,    # log eps = 1.05 (depleted)
    "Li I": 1.0e-9,
    "C": 2.69e-4,    # log eps = 8.43 (total carbon)
    "O": 4.90e-4,    # log eps = 8.69 (total oxygen)
    "N": 6.76e-5,    # log eps = 7.83 (total nitrogen)
}

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


@lru_cache(maxsize=None)
def _count_co(species: str) -> tuple[int, int]:
    base = species.split()[0].replace("+", "").replace("-", "")
    n_c = 0
    n_o = 0
    for element, count_str in _ELEMENT_RE.findall(base):
        count = int(count_str) if count_str else 1
        if element == "C":
            n_c += count
        elif element == "O":
            n_o += count
    return n_c, n_o


class FreeVMR:

    def __init__(
        self,
        n_nodes: int = chem_config.N_VMR_NODES,
        log_vmr_min: float = chem_config.LOG_VMR_MIN,
        log_vmr_max: float = chem_config.LOG_VMR_MAX,
        h2_he_ratio: float = chem_config.H2_HE_RATIO,
    ):
        self.n_nodes = n_nodes
        self.log_vmr_min = log_vmr_min
        self.log_vmr_max = log_vmr_max
        self.h2_he_ratio = h2_he_ratio

    def _sample_vmr_profile(
        self,
        name: str,
        log_p: jnp.ndarray,
        log_p_nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        logVMR_nodes = []
        for i in range(self.n_nodes):
            logVMR_i = numpyro.sample(
                f"logVMR_{name}_node{i}",
                dist.Uniform(self.log_vmr_min, self.log_vmr_max),
            )
            logVMR_nodes.append(logVMR_i)
        logVMR_nodes = jnp.array(logVMR_nodes)

        logVMR_profile = jnp.interp(log_p, log_p_nodes, logVMR_nodes)
        return jnp.power(10.0, logVMR_profile)

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        log_p = jnp.log10(art.pressure)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), self.n_nodes)
        n_layers = art.pressure.size
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_prof = self._sample_vmr_profile(mol, log_p, log_p_nodes)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_prof = self._sample_vmr_profile(atom, log_p, log_p_nodes)
            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize at each layer if total trace VMR exceeds 1 (handles empty naturally)
        all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
        sum_trace = jnp.sum(all_profiles, axis=0) if all_profiles.size > 0 else jnp.zeros(n_layers)
        scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
        all_profiles = all_profiles * scale[None, :] if all_profiles.size > 0 else all_profiles
        vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
        vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
        vmr_trace_tot = jnp.sum(all_profiles, axis=0) if all_profiles.size > 0 else jnp.zeros(n_layers)

        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # MMR_i = VMR_i * (M_i / mmw) - ensure correct 2D shape for empty cases
        mmr_mols = (
            jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ]) if n_mols > 0 else jnp.zeros((0, n_layers))
        )

        mmr_atoms = (
            jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ]) if n_atoms > 0 else jnp.zeros((0, n_layers))
        )

        # For scalar outputs, use column-averaged values (pressure-weighted would be better)
        # Here we just use simple mean for consistency with downstream code that expects scalars
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )

# TODO: equilibrium chemistry solver might be a good prior/baseline. but really going to need to either keep VMRs free or use a hybrid of quench + equilibrum + disequilibrium + photochemistry + ... for realistic modeling
# TODO: partial ionization (Saha)
# TODO: H- continuum affecting line-to-continuum ratio
# TODO: vertical transport parameterized by K_zz and then like tchem and t_dyn or more simply the pestimate of quench level
# TODO: use CO as a proxy/tracer for the overall metallicity M/H of the atmosphere
# TODO: The continuum problem. Public low resolution spectra? R ~ 10^2 - 10^3??


class EquilibriumChemistry(CompositionSolver):
    """Simple equilibrium-like chemistry with [M/H] and C/O as free parameters."""

    def __init__(
        self,
        metallicity_range: tuple[float, float] = chem_config.METALLICITY_RANGE,
        co_ratio_range: tuple[float, float] = chem_config.CO_RATIO_RANGE,
        h2_he_ratio: float = chem_config.H2_HE_RATIO,
    ):
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self._c_solar = SOLAR_ABUNDANCES["C"]
        self._o_solar = SOLAR_ABUNDANCES["O"]

    def _co_scales(self, co_ratio: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        total_co = self._c_solar + self._o_solar
        c_target = total_co * (co_ratio / (1.0 + co_ratio))
        o_target = total_co * (1.0 / (1.0 + co_ratio))
        c_scale = c_target / self._c_solar
        o_scale = o_target / self._o_solar
        return c_scale, o_scale

    def _co_factor(
        self,
        n_c: int,
        n_o: int,
        c_scale: jnp.ndarray,
        o_scale: jnp.ndarray,
    ) -> jnp.ndarray:
        if n_c == 0 and n_o == 0:
            return jnp.array(1.0)
        if n_c > 0 and n_o > 0:
            return jnp.minimum(c_scale, o_scale)
        if n_c > 0:
            return c_scale
        return o_scale

    def _vmr_scalar(
        self,
        species: str,
        metallicity: jnp.ndarray,
        c_scale: jnp.ndarray,
        o_scale: jnp.ndarray,
    ) -> jnp.ndarray:
        solar = SOLAR_ABUNDANCES.get(species, 1e-10)
        n_c, n_o = _count_co(species)
        return solar * metallicity * self._co_factor(n_c, n_o, c_scale, o_scale)

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        c_scale, o_scale = self._co_scales(co_ratio)

        species = list(mol_names) + list(atom_names)
        n_mols = len(mol_names)
        n_atoms = len(atom_names)
        n_species = len(species)

        if n_species > 0:
            solar_arr = jnp.array([SOLAR_ABUNDANCES.get(s, 1e-10) for s in species])
            counts = [_count_co(s) for s in species]
            n_c = jnp.array([c[0] for c in counts], dtype=solar_arr.dtype)
            n_o = jnp.array([c[1] for c in counts], dtype=solar_arr.dtype)

            co_factor = jnp.where((n_c > 0) & (n_o > 0), jnp.minimum(c_scale, o_scale), 1.0)
            co_factor = jnp.where((n_c > 0) & (n_o == 0), c_scale, co_factor)
            co_factor = jnp.where((n_o > 0) & (n_c == 0), o_scale, co_factor)

            vmr_trace_arr = solar_arr * metallicity * co_factor
            sum_trace = jnp.sum(vmr_trace_arr)
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
            vmr_trace_arr = vmr_trace_arr * scale

            vmr_mols_arr = vmr_trace_arr[:n_mols]
            vmr_atoms_arr = vmr_trace_arr[n_mols:]
            vmr_mols_scalar = [vmr_mols_arr[i] for i in range(n_mols)]
            vmr_atoms_scalar = [vmr_atoms_arr[i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(vmr_trace_arr)
        else:
            vmr_mols_arr = jnp.zeros((0,))
            vmr_atoms_arr = jnp.zeros((0,))
            vmr_mols_scalar = []
            vmr_atoms_scalar = []
            vmr_trace_tot = jnp.array(0.0)

        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe = (1.0 - vmr_trace_tot) * he_frac

        mol_masses_arr = jnp.array(mol_masses)
        atom_masses_arr = jnp.array(atom_masses)

        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw = mass_H2 * vmrH2 + mass_He * vmrHe
        if n_mols > 0:
            mmw = mmw + jnp.sum(mol_masses_arr * vmr_mols_arr)
        if n_atoms > 0:
            mmw = mmw + jnp.sum(atom_masses_arr * vmr_atoms_arr)

        ones_prof = art.constant_mmr_profile(1.0)
        vmrH2_prof = vmrH2 * ones_prof
        vmrHe_prof = vmrHe * ones_prof
        mmw_prof = mmw * ones_prof

        if n_mols > 0:
            mmr_mols = (
                (vmr_mols_arr * (mol_masses_arr / mmw))[:, None] * ones_prof[None, :]
            )
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atoms = (
                (vmr_atoms_arr * (atom_masses_arr / mmw))[:, None] * ones_prof[None, :]
            )
        else:
            mmr_atoms = jnp.zeros((0, art.pressure.size))

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )

class QuenchedChemistry(CompositionSolver):
    pass

class Photochemistry:
    pass

class DisequilibriumChemistry:
    pass
