from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
import re
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyfastchem
from exojax.database import molinfo

from config import chemistry_config as chem_config

logger = logging.getLogger(__name__)


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
        Tarr: jnp.ndarray | None = None,
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
        Tarr: jnp.ndarray | None = None,
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


# ---------------------------------------------------------------------------
# FastChem species name mapping
# Maps codebase names → FastChem gas species symbols
# FastChem uses Hill notation for molecules and element symbols for atoms.
# Ions use "Element1+" for singly ionized, etc.
# ---------------------------------------------------------------------------
_FASTCHEM_SPECIES_MAP: dict[str, str] = {
    # Molecules (HITEMP + ExoMol)
    "H2O": "H2O1",
    "CO": "C1O1",
    "CO2": "C1O2",
    "OH": "H1O1",
    "NO": "N1O1",
    "CH4": "C1H4",
    "NH3": "H3N1",
    "HCN": "C1H1N1",
    "C2H2": "C2H2",
    "C2H4": "C2H4",
    "H2S": "H2S1",
    "SO": "O1S1",
    "SO2": "O2S1",
    "SiO": "O1Si1",
    "TiH": "H1Ti1",
    "MgH": "H1Mg1",
    "AlH": "Al1H1",
    "SiH": "H1Si1",
    "NaH": "H1Na1",
    "KH": "H1K1",
    "TiO": "O1Ti1",
    "VO": "O1V1",
    "FeH": "Fe1H1",
    "CaH": "Ca1H1",
    "CrH": "Cr1H1",
    "AlO": "Al1O1",
    "PH3": "H3P1",
    # Neutral atoms ("X I" → "X")
    "Fe": "Fe", "Fe I": "Fe",
    "Na": "Na", "Na I": "Na",
    "K": "K", "K I": "K",
    "Ca": "Ca", "Ca I": "Ca",
    "Mg": "Mg", "Mg I": "Mg",
    "Ti": "Ti", "Ti I": "Ti",
    "V": "V", "V I": "V",
    "Cr": "Cr", "Cr I": "Cr",
    "Mn": "Mn", "Mn I": "Mn",
    "Ni": "Ni", "Ni I": "Ni",
    "Si": "Si", "Si I": "Si",
    "Al": "Al", "Al I": "Al",
    "Li": "Li", "Li I": "Li",
    "Co": "Co", "Co I": "Co",
    "Cu": "Cu", "Cu I": "Cu",
    "Zn": "Zn", "Zn I": "Zn",
    "Sr": "Sr", "Sr I": "Sr",
    "Ba": "Ba", "Ba I": "Ba",
    "Sc": "Sc", "Sc I": "Sc",
    "Y": "Y", "Y I": "Y",
    "Zr": "Zr", "Zr I": "Zr",
    "B": "B", "B I": "B",
    "Be": "Be", "Be I": "Be",
    "Ga": "Ga", "Ga I": "Ga",
    "Ge": "Ge", "Ge I": "Ge",
    "Rb": "Rb", "Rb I": "Rb",
    "Cs": "Cs", "Cs I": "Cs",
    "Nb": "Nb", "Nb I": "Nb",
    "Mo": "Mo", "Mo I": "Mo",
    "Ru": "Ru", "Ru I": "Ru",
    "Rh": "Rh", "Rh I": "Rh",
    "Pd": "Pd", "Pd I": "Pd",
    "In": "In", "In I": "In",
    "Sn": "Sn", "Sn I": "Sn",
    "Hf": "Hf", "Hf I": "Hf",
    "W": "W", "W I": "W",
    "Os": "Os", "Os I": "Os",
    "Ir": "Ir", "Ir I": "Ir",
    "Tl": "Tl", "Tl I": "Tl",
    "Pb": "Pb", "Pb I": "Pb",
    # Singly ionized atoms ("X II" → "X1+")
    "Fe II": "Fe1+",
    "Ca II": "Ca1+",
    "Mg II": "Mg1+",
    "Ti II": "Ti1+",
    "Cr II": "Cr1+",
    "Sr II": "Sr1+",
    "Ba II": "Ba1+",
    "Sc II": "Sc1+",
    "Y II": "Y1+",
}


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
        Tarr: jnp.ndarray | None = None,
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
        Tarr: jnp.ndarray | None = None,
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
class FastChem(CompositionSolver):
    """Equilibrium chemistry via FastChem2.

    Pre-computes a 2D VMR grid over (T, P) at grid-build time,
    then interpolates that grid in pure JAX during sampling.
    """

    def __init__(
        self,
        fastchem_parameter_file: str | Path | None = None,
        log_metallicity: float = 0.0,
        co_ratio: float = SOLAR_ABUNDANCES["C"] / SOLAR_ABUNDANCES["O"],
        n_temp: int = chem_config.FASTCHEM_N_TEMP,
        n_pressure: int = chem_config.FASTCHEM_N_PRESSURE,
        t_min: float = chem_config.FASTCHEM_T_MIN,
        t_max: float = chem_config.FASTCHEM_T_MAX,
        cache_dir: str | Path = chem_config.FASTCHEM_CACHE_DIR,
    ):
        self.fastchem_parameter_file = fastchem_parameter_file
        self.log_metallicity = float(log_metallicity)
        self.co_ratio = float(co_ratio)
        self.n_temp = n_temp
        self.n_pressure = n_pressure
        self.t_min = t_min
        self.t_max = t_max
        self.cache_dir = Path(cache_dir)

        # Grid axes (P axis set at build time from art.pressure)
        self._T_grid = np.linspace(t_min, t_max, n_temp)

        # Populated by build_grid()
        self._vmr_grids: dict[str, jnp.ndarray] | None = None
        self._mmw_grid: jnp.ndarray | None = None
        self._log_P_grid: np.ndarray | None = None
        self._species_built: list[str] | None = None

    def _run_fastchem(
        self,
        fc: pyfastchem.FastChem,
        T_flat: np.ndarray,
        P_flat: np.ndarray,
    ) -> pyfastchem.FastChemOutput:
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = T_flat.tolist()
        input_data.pressure = P_flat.tolist()
        fc.calcDensities(input_data, output_data)
        return output_data

    def _cache_key(self, pressure_bar: np.ndarray, species_names: list[str]) -> str:
        h = hashlib.sha256()
        h.update(f"T:{self.n_temp},{self.t_min},{self.t_max}".encode())
        h.update(f"P:{pressure_bar.size},{pressure_bar.min():.6e},{pressure_bar.max():.6e}".encode())
        h.update(f"MH:{self.log_metallicity:.6f}".encode())
        h.update(f"CO:{self.co_ratio:.6f}".encode())
        h.update(f"species:{sorted(species_names)}".encode())
        if self.fastchem_parameter_file is not None:
            h.update(f"par:{self.fastchem_parameter_file}".encode())
        return h.hexdigest()[:16]

    def build_grid(
        self,
        pressure_bar: np.ndarray,
        species_names: list[str],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(pressure_bar, species_names)
        cache_file = self.cache_dir / f"fastchem_grid_{cache_key}.npz"

        # Resample pressure to grid resolution
        if pressure_bar.size != self.n_pressure:
            log_p_orig = np.log10(pressure_bar)
            log_p_grid = np.linspace(log_p_orig.min(), log_p_orig.max(), self.n_pressure)
        else:
            log_p_grid = np.log10(pressure_bar)
        self._log_P_grid = log_p_grid
        P_grid = 10.0 ** log_p_grid

        # Try loading from cache
        if cache_file.exists():
            logger.info("Loading FastChem grid from cache: %s", cache_file)
            data = np.load(cache_file, allow_pickle=True)
            self._vmr_grids = {
                name: jnp.array(data[f"vmr_{name}"])
                for name in species_names
                if f"vmr_{name}" in data
            }
            self._vmr_grids["H2"] = jnp.array(data["vmr_H2"])
            self._vmr_grids["He"] = jnp.array(data["vmr_He"])
            self._mmw_grid = jnp.array(data["mmw"])
            self._species_built = list(species_names)
            return

        logger.info("Building FastChem grid (%d x %d)...", self.n_temp, self.n_pressure)

        # Initialize FastChem
        if self.fastchem_parameter_file is not None:
            fc = pyfastchem.FastChem(str(self.fastchem_parameter_file), 0)
        else:
            raise ValueError(
                "fastchem_parameter_file is required. Provide the path to a "
                "FastChem parameter file (parameters.dat) from the FastChem "
                "repository: https://github.com/NewStrangeWorlds/FastChem"
            )

        # Get element indices for Fe, C, O, H (needed for metallicity/C/O scaling)
        idx_Fe = fc.getElementIndex("Fe")
        idx_C = fc.getElementIndex("C")
        idx_O = fc.getElementIndex("O")
        idx_H = fc.getElementIndex("H")

        # Store solar abundances (log10 relative to H, as loaded by FastChem)
        n_elements = fc.getElementNumber()
        solar_abundances = np.array([fc.getElementAbundance(i) for i in range(n_elements)])

        # Resolve FastChem species indices
        fc_species_idx: dict[str, int] = {}
        for name in species_names:
            fc_symbol = _FASTCHEM_SPECIES_MAP.get(name)
            if fc_symbol is None:
                logger.warning("No FastChem mapping for species '%s', skipping.", name)
                continue
            idx = fc.getGasSpeciesIndex(fc_symbol)
            if idx == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
                logger.warning("FastChem does not know species '%s' (symbol '%s'), skipping.",
                               name, fc_symbol)
                continue
            fc_species_idx[name] = idx

        # H2 and He indices
        idx_H2 = fc.getGasSpeciesIndex("H2")
        idx_He = fc.getGasSpeciesIndex("He")

        n_T = self.n_temp
        n_P = self.n_pressure
        # Allocate grids: (n_T, n_P)
        grid_shape = (n_T, n_P)
        vmr_grids_np: dict[str, np.ndarray] = {
            name: np.zeros(grid_shape) for name in fc_species_idx
        }
        vmr_H2_grid = np.zeros(grid_shape)
        vmr_He_grid = np.zeros(grid_shape)
        mmw_grid_np = np.zeros(grid_shape)

        # Build the flat T×P arrays for FastChem input
        T_flat = np.repeat(self._T_grid, n_P)
        P_flat = np.tile(P_grid * 1e6, n_T)  # bar → dyne/cm² (cgs)

        metallicity_factor = 10.0 ** self.log_metallicity

        # Scale element abundances
        abundances = solar_abundances.copy()

        # Scale all metals by metallicity (everything except H, He)
        for ie in range(n_elements):
            if ie != idx_H and fc.getElementSymbol(ie) != "He":
                abundances[ie] = solar_abundances[ie] * metallicity_factor

        # Adjust C and O to achieve target C/O ratio
        solar_C = solar_abundances[idx_C] * metallicity_factor
        solar_O = solar_abundances[idx_O] * metallicity_factor
        total_CO = solar_C + solar_O
        abundances[idx_C] = total_CO * (self.co_ratio / (1.0 + self.co_ratio))
        abundances[idx_O] = total_CO * (1.0 / (1.0 + self.co_ratio))

        fc.setElementAbundances(abundances.tolist())

        # Run FastChem
        output_data = self._run_fastchem(fc, T_flat, P_flat)

        # Extract number densities → VMR
        n_densities = np.array(output_data.number_densities)
        # n_densities shape: (n_T*n_P, n_species)
        total_n = np.array(output_data.total_element_density)
        # total_n shape: (n_T*n_P,)

        # VMR = n_species / n_total
        for name, sp_idx in fc_species_idx.items():
            vmr = n_densities[:, sp_idx] / np.clip(total_n, 1e-300, None)
            vmr_grids_np[name][:, :] = vmr.reshape(n_T, n_P)

        if idx_H2 != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            vmr = n_densities[:, idx_H2] / np.clip(total_n, 1e-300, None)
            vmr_H2_grid[:, :] = vmr.reshape(n_T, n_P)

        if idx_He != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            vmr = n_densities[:, idx_He] / np.clip(total_n, 1e-300, None)
            vmr_He_grid[:, :] = vmr.reshape(n_T, n_P)

        mmw_arr = np.array(output_data.mean_molecular_weight)
        mmw_grid_np[:, :] = mmw_arr.reshape(n_T, n_P)

        # Convert to JAX arrays and store
        self._vmr_grids = {name: jnp.array(g) for name, g in vmr_grids_np.items()}
        self._vmr_grids["H2"] = jnp.array(vmr_H2_grid)
        self._vmr_grids["He"] = jnp.array(vmr_He_grid)
        self._mmw_grid = jnp.array(mmw_grid_np)
        self._species_built = list(species_names)

        # Save to cache
        save_dict = {
            f"vmr_{name}": np.asarray(g) for name, g in vmr_grids_np.items()
        }
        save_dict["vmr_H2"] = np.asarray(vmr_H2_grid)
        save_dict["vmr_He"] = np.asarray(vmr_He_grid)
        save_dict["mmw"] = np.asarray(mmw_grid_np)
        np.savez_compressed(cache_file, **save_dict)
        logger.info("FastChem grid saved to cache: %s", cache_file)

    def _interp_2d(
        self,
        grid: jnp.ndarray,
        Tarr: jnp.ndarray,
        log_P: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pure JAX 2D interpolation: grid(n_T, n_P) → (n_layers,)."""
        T_grid = jnp.array(self._T_grid)
        log_P_grid = jnp.array(self._log_P_grid)

        # Per-layer bilinear interp at (T_i, P_i) on the (n_T, n_P) grid
        f_T = jnp.interp(Tarr, T_grid, jnp.arange(T_grid.size))
        i_T = jnp.clip(jnp.floor(f_T).astype(int), 0, T_grid.size - 2)
        w_T = f_T - i_T

        f_P = jnp.interp(log_P, log_P_grid, jnp.arange(log_P_grid.size))
        i_P = jnp.clip(jnp.floor(f_P).astype(int), 0, log_P_grid.size - 2)
        w_P = f_P - i_P

        v00 = grid[i_T, i_P]
        v01 = grid[i_T, i_P + 1]
        v10 = grid[i_T + 1, i_P]
        v11 = grid[i_T + 1, i_P + 1]

        result = (
            v00 * (1 - w_T) * (1 - w_P)
            + v01 * (1 - w_T) * w_P
            + v10 * w_T * (1 - w_P)
            + v11 * w_T * w_P
        )
        return result

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        # Lazy grid build on first call
        all_names = list(mol_names) + list(atom_names)
        if self._vmr_grids is None:
            self.build_grid(np.asarray(art.pressure), all_names)

        # If no temperature profile provided, use isothermal midpoint fallback
        if Tarr is None:
            T_mid = (self.t_min + self.t_max) / 2.0
            Tarr = jnp.full(art.pressure.shape, T_mid)

        log_P = jnp.log10(art.pressure)
        n_layers = art.pressure.size
        n_mols = len(mol_names)
        n_atoms = len(atom_names)

        # Interpolate VMR profiles for each species
        vmr_mols_profiles = []
        for mol in mol_names:
            if mol in self._vmr_grids:
                vmr_prof = self._interp_2d(self._vmr_grids[mol], Tarr, log_P)
            else:
                vmr_prof = jnp.full(n_layers, 1e-30)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            if atom in self._vmr_grids:
                vmr_prof = self._interp_2d(self._vmr_grids[atom], Tarr, log_P)
            else:
                vmr_prof = jnp.full(n_layers, 1e-30)
            vmr_atoms_profiles.append(vmr_prof)

        # H2 and He profiles
        vmrH2_prof = self._interp_2d(self._vmr_grids["H2"], Tarr, log_P)
        vmrHe_prof = self._interp_2d(self._vmr_grids["He"], Tarr, log_P)

        # MMW profile
        mmw_prof = self._interp_2d(self._mmw_grid, Tarr, log_P)

        # Compute MMR from VMR: MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Column-averaged scalars for CompositionState
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

class QuenchedChemistry(CompositionSolver):
    pass


class FastChemCond(FastChem):
    """FastChem chemistry with condensation enabled."""

    def __init__(
        self,
        *args,
        rainout_condensation: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.equilibrium_condensation = True
        self.rainout_condensation = rainout_condensation

    def _run_fastchem(
        self,
        fc: pyfastchem.FastChem,
        T_flat: np.ndarray,
        P_flat: np.ndarray,
    ) -> pyfastchem.FastChemOutput:
        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = T_flat.tolist()
        input_data.pressure = P_flat.tolist()
        input_data.equilibrium_condensation = self.equilibrium_condensation
        input_data.rainout_condensation = self.rainout_condensation
        fc.calcDensities(input_data, output_data)
        return output_data


class Photochemistry:
    pass

class DisequilibriumChemistry:
    pass
