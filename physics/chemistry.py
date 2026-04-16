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
from config import numerics_config as numerics_config

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOLAR_ABUNDANCE_FILE = _REPO_ROOT / chem_config.SOLAR_ABUNDANCE_FILE


@lru_cache(maxsize=1)
def _load_solar_element_abundances() -> dict[str, float]:
    abundances: dict[str, float] = {}

    with _SOLAR_ABUNDANCE_FILE.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            symbol = parts[0]
            if symbol == "e-":
                continue
            try:
                log_eps = float(parts[1])
            except ValueError:
                continue
            abundances[symbol] = float(10.0 ** (log_eps - 12.0))

    if "H" not in abundances:
        raise ValueError(f"Hydrogen abundance missing from '{_SOLAR_ABUNDANCE_FILE}'.")
    abundances["H"] = 1.0
    return abundances

# standardized output of each chemistry solver
class CompositionState(NamedTuple):

    vmr_mols: list[jnp.ndarray]  # Scalar VMR per molecule
    vmr_atoms: list[jnp.ndarray]  # Scalar VMR per atom
    vmrH2: jnp.ndarray  # H2 VMR (scalar)
    vmrHe: jnp.ndarray  # He VMR (scalar)
    mmw: jnp.ndarray  # Mean molecular weight (scalar)
    mmr_mols: jnp.ndarray  # MMR profiles (n_mols, n_layers)
    mmr_atoms: jnp.ndarray  # MMR profiles (n_atoms, n_layers)
    vmrH2_profile: jnp.ndarray  # H2 VMR profile (n_layers,)
    vmrHe_profile: jnp.ndarray  # He VMR profile (n_layers,)
    mmw_profile: jnp.ndarray  # MMW profile (n_layers,)
    continuum_vmr_profiles: dict[str, jnp.ndarray]  # Hidden continuum VMR profiles

# anything with a CompositionState is a chemistry solver
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

######## CHEMISTRY MODELS #######
class ConstantVMR:

    # bounds for log-uniform VMR sampling of trace spcecies
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
        
        # sample over log-uniform VMRs for trace molecules and atoms, then renormalize to ensure total VMR < 1
        vmr_mols_raw = []
        for mol in mol_names:
            logVMR = numpyro.sample(f"logVMR_{mol}", dist.Uniform(self.log_vmr_min, self.log_vmr_max))
            vmr_mols_raw.append(jnp.power(10.0, logVMR))

        vmr_atoms_raw = []
        for atom in atom_names:
            logVMR = numpyro.sample(f"logVMR_{atom}", dist.Uniform(self.log_vmr_min, self.log_vmr_max))
            vmr_atoms_raw.append(jnp.power(10.0, logVMR))

        n_mols = len(vmr_mols_raw)
        n_atoms = len(vmr_atoms_raw)

        # Normalize trace species VMRs
        vmr_trace_array = jnp.array(vmr_mols_raw + vmr_atoms_raw)
        sum_trace = jnp.sum(vmr_trace_array)
        scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
        vmr_trace_array = vmr_trace_array * scale
        vmr_mols_scalar = [vmr_trace_array[i] for i in range(n_mols)]
        vmr_atoms_scalar = [vmr_trace_array[n_mols + i] for i in range(n_atoms)]
        vmr_trace_tot = jnp.sum(vmr_trace_array)

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
        if n_mols > 0:
            mmr_mol_profiles = []
            for vmr, mass in zip(vmr_mols_scalar, mol_masses):
                mmr_mol_profiles.append(art.constant_mmr_profile(vmr * (mass / mmw)))
            mmr_mols = jnp.array(mmr_mol_profiles)
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atom_profiles = []
            for vmr, mass in zip(vmr_atoms_scalar, atom_masses):
                mmr_atom_profiles.append(art.constant_mmr_profile(vmr * (mass / mmw)))
            mmr_atoms = jnp.array(mmr_atom_profiles)
        else:
            mmr_atoms = jnp.zeros((0, art.pressure.size))

        # Step 6: Create constant profiles for CIA inputs and mmw
        vmrH2_profile = art.constant_mmr_profile(vmrH2)
        vmrHe_profile = art.constant_mmr_profile(vmrHe)
        mmw_profile = art.constant_mmr_profile(mmw)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            continuum_vmr_profiles={},
        )

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _solar_abundance_for_species(species: str) -> float:
    base = species.split()[0].replace("+", "").replace("-", "")
    counts: list[tuple[str, int]] = []
    for element, count_str in _ELEMENT_RE.findall(base):
        count = int(count_str) if count_str else 1
        counts.append((element, count))
    if not counts:
        return numerics_config.TRACE_SPECIES_FLOOR

    limiting = np.inf
    for element, count in counts:
        elem_abundance = SOLAR_ELEMENT_ABUNDANCES.get(element)
        if elem_abundance is None:
            return numerics_config.TRACE_SPECIES_FLOOR
        limiting = min(limiting, elem_abundance / float(count))
    return float(max(limiting, numerics_config.TRACE_SPECIES_FLOOR))


SOLAR_ELEMENT_ABUNDANCES = _load_solar_element_abundances()


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
    "H-": "H1-",
    "e-": "e-",
    "H": "H",
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
    "H I": "H",
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

_CONTINUUM_SPECIES_ALIASES: dict[str, str] = {
    "H": "H",
    "H I": "H",
    "e-": "e-",
    "H-": "H-",
}
_CONTINUUM_SPECIES_MASSES: dict[str, float] = {
    "H": float(molinfo.molmass_isotope("H", db_HIT=False)),
    "H-": float(molinfo.molmass_isotope("H", db_HIT=False) + 5.48579909065e-4),
    "e-": 5.48579909065e-4,
}


def _canonical_continuum_species_name(species: str) -> str:
    return _CONTINUUM_SPECIES_ALIASES.get(species, species)


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
        scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
        all_profiles = all_profiles * scale[None, :] if all_profiles.size > 0 else all_profiles
        scaled_vmr_mols_profiles = []
        for i in range(n_mols):
            scaled_vmr_mols_profiles.append(all_profiles[i])
        vmr_mols_profiles = scaled_vmr_mols_profiles
        scaled_vmr_atoms_profiles = []
        for i in range(n_atoms):
            scaled_vmr_atoms_profiles.append(all_profiles[n_mols + i])
        vmr_atoms_profiles = scaled_vmr_atoms_profiles
        vmr_trace_tot = jnp.sum(all_profiles, axis=0) if all_profiles.size > 0 else jnp.zeros(n_layers)

        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_profile = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_profile = (1.0 - vmr_trace_tot) * he_frac

        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_profile = mass_H2 * vmrH2_profile + mass_He * vmrHe_profile
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_profile = mmw_profile + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_profile = mmw_profile + mass * vmr_prof

        # MMR_i = VMR_i * (M_i / mmw) - ensure correct 2D shape for empty cases
        if n_mols > 0:
            mmr_mol_profiles = []
            for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
                mmr_mol_profiles.append(vmr_prof * (mass / mmw_profile))
            mmr_mols = jnp.array(mmr_mol_profiles)
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atom_profiles = []
            for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
                mmr_atom_profiles.append(vmr_prof * (mass / mmw_profile))
            mmr_atoms = jnp.array(mmr_atom_profiles)
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # For scalar outputs, use column-averaged values (pressure-weighted would be better)
        # Here we just use simple mean for consistency with downstream code that expects scalars
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_profile)
        vmrHe = jnp.mean(vmrHe_profile)
        mmw = jnp.mean(mmw_profile)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            continuum_vmr_profiles={},
        )

# TODO: extend the current equilibrium / hybrid baseline with quenching,
# disequilibrium chemistry, and photochemistry when the free-VMR model is not sufficient
# TODO: partial ionization (Saha)
# TODO: H- continuum affecting line-to-continuum ratio
# TODO: vertical transport parameterized by K_zz and then like tchem and t_dyn or more simply the pestimate of quench level
# TODO: use CO as a proxy/tracer for the overall metallicity M/H of the atmosphere
# TODO: The continuum problem. Public low resolution spectra? R ~ 10^2 - 10^3??
#class QuenchedChemistry(CompositionSolver):
#    pass
#class Photochemistry:
#    pass

#class DisequilibriumChemistry:
#    pass

#class ParametricMetallicityCOChemistry(CompositionSolver):
#    """Simple equilibrium-like chemistry with [M/H] and C/O as free parameters."""

    def __init__(
        self,
        metallicity_range: tuple[float, float] = chem_config.METALLICITY_RANGE,
        co_ratio_range: tuple[float, float] = chem_config.CO_RATIO_RANGE,
        h2_he_ratio: float = chem_config.H2_HE_RATIO,
    ):
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self._c_solar = SOLAR_ELEMENT_ABUNDANCES["C"]
        self._o_solar = SOLAR_ELEMENT_ABUNDANCES["O"]

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
        solar = _solar_abundance_for_species(species)
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
            solar_arr = jnp.array([_solar_abundance_for_species(s) for s in species])
            counts = [_count_co(s) for s in species]
            n_c = jnp.array([c[0] for c in counts], dtype=solar_arr.dtype)
            n_o = jnp.array([c[1] for c in counts], dtype=solar_arr.dtype)

            co_factor = jnp.where((n_c > 0) & (n_o > 0), jnp.minimum(c_scale, o_scale), 1.0)
            co_factor = jnp.where((n_c > 0) & (n_o == 0), c_scale, co_factor)
            co_factor = jnp.where((n_o > 0) & (n_c == 0), o_scale, co_factor)

            vmr_trace_array = solar_arr * metallicity * co_factor
            sum_trace = jnp.sum(vmr_trace_array)
            scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
            vmr_trace_array = vmr_trace_array * scale

            vmr_mols_arr = vmr_trace_array[:n_mols]
            vmr_atoms_arr = vmr_trace_array[n_mols:]
            vmr_mols_scalar = [vmr_mols_arr[i] for i in range(n_mols)]
            vmr_atoms_scalar = [vmr_atoms_arr[i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(vmr_trace_array)
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
        vmrH2_profile = vmrH2 * ones_prof
        vmrHe_profile = vmrHe * ones_prof
        mmw_profile = mmw * ones_prof

        if n_mols > 0:
            mmr_mols = ((vmr_mols_arr * (mol_masses_arr / mmw))[:, None] * ones_prof[None, :])
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atoms = ((vmr_atoms_arr * (atom_masses_arr / mmw))[:, None] * ones_prof[None, :])
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
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            continuum_vmr_profiles={},
        )
class FastChemEquilibriumChemistry(CompositionSolver):
    """Equilibrium chemistry via FastChem2.

    Pre-computes a 2D VMR grid over (T, P) at grid-build time,
    then interpolates that grid in pure JAX during sampling.
    """

    def __init__(
        self,
        fastchem_parameter_file: str | Path | None = None,
        log_metallicity: float = 0.0,
        co_ratio: float = SOLAR_ELEMENT_ABUNDANCES["C"] / SOLAR_ELEMENT_ABUNDANCES["O"],
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
            self._vmr_grids = {}
            for name in species_names:
                key = f"vmr_{name}"
                if key in data:
                    self._vmr_grids[name] = jnp.array(data[key])
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
        vmr_grids_np: dict[str, np.ndarray] = {name: np.zeros(grid_shape) for name in fc_species_idx}
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
            vmr = n_densities[:, sp_idx] / np.clip(total_n, numerics_config.F64_FLOOR, None)
            vmr_grids_np[name][:, :] = vmr.reshape(n_T, n_P)

        if idx_H2 != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            vmr = n_densities[:, idx_H2] / np.clip(total_n, numerics_config.F64_FLOOR, None)
            vmr_H2_grid[:, :] = vmr.reshape(n_T, n_P)

        if idx_He != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            vmr = n_densities[:, idx_He] / np.clip(total_n, numerics_config.F64_FLOOR, None)
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
                vmr_prof = jnp.full(n_layers, numerics_config.TRACE_SPECIES_FLOOR)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            if atom in self._vmr_grids:
                vmr_prof = self._interp_2d(self._vmr_grids[atom], Tarr, log_P)
            else:
                vmr_prof = jnp.full(n_layers, numerics_config.TRACE_SPECIES_FLOOR)
            vmr_atoms_profiles.append(vmr_prof)

        # H2 and He profiles
        vmrH2_profile = self._interp_2d(self._vmr_grids["H2"], Tarr, log_P)
        vmrHe_profile = self._interp_2d(self._vmr_grids["He"], Tarr, log_P)

        # MMW profile
        mmw_profile = self._interp_2d(self._mmw_grid, Tarr, log_P)

        # Compute MMR from VMR: MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_profile)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_profile)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Column-averaged scalars for CompositionState
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_profile)
        vmrHe = jnp.mean(vmrHe_profile)
        mmw = jnp.mean(mmw_profile)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            continuum_vmr_profiles={},
        )


class FastChemEquilibriumCondensationChemistry(FastChemEquilibriumChemistry):
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


class FastChemHybridChemistry(FastChemEquilibriumChemistry):
    """Hybrid chemistry with free trace VMRs and FastChem continuum species.

    This solver is designed for HMC-NUTS: FastChem is evaluated offline onto a
    cached 4D grid over ([M/H], C/O, T, P), and runtime sampling uses pure JAX
    interpolation to inject continuum-relevant species (e.g. H-, e-, H).

    The bulk H2/He reservoir is still reconstructed from the remaining
    abundance using the configured H2/He ratio. FastChem's bulk-gas state is
    not yet used directly in this hybrid branch.
    """

    def __init__(
        self,
        *args,
        continuum_species: tuple[str, ...] = chem_config.FASTCHEM_HYBRID_CONTINUUM_SPECIES,
        metallicity_range: tuple[float, float] = chem_config.FASTCHEM_HYBRID_METALLICITY_RANGE,
        co_ratio_range: tuple[float, float] = chem_config.FASTCHEM_HYBRID_CO_RATIO_RANGE,
        n_metallicity: int = chem_config.FASTCHEM_HYBRID_N_METALLICITY,
        n_co_ratio: int = chem_config.FASTCHEM_HYBRID_N_CO_RATIO,
        log_vmr_min: float = chem_config.LOG_VMR_MIN,
        log_vmr_max: float = chem_config.LOG_VMR_MAX,
        h2_he_ratio: float = chem_config.H2_HE_RATIO,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if n_metallicity < 2:
            raise ValueError("n_metallicity must be >= 2.")
        if n_co_ratio < 2:
            raise ValueError("n_co_ratio must be >= 2.")

        self.continuum_species = tuple(continuum_species)
        self._canonical_continuum_species = tuple(dict.fromkeys(_canonical_continuum_species_name(species) for species in self.continuum_species))
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.n_metallicity = int(n_metallicity)
        self.n_co_ratio = int(n_co_ratio)
        self.log_vmr_min = float(log_vmr_min)
        self.log_vmr_max = float(log_vmr_max)
        self.h2_he_ratio = float(h2_he_ratio)
        self._log_metallicity_grid = np.linspace(
            self.metallicity_range[0],
            self.metallicity_range[1],
            self.n_metallicity,
        )
        self._co_ratio_grid = np.linspace(
            self.co_ratio_range[0],
            self.co_ratio_range[1],
            self.n_co_ratio,
        )
        self._hybrid_vmr_grids: dict[str, jnp.ndarray] | None = None
        self._hybrid_species_built: list[str] | None = None

        self._free_solver = ConstantVMR(
            log_vmr_min=self.log_vmr_min,
            log_vmr_max=self.log_vmr_max,
            h2_he_ratio=self.h2_he_ratio,
        )
        if not self.requires_hybrid_parameters():
            logger.warning(
                "FastChemHybridChemistry requires hidden 'H'/'H I' and 'e-' "
                "continuum drivers for the H- continuum. It will fall back to "
                "ConstantVMR behavior."
            )

    def _canonical_species_name(self, species: str) -> str:
        return _canonical_continuum_species_name(species)

    def is_hybrid_managed_species(self, species: str) -> bool:
        return self._canonical_species_name(species) in self._canonical_continuum_species

    def hidden_continuum_species(self) -> tuple[str, ...]:
        return tuple(
            species for species in self._canonical_continuum_species
            if species in _FASTCHEM_SPECIES_MAP
        )

    def opacity_driver_species(self) -> tuple[str, ...]:
        drivers = []
        if "H" in self._canonical_continuum_species:
            drivers.append("H")
        if "e-" in self._canonical_continuum_species:
            drivers.append("e-")
        return tuple(drivers)

    def requires_hybrid_parameters(self) -> bool:
        return {"H", "e-"}.issubset(self.opacity_driver_species())

    def _hybrid_cache_key(self, pressure_bar: np.ndarray, species_names: list[str]) -> str:
        h = hashlib.sha256()
        h.update("v2".encode())
        h.update(f"T:{self.n_temp},{self.t_min},{self.t_max}".encode())
        h.update(f"P:{pressure_bar.size},{pressure_bar.min():.6e},{pressure_bar.max():.6e}".encode())
        h.update(
            (
                f"MH:{self.n_metallicity},"
                f"{self.metallicity_range[0]:.6f},"
                f"{self.metallicity_range[1]:.6f}"
            ).encode()
        )
        h.update(
            (
                f"CO:{self.n_co_ratio},"
                f"{self.co_ratio_range[0]:.6f},"
                f"{self.co_ratio_range[1]:.6f}"
            ).encode()
        )
        h.update(f"species:{sorted(species_names)}".encode())
        if self.fastchem_parameter_file is not None:
            h.update(f"par:{self.fastchem_parameter_file}".encode())
        return h.hexdigest()[:16]

    def _resolve_fastchem_species_indices(
        self,
        fc: pyfastchem.FastChem,
        species_names: list[str],
    ) -> dict[str, int]:
        fc_species_idx: dict[str, int] = {}
        for name in species_names:
            fc_symbol = _FASTCHEM_SPECIES_MAP.get(name)
            if fc_symbol is None:
                logger.warning("No FastChem mapping for species '%s', skipping.", name)
                continue
            idx = fc.getGasSpeciesIndex(fc_symbol)
            if idx == pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
                logger.warning(
                    "FastChem does not know species '%s' (symbol '%s'), skipping.",
                    name,
                    fc_symbol,
                )
                continue
            fc_species_idx[name] = idx
        return fc_species_idx

    def _set_fastchem_abundances(
        self,
        fc: pyfastchem.FastChem,
        solar_abundances: np.ndarray,
        idx_c: int,
        idx_o: int,
        idx_h: int,
        log_metallicity: float,
        co_ratio: float,
    ) -> None:
        n_elements = fc.getElementNumber()
        metallicity_factor = 10.0 ** float(log_metallicity)
        co_ratio_safe = max(float(co_ratio), 1.0e-8)

        abundances = solar_abundances.copy()
        for ie in range(n_elements):
            if ie != idx_h and fc.getElementSymbol(ie) != "He":
                abundances[ie] = solar_abundances[ie] * metallicity_factor

        solar_c = solar_abundances[idx_c] * metallicity_factor
        solar_o = solar_abundances[idx_o] * metallicity_factor
        total_co = solar_c + solar_o
        abundances[idx_c] = total_co * (co_ratio_safe / (1.0 + co_ratio_safe))
        abundances[idx_o] = total_co * (1.0 / (1.0 + co_ratio_safe))
        fc.setElementAbundances(abundances.tolist())

    def _build_hybrid_grid(
        self,
        pressure_bar: np.ndarray,
        species_names: list[str],
    ) -> None:
        if self.fastchem_parameter_file is None:
            raise ValueError("fastchem_parameter_file is required for FastChemHybridChemistry.")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        species_names = sorted(set(species_names))
        cache_key = self._hybrid_cache_key(pressure_bar, species_names)
        cache_file = self.cache_dir / f"fastchem_hybrid_grid_{cache_key}.npz"

        if pressure_bar.size != self.n_pressure:
            log_p_orig = np.log10(pressure_bar)
            log_p_grid = np.linspace(log_p_orig.min(), log_p_orig.max(), self.n_pressure)
        else:
            log_p_grid = np.log10(pressure_bar)
        self._log_P_grid = log_p_grid
        p_grid = 10.0 ** log_p_grid

        if cache_file.exists():
            logger.info("Loading FastChem hybrid grid from cache: %s", cache_file)
            data = np.load(cache_file, allow_pickle=True)
            self._hybrid_vmr_grids = {}
            for name in species_names:
                key = f"vmr_{name}"
                if key in data:
                    self._hybrid_vmr_grids[name] = jnp.array(data[key])
            self._hybrid_species_built = list(species_names)
            return

        logger.info(
            "Building FastChem hybrid grid (%d x %d x %d x %d)...",
            self.n_metallicity,
            self.n_co_ratio,
            self.n_temp,
            self.n_pressure,
        )

        fc = pyfastchem.FastChem(str(self.fastchem_parameter_file), 0)
        n_elements = fc.getElementNumber()
        idx_c = fc.getElementIndex("C")
        idx_o = fc.getElementIndex("O")
        idx_h = fc.getElementIndex("H")
        solar_abundances = np.array([fc.getElementAbundance(i) for i in range(n_elements)])

        fc_species_idx = self._resolve_fastchem_species_indices(fc, species_names)
        if not fc_species_idx:
            self._hybrid_vmr_grids = {}
            self._hybrid_species_built = list(species_names)
            return

        grid_shape = (self.n_metallicity, self.n_co_ratio, self.n_temp, self.n_pressure)
        vmr_grids_np: dict[str, np.ndarray] = {
            name: np.zeros(grid_shape, dtype=np.float64)
            for name in fc_species_idx
        }

        t_flat = np.repeat(self._T_grid, self.n_pressure)
        p_flat = np.tile(p_grid * 1e6, self.n_temp)

        for i_mh, log_metallicity in enumerate(self._log_metallicity_grid):
            for i_co, co_ratio in enumerate(self._co_ratio_grid):
                self._set_fastchem_abundances(
                    fc,
                    solar_abundances,
                    idx_c,
                    idx_o,
                    idx_h,
                    float(log_metallicity),
                    float(co_ratio),
                )

                output_data = self._run_fastchem(fc, t_flat, p_flat)
                n_densities = np.array(output_data.number_densities)
                total_n = np.clip(
                    np.array(output_data.total_element_density),
                    numerics_config.F64_FLOOR,
                    None,
                )

                for name, sp_idx in fc_species_idx.items():
                    vmr = n_densities[:, sp_idx] / total_n
                    vmr_grids_np[name][i_mh, i_co, :, :] = vmr.reshape(self.n_temp, self.n_pressure)

        self._hybrid_vmr_grids = {name: jnp.array(g) for name, g in vmr_grids_np.items()}
        self._hybrid_species_built = list(species_names)

        save_dict = {
            "log_metallicity_grid": np.asarray(self._log_metallicity_grid),
            "co_ratio_grid": np.asarray(self._co_ratio_grid),
            "T_grid": np.asarray(self._T_grid),
            "log_P_grid": np.asarray(self._log_P_grid),
        }
        for name, grid in vmr_grids_np.items():
            save_dict[f"vmr_{name}"] = np.asarray(grid)
        np.savez_compressed(cache_file, **save_dict)
        logger.info("FastChem hybrid grid saved to cache: %s", cache_file)

    def _interp_4d(
        self,
        grid: jnp.ndarray,
        log_metallicity: jnp.ndarray,
        co_ratio: jnp.ndarray,
        Tarr: jnp.ndarray,
        log_P: jnp.ndarray,
    ) -> jnp.ndarray:
        mh_grid = jnp.array(self._log_metallicity_grid)
        co_grid = jnp.array(self._co_ratio_grid)
        t_grid = jnp.array(self._T_grid)
        log_p_grid = jnp.array(self._log_P_grid)

        mh_val = jnp.clip(log_metallicity, mh_grid[0], mh_grid[-1])
        co_val = jnp.clip(co_ratio, co_grid[0], co_grid[-1])
        t_val = jnp.clip(Tarr, t_grid[0], t_grid[-1])
        log_p_val = jnp.clip(log_P, log_p_grid[0], log_p_grid[-1])

        f_mh = jnp.interp(mh_val, mh_grid, jnp.arange(mh_grid.size))
        i_mh = jnp.clip(jnp.floor(f_mh).astype(int), 0, mh_grid.size - 2)
        w_mh = f_mh - i_mh

        f_co = jnp.interp(co_val, co_grid, jnp.arange(co_grid.size))
        i_co = jnp.clip(jnp.floor(f_co).astype(int), 0, co_grid.size - 2)
        w_co = f_co - i_co

        f_t = jnp.interp(t_val, t_grid, jnp.arange(t_grid.size))
        i_t = jnp.clip(jnp.floor(f_t).astype(int), 0, t_grid.size - 2)
        w_t = f_t - i_t

        f_p = jnp.interp(log_p_val, log_p_grid, jnp.arange(log_p_grid.size))
        i_p = jnp.clip(jnp.floor(f_p).astype(int), 0, log_p_grid.size - 2)
        w_p = f_p - i_p

        result = jnp.zeros_like(t_val)
        for d_mh in (0, 1):
            w0 = w_mh if d_mh else (1.0 - w_mh)
            idx_mh = i_mh + d_mh
            for d_co in (0, 1):
                w1 = w_co if d_co else (1.0 - w_co)
                idx_co = i_co + d_co
                for d_t in (0, 1):
                    w2 = w_t if d_t else (1.0 - w_t)
                    idx_t = i_t + d_t
                    for d_p in (0, 1):
                        w3 = w_p if d_p else (1.0 - w_p)
                        idx_p = i_p + d_p
                        result = result + (w0 * w1 * w2 * w3) * grid[idx_mh, idx_co, idx_t, idx_p]
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
        free_mol = [
            (n, m) for n, m in zip(mol_names, mol_masses)
            if not self.is_hybrid_managed_species(n)
        ]
        free_atom = [
            (n, m) for n, m in zip(atom_names, atom_masses)
            if not self.is_hybrid_managed_species(n)
        ]

        free_mol_names = [n for n, _ in free_mol]
        free_mol_masses = [m for _, m in free_mol]
        free_atom_names = [n for n, _ in free_atom]
        free_atom_masses = [m for _, m in free_atom]

        base = self._free_solver.sample(
            free_mol_names,
            free_mol_masses,
            free_atom_names,
            free_atom_masses,
            art,
            Tarr=Tarr,
        )

        if not self.requires_hybrid_parameters():
            return base

        if Tarr is None:
            T_mid = (self.t_min + self.t_max) / 2.0
            Tarr = jnp.full(art.pressure.shape, T_mid)

        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        log_P = jnp.log10(art.pressure)
        n_layers = art.pressure.size

        mol_profile_map = {
            name: jnp.full(n_layers, numerics_config.TRACE_SPECIES_FLOOR)
            for name in mol_names
        }
        atom_profile_map = {
            name: jnp.full(n_layers, numerics_config.TRACE_SPECIES_FLOOR)
            for name in atom_names
        }
        for i, name in enumerate(free_mol_names):
            mol_profile_map[name] = jnp.full(n_layers, base.vmr_mols[i])
        for i, name in enumerate(free_atom_names):
            atom_profile_map[name] = jnp.full(n_layers, base.vmr_atoms[i])

        continuum_profile_map = {
            species: jnp.full(n_layers, numerics_config.TRACE_SPECIES_FLOOR)
            for species in self.hidden_continuum_species()
        }
        needed = list(continuum_profile_map)
        if needed:
            if self._hybrid_vmr_grids is None or any(s not in self._hybrid_vmr_grids for s in needed):
                self._build_hybrid_grid(np.asarray(art.pressure), needed)

            overrides = {}
            for species in needed:
                if species in self._hybrid_vmr_grids:
                    overrides[species] = self._interp_4d(
                        self._hybrid_vmr_grids[species],
                        log_metallicity,
                        co_ratio,
                        Tarr,
                        log_P,
                    )

            for species, vmr_prof in overrides.items():
                continuum_profile_map[species] = vmr_prof
                for name in mol_profile_map:
                    if self._canonical_species_name(name) == species:
                        mol_profile_map[name] = vmr_prof
                for name in atom_profile_map:
                    if self._canonical_species_name(name) == species:
                        atom_profile_map[name] = vmr_prof

        vmr_mols_profiles = [mol_profile_map[name] for name in mol_names]
        vmr_atoms_profiles = [atom_profile_map[name] for name in atom_names]
        continuum_profiles = [continuum_profile_map[name] for name in continuum_profile_map]

        if (len(vmr_mols_profiles) + len(vmr_atoms_profiles) + len(continuum_profiles)) > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles + continuum_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 1.0, 1.0 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(len(vmr_mols_profiles))]
            vmr_atoms_profiles = [all_profiles[len(vmr_mols_profiles) + i] for i in range(len(vmr_atoms_profiles))]
            continuum_offset = len(vmr_mols_profiles) + len(vmr_atoms_profiles)
            continuum_profile_map = {
                name: all_profiles[continuum_offset + i]
                for i, name in enumerate(continuum_profile_map)
            }
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_profile = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_profile = (1.0 - vmr_trace_tot) * he_frac

        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_profile = mass_H2 * vmrH2_profile + mass_He * vmrHe_profile
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_profile = mmw_profile + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_profile = mmw_profile + mass * vmr_prof
        for species, vmr_prof in continuum_profile_map.items():
            if species in _CONTINUUM_SPECIES_MASSES:
                mmw_profile = mmw_profile + _CONTINUUM_SPECIES_MASSES[species] * vmr_prof

        if len(vmr_mols_profiles) > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_profile)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if len(vmr_atoms_profiles) > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_profile)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_profile)
        vmrHe = jnp.mean(vmrHe_profile)
        mmw = jnp.mean(mmw_profile)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_profile=vmrH2_profile,
            vmrHe_profile=vmrHe_profile,
            mmw_profile=mmw_profile,
            continuum_vmr_profiles=continuum_profile_map,
        )
