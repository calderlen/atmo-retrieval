"""Chemistry/composition solvers for atmospheric retrieval.

This module provides pluggable composition solvers that sample VMRs and compute
derived quantities (MMR profiles, mean molecular weight, etc.) for use in
atmospheric models.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from exojax.database import molinfo


class CompositionState(NamedTuple):
    """Output from a composition solver.

    All scalar quantities are JAX arrays with shape ().
    Profile quantities have shape (n_species, n_layers) or (n_layers,).
    """

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
    """Protocol for composition/chemistry solvers."""

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        """Sample composition and return derived quantities.

        Args:
            mol_names: List of molecule names (e.g., ["H2O", "CO"])
            mol_masses: Molecular masses in AMU, same order as mol_names
            atom_names: List of atomic species names (e.g., ["Fe", "Mg"])
            atom_masses: Atomic masses in AMU, same order as atom_names
            art: ExoJAX art object (provides pressure grid and constant_mmr_profile)

        Returns:
            CompositionState with all derived quantities
        """
        ...


class ConstantVMR:
    """Vertically constant VMR with uniform log-prior.

    This is the default chemistry solver that samples a single log(VMR) value
    per species from a uniform prior, then:
    1. Renormalizes if total trace VMR exceeds 1
    2. Fills remainder with H2/He at solar ratio (6:1)
    3. Computes mean molecular weight
    4. Converts VMR to MMR profiles (constant with altitude)
    """

    def __init__(
        self,
        log_vmr_min: float = -15.0,
        log_vmr_max: float = 0.0,
        h2_he_ratio: float = 6.0,
    ):
        """Initialize the constant VMR solver.

        Args:
            log_vmr_min: Lower bound for log10(VMR) prior
            log_vmr_max: Upper bound for log10(VMR) prior
            h2_he_ratio: H2/He number ratio (solar is ~6)
        """
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
        """Sample composition with vertically constant VMR."""
        # Step 1: Sample VMRs for all species (raw, may sum to > 1)
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

        # Step 2: Renormalize trace VMRs if they sum to > 1
        n_mols = len(vmr_mols_raw)
        n_atoms = len(vmr_atoms_raw)
        if n_mols + n_atoms > 0:
            vmr_trace_arr = jnp.array(vmr_mols_raw + vmr_atoms_raw)
            sum_trace = jnp.sum(vmr_trace_arr)
            # Scale down if sum exceeds 1 (leave tiny room for H2/He)
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
            vmr_trace_arr = vmr_trace_arr * scale
            # Split back into molecules and atoms
            vmr_mols_scalar = [vmr_trace_arr[i] for i in range(n_mols)]
            vmr_atoms_scalar = [vmr_trace_arr[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(vmr_trace_arr)
        else:
            vmr_mols_scalar = []
            vmr_atoms_scalar = []
            vmr_trace_tot = jnp.array(0.0)

        # Step 3: Fill remainder with H2/He (solar ratio)
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe = (1.0 - vmr_trace_tot) * he_frac

        # Step 4: Compute mean molecular weight from (renormalized) VMRs
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw = mass_H2 * vmrH2 + mass_He * vmrHe
        if n_mols > 0:
            mmw = mmw + sum(m * v for m, v in zip(mol_masses, vmr_mols_scalar))
        if n_atoms > 0:
            mmw = mmw + sum(m * v for m, v in zip(atom_masses, vmr_atoms_scalar))

        # Step 5: Convert VMR to MMR and create profiles
        # MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array(
                [
                    art.constant_mmr_profile(vmr * (mass / mmw))
                    for vmr, mass in zip(vmr_mols_scalar, mol_masses)
                ]
            )
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atoms = jnp.array(
                [
                    art.constant_mmr_profile(vmr * (mass / mmw))
                    for vmr, mass in zip(vmr_atoms_scalar, atom_masses)
                ]
            )
        else:
            mmr_atoms = jnp.zeros((0, art.pressure.size))

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


class FreeVMR:
    """Vertically-varying VMR: sample at N nodes, interpolate in log-P.

    Similar to numpyro_free_temperature in pt.py, this samples log(VMR) at
    a set of pressure nodes and interpolates to the full pressure grid.
    """

    def __init__(
        self,
        n_nodes: int = 5,
        log_vmr_min: float = -15.0,
        log_vmr_max: float = 0.0,
        h2_he_ratio: float = 6.0,
    ):
        """Initialize the free VMR solver.

        Args:
            n_nodes: Number of pressure nodes for VMR interpolation
            log_vmr_min: Lower bound for log10(VMR) prior
            log_vmr_max: Upper bound for log10(VMR) prior
            h2_he_ratio: H2/He number ratio (solar is ~6)
        """
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
        """Sample log(VMR) at nodes and interpolate to full pressure grid.

        Args:
            name: Species name (used for parameter naming)
            log_p: Log10 pressure grid (full)
            log_p_nodes: Log10 pressure at nodes

        Returns:
            VMR profile (linear, not log) on full pressure grid
        """
        logVMR_nodes = []
        for i in range(self.n_nodes):
            logVMR_i = numpyro.sample(
                f"logVMR_{name}_node{i}",
                dist.Uniform(self.log_vmr_min, self.log_vmr_max),
            )
            logVMR_nodes.append(logVMR_i)
        logVMR_nodes = jnp.array(logVMR_nodes)

        # Interpolate in log-P space
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
        """Sample composition with vertically-varying VMR profiles."""
        log_p = jnp.log10(art.pressure)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), self.n_nodes)
        n_layers = art.pressure.size

        # Step 1: Sample VMR profiles for all species
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

        # Step 2: Renormalize at each layer if total trace VMR exceeds 1
        if n_mols + n_atoms > 0:
            # Stack all profiles: shape (n_species, n_layers)
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)  # (n_layers,)
            # Scale down where sum exceeds 1
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            # Split back
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)  # (n_layers,)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Step 3: Fill remainder with H2/He at each layer
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Step 4: Compute mean molecular weight profile
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        if n_mols > 0:
            for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
                mmw_prof = mmw_prof + mass * vmr_prof
        if n_atoms > 0:
            for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
                mmw_prof = mmw_prof + mass * vmr_prof

        # Step 5: Convert VMR to MMR profiles
        # MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array(
                [
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
                ]
            )
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array(
                [
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
                ]
            )
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

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
