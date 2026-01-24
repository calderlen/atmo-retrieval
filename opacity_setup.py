"""Opacity setup for molecular and atomic species."""

from typing import Callable
import numpy as np
import jax.numpy as jnp
from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa


def setup_cia_opacities(cia_paths: dict[str, str], nu_grid: np.ndarray) -> dict[str, OpaCIA]:
    """Setup CIA opacities for H2-H2 and H2-He."""
    opa_cias = {}
    for name, path in cia_paths.items():
        cdb = CdbCIA(path, nurange=nu_grid)
        opa_cias[name] = OpaCIA(cdb, nu_grid=nu_grid)
    return opa_cias


def build_premodit_from_snapshot(
    snapshot: object,
    molmass: float,
    mol: str,
    nu_grid: np.ndarray,
    ndiv: int,
    diffmode: int,
    Tlow: float,
    Thigh: float,
) -> OpaPremodit:
    """Create preMODIT opacity from database snapshot and save it."""
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid,
        nstitch=ndiv,
        diffmode=diffmode,
        auto_trange=[Tlow, Thigh],
        dit_grid_resolution=1,
        allow_32bit=True,
        cutwing=1 / (2 * ndiv),
    )
    saveopa(opa, f"opa_{mol}.zarr", format="zarr", aux={"molmass": molmass})
    return opa


def load_or_build_opacity(
    mol: str,
    path: str,
    mdb_factory: Callable[[str], object],
    opa_load: bool,
    nu_grid: np.ndarray,
    ndiv: int,
    diffmode: int,
    Tlow: float,
    Thigh: float,
) -> tuple[OpaPremodit, float]:
    """Load saved opacity or build from database snapshot."""
    if opa_load:
        try:
            opa = OpaPremodit.from_saved_opa(f"opa_{mol}.zarr", strict=False)
            return opa, opa.aux["molmass"]
        except Exception:
            print(f"  Warning: Could not load saved opacity for {mol}, building from database...")

    mdb = mdb_factory(path)
    molmass = mdb.molmass
    opa = build_premodit_from_snapshot(
        mdb.to_snapshot(), molmass, mol, nu_grid, ndiv, diffmode, Tlow, Thigh
    )
    del mdb
    return opa, molmass


def load_molecular_opacities(
    molpath_hitemp: dict[str, str],
    molpath_exomol: dict[str, str],
    nu_grid: np.ndarray,
    opa_load: bool,
    ndiv: int,
    diffmode: int,
    Tlow: float,
    Thigh: float,
) -> tuple[dict[str, OpaPremodit], jnp.ndarray]:
    """Load or create all molecular opacities for HITEMP and ExoMol."""
    opa_mols = {}
    molmass_list = []

    print("Loading HITEMP/ExoMol databases...")

    # HITEMP molecules
    for mol, path in molpath_hitemp.items():
        print(f"  * {mol} (HITEMP)")
        mdb_factory = lambda p: MdbHitemp(p, nu_grid, gpu_transfer=False, isotope=1)
        opa, molmass = load_or_build_opacity(
            mol, path, mdb_factory, opa_load, nu_grid, ndiv, diffmode, Tlow, Thigh
        )
        opa_mols[mol] = opa
        molmass_list.append(molmass)

    # ExoMol molecules
    for mol, path in molpath_exomol.items():
        print(f"  * {mol} (ExoMol)")
        mdb_factory = lambda p: MdbExomol(p, nu_grid, gpu_transfer=False)
        opa, molmass = load_or_build_opacity(
            mol, path, mdb_factory, opa_load, nu_grid, ndiv, diffmode, Tlow, Thigh
        )
        opa_mols[mol] = opa
        molmass_list.append(molmass)

    return opa_mols, jnp.array(molmass_list)


def load_atomic_opacities(
    atomic_species: dict[str, dict],
    nu_grid: np.ndarray,
    opa_load: bool,
    ndiv: int,
    diffmode: int,
    Tlow: float,
    Thigh: float,
) -> tuple[dict[str, object], jnp.ndarray]:
    """Load atomic line opacities (e.g., Na, K, Ca)."""
    opa_atoms = {}
    atommass_list = []

    print("Loading atomic line databases...")

    # Note: Atomic line handling depends on the database format
    # This is a placeholder - actual implementation depends on your atomic line source
    # ExoJAX can use Kurucz/VALD format

    for atom, config in atomic_species.items():
        print(f"  * {atom}")
        # Placeholder for atomic line loading
        pass

    return opa_atoms, jnp.array(atommass_list)
