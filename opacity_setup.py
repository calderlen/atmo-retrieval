import jax.numpy as jnp
from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa


def setup_cia_opacities(cia_paths:dict, nu_grid:np.ndarray) -> dict:
    opa_cias = {}
    for name, path in cia_paths.items():
        cdb = CdbCIA(path, nurange=nu_grid)
        opa_cias[name] = OpaCIA(cdb, nu_grid=nu_grid)
    return opa_cias




# TODO: with this function and all the following functions and for that matter, all of the functions in the codebase, specify a paramater's type and output type using type hints, not docstrings can keep the little comments for now but that's it


def build_premodit_from_snapshot(snapshot, molmass, mol, nu_grid, ndiv, diffmode, Tlow, Thigh):
    """
    Create preMODIT opacity from database snapshot and save it.

    Parameters
    ----------
    snapshot : object
        Molecular database snapshot
    molmass : float
        Molecular mass
    mol : str
        Molecule name
    nu_grid : np.ndarray
        Wavenumber grid
    ndiv : int
        Number of stitch blocks
    diffmode : int
        Diffmode parameter
    Tlow : float
        Low temperature bound [K]
    Thigh : float
        High temperature bound [K]

    Returns
    -------
    opa : OpaPremodit
        Opacity object
    """
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


def load_or_build_opacity(mol, path, mdb_factory, opa_load, nu_grid, ndiv, diffmode, Tlow, Thigh):
    """
    Load saved opacity or build from database snapshot.

    Parameters
    ----------
    mol : str
        Molecule name
    path : str
        Path to molecular database
    mdb_factory : callable
        Factory function to create MDB object
    opa_load : bool
        If True, attempt to load saved opacity
    nu_grid : np.ndarray
        Wavenumber grid
    ndiv : int
        Number of stitch blocks
    diffmode : int
        Diffmode parameter
    Tlow : float
        Low temperature [K]
    Thigh : float
        High temperature [K]

    Returns
    -------
    opa : OpaPremodit
        Opacity object
    molmass : float
        Molecular mass
    """
    if opa_load:
        try:
            opa = OpaPremodit.from_saved_opa(f"opa_{mol}.zarr", strict=False)
            return opa, opa.aux["molmass"]
        except:
            print(f"  Warning: Could not load saved opacity for {mol}, building from database...")

    mdb = mdb_factory(path)
    molmass = mdb.molmass
    opa = build_premodit_from_snapshot(
        mdb.to_snapshot(), molmass, mol, nu_grid, ndiv, diffmode, Tlow, Thigh
    )
    del mdb
    return opa, molmass


def load_molecular_opacities(molpath_hitemp, molpath_exomol, nu_grid, opa_load, ndiv, diffmode, Tlow, Thigh):
    """
    Load or create all molecular opacities for HITEMP and ExoMol.

    Parameters
    ----------
    molpath_hitemp : dict
        HITEMP molecule paths
    molpath_exomol : dict
        ExoMol molecule paths
    nu_grid : np.ndarray
        Wavenumber grid
    opa_load : bool
        Load saved opacities if True
    ndiv : int
        preMODIT stitch blocks
    diffmode : int
        Diffmode parameter
    Tlow : float
        Low temperature [K]
    Thigh : float
        High temperature [K]

    Returns
    -------
    opa_mols : dict
        Dictionary of molecular opacities
    molmass_arr : jnp.ndarray
        Array of molecular masses
    """
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


def load_atomic_opacities(atomic_species, nu_grid, opa_load, ndiv, diffmode, Tlow, Thigh):
    """
    Load atomic line opacities (e.g., Na, K, Ca).

    Parameters
    ----------
    atomic_species : dict
        Dictionary of atomic species configurations
    nu_grid : np.ndarray
        Wavenumber grid
    opa_load : bool
        Load saved opacities if True
    ndiv : int
        preMODIT stitch blocks
    diffmode : int
        Diffmode parameter
    Tlow : float
        Low temperature [K]
    Thigh : float
        High temperature [K]

    Returns
    -------
    opa_atoms : dict
        Dictionary of atomic opacities
    atommass_arr : jnp.ndarray
        Array of atomic masses
    """
    opa_atoms = {}
    atommass_list = []

    print("Loading atomic line databases...")

    # Note: Atomic line handling depends on the database format
    # This is a placeholder - actual implementation depends on your atomic line source
    # ExoJAX can use Kurucz/VALD format

    for atom, config in atomic_species.items():
        print(f"  * {atom}")
        # Placeholder for atomic line loading
        # You'll need to implement this based on your atomic line database
        # Example: MdbKurucz or custom loader

    return opa_atoms, jnp.array(atommass_list)
