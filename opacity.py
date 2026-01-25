"""Opacity setup for molecular and atomic species."""

from typing import Callable
import numpy as np
import jax.numpy as jnp
import pathlib
from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa

# Opacity cache directory (relative to project root)
from config.paths_config import PROJECT_ROOT
OPA_CACHE_DIR = PROJECT_ROOT / "input" / ".opa_cache"
OPA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _patch_path_contains():
    if not hasattr(pathlib.Path, "__contains__"):
        def _contains(self, item):
            return str(item) in str(self)
        pathlib.Path.__contains__ = _contains


def _patch_radis_dbmanager():
    try:
        from radis.api import dbmanager as radis_dbmanager
    except Exception:
        return
    orig_init = getattr(radis_dbmanager.DatabaseManager, "__init__", None)
    if orig_init is None or getattr(orig_init, "_path_patched", False):
        return

    def _init(self, name, molecule, local_databases, *args, **kwargs):
        return orig_init(self, name, molecule, str(local_databases), *args, **kwargs)

    _init._path_patched = True
    radis_dbmanager.DatabaseManager.__init__ = _init


_patch_path_contains()
_patch_radis_dbmanager()


def _patch_radis_download():
    try:
        from radis.api import dbmanager as radis_dbmanager
    except Exception:
        return
    orig = getattr(radis_dbmanager.DatabaseManager, "download_and_parse", None)
    if orig is None or getattr(orig, "_no_head_patched", False):
        return

    def download_and_parse(self, urlnames, local_files, N_files_total=None):
        import re
        import requests
        from time import time

        if N_files_total is None:
            all_local_files, _ = self.get_filenames()
            N_files_total = len(all_local_files)

        verbose = self.verbose
        molecule = self.molecule
        parallel = self.parallel

        t0 = time()
        pbar_Ntot_estimate_factor = None
        if len(urlnames) != N_files_total:
            pbar_Ntot_estimate_factor = len(urlnames) / N_files_total

        Nlines_total = 0
        Ntotal_downloads = len(local_files)

        def download_and_parse_one_file(urlname, local_file, Ndownload):
            if verbose:
                inputf = urlname.split("/")[-1]
                print(
                    f"Downloading {inputf} for {molecule} ({Ndownload}/{Ntotal_downloads})."
                )

            db_path = str(self.local_databases).lower()
            if "hitemp" in db_path:
                from radis.api.hitempapi import login_to_hitran
                session = login_to_hitran(verbose=verbose)
            else:
                session = requests.Session()

            headers = {
                "Accept": "application/zip, application/octet-stream",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }

            try:
                response = session.get(
                    urlname, headers=headers, stream=True, allow_redirects=True
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    raise requests.HTTPError(
                        f"Received HTML instead of a data file for {urlname}. "
                        "HITRAN login is likely required. Set HITRAN_USERNAME and HITRAN_PASSWORD."
                    )

                temp_file_name = urlname.split("/")[-1]
                temp_file_name = re.sub(r'[<>:\"/\\\\|?*&=]', "_", temp_file_name)
                # Save downloads to the database directory, not project root
                temp_file_path = str(self.local_databases / temp_file_name)

                with open(temp_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                opener = radis_dbmanager.RequestsFileOpener(temp_file_path)

                Nlines = self.parse_to_local_file(
                    opener,
                    urlname,
                    local_file,
                    pbar_active=(not parallel),
                    pbar_t0=time() - t0,
                    pbar_Ntot_estimate_factor=pbar_Ntot_estimate_factor,
                    pbar_Nlines_already=Nlines_total,
                    pbar_last=(Ndownload == Ntotal_downloads),
                )

            except requests.RequestException as err:
                raise type(err)(
                    f"Problem downloading: {urlname}. Error: {str(err)}"
                ).with_traceback(err.__traceback__)
            except Exception as err:
                raise type(err)(
                    f"Problem parsing downloaded file from {urlname}. "
                    "Check the error above. It may arise if the file wasn't properly downloaded.\n\n"
                    f"Error: {str(err)}"
                ).with_traceback(err.__traceback__)

            return Nlines

        if parallel and len(local_files) > self.minimum_nfiles:
            nJobs = self.nJobs
            batch_size = self.batch_size
            if self.verbose:
                print(
                    f"Downloading and parsing {urlnames} to {local_files} "
                    + f"({len(local_files)}) files), in parallel ({nJobs} jobs)"
                )
            Nlines_total = sum(
                radis_dbmanager.Parallel(
                    n_jobs=nJobs, batch_size=batch_size, verbose=self.verbose
                )(
                    radis_dbmanager.delayed(download_and_parse_one_file)(
                        urlname, local_file, Ndownload
                    )
                    for urlname, local_file, Ndownload in zip(
                        urlnames, local_files, range(1, len(local_files) + 1)
                    )
                )
            )
        else:
            for urlname, local_file, Ndownload in zip(
                urlnames, local_files, range(1, len(local_files) + 1)
            ):
                download_and_parse_one_file(urlname, local_file, Ndownload)

    download_and_parse._no_head_patched = True
    radis_dbmanager.DatabaseManager.download_and_parse = download_and_parse


_patch_radis_download()


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
    opa_path = OPA_CACHE_DIR / f"opa_{mol}.zarr"
    saveopa(opa, str(opa_path), format="zarr", aux={"molmass": molmass})
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
    path = str(path)
    if opa_load:
        try:
            opa_path = OPA_CACHE_DIR / f"opa_{mol}.zarr"
            opa = OpaPremodit.from_saved_opa(str(opa_path), strict=False)
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
    db_exoatom: str | None = None,
) -> tuple[dict[str, OpaPremodit], jnp.ndarray]:
    """Load atomic LINE LISTS and compute opacities using preMODIT.
    
    Workflow:
        1. Load atomic line lists from ExoAtom (ExoMol format)
        2. Compute cross-sections using preMODIT
        3. Cache computed opacities to .zarr for fast reloading
    
    ExoAtom provides atomic line lists (Na, K, Fe, etc.) in ExoMol format.
    See: https://exomol.com/data/atoms/
    
    Args:
        atomic_species: Dict mapping atom names to config, e.g.:
            {"Na": {"element": "Na", "ionization": 0}}  # Na I (neutral)
            {"Fe": {"element": "Fe", "ionization": 0}}  # Fe I
        nu_grid: Wavenumber grid
        opa_load: Try loading cached .zarr opacity first
        ndiv: Number of DIT divisions for preMODIT
        diffmode: DIT diff mode
        Tlow, Thigh: Temperature range [K]
        db_exoatom: Path to ExoAtom line list directory
        
    Returns:
        (opa_atoms dict, atomic_masses array)
        
    Note:
        Download atomic LINE LISTS from https://exomol.com/data/atoms/
        Directory structure: {db_exoatom}/{element}/{isotope}/{linelist}/
        Example: input/.db_ExoAtom/Na/23Na/Kurucz/
    """
    opa_atoms = {}
    atommass_list = []
    
    if not atomic_species:
        return opa_atoms, jnp.array(atommass_list)
    
    # Default ExoAtom path
    if db_exoatom is None:
        from config.paths_config import PROJECT_ROOT
        db_exoatom = PROJECT_ROOT / "input" / ".db_ExoAtom"
    
    print("Loading atomic line databases (ExoAtom)...")
    
    # Atomic mass lookup (most common isotopes)
    ATOMIC_MASSES = {
        "Al": 26.98,   # 27Al
        "B": 10.81,    # 11B
        "Ba": 137.33,  # 138Ba
        "Be": 9.01,    # 9Be
        "Ca": 40.08,   # 40Ca
        "Co": 58.93,   # 59Co
        "Cr": 52.00,   # 52Cr
        "Cs": 132.91,  # 133Cs
        "Cu": 63.55,   # 63Cu
        "Fe": 55.85,   # 56Fe
        "Ga": 69.72,   # 69Ga
        "Ge": 72.63,   # 74Ge
        "Hf": 178.49,  # 180Hf
        "In": 114.82,  # 115In
        "Ir": 192.22,  # 193Ir
        "K": 39.10,    # 39K
        "Li": 6.94,    # 7Li
        "Mg": 24.31,   # 24Mg
        "Mn": 54.94,   # 55Mn
        "Mo": 95.95,   # 98Mo
        "Na": 22.99,   # 23Na
        "Nb": 92.91,   # 93Nb
        "Ni": 58.69,   # 58Ni
        "Os": 190.23,  # 192Os
        "Pb": 207.2,   # 208Pb
        "Pd": 106.42,  # 106Pd
        "Rb": 85.47,   # 85Rb
        "Rh": 102.91,  # 103Rh
        "Ru": 101.07,  # 102Ru
        "Sc": 44.96,   # 45Sc
        "Si": 28.09,   # 28Si
        "Sn": 118.71,  # 120Sn
        "Sr": 87.62,   # 88Sr
        "Ti": 47.87,   # 48Ti
        "Tl": 204.38,  # 205Tl
        "V": 50.94,    # 51V
        "W": 183.84,   # 184W
        "Y": 88.91,    # 89Y
        "Zn": 65.38,   # 64Zn
        "Zr": 91.22,   # 90Zr
    }

    # ExoAtom naming conventions: element -> path
    # Format: {element}/{mass}{element}/Kurucz
    # For ions, append "_I" or "_II" suffix when loading
    EXOATOM_PATHS = {
        "Al": "Al/27Al/Kurucz",
        "B": "B/11B/Kurucz",
        "Ba": "Ba/138Ba/Kurucz",
        "Be": "Be/9Be/Kurucz",
        "Ca": "Ca/40Ca/Kurucz",
        "Co": "Co/59Co/Kurucz",
        "Cr": "Cr/52Cr/Kurucz",
        "Cs": "Cs/133Cs/Kurucz",
        "Cu": "Cu/63Cu/Kurucz",
        "Fe": "Fe/56Fe/Kurucz",
        "Ga": "Ga/69Ga/Kurucz",
        "Ge": "Ge/74Ge/Kurucz",
        "Hf": "Hf/180Hf/Kurucz",
        "In": "In/115In/Kurucz",
        "Ir": "Ir/193Ir/Kurucz",
        "K": "K/39K/Kurucz",
        "Li": "Li/7Li/Kurucz",
        "Mg": "Mg/24Mg/Kurucz",
        "Mn": "Mn/55Mn/Kurucz",
        "Mo": "Mo/98Mo/Kurucz",
        "Na": "Na/23Na/Kurucz",
        "Nb": "Nb/93Nb/Kurucz",
        "Ni": "Ni/58Ni/Kurucz",
        "Os": "Os/192Os/Kurucz",
        "Pb": "Pb/208Pb/Kurucz",
        "Pd": "Pd/106Pd/Kurucz",
        "Rb": "Rb/85Rb/Kurucz",
        "Rh": "Rh/103Rh/Kurucz",
        "Ru": "Ru/102Ru/Kurucz",
        "Sc": "Sc/45Sc/Kurucz",
        "Si": "Si/28Si/Kurucz",
        "Sn": "Sn/120Sn/Kurucz",
        "Sr": "Sr/88Sr/Kurucz",
        "Ti": "Ti/48Ti/Kurucz",
        "Tl": "Tl/205Tl/Kurucz",
        "V": "V/51V/Kurucz",
        "W": "W/184W/Kurucz",
        "Y": "Y/89Y/Kurucz",
        "Zn": "Zn/64Zn/Kurucz",
        "Zr": "Zr/90Zr/Kurucz",
    }
    
    for atom, atom_config in atomic_species.items():
        element = atom_config.get("element", atom)
        ionization = atom_config.get("ionization", 0)

        if element not in EXOATOM_PATHS:
            print(f"  * {atom} (not in ExoAtom database, skipping)")
            continue

        # Build path - for ions, try ionization-specific subdirectory first
        base_path = EXOATOM_PATHS[element]
        if ionization > 0:
            # Try ion-specific path: e.g., Fe/56Fe/Kurucz_II or Fe/56Fe_II/Kurucz
            ion_suffix = "I" * (ionization + 1)  # I for neutral, II for +1, III for +2
            ion_paths_to_try = [
                f"{base_path}_{ion_suffix}",  # Fe/56Fe/Kurucz_II
                base_path.replace("/Kurucz", f"_{ion_suffix}/Kurucz"),  # Fe/56Fe_II/Kurucz
                base_path,  # Fall back to neutral path (some databases combine all ions)
            ]
        else:
            ion_paths_to_try = [base_path]

        exoatom_path = None
        for try_path in ion_paths_to_try:
            candidate = pathlib.Path(db_exoatom) / try_path
            if candidate.exists():
                exoatom_path = candidate
                break

        if exoatom_path is None:
            print(f"  * {atom} (ExoAtom data not found, skipping)")
            print(f"    Tried paths: {[str(pathlib.Path(db_exoatom) / p) for p in ion_paths_to_try]}")
            print(f"    Download from: https://exomol.com/data/atoms/{element}/")
            continue

        print(f"  * {atom} (ExoAtom)")

        try:
            # ExoAtom uses ExoMol format, so we can use MdbExomol
            mdb_factory = lambda p: MdbExomol(str(p), nu_grid, gpu_transfer=False)
            # Use sanitized name for cache (replace space with underscore)
            cache_name = f"atom_{atom.replace(' ', '_')}"
            opa, molmass = load_or_build_opacity(
                cache_name,
                str(exoatom_path),
                mdb_factory,
                opa_load,
                nu_grid,
                ndiv,
                diffmode,
                Tlow,
                Thigh,
            )
            opa_atoms[atom] = opa
            atommass_list.append(ATOMIC_MASSES.get(element, molmass))
        except Exception as e:
            print(f"    Warning: Failed to load {atom}: {e}")
            continue
    
    if not opa_atoms:
        print("  No atomic opacities loaded (data not available or download required)")
    
    return opa_atoms, jnp.array(atommass_list)
