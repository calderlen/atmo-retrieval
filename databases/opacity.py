"""Opacity setup for molecular and atomic species."""

from typing import Callable
import os
import numpy as np
import jax.numpy as jnp
import pathlib

# RADIS uses numba cache=True in some environments where caching is unsupported.
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")

from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa



# Opacity cache directory (relative to project root)
from config.paths_config import PROJECT_ROOT
OPA_CACHE_DIR = PROJECT_ROOT / "input" / ".opa_cache"
OPA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Atomic mass lookup (most common isotopes)
ATOMIC_MASSES = {
    "H": 1.008,   # 1H
    "He": 4.0026, # 4He
    "Al": 26.98,   # 27Al
    "B": 10.81,    # 11B
    "Ba": 137.33,  # 138Ba
    "Be": 9.01,    # 9Be
    "C": 12.01,    # 12C
    "Ca": 40.08,   # 40Ca
    "Cl": 35.45,   # 35Cl
    "Co": 58.93,   # 59Co
    "Cr": 52.00,   # 52Cr
    "Cs": 132.91,  # 133Cs
    "Cu": 63.55,   # 63Cu
    "Fe": 55.85,   # 56Fe
    "F": 19.00,    # 19F
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
    "N": 14.01,    # 14N
    "Na": 22.99,   # 23Na
    "Nb": 92.91,   # 93Nb
    "Ni": 58.69,   # 58Ni
    "O": 16.00,    # 16O
    "Os": 190.23,  # 192Os
    "P": 30.97,    # 31P
    "Pb": 207.2,   # 208Pb
    "Pd": 106.42,  # 106Pd
    "Rb": 85.47,   # 85Rb
    "Rh": 102.91,  # 103Rh
    "Ru": 101.07,  # 102Ru
    "S": 32.06,    # 32S
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

def _patch_path_contains():
    if not hasattr(pathlib.Path, "__contains__"):
        def _contains(self, item):
            return str(item) in str(self)
        pathlib.Path.__contains__ = _contains


def _patch_radis_dbmanager():
    from radis.api import dbmanager as radis_dbmanager
    orig_init = getattr(radis_dbmanager.DatabaseManager, "__init__", None)
    if orig_init is None or getattr(orig_init, "_path_patched", False):
        return

    def _init(self, name, molecule, local_databases, *args, **kwargs):
        base_dir = pathlib.Path(local_databases).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        return orig_init(self, name, molecule, str(base_dir), *args, **kwargs)

    _init._path_patched = True
    radis_dbmanager.DatabaseManager.__init__ = _init


_patch_path_contains()
_patch_radis_dbmanager()


def _patch_radis_download():
    from radis.api import dbmanager as radis_dbmanager
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
            base_dir = pathlib.Path(self.local_databases).expanduser().resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            target_path = base_dir / temp_file_name
            legacy_path = pathlib.Path(temp_file_name).resolve()
            if legacy_path.exists() and not target_path.exists():
                legacy_path.rename(target_path)
            temp_file_path = str(target_path)

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
        cdb = CdbCIA(str(path), nurange=nu_grid)
        if np.asarray(cdb.nucia).size == 0:
            raise ValueError(f"CIA {name} has no overlap with nu_grid.")
        opa_cias[name] = OpaCIA(cdb, nu_grid=nu_grid)
    return opa_cias


def _opa_grid_matches(opa: OpaPremodit, nu_grid: np.ndarray) -> bool:
    """Check whether a cached opacity matches the current wavenumber grid."""
    opa_grid = np.asarray(opa.nu_grid)
    nu_grid = np.asarray(nu_grid)
    if opa_grid.shape != nu_grid.shape:
        return False
    if opa_grid.size == 0:
        return False
    return np.isclose(opa_grid[0], nu_grid[0]) and np.isclose(opa_grid[-1], nu_grid[-1])


def _resolve_cutwing(ndiv: int, cutwing: float | None) -> float:
    """Resolve line-wing truncation parameter for preMODIT."""
    if cutwing is None:
        return 1.0 / (2 * max(int(ndiv), 1))
    return float(cutwing)


def _opa_settings_match(
    opa: OpaPremodit, ndiv: int, diffmode: int, cutwing: float
) -> bool:
    """Check whether cached opacity settings match stitching parameters."""
    aux = getattr(opa, "aux", {}) or {}
    nstitch = aux.get("nstitch")
    if nstitch is None or int(nstitch) != int(ndiv):
        return False
    dm = aux.get("diffmode")
    if dm is None or int(dm) != int(diffmode):
        return False
    cw = aux.get("cutwing")
    if cw is None or not np.isclose(float(cw), float(cutwing)):
        return False
    return True


def build_premodit_from_snapshot(
    snapshot: object,
    molmass: float,
    mol: str,
    nu_grid: np.ndarray,
    ndiv: int,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
) -> OpaPremodit:
    """Create preMODIT opacity from database snapshot and save it."""
    cutwing_val = _resolve_cutwing(ndiv, cutwing)
    if ndiv > 1 and diffmode not in (0,):
        print(
            f"  Warning: stitching (ndiv={ndiv}) is recommended with forward-mode "
            f"differentiation (diffmode=0). Current diffmode={diffmode}."
        )
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid,
        nstitch=ndiv,
        diffmode=diffmode,
        auto_trange=[T_low, T_high],
        dit_grid_resolution=1,
        allow_32bit=True,
        cutwing=cutwing_val,
    )
    opa_path = OPA_CACHE_DIR / f"opa_{mol}.zarr"
    saveopa(
        opa,
        str(opa_path),
        format="zarr",
        aux={
            "molmass": molmass,
            "nstitch": int(ndiv),
            "cutwing": float(cutwing_val),
            "diffmode": int(diffmode),
        },
    )
    return opa


def load_or_build_opacity(
    mol: str,
    path: str,
    mdb_factory: Callable[[str], object],
    opa_load: bool,
    nu_grid: np.ndarray,
    ndiv: int,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
    load_only: bool = False,
) -> tuple[OpaPremodit | None, float | None]:
    """Load saved opacity or build from database snapshot."""
    path = str(pathlib.Path(path).expanduser().resolve())
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    cutwing_val = _resolve_cutwing(ndiv, cutwing)
    if opa_load:
        opa_path = OPA_CACHE_DIR / f"opa_{mol}.zarr"
        legacy_opa_path = PROJECT_ROOT / f"opa_{mol}.zarr"
        if legacy_opa_path.exists() and not opa_path.exists():
            opa_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_opa_path.rename(opa_path)
        opa = OpaPremodit.from_saved_opa(str(opa_path), strict=False)
        if not _opa_grid_matches(opa, nu_grid):
            raise ValueError("Cached opacity grid mismatch.")
        if not _opa_settings_match(opa, ndiv, diffmode, cutwing_val):
            raise ValueError("Cached opacity stitching settings mismatch.")
        return opa, opa.aux["molmass"]
    elif load_only:
        print(f"  Warning: OPA_LOAD disabled; skipping {mol} (load-only).")
        return None, None

    mdb = mdb_factory(path)
    molmass = mdb.molmass
    opa = build_premodit_from_snapshot(
        mdb.to_snapshot(),
        molmass,
        mol,
        nu_grid,
        ndiv,
        diffmode,
        T_low,
        T_high,
        cutwing=cutwing_val,
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
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
    load_only: bool = False,
    on_species_loaded: Callable[[str], None] | None = None,
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
            mol,
            path,
            mdb_factory,
            opa_load,
            nu_grid,
            ndiv,
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing,
            load_only=load_only,
        )
        if opa is None or molmass is None:
            continue
        opa_mols[mol] = opa
        molmass_list.append(molmass)
        if on_species_loaded is not None:
            on_species_loaded(f"mol:{mol}")

    # ExoMol molecules
    for mol, path in molpath_exomol.items():
        print(f"  * {mol} (ExoMol)")
        mdb_factory = lambda p: MdbExomol(p, nu_grid, gpu_transfer=False)
        opa, molmass = load_or_build_opacity(
            mol,
            path,
            mdb_factory,
            opa_load,
            nu_grid,
            ndiv,
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing,
            load_only=load_only,
        )
        if opa is None or molmass is None:
            continue
        opa_mols[mol] = opa
        molmass_list.append(molmass)
        if on_species_loaded is not None:
            on_species_loaded(f"mol:{mol}")

    return opa_mols, jnp.array(molmass_list)


def load_atomic_opacities(
    atomic_species: dict[str, dict],
    nu_grid: np.ndarray,
    opa_load: bool,
    ndiv: int,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
    load_only: bool = False,
    on_species_loaded: Callable[[str], None] | None = None,
    db_kurucz: str | None = None,
    db_vald: str | None = None,
    auto_download: bool = True,
    use_kurucz: bool | None = None,
    use_vald: bool | None = None,
) -> tuple[dict[str, OpaPremodit], jnp.ndarray]:
    """Load atomic opacities from Kurucz or VALD databases.

    Kurucz line lists are auto-downloaded from kurucz.harvard.edu.
    VALD requires manual download from vald.astro.uu.se.

    Source priority:
        1. Cached .zarr opacity (if opa_load=True)
        2. Kurucz gfall (if use_kurucz=True, auto-download enabled)
        3. VALD3 extract file (if use_vald=True and file available)

    Args:
        use_kurucz: Enable Kurucz database (default: from config.USE_KURUCZ)
        use_vald: Enable VALD database (default: from config.USE_VALD)
    """
    opa_atoms = {}
    atommass_list = []

    if not atomic_species:
        return opa_atoms, jnp.array(atommass_list)

    # Get defaults from config if not specified
    from config.paths_config import DB_KURUCZ, DB_VALD, USE_KURUCZ, USE_VALD
    if use_kurucz is None:
        use_kurucz = USE_KURUCZ
    if use_vald is None:
        use_vald = USE_VALD

    if not use_kurucz and not use_vald:
        print("  Warning: Both USE_KURUCZ and USE_VALD are False, skipping atomic opacities")
        return opa_atoms, jnp.array(atommass_list)

    cutwing_val = _resolve_cutwing(ndiv, cutwing)

    # Import Kurucz/VALD helpers
    from databases.atomic import (
        load_kurucz_atomic,
        load_vald_atomic,
        create_atomic_snapshot,
        resolve_vald_file,
        parse_species,
        ATOMIC_MASSES,
    )

    if db_kurucz is None:
        db_kurucz = DB_KURUCZ
    if db_vald is None:
        db_vald = DB_VALD
    db_kurucz = pathlib.Path(db_kurucz) if use_kurucz else None
    db_vald = pathlib.Path(db_vald) if use_vald and db_vald is not None else None
    vald_file = resolve_vald_file(db_vald) if db_vald is not None else None

    sources_str = "/".join(filter(None, ["Kurucz" if use_kurucz else None, "VALD" if use_vald else None]))
    print(f"Loading atomic line databases ({sources_str})...")

    for atom, atom_meta in atomic_species.items():
        cache_name = f"atom_{atom.replace(' ', '_')}"

        if opa_load:
            opa_path = OPA_CACHE_DIR / f"opa_{cache_name}.zarr"
            opa = OpaPremodit.from_saved_opa(str(opa_path), strict=False)
            if not _opa_grid_matches(opa, nu_grid):
                raise ValueError("Cached opacity grid mismatch.")
            if not _opa_settings_match(opa, ndiv, diffmode, cutwing_val):
                raise ValueError("Cached opacity stitching settings mismatch.")
            molmass = opa.aux.get("molmass", None)
            if molmass is None:
                raise KeyError("Missing molmass in cached opacity.")
            opa_atoms[atom] = opa
            atommass_list.append(molmass)
            print(f"  * {atom} (cached)")
            if on_species_loaded is not None:
                on_species_loaded(f"atom:{atom}")
            continue
        elif load_only:
            print(f"  Warning: OPA_LOAD disabled; skipping {atom} (load-only).")
            continue

        if use_kurucz and db_kurucz is not None:
            adb, mask = load_kurucz_atomic(atom, nu_grid, db_kurucz, auto_download=auto_download)
            snapshot, molmass = create_atomic_snapshot(adb, mask=mask)
            source = "Kurucz"
        elif use_vald and vald_file is not None:
            adb, mask = load_vald_atomic(atom, nu_grid, vald_file)
            snapshot, molmass = create_atomic_snapshot(adb, mask=mask)
            source = "VALD"
        else:
            raise ValueError(f"No atomic database source available for {atom}.")

        # Fix missing molmass
        if molmass is None or (isinstance(molmass, float) and np.isnan(molmass)):
            element, _ = parse_species(atom)
            molmass = ATOMIC_MASSES.get(element, molmass)

        if molmass is None:
            raise ValueError(f"Could not determine atomic mass for {atom}.")

        print(f"  * {atom} ({source})")
        opa = build_premodit_from_snapshot(
            snapshot,
            molmass,
            cache_name,
            nu_grid,
            ndiv,
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing_val,
        )
        opa_atoms[atom] = opa
        atommass_list.append(molmass)
        if on_species_loaded is not None:
            on_species_loaded(f"atom:{atom}")

    if not opa_atoms:
        print("  No atomic opacities loaded (data not available or download required)")

    return opa_atoms, jnp.array(atommass_list)
