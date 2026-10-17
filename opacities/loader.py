"""Opacity setup for molecular and atomic species."""

import hashlib
import re
from time import time
from typing import Callable
import numpy as np
import jax.numpy as jnp
import pathlib

import requests

import _numba_compat  # Configure caching before ExoJAX imports RADIS.

from exojax.database.contdb import CdbCIA
from exojax.opacity.opacont import OpaCIA
from exojax.database.api import MdbHitemp, MdbExomol
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity import saveopa
from radis.api import dbmanager as radis_dbmanager
from radis.api.hitempapi import login_to_hitran



# Opacity cache directory (relative to project root)
from config import DB_KURUCZ, DB_VALD, OPA_CACHE_DIR, USE_KURUCZ, USE_VALD
from .atomic_sources import (
    load_kurucz_atomic,
    load_vald_atomic,
    create_atomic_snapshot,
    resolve_vald_file,
)


def premodit_cache_signature(
    nu_grid: np.ndarray,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
) -> str:
    """Return a stable signature for the numerical PreMODIT grid settings."""
    grid = np.ascontiguousarray(np.asarray(nu_grid, dtype="<f8"))
    settings = np.asarray(
        [float(diffmode), float(T_low), float(T_high), _resolve_cutwing(cutwing)],
        dtype="<f8",
    )
    cache_key = hashlib.sha1()
    cache_key.update(b"premodit-opacity-v1\0")
    cache_key.update(grid.tobytes())
    cache_key.update(settings.tobytes())
    return cache_key.hexdigest()[:12]


def premodit_cache_path(
    name: str,
    nu_grid: np.ndarray,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
) -> pathlib.Path:
    """Return the coexistence-safe cache path for one PreMODIT opacity."""
    signature = premodit_cache_signature(
        nu_grid,
        diffmode,
        T_low,
        T_high,
        cutwing,
    )
    return OPA_CACHE_DIR / f"opa_{name}_{signature}.zarr"


def _patch_radis_download():
    def download_and_parse(self, urlnames, local_files, N_files_total=None):
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
                print(f"Downloading {inputf} for {molecule} ({Ndownload}/{Ntotal_downloads}).")

            db_path = str(self.local_databases).lower()
            if "hitemp" in db_path:
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

            temp_file_name = urlname.split("/")[-1]
            temp_file_name = re.sub(r'[<>:\"/\\\\|?*&=]', "_", temp_file_name)
            # Save downloads to the database directory, not project root
            base_dir = pathlib.Path(self.local_databases).expanduser().resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            target_path = base_dir / temp_file_name
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

    radis_dbmanager.DatabaseManager.download_and_parse = download_and_parse


_patch_radis_download()


def setup_cia_opacities(cia_paths: dict[str, str], nu_grid: np.ndarray) -> dict[str, OpaCIA]:
    """Setup CIA opacities for H2-H2 and H2-He."""
    opa_cias = {}

    for name, path in cia_paths.items():
        cdb = CdbCIA(str(path), nurange=nu_grid)
        if np.asarray(cdb.nucia).size == 0:
            continue
        opa_cias[name] = OpaCIA(cdb, nu_grid=nu_grid)

    return opa_cias


def _resolve_cutwing(cutwing: float | None) -> float:
    """Resolve line-wing truncation parameter for preMODIT."""
    if cutwing is None:
        return 0.5
    return float(cutwing)


def build_premodit_from_snapshot(
    snapshot: object,
    molmass: float,
    mol: str,
    nu_grid: np.ndarray,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
) -> OpaPremodit:
    """Create preMODIT opacity from database snapshot and save it."""
    cutwing_val = _resolve_cutwing(cutwing)
    opa = OpaPremodit.from_snapshot(
        snapshot,
        nu_grid,
        diffmode=diffmode,
        auto_trange=[T_low, T_high],
        dit_grid_resolution=1,
        allow_32bit=True,
        cutwing=cutwing_val,
    )
    opa_path = premodit_cache_path(
        mol,
        nu_grid,
        diffmode,
        T_low,
        T_high,
        cutwing_val,
    )
    saveopa(opa, str(opa_path), format="zarr", aux={"molmass": molmass})
    return opa


def load_or_build_opacity(
    mol: str,
    path: str,
    mdb_factory: Callable[[str], object],
    opa_load: bool,
    nu_grid: np.ndarray,
    diffmode: int,
    T_low: float,
    T_high: float,
    cutwing: float | None = None,
    load_only: bool = False,
) -> tuple[OpaPremodit | None, float | None]:
    """Load a saved opacity when opa_load is enabled, otherwise build it."""
    path = str(pathlib.Path(path).expanduser().resolve())
    cutwing_val = _resolve_cutwing(cutwing)
    if opa_load:
        opa_path = premodit_cache_path(
            mol,
            nu_grid,
            diffmode,
            T_low,
            T_high,
            cutwing_val,
        )
        opa = OpaPremodit.from_saved_opa(str(opa_path), strict=False)
        return opa, float(opa.aux["molmass"])
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
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing,
            load_only=load_only,
        )
        if opa is None:
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
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing,
            load_only=load_only,
        )
        if opa is None:
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

    With opa_load=True, read the saved .zarr opacity directly. Otherwise build
    from Kurucz when enabled, or from the configured VALD extract.

    Args:
        use_kurucz: Enable Kurucz database (default: from config.USE_KURUCZ)
        use_vald: Enable VALD database (default: from config.USE_VALD)
    """
    opa_atoms = {}
    atommass_list = []

    if not atomic_species:
        return opa_atoms, jnp.array(atommass_list)

    # Get defaults from config if not specified
    if use_kurucz is None:
        use_kurucz = USE_KURUCZ
    if use_vald is None:
        use_vald = USE_VALD

    if not use_kurucz and not use_vald:
        return opa_atoms, jnp.array(atommass_list)

    cutwing_val = _resolve_cutwing(cutwing)

    if db_kurucz is None:
        db_kurucz = DB_KURUCZ
    if db_vald is None:
        db_vald = DB_VALD
    db_kurucz = pathlib.Path(db_kurucz) if use_kurucz else None
    db_vald = pathlib.Path(db_vald) if use_vald else None

    sources_str = "/".join(filter(None, ["Kurucz" if use_kurucz else None, "VALD" if use_vald else None]))
    print(f"Loading atomic line databases ({sources_str})...")

    for atom in atomic_species:
        cache_name = f"atom_{atom.replace(' ', '_')}"

        if opa_load:
            opa_path = premodit_cache_path(
                cache_name,
                nu_grid,
                diffmode,
                T_low,
                T_high,
                cutwing_val,
            )
            opa = OpaPremodit.from_saved_opa(str(opa_path), strict=False)
            molmass = float(opa.aux["molmass"])
            opa_atoms[atom] = opa
            atommass_list.append(molmass)
            print(f"  * {atom} (cached)")
            if on_species_loaded is not None:
                on_species_loaded(f"atom:{atom}")
            continue
        elif load_only:
            print(f"  Warning: OPA_LOAD disabled; skipping {atom} (load-only).")
            continue

        if use_kurucz:
            adb, mask = load_kurucz_atomic(atom, nu_grid, db_kurucz, auto_download=auto_download)
            snapshot, molmass = create_atomic_snapshot(adb, mask=mask)
            source = "Kurucz"
        else:
            adb, mask = load_vald_atomic(atom, nu_grid, resolve_vald_file(db_vald))
            snapshot, molmass = create_atomic_snapshot(adb, mask=mask)
            source = "VALD"
        print(f"  * {atom} ({source})")
        opa = build_premodit_from_snapshot(
            snapshot,
            molmass,
            cache_name,
            nu_grid,
            diffmode,
            T_low,
            T_high,
            cutwing=cutwing_val,
        )
        opa_atoms[atom] = opa
        atommass_list.append(molmass)
        if on_species_loaded is not None:
            on_species_loaded(f"atom:{atom}")

    return opa_atoms, jnp.array(atommass_list)
