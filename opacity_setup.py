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
from config.paths import PROJECT_ROOT
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
