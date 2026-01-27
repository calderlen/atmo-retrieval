"""Atomic line database helpers using native ExoJAX APIs (Kurucz/VALD)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import urllib.request

import numpy as np

try:
    # Official ExoJAX atomic database APIs (Kurucz/VALD).
    from exojax.database.kurucz.api import AdbKurucz
    from exojax.database.vald.api import AdbVald
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "ExoJAX Kurucz/VALD APIs are not available. "
        "Install/upgrade ExoJAX so `exojax.database.kurucz.api` and "
        "`exojax.database.vald.api` are present."
    ) from exc

try:
    from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot
except Exception as exc:  # pragma: no cover
    raise ImportError("ExoJAX MDBSnapshot contracts are required for atomic snapshots.") from exc


KURUCZ_BASE_URL = "http://kurucz.harvard.edu/linelists/gfall"

# Kurucz element codes for gfall downloads (atomic number)
KURUCZ_ELEMENT_CODES = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "In": 49,
    "Sn": 50,
    "Cs": 55,
    "Ba": 56,
    "Hf": 72,
    "W": 74,
    "Os": 76,
    "Ir": 77,
    "Tl": 81,
    "Pb": 82,
}

# Fallback atomic masses (amu) for mean molecular weight bookkeeping
ATOMIC_MASSES = {
    "Al": 26.98,
    "B": 10.81,
    "Ba": 137.33,
    "Be": 9.01,
    "Ca": 40.08,
    "Co": 58.93,
    "Cr": 52.00,
    "Cs": 132.91,
    "Cu": 63.55,
    "Fe": 55.85,
    "Ga": 69.72,
    "Ge": 72.63,
    "Hf": 178.49,
    "In": 114.82,
    "Ir": 192.22,
    "K": 39.10,
    "Li": 6.94,
    "Mg": 24.31,
    "Mn": 54.94,
    "Mo": 95.95,
    "Na": 22.99,
    "Nb": 92.91,
    "Ni": 58.69,
    "Os": 190.23,
    "Pb": 207.2,
    "Pd": 106.42,
    "Rb": 85.47,
    "Rh": 102.91,
    "Ru": 101.07,
    "Sc": 44.96,
    "Si": 28.09,
    "Sn": 118.71,
    "Sr": 87.62,
    "Ti": 47.87,
    "Tl": 204.38,
    "V": 50.94,
    "W": 183.84,
    "Y": 88.91,
    "Zn": 65.38,
    "Zr": 91.22,
}

ROMAN_TO_INT = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
}


def parse_species(species: str) -> Tuple[str, int]:
    """Parse species label into element and ionization (0=neutral, 1=II, ...)."""
    parts = species.strip().split()
    if not parts:
        raise ValueError("Empty species name.")
    element = parts[0]
    ionization = 0
    if len(parts) > 1:
        roman = parts[1]
        if roman not in ROMAN_TO_INT:
            raise ValueError(f"Unsupported ionization label: {species}")
        ionization = ROMAN_TO_INT[roman] - 1
    return element, ionization


def element_to_atomic_number(element: str) -> int:
    if element not in KURUCZ_ELEMENT_CODES:
        raise KeyError(f"Element not in Kurucz mapping: {element}")
    return KURUCZ_ELEMENT_CODES[element]


def ionization_to_iion(ionization: int) -> int:
    # iion convention: neutral=1, singly ionized=2, etc.
    return ionization + 1


def _kurucz_filename(code: int, ionization: int) -> str:
    if ionization < 0 or ionization > 99:
        raise ValueError(f"Unsupported ionization state: {ionization}")
    return f"gf{code:02d}{ionization:02d}.all"


def download_kurucz_gfall(
    element: str,
    ionization: int,
    output_dir: Path,
    timeout: int = 60,
) -> Path:
    """Download Kurucz gfall line list for element/ionization."""
    code = element_to_atomic_number(element)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _kurucz_filename(code, ionization)
    dest = output_dir / filename
    if dest.exists():
        return dest

    url = f"{KURUCZ_BASE_URL}/{filename}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"Failed to download {url} (HTTP {resp.status})")
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return dest


def load_kurucz_atomic(
    species: str,
    nu_grid: np.ndarray,
    db_kurucz: Path,
    auto_download: bool = True,
) -> Tuple[AdbKurucz, np.ndarray | None]:
    """Load atomic data from Kurucz gfall files."""
    element, ionization = parse_species(species)
    iion = ionization_to_iion(ionization)
    code = element_to_atomic_number(element)

    db_kurucz = Path(db_kurucz)
    filename = _kurucz_filename(code, ionization)
    filepath = db_kurucz / filename
    if auto_download and not filepath.exists():
        filepath = download_kurucz_gfall(element, ionization, db_kurucz)
    if not filepath.exists():
        raise FileNotFoundError(f"Kurucz gfall file not found: {filepath}")

    try:
        adb = AdbKurucz(str(filepath), nu_grid, gpu_transfer=False)
    except AttributeError as exc:
        # ExoJAX 2.2.x can access ielem/iion even when gpu_transfer=False.
        # Retry with gpu_transfer=True for compatibility.
        if "ielem" in str(exc) or "iion" in str(exc):
            adb = AdbKurucz(str(filepath), nu_grid, gpu_transfer=True)
        else:
            raise
    mask = _species_mask(adb, element, iion, code)
    return adb, mask


def load_vald_atomic(
    species: str,
    nu_grid: np.ndarray,
    vald_file: Path,
) -> Tuple[AdbVald, np.ndarray | None]:
    """Load atomic data from VALD3 extract file."""
    element, ionization = parse_species(species)
    iion = ionization_to_iion(ionization)
    code = element_to_atomic_number(element)
    try:
        adb = AdbVald(str(vald_file), nu_grid, gpu_transfer=False)
    except AttributeError as exc:
        if "ielem" in str(exc) or "iion" in str(exc):
            adb = AdbVald(str(vald_file), nu_grid, gpu_transfer=True)
        else:
            raise
    mask = _species_mask(adb, element, iion, code)
    return adb, mask


def resolve_vald_file(db_vald: Path) -> Path | None:
    """Find a VALD extract file in a directory, or validate a direct file path."""
    db_vald = Path(db_vald)
    if db_vald.is_file():
        return db_vald
    if not db_vald.exists():
        return None

    preferred = [
        "vald_extract.txt",
        "vald_extract.gz",
        "vald_extract.dat",
    ]
    for name in preferred:
        candidate = db_vald / name
        if candidate.exists():
            return candidate

    for ext in ("*.gz", "*.txt", "*.dat"):
        matches = sorted(db_vald.glob(ext))
        if matches:
            return matches[0]
    return None


def create_atomic_snapshot(
    adb: object,
    mask: np.ndarray | None = None,
    Tref: float = 296.0,
) -> Tuple[MDBSnapshot, float]:
    """Convert AdbKurucz/AdbVald to a snapshot for OpaPremodit."""
    nu_lines = _maybe_mask(_get_adb_attr(adb, "nu_lines"), mask)
    elower = _maybe_mask(_get_adb_attr(adb, "elower"), mask)
    Sij0 = _maybe_mask(_get_adb_attr(adb, "Sij0"), mask)

    molmass = _extract_molmass(adb, mask)

    T_gQT, gQT = _extract_partition(adb, mask, Tref)

    lines = Lines(
        nu_lines=nu_lines,
        elower=elower,
        line_strength_ref_original=Sij0,
    )

    meta = MDBMeta(
        dbtype="exomol",
        molmass=float(molmass),
        T_gQT=T_gQT,
        gQT=gQT,
    )

    nlines = len(nu_lines)
    n_Texp = _maybe_mask(_get_adb_attr(adb, "n_Texp"), mask)
    alpha_ref = _maybe_mask(_get_adb_attr(adb, "alpha_ref"), mask)
    if n_Texp is None:
        n_Texp = np.full(nlines, 0.5)
    if alpha_ref is None:
        alpha_ref = np.full(nlines, 0.05)

    snapshot = MDBSnapshot(meta=meta, lines=lines, n_Texp=n_Texp, alpha_ref=alpha_ref)
    return snapshot, float(molmass)


def _maybe_mask(arr: Iterable | None, mask: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    arr_np = np.asarray(arr)
    if mask is None:
        return arr_np
    return arr_np[mask]


def _get_adb_attr(adb: object, name: str, default=None):
    """Get attribute from AdbKurucz/AdbVald, checking underscore prefix fallback.

    ExoJAX uses underscore-prefixed names (e.g., _elower) when gpu_transfer=False,
    and non-prefixed names (e.g., elower) when gpu_transfer=True.
    """
    if hasattr(adb, name):
        return getattr(adb, name)
    underscore_name = f"_{name}"
    if hasattr(adb, underscore_name):
        return getattr(adb, underscore_name)
    return default


def _extract_molmass(adb: object, mask: np.ndarray | None) -> float:
    amass = _get_adb_attr(adb, "atomicmass")
    if amass is not None:
        amass = _maybe_mask(amass, mask)
        if amass is not None and len(amass) > 0:
            return float(np.asarray(amass)[0])
    # Fallback to periodic table lookup by element label if available
    ielem = _get_adb_attr(adb, "ielem")
    if ielem is not None:
        ielem = _maybe_mask(ielem, mask)
        if ielem is not None and len(ielem) > 0:
            element = _element_from_atomic_number(int(np.asarray(ielem)[0]))
            return float(ATOMIC_MASSES.get(element, np.nan))
    return float("nan")


def _extract_partition(adb: object, mask: np.ndarray | None, Tref: float) -> Tuple[np.ndarray, np.ndarray]:
    # Prefer the 284-species partition function grid if available.
    gQT_284 = _get_adb_attr(adb, "gQT_284species")
    T_gQT = _get_adb_attr(adb, "T_gQT")
    qtmask_arr = _get_adb_attr(adb, "QTmask")
    if gQT_284 is not None and T_gQT is not None and qtmask_arr is not None:
        qtmask = _maybe_mask(qtmask_arr, mask)
        if qtmask is not None and len(qtmask) > 0:
            qt_idx = int(np.asarray(qtmask)[0])
            gQT = np.asarray(gQT_284)[qt_idx]
            return np.asarray(T_gQT), gQT

    # Fallback: flat partition function
    return np.array([Tref], dtype=float), np.array([1.0], dtype=float)


def _species_mask(adb: object, element: str, iion: int, ielem: int | None = None) -> np.ndarray | None:
    mask = None
    # Check for iion or _iion (ExoJAX uses _iion when gpu_transfer=False)
    iion_attr = "iion" if hasattr(adb, "iion") else "_iion" if hasattr(adb, "_iion") else None
    if iion_attr:
        iion_arr = np.asarray(getattr(adb, iion_attr))
        mask = (iion_arr == iion) if mask is None else (mask & (iion_arr == iion))

    # Check for ielem or _ielem (ExoJAX uses _ielem when gpu_transfer=False)
    ielem_attr = "ielem" if hasattr(adb, "ielem") else "_ielem" if hasattr(adb, "_ielem") else None
    if ielem_attr:
        if ielem is None:
            ielem = element_to_atomic_number(element)
        ielem_arr = np.asarray(getattr(adb, ielem_attr))
        mask = (ielem_arr == ielem) if mask is None else (mask & (ielem_arr == ielem))

    if mask is not None and not np.any(mask):
        raise ValueError(f"No lines found for {element} ionization {iion}")
    return mask


def _element_from_atomic_number(atomic_number: int) -> str:
    for el, num in KURUCZ_ELEMENT_CODES.items():
        if num == atomic_number:
            return el
    return "X"
