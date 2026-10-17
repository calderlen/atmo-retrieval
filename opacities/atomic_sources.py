"""Atomic line database helpers using native ExoJAX APIs (Kurucz/VALD)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple
import urllib.request

import numpy as np

# Official ExoJAX atomic database APIs (Kurucz/VALD).
from exojax.database.kurucz.api import AdbKurucz
from exojax.database.vald.api import AdbVald
from exojax.database.contracts import MDBMeta, Lines, MDBSnapshot


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
    element = parts[0]
    ionization = 0
    if len(parts) > 1:
        roman = parts[1]
        ionization = ROMAN_TO_INT[roman] - 1
    return element, ionization


def element_to_atomic_number(element: str) -> int:
    return KURUCZ_ELEMENT_CODES[element]


def ionization_to_iion(ionization: int) -> int:
    # iion convention: neutral=1, singly ionized=2, etc.
    return ionization + 1


def _kurucz_filename(code: int, ionization: int) -> str:
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
) -> Tuple[AdbKurucz, np.ndarray]:
    """Load atomic data from Kurucz gfall files."""
    element, ionization = parse_species(species)
    iion = ionization_to_iion(ionization)
    code = element_to_atomic_number(element)

    db_kurucz = Path(db_kurucz)
    filename = _kurucz_filename(code, ionization)
    filepath = db_kurucz / filename
    if auto_download and not filepath.exists():
        filepath = download_kurucz_gfall(element, ionization, db_kurucz)

    # ExoJAX 2.2.x accesses jnp-backed ielem/iion attributes during __init__,
    # so gpu_transfer must remain enabled for atomic backends.
    adb = AdbKurucz(str(filepath), nu_grid, gpu_transfer=True)
    mask = _species_mask(adb, iion, code)
    return adb, mask


def load_vald_atomic(
    species: str,
    nu_grid: np.ndarray,
    vald_file: Path,
) -> Tuple[AdbVald, np.ndarray]:
    """Load atomic data from VALD3 extract file."""
    element, ionization = parse_species(species)
    iion = ionization_to_iion(ionization)
    code = element_to_atomic_number(element)
    # ExoJAX 2.2.x accesses jnp-backed ielem/iion attributes during __init__,
    # so gpu_transfer must remain enabled for atomic backends.
    adb = AdbVald(str(vald_file), nu_grid, gpu_transfer=True)
    mask = _species_mask(adb, iion, code)
    return adb, mask


def resolve_vald_file(db_vald: Path) -> Path:
    """Use a direct VALD extract path or find an extract in a directory."""
    db_vald = Path(db_vald)
    if db_vald.is_file():
        return db_vald
    preferred = [
        "vald_extract.txt",
        "vald_extract.gz",
        "vald_extract.dat",
    ]
    for name in preferred:
        candidate = db_vald / name
        if candidate.exists():
            return candidate

    matches = [
        match
        for ext in ("*.gz", "*.txt", "*.dat")
        for match in sorted(db_vald.glob(ext))
    ]
    return matches[0]


def create_atomic_snapshot(
    adb: object,
    mask: np.ndarray | None = None,
) -> Tuple[MDBSnapshot, float]:
    """Convert AdbKurucz/AdbVald to a snapshot for OpaPremodit."""
    # Atomic preMODIT is numerically sensitive to dtype. In practice Kurucz/VALD
    # lower-state energies arrive as float32, which overflows the line-strength
    # temperature scaling for hot atmospheres and yields NaN xsmatrix values.
    nu_lines = np.asarray(_maybe_mask(adb.nu_lines, mask), dtype=np.float64)
    elower = np.asarray(_maybe_mask(adb.elower, mask), dtype=np.float64)
    Sij0 = np.asarray(_maybe_mask(adb.Sij0, mask), dtype=np.float64)

    molmass = float(_maybe_mask(adb.atomicmass, mask)[0])
    qt_idx = int(_maybe_mask(adb.QTmask, mask)[0])
    T_gQT = np.asarray(adb.T_gQT, dtype=np.float64)
    gQT = np.asarray(adb.gQT_284species, dtype=np.float64)[qt_idx]

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
    # Kurucz/VALD do not supply these ExoMol broadening parameters by default.
    n_Texp = getattr(adb, "n_Texp", None)
    alpha_ref = getattr(adb, "alpha_ref", None)
    if n_Texp is None:
        n_Texp = np.full(nlines, 0.5, dtype=np.float64)
    else:
        n_Texp = np.asarray(_maybe_mask(n_Texp, mask), dtype=np.float64)
    if alpha_ref is None:
        alpha_ref = np.full(nlines, 0.05, dtype=np.float64)
    else:
        alpha_ref = np.asarray(_maybe_mask(alpha_ref, mask), dtype=np.float64)

    snapshot = MDBSnapshot(meta=meta, lines=lines, n_Texp=n_Texp, alpha_ref=alpha_ref)
    return snapshot, float(molmass)


def _maybe_mask(arr: Iterable, mask: np.ndarray | None) -> np.ndarray:
    arr_np = np.asarray(arr)
    if mask is None:
        return arr_np
    return arr_np[mask]


def _species_mask(adb: object, iion: int, ielem: int) -> np.ndarray:
    return (np.asarray(adb.iion) == iion) & (np.asarray(adb.ielem) == ielem)
