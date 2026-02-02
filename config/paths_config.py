"""
Database paths, data paths, and output configuration.
"""

import os
from pathlib import Path

from .planets_config import PLANET, EPHEMERIS

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================

# Root of the project (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent

INPUT_DIR = PROJECT_ROOT / "input"
INPUT_DIR.mkdir(exist_ok=True)

# ==============================================================================
# DATABASE PATHS
# ==============================================================================

# Molecular databases (override with env vars if set)
DB_HITEMP = Path(os.environ.get("HITEMP_DIR") or INPUT_DIR / ".db_HITEMP")
DB_EXOMOL = Path(os.environ.get("EXOMOL_DIR") or INPUT_DIR / ".db_ExoMol")
DB_EXOATOM = Path(os.environ.get("EXOATOM_DIR") or INPUT_DIR / ".db_ExoAtom")
DB_KURUCZ = Path(os.environ.get("KURUCZ_DIR") or INPUT_DIR / ".db_kurucz")
DB_VALD = Path(os.environ.get("VALD_DIR") or INPUT_DIR / ".db_VALD")
DB_CIA = Path(os.environ.get("CIA_DIR") or INPUT_DIR / ".db_CIA")

# CIA paths
CIA_PATHS = {
    "H2H2": DB_CIA / "H2-H2_2011.cia",
    "H2He": DB_CIA / "H2-He_2011.cia",
}

# Molecular line lists (HITEMP)
MOLPATH_HITEMP = {
    "H2O": DB_HITEMP / "H2O",
    "CO": DB_HITEMP / "CO",
    "OH": DB_HITEMP / "OH",
}

# Molecular line lists (ExoMol)
MOLPATH_EXOMOL = {
    "TiO": DB_EXOMOL / "TiO/48Ti-16O/Toto",
    "VO": DB_EXOMOL / "VO/51V-16O/VOMYT",
    "FeH": DB_EXOMOL / "FeH/56Fe-1H/MoLLIST",
    "CaH": DB_EXOMOL / "CaH/40Ca-1H/XAB",
    "CrH": DB_EXOMOL / "CrH/52Cr-1H/MoLLIST",
    "AlO": DB_EXOMOL / "AlO/27Al-16O/ATP",
}

# Atomic line lists (Kurucz/VALD)
# Format: "Element_I" for neutral, "Element_II" for singly ionized
# Key names match spectroscopic notation (e.g., "Fe I", "Fe II")
ATOMIC_SPECIES = {
    # Neutral atoms (ionization = 0)
    "Al I": {"element": "Al", "ionization": 0},
    "B I": {"element": "B", "ionization": 0},
    "Ba I": {"element": "Ba", "ionization": 0},
    "Be I": {"element": "Be", "ionization": 0},
    "Ca I": {"element": "Ca", "ionization": 0},
    "Co I": {"element": "Co", "ionization": 0},
    "Cr I": {"element": "Cr", "ionization": 0},
    "Cs I": {"element": "Cs", "ionization": 0},
    "Cu I": {"element": "Cu", "ionization": 0},
    "Fe I": {"element": "Fe", "ionization": 0},
    "Ga I": {"element": "Ga", "ionization": 0},
    "Ge I": {"element": "Ge", "ionization": 0},
    "Hf I": {"element": "Hf", "ionization": 0},
    "In I": {"element": "In", "ionization": 0},
    "Ir I": {"element": "Ir", "ionization": 0},
    "K I": {"element": "K", "ionization": 0},
    "Li I": {"element": "Li", "ionization": 0},
    "Mg I": {"element": "Mg", "ionization": 0},
    "Mn I": {"element": "Mn", "ionization": 0},
    "Mo I": {"element": "Mo", "ionization": 0},
    "Na I": {"element": "Na", "ionization": 0},
    "Nb I": {"element": "Nb", "ionization": 0},
    "Ni I": {"element": "Ni", "ionization": 0},
    "Os I": {"element": "Os", "ionization": 0},
    "Pb I": {"element": "Pb", "ionization": 0},
    "Pd I": {"element": "Pd", "ionization": 0},
    "Rb I": {"element": "Rb", "ionization": 0},
    "Rh I": {"element": "Rh", "ionization": 0},
    "Ru I": {"element": "Ru", "ionization": 0},
    "Sc I": {"element": "Sc", "ionization": 0},
    "Si I": {"element": "Si", "ionization": 0},
    "Sn I": {"element": "Sn", "ionization": 0},
    "Sr I": {"element": "Sr", "ionization": 0},
    "Ti I": {"element": "Ti", "ionization": 0},
    "Tl I": {"element": "Tl", "ionization": 0},
    "V I": {"element": "V", "ionization": 0},
    "W I": {"element": "W", "ionization": 0},
    "Y I": {"element": "Y", "ionization": 0},
    "Zn I": {"element": "Zn", "ionization": 0},
    "Zr I": {"element": "Zr", "ionization": 0},
    # Singly ionized atoms (ionization = 1)
    "Ba II": {"element": "Ba", "ionization": 1},
    "Ca II": {"element": "Ca", "ionization": 1},
    "Cr II": {"element": "Cr", "ionization": 1},
    "Fe II": {"element": "Fe", "ionization": 1},
    "Mg II": {"element": "Mg", "ionization": 1},
    "Sc II": {"element": "Sc", "ionization": 1},
    "Sr II": {"element": "Sr", "ionization": 1},
    "Ti II": {"element": "Ti", "ionization": 1},
    "Y II": {"element": "Y", "ionization": 1},
}

# ==============================================================================
# DATA PATHS
# ==============================================================================

# Import here to avoid circular import at module level
from .instrument_config import OBSERVING_MODE
from .model_config import RETRIEVAL_MODE


def get_data_dir(planet: str | None = None, arm: str | None = None, epoch: str | None = None) -> Path:
    """Get data directory for a planet, arm, and optional epoch."""
    planet = planet or PLANET
    arm = arm or OBSERVING_MODE
    base = INPUT_DIR / "spectra" / planet.lower().replace("-", "")
    if epoch:
        return base / epoch / arm
    return base / arm


def get_transmission_paths(planet: str | None = None, arm: str | None = None, epoch: str | None = None) -> dict[str, Path]:
    """Get paths to transmission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch)
    return {
        "wavelength": data_dir / "wavelength_transmission.npy",
        "spectrum": data_dir / "spectrum_transmission.npy",
        "uncertainty": data_dir / "uncertainty_transmission.npy",
    }


def get_emission_paths(planet: str | None = None, arm: str | None = None, epoch: str | None = None) -> dict[str, Path]:
    """Get paths to emission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch)
    return {
        "wavelength": data_dir / "wavelength_emission.npy",
        "spectrum": data_dir / "spectrum_emission.npy",
        "uncertainty": data_dir / "uncertainty_emission.npy",
    }


DATA_DIR = get_data_dir()
TRANSMISSION_DATA = get_transmission_paths()
EMISSION_DATA = get_emission_paths()

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================


def get_output_dir(
    planet: str | None = None,
    ephemeris: str | None = None,
    mode: str | None = None,
) -> Path:
    """Get output directory: output/{planet}/{ephemeris}/{mode}/"""
    planet = planet or PLANET
    ephemeris = ephemeris or EPHEMERIS
    mode = mode or RETRIEVAL_MODE

    return PROJECT_ROOT / "output" / planet.lower().replace("-", "") / ephemeris / mode


def create_timestamped_dir(base_dir: str | Path) -> Path:
    """Create timestamped subdirectory within base directory.

    Args:
        base_dir: Base output directory (e.g., output/kelt20b/jpl/transmission/)

    Returns:
        Path to timestamped directory (e.g., output/.../2026-01-24_14-30-45/)
    """
    from datetime import datetime

    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_dir = base_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)

    return timestamped_dir


# Default output directory (lazy - will be set by CLI or on first use)
DIR_SAVE = None  # Set by CLI via get_output_dir()

# Opacity loading/saving
OPA_LOAD = True
OPA_SAVE = False

# Atomic database preferences
# Kurucz: auto-downloaded from kurucz.harvard.edu
# VALD: requires manual download from vald.astro.uu.se (place in .db_VALD/)
USE_KURUCZ = True
USE_VALD = True

# ==============================================================================
# DEFAULT SPECIES SELECTION
# ==============================================================================
# Species detected in literature (used by default unless --all-species is set)
# Based on high-resolution detections from multiple instruments (PEPSI, HARPS-N,
# CARMENES, EXPRES, FIES) - see literature compilation table.
#
# To use all available species instead of this subset, pass --all-species to CLI.
# To override with a custom list, use --atoms "Fe I,Na I" or --molecules "H2O,CO".

DEFAULT_SPECIES = {
    "atoms": [
        "Na I",   # Detected in multiple studies (PEPSI, HARPS-N, CARMENES, EXPRES)
        "Mg I",   # Detected (EXPRES)
        "Ca II",  # Detected (CARMENES, HARPS-N)
        "Cr I",   # Detected (PEPSI)
        "Cr II",  # Detected (EXPRES)
        "Fe I",   # Strong detection in most studies
        "Fe II",  # Strong detection in most studies
    ],
    "molecules": [
        "FeH",    # Detected (CARMENES)
    ],
}

# Set to True to use DEFAULT_SPECIES by default, False to use all available species
USE_DEFAULT_SPECIES = True
