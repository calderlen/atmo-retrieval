"""
Database paths, data paths, and output configuration.
"""

import os
from pathlib import Path

from .planets import PLANET, EPHEMERIS

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
DB_KURUCZ = Path(os.environ.get("KURUCZ_DIR") or INPUT_DIR / ".db_kurucz")
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
ATOMIC_SPECIES = {
    "Na": {"element": "Na", "ionization": 0},  # Na I
    "K": {"element": "K", "ionization": 0},    # K I
    "Ca": {"element": "Ca", "ionization": 1},  # Ca II
    "Fe": {"element": "Fe", "ionization": 0},  # Fe I
    "Ti": {"element": "Ti", "ionization": 0},  # Ti I
    "V": {"element": "V", "ionization": 0},    # V I
}

# ==============================================================================
# DATA PATHS
# ==============================================================================

# Import here to avoid circular import at module level
from .instrument import OBSERVING_MODE
from .model import RETRIEVAL_MODE


def get_data_dir(planet: str | None = None, arm: str | None = None) -> Path:
    """Get data directory for a planet and arm."""
    planet = planet or PLANET
    arm = arm or OBSERVING_MODE
    return INPUT_DIR / "spectra" / planet.lower().replace("-", "") / arm


def get_transmission_paths(planet: str | None = None, arm: str | None = None) -> dict[str, Path]:
    """Get paths to transmission data files."""
    data_dir = get_data_dir(planet, arm=arm)
    return {
        "wavelength": data_dir / "wavelength_transmission.npy",
        "spectrum": data_dir / "spectrum_transmission.npy",
        "uncertainty": data_dir / "uncertainty_transmission.npy",
    }


def get_emission_paths(planet: str | None = None, arm: str | None = None) -> dict[str, Path]:
    """Get paths to emission data files."""
    data_dir = get_data_dir(planet, arm=arm)
    return {
        "wavelength": data_dir / "wavelength_emission.npy",
        "spectrum": data_dir / "spectrum_emission.npy",
        "uncertainty": data_dir / "uncertainty_emission.npy",
    }


# Legacy compatibility
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
