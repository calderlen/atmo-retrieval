"""
KELT-20b Ultra-Hot Jupiter Retrieval Configuration
===================================================

System parameters, paths, and configuration for PEPSI observations.
"""

import os
from uncertainties import ufloat

# ==============================================================================
# TARGET SYSTEM PARAMETERS
# ==============================================================================

# Active planet and ephemeris (can be overridden via CLI)
PLANET = "KELT-20b"
EPHEMERIS = "Duck24"

# Planet parameters dictionary
# Format: PLANETS[planet_name][ephemeris_source]
PLANETS = {
    "KELT-20b": {
        "Duck24": {
            "period": ufloat(3.47410151, 0.00000012),  # days
            "epoch": ufloat(2459757.811, 0.000019),    # BJD_TDB
            "duration": 0.14762,                        # days
            "inclination": ufloat(86.065, 0.073),      # degrees
            "M_star": ufloat(1.76, 0.19),              # M_Sun
            "R_star": ufloat(1.60, 0.06),              # R_Sun
            "T_star": 8980,                            # K (A2V star)
            "M_p": ufloat(3.382, 0.13),                # M_J (3-sigma)
            "R_p": ufloat(1.741, 0.07),                # R_J
            "a": 0.0542,                                # AU
            "RV_abs": ufloat(0.0, 0.0),                # km/s
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
        "Singh24": {
            "period": ufloat(3.4741039, 0.0000040),
            "epoch": ufloat(2459406.927174, 0.000024),
            "duration": 0.1475,
            "inclination": ufloat(86.03, 0.05),
            "M_star": ufloat(1.76, 0.19),
            "R_star": ufloat(1.60, 0.06),
            "T_star": 8980,
            "M_p": ufloat(3.382, 0.13),
            "R_p": ufloat(1.741, 0.07),
            "a": 0.0542,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
        "Lund17": {
            "period": ufloat(3.4741085, 0.0000019),
            "epoch": ufloat(2457503.120049, 0.000190),
            "duration": 0.14898,
            "inclination": ufloat(86.12, 0.28),
            "M_star": ufloat(1.89, 0.06),
            "R_star": ufloat(1.60, 0.06),
            "T_star": 8980,
            "M_p": ufloat(3.382, 0.13),
            "R_p": ufloat(1.735, 0.07),
            "a": 0.0542,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
    },
    "WASP-76b": {
        "West16": {
            "period": ufloat(1.809886, 0.000001),
            "epoch": ufloat(2456107.85507, 0.00034),
            "duration": 3.694 / 24.0,
            "inclination": ufloat(88.0, 1.6),
            "M_star": ufloat(1.46, 0.07),
            "R_star": ufloat(1.73, 0.04),
            "T_star": 6329,
            "M_p": ufloat(0.92, 0.03),
            "R_p": ufloat(1.83, 0.06),
            "a": 0.033,
            "RV_abs": ufloat(-1.152, 0.0033),
            "RA": "01h46m31.90s",
            "Dec": "+02d42m01.40s",
        },
    },
    "KELT-9b": {
        "Gaudi17": {
            "period": ufloat(1.4811235, 0.0000011),
            "epoch": ufloat(2457095.68572, 0.00014),
            "duration": 3.9158 / 24.0,
            "inclination": ufloat(86.79, 0.25),
            "M_star": ufloat(2.11, 0.78),
            "R_star": ufloat(2.362, 0.075),
            "T_star": 10170,
            "M_p": ufloat(2.17, 0.56),
            "R_p": ufloat(1.891, 0.061),
            "a": 0.03462,
            "RV_abs": ufloat(-37.11, 1.0),
            "RA": "20h31m26.38s",
            "Dec": "+39d56m20.10s",
        },
    },
    "WASP-12b": {
        "Ivshina22": {
            "period": ufloat(1.091419108, 5.5e-08),
            "epoch": ufloat(2457010.512173, 7e-05),
            "duration": 3.0408 / 24.0,
            "inclination": ufloat(83.3, 1.1),
            "M_star": ufloat(1.38, 0.18),
            "R_star": ufloat(1.619, 0.065),
            "T_star": 6300,
            "M_p": ufloat(1.39, 0.12),
            "R_p": ufloat(1.937, 0.064),
            "a": 0.0234,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "06h30m32.79s",
            "Dec": "+29d40m20.16s",
        },
    },
    "WASP-33b": {
        "Ivshina22": {
            "period": ufloat(1.219870, 0.000001),
            "epoch": ufloat(2454163.22367, 0.00022),
            "duration": 2.854 / 24.0,
            "inclination": ufloat(86.63, 0.03),
            "M_star": ufloat(1.495, 0.031),
            "R_star": ufloat(1.509, 0.043),
            "T_star": 7430,
            "M_p": ufloat(2.093, 0.139),
            "R_p": ufloat(1.593, 0.054),
            "a": 0.02558,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "02h26m51.06s",
            "Dec": "+37d33m01.60s",
        },
    },
    "WASP-18b": {
        "Cortes-Zuleta20": {
            "period": ufloat(0.94145223, 0.00000024),
            "epoch": ufloat(2456740.80560, 0.00019),
            "duration": 2.21 / 24.0,
            "inclination": ufloat(83.5, 2.0),
            "M_star": ufloat(1.294, 0.063),
            "R_star": ufloat(1.23, 0.05),
            "T_star": 6400,
            "M_p": ufloat(10.20, 0.35),
            "R_p": ufloat(1.240, 0.079),
            "a": 0.02047,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "01h37m25.07s",
            "Dec": "-45d40m40.06s",
        },
    },
    "WASP-189b": {
        "Anderson18": {
            "period": ufloat(2.7240308, 0.0000028),
            "epoch": ufloat(2458926.5416960, 0.0000650),
            "duration": 4.3336 / 24.0,
            "inclination": ufloat(84.03, 0.14),
            "M_star": ufloat(2.030, 0.066),
            "R_star": ufloat(2.36, 0.030),
            "T_star": 8000,
            "M_p": ufloat(1.99, 0.16),
            "R_p": ufloat(1.619, 0.021),
            "a": 0.05053,
            "RV_abs": ufloat(-22.4, 0.0),
            "RA": "15h02m44.82s",
            "Dec": "-03d01m53.35s",
        },
    },
    "MASCARA-1b": {
        "Talens17": {
            "period": ufloat(2.14877381, 0.00000088),
            "epoch": ufloat(2458833.488151, 0.000092),
            "duration": 4.226 / 24.0,
            "inclination": ufloat(88.45, 0.17),
            "M_star": ufloat(1.900, 0.068),
            "R_star": ufloat(2.082, 0.038),
            "T_star": 7554,
            "M_p": ufloat(3.7, 0.9),
            "R_p": ufloat(1.597, 0.037),
            "a": 0.04034,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "21h10m12.37s",
            "Dec": "+10d44m20.03s",
        },
    },
    "TOI-1431b": {
        "Addison21": {
            "period": ufloat(2.650237, 0.000003),
            "epoch": ufloat(2458739.17737, 0.00007),
            "duration": 2.489 / 24.0,
            "inclination": ufloat(80.13, 0.13),
            "M_star": ufloat(1.90, 0.10),
            "R_star": ufloat(1.92, 0.07),
            "T_star": 7690,
            "M_p": ufloat(3.12, 0.18),
            "R_p": ufloat(1.49, 0.05),
            "a": 0.046,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "21h04m48.89s",
            "Dec": "+55d35m16.88s",
            "Ks_expected": 294.1,  # m/s
        },
    },
    "TOI-1518b": {
        "Cabot21": {
            "period": ufloat(1.902603, 0.000011),
            "epoch": ufloat(2458787.049255, 0.000094),
            "duration": 2.365 / 24.0,
            "inclination": ufloat(77.84, 0.26),
            "M_star": ufloat(1.79, 0.26),
            "R_star": ufloat(1.95, 0.08),
            "T_star": 7300,
            "M_p": ufloat(2.3, 2.3),
            "R_p": ufloat(1.875, 0.053),
            "a": 0.0389,
            "RV_abs": ufloat(0.0, 0.0),
            "RA": "23h29m04.20s",
            "Dec": "+67d02m05.30s",
        },
    },
}


def get_params(planet: str | None = None, ephemeris: str | None = None) -> dict:
    """Get planet parameters for the specified planet and ephemeris."""
    planet = planet or PLANET
    ephemeris = ephemeris or EPHEMERIS
    
    if planet not in PLANETS:
        raise ValueError(f"Unknown planet: {planet}. Available: {list(PLANETS.keys())}")
    
    if ephemeris not in PLANETS[planet]:
        available = list(PLANETS[planet].keys())
        raise ValueError(f"Unknown ephemeris '{ephemeris}' for {planet}. Available: {available}")
    
    return PLANETS[planet][ephemeris]


def list_planets() -> list[str]:
    """List all available planets."""
    return list(PLANETS.keys())


def list_ephemerides(planet: str | None = None) -> list[str]:
    """List available ephemerides for a planet."""
    planet = planet or PLANET
    if planet not in PLANETS:
        raise ValueError(f"Unknown planet: {planet}")
    return list(PLANETS[planet].keys())


# ==============================================================================
# OBSERVATION PARAMETERS - PEPSI/LBT
# ==============================================================================

OBSERVING_MODE = "red"  # Options: "blue", "green", "red", "full"

WAVELENGTH_RANGES = {
    "blue": (4752, 5425),    # Angstroms, PEPSI blue arm
    "green": (4760, 6570),   # Angstroms, CD3+CD4 (approximate)
    "red": (6231, 7427),     # Angstroms, PEPSI red arm
    "full": (4752, 7427),    # Angstroms, both arms combined
}

WAV_MIN, WAV_MAX = WAVELENGTH_RANGES[OBSERVING_MODE]

RESOLUTION = 120000  # PEPSI high-resolution mode

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 100  # Number of atmospheric layers

# Pressure range [bar]
PRESSURE_TOP = 1e-8   # Extended for ultra-hot Jupiter
PRESSURE_BTM = 1e2

# Temperature range [K] - Ultra-hot Jupiter
TLOW = 1500.0   # Cooler nightside
THIGH = 4500.0  # Very hot dayside with thermal inversion

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 100000  # Higher resolution for PEPSI
WAV_MIN_OFFSET = 100  # Angstroms
WAV_MAX_OFFSET = 100  # Angstroms

# preMODIT parameters
NDIV = 8  # More divisions for higher resolution

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0  # Lower than WASP-39b (UHJs have minimal clouds on dayside)

# ==============================================================================
# DATABASE PATHS (relative to input/)
# ==============================================================================

INPUT_DIR = os.path.join(os.path.dirname(__file__), "input")
os.makedirs(INPUT_DIR, exist_ok=True)

# Molecular databases (override with env vars if set)
DB_HITEMP = os.environ.get("HITEMP_DIR") or os.path.join(INPUT_DIR, ".db_HITEMP/")
DB_EXOMOL = os.environ.get("EXOMOL_DIR") or os.path.join(INPUT_DIR, ".db_ExoMol/")
DB_KURUCZ = os.environ.get("KURUCZ_DIR") or os.path.join(INPUT_DIR, ".db_kurucz/")
DB_CIA = os.environ.get("CIA_DIR") or os.path.join(INPUT_DIR, ".db_CIA/")

# CIA paths
CIA_PATHS = {
    "H2H2": os.path.join(DB_CIA, "H2-H2_2011.cia"),
    "H2He": os.path.join(DB_CIA, "H2-He_2011.cia"),
}

# Molecular line lists (HITEMP)
MOLPATH_HITEMP = {
    "H2O": os.path.join(DB_HITEMP, "H2O/"),
    "CO": os.path.join(DB_HITEMP, "CO/"),
    "OH": os.path.join(DB_HITEMP, "OH/"),
}

# Molecular line lists (ExoMol)
MOLPATH_EXOMOL = {
    "TiO": os.path.join(DB_EXOMOL, "TiO/48Ti-16O/Toto/"),
    "VO": os.path.join(DB_EXOMOL, "VO/51V-16O/VOMYT/"),
    "FeH": os.path.join(DB_EXOMOL, "FeH/56Fe-1H/MoLLIST/"),
    "CaH": os.path.join(DB_EXOMOL, "CaH/40Ca-1H/XAB/"),
    "CrH": os.path.join(DB_EXOMOL, "CrH/52Cr-1H/MoLLIST/"),
    "AlO": os.path.join(DB_EXOMOL, "AlO/27Al-16O/ATP/"),
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
# TELLURIC LINE MODELING
# ==============================================================================

ENABLE_TELLURICS = True

TELLURIC_SPECIES = {
    "H2O": os.path.join(DB_HITEMP, "H2O/"),
    "O2": os.path.join(DB_HITEMP, "O2/"),
}

# Typical telluric parameters (can be free parameters in retrieval)
TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass

# ==============================================================================
# DATA PATHS
# ==============================================================================

def get_data_dir(planet: str | None = None, arm: str | None = None) -> str:
    """Get data directory for a planet and arm."""
    planet = planet or PLANET
    arm = arm or OBSERVING_MODE
    return os.path.join(INPUT_DIR, "spectra", planet.lower().replace("-", ""), arm)


def get_transmission_paths(planet: str | None = None, arm: str | None = None) -> dict[str, str]:
    """Get paths to transmission data files."""
    data_dir = get_data_dir(planet, arm=arm)
    return {
        "wavelength": os.path.join(data_dir, "wavelength_transmission.npy"),
        "spectrum": os.path.join(data_dir, "spectrum_transmission.npy"),
        "uncertainty": os.path.join(data_dir, "uncertainty_transmission.npy"),
    }


def get_emission_paths(planet: str | None = None, arm: str | None = None) -> dict[str, str]:
    """Get paths to emission data files."""
    data_dir = get_data_dir(planet, arm=arm)
    return {
        "wavelength": os.path.join(data_dir, "wavelength_emission.npy"),
        "spectrum": os.path.join(data_dir, "spectrum_emission.npy"),
        "uncertainty": os.path.join(data_dir, "uncertainty_emission.npy"),
    }


# Legacy compatibility
DATA_DIR = get_data_dir()
TRANSMISSION_DATA = get_transmission_paths()
EMISSION_DATA = get_emission_paths()

# ==============================================================================
# RETRIEVAL MODE (must be before output config)
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

def get_output_dir(
    planet: str | None = None,
    ephemeris: str | None = None,
    mode: str | None = None,
) -> str:
    """Get output directory: output/{planet}/{ephemeris}/{mode}/"""
    planet = planet or PLANET
    ephemeris = ephemeris or EPHEMERIS
    mode = mode or RETRIEVAL_MODE

    base = os.path.join(os.path.dirname(__file__), "output")
    return os.path.join(base, planet.lower().replace("-", ""), ephemeris, mode)


def create_timestamped_dir(base_dir: str) -> str:
    """Create timestamped subdirectory within base directory.

    Args:
        base_dir: Base output directory (e.g., output/kelt20b/jpl/transmission/)

    Returns:
        Path to timestamped directory (e.g., output/.../2026-01-24_14-30-45/)
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_dir = os.path.join(base_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)

    return timestamped_dir


def save_run_config(
    output_dir: str,
    mode: str,
    temperature_profile: str,
    skip_svi: bool,
    svi_only: bool,
    seed: int,
) -> None:
    """Save run configuration to log file.

    Args:
        output_dir: Directory to save config log
        mode: Retrieval mode (transmission/emission)
        temperature_profile: Temperature profile type
        skip_svi: Whether SVI was skipped
        svi_only: Whether only SVI was run
        seed: Random seed
    """
    from datetime import datetime
    import platform
    import jax

    log_path = os.path.join(output_dir, "run_config.log")

    params = get_params()

    with open(log_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("RETRIEVAL RUN CONFIGURATION\n")
        f.write("="*70 + "\n\n")

        # Timestamp
        f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {seed}\n\n")

        # System info
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"JAX version: {jax.__version__}\n")
        f.write(f"JAX backend: {jax.default_backend()}\n")
        f.write(f"JAX devices: {jax.devices()}\n\n")

        # Target info
        f.write("TARGET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Planet: {PLANET}\n")
        f.write(f"Ephemeris: {EPHEMERIS}\n")
        f.write(f"Period: {params['period']}\n")
        f.write(f"R_p: {params['R_p']}\n")
        f.write(f"M_p: {params['M_p']}\n")
        f.write(f"R_star: {params['R_star']}\n")
        f.write(f"T_star: {params['T_star']}\n\n")

        # Retrieval mode
        f.write("RETRIEVAL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Temperature profile: {temperature_profile}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        # Wavelength/spectral setup
        f.write("SPECTRAL SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Observing mode: {OBSERVING_MODE}\n")
        f.write(f"Wavelength range: {WAV_MIN} - {WAV_MAX} Angstroms\n")
        f.write(f"Spectral points: {N_SPECTRAL_POINTS}\n")
        f.write(f"Resolution: R = {RESOLUTION:,}\n\n")

        # Atmospheric setup
        f.write("ATMOSPHERIC SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Layers: {NLAYER}\n")
        f.write(f"Pressure range: {PRESSURE_TOP:.2e} - {PRESSURE_BTM:.2e} bar\n")
        f.write(f"Temperature range: {TLOW} - {THIGH} K\n")
        f.write(f"Cloud width: {CLOUD_WIDTH}\n")
        f.write(f"Cloud integrated tau: {CLOUD_INTEGRATED_TAU}\n\n")

        # Molecules
        f.write("OPACITY SOURCES\n")
        f.write("-" * 70 + "\n")
        f.write("Molecules (HITEMP):\n")
        for mol in MOLPATH_HITEMP.keys():
            f.write(f"  - {mol}\n")
        f.write("Molecules (ExoMol):\n")
        for mol in MOLPATH_EXOMOL.keys():
            f.write(f"  - {mol}\n")
        f.write(f"\nCIA sources: H2-H2, H2-He\n")
        f.write(f"Opacity loading: {OPA_LOAD}\n")
        f.write(f"Opacity saving: {OPA_SAVE}\n\n")

        # Inference parameters
        f.write("INFERENCE PARAMETERS\n")
        f.write("-" * 70 + "\n")
        if not skip_svi:
            f.write(f"SVI steps: {SVI_NUM_STEPS:,}\n")
            f.write(f"SVI learning rate: {SVI_LEARNING_RATE}\n")
        else:
            f.write("SVI: SKIPPED\n")

        if not svi_only:
            f.write(f"\nMCMC warmup: {MCMC_NUM_WARMUP:,}\n")
            f.write(f"MCMC samples: {MCMC_NUM_SAMPLES:,}\n")
            f.write(f"MCMC chains: {MCMC_NUM_CHAINS}\n")
            f.write(f"MCMC max tree depth: {MCMC_MAX_TREE_DEPTH}\n")
        else:
            f.write("\nMCMC: SKIPPED (SVI only)\n")

        # Tellurics
        if ENABLE_TELLURICS:
            f.write(f"\nTelluric correction: ENABLED\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Run configuration saved to {log_path}")


# Default output directory (lazy - will be set by CLI or on first use)
DIR_SAVE = None  # Set by CLI via get_output_dir()

# Opacity loading/saving
OPA_LOAD = True
OPA_SAVE = False

# ==============================================================================
# INFERENCE PARAMETERS
# ==============================================================================

# SVI parameters
SVI_NUM_STEPS = 2000
SVI_LEARNING_RATE = 0.005

# MCMC parameters
MCMC_NUM_WARMUP = 2000
MCMC_NUM_SAMPLES = 2000
MCMC_MAX_TREE_DEPTH = 6

# Parallel chains for better convergence
MCMC_NUM_CHAINS = 4
