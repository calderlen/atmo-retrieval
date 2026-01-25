"""
Instrument configuration for high-resolution spectrographs.
"""

# ==============================================================================
# ACTIVE INSTRUMENT
# ==============================================================================

INSTRUMENT = "PEPSI"
OBSERVATORY = "lbt"  # For astropy EarthLocation.of_site()

# ==============================================================================
# SPECTRAL SETTINGS
# ==============================================================================

RESOLUTION = 120000  # Spectral resolving power R = λ/Δλ

# Arm/mode configurations: wavelength range (Angstroms) and file identifiers
ARMS = {
    "blue": {
        "range": (4752, 5425),
        "file_prefix": "pepsib",
    },
    "red": {
        "range": (6231, 7427),
        "file_prefix": "pepsir",
    },
    "green": {
        "range": (4760, 6570),  # CD3+CD4 approximate
        "file_prefix": "pepsig",
    },
    "full": {
        "range": (4752, 7427),  # Both arms combined
        "file_prefix": None,  # No single file prefix for combined
    },
}

# Active observing mode
OBSERVING_MODE = "red"

# Convenience accessors for active mode
def get_wavelength_range(mode: str | None = None) -> tuple[float, float]:
    """Get wavelength range for the specified or active mode."""
    mode = mode or OBSERVING_MODE
    return ARMS[mode]["range"]

def get_file_prefix(mode: str | None = None) -> str | None:
    """Get file prefix for the specified or active mode."""
    mode = mode or OBSERVING_MODE
    return ARMS[mode]["file_prefix"]

WAV_MIN, WAV_MAX = get_wavelength_range()

# ==============================================================================
# FITS HEADER KEY MAPPINGS
# ==============================================================================

# Maps logical names to instrument-specific FITS header keys
HEADER_KEYS = {
    "jd": "JD-OBS",          # Mid-exposure Julian Date
    "snr": "SNR",            # Signal-to-noise ratio
    "exptime": "EXPTIME",    # Exposure time
    "airmass": "AIRMASS",    # Airmass
    "radvel": "RADVEL",      # Radial velocity correction
    "obsvel": "OBSVEL",      # Observatory velocity
    "ssbvel": "SSBVEL",      # Solar system barycenter velocity
}

# ==============================================================================
# DATA FILE PATTERNS
# ==============================================================================

def get_data_patterns(
    observation_epoch: str,
    planet_name: str,
    arm: str,
    do_molecfit: bool = True,
    data_dir: str = "input",
) -> list[str]:
    """Get glob patterns for finding PEPSI data files.

    Args:
        observation_epoch: Observation date string (YYYYMMDD)
        planet_name: Planet name
        arm: Spectrograph arm (blue/red)
        do_molecfit: Whether to look for molecfit-corrected files
        data_dir: Base data directory

    Returns:
        List of glob patterns to try in order
    """
    file_prefix = get_file_prefix(arm)
    if file_prefix is None:
        raise ValueError(f"No file prefix defined for arm '{arm}'")

    year = int(observation_epoch[0:4])

    # PEPSI file extensions (order matters - try newest first)
    pepsi_exts = ["nor", "avr"]
    if year >= 2024:
        pepsi_exts.insert(0, "bwl")

    patterns = []
    base_path = f"{data_dir}/{observation_epoch}_{planet_name}"

    if do_molecfit:
        for ext in pepsi_exts:
            patterns.append(
                f"{base_path}/molecfit_weak/SCIENCE_TELLURIC_CORR_{file_prefix}*.dxt.{ext}.fits"
            )
            patterns.append(
                f"{base_path}/**/SCIENCE_TELLURIC_CORR_{file_prefix}*.dxt.{ext}.fits"
            )
    else:
        for ext in pepsi_exts:
            patterns.append(f"{base_path}/{file_prefix}*.dxt.{ext}")
            patterns.append(f"{base_path}/**/{file_prefix}*.dxt.{ext}")

    return patterns

# ==============================================================================
# MOLECFIT SETTINGS
# ==============================================================================

# Column names in FITS files (differ between raw and molecfit-corrected)
FITS_COLUMNS = {
    "molecfit": {
        "wave": "lambda",
        "flux": "flux",
        "error": "error",
        "wave_unit": "micron",  # molecfit outputs in microns
    },
    "raw": {
        "wave": "Arg",
        "flux": "Fun",
        "error": "Var",  # Note: this is variance, needs sqrt
        "wave_unit": "angstrom",
    },
}
