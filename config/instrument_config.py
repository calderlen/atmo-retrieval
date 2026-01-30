"""
Instrument configuration for high-resolution spectrographs.

Structure: INSTRUMENTS[observatory][instrument] contains instrument config with:
- resolution: Spectral resolving power R = λ/Δλ
- header_keys: FITS header key mappings
- fits_columns: Column names for raw/molecfit data
- get_data_patterns: Callable to get file glob patterns
- modes: Dict of observing modes, each with:
  - range: (wav_min, wav_max) in Angstroms
  - file_prefix: Prefix for data files
"""

from typing import Callable

# ==============================================================================
# ACTIVE SELECTION (global state, can be modified at runtime)
# ==============================================================================

OBSERVATORY = "lbt"
INSTRUMENT = "PEPSI"
OBSERVING_MODE = "full"
RESOLUTION_MODE = "uhr"  # Options: "standard" (R=50k), "hr" (R=120k), "uhr" (R=270k)


# ==============================================================================
# PEPSI-SPECIFIC DATA PATTERNS FUNCTION
# ==============================================================================

def _pepsi_data_patterns(
    observation_epoch: str,
    planet_name: str,
    mode: str,
    file_prefix: str,
    do_molecfit: bool = True,
    data_dir: str = "input",
) -> list[str]:
    """Get glob patterns for finding PEPSI data files.

    Args:
        observation_epoch: Observation date string (YYYYMMDD)
        planet_name: Planet name
        mode: Spectrograph arm (blue/red/green/full)
        file_prefix: File prefix for this mode
        do_molecfit: Whether to look for molecfit-corrected files
        data_dir: Base data directory

    Returns:
        List of glob patterns to try in order
    """
    if file_prefix is None:
        raise ValueError(f"No file prefix defined for mode '{mode}'")

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
# INSTRUMENT DATABASE
# ==============================================================================

# Common PEPSI header keys (shared across all modes)
_PEPSI_HEADER_KEYS = {
    "jd": "JD-OBS",          # Mid-exposure Julian Date
    "snr": "SNR",            # Signal-to-noise ratio
    "exptime": "EXPTIME",    # Exposure time
    "airmass": "AIRMASS",    # Airmass
    "radvel": "RADVEL",      # Radial velocity correction
    "obsvel": "OBSVEL",      # Observatory velocity
    "ssbvel": "SSBVEL",      # Solar system barycenter velocity
}

# Common PEPSI FITS columns (shared across all modes)
_PEPSI_FITS_COLUMNS = {
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


# TODO: maybe hardcoding the telluric regions is a bad idea
# ==============================================================================
# TELLURIC REGIONS (wavelength ranges in Angstroms)
# ==============================================================================

# From Lenhart et al. 2026 Table 2 (PEPSI observations)
TELLURIC_REGIONS: dict[str, dict[str, list[tuple[float, float]]]] = {
    "red": {
        # Regions with >1% line depth in adjacent telluric lines
        "telluric": [
            (6278, 6328),   # O2 B-band wing
            (6459, 6527),   # H2O
            (6867, 6867.5), # O2 B-band edge
            (6930, 7168),   # H2O + O2 A-band
            (7312, 7500),   # H2O
        ],
        # Deep absorption - mask if molecfit was used (set flux=0, err=1)
        "deep_mask": [
            (6867.5, 6930),  # O2 B-band core
            (7168, 7312),    # Deep H2O
        ],
    },
    "blue": {
        # Blue arm lacks significant tellurics (Smette et al. 2015)
        "telluric": [],
        "deep_mask": [],
    },
}


INSTRUMENTS: dict[str, dict[str, dict]] = {
    "lbt": {
        "PEPSI": {
            "resolution": 270000,  # Default to UHR mode
            "resolution_modes": {
                "standard": 50000,   # 300 µm fiber
                "hr": 120000,        # 200 µm fiber (High Resolution)
                "uhr": 270000,       # 100 µm fiber (Ultra-High Resolution)
            },
            "header_keys": _PEPSI_HEADER_KEYS,
            "fits_columns": _PEPSI_FITS_COLUMNS,
            "get_data_patterns": _pepsi_data_patterns,
            "modes": {
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
                    "file_prefix": None,    # No single file prefix for combined
                },
            },
        },
    },
}


# ==============================================================================
# HELPER FUNCTIONS (use global state by default)
# ==============================================================================

def get_instrument_config(
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict:
    """Get instrument configuration dict.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)

    Returns:
        Instrument config dict with 'resolution', 'modes', 'get_data_patterns'
    """
    obs = observatory or OBSERVATORY
    inst = instrument or INSTRUMENT
    return INSTRUMENTS[obs][inst]


def get_mode_config(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> dict:
    """Get observing mode configuration dict.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)
        mode: Observing mode (default: active OBSERVING_MODE)

    Returns:
        Mode config dict with 'range', 'file_prefix'
    """
    obs = observatory or OBSERVATORY
    inst = instrument or INSTRUMENT
    m = mode or OBSERVING_MODE
    return INSTRUMENTS[obs][inst]["modes"][m]


def get_resolution(
    observatory: str | None = None,
    instrument: str | None = None,
    resolution_mode: str | None = None,
) -> int:
    """Get spectral resolving power R = λ/Δλ for instrument.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)
        resolution_mode: Resolution mode (default: active RESOLUTION_MODE).
            For PEPSI: "standard" (R=50k), "hr" (R=120k), "uhr" (R=270k)

    Returns:
        Spectral resolution
    """
    config = get_instrument_config(observatory, instrument)
    res_mode = resolution_mode or RESOLUTION_MODE

    # If instrument has resolution_modes, use that; otherwise fall back to default
    if "resolution_modes" in config and res_mode in config["resolution_modes"]:
        return config["resolution_modes"][res_mode]
    return config["resolution"]


def get_wavelength_range(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> tuple[float, float]:
    """Get wavelength range (Angstroms) for observing mode.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)
        mode: Observing mode (default: active OBSERVING_MODE)

    Returns:
        (wav_min, wav_max) tuple in Angstroms
    """
    return get_mode_config(observatory, instrument, mode)["range"]


def get_file_prefix(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> str | None:
    """Get file prefix for observing mode.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)
        mode: Observing mode (default: active OBSERVING_MODE)

    Returns:
        File prefix string or None
    """
    return get_mode_config(observatory, instrument, mode)["file_prefix"]


def get_header_keys(
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict[str, str]:
    """Get FITS header key mappings for instrument.

    Args:
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)

    Returns:
        Dict mapping logical names to FITS header keys
    """
    return get_instrument_config(observatory, instrument)["header_keys"]


def get_fits_columns(
    molecfit: bool = True,
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict[str, str]:
    """Get FITS column names for data files.

    Args:
        molecfit: If True, return molecfit column names; else raw
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)

    Returns:
        Dict with 'wave', 'flux', 'error', 'wave_unit' keys
    """
    cols = get_instrument_config(observatory, instrument)["fits_columns"]
    return cols["molecfit" if molecfit else "raw"]


def get_data_patterns(
    observation_epoch: str,
    planet_name: str,
    mode: str | None = None,
    do_molecfit: bool = True,
    data_dir: str = "input",
    observatory: str | None = None,
    instrument: str | None = None,
) -> list[str]:
    """Get glob patterns for finding data files.

    Args:
        observation_epoch: Observation date string (YYYYMMDD)
        planet_name: Planet name
        mode: Observing mode (default: active OBSERVING_MODE)
        do_molecfit: Whether to look for molecfit-corrected files
        data_dir: Base data directory
        observatory: Observatory name (default: active OBSERVATORY)
        instrument: Instrument name (default: active INSTRUMENT)

    Returns:
        List of glob patterns to try in order
    """
    m = mode or OBSERVING_MODE
    inst_config = get_instrument_config(observatory, instrument)
    mode_config = get_mode_config(observatory, instrument, m)

    patterns_fn = inst_config["get_data_patterns"]
    return patterns_fn(
        observation_epoch=observation_epoch,
        planet_name=planet_name,
        mode=m,
        file_prefix=mode_config["file_prefix"],
        do_molecfit=do_molecfit,
        data_dir=data_dir,
    )
