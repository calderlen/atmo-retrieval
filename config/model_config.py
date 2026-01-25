"""
Radiative transfer model parameters and spectral grid settings.

Instrument-specific settings (resolution, wavelength ranges) are in instrument.py.
"""

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 100  # Number of atmospheric layers

# Pressure range [bar]
PRESSURE_TOP = 1e-8 
PRESSURE_BTM = 1e2

# Temperature range [K]
TLOW = 1500.0
THIGH = 4500.0

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 250000
WAV_MIN_OFFSET = 100  # Angstroms
WAV_MAX_OFFSET = 100  # Angstroms

# preMODIT parameters
NDIV = 8 

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0 

# ==============================================================================
# TELLURIC LINE MODELING
# ==============================================================================

ENABLE_TELLURICS = True

TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass
