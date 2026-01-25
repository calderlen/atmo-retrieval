"""
Radiative transfer model parameters, spectral grid, and observation settings.
"""

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
# RETRIEVAL MODE
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

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
# TELLURIC LINE MODELING
# ==============================================================================

ENABLE_TELLURICS = True

# Typical telluric parameters (can be free parameters in retrieval)
TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass
