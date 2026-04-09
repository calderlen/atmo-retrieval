"""
Radiative transfer model parameters and spectral grid settings.

Instrument-specific settings (resolution, wavelength ranges) are in instrument.py.
"""

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

# Default P-T profile
PT_PROFILE_DEFAULT = "guillot"  # Options: "guillot", "isothermal", "free`"

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 20 # number of atmospheric layers

# Pressure range [bar]
PRESSURE_TOP = 1e-8 
PRESSURE_BTM = 1e0

# Temperature range [K]
T_LOW = 1500.0
T_HIGH = 4500.0

# Guillot profile defaults and bounds
TINT_FIXED = 100.0
LOG_KAPPA_IR_BOUNDS = (-4.0, 0.0)
LOG_GAMMA_BOUNDS = (-2.0, 2.0)

# Model/inference defaults
DEFAULT_KP = 169.0  # planet radial velocity semi-amplitude [km/s]
DEFAULT_KP_ERR = 20.0 
DEFAULT_RV_ABS = 0.0 # absolute RV shift [km/s]
DEFAULT_RV_ABS_ERR = 1.0
DEFAULT_TSTAR = 6000.0 # stellar temperature [K] 
DEFAULT_RP_ERR = 0.1 # planet radius error (relative to Rp/Rs)
DEFAULT_MP_ERR = 0.1 # planet mass error (relative to Mp/Ms)
DEFAULT_RSTAR_ERR = 0.1 # stellar radius error (relative to Rstar)

# Posterior reconstruction defaults
DEFAULT_POSTERIOR_RP = 1.5 # Maximum Rp/Rs for posterior reconstruction
DEFAULT_POSTERIOR_MP = 1.0 # Maximum Mp/Ms for posterior reconstruction

# Pipeline behavior defaults
SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT = True # Whether to subtract per-exposure mean from model and data before computing likelihood. Should be True for CCF-like likelihoods, but can be False for full-spectrum Gaussian likelihoods.
APPLY_SYSREM_DEFAULT = True # Whether to apply SysRem-like filtering to model and data before computing likelihood. Should be True for CCF-like likelihoods, but can be False for full-spectrum Gaussian likelihoods. Requires U and V from data preprocessing.

# Phase modeling defaults
DEFAULT_PHASE_MODE = "global" 

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 500000
#N_SPECTRAL_POINTS = 50000
WAV_MIN_OFFSET = 100  # Angstroms
WAV_MAX_OFFSET = 100  # Angstroms

# preMODIT parameters
# Line-wing truncation (relative to grid spacing). Set to None to use the default.
PREMODIT_CUTWING = None

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
