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
NLAYER = 10 # number of atmospheric layers (runtime profiles in config/runtime_profiles.py override)

# Pressure range [bar]
PRESSURE_TOP = 1e-8 
PRESSURE_BTM = 1e0

# Temperature range [K]
# Sets PreModit auto_trange, ART layer T clipping, and FastChem T clipping.
# [1500, 5500] gives a PreModit robust range of 1451.74 - 5825.62 K (dE=875, Tref, Twt
# chosen by the LUT), covering Guillot upper-atmosphere draws (observed up to ~5472 K
# under the current priors) while only increasing the LBD+xsmatrix scratch tensor by
# ~14% over the historical [1500, 4500] setting. Widening the cold edge below 1500 K
# requires a smaller dE and runs past the 10 GB GPU budget.
T_LOW = 1500.0
T_HIGH = 5500.0

# Guillot profile defaults and bounds
TINT_FIXED = 100.0
# LOG_KAPPA_IR_BOUNDS: log10(kappa_IR [cm^2/g]). kappa_IR is the Rosseland-mean
# IR opacity of the atmosphere. Hot-Jupiter retrieval literature (Guillot 2010,
# Line et al. 2013, Molliere et al. 2015) places this in 1e-3 - 1e-1 cm^2/g for
# solar-composition atmospheres. The previous (-4, 0) range extended four orders
# of magnitude wider than physical on both ends, contributing to the
# upper-atmosphere temperature runaway (tau = kappa_IR * P / g).
LOG_KAPPA_IR_BOUNDS = (-3.0, -1.0)
# LOG_GAMMA_BOUNDS: log10(gamma) where gamma = kappa_V / kappa_IR. Physically
# plausible hot-Jupiter values span roughly 0.1 - 3 (Guillot 2010 Fig. 4,
# Fortney et al. 2008, Line+ 2013), with gamma > 10 indicating an extreme
# stratospheric absorber. The previous (0, 2) range allowed gamma up to 100,
# which at Tirr = 5500 K drives Guillot's top-of-atmosphere T beyond 14,000 K
# (T_top^4 ~ (3/4) Tirr^4 gamma/sqrt(3)) - well outside PreModit's robust range
# and FastChem's tabulated grid, producing NaN cross sections / VMRs and a
# NaN logL. (-1, 1) covers the physical range while keeping the bulk of prior
# mass inside [T_LOW, T_HIGH]; the clip in _sample_atmosphere_state still
# catches the gamma > 3 tail where Guillot would overshoot.
LOG_GAMMA_BOUNDS = (-1.0, 1.0)

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
