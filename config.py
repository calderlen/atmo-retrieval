"""
KELT-20b Ultra-Hot Jupiter Retrieval Configuration
===================================================

System parameters, paths, and configuration for PEPSI observations.
"""

import os
from exojax.utils.constants import RJ, Rs, MJ

# ==============================================================================
# TARGET SYSTEM PARAMETERS - KELT-20b (Ultra-Hot Jupiter)
# ==============================================================================

# Orbital parameters
PERIOD_DAY = 3.4741095  # Lund et al. 2017
SEMI_MAJOR_AXIS = 0.0542  # AU

# Planet parameters (Lund et al. 2017, updated)
MP_MEAN, MP_STD = 3.382, 0.13  # [M_J]
RP_MEAN, RP_STD = 1.735, 0.07  # [R_J]

# Stellar parameters
RSTAR_MEAN, RSTAR_STD = 1.60, 0.06  # [R_Sun]
MSTAR_MEAN, MSTAR_STD = 1.89, 0.06  # [M_Sun]
TSTAR = 8980  # K (A2V star)

# ==============================================================================
# OBSERVATION PARAMETERS - PEPSI/LBT
# ==============================================================================

# Wavelength range depends on observing mode
# Options: "blue" (383-476 nm), "green" (476-657 nm), "red" (650-907 nm)
OBSERVING_MODE = "red"  # Change based on your data

WAVELENGTH_RANGES = {
    "blue": (383, 476),    # nm
    "green": (476, 657),   # nm
    "red": (650, 907),     # nm
    "full": (383, 907),    # If combining multiple modes
}

WAV_MIN, WAV_MAX = WAVELENGTH_RANGES[OBSERVING_MODE]

# PEPSI resolution (depends on fiber mode)
# 200 μm fiber: R~120,000
# 100 μm fiber: R~250,000
PEPSI_RESOLUTION = 120000  # Adjust based on your setup

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

N_SPECTRAL_POINTS = 100_000  # Higher resolution for PEPSI
WAV_MIN_OFFSET = 10  # nm
WAV_MAX_OFFSET = 10  # nm

# preMODIT parameters
NDIV = 8  # More divisions for higher resolution

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

# Ultra-hot Jupiters may have minimal clouds on dayside
# but could have condensates on nightside
CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0  # Lower than WASP-39b

# ==============================================================================
# DATABASE PATHS
# ==============================================================================

# Molecular databases
DB_HITEMP = os.path.expanduser("~/.db_HITEMP/")
DB_EXOMOL = os.path.expanduser("~/.db_ExoMol/")
DB_KURUCZ = os.path.expanduser("~/.db_kurucz/")  # For atomic lines

# CIA paths
CIA_PATHS = {
    "H2H2": os.path.join(DB_HITEMP, "../.db_CIA/H2-H2_2011.cia"),
    "H2He": os.path.join(DB_HITEMP, "../.db_CIA/H2-He_2011.cia"),
}

# Molecular line lists for ultra-hot Jupiter
MOLPATH_HITEMP = {
    "H2O": f"{DB_HITEMP}H2O/",
    "CO": f"{DB_HITEMP}CO/",
    "OH": f"{DB_HITEMP}OH/",
}

MOLPATH_EXOMOL = {
    "TiO": f"{DB_EXOMOL}TiO/48Ti-16O/Toto/",
    "VO": f"{DB_EXOMOL}VO/51V-16O/VOMYT/",
    "FeH": f"{DB_EXOMOL}FeH/56Fe-1H/MoLLIST/",
    "CaH": f"{DB_EXOMOL}CaH/40Ca-1H/MoLLIST/",
    "CrH": f"{DB_EXOMOL}CrH/52Cr-1H/MoLLIST/",
    "AlO": f"{DB_EXOMOL}AlO/27Al-16O/ATP/",
    # Add more as needed
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

# Enable telluric correction for ground-based observations
ENABLE_TELLURICS = True

# Telluric species (Earth's atmosphere)
TELLURIC_SPECIES = {
    "H2O": f"{DB_HITEMP}H2O/",  # Water vapor (dominant in optical)
    "O2": f"{DB_HITEMP}O2/",    # Oxygen
    # Add more if needed
}

# Typical telluric parameters (can be free parameters in retrieval)
TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass

# ==============================================================================
# DATA PATHS
# ==============================================================================

DATA_DIR = "data/kelt20b_pepsi"
os.makedirs(DATA_DIR, exist_ok=True)

# Adjust these based on your actual data files
TRANSMISSION_DATA = {
    "wavelength": f"{DATA_DIR}/wavelength_transmission.npy",
    "spectrum": f"{DATA_DIR}/spectrum_transmission.npy",
    "uncertainty": f"{DATA_DIR}/uncertainty_transmission.npy",
}

EMISSION_DATA = {
    "wavelength": f"{DATA_DIR}/wavelength_emission.npy",
    "spectrum": f"{DATA_DIR}/spectrum_emission.npy",
    "uncertainty": f"{DATA_DIR}/uncertainty_emission.npy",
}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

DIR_SAVE = "output_kelt20b"
os.makedirs(DIR_SAVE, exist_ok=True)

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

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

# Options: "transmission", "emission", "combined"
RETRIEVAL_MODE = "transmission"  # Change based on your analysis

# For combined retrieval
COMBINE_TRANSMISSION_EMISSION = False
