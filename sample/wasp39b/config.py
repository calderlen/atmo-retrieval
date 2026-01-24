"""
WASP-39b Retrieval Configuration
=================================

System parameters, paths, and configuration flags.
"""

import os
from exojax.utils.constants import RJ, Rs, MJ

# Planet-star system parameters
PERIOD_DAY = 4.05528
MP_MEAN, MP_STD = 0.281, 0.032  # [M_J]
RSTAR_MEAN, RSTAR_STD = 0.939, 0.022  # [R_Sun]

# Output directory
DIR_SAVE = "output_wasp39b"
os.makedirs(DIR_SAVE, exist_ok=True)

# Opacity loading/saving flags
OPA_LOAD = True
OPA_SAVE = False

# Atmospheric RT parameters
DIFFMODE = 0
NLAYER = 120
PRESSURE_TOP = 1e-11  # [bar]
PRESSURE_BTM = 1e1    # [bar]
TLOW = 500.0          # [K]
THIGH = 2000.0        # [K]

# Wavenumber grid
N_SPECTRAL_POINTS = 30_000
WAV_MIN_OFFSET = 15  # nm
WAV_MAX_OFFSET = 15  # nm

# preMODIT parameters
NDIV = 6  # stitch blocks

# Cloud parameters
CLOUD_WIDTH = 1.0 / 25.0  # Fixed cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 50.0  # Integrated optical depth

# Database paths
DB_HITEMP = "path_to/.db_HITEMP/"
DB_EXOMOL = "path_to/.db_ExoMol/"

# CIA paths
CIA_PATHS = {
    "H2H2": "path_to/.db_CIA/H2-H2_2011.cia",
    "H2He": "path_to/.db_CIA/H2-He_2011.cia",
}

# Molecular database paths
MOLPATH_HITEMP = {
    "H2O": f"{DB_HITEMP}H2O/",
    "CO": f"{DB_HITEMP}CO/",
    "CO2": f"{DB_HITEMP}CO2/",
}

MOLPATH_EXOMOL = {
    "H2S": f"{DB_EXOMOL}H2S/1H2-32S/AYT2/",
    "SO2": f"{DB_EXOMOL}SO2/32S-16O2/ExoAmes/",
    "SiO": f"{DB_EXOMOL}SiO/28Si-16O/SiOUVenIR/",
}

# Data paths
DATA_DIR = "WASP39b_NIRSpec_data"
WAV_OBS_PATH = f"{DATA_DIR}/wavelength.npy"
RP_MEAN_PATH = f"{DATA_DIR}/wasp39b_nirspec_g395h_rp_mean.npy"
RP_STD_PATH = f"{DATA_DIR}/wasp39b_nirspec_g395h_rp_std.npy"
RESOLUTION_CURVE_PATH = f"{DATA_DIR}/jwst_nirspec_g395h_disp.fits"

# Inference parameters
SVI_NUM_STEPS = 1000
SVI_LEARNING_RATE = 0.005
MCMC_NUM_WARMUP = 1000
MCMC_NUM_SAMPLES = 1000
MCMC_MAX_TREE_DEPTH = 5
