"""
Defaults for data preparation and utility scripts.
"""

from .planets_config import PLANET
from .instrument_config import OBSERVING_MODE

# Data prep defaults
DEFAULT_DATA_PLANET = PLANET
DEFAULT_DATA_ARM = OBSERVING_MODE
DEFAULT_USE_MOLECFIT = True
DEFAULT_RAW_DATA_DIR = "input/raw"
DEFAULT_BARYCORR = False
DEFAULT_INTRODUCED_SHIFT = True

# Data loading defaults
DEFAULT_DATA_FORMAT = "auto"

# Binning defaults
DEFAULT_BIN_SIZE = 50

# Doppler shadow fitting defaults
DEFAULT_SHADOW_SCALING = 1.0
DEFAULT_FIT_PARAM_FALLBACK = 1.0

# Wavelength shift defaults
DEFAULT_INTRODUCED_SHIFT_MPS = 0.0

# Misc utility defaults
DEFAULT_BIN_INFO_COUNT = 0
DEFAULT_TRACKER_MAX_USED = 0.0

# SYSREM defaults
DEFAULT_SYSREM_N_SYSTEMATICS_RED = [5, 5]
DEFAULT_SYSREM_N_SYSTEMATICS_OTHER = [5]
