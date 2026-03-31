"""
Defaults for data preparation and utility scripts.
"""

from .planets_config import PLANET
from .instrument_config import OBSERVING_MODE

# Data prep defaults
DEFAULT_DATA_PLANET = PLANET
DEFAULT_DATA_ARM = OBSERVING_MODE
DEFAULT_USE_MOLECFIT = True
DEFAULT_RAW_DATA_DIR = "input/hrs"
DEFAULT_BARYCORR = False
DEFAULT_INTRODUCED_SHIFT = True

# Data loading defaults
# Default to time-series input so the main CLI and phase-binned path work without
# extra flags. Use --data-format spectrum for collapsed retrieval products.
DEFAULT_DATA_FORMAT = "timeseries"

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
DEFAULT_SYSREM_MAX_SYSTEMATICS_RED = [10, 10]
DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER = [10]
DEFAULT_SYSREM_STOP_TOL = 1e-4
