"""Defaults and constants for standalone TESS processing."""

from .paths_config import INPUT_DIR

# ==============================================================================
# TESS FILES / PATHS
# ==============================================================================

TESS_BANDPASS_URL = "https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv"
TESS_BANDPASS_PATH = INPUT_DIR / "bandpasses" / "tess-response-function-v2.0.csv"
TESS_DEFAULT_ECLIPSE_TBL = INPUT_DIR / "lowres_spectra" / "KELT_20_b_3.12080_5244_1.tbl"
TESS_OUTPUT_SUBDIR = "tess_proc"
TESS_TARGET_WAVELENGTH_ANGSTROM = 8000.0

# ==============================================================================
# PHYSICAL CONSTANTS (SI)
# ==============================================================================

R_SUN_M = 6.957e8
AU_M = 1.495978707e11
RJUP_M = 7.1492e7

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZMANN = 1.380649e-23

NUMERIC_EPS = 1.0e-30

# ==============================================================================
# MODEL / PRIOR DEFAULTS
# ==============================================================================

TESS_DEFAULT_IMPACT_PARAM = 0.5
TESS_DEFAULT_GAMMA1 = 0.3
TESS_DEFAULT_GAMMA2 = 0.2

TESS_RADIUS_RATIO_FALLBACK_FRAC = 0.05
TESS_RADIUS_RATIO_SIGMA_MIN = 1.0e-4

# ==============================================================================
# CLI / INFERENCE DEFAULTS
# ==============================================================================

TESS_DEFAULT_TDAY_MIN_K = 500.0
TESS_DEFAULT_TDAY_MAX_K = 5000.0
TESS_DEFAULT_ALBEDO_MIN = 0.0
TESS_DEFAULT_ALBEDO_MAX = 1.0

TESS_DEFAULT_ECLIPSE_MODEL_SIGMA_PPM = 5.0

TESS_DEFAULT_NUM_WARMUP = 1000
TESS_DEFAULT_NUM_SAMPLES = 2000
TESS_DEFAULT_NUM_CHAINS = 1
TESS_DEFAULT_MAX_TREE_DEPTH = 10
TESS_DEFAULT_SEED = 42
