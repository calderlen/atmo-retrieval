"""
Chemistry and composition model parameters.

These are defaults for composition solvers and chemistry models.
"""

# ==============================================================================
# VMR PRIOR RANGES
# ==============================================================================

# Logarithmic VMR prior bounds for trace species
LOG_VMR_MIN = -15.0  # Minimum log10(VMR)
LOG_VMR_MAX = 0.0    # Maximum log10(VMR)

# ==============================================================================
# BULK COMPOSITION
# ==============================================================================

# H2/He number ratio (solar ~10-11, hot Jupiters often use ~6-7)
H2_HE_RATIO = 6.0

# ==============================================================================
# FREE CHEMISTRY PROFILE PARAMETERIZATION
# ==============================================================================

# Number of nodes for altitude-dependent VMR profiles
N_VMR_NODES = 5

# ==============================================================================
# EQUILIBRIUM CHEMISTRY
# ==============================================================================

# Metallicity [M/H] prior range (log10 relative to solar)
METALLICITY_RANGE = (-2.0, 3.0)

# C/O ratio prior range (solar ~ 0.55)
CO_RATIO_RANGE = (0.1, 2.0)

# ==============================================================================
# FASTCHEM GRID PARAMETERS
# ==============================================================================

FASTCHEM_N_TEMP = 50
FASTCHEM_N_PRESSURE = 50
FASTCHEM_T_MIN = 500.0
FASTCHEM_T_MAX = 5000.0
FASTCHEM_CACHE_DIR = "input/.fastchem_cache"
FASTCHEM_DATA_DIR = None  # None = use pyfastchem defaults

# ==============================================================================
# DISEQUILIBRIUM CHEMISTRY
# ==============================================================================

# Eddy diffusion coefficient Kzz [cm^2/s] prior range
LOG_KZZ_RANGE = (6.0, 12.0)

# Quench pressure range [bar]
LOG_QUENCH_P_RANGE = (-6.0, 2.0)
