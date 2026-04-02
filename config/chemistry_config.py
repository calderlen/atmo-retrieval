"""
Chemistry and composition model parameters.

These are defaults for composition solvers and chemistry models.
"""

# ==============================================================================
# VMR PRIOR RANGES
# ==============================================================================

# Logarithmic VMR prior bounds for trace species
LOG_VMR_MIN = -12.0  # Minimum log10(VMR)
LOG_VMR_MAX = -2.0   # Maximum log10(VMR)

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

# Solar elemental abundance table (Asplund 2020; log epsilon format)
SOLAR_ABUNDANCE_FILE = "input/abundances/asplund_2020_extended"

# ==============================================================================
# FASTCHEM GRID PARAMETERS
# ==============================================================================

FASTCHEM_N_TEMP = 50
FASTCHEM_N_PRESSURE = 50
FASTCHEM_T_MIN = 500.0
FASTCHEM_T_MAX = 5000.0
FASTCHEM_CACHE_DIR = "input/.fastchem_cache"
FASTCHEM_DATA_DIR = None  # None = use pyfastchem defaults
FASTCHEM_PARAMETER_FILE = None  # Path to FastChem parameters.dat

# Chemistry solver selection
CHEMISTRY_MODEL_DEFAULT = "constant"

# Hybrid FastChem grid settings (NUTS-safe via JAX interpolation)
FASTCHEM_HYBRID_CONTINUUM_SPECIES = ("H-", "e-", "H")
FASTCHEM_HYBRID_N_METALLICITY = 17
FASTCHEM_HYBRID_N_CO_RATIO = 17
FASTCHEM_HYBRID_METALLICITY_RANGE = METALLICITY_RANGE
FASTCHEM_HYBRID_CO_RATIO_RANGE = CO_RATIO_RANGE

# ==============================================================================
# DISEQUILIBRIUM CHEMISTRY
# ==============================================================================

# Eddy diffusion coefficient Kzz [cm^2/s] prior range
LOG_KZZ_RANGE = (6.0, 12.0)

# Quench pressure range [bar]
LOG_QUENCH_P_RANGE = (-6.0, 2.0)
