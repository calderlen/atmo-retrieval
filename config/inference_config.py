"""
Inference parameters for SVI and MCMC sampling.
"""

# ==============================================================================
# SVI PARAMETERS
# ==============================================================================

SVI_NUM_STEPS = 2000
SVI_LEARNING_RATE = 0.005

# ==============================================================================
# MCMC PARAMETERS
# ==============================================================================

MCMC_NUM_WARMUP = 2000
MCMC_NUM_SAMPLES = 2000
MCMC_MAX_TREE_DEPTH = 10

# Parallel chains
MCMC_NUM_CHAINS = 4

# ==============================================================================
# INFERENCE BEHAVIOR DEFAULTS
# ==============================================================================

INIT_TO_MEDIAN_SAMPLES = 100

# Quick mode defaults
QUICK_SVI_STEPS = 100
QUICK_MCMC_WARMUP = 100
QUICK_MCMC_SAMPLES = 100
QUICK_MCMC_CHAINS = 1
