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
