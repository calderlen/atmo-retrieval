"""
Inference parameters for SVI and MCMC sampling.
"""

# ==============================================================================
# SVI PARAMETERS
# ==============================================================================

SVI_NUM_STEPS = 2000
SVI_LEARNING_RATE = 0.001
SVI_LR_DECAY_STEPS = None
SVI_LR_DECAY_RATE = None

# ==============================================================================
# MCMC PARAMETERS
# ==============================================================================

MCMC_NUM_WARMUP = 2000
MCMC_NUM_SAMPLES = 2000
MCMC_MAX_TREE_DEPTH = 10

# Parallel chains
MCMC_NUM_CHAINS = 1
# TODO: if MCMC_NUM_CHAINS = 4 w/o parallel gpus then 4 chains will run sequentially, which is fine for testing but not ideal. see how t correctly make this code run in parallel on GPUs, then change this parameter before a run



# ==============================================================================
# INFERENCE BEHAVIOR DEFAULTS
# ==============================================================================

INIT_TO_MEDIAN_SAMPLES = 100

# Quick mode defaults
QUICK_SVI_STEPS = 100
QUICK_MCMC_WARMUP = 100
QUICK_MCMC_SAMPLES = 100
QUICK_MCMC_CHAINS = 1
