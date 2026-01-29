"""
Radiative transfer model parameters and spectral grid settings.

Instrument-specific settings (resolution, wavelength ranges) are in instrument.py.
"""

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 100  # Number of atmospheric layers

# Pressure range [bar]
PRESSURE_TOP = 1e-8 
PRESSURE_BTM = 1e2

# Temperature range [K]
T_LOW = 1500.0
T_HIGH = 4500.0

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 250000
WAV_MIN_OFFSET = 100  # Angstroms
WAV_MAX_OFFSET = 100  # Angstroms

# preMODIT parameters
# NDIV controls wavenumber stitching (OpaPremodit nstitch) to reduce device memory.
NDIV = 250
# Line-wing truncation (relative to grid spacing). Set to None to auto-scale as 1/(2*NDIV).
PREMODIT_CUTWING = None

# ==============================================================================
# INFERENCE GRID STITCHING (OPTIONAL)
# ==============================================================================
# Chunk the forward model across the wavenumber grid to reduce GPU memory during
# inference. Enable only if full-grid inference is too large.
ENABLE_INFERENCE_STITCHING = True
# Target number of grid points in each chunk core (excludes guard points).
# Ignored if INFERENCE_STITCH_NCHUNKS is set.
INFERENCE_STITCH_CHUNK_POINTS = 50000
# Explicit number of chunks. If set, overrides INFERENCE_STITCH_CHUNK_POINTS.
INFERENCE_STITCH_NCHUNKS = None
# Guard band in km/s to mitigate edge effects from broadening + Doppler shifts.
# If INFERENCE_STITCH_GUARD_POINTS is set, this is ignored.
INFERENCE_STITCH_GUARD_KMS = 300.0
# Explicit guard size in grid points (overrides INFERENCE_STITCH_GUARD_KMS).
INFERENCE_STITCH_GUARD_POINTS = None
# Minimum guard size in grid points (applies when guard is computed from km/s).
INFERENCE_STITCH_MIN_GUARD_POINTS = 256

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0 

# ==============================================================================
# TELLURIC LINE MODELING
# ==============================================================================

ENABLE_TELLURICS = True

TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass
