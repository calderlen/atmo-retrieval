"""
Ultra-Hot Jupiter Atmospheric Retrieval Configuration
======================================================

Configuration is split into logical modules:
- planets: System parameters from published literature
- instrument: Spectrograph and observatory settings
- model: RT parameters, spectral grid settings
- paths: Database paths, data paths, output configuration
- inference: SVI and MCMC sampling parameters
"""

# Re-export everything for convenience
from .planets_config import (
    PLANET,
    EPHEMERIS,
    DEFAULT_LAMBDA_ANGLE,
    DEFAULT_PERIOD_DAY,
    PLANETS,
    PHASE_BINS,
    get_params,
    list_planets,
    list_ephemerides,
)

from .instrument_config import (
    # Global state (can be modified at runtime)
    INSTRUMENT,
    OBSERVATORY,
    OBSERVING_MODE,
    # Main config database
    INSTRUMENTS,
    TELLURIC_REGIONS,
    # Helper functions (use global state by default)
    get_instrument_config,
    get_mode_config,
    get_resolution,
    get_wavelength_range,
    get_file_prefix,
    get_header_keys,
    get_fits_columns,
    get_data_patterns,
)

from .model_config import (
    RETRIEVAL_MODE,
    PT_PROFILE_DEFAULT,
    DIFFMODE,
    NLAYER,
    PRESSURE_TOP,
    PRESSURE_BTM,
    T_LOW,
    T_HIGH,
    TINT_FIXED,
    LOG_KAPPA_IR_BOUNDS,
    LOG_GAMMA_BOUNDS,
    DEFAULT_KP,
    DEFAULT_KP_ERR,
    DEFAULT_RV_ABS,
    DEFAULT_RV_ABS_ERR,
    DEFAULT_TSTAR,
    DEFAULT_RP_ERR,
    DEFAULT_MP_ERR,
    DEFAULT_RSTAR_ERR,
    STITCH_MIN_GUARD_POINTS,
    STITCH_VSINI_MAX,
    STITCH_VRMAX,
    DEFAULT_RP_MEAN,
    DEFAULT_KP_MEAN,
    DEFAULT_KP_ERR_MEAN,
    DEFAULT_RV_ABS_MEAN,
    DEFAULT_RV_ABS_ERR_MEAN,
    DEFAULT_RV_GUARD_EXTRA,
    DEFAULT_SIGMA_V_FACTOR,
    DEFAULT_POSTERIOR_RP,
    DEFAULT_POSTERIOR_MP,
    SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT,
    APPLY_SYSREM_DEFAULT,
    DEFAULT_PHASE_MODE,
    N_SPECTRAL_POINTS,
    WAV_MIN_OFFSET,
    WAV_MAX_OFFSET,
    NDIV,
    PREMODIT_CUTWING,
    ENABLE_INFERENCE_STITCHING,
    INFERENCE_STITCH_CHUNK_POINTS,
    INFERENCE_STITCH_NCHUNKS,
    INFERENCE_STITCH_GUARD_KMS,
    INFERENCE_STITCH_GUARD_POINTS,
    INFERENCE_STITCH_MIN_GUARD_POINTS,
    CLOUD_WIDTH,
    CLOUD_INTEGRATED_TAU,
    ENABLE_TELLURICS,
    TELLURIC_PWV,
    TELLURIC_AIRMASS,
)

from .paths_config import (
    PROJECT_ROOT,
    INPUT_DIR,
    DB_HITEMP,
    DB_EXOMOL,
    DB_EXOATOM,
    DB_KURUCZ,
    DB_VALD,
    DB_CIA,
    CIA_PATHS,
    MOLPATH_HITEMP,
    MOLPATH_EXOMOL,
    ATOMIC_SPECIES,
    DEFAULT_SPECIES,
    USE_DEFAULT_SPECIES,
    get_data_dir,
    get_transmission_paths,
    get_emission_paths,
    DATA_DIR,
    TRANSMISSION_DATA,
    EMISSION_DATA,
    get_output_dir,
    create_timestamped_dir,
    DEFAULT_PHASE_BINNED_OUTPUT_DIR,
    DIR_SAVE,
    OPA_LOAD,
    OPA_SAVE,
    USE_KURUCZ,
    USE_VALD,
)

from .inference_config import (
    SVI_NUM_STEPS,
    SVI_LEARNING_RATE,
    MCMC_NUM_WARMUP,
    MCMC_NUM_SAMPLES,
    MCMC_MAX_TREE_DEPTH,
    MCMC_NUM_CHAINS,
    INIT_TO_MEDIAN_SAMPLES,
    QUICK_SVI_STEPS,
    QUICK_MCMC_WARMUP,
    QUICK_MCMC_SAMPLES,
    QUICK_MCMC_CHAINS,
)

from .chemistry_config import (
    LOG_VMR_MIN,
    LOG_VMR_MAX,
    H2_HE_RATIO,
    N_VMR_NODES,
    METALLICITY_RANGE,
    CO_RATIO_RANGE,
    LOG_KZZ_RANGE,
    LOG_QUENCH_P_RANGE,
)

from .data_config import (
    DEFAULT_DATA_PLANET,
    DEFAULT_DATA_ARM,
    DEFAULT_USE_MOLECFIT,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_BARYCORR,
    DEFAULT_INTRODUCED_SHIFT,
    DEFAULT_DATA_FORMAT,
    DEFAULT_BIN_SIZE,
    DEFAULT_SHADOW_SCALING,
    DEFAULT_FIT_PARAM_FALLBACK,
    DEFAULT_INTRODUCED_SHIFT_MPS,
    DEFAULT_SYSREM_N_SYSTEMATICS_RED,
    DEFAULT_SYSREM_N_SYSTEMATICS_OTHER,
    DEFAULT_BIN_INFO_COUNT,
    DEFAULT_TRACKER_MAX_USED,
)

from .tellurics_config import (
    TELLURIC_SPECIES_DEFAULT,
    TELLURIC_N_GRID,
    TELLURIC_T_RANGE,
    TELLURIC_MARGIN_CM1,
    TELLURIC_VRMAX,
)


def save_run_config(
    output_dir: str,
    mode: str,
    pt_profile: str,
    skip_svi: bool,
    svi_only: bool,
    seed: int,
) -> None:
    """Save run configuration to log file.

    Args:
        output_dir: Directory to save config log
        mode: Retrieval mode (transmission/emission)
        pt_profile: P-T profile type
        skip_svi: Whether SVI was skipped
        svi_only: Whether only SVI was run
        seed: Random seed
    """
    import os
    from datetime import datetime
    import platform
    import jax

    log_path = os.path.join(output_dir, "run_config.log")

    params = get_params()

    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RETRIEVAL RUN CONFIGURATION\n")
        f.write("=" * 70 + "\n\n")

        # Timestamp
        f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {seed}\n\n")

        # System info
        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"JAX version: {jax.__version__}\n")
        f.write(f"JAX backend: {jax.default_backend()}\n")
        f.write(f"JAX devices: {jax.devices()}\n\n")

        # Target info
        f.write("TARGET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Planet: {PLANET}\n")
        f.write(f"Ephemeris: {EPHEMERIS}\n")
        f.write(f"Period: {params['period']}\n")
        f.write(f"R_p: {params['R_p']}\n")
        f.write(f"M_p: {params['M_p']}\n")
        f.write(f"R_star: {params['R_star']}\n")
        f.write(f"T_star: {params['T_star']}\n\n")

        # Retrieval mode
        f.write("RETRIEVAL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"P-T profile: {pt_profile}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        # Wavelength/spectral setup
        f.write("SPECTRAL SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Observatory: {OBSERVATORY}\n")
        f.write(f"Instrument: {INSTRUMENT}\n")
        f.write(f"Observing mode: {OBSERVING_MODE}\n")
        wav_min, wav_max = get_wavelength_range()
        f.write(f"Wavelength range: {wav_min} - {wav_max} Angstroms\n")
        f.write(f"Spectral points: {N_SPECTRAL_POINTS}\n")
        f.write(f"Resolution: R = {get_resolution():,}\n\n")

        # preMODIT stitching
        f.write("preMODIT STITCHING\n")
        f.write("-" * 70 + "\n")
        f.write(f"NDIV (nstitch): {NDIV}\n")
        f.write(f"Cutwing: {PREMODIT_CUTWING}\n\n")

        # Inference stitching
        f.write("INFERENCE GRID STITCHING\n")
        f.write("-" * 70 + "\n")
        f.write(f"Enabled: {ENABLE_INFERENCE_STITCHING}\n")
        f.write(f"Chunk points: {INFERENCE_STITCH_CHUNK_POINTS}\n")
        f.write(f"Chunks (override): {INFERENCE_STITCH_NCHUNKS}\n")
        f.write(f"Guard (km/s): {INFERENCE_STITCH_GUARD_KMS}\n")
        f.write(f"Guard points (override): {INFERENCE_STITCH_GUARD_POINTS}\n")
        f.write(f"Min guard points: {INFERENCE_STITCH_MIN_GUARD_POINTS}\n\n")

        # Atmospheric setup
        f.write("ATMOSPHERIC SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Layers: {NLAYER}\n")
        f.write(f"Pressure range: {PRESSURE_TOP:.2e} - {PRESSURE_BTM:.2e} bar\n")
        f.write(f"Temperature range: {T_LOW} - {T_HIGH} K\n")
        f.write(f"Cloud width: {CLOUD_WIDTH}\n")
        f.write(f"Cloud integrated tau: {CLOUD_INTEGRATED_TAU}\n\n")

        # Molecules and atoms
        f.write("OPACITY SOURCES\n")
        f.write("-" * 70 + "\n")
        f.write("Molecules (HITEMP):\n")
        for mol in MOLPATH_HITEMP.keys():
            f.write(f"  - {mol}\n")
        f.write("Molecules (ExoMol):\n")
        for mol in MOLPATH_EXOMOL.keys():
            f.write(f"  - {mol}\n")
        f.write("Atomic species:\n")
        for atom in ATOMIC_SPECIES.keys():
            f.write(f"  - {atom}\n")
        f.write("\nCIA sources: H2-H2, H2-He\n")
        f.write(f"Opacity loading: {OPA_LOAD}\n")
        f.write(f"Opacity saving: {OPA_SAVE}\n\n")

        # Inference parameters
        f.write("INFERENCE PARAMETERS\n")
        f.write("-" * 70 + "\n")
        if not skip_svi:
            f.write(f"SVI steps: {SVI_NUM_STEPS:,}\n")
            f.write(f"SVI learning rate: {SVI_LEARNING_RATE}\n")
        else:
            f.write("SVI: SKIPPED\n")

        if not svi_only:
            f.write(f"\nMCMC warmup: {MCMC_NUM_WARMUP:,}\n")
            f.write(f"MCMC samples: {MCMC_NUM_SAMPLES:,}\n")
            f.write(f"MCMC chains: {MCMC_NUM_CHAINS}\n")
            f.write(f"MCMC max tree depth: {MCMC_MAX_TREE_DEPTH}\n")
        else:
            f.write("\nMCMC: SKIPPED (SVI only)\n")

        # Tellurics
        if ENABLE_TELLURICS:
            f.write("\nTelluric correction: ENABLED\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Run configuration saved to {log_path}")
