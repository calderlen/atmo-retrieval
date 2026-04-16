"""
Ultra-Hot Jupiter Atmospheric Retrieval Configuration
======================================================

Configuration is split into logical modules:
- planets: System parameters from published literature
- instrument: Spectrograph and observatory settings
- model: RT parameters, spectral grid settings
- paths: Database paths, data paths, output configuration
- inference: SVI and MCMC sampling parameters
- photometry: bandpass and broadband-observation defaults
"""

import os
from datetime import datetime
import platform
import jax

from . import chemistry_config as _chemistry_config
from . import data_config as _data_config
from . import inference_config as _inference_config
from . import instrument_config as _instrument_config
from . import model_config as _model_config
from . import numerics_config as _numerics_config
from . import paths_config as _paths_config
from . import photometry_config as _photometry_config
from . import planets_config as _planets_config
from . import tellurics_config as _tellurics_config

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
    RESOLUTION_MODE,
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
    DEFAULT_POSTERIOR_RP,
    DEFAULT_POSTERIOR_MP,
    SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT,
    APPLY_SYSREM_DEFAULT,
    DEFAULT_PHASE_MODE,
    N_SPECTRAL_POINTS,
    WAV_MIN_OFFSET,
    WAV_MAX_OFFSET,
    PREMODIT_CUTWING,
    CLOUD_WIDTH,
    CLOUD_INTEGRATED_TAU,
    ENABLE_TELLURICS,
    TELLURIC_PWV,
    TELLURIC_AIRMASS,
)

from .paths_config import (
    PROJECT_ROOT,
    INPUT_DIR,
    REFERENCE_DIR,
    REFERENCE_BANDPASS_DIR,
    REFERENCE_ABUNDANCE_DIR,
    CACHE_DIR,
    OPA_CACHE_DIR,
    DB_ROOT_DIR,
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
    get_raw_hrs_dir,
    get_data_dir,
    get_lowres_dir,
    get_phot_dir,
    get_transmission_paths,
    get_emission_paths,
    PHOENIX_CACHE_DIR,
    DATA_DIR,
    RAW_HRS_DIR,
    LOWRES_DIR,
    PHOT_DIR,
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
    SVI_LR_DECAY_STEPS,
    SVI_LR_DECAY_RATE,
    MCMC_NUM_WARMUP,
    MCMC_NUM_SAMPLES,
    MCMC_MAX_TREE_DEPTH,
    MCMC_NUM_CHAINS,
    MCMC_CHAIN_METHOD,
    MCMC_REQUIRE_GPU_PER_CHAIN,
    INIT_TO_MEDIAN_SAMPLES,
    QUICK_SVI_STEPS,
    QUICK_MCMC_WARMUP,
    QUICK_MCMC_SAMPLES,
    QUICK_MCMC_CHAINS,
)

from .chemistry_config import (
    CHEMISTRY_MODEL_DEFAULT,
    LOG_VMR_MIN,
    LOG_VMR_MAX,
    H2_HE_RATIO,
    N_VMR_NODES,
    METALLICITY_RANGE,
    CO_RATIO_RANGE,
    SOLAR_ABUNDANCE_FILE,
    FASTCHEM_N_TEMP,
    FASTCHEM_N_PRESSURE,
    FASTCHEM_T_MIN,
    FASTCHEM_T_MAX,
    FASTCHEM_CACHE_DIR,
    FASTCHEM_DATA_DIR,
    FASTCHEM_PARAMETER_FILE,
    FASTCHEM_HYBRID_CONTINUUM_SPECIES,
    FASTCHEM_HYBRID_N_METALLICITY,
    FASTCHEM_HYBRID_N_CO_RATIO,
    FASTCHEM_HYBRID_METALLICITY_RANGE,
    FASTCHEM_HYBRID_CO_RATIO_RANGE,
    LOG_KZZ_RANGE,
    LOG_QUENCH_P_RANGE,
)

from .numerics_config import (
    F32_EPS,
    F32_GRAVITY_FLOOR,
    F32_LENGTHSCALE_FLOOR,
    F32_FLOOR_RECIP,
    F32_FLOOR_RECIPSQ,
    F32_STDDEV_FLOOR,
    F64_FLOOR,
    TRACE_SPECIES_FLOOR,
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
    DEFAULT_SYSREM_MAX_SYSTEMATICS_RED,
    DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER,
    DEFAULT_SYSREM_STOP_TOL,
    DEFAULT_BIN_INFO_COUNT,
    DEFAULT_TRACKER_MAX_USED,
)

from .photometry_config import (
    TESS_BANDPASS_URL,
    TESS_BANDPASS_PATH,
    AU_M,
)

from .tellurics_config import (
    TELLURIC_SPECIES_DEFAULT,
    TELLURIC_N_GRID,
    TELLURIC_T_RANGE,
    TELLURIC_MARGIN_CM1,
    TELLURIC_VRMAX,
)

from .runtime_profiles import (
    CONFIG_PROFILE_ENVVAR,
    CONFIG_PROFILES,
    DEFAULT_RUNTIME_PROFILE,
)


_RUNTIME_CONFIG_MODULES = (
    _planets_config,
    _instrument_config,
    _model_config,
    _paths_config,
    _inference_config,
    _chemistry_config,
    _numerics_config,
    _data_config,
    _photometry_config,
    _tellurics_config,
)

_active_runtime_profile = DEFAULT_RUNTIME_PROFILE


def set_runtime_config(name: str, value) -> None:
    """Update a config variable across the package and source modules."""
    globals()[name] = value
    for module in _RUNTIME_CONFIG_MODULES:
        if hasattr(module, name):
            setattr(module, name, value)


def list_runtime_profiles() -> tuple[str, ...]:
    """Return available named runtime profiles."""
    return tuple(CONFIG_PROFILES.keys())


def get_runtime_profile_name() -> str:
    """Return the currently active runtime profile name."""
    return _active_runtime_profile


def get_runtime_profile(profile_name: str | None = None) -> dict:
    """Return the profile definition for the active or requested profile."""
    normalized = _normalize_runtime_profile_name(profile_name if profile_name is not None else _active_runtime_profile)
    return CONFIG_PROFILES[normalized]


def _normalize_runtime_profile_name(profile_name: str) -> str:
    normalized = str(profile_name).strip().lower()
    return normalized


def apply_runtime_profile(profile_name: str) -> str:
    """Apply a named runtime profile across config modules."""
    global _active_runtime_profile

    normalized = _normalize_runtime_profile_name(profile_name)
    profile = CONFIG_PROFILES[normalized]
    for name, value in profile["overrides"].items():
        set_runtime_config(name, value)
    _active_runtime_profile = normalized
    return normalized


apply_runtime_profile(os.environ.get(CONFIG_PROFILE_ENVVAR) or DEFAULT_RUNTIME_PROFILE)


def save_run_config(
    output_dir: str,
    mode: str,
    pt_profile: str,
    skip_svi: bool,
    svi_only: bool,
    seed: int,
    chemistry_model: str | None = None,
    epoch: str | None = None,
    phoenix_spectrum_path: str | None = None,
    phoenix_cache_dir: str | None = None,
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
        if epoch is not None:
            f.write(f"Epoch: {epoch}\n")
        f.write(f"Period: {params['period']}\n")
        f.write(f"R_p: {params['R_p']}\n")
        f.write(f"M_p: {params['M_p']}\n")
        f.write(f"R_star: {params['R_star']}\n")
        f.write(f"T_star: {params['T_star']}\n")
        f.write(f"Systemic velocity (fixed): {params.get('RV_abs')}\n\n")

        # Retrieval mode
        f.write("RETRIEVAL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Config profile: {get_runtime_profile_name()}\n")
        f.write(f"P-T profile: {pt_profile}\n")
        if chemistry_model is not None:
            f.write(f"Chemistry model: {chemistry_model}\n")
        if phoenix_spectrum_path is not None:
            f.write(f"PHOENIX spectrum: {phoenix_spectrum_path}\n")
        if phoenix_cache_dir is not None:
            f.write(f"PHOENIX cache dir: {phoenix_cache_dir}\n")
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
        f.write(f"Resolution mode: {RESOLUTION_MODE}\n")
        f.write(f"Resolution: R = {get_resolution():,}\n\n")

        # preMODIT grid
        f.write("PREMODIT GRID\n")
        f.write("-" * 70 + "\n")
        f.write(f"Cutwing: {PREMODIT_CUTWING}\n\n")

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
            f.write("Vsys handling: fixed at systemic velocity\n")
            if SVI_LR_DECAY_STEPS is not None and SVI_LR_DECAY_RATE is not None:
                f.write(
                    "SVI LR schedule: "
                    f"exponential_decay(steps={SVI_LR_DECAY_STEPS}, "
                    f"rate={SVI_LR_DECAY_RATE})\n"
                )
            else:
                f.write("SVI LR schedule: constant\n")
        else:
            f.write("SVI: SKIPPED\n")

        if not svi_only:
            f.write(f"\nMCMC warmup: {MCMC_NUM_WARMUP:,}\n")
            f.write(f"MCMC samples: {MCMC_NUM_SAMPLES:,}\n")
            f.write(f"MCMC chains: {MCMC_NUM_CHAINS}\n")
            f.write(f"MCMC chain method: {MCMC_CHAIN_METHOD}\n")
            f.write(f"MCMC require GPU per chain: {MCMC_REQUIRE_GPU_PER_CHAIN}\n")
            f.write(f"MCMC max tree depth: {MCMC_MAX_TREE_DEPTH}\n")
        else:
            f.write("\nMCMC: SKIPPED (SVI only)\n")

        # Tellurics
        if ENABLE_TELLURICS:
            f.write("\nTelluric correction: ENABLED\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Run configuration saved to {log_path}")
