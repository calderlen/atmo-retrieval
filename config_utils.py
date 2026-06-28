"""Helper APIs for the flat :mod:`config` module."""

from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path

import config


_REMOVED_RUNTIME_CONFIG_NAMES = frozenset(
    {
        "PRESSURE" + "_TOP",
        "PRESSURE" + "_BTM",
        "PT" + "_PROFILE" + "_DEFAULT",
    }
)


def get_params(planet: str | None = None, ephemeris: str | None = None) -> dict:
    """Get planet parameters for the specified planet and ephemeris."""
    planet = planet or config.PLANET
    ephemeris = ephemeris or config.EPHEMERIS
    return config.PLANETS[planet][ephemeris]


def list_planets() -> list[str]:
    """List all available planets."""
    return list(config.PLANETS.keys())


def list_ephemerides(planet: str | None = None) -> list[str]:
    """List available ephemerides for a planet."""
    planet = planet or config.PLANET
    return list(config.PLANETS[planet].keys())


def _pepsi_data_patterns(
    observation_epoch: str,
    planet_name: str,
    mode: str,
    file_prefix: str,
    do_molecfit: bool = True,
    data_dir: str = "input",
) -> list[str]:
    """Get glob patterns for finding PEPSI data files."""
    if file_prefix is None:
        raise ValueError(f"No file prefix defined for mode '{mode}'")

    year = int(observation_epoch[0:4])

    pepsi_coadd_exts = ["nor", "avr"]
    if year >= 2024:
        pepsi_coadd_exts.insert(0, "bwl")
    pepsi_coadd_extraction_modes = ("dxt", "sxt")
    pepsi_sxs_exts = ["i"]

    patterns = []
    base_path = str(data_dir)

    if do_molecfit:
        for mode in pepsi_coadd_extraction_modes:
            for ext in pepsi_coadd_exts:
                patterns.append(f"{base_path}/molecfit_weak/SCIENCE_TELLURIC_CORR_{file_prefix}*.{mode}.{ext}.fits")
                patterns.append(f"{base_path}/**/SCIENCE_TELLURIC_CORR_{file_prefix}*.{mode}.{ext}.fits")
        for ext in pepsi_sxs_exts:
            patterns.append(f"{base_path}/molecfit_weak/SCIENCE_TELLURIC_CORR_{file_prefix}*.sxs.{ext}.fits")
            patterns.append(f"{base_path}/**/SCIENCE_TELLURIC_CORR_{file_prefix}*.sxs.{ext}.fits")
    else:
        for mode in pepsi_coadd_extraction_modes:
            for ext in pepsi_coadd_exts:
                patterns.append(f"{base_path}/{file_prefix}*.{mode}.{ext}")
                patterns.append(f"{base_path}/**/{file_prefix}*.{mode}.{ext}")
        for ext in pepsi_sxs_exts:
            patterns.append(f"{base_path}/{file_prefix}*.sxs.{ext}")
            patterns.append(f"{base_path}/**/{file_prefix}*.sxs.{ext}")

    return patterns


def get_instrument_config(
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict:
    """Get instrument configuration dict."""
    obs = observatory or config.OBSERVATORY
    inst = instrument or config.INSTRUMENT
    return config.INSTRUMENTS[obs][inst]


def get_mode_config(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> dict:
    """Get observing mode configuration dict."""
    obs = observatory or config.OBSERVATORY
    inst = instrument or config.INSTRUMENT
    m = mode or config.OBSERVING_MODE
    return config.INSTRUMENTS[obs][inst]["modes"][m]


def get_resolution(
    observatory: str | None = None,
    instrument: str | None = None,
    resolution_mode: str | None = None,
) -> int:
    """Get spectral resolving power R = lambda / delta lambda for instrument."""
    instrument_config = get_instrument_config(observatory, instrument)
    res_mode = resolution_mode or config.RESOLUTION_MODE

    if "resolution_modes" in instrument_config and res_mode in instrument_config["resolution_modes"]:
        return instrument_config["resolution_modes"][res_mode]
    return instrument_config["resolution"]


def _normalize_retrieval_mode(mode: str | None = None) -> str:
    normalized = str(mode or config.RETRIEVAL_MODE).strip().lower()
    if normalized not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported retrieval mode: {mode!r}")
    return normalized


def get_pressure_bounds_for_mode(mode: str | None = None) -> tuple[float, float]:
    """Return the atmospheric pressure range in bar for a retrieval mode."""
    normalized = _normalize_retrieval_mode(mode)
    if normalized == "transmission":
        return (
            config.TRANSMISSION_PRESSURE_TOP,
            config.TRANSMISSION_PRESSURE_BTM,
        )
    return (
        config.EMISSION_PRESSURE_TOP,
        config.EMISSION_PRESSURE_BTM,
    )


def get_pt_profile_default_for_mode(mode: str | None = None) -> str:
    """Return the default P-T profile for a retrieval mode."""
    normalized = _normalize_retrieval_mode(mode)
    if normalized == "transmission":
        return config.TRANSMISSION_PT_PROFILE_DEFAULT
    return config.EMISSION_PT_PROFILE_DEFAULT


def resolve_pt_profile_for_mode(mode: str | None = None, pt_profile: str | None = None) -> str:
    """Return an explicit P-T profile or the default for the retrieval mode."""
    if pt_profile is not None:
        return pt_profile
    return get_pt_profile_default_for_mode(mode)


def get_wavelength_range(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> tuple[float, float]:
    """Get wavelength range in Angstroms for observing mode."""
    return get_mode_config(observatory, instrument, mode)["range"]


def get_file_prefix(
    observatory: str | None = None,
    instrument: str | None = None,
    mode: str | None = None,
) -> str | None:
    """Get file prefix for observing mode."""
    return get_mode_config(observatory, instrument, mode)["file_prefix"]


def get_header_keys(
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict[str, str]:
    """Get FITS header key mappings for instrument."""
    return get_instrument_config(observatory, instrument)["header_keys"]


def get_fits_columns(
    molecfit: bool = True,
    observatory: str | None = None,
    instrument: str | None = None,
) -> dict[str, str]:
    """Get FITS column names for data files."""
    cols = get_instrument_config(observatory, instrument)["fits_columns"]
    return cols["molecfit" if molecfit else "raw"]


def get_data_patterns(
    observation_epoch: str,
    planet_name: str,
    mode: str | None = None,
    do_molecfit: bool = True,
    data_dir: str = "input",
    observatory: str | None = None,
    instrument: str | None = None,
) -> list[str]:
    """Get glob patterns for finding data files."""
    m = mode or config.OBSERVING_MODE
    inst_config = get_instrument_config(observatory, instrument)
    mode_config = get_mode_config(observatory, instrument, m)

    family = inst_config.get("data_pattern_family")
    if family == "pepsi":
        return _pepsi_data_patterns(
            observation_epoch=observation_epoch,
            planet_name=planet_name,
            mode=m,
            file_prefix=mode_config["file_prefix"],
            do_molecfit=do_molecfit,
            data_dir=data_dir,
        )
    raise ValueError(f"Unsupported data pattern family: {family!r}")


def _planet_slug(planet: str) -> str:
    return planet.strip().lower().replace("-", "").replace(" ", "")


def _normalize_retrieval_mode(mode: str | None) -> str:
    resolved = (mode or config.RETRIEVAL_MODE).strip().lower()
    if resolved not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported retrieval mode: {mode!r}")
    return resolved


def get_raw_hrs_dir(
    planet: str | None = None,
    *,
    epoch: str | None = None,
    mode: str | None = None,
) -> Path:
    """Get raw high-resolution exposure directory for a planet and epoch."""
    planet_slug = _planet_slug(planet or config.PLANET)
    resolved_mode = _normalize_retrieval_mode(mode)
    base = config.INPUT_DIR / "hrs" / resolved_mode / "raw" / planet_slug
    if epoch:
        return base / epoch
    return base


def get_data_dir(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
    *,
    mode: str | None = None,
) -> Path:
    """Get processed high-resolution data directory."""
    planet_slug = _planet_slug(planet or config.PLANET)
    resolved_mode = _normalize_retrieval_mode(mode)
    resolved_arm = arm or config.OBSERVING_MODE
    if resolved_arm == "full":
        raise ValueError(
            "arm='full' has no single on-disk directory; red and blue are stored "
            "separately under <epoch>/red and <epoch>/blue. Use "
            "get_full_arm_data_dirs() instead, or pass arm='red'/'blue'."
        )
    base = config.INPUT_DIR / "hrs" / resolved_mode / planet_slug
    if epoch:
        return base / epoch / resolved_arm
    return base / resolved_arm


def get_full_arm_data_dirs(
    planet: str | None = None,
    epoch: str | None = None,
    *,
    mode: str | None = None,
) -> dict[str, Path]:
    """Return per-arm data directories for a full-arm retrieval."""
    return {
        arm: get_data_dir(planet=planet, arm=arm, epoch=epoch, mode=mode)
        for arm in config.FULL_ARM_MEMBERS
    }


def get_lowres_dir(
    planet: str | None = None,
    *,
    mode: str | None = None,
    raw: bool = False,
) -> Path:
    """Get low-resolution spectrum directory for a planet and mode."""
    planet_slug = _planet_slug(planet or config.PLANET)
    resolved_mode = _normalize_retrieval_mode(mode)
    base = config.INPUT_DIR / "lrs" / resolved_mode
    if raw:
        return base / "raw" / planet_slug
    return base / planet_slug


def get_phot_dir(
    planet: str | None = None,
    *,
    mode: str | None = None,
    raw: bool = False,
) -> Path:
    """Get broadband photometry directory for a planet and mode."""
    planet_slug = _planet_slug(planet or config.PLANET)
    resolved_mode = _normalize_retrieval_mode(mode)
    base = config.INPUT_DIR / "phot" / resolved_mode
    if raw:
        return base / "raw" / planet_slug
    return base / planet_slug


def get_transmission_paths(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
) -> dict[str, Path]:
    """Get paths to transmission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch, mode="transmission")
    return {
        "wavelength": data_dir / "wavelength_transmission.npy",
        "spectrum": data_dir / "spectrum_transmission.npy",
        "uncertainty": data_dir / "uncertainty_transmission.npy",
    }


def get_emission_paths(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
) -> dict[str, Path]:
    """Get paths to emission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch, mode="emission")
    return {
        "wavelength": data_dir / "wavelength_emission.npy",
        "spectrum": data_dir / "spectrum_emission.npy",
        "uncertainty": data_dir / "uncertainty_emission.npy",
    }


def get_output_dir(
    planet: str | None = None,
    ephemeris: str | None = None,
    mode: str | None = None,
) -> Path:
    """Get output directory: output/{planet}/{ephemeris}/{mode}/."""
    planet = planet or config.PLANET
    ephemeris = ephemeris or config.EPHEMERIS
    mode = mode or config.RETRIEVAL_MODE
    return config.PROJECT_ROOT / "output" / planet.lower().replace("-", "") / ephemeris / mode


def create_timestamped_dir(base_dir: str | Path) -> Path:
    """Create timestamped subdirectory within a base directory."""
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_dir = base_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir


def set_runtime_config(name: str, value) -> None:
    """Update a config variable at module scope."""
    if name in _REMOVED_RUNTIME_CONFIG_NAMES:
        raise ValueError(
            f"{name} was removed. Use explicit mode-specific atmospheric config names."
        )
    setattr(config, name, value)


def list_runtime_profiles() -> tuple[str, ...]:
    """Return available named runtime profiles."""
    return tuple(config.CONFIG_PROFILES.keys())


def get_runtime_profile_name() -> str:
    """Return the currently active runtime profile name."""
    return config._active_runtime_profile


def get_runtime_profile(profile_name: str | None = None) -> dict:
    """Return the profile definition for the active or requested profile."""
    normalized = _normalize_runtime_profile_name(
        profile_name if profile_name is not None else config._active_runtime_profile
    )
    return config.CONFIG_PROFILES[normalized]


def _normalize_runtime_profile_name(profile_name: str) -> str:
    return str(profile_name).strip().lower()


def apply_runtime_profile(profile_name: str) -> str:
    """Apply a named runtime profile across config variables."""
    normalized = _normalize_runtime_profile_name(profile_name)
    profile = config.CONFIG_PROFILES[normalized]
    for name, value in profile["overrides"].items():
        set_runtime_config(name, value)
    config._active_runtime_profile = normalized
    return normalized


def save_run_config(
    output_dir: str,
    mode: str,
    pt_profile: str,
    skip_svi: bool,
    svi_only: bool,
    seed: int,
    chemistry_model: str | None = None,
    epoch: str | list[str] | tuple[str, ...] | None = None,
    phoenix_spectrum_path: str | None = None,
    phoenix_cache_dir: str | None = None,
    save_mcmc_diagnostics: bool = True,
    sigma_scale: float = 1.0,
    spectral_stride: int = 1,
    spectral_offset: int = 0,
    diagnostic_label: str | None = None,
    apply_sysrem_override: bool | None = None,
) -> None:
    """Save run configuration to log file."""
    import jax

    log_path = os.path.join(output_dir, "run_config.log")
    params = get_params()

    epoch_values: list[str] = []
    if epoch is None:
        epoch_values = []
    elif isinstance(epoch, str):
        if epoch.strip():
            epoch_values = [epoch.strip()]
    else:
        for value in epoch:
            text = str(value).strip()
            if text:
                epoch_values.append(text)

    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RETRIEVAL RUN CONFIGURATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {seed}\n\n")

        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"JAX version: {jax.__version__}\n")
        f.write(f"JAX backend: {jax.default_backend()}\n")
        f.write(f"JAX devices: {jax.devices()}\n\n")

        f.write("TARGET\n")
        f.write("-" * 70 + "\n")
        f.write(f"Planet: {config.PLANET}\n")
        f.write(f"Ephemeris: {config.EPHEMERIS}\n")
        if epoch_values:
            f.write(f"Epoch: {epoch_values[0]}\n")
            if len(epoch_values) > 1:
                f.write(f"Epochs: {', '.join(epoch_values)}\n")
        f.write(f"Period: {params['period']}\n")
        f.write(f"R_p: {params['R_p']}\n")
        f.write(f"M_p: {params['M_p']}\n")
        f.write(f"R_star: {params['R_star']}\n")
        f.write(f"T_star: {params['T_star']}\n")
        f.write(f"Systemic velocity (fixed): {params.get('RV_abs')}\n\n")

        f.write("RETRIEVAL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Config profile: {get_runtime_profile_name()}\n")
        f.write(f"P-T profile: {pt_profile}\n")
        if diagnostic_label is not None:
            f.write(f"Diagnostic label: {diagnostic_label}\n")
        if chemistry_model is not None:
            f.write(f"Chemistry model: {chemistry_model}\n")
        if phoenix_spectrum_path is not None:
            f.write(f"PHOENIX spectrum: {phoenix_spectrum_path}\n")
        if phoenix_cache_dir is not None:
            f.write(f"PHOENIX cache dir: {phoenix_cache_dir}\n")
        f.write(f"Output directory: {output_dir}\n\n")

        f.write("SPECTRAL SETUP\n")
        f.write("-" * 70 + "\n")
        f.write(f"Observatory: {config.OBSERVATORY}\n")
        f.write(f"Instrument: {config.INSTRUMENT}\n")
        f.write(f"Observing mode: {config.OBSERVING_MODE}\n")
        wav_min, wav_max = get_wavelength_range()
        f.write(f"Wavelength range: {wav_min} - {wav_max} Angstroms\n")
        f.write(f"Spectral points: {config.N_SPECTRAL_POINTS}\n")
        f.write(f"Resolution mode: {config.RESOLUTION_MODE}\n")
        f.write(f"Resolution: R = {get_resolution():,}\n\n")

        f.write("PREMODIT GRID\n")
        f.write("-" * 70 + "\n")
        f.write(f"Cutwing: {config.PREMODIT_CUTWING}\n\n")

        f.write("ATMOSPHERIC SETUP\n")
        f.write("-" * 70 + "\n")
        pressure_top, pressure_btm = get_pressure_bounds_for_mode(mode)
        f.write(f"Layers: {config.NLAYER}\n")
        f.write(f"Pressure range: {pressure_top:.2e} - {pressure_btm:.2e} bar\n")
        f.write(f"Temperature range: {config.T_LOW} - {config.T_HIGH} K\n")
        f.write(f"Cloud width: {config.CLOUD_WIDTH}\n")
        f.write(f"Cloud integrated tau: {config.CLOUD_INTEGRATED_TAU}\n\n")

        f.write("OPACITY SOURCES\n")
        f.write("-" * 70 + "\n")
        f.write("Molecules (HITEMP):\n")
        for mol in config.MOLPATH_HITEMP.keys():
            f.write(f"  - {mol}\n")
        f.write("Molecules (ExoMol):\n")
        for mol in config.MOLPATH_EXOMOL.keys():
            f.write(f"  - {mol}\n")
        f.write("Atomic species:\n")
        for atom in config.ATOMIC_SPECIES.keys():
            f.write(f"  - {atom}\n")
        f.write("\nCIA sources: H2-H2, H2-He\n")
        f.write(f"Opacity loading: {config.OPA_LOAD}\n")
        f.write(f"Opacity saving: {config.OPA_SAVE}\n\n")

        f.write("INFERENCE PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Save MCMC diagnostics: {save_mcmc_diagnostics}\n")
        f.write(f"Spectroscopic sigma scale: {sigma_scale}\n")
        f.write(f"Spectral stride: {spectral_stride}\n")
        f.write(f"Spectral offset: {spectral_offset}\n")
        if apply_sysrem_override is not None:
            f.write(f"SYSREM override: {apply_sysrem_override}\n")
        if not skip_svi:
            f.write(f"SVI steps: {config.SVI_NUM_STEPS:,}\n")
            f.write(f"SVI learning rate: {config.SVI_LEARNING_RATE}\n")
            f.write("Vsys handling: fixed at systemic velocity\n")
            if config.SVI_LR_DECAY_STEPS is not None and config.SVI_LR_DECAY_RATE is not None:
                f.write(
                    "SVI LR schedule: "
                    f"exponential_decay(steps={config.SVI_LR_DECAY_STEPS}, "
                    f"rate={config.SVI_LR_DECAY_RATE})\n"
                )
            else:
                f.write("SVI LR schedule: constant\n")
        else:
            f.write("SVI: SKIPPED\n")

        if not svi_only:
            f.write(f"\nMCMC warmup: {config.MCMC_NUM_WARMUP:,}\n")
            f.write(f"MCMC samples: {config.MCMC_NUM_SAMPLES:,}\n")
            f.write(f"MCMC chains: {config.MCMC_NUM_CHAINS}\n")
            f.write(f"MCMC chain method: {config.MCMC_CHAIN_METHOD}\n")
            f.write(f"MCMC require GPU per chain: {config.MCMC_REQUIRE_GPU_PER_CHAIN}\n")
            f.write(f"MCMC max tree depth: {config.MCMC_MAX_TREE_DEPTH}\n")
        else:
            f.write("\nMCMC: SKIPPED (SVI diagnostic approximation only)\n")

        if config.ENABLE_TELLURICS:
            f.write("\nTelluric correction: ENABLED\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Run configuration saved to {log_path}")
