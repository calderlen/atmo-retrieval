"""Helper APIs for the flat :mod:`config` module."""

from __future__ import annotations

import math
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

import config


_AU_KM = 149_597_870.7
_DAY_SECONDS = 86_400.0
_JUPITER_MASS_IN_SOLAR_MASSES = 0.0009545942339693249

TIMING_PARAMETER_FIELDS = (
    "period",
    "period_err",
    "epoch",
    "epoch_err",
    "epoch_scale",
    "epoch_reference",
    "timing_model",
    "dperiod_depoch",
    "transit_midpoints_bjd_tdb",
    "duration",
    "duration_err",
    "timing_source",
    "RA",
    "Dec",
)

SHADOW_PARAMETER_FIELDS = (
    "v_sini_star",
    "v_sini_star_err",
    "lambda_angle",
    "lambda_angle_err",
    "gamma1",
    "gamma2",
    "b",
    "rp_rs",
    "a_rs",
    "inclination",
    "inclination_err",
    "a",
    "a_err",
    "M_star",
    "M_star_err",
    "M_p",
    "M_p_err",
    "eccentricity",
    "eccentricity_err",
    "eccentricity_reported",
    "eccentricity_model",
    "eccentricity_adoption_reason",
    "omega",
    "omega_err",
    "Kp",
    "Kp_err",
    "Kp_source",
    "Kp_is_derived",
    "Kp_derivation",
    "geometry_source",
    "rotation_source",
    "limb_darkening_source",
)


_REMOVED_RUNTIME_CONFIG_NAMES = frozenset(
    {
        "PRESSURE" + "_TOP",
        "PRESSURE" + "_BTM",
        "PT" + "_PROFILE" + "_DEFAULT",
    }
)


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def derive_planet_kp(planet_params: dict) -> dict:
    """Derive the planet barycentric RV semiamplitude from orbital parameters.

    The adopted convention is

    ``Kp = 2*pi*a_p*sin(i) / (P*sqrt(1-e**2))``,

    where ``a_p = a*M_star/(M_star + M_p)``. Configuration units are AU,
    days, degrees, solar masses, and Jupiter masses; the result is km/s.
    No edge-on, circular, or negligible-planet-mass fallback is used.
    """

    required = {
        name: _finite_float(planet_params.get(name))
        for name in (
            "a",
            "period",
            "inclination",
            "eccentricity",
            "M_star",
            "M_p",
        )
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Cannot derive Kp without finite " + ", ".join(missing) + "."
        )

    semi_major_axis_au = float(required["a"])
    period_day = float(required["period"])
    inclination_deg = float(required["inclination"])
    eccentricity = float(required["eccentricity"])
    stellar_mass_msun = float(required["M_star"])
    planet_mass_mjup = float(required["M_p"])
    if semi_major_axis_au <= 0.0 or period_day <= 0.0 or stellar_mass_msun <= 0.0:
        raise ValueError("Kp derivation requires positive a, period, and M_star.")
    if planet_mass_mjup < 0.0:
        raise ValueError("Kp derivation requires non-negative M_p.")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("Kp derivation requires 0 <= eccentricity < 1.")
    sine_inclination = math.sin(math.radians(inclination_deg))
    if sine_inclination <= 0.0:
        raise ValueError("Kp derivation requires 0 < inclination < 180 degrees.")

    planet_mass_msun = planet_mass_mjup * _JUPITER_MASS_IN_SOLAR_MASSES
    planet_semimajor_axis_au = (
        semi_major_axis_au
        * stellar_mass_msun
        / (stellar_mass_msun + planet_mass_msun)
    )
    kp_kms = (
        2.0
        * math.pi
        * planet_semimajor_axis_au
        * _AU_KM
        * sine_inclination
        / (
            period_day
            * _DAY_SECONDS
            * math.sqrt(1.0 - eccentricity**2)
        )
    )
    return {
        "Kp": float(kp_kms),
        "Kp_err": float("nan"),
        "Kp_source": "derived_from_orbit",
        "Kp_is_derived": True,
        "Kp_derivation": {
            "formula": "2*pi*a_planet*sin(inclination)/(period*sqrt(1-eccentricity^2))",
            "a_planet_definition": "a*M_star/(M_star+M_p)",
            "input_units": {
                "a": "AU",
                "period": "day",
                "inclination": "degree",
                "M_star": "solar_mass",
                "M_p": "jupiter_mass",
            },
            "uncertainty": "not_propagated_without_coherent_posterior_samples",
            "inputs": {
                name: float(value) for name, value in required.items()
            },
        },
    }


def _with_resolved_kp(params: dict, *, source: str) -> dict:
    result = dict(params)
    if _finite_float(result.get("Kp")) is not None:
        result.setdefault("Kp_is_derived", False)
        result.setdefault("Kp_source", f"configured:{source}")
        return result
    try:
        result.update(derive_planet_kp(result))
    except ValueError:
        result.setdefault("Kp_is_derived", False)
        result.setdefault("Kp_source", "unresolved")
    return result


def get_params(planet: str | None = None, ephemeris: str | None = None) -> dict:
    """Get a copied parameter block with derivable quantities resolved."""
    planet = planet or config.PLANET
    ephemeris = ephemeris or config.EPHEMERIS
    return _with_resolved_kp(config.PLANETS[planet][ephemeris], source=ephemeris)


def resolve_parameters(
    *,
    planet: str,
    source: str,
    fields: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Resolve one explicit parameter source, optionally restricted to fields."""

    params = get_params(planet, source)
    if fields is None:
        return params
    return {name: params.get(name) for name in fields}


def resolve_parameter_domains(
    *,
    planet: str,
    timing_source: str,
    shadow_source: str = "Recommended",
) -> dict:
    """Combine timing and Doppler-shadow domains without monolithic routing.

    Timing quantities remain tied to ``timing_source``. Stellar profile,
    transit geometry, obliquity, eccentric geometry, and Kp are taken from the
    independently selected ``shadow_source``. A missing shadow-source Kp is
    re-derived after merging so the selected timing period is honored.
    """

    timing = get_params(planet, timing_source)
    shadow = get_params(planet, shadow_source)
    result = dict(timing)
    for name in SHADOW_PARAMETER_FIELDS:
        if name in shadow:
            result[name] = shadow[name]

    if bool(shadow.get("Kp_is_derived", False)):
        result["Kp"] = float("nan")
        result["Kp_err"] = float("nan")
        result.pop("Kp_derivation", None)
        result = _with_resolved_kp(
            result,
            source=f"timing={timing_source};shadow={shadow_source}",
        )

    result["parameter_resolution"] = {
        "timing_source": timing_source,
        "shadow_source": shadow_source,
        "timing_fields": list(TIMING_PARAMETER_FIELDS),
        "shadow_fields": list(SHADOW_PARAMETER_FIELDS),
    }
    return result


def list_planets() -> list[str]:
    """List all available planets."""
    return list(config.PLANETS.keys())


def list_ephemerides(planet: str | None = None) -> list[str]:
    """List available ephemerides for a planet."""
    planet = planet or config.PLANET
    return list(config.PLANETS[planet].keys())


def resolve_transit_midpoint(
    bjd_tdb: np.ndarray,
    planet_params: dict,
    *,
    reference_epoch_bjd_tdb: float,
    observation_epoch: str | None = None,
) -> float:
    """Resolve the transit midpoint nearest an observing sequence.

    Supported timing models are ``linear``, ``quadratic``, and ``ttv_table``.
    A missing ``timing_model`` retains the historical linear behavior.  TTV
    tables fail closed when no midpoint is configured for the requested raw
    observing epoch.
    """

    times = np.asarray(bjd_tdb, dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("BJD_TDB observation times must be finite and non-empty.")

    period = float(planet_params["period"])
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("The orbital period must be finite and positive.")
    reference = float(reference_epoch_bjd_tdb)
    if not np.isfinite(reference):
        raise ValueError("The reference ephemeris epoch must be finite.")

    model = str(planet_params.get("timing_model", "linear")).strip().lower()
    observation_midpoint = 0.5 * (float(np.min(times)) + float(np.max(times)))

    if model == "ttv_table":
        table = planet_params.get("transit_midpoints_bjd_tdb")
        if not isinstance(table, dict):
            raise ValueError(
                "timing_model='ttv_table' requires transit_midpoints_bjd_tdb."
            )
        key = None if observation_epoch is None else str(observation_epoch)
        if key is None or key not in table:
            raise ValueError(
                "No TTV midpoint is configured for observation epoch "
                f"{observation_epoch!r}."
            )
        midpoint = float(table[key])
        if not np.isfinite(midpoint):
            raise ValueError(
                f"The TTV midpoint configured for {key!r} must be finite."
            )
        return midpoint

    epoch_number = round((observation_midpoint - reference) / period)
    if model == "linear":
        return float(reference + epoch_number * period)

    if model == "quadratic":
        dperiod_depoch = float(planet_params.get("dperiod_depoch", np.nan))
        if not np.isfinite(dperiod_depoch):
            raise ValueError(
                "timing_model='quadratic' requires finite dperiod_depoch."
            )
        return float(
            reference
            + epoch_number * period
            + 0.5 * dperiod_depoch * epoch_number**2
        )

    raise ValueError(
        f"Unsupported timing_model={model!r}; expected linear, quadratic, or ttv_table."
    )


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
    pepsi_slice_extraction_modes = ("dxs", "sxs")
    pepsi_slice_exts = ["i"]

    patterns = []
    base_path = str(data_dir)

    if do_molecfit:
        for mode in pepsi_coadd_extraction_modes:
            for ext in pepsi_coadd_exts:
                patterns.append(f"{base_path}/molecfit_weak/SCIENCE_TELLURIC_CORR_{file_prefix}*.{mode}.{ext}.fits")
                patterns.append(f"{base_path}/**/SCIENCE_TELLURIC_CORR_{file_prefix}*.{mode}.{ext}.fits")
        for extraction_mode in pepsi_slice_extraction_modes:
            for ext in pepsi_slice_exts:
                patterns.append(
                    f"{base_path}/molecfit_weak/SCIENCE_TELLURIC_CORR_"
                    f"{file_prefix}*.{extraction_mode}.{ext}.fits"
                )
                patterns.append(
                    f"{base_path}/**/SCIENCE_TELLURIC_CORR_"
                    f"{file_prefix}*.{extraction_mode}.{ext}.fits"
                )
    else:
        for mode in pepsi_coadd_extraction_modes:
            for ext in pepsi_coadd_exts:
                patterns.append(f"{base_path}/{file_prefix}*.{mode}.{ext}")
                patterns.append(f"{base_path}/**/{file_prefix}*.{mode}.{ext}")
        for extraction_mode in pepsi_slice_extraction_modes:
            for ext in pepsi_slice_exts:
                patterns.append(
                    f"{base_path}/{file_prefix}*.{extraction_mode}.{ext}"
                )
                patterns.append(
                    f"{base_path}/**/{file_prefix}*.{extraction_mode}.{ext}"
                )

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


def get_transmission_reference_pressure_bar() -> float:
    """Return and validate the adopted transmission radius reference pressure.

    The reference radius is passed to ExoJAX as ``radius_btm``.  Requiring the
    adopted reference pressure to equal the configured transmission lower
    boundary makes that radius-pressure association explicit and prevents a
    pressure-grid change from silently changing the meaning of the radius.
    """
    reference_pressure_bar = float(config.TRANSMISSION_REFERENCE_PRESSURE_BAR)
    _, pressure_btm = get_pressure_bounds_for_mode("transmission")
    if not math.isfinite(reference_pressure_bar) or reference_pressure_bar <= 0.0:
        raise ValueError(
            "TRANSMISSION_REFERENCE_PRESSURE_BAR must be a finite positive pressure"
        )
    if not math.isclose(reference_pressure_bar, pressure_btm, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(
            "Transmission reference pressure must equal the transmission pressure "
            f"lower boundary: P_ref={reference_pressure_bar:g} bar, "
            f"P_btm={pressure_btm:g} bar"
        )
    return reference_pressure_bar


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


def resolve_pt_profile_for_region(
    mode: str,
    *,
    primary_pt_profile: str,
    is_primary: bool,
    pt_profile: str | None = None,
) -> str:
    """Resolve a region override, the primary selection, or the region-mode default."""
    if pt_profile is not None:
        return pt_profile
    if is_primary:
        return primary_pt_profile
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


def get_hrs_observation_arms(
    planet: str,
    epoch: str,
    *,
    mode: str | None = None,
) -> tuple[str, ...]:
    """Return the explicitly active PEPSI arms for one observation."""

    resolved_mode = _normalize_retrieval_mode(mode)
    requested_key = (resolved_mode, _planet_slug(planet), str(epoch))
    matches = [
        tuple(arms)
        for (configured_mode, configured_planet, configured_epoch), arms
        in config.HRS_OBSERVATION_ARMS.items()
        if (
            configured_mode.strip().lower(),
            _planet_slug(configured_planet),
            str(configured_epoch),
        )
        == requested_key
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate HRS observation-arm configuration for {requested_key}."
        )
    arms = matches[0] if matches else tuple(config.FULL_ARM_MEMBERS)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError(f"Invalid HRS observation arms for {requested_key}: {arms!r}.")
    invalid = set(arms).difference(config.FULL_ARM_MEMBERS)
    if invalid:
        raise ValueError(
            f"Invalid HRS observation arms for {requested_key}: {sorted(invalid)}."
        )
    return arms


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


def get_timeseries_data_dir(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
    *,
    mode: str | None = None,
) -> Path:
    """Get the retrieval-ready, phase-selected time-series directory."""
    return get_data_dir(
        planet=planet,
        arm=arm,
        epoch=epoch,
        mode=mode,
    ) / "timeseries"


def get_collapse_source_dir(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
    *,
    mode: str | None = None,
) -> Path:
    """Get the full-exposure source-cube directory used by 1D collapsers."""
    return get_data_dir(
        planet=planet,
        arm=arm,
        epoch=epoch,
        mode=mode,
    ) / "collapse_source"


def get_full_arm_timeseries_dirs(
    planet: str | None = None,
    epoch: str | None = None,
    *,
    mode: str | None = None,
) -> dict[str, Path]:
    """Return retrieval-ready time-series directories for both arms."""
    return {
        arm: get_timeseries_data_dir(
            planet=planet,
            arm=arm,
            epoch=epoch,
            mode=mode,
        )
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
    *,
    collapsed: bool = False,
) -> dict[str, Path]:
    """Get paths to transmission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch, mode="transmission")
    if collapsed:
        data_dir = data_dir / "collapsed" / "full_transit"
    return {
        "wavelength": data_dir / "wavelength_transmission.npy",
        "spectrum": data_dir / "spectrum_transmission.npy",
        "uncertainty": data_dir / "uncertainty_transmission.npy",
    }


def get_collapsed_transmission_dir(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
) -> Path:
    """Get the directory for a collapsed full-transit spectrum."""
    return get_transmission_paths(
        planet=planet,
        arm=arm,
        epoch=epoch,
        collapsed=True,
    )["wavelength"].parent


def get_emission_paths(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
    *,
    selection: str | None = None,
) -> dict[str, Path]:
    """Get paths to emission data files."""
    data_dir = get_data_dir(planet, arm=arm, epoch=epoch, mode="emission")
    if selection is not None:
        selection_name = str(selection).strip().lower().replace("-", "_")
        if selection_name in {"full", "full_transit"}:
            selection_name = "full_emission"
        if selection_name not in {
            "full_emission",
            "pre_eclipse",
            "post_eclipse",
        }:
            raise ValueError(f"Unsupported collapsed emission selection: {selection!r}")
        data_dir = data_dir / "collapsed" / selection_name
    return {
        "wavelength": data_dir / "wavelength_emission.npy",
        "spectrum": data_dir / "spectrum_emission.npy",
        "uncertainty": data_dir / "uncertainty_emission.npy",
    }


def get_collapsed_emission_dir(
    planet: str | None = None,
    arm: str | None = None,
    epoch: str | None = None,
    *,
    selection: str,
) -> Path:
    """Get the directory for one phase-selected collapsed emission spectrum."""
    return get_emission_paths(
        planet=planet,
        arm=arm,
        epoch=epoch,
        selection=selection,
    )["wavelength"].parent


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
    phoenix_cache_dir: str | None = None,
    save_mcmc_diagnostics: bool = True,
    sigma_scale: float = 1.0,
    spectral_stride: int = 1,
    spectral_offset: int = 0,
    diagnostic_label: str | None = None,
    apply_sysrem_override: bool | None = None,
    emission_selection: str | None = None,
    reference_pressure_bar: float | None = None,
    retrieval_intent: dict | None = None,
    model_param_overrides: dict[str, float] | None = None,
) -> None:
    """Save run configuration to log file."""
    import jax

    log_path = os.path.join(output_dir, "run_config.log")
    params = dict(get_params())
    if model_param_overrides:
        params.update(model_param_overrides)
    normalized_mode = _normalize_retrieval_mode(mode)
    if normalized_mode == "transmission":
        expected_reference_pressure_bar = get_transmission_reference_pressure_bar()
        if reference_pressure_bar is None:
            reference_pressure_bar = expected_reference_pressure_bar
        elif not math.isclose(
            float(reference_pressure_bar),
            expected_reference_pressure_bar,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "reference_pressure_bar must equal the configured transmission "
                f"boundary ({expected_reference_pressure_bar:g} bar)"
            )

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
        if emission_selection is not None:
            f.write(f"Collapsed emission selection: {emission_selection}\n")
        f.write(f"Period: {params['period']}\n")
        f.write(f"R_p: {params['R_p']}\n")
        f.write(f"M_p: {params['M_p']}\n")
        f.write(f"R_star: {params['R_star']}\n")
        f.write(f"T_star: {params['T_star']}\n")
        f.write(
            "Absolute stellar systemic velocity (metadata only): "
            f"{params.get('RV_abs')}\n\n"
        )

        f.write("RETRIEVAL CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Config profile: {get_runtime_profile_name()}\n")
        f.write(f"P-T profile: {pt_profile}\n")
        if normalized_mode == "transmission":
            f.write("Radius convention: adopted reference-radius prior\n")
            f.write(f"R_ref prior center: {params['R_p']} R_J\n")
            f.write(f"R_ref prior width: {params['R_p_err']} R_J\n")
            f.write(f"Reference pressure P_ref: {reference_pressure_bar:g} bar\n")
            f.write(
                "RT radius input: R_ref passed as radius_btm at P_ref\n"
            )
            f.write(
                "Interpretation: adopted modeling convention; not a direct "
                "measurement of R_1bar\n"
            )
        if diagnostic_label is not None:
            f.write(f"Diagnostic label: {diagnostic_label}\n")
        if chemistry_model is not None:
            f.write(f"Chemistry model: {chemistry_model}\n")
        if retrieval_intent is not None:
            f.write("Resolved retrieval intent (JSON):\n")
            f.write(json.dumps(retrieval_intent, indent=2, sort_keys=True))
            f.write("\n")
        if mode == "emission":
            f.write("PHOENIX spectrum source: auto-fetch/cache\n")
            f.write(
                f"PHOENIX cache dir: {phoenix_cache_dir or config.PHOENIX_CACHE_DIR}\n"
            )
            f.write("PHOENIX stellar rotation: applied by retrieval\n")
            f.write("PHOENIX instrumental profile: applied by retrieval\n")
            f.write(f"Stellar v sin(i): {params.get('v_sini_star')} km/s\n")
            f.write("Stellar denominator velocity: 0 km/s\n")
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
        if normalized_mode == "transmission":
            f.write(f"Reference pressure equals RT lower boundary: {reference_pressure_bar:g} bar\n")
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
            f.write("Velocity offset: global v_sys ~ Normal(0, 10 km/s) in stellar-rest frame\n")
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
