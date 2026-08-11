"""Shared PEPSI loading and provenance services for HRS preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import config_utils
from dataio.collapse_transmission_timeseries_to_1d import get_pepsi_data
from dataio.stellar_lsd import load_stellar_velocity_result


HRS_MODES = frozenset({"transmission", "emission"})


def _validated_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in HRS_MODES:
        raise ValueError(
            f"Unsupported HRS preparation mode {mode!r}; expected transmission or emission."
        )
    return normalized


def output_dir_for(
    *,
    mode: str,
    planet: str,
    epoch: str,
    arm: str,
    product_kind: str,
) -> Path:
    """Return the canonical output directory for one prepared HRS arm."""

    mode = _validated_mode(mode)
    if product_kind == "collapse-source":
        return config_utils.get_collapse_source_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode=mode,
        )
    return config_utils.get_timeseries_data_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        mode=mode,
    )


def raw_input_dir_for(*, mode: str, planet: str, epoch: str) -> Path:
    """Return the configured raw PEPSI directory for a preparation mode."""

    return config_utils.get_raw_hrs_dir(
        planet=planet,
        epoch=epoch,
        mode=_validated_mode(mode),
    )


def resolve_stellar_velocity_correction(
    *,
    mode: str,
    planet: str,
    epoch: str,
    arm: str,
) -> tuple[float, dict[str, Any]]:
    """Load an accepted LSD result and return its correction plus provenance."""

    mode = _validated_mode(mode)
    result_path = config_utils.get_data_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        mode=mode,
    ).parent / "stellar_velocity_lsd.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            "Cannot create stellar-rest product for "
            f"{planet} {mode} {epoch} {arm}: required accepted "
            f"stellar-velocity result is missing at {result_path}."
        )
    try:
        result = load_stellar_velocity_result(
            result_path,
            planet=planet,
            mode=mode,
            epoch=epoch,
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot create stellar-rest product for "
            f"{planet} {mode} {epoch} {arm}: {exc}"
        ) from exc

    velocity = float(result["systemic_velocity_kms"])
    source_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    print(
        f"Using LSD stellar-rest correction: {velocity:+.4f} km/s "
        f"from {result_path}."
    )
    return velocity, {
        "applied": True,
        "accepted_for_stellar_rest": True,
        "method": result["method"],
        "result_schema_version": int(result["schema_version"]),
        "source_file": str(result_path),
        "source_sha256": source_sha256,
        "systemic_velocity_kms": velocity,
        "systemic_velocity_stat_err_kms": float(
            result["systemic_velocity_stat_err_kms"]
        ),
        "systemic_velocity_err_kms": float(result["systemic_velocity_err_kms"]),
        "systematic_error_floor_kms": float(result["systematic_error_floor_kms"]),
        "wavelength_frame": result["wavelength_frame"],
        "wavelength_medium": result["wavelength_medium"],
        "template_sha256": result.get("template_sha256"),
        "correction_convention": result.get("correction_convention"),
    }


def requested_stellar_velocity_correction(
    *,
    enabled: bool,
    mode: str,
    planet: str,
    epoch: str,
    arm: str,
) -> tuple[float | None, dict[str, Any]]:
    """Resolve an optional stellar-rest correction for a preparation command."""

    mode = _validated_mode(mode)
    if not enabled:
        return None, {
            "applied": False,
            "reason": "not_requested_barycentric_output",
            "planet": planet,
            "mode": mode,
        }
    return resolve_stellar_velocity_correction(
        mode=mode,
        planet=planet,
        epoch=epoch,
        arm=arm,
    )


def _unwrap_loader_result(result: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def load_hrs_arm(
    *,
    mode: str,
    arm: str,
    epoch: str,
    planet: str,
    molecfit: bool,
    regrid: bool,
    subtract_median: bool,
    run_sysrem: bool,
    stellar_rest_velocity_kms: float | None,
    edge_trim_widths_A: tuple[float, float] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Load and preprocess one PEPSI arm with the shared fallback policy."""

    mode = _validated_mode(mode)
    if arm == "full":
        raise ValueError(
            "load_hrs_arm() is per-arm; 'full' must be expanded into its "
            "constituent arms by the caller."
        )
    if arm not in {"blue", "red"}:
        raise ValueError(f"Unsupported PEPSI arm {arm!r}; expected blue or red.")

    raw_dir = raw_input_dir_for(mode=mode, planet=planet, epoch=epoch)
    prefer_molecfit = bool(molecfit) and arm != "blue"

    def load(*, use_molecfit: bool):
        return get_pepsi_data(
            arm=arm,
            observation_epoch=epoch,
            planet_name=planet,
            do_molecfit=use_molecfit,
            data_dir=raw_dir,
            regrid=regrid,
            subtract_median=subtract_median,
            run_sysrem=run_sysrem,
            wavelength_frame="barycentric",
            stellar_rest_velocity_kms=stellar_rest_velocity_kms,
            edge_trim_widths_A=edge_trim_widths_A,
            data_mode=mode,
        )

    result = load(use_molecfit=prefer_molecfit)
    if result is None and prefer_molecfit:
        print(f"  No molecfit files for {arm}; retrying with raw files.")
        result = load(use_molecfit=False)
    if result is None:
        raise FileNotFoundError(
            f"Could not load {arm}-arm PEPSI {mode} data for {planet} {epoch} "
            f"from {raw_dir}."
        )
    return _unwrap_loader_result(result)
