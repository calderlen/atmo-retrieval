"""Shared PEPSI loading and provenance services for HRS preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

import config_utils
from dataio.collapse_transmission_timeseries_to_1d import (
    get_pepsi_data,
    get_sysrem_chunk_indices,
    get_sysrem_ignore_mask,
)
from dataio.stellar_lsd import load_stellar_velocity_result


def output_dir_for(
    *,
    mode: str,
    planet: str,
    epoch: str,
    arm: str,
    product_kind: str,
) -> Path:
    """Return the canonical output directory for one prepared HRS arm."""

    mode = str(mode).strip().lower()
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
        mode=str(mode).strip().lower(),
    )


def resolve_stellar_velocity_correction(
    *,
    mode: str,
    planet: str,
    epoch: str,
    arm: str,
) -> tuple[float, dict[str, Any]]:
    """Load an accepted LSD result and return its correction plus provenance."""

    mode = str(mode).strip().lower()
    result_path = config_utils.get_data_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        mode=mode,
    ).parent / "stellar_velocity_lsd.json"
    result = load_stellar_velocity_result(
        result_path,
        planet=planet,
        mode=mode,
        epoch=epoch,
    )

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

    mode = str(mode).strip().lower()
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

    mode = str(mode).strip().lower()

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
    return result


def valid_sorted_column_indices(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    *,
    arm: str | None = None,
    edge_trim_widths_A: tuple[float, float] | None = None,
) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)


    valid = np.isfinite(wavelength) & (wavelength > 0.0)
    valid &= np.all(np.isfinite(data), axis=0)
    valid &= np.all(np.isfinite(sigma), axis=0)
    valid &= np.all(sigma > 0.0, axis=0)
    if arm is not None:
        valid &= ~get_sysrem_ignore_mask(
            wavelength,
            arm,
            explicit_edge_trim_widths_A=edge_trim_widths_A,
        )

    if not np.any(valid):
        raise ValueError("No valid spectral columns remain after masking.")

    valid_indices = np.flatnonzero(valid)
    sort_idx = np.argsort(wavelength[valid_indices])
    return valid_indices[sort_idx]


def save_sysrem_products(
    output_dir: Path,
    *,
    wavelength: np.ndarray,
    arm: str,
    n_source_exposures: int,
    source_sigma: np.ndarray,
    extras: dict[str, Any],
    run_sysrem: bool,
) -> None:
    """Save the full-exposure basis and pixel weights, or remove stale products."""
    if not run_sysrem:
        for filename in ("U_sysrem.npz", "pre_sysrem_data.npy", "pre_sysrem_sigma.npy"):
            (output_dir / filename).unlink(missing_ok=True)
        return

    U_full = np.asarray(extras["U_sysrem"])
    chunk_names, chunk_indices, _ = get_sysrem_chunk_indices(wavelength, arm)
    chunk_labels = np.full(wavelength.size, -1, dtype=int)
    for chunk_id, indices in enumerate(chunk_indices):
        chunk_labels[np.asarray(indices, dtype=int)] = chunk_id

    basis = np.asarray(U_full, dtype=float)
    if basis.ndim == 2:
        basis = basis[:, :, np.newaxis]
    basis_counts = np.sum(np.any(np.isfinite(basis), axis=0), axis=0).astype(int)
    diagnostic_keys = (
        "sysrem_stddev_before",
        "sysrem_stddev_after",
        "sysrem_delta_stddev",
        "sysrem_component_attempted",
        "sysrem_component_accepted",
        "sysrem_min_systematics",
        "sysrem_max_systematics",
        "sysrem_stop_delta_stddev",
    )
    diagnostics = {
        key: np.asarray(extras[key])
        for key in diagnostic_keys
        if extras.get(key) is not None
    }
    np.savez_compressed(
        output_dir / "U_sysrem.npz",
        U_sysrem=U_full,
        chunk_labels=chunk_labels,
        basis_counts=basis_counts,
        projection_sigma=source_sigma,
        chunk_names=np.asarray(chunk_names, dtype="U32"),
        **diagnostics,
    )
    print(
        "  Saved chunked SYSREM bundle: "
        f"{len(chunk_names)} chunks, basis counts={basis_counts.tolist()}"
    )
    if "sysrem_delta_stddev" in diagnostics:
        print("  Saved SYSREM component diagnostics: sysrem_delta_stddev and acceptance masks")
