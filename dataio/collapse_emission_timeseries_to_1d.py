#!/usr/bin/env python
"""Collapse prepared emission time series into phase-selected 1D spectra.

Each requested epoch and spectrograph arm is collapsed independently.  This
preserves night-specific wavelength grids and uncertainties so multi-night
retrievals can combine the resulting products as separate likelihood
components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import config
import config_utils
from dataio.orbital_velocity import planet_radial_velocity_kms
from config import FULL_ARM_MEMBERS


SPEED_OF_LIGHT_KMS = 299792.458
EMISSION_COLLAPSE_SCHEMA_VERSION = 4
COLLAPSE_COVERAGE_POLICY = (
    "all_selected_exposures_in_bounds_and_within_native_segment"
)

EMISSION_COLLAPSE_SELECTIONS = (
    "full_emission",
    "pre_eclipse",
    "post_eclipse",
)

_SELECTION_ALIASES = {
    "full": "full_emission",
    "full-emission": "full_emission",
    "full_emission": "full_emission",
    "full-transit": "full_emission",
    "full_transit": "full_emission",
    "pre-eclipse": "pre_eclipse",
    "pre_eclipse": "pre_eclipse",
    "post-eclipse": "post_eclipse",
    "post_eclipse": "post_eclipse",
}

_PRODUCT_FILENAMES = (
    "wavelength_emission.npy",
    "spectrum_emission.npy",
    "uncertainty_emission.npy",
    "emission_collapse_operator.npz",
)


def canonicalize_emission_selection(selection: str) -> str:
    """Return the canonical name for an emission-collapse selection."""
    key = str(selection).strip().lower()
    return _SELECTION_ALIASES.get(key, key.replace("-", "_"))


def emission_eclipse_boundaries(
    planet_params: dict[str, Any],
) -> tuple[float, float]:
    """Return secondary-eclipse ingress and egress phases around phase 0.5."""
    duration = planet_params["duration"]
    period = planet_params["period"]
    half_width = float(duration) / (2.0 * float(period))
    return 0.5 - half_width, 0.5 + half_width


def emission_selection_mask(
    phase: np.ndarray,
    selection: str,
    planet_params: dict[str, Any],
) -> np.ndarray:
    """Select out-of-eclipse dayside exposures for a collapsed spectrum.

    ``pre_eclipse`` covers phase 0.25 through secondary-eclipse ingress.
    ``post_eclipse`` covers secondary-eclipse egress through phase 0.75.
    ``full_emission`` is the union of those two selections.
    """
    canonical = canonicalize_emission_selection(selection)
    phase_01 = np.mod(np.asarray(phase, dtype=float), 1.0)
    ingress, egress = emission_eclipse_boundaries(planet_params)
    tolerance = 1.0e-12

    pre = (
        (phase_01 >= 0.25 - tolerance)
        & (phase_01 < ingress - tolerance)
    )
    post = (
        (phase_01 > egress + tolerance)
        & (phase_01 <= 0.75 + tolerance)
    )
    if canonical == "pre_eclipse":
        return pre
    if canonical == "post_eclipse":
        return post
    return pre | post


def describe_emission_selection(
    selection: str,
    planet_params: dict[str, Any],
) -> str:
    """Return a reproducible human-readable definition of a phase selection."""
    canonical = canonicalize_emission_selection(selection)
    ingress, egress = emission_eclipse_boundaries(planet_params)
    if canonical == "pre_eclipse":
        return f"phase(mod 1) in [0.25, {ingress:.12f})"
    if canonical == "post_eclipse":
        return f"phase(mod 1) in ({egress:.12f}, 0.75]"
    return (
        f"phase(mod 1) in [0.25, {ingress:.12f}) or "
        f"({egress:.12f}, 0.75]"
    )


def collapsed_emission_dir(
    *,
    planet: str,
    epoch: str,
    arm: str,
    selection: str,
) -> Path:
    """Return the standard directory for one collapsed emission product."""
    canonical = canonicalize_emission_selection(selection)
    return (
        config_utils.get_data_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode="emission",
        )
        / "collapsed"
        / canonical
    )


def load_frozen_sysrem_arrays(
    data_dir: Path,
) -> dict[str, np.ndarray] | None:
    """Load the frozen chunked SYSREM operator saved with a source cube."""
    path = data_dir / "U_sysrem.npz"
    if not path.exists():
        return None
    with np.load(path) as raw:
        if "projection_sigma" not in raw.files:
            if "V_chunk_diag" in raw.files:
                raise ValueError(
                    f"{path} uses the retired V_chunk_diag approximation. "
                    "Regenerate the source cube to retain per-pixel SYSREM "
                    "projection uncertainties."
                )
            raise ValueError(
                f"{path} is missing projection_sigma; regenerate the source cube."
            )
        U = np.asarray(raw["U_sysrem"], dtype=float)
        labels = np.asarray(raw["chunk_labels"], dtype=np.int32)
        counts = np.asarray(raw["basis_counts"], dtype=np.int32)
        projection_sigma = np.asarray(raw["projection_sigma"], dtype=float)
    if U.ndim == 2:
        U = U[:, :, None]
    expected_sigma_shape = (U.shape[0], labels.size)
    if projection_sigma.shape != expected_sigma_shape:
        raise ValueError(
            f"{path} projection_sigma has shape {projection_sigma.shape}; "
            f"expected {expected_sigma_shape}."
        )
    if np.any(~np.isfinite(projection_sigma)) or np.any(
        projection_sigma <= 0.0
    ):
        raise ValueError(
            f"{path} projection_sigma must contain finite positive values."
        )
    return {
        "sysrem_U": U,
        "sysrem_chunk_labels": labels,
        "sysrem_basis_counts": counts,
        "sysrem_projection_sigma": projection_sigma,
    }


def _validate_timeseries_arrays(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if not np.all(np.isfinite(data)):
        raise ValueError("data contains non-finite values; regenerate the prepared cube.")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            "sigma must contain only finite positive values; regenerate the prepared cube."
        )

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    data = data[:, order]
    sigma = sigma[:, order]
    return wavelength, data, sigma, phase


def collapse_selected_emission_exposures(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    *,
    kp_kms: float,
    eccentricity: float = 0.0,
    omega_deg: float | None = None,
    bin_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift selected residual spectra to the planet frame and coadd them."""
    wavelength, data, sigma, phase = _validate_timeseries_arrays(
        wavelength,
        data,
        sigma,
        phase,
    )

    operator = build_emission_collapse_operator(
        wavelength,
        sigma,
        phase,
        kp_kms=kp_kms,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
        bin_size=bin_size,
    )
    return apply_emission_collapse_operator(data, operator)


def apply_emission_collapse_operator(
    data: np.ndarray,
    operator: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one validated planet-frame shift/coadd/bin operator to data."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D; got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")

    left_indices = np.asarray(operator["shift_left_indices"], dtype=np.int32)
    fractions = np.asarray(operator["shift_fractions"], dtype=float)
    coadd_weights = np.asarray(operator["coadd_weights"], dtype=float)
    bin_indices = np.asarray(operator["bin_indices"], dtype=np.int32)
    bin_weights = np.asarray(operator["bin_weights"], dtype=float)
    output_wavelength = np.asarray(operator["output_wavelength"], dtype=float)
    output_uncertainty = np.asarray(operator["output_uncertainty"], dtype=float)

    if left_indices.shape != fractions.shape or left_indices.shape != coadd_weights.shape:
        raise ValueError(
            "shift_left_indices, shift_fractions, and coadd_weights must have "
            f"the same shape; got {left_indices.shape}, {fractions.shape}, and "
            f"{coadd_weights.shape}."
        )
    if left_indices.shape[0] != data.shape[0]:
        raise ValueError(
            "Collapse operator exposure count does not match data: "
            f"{left_indices.shape[0]} versus {data.shape[0]}."
        )
    if np.any(left_indices < 0) or np.any(left_indices + 1 >= data.shape[1]):
        raise ValueError("Collapse operator contains out-of-range source indices.")
    if np.any(~np.isfinite(fractions)) or np.any(
        (fractions < 0.0) | (fractions > 1.0)
    ):
        raise ValueError("Collapse operator fractions must be finite and within [0, 1].")
    if np.any(~np.isfinite(coadd_weights)) or np.any(coadd_weights < 0.0):
        raise ValueError("Collapse operator coadd weights must be finite and nonnegative.")
    if not np.allclose(
        np.sum(coadd_weights, axis=0),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Collapse operator coadd weights must sum to one.")
    if bin_indices.ndim != 1 or bin_weights.shape != bin_indices.shape:
        raise ValueError("bin_indices and bin_weights must be matching 1D arrays.")
    if bin_indices.size != left_indices.shape[1]:
        raise ValueError(
            "Every shifted wavelength must have one bin assignment; got "
            f"{left_indices.shape[1]} shifted wavelengths and "
            f"{bin_indices.size} bin assignments."
        )
    if np.any(bin_indices < 0) or np.any(bin_indices >= output_wavelength.size):
        raise ValueError("Collapse operator contains out-of-range bin indices.")

    exposure_indices = np.arange(data.shape[0])[:, None]
    left_data = data[exposure_indices, left_indices]
    right_data = data[exposure_indices, left_indices + 1]
    shifted_data = left_data + fractions * (right_data - left_data)
    spectrum_unbinned = np.sum(coadd_weights * shifted_data, axis=0)
    spectrum_binned = np.bincount(
        bin_indices,
        weights=bin_weights * spectrum_unbinned,
        minlength=output_wavelength.size,
    )
    return output_wavelength, spectrum_binned, output_uncertainty


def build_emission_collapse_operator(
    wavelength: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    *,
    kp_kms: float,
    velocity_offset_kms: float = 0.0,
    eccentricity: float = 0.0,
    omega_deg: float | None = None,
    bin_size: int = 1,
    max_native_gap_factor: float = config.DEFAULT_REGRID_MAX_NATIVE_GAP_FACTOR,
) -> dict[str, np.ndarray]:
    """Build a coverage-safe shift, coadd, and binning operator."""
    wavelength = np.asarray(wavelength, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength must be a 1D array with at least two pixels.")
    if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be finite and strictly increasing.")
    if sigma.ndim != 2 or sigma.shape != (phase.size, wavelength.size):
        raise ValueError(
            "sigma must have shape (phase.size, wavelength.size); got "
            f"{sigma.shape}, expected {(phase.size, wavelength.size)}."
        )
    if phase.size == 0 or np.any(~np.isfinite(phase)):
        raise ValueError("phase must contain at least one finite exposure phase.")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must contain only finite positive values.")
    if not np.isfinite(kp_kms):
        raise ValueError("kp_kms must be finite.")
    if not np.isfinite(velocity_offset_kms):
        raise ValueError("velocity_offset_kms must be finite.")
    if int(bin_size) != bin_size or int(bin_size) < 1:
        raise ValueError("bin_size must be a positive integer.")
    bin_size = int(bin_size)
    if not np.isfinite(max_native_gap_factor) or max_native_gap_factor <= 0.0:
        raise ValueError("max_native_gap_factor must be finite and positive.")

    spacing = np.diff(wavelength)
    typical_spacing = float(np.median(spacing))
    maximum_gap = float(max_native_gap_factor) * typical_spacing

    velocities = planet_radial_velocity_kms(
        phase,
        kp_kms=float(kp_kms),
        eccentricity=float(eccentricity),
        omega_deg=omega_deg,
    )
    velocities = velocities + float(velocity_offset_kms)
    beta = velocities / SPEED_OF_LIGHT_KMS
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("Planet-frame velocity must remain below the speed of light.")
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    shift_query_full = doppler_factor[:, None] * wavelength[None, :]
    boundary_tolerance = (
        8.0
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(wavelength))))
    )
    within_bounds = (
        (shift_query_full >= wavelength[0] - boundary_tolerance)
        & (shift_query_full <= wavelength[-1] + boundary_tolerance)
    )

    shift_right_indices_full = np.searchsorted(
        wavelength,
        shift_query_full,
        side="left",
    )
    shift_right_indices_full = np.clip(
        shift_right_indices_full,
        1,
        wavelength.size - 1,
    )
    shift_left_indices_full = shift_right_indices_full - 1
    left_wavelength_full = wavelength[shift_left_indices_full]
    right_wavelength_full = wavelength[shift_right_indices_full]
    bracket_width_full = right_wavelength_full - left_wavelength_full

    # With side="left", an exact sample at the first pixel after a gap is
    # initially paired with the final pixel before that gap. Move it to the
    # first valid bracket within its own native segment.
    exact_right = np.isclose(
        shift_query_full,
        right_wavelength_full,
        rtol=0.0,
        atol=boundary_tolerance,
    )
    right_has_native_neighbor = shift_right_indices_full < wavelength.size - 1
    next_indices = np.minimum(shift_right_indices_full + 1, wavelength.size - 1)
    next_width = wavelength[next_indices] - right_wavelength_full
    move_exact_segment_start = (
        (bracket_width_full > maximum_gap)
        & exact_right
        & right_has_native_neighbor
        & (next_width <= maximum_gap)
    )
    shift_left_indices_full = np.where(
        move_exact_segment_start,
        shift_right_indices_full,
        shift_left_indices_full,
    )
    shift_right_indices_full = np.where(
        move_exact_segment_start,
        next_indices,
        shift_right_indices_full,
    )
    left_wavelength_full = wavelength[shift_left_indices_full]
    right_wavelength_full = wavelength[shift_right_indices_full]
    bracket_width_full = right_wavelength_full - left_wavelength_full
    shift_fractions_full = (
        (shift_query_full - left_wavelength_full)
        / bracket_width_full
    )

    gap_safe = bracket_width_full <= maximum_gap
    fraction_tolerance = 64.0 * np.finfo(float).eps
    fraction_safe = (
        np.isfinite(shift_fractions_full)
        & (shift_fractions_full >= -fraction_tolerance)
        & (shift_fractions_full <= 1.0 + fraction_tolerance)
    )
    valid_shift = within_bounds & gap_safe & fraction_safe
    common_coverage = np.all(valid_shift, axis=0)
    if not np.any(common_coverage):
        raise ValueError(
            "No wavelengths retain in-bounds, gap-safe coverage from every "
            "selected exposure after the planet-frame shift."
        )

    covered_source_indices = np.flatnonzero(common_coverage).astype(np.int32)
    covered_wavelength = wavelength[common_coverage]
    shift_left_indices = shift_left_indices_full[:, common_coverage]
    shift_fractions = np.clip(
        shift_fractions_full[:, common_coverage],
        0.0,
        1.0,
    )

    exposure_indices = np.arange(sigma.shape[0])[:, None]
    left_sigma = sigma[exposure_indices, shift_left_indices]
    right_sigma = sigma[exposure_indices, shift_left_indices + 1]
    shifted_sigma = left_sigma + shift_fractions * (right_sigma - left_sigma)
    if np.any(~np.isfinite(shifted_sigma)) or np.any(shifted_sigma <= 0.0):
        raise ValueError(
            "Planet-frame interpolation produced non-finite or nonpositive "
            "uncertainties."
        )
    inverse_variance = 1.0 / shifted_sigma**2
    weight_sum = np.sum(inverse_variance, axis=0)
    if np.any(~np.isfinite(weight_sum)) or np.any(weight_sum <= 0.0):
        raise ValueError("Planet-frame coadd has invalid inverse-variance sums.")
    coadd_weights = inverse_variance / weight_sum[None, :]
    uncertainty_unbinned = np.sqrt(1.0 / weight_sum)

    # Restart bin numbering whenever coverage removes a source pixel or the
    # retained wavelengths cross a native/masked gap. Thus a non-default bin
    # size can never combine disconnected wavelength segments.
    segment_breaks = np.flatnonzero(
        (np.diff(covered_source_indices) != 1)
        | (np.diff(covered_wavelength) > maximum_gap)
    ) + 1
    segment_starts = np.r_[0, segment_breaks]
    segment_stops = np.r_[segment_breaks, covered_wavelength.size]
    bin_indices = np.empty(covered_wavelength.size, dtype=np.int32)
    next_bin = 0
    for start, stop in zip(segment_starts, segment_stops):
        local_bins = np.arange(stop - start, dtype=np.int32) // bin_size
        bin_indices[start:stop] = next_bin + local_bins
        next_bin += int(local_bins[-1]) + 1

    n_bins = next_bin
    bin_inverse_variance = 1.0 / uncertainty_unbinned**2
    bin_weight_sum = np.bincount(
        bin_indices,
        weights=bin_inverse_variance,
        minlength=n_bins,
    )
    bin_weights = bin_inverse_variance / bin_weight_sum[bin_indices]
    output_wavelength = np.bincount(
        bin_indices,
        weights=bin_weights * covered_wavelength,
        minlength=n_bins,
    )
    output_uncertainty = np.sqrt(1.0 / bin_weight_sum)

    all_exposures_in_bounds = np.all(within_bounds, axis=0)
    all_exposures_gap_safe = np.all(gap_safe, axis=0)

    return {
        "planet_velocity_kms": np.asarray(velocities, dtype=float),
        "velocity_offset_kms": np.asarray(velocity_offset_kms, dtype=float),
        "eccentricity": np.asarray(eccentricity, dtype=float),
        "omega_planet_deg": np.asarray(
            np.nan if omega_deg is None else omega_deg,
            dtype=float,
        ),
        "shift_left_indices": shift_left_indices.astype(np.int32),
        "shift_fractions": shift_fractions,
        "coadd_weights": coadd_weights,
        "bin_indices": bin_indices,
        "bin_weights": bin_weights,
        "output_wavelength": output_wavelength,
        "output_uncertainty": output_uncertainty,
        "covered_source_indices": covered_source_indices,
        "n_source_wavelengths": np.asarray(wavelength.size, dtype=np.int32),
        "n_covered_wavelengths": np.asarray(
            covered_wavelength.size,
            dtype=np.int32,
        ),
        "n_dropped_out_of_bounds": np.asarray(
            np.count_nonzero(~all_exposures_in_bounds),
            dtype=np.int32,
        ),
        "n_dropped_gap_crossing": np.asarray(
            np.count_nonzero(
                all_exposures_in_bounds & ~all_exposures_gap_safe
            ),
            dtype=np.int32,
        ),
        "max_native_gap_factor": np.asarray(
            max_native_gap_factor,
            dtype=float,
        ),
        "max_native_gap_angstrom": np.asarray(maximum_gap, dtype=float),
    }


def _load_prepared_timeseries(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    expected = {
        "wavelength": data_dir / "wavelength.npy",
        "data": data_dir / "data.npy",
        "sigma": data_dir / "sigma.npy",
        "phase": data_dir / "phase.npy",
    }
    missing = [path.name for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is missing prepared emission time-series files: "
            + ", ".join(missing)
        )

    metadata_path = data_dir / "timeseries_prep.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse {metadata_path}: {exc}") from exc
    arrays = (
        np.load(expected["wavelength"]),
        np.load(expected["data"]),
        np.load(expected["sigma"]),
        np.load(expected["phase"]),
    )
    wavelength, data, sigma, phase = _validate_timeseries_arrays(*arrays)
    return wavelength, data, sigma, phase, metadata


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "collapse_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_stale_product_arrays(output_dir: Path) -> None:
    for filename in _PRODUCT_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)


def collapse_epoch_arm(
    *,
    planet: str,
    ephemeris: str,
    shadow_source: str = "Recommended",
    epoch: str,
    arm: str,
    selection: str,
    kp_kms: float,
    bin_size: int,
    min_exposures: int,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build one phase-selected, one-night, one-arm emission spectrum."""
    canonical = canonicalize_emission_selection(selection)
    params = config_utils.resolve_parameter_domains(
        planet=planet,
        timing_source=ephemeris,
        shadow_source=shadow_source,
    )
    if source_dir is None:
        data_dir = config_utils.get_collapse_source_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode="emission",
        )
    else:
        data_dir = Path(source_dir)
    if output_dir is None:
        output_dir = collapsed_emission_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            selection=canonical,
        )
    else:
        output_dir = Path(output_dir)
    wavelength, data, sigma, phase, source_metadata = _load_prepared_timeseries(data_dir)
    frozen_sysrem = load_frozen_sysrem_arrays(data_dir)
    if source_metadata.get("run_sysrem") is True and frozen_sysrem is None:
        raise FileNotFoundError(
            f"{data_dir} declares run_sysrem=true but U_sysrem.npz is missing."
        )
    mask = emission_selection_mask(phase, canonical, params)
    selected_indices = np.flatnonzero(mask)
    phase_01 = np.mod(phase, 1.0)
    ingress, egress = emission_eclipse_boundaries(params)

    metadata: dict[str, Any] = {
        "schema_version": EMISSION_COLLAPSE_SCHEMA_VERSION,
        "product_kind": "collapsed_emission_spectrum",
        "observable_kind": "continuum_removed_emission_line_contrast",
        "model_preprocessing": (
            "time_median_subtraction_then_frozen_per_pixel_sysrem_then_"
            "planet_frame_inverse_variance_"
            "coadd_then_subtract_inverse_variance_weighted_constant"
        ),
        "status": "ready",
        "planet": planet,
        "ephemeris": ephemeris,
        "shadow_source": shadow_source,
        "parameter_resolution": params.get("parameter_resolution"),
        "epoch": epoch,
        "arm": arm,
        "selection": canonical,
        "selection_definition": describe_emission_selection(canonical, params),
        "eclipse_ingress_phase": ingress,
        "eclipse_egress_phase": egress,
        "kp_reference_kms": float(kp_kms),
        "planet_velocity_model": (
            "keplerian_transit_centered"
            if float(params.get("eccentricity", 0.0)) != 0.0
            else "circular_sinusoid"
        ),
        "eccentricity": float(params.get("eccentricity", 0.0)),
        "omega_planet_deg": (
            None
            if params.get("omega") is None
            else float(params.get("omega"))
        ),
        "velocity_offset_reference_kms": 0.0,
        "source_data_dir": str(data_dir),
        "source_phase_bin": source_metadata.get("phase_bin"),
        "source_subtract_median": source_metadata.get("subtract_median"),
        "source_run_sysrem": source_metadata.get("run_sysrem"),
        "wavelength_medium": source_metadata.get("wavelength_medium"),
        "wavelength_frame": source_metadata.get("wavelength_frame"),
        "wavelength_frame_contract": source_metadata.get(
            "wavelength_frame_contract"
        ),
        "arm_edge_trim": source_metadata.get("arm_edge_trim"),
        "n_source_exposures": int(phase.size),
        "n_selected_exposures": int(selected_indices.size),
        "selected_exposure_indices": selected_indices.tolist(),
        "bin_size": int(bin_size),
    }

    if selected_indices.size < min_exposures:
        metadata.update(
            {
                "status": "skipped",
                "skip_reason": (
                    f"selected {selected_indices.size} exposures, fewer than "
                    f"min_exposures={min_exposures}"
                ),
                "phase_min": None,
                "phase_max": None,
                "n_output_wavelengths": 0,
            }
        )
        _clear_stale_product_arrays(output_dir)
        _write_metadata(output_dir, metadata)
        return metadata

    selected_phase = phase[selected_indices]
    selected_sigma = sigma[selected_indices]
    collapse_operator = build_emission_collapse_operator(
        wavelength,
        selected_sigma,
        selected_phase,
        kp_kms=kp_kms,
        eccentricity=float(params.get("eccentricity", 0.0)),
        omega_deg=params.get("omega"),
        bin_size=bin_size,
    )
    wavelength_1d, spectrum_1d, uncertainty_1d = apply_emission_collapse_operator(
        data[selected_indices],
        collapse_operator,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / _PRODUCT_FILENAMES[0], wavelength_1d)
    np.save(output_dir / _PRODUCT_FILENAMES[1], spectrum_1d)
    np.save(output_dir / _PRODUCT_FILENAMES[2], uncertainty_1d)
    np.savez_compressed(
        output_dir / _PRODUCT_FILENAMES[3],
        schema_version=np.asarray(
            EMISSION_COLLAPSE_SCHEMA_VERSION,
            dtype=np.int32,
        ),
        source_wavelength=np.asarray(wavelength, dtype=float),
        source_phase=np.asarray(phase, dtype=float),
        selected_exposure_indices=np.asarray(selected_indices, dtype=np.int32),
        shift_left_indices=collapse_operator["shift_left_indices"],
        shift_fractions=collapse_operator["shift_fractions"],
        coadd_weights=collapse_operator["coadd_weights"],
        bin_indices=collapse_operator["bin_indices"],
        bin_weights=collapse_operator["bin_weights"],
        output_wavelength=collapse_operator["output_wavelength"],
        covered_source_indices=collapse_operator["covered_source_indices"],
        n_source_wavelengths=collapse_operator["n_source_wavelengths"],
        n_covered_wavelengths=collapse_operator["n_covered_wavelengths"],
        n_dropped_out_of_bounds=collapse_operator[
            "n_dropped_out_of_bounds"
        ],
        n_dropped_gap_crossing=collapse_operator[
            "n_dropped_gap_crossing"
        ],
        max_native_gap_factor=collapse_operator["max_native_gap_factor"],
        max_native_gap_angstrom=collapse_operator[
            "max_native_gap_angstrom"
        ],
        coverage_policy=np.asarray(COLLAPSE_COVERAGE_POLICY),
        kp_reference_kms=np.asarray(kp_kms, dtype=float),
        planet_velocity_kms=collapse_operator["planet_velocity_kms"],
        eccentricity=collapse_operator["eccentricity"],
        omega_planet_deg=collapse_operator["omega_planet_deg"],
        velocity_offset_reference_kms=np.asarray(0.0, dtype=float),
        has_sysrem=np.asarray(frozen_sysrem is not None, dtype=bool),
        **({} if frozen_sysrem is None else frozen_sysrem),
    )
    metadata.update(
        {
            "collapse_operator_file": _PRODUCT_FILENAMES[3],
            "phase_min": float(np.min(phase_01[selected_indices])),
            "phase_max": float(np.max(phase_01[selected_indices])),
            "wavelength_coverage_policy": COLLAPSE_COVERAGE_POLICY,
            "n_source_wavelengths": int(wavelength.size),
            "n_covered_unbinned_wavelengths": int(
                collapse_operator["n_covered_wavelengths"]
            ),
            "n_dropped_wavelengths": int(
                wavelength.size
                - int(collapse_operator["n_covered_wavelengths"])
            ),
            "n_dropped_out_of_bounds": int(
                collapse_operator["n_dropped_out_of_bounds"]
            ),
            "n_dropped_gap_crossing": int(
                collapse_operator["n_dropped_gap_crossing"]
            ),
            "retained_unbinned_wavelength_fraction": float(
                int(collapse_operator["n_covered_wavelengths"])
                / wavelength.size
            ),
            "max_native_gap_factor": float(
                collapse_operator["max_native_gap_factor"]
            ),
            "max_native_gap_angstrom": float(
                collapse_operator["max_native_gap_angstrom"]
            ),
            "shift_fraction_min": float(
                np.min(collapse_operator["shift_fractions"])
            ),
            "shift_fraction_max": float(
                np.max(collapse_operator["shift_fractions"])
            ),
            "wavelength_min_angstrom": float(np.min(wavelength_1d)),
            "wavelength_max_angstrom": float(np.max(wavelength_1d)),
            "n_output_wavelengths": int(wavelength_1d.size),
        }
    )
    _write_metadata(output_dir, metadata)
    return metadata


def _unique_canonical_selections(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        canonical = canonicalize_emission_selection(value)
        if canonical not in result:
            result.append(canonical)
    return tuple(result)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse prepared emission time-series cubes into full, pre-eclipse, "
            "and post-eclipse 1D spectra."
        )
    )
    parser.add_argument("--planet", required=True, help="Configured planet name")
    parser.add_argument(
        "--ephemeris",
        default=config.EPHEMERIS,
        help=f"Ephemeris configuration (default: {config.EPHEMERIS})",
    )
    parser.add_argument(
        "--shadow-source",
        default="Recommended",
        help="Independent orbital-geometry source (default: Recommended)",
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        required=True,
        help="One or more observation epochs (YYYYMMDD)",
    )
    parser.add_argument(
        "--arm",
        choices=["red", "blue", "full"],
        default=config.DEFAULT_DATA_ARM,
        help="Spectrograph arm; full processes red and blue independently",
    )
    parser.add_argument(
        "--selection",
        nargs="+",
        default=list(EMISSION_COLLAPSE_SELECTIONS),
        help=(
            "Selections to build. Canonical values: "
            + ", ".join(EMISSION_COLLAPSE_SELECTIONS)
            + ". The alias full-transit maps to full_emission."
        ),
    )
    parser.add_argument(
        "--kp-kms",
        type=float,
        default=None,
        help="Reference orbital semi-amplitude for the rest-frame shift (default: planet Kp)",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1,
        help="Inverse-variance spectral bin size (default: 1, no binning)",
    )
    parser.add_argument(
        "--min-exposures",
        type=int,
        default=1,
        help="Minimum selected exposures required for a nightly product (default: 1)",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    epochs = tuple(str(epoch).strip() for epoch in args.epoch if str(epoch).strip())
    params = config_utils.resolve_parameter_domains(
        planet=args.planet,
        timing_source=args.ephemeris,
        shadow_source=args.shadow_source,
    )
    kp_kms = params.get("Kp") if args.kp_kms is None else args.kp_kms

    selections = _unique_canonical_selections(args.selection)
    arms = FULL_ARM_MEMBERS if args.arm == "full" else (args.arm,)
    ready = 0
    skipped = 0

    for epoch in epochs:
        for arm in arms:
            for selection in selections:
                result = collapse_epoch_arm(
                    planet=args.planet,
                    ephemeris=args.ephemeris,
                    shadow_source=args.shadow_source,
                    epoch=str(epoch),
                    arm=arm,
                    selection=selection,
                    kp_kms=float(kp_kms),
                    bin_size=int(args.bin_size),
                    min_exposures=int(args.min_exposures),
                )
                label = f"{epoch}/{arm}/{selection}"
                if result["status"] == "ready":
                    ready += 1
                    print(
                        f"READY {label}: {result['n_selected_exposures']} exposures, "
                        f"{result['n_output_wavelengths']} wavelengths"
                    )
                else:
                    skipped += 1
                    print(f"SKIP  {label}: {result['skip_reason']}")

    print(f"\nCollapsed emission products complete: ready={ready}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
