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
from config import FULL_ARM_MEMBERS


SPEED_OF_LIGHT_KMS = 299792.458

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
        bin_size=bin_size,
    )
    exposure_indices = np.arange(data.shape[0])[:, None]
    left_indices = operator["shift_left_indices"]
    fractions = operator["shift_fractions"]
    left_data = data[exposure_indices, left_indices]
    right_data = data[exposure_indices, left_indices + 1]
    shifted_data = left_data + fractions * (right_data - left_data)
    spectrum_unbinned = np.sum(
        operator["coadd_weights"] * shifted_data,
        axis=0,
    )
    retained = operator["bin_indices"].size
    spectrum_binned = np.bincount(
        operator["bin_indices"],
        weights=operator["bin_weights"] * spectrum_unbinned[:retained],
        minlength=operator["output_wavelength"].size,
    )
    return (
        operator["output_wavelength"],
        spectrum_binned,
        operator["output_uncertainty"],
    )


def build_emission_collapse_operator(
    wavelength: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    *,
    kp_kms: float,
    bin_size: int = 1,
) -> dict[str, np.ndarray]:
    """Build the fixed shift, coadd, and binning operator for one selection."""
    wavelength = np.asarray(wavelength, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    phase = np.asarray(phase, dtype=float)

    velocities = float(kp_kms) * np.sin(2.0 * np.pi * phase)
    beta = velocities / SPEED_OF_LIGHT_KMS
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    shift_query_full = doppler_factor[:, None] * wavelength[None, :]
    covered_wavelength = wavelength
    shift_query_wavelength = shift_query_full
    shift_right_indices = np.searchsorted(
        wavelength,
        shift_query_wavelength,
        side="left",
    )
    shift_right_indices = np.clip(
        shift_right_indices,
        1,
        wavelength.size - 1,
    )
    shift_left_indices = shift_right_indices - 1
    left_wavelength = wavelength[shift_left_indices]
    right_wavelength = wavelength[shift_right_indices]
    shift_fractions = (
        (shift_query_wavelength - left_wavelength)
        / (right_wavelength - left_wavelength)
    )
    exposure_indices = np.arange(sigma.shape[0])[:, None]
    left_sigma = sigma[exposure_indices, shift_left_indices]
    right_sigma = sigma[exposure_indices, shift_right_indices]
    shifted_sigma = left_sigma + shift_fractions * (right_sigma - left_sigma)
    inverse_variance = 1.0 / shifted_sigma**2
    weight_sum = np.sum(inverse_variance, axis=0)
    coadd_weights = inverse_variance / weight_sum[None, :]
    uncertainty_unbinned = np.sqrt(1.0 / weight_sum)

    n_bins = covered_wavelength.size // bin_size
    retained = n_bins * bin_size
    bin_indices = np.repeat(np.arange(n_bins, dtype=np.int32), bin_size)
    wavelength_retained = covered_wavelength[:retained]
    uncertainty_retained = uncertainty_unbinned[:retained]
    bin_inverse_variance = 1.0 / uncertainty_retained**2
    bin_weight_sum = np.bincount(
        bin_indices,
        weights=bin_inverse_variance,
        minlength=n_bins,
    )
    bin_weights = bin_inverse_variance / bin_weight_sum[bin_indices]
    output_wavelength = np.bincount(
        bin_indices,
        weights=bin_weights * wavelength_retained,
        minlength=n_bins,
    )
    output_uncertainty = np.sqrt(1.0 / bin_weight_sum)

    return {
        "shift_left_indices": shift_left_indices.astype(np.int32),
        "shift_fractions": shift_fractions,
        "coadd_weights": coadd_weights,
        "bin_indices": bin_indices,
        "bin_weights": bin_weights,
        "output_wavelength": output_wavelength,
        "output_uncertainty": output_uncertainty,
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
    epoch: str,
    arm: str,
    selection: str,
    kp_kms: float,
    bin_size: int,
    min_exposures: int,
) -> dict[str, Any]:
    """Build one phase-selected, one-night, one-arm emission spectrum."""
    canonical = canonicalize_emission_selection(selection)
    params = config_utils.get_params(planet, ephemeris)
    data_dir = config_utils.get_collapse_source_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        mode="emission",
    )
    output_dir = collapsed_emission_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        selection=canonical,
    )
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
        "schema_version": 3,
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
        "epoch": epoch,
        "arm": arm,
        "selection": canonical,
        "selection_definition": describe_emission_selection(canonical, params),
        "eclipse_ingress_phase": ingress,
        "eclipse_egress_phase": egress,
        "kp_reference_kms": float(kp_kms),
        "velocity_offset_reference_kms": 0.0,
        "source_data_dir": str(data_dir),
        "source_phase_bin": source_metadata.get("phase_bin"),
        "source_subtract_median": source_metadata.get("subtract_median"),
        "source_run_sysrem": source_metadata.get("run_sysrem"),
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
        bin_size=bin_size,
    )
    wavelength_1d, spectrum_1d, uncertainty_1d = collapse_selected_emission_exposures(
        wavelength,
        data[selected_indices],
        selected_sigma,
        selected_phase,
        kp_kms=kp_kms,
        bin_size=bin_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / _PRODUCT_FILENAMES[0], wavelength_1d)
    np.save(output_dir / _PRODUCT_FILENAMES[1], spectrum_1d)
    np.save(output_dir / _PRODUCT_FILENAMES[2], uncertainty_1d)
    np.savez_compressed(
        output_dir / _PRODUCT_FILENAMES[3],
        schema_version=np.asarray(3, dtype=np.int32),
        source_wavelength=np.asarray(wavelength, dtype=float),
        source_phase=np.asarray(phase, dtype=float),
        selected_exposure_indices=np.asarray(selected_indices, dtype=np.int32),
        shift_left_indices=collapse_operator["shift_left_indices"],
        shift_fractions=collapse_operator["shift_fractions"],
        coadd_weights=collapse_operator["coadd_weights"],
        bin_indices=collapse_operator["bin_indices"],
        bin_weights=collapse_operator["bin_weights"],
        output_wavelength=collapse_operator["output_wavelength"],
        kp_reference_kms=np.asarray(kp_kms, dtype=float),
        velocity_offset_reference_kms=np.asarray(0.0, dtype=float),
        has_sysrem=np.asarray(frozen_sysrem is not None, dtype=bool),
        **({} if frozen_sysrem is None else frozen_sysrem),
    )
    metadata.update(
        {
            "collapse_operator_file": _PRODUCT_FILENAMES[3],
            "phase_min": float(np.min(phase_01[selected_indices])),
            "phase_max": float(np.max(phase_01[selected_indices])),
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
    params = config_utils.get_params(args.planet, args.ephemeris)
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
