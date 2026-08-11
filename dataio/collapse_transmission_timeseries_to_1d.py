"""Shared PEPSI preprocessing and collapsed-transmission product builder.

The time-series preparation and retrieval paths import the helpers defined
here. The command-line entry point consumes a dedicated full-exposure
``collapse_source`` cube and writes a self-contained 1D retrieval product.
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from exojax.database.core_atom.io import air_to_vac

import config
import config_utils
from config import (
    EPHEMERIS,
    OBSERVATORY,
    PHASE_BINS,
    PLANETS,
    TELLURIC_REGIONS,
)
from config_utils import (
    get_data_patterns,
    get_header_keys,
    get_fits_columns,
    get_resolution,
)
from dataio.lsd_doppler_shadow import (
    FIXED_LSD_SHADOW_METHOD,
    FIXED_LSD_SHADOW_SCHEMA_VERSION,
)
from dataio.exposure_selection import (
    ScienceExposureSelection,
    apply_paired_snr_quality_rule,
    select_science_exposures,
)
from dataio.orbital_velocity import planet_radial_velocity_kms


PEPSI_INPUT_TIME_SCALE = "utc"
PEPSI_BJD_TDB_VALIDATION_TOLERANCE_SECONDS = 0.1
TRANSMISSION_COLLAPSE_SCHEMA_VERSION = 4


def _calendar_mid_exposure_jd_utc(header, exptime_seconds: float) -> float | None:
    """Reconstruct a UTC midpoint from PEPSI calendar start-time headers."""

    date_obs = str(header.get("DATE-OBS", "")).strip()
    ut_obs = str(header.get("UT-OBS", "")).strip()
    if not date_obs or not ut_obs:
        return None
    if not np.isfinite(exptime_seconds) or exptime_seconds <= 0.0:
        return None

    value = f"{date_obs} {ut_obs}"
    formats = (
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for date_format in formats:
        try:
            start_utc = datetime.strptime(value, date_format)
        except ValueError:
            continue
        return float(
            Time(start_utc, scale=PEPSI_INPUT_TIME_SCALE).jd
            + 0.5 * exptime_seconds / 86400.0
        )
    return None


def compute_contact_phases(params: dict) -> dict[str, float]:
    period = params["period"]
    duration = params["duration"]
    half_dur_phase = (duration / period) / 2

    if bool(params.get("grazing_transit", False)):
        # A grazing event has a valid T14 interval but no T2/T3 contacts.
        # Keep those contacts explicitly undefined rather than inventing a
        # zero-width T23 interval.
        return {
            "T1": -half_dur_phase,
            "T2": np.nan,
            "T3": np.nan,
            "T4": half_dur_phase,
        }

    tau = params["tau"]
    tau_phase = tau / period
    return {
        "T1": -half_dur_phase,
        "T2": -half_dur_phase + tau_phase,
        "T3": half_dur_phase - tau_phase,
        "T4": half_dur_phase,
    }


def active_transit_mask(phase: np.ndarray, params: dict) -> np.ndarray:
    """Return the exposures used by the transmission collapse workflow."""

    phase = np.asarray(phase, dtype=float)
    contacts = compute_contact_phases(params)
    if bool(params.get("grazing_transit", False)):
        return (phase >= contacts["T1"]) & (phase <= contacts["T4"])
    return (phase >= contacts["T2"]) & (phase <= contacts["T3"])


def get_phase_boundaries(params: dict) -> dict[str, tuple[float, float]]:
    c = compute_contact_phases(params)
    return {
        "T12": (c["T1"], c["T2"]),
        "T23": (c["T2"], c["T3"]),
        "T34": (c["T3"], c["T4"]),
    }


# ==============================================================================
# DATA PREPARATION FUNCTIONS
# ==============================================================================

def regrid_to_common_wavelength(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    *,
    max_native_gap_factor: float = config.DEFAULT_REGRID_MAX_NATIVE_GAP_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid all spectra to the first spectrum's wavelength grid.

    Corrects for sub-pixel drift between exposures by interpolating all spectra
    onto a common wavelength grid (the first spectrum's grid). Interpolation is
    only performed inside valid native wavelength segments; invalid pixels,
    uncovered edges, and large native jumps are left as NaN/inf so downstream
    export can drop them instead of treating invented edge values as data.

    Args:
        wave: Wavelength arrays, shape (n_spectra, npix)
        flux: Flux arrays, shape (n_spectra, npix)
        error: Error arrays, shape (n_spectra, npix)
        max_native_gap_factor: Split native wavelength rows wherever the local
            pixel spacing exceeds this factor times the median positive spacing.

    Returns:
        common_wave: Common wavelength grid, shape (npix,)
        flux: Regridded flux, shape (n_spectra, npix)
        error: Regridded error, shape (n_spectra, npix)
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)

    common_wave = wave[0, :].copy()
    n_spectra = wave.shape[0]
    regridded_flux = np.full(flux.shape, np.nan, dtype=float)
    regridded_error = np.full(error.shape, np.inf, dtype=float)

    for i in range(n_spectra):
        valid = (
            np.isfinite(wave[i, :])
            & (wave[i, :] > 0.0)
            & np.isfinite(flux[i, :])
            & np.isfinite(error[i, :])
            & (error[i, :] > 0.0)
        )
        if np.sum(valid) < 2:
            continue

        native_wave = wave[i, valid]
        native_flux = flux[i, valid]
        native_error = error[i, valid]

        sort_idx = np.argsort(native_wave)
        native_wave = native_wave[sort_idx]
        native_flux = native_flux[sort_idx]
        native_error = native_error[sort_idx]

        native_wave, unique_idx = np.unique(native_wave, return_index=True)
        native_flux = native_flux[unique_idx]
        native_error = native_error[unique_idx]
        if native_wave.size < 2:
            continue

        spacing = np.diff(native_wave)
        positive_spacing = spacing[spacing > 0.0]
        if positive_spacing.size == 0:
            continue

        max_gap = float(max_native_gap_factor) * float(np.median(positive_spacing))
        segment_starts = np.r_[0, np.flatnonzero(spacing > max_gap) + 1]
        segment_stops = np.r_[segment_starts[1:], native_wave.size]

        for start, stop in zip(segment_starts, segment_stops):
            if stop - start < 2:
                continue
            segment_wave = native_wave[start:stop]
            segment_flux = native_flux[start:stop]
            segment_error = native_error[start:stop]
            covered = (
                np.isfinite(common_wave)
                & (common_wave >= segment_wave[0])
                & (common_wave <= segment_wave[-1])
            )
            if not np.any(covered):
                continue
            regridded_flux[i, covered] = np.interp(
                common_wave[covered],
                segment_wave,
                segment_flux,
            )
            regridded_error[i, covered] = np.interp(
                common_wave[covered],
                segment_wave,
                segment_error,
            )

    return common_wave, regridded_flux, regridded_error


def keep_finite_common_columns(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop common-grid columns that are not valid for every exposure."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = np.isfinite(wave) & (wave > 0.0)
    valid &= np.all(np.isfinite(flux), axis=0)
    valid &= np.all(np.isfinite(error), axis=0)
    valid &= np.all(error > 0.0, axis=0)
    if not np.any(valid):
        raise ValueError("No valid common-grid columns remain after regridding.")
    return wave[valid], flux[:, valid], error[:, valid], valid


def subtract_median_spectrum(
    flux: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract median spectrum to remove stellar lines and time-invariant tellurics.

    Computes the per-wavelength median across all exposures and subtracts it,
    yielding residual spectra. Error propagation accounts for the uncertainty
    in the median estimate.

    Args:
        flux: Flux arrays, shape (n_spectra, npix)
        error: Error arrays, shape (n_spectra, npix)

    Returns:
        residual_flux: Median-subtracted flux, shape (n_spectra, npix)
        residual_error: Propagated uncertainties, shape (n_spectra, npix)
        median_flux: The subtracted median spectrum, shape (npix,)
    """
    n_spectra = flux.shape[0]
    npix = flux.shape[1]

    # Compute median spectrum
    median_flux = np.median(flux, axis=0)

    # Median error estimation from https://mathworld.wolfram.com/StatisticalMedian.html
    # For large n, var(median) ≈ pi/(2n) * var(mean)
    little_n = (n_spectra - 1) / 2
    correction_factor = np.sqrt(4 * little_n / (np.pi * n_spectra))
    median_error = np.sqrt(np.sum(error**2, axis=0)) / n_spectra / correction_factor

    # Subtract median from each spectrum
    residual_flux = flux - median_flux[np.newaxis, :]

    # Propagate errors: sigma_residual^2 = sigma_flux^2 + sigma_median^2
    residual_error = np.sqrt(error**2 + median_error[np.newaxis, :]**2)

    return residual_flux, residual_error, median_flux


def get_sysrem_chunk_indices(
    wave: np.ndarray,
    arm: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Partition wavelength columns into SYSREM chunks.

    The current scheme is:
    - ``red``: two chunks, non-telluric and telluric
    - ``blue``: one chunk, non-telluric only

    ``arm='full'`` is not supported: run SYSREM independently per arm.
    """
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_sysrem_chunk_indices. "
            "Run SYSREM independently for 'red' and 'blue'."
        )

    wave = np.asarray(wave, dtype=float)

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})

    telluric_mask = np.zeros(wave.shape[0], dtype=bool)
    for wmin, wmax in telluric_config["telluric"]:
        telluric_mask |= (wave > wmin) & (wave <= wmax)

    no_tellurics = np.where(~telluric_mask)[0]
    if arm == "red":
        has_tellurics = np.where(telluric_mask)[0]
        return ("non_telluric", "telluric"), (no_tellurics, has_tellurics), telluric_mask

    return ("non_telluric",), (no_tellurics,), telluric_mask


def get_sysrem_deep_mask(
    wave: np.ndarray,
    arm: str,
) -> np.ndarray:
    """Return wavelengths that should be ignored after Molecfit correction."""
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_sysrem_deep_mask. "
            "Call with 'red' or 'blue' separately."
        )
    wave = np.asarray(wave, dtype=float)

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})

    deep_mask = np.zeros(wave.shape[0], dtype=bool)
    for wmin, wmax in telluric_config["deep_mask"]:
        deep_mask |= (wave >= wmin) & (wave < wmax)
    return deep_mask


def get_telluric_edge_mask(
    wave: np.ndarray,
    arm: str,
    *,
    edge_width_angstrom: float = config.DEFAULT_TELLURIC_EDGE_MASK_WIDTH_A,
) -> np.ndarray:
    """Return columns close to configured telluric/deep-mask boundaries."""
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_telluric_edge_mask. "
            "Call with 'red' or 'blue' separately."
        )
    wave = np.asarray(wave, dtype=float)

    edge_width_angstrom = float(edge_width_angstrom)
    edge_mask = np.zeros(wave.shape[0], dtype=bool)
    if edge_width_angstrom <= 0.0:
        return edge_mask

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})
    boundaries: list[float] = []
    for key in ("telluric", "deep_mask"):
        for wmin, wmax in telluric_config[key]:
            boundaries.extend((float(wmin), float(wmax)))
    for boundary in sorted(set(boundaries)):
        edge_mask |= np.abs(wave - boundary) <= edge_width_angstrom
    return edge_mask


def _validated_edge_trim_widths(
    explicit_widths_A: tuple[float, float],
) -> tuple[float, float]:
    """Validate the exact left/right widths supplied by a manifest consumer."""

    if len(explicit_widths_A) != 2:
        raise ValueError("Explicit edge-trim widths must contain left and right values.")
    left_width, right_width = (float(value) for value in explicit_widths_A)
    if (
        not np.isfinite(left_width)
        or not np.isfinite(right_width)
        or left_width < 0.0
        or right_width < 0.0
    ):
        raise ValueError(
            "Explicit edge-trim widths must be finite, non-negative Angstrom values."
        )
    return left_width, right_width


def get_arm_edge_trim_mask(
    wave: np.ndarray,
    arm: str,
    *,
    explicit_widths_A: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return columns selected by exact, explicitly supplied trim widths."""
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_arm_edge_trim_mask. "
            "Call with 'red' or 'blue' separately."
        )
    wave = np.asarray(wave, dtype=float)

    finite = np.isfinite(wave)
    edge_mask = np.zeros(wave.shape[0], dtype=bool)
    if not np.any(finite) or explicit_widths_A is None:
        return edge_mask

    left_width, right_width = _validated_edge_trim_widths(explicit_widths_A)
    if left_width <= 0.0 and right_width <= 0.0:
        return edge_mask

    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    if left_width > 0.0:
        edge_mask |= finite & (wave < lo + left_width)
    if right_width > 0.0:
        edge_mask |= finite & (wave > hi - right_width)
    return edge_mask


def arm_edge_trim_metadata(
    wave: np.ndarray,
    arm: str,
    *,
    planet: str | None = None,
    mode: str | None = None,
    epoch: str | None = None,
    explicit_widths_A: tuple[float, float] | None = None,
    source: str | None = None,
) -> dict[str, float | int | str]:
    """Describe the configured arm-edge trim for a wavelength grid."""
    wave = np.asarray(wave, dtype=float)
    finite = np.isfinite(wave)
    widths = (
        None
        if explicit_widths_A is None
        else _validated_edge_trim_widths(explicit_widths_A)
    )
    left_width, right_width = (0.0, 0.0) if widths is None else widths
    if not np.any(finite):
        return {
            "left_trim_A": left_width,
            "right_trim_A": right_width,
            "planet": planet or "",
            "mode": mode or "",
            "epoch": epoch or "",
            "raw_min_A": float("nan"),
            "raw_max_A": float("nan"),
            "keep_min_A": float("nan"),
            "keep_max_A": float("nan"),
            "n_trimmed_columns": 0,
            "source": source or "none",
        }

    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    trim_mask = get_arm_edge_trim_mask(
        wave,
        arm,
        explicit_widths_A=explicit_widths_A,
    )
    return {
        "left_trim_A": left_width,
        "right_trim_A": right_width,
        "planet": planet or "",
        "mode": mode or "",
        "epoch": epoch or "",
        "raw_min_A": lo,
        "raw_max_A": hi,
        "keep_min_A": lo + left_width,
        "keep_max_A": hi - right_width,
        "n_trimmed_columns": int(np.count_nonzero(trim_mask)),
        "source": source or "none",
    }


def get_sysrem_ignore_mask(
    wave: np.ndarray,
    arm: str,
    *,
    explicit_edge_trim_widths_A: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return data-quality columns to ignore in SYSREM output."""
    return (
        get_sysrem_deep_mask(wave, arm)
        | get_telluric_edge_mask(wave, arm)
        | get_arm_edge_trim_mask(
            wave,
            arm,
            explicit_widths_A=explicit_edge_trim_widths_A,
        )
    )


def get_sysrem_max_systematics(arm: str) -> list[int]:
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_sysrem_max_systematics. "
            "Call with 'red' or 'blue' separately."
        )
    if arm == "red":
        return list(config.DEFAULT_SYSREM_MAX_SYSTEMATICS_RED)
    return list(config.DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER)


def get_sysrem_min_systematics(arm: str) -> list[int]:
    if arm == "full":
        raise ValueError(
            "arm='full' is not supported by get_sysrem_min_systematics. "
            "Call with 'red' or 'blue' separately."
        )
    if arm == "red":
        return list(config.DEFAULT_SYSREM_MIN_SYSTEMATICS_RED)
    return list(config.DEFAULT_SYSREM_MIN_SYSTEMATICS_OTHER)


def do_sysrem(
    wave: np.ndarray,
    residual_flux: np.ndarray,
    residual_error: np.ndarray,
    arm: str,
    airmass: np.ndarray,
    niter: int = 10,
    do_molecfit: bool = True,
    stop_delta_stddev: float = config.DEFAULT_SYSREM_STOP_TOL,
    return_diagnostics: bool = False,
    planet_name: str | None = None,
    data_mode: str | None = None,
    observation_epoch: str | None = None,
    edge_trim_widths_A: tuple[float, float] | None = None,
) -> tuple[np.ndarray, ...]:
    """Run SYSREM with separate treatment for telluric and non-telluric regions.

    The first systematic is initialized with airmass (physically motivated for
    telluric residuals), subsequent systematics start with unity.

    Args:
        wave: Common wavelength grid, shape (npix,)
        residual_flux: Median-subtracted flux, shape (n_spectra, npix)
        residual_error: Uncertainties, shape (n_spectra, npix)
        arm: Spectrograph arm ('red' or 'blue')
        airmass: Per-exposure airmass values, shape (n_spectra,)
        niter: Number of iterations per systematic (default: 10)
        do_molecfit: If True, mask deep telluric regions (default: True)
        stop_delta_stddev: Minimum sigma improvement required to accept a component
            after the configured minimum component count has been reached
        return_diagnostics: If True, append per-component SYSREM attempt
            diagnostics to the returned tuple

    Returns:
        corrected_flux: Systematics-corrected flux, shape (n_spectra, npix)
        corrected_error: Propagated uncertainties, shape (n_spectra, npix)
        U_sysrem: Systematic vectors, shape (n_spectra, n_sys_used_max, n_chunks)
            Columns not used in a given chunk are NaN.
        no_tellurics: Indices of non-telluric pixels
    """
    n_spectra = residual_flux.shape[0]

    # Work with copies to avoid modifying input
    corrected_flux = residual_flux.copy()
    corrected_error = residual_error.copy()

    _, chunk_indices, _telluric_mask = get_sysrem_chunk_indices(wave, arm)
    ignore_mask = get_arm_edge_trim_mask(
        wave,
        arm,
        explicit_widths_A=edge_trim_widths_A,
    )
    if do_molecfit:
        ignore_mask |= get_sysrem_deep_mask(wave, arm) | get_telluric_edge_mask(wave, arm)
    if np.any(ignore_mask):
        corrected_flux[:, ignore_mask] = 0.0
        corrected_error[:, ignore_mask] = 1.0
        chunk_indices = tuple(
            np.asarray(indices, dtype=int)[~ignore_mask[np.asarray(indices, dtype=int)]]
            for indices in chunk_indices
        )

    no_tellurics = np.asarray(chunk_indices[0], dtype=int)
    chunks = len(chunk_indices)

    max_systematics = get_sysrem_max_systematics(arm)
    min_systematics = get_sysrem_min_systematics(arm)

    # Ensure per-chunk settings have the correct length.
    if len(max_systematics) < chunks:
        max_systematics = max_systematics + [max_systematics[-1]] * (chunks - len(max_systematics))
    if len(min_systematics) < chunks:
        min_systematics = min_systematics + [min_systematics[-1]] * (chunks - len(min_systematics))

    for chunk, (min_n_sys, max_n_sys_chunk) in enumerate(zip(min_systematics, max_systematics)):
        if min_n_sys < 0:
            raise ValueError(f"SYSREM minimum systematics must be non-negative; got {min_n_sys}.")
        if min_n_sys > max_n_sys_chunk:
            raise ValueError(
                f"SYSREM chunk {chunk + 1} minimum systematics ({min_n_sys}) "
                f"exceeds maximum systematics ({max_n_sys_chunk})."
            )

    max_n_sys = max(max_systematics)
    U_sysrem = np.full((n_spectra, max_n_sys, chunks), np.nan, dtype=float)
    n_systematics_used = [0] * chunks
    sysrem_stddev_before = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_stddev_after = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_delta_stddev = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_component_attempted = np.zeros((max_n_sys, chunks), dtype=bool)
    sysrem_component_accepted = np.zeros((max_n_sys, chunks), dtype=bool)

    for chunk in range(chunks):
        this_one = chunk_indices[chunk]
        if len(this_one) == 0:
            continue

        npixhere = len(this_one)
        n_sys_chunk = max_systematics[chunk]
        min_sys_chunk = min_systematics[chunk]

        stddev_prev = np.std(corrected_flux[:, this_one])

        for system in range(n_sys_chunk):
            c = np.zeros(npixhere)
            sigma_c = np.zeros(npixhere)
            sigma_a = np.zeros(n_spectra)

            # Initialize: first systematic uses airmass, others use unity
            if system == 0:
                a = np.array(airmass, dtype=float)
            else:
                a = np.ones(n_spectra)

            for iteration in range(niter):
                # Minimize c for each pixel
                for s in range(npixhere):
                    pix_idx = this_one[s]
                    err_squared = corrected_error[:, pix_idx]**2

                    numerator = np.sum(a * corrected_flux[:, pix_idx] / err_squared)
                    denominator = np.sum(a**2 / err_squared)

                    # Error propagation
                    abs_a = np.abs(a)
                    saoa = np.divide(
                        sigma_a,
                        abs_a,
                        out=np.zeros_like(sigma_a),
                        where=abs_a != 0.0,
                    )
                    abs_flux = np.abs(corrected_flux[:, pix_idx])
                    eof = np.divide(
                        corrected_error[:, pix_idx],
                        abs_flux,
                        out=np.zeros_like(corrected_error[:, pix_idx]),
                        where=abs_flux != 0.0,
                    )

                    sigma_1 = np.abs(a * corrected_flux[:, pix_idx] / err_squared) * np.sqrt(saoa**2 + eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))

                    sigma_2 = np.sqrt(2.0) * np.abs(a) * sigma_a / err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))

                    if denominator != 0:
                        c[s] = numerator / denominator
                        if numerator != 0 and sigma_denominator >= 0:
                            sigma_c[s] = np.abs(c[s]) * np.sqrt(
                                (sigma_numerator / np.abs(numerator))**2 +
                                (sigma_denominator / np.abs(denominator))**2
                            )
                    else:
                        c[s] = 0.0
                        sigma_c[s] = 0.0

                # Using c, minimize a for each epoch
                for ep in range(n_spectra):
                    pix_indices = this_one
                    err_squared = corrected_error[ep, pix_indices]**2

                    numerator = np.sum(c * corrected_flux[ep, pix_indices] / err_squared)
                    denominator = np.sum(c**2 / err_squared)

                    # Error propagation
                    abs_c = np.abs(c)
                    scoc = np.divide(
                        sigma_c,
                        abs_c,
                        out=np.zeros_like(sigma_c),
                        where=abs_c != 0.0,
                    )
                    abs_flux = np.abs(corrected_flux[ep, pix_indices])
                    eof = np.divide(
                        corrected_error[ep, pix_indices],
                        abs_flux,
                        out=np.zeros_like(corrected_error[ep, pix_indices]),
                        where=abs_flux != 0.0,
                    )

                    sigma_1 = np.abs(c * corrected_flux[ep, pix_indices] / err_squared) * np.sqrt(scoc**2 + eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))

                    sigma_2 = np.sqrt(2.0) * np.abs(c) * sigma_c / err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))

                    if denominator != 0:
                        a[ep] = numerator / denominator
                        if numerator != 0 and sigma_denominator >= 0:
                            sigma_a[ep] = np.abs(a[ep]) * np.sqrt(
                                (sigma_numerator / np.abs(numerator))**2 +
                                (sigma_denominator / np.abs(denominator))**2
                            )
                    else:
                        a[ep] = 0.0
                        sigma_a[ep] = 0.0

            syserr = a[:, np.newaxis] * c[np.newaxis, :]

            # Error on systematic term
            abs_a = np.abs(a)
            abs_c = np.abs(c)
            ratio_a = np.divide(
                sigma_a,
                abs_a,
                out=np.zeros_like(sigma_a),
                where=abs_a != 0.0,
            )[:, np.newaxis]
            ratio_c = np.divide(
                sigma_c,
                abs_c,
                out=np.zeros_like(sigma_c),
                where=abs_c != 0.0,
            )[np.newaxis, :]
            sigma_syserr = np.abs(syserr) * np.sqrt(ratio_a**2 + ratio_c**2)

            # Stop when additional components stop improving scatter enough
            trial_flux = corrected_flux[:, this_one] - syserr
            stddev_next = np.std(trial_flux)
            delta_stddev = stddev_prev - stddev_next
            sysrem_stddev_before[system, chunk] = stddev_prev
            sysrem_stddev_after[system, chunk] = stddev_next
            sysrem_delta_stddev[system, chunk] = delta_stddev
            sysrem_component_attempted[system, chunk] = True
            if (
                n_systematics_used[chunk] >= min_sys_chunk
                and delta_stddev <= stop_delta_stddev
            ):
                print(
                    f"  SYSREM chunk {chunk + 1}/{chunks}, component {system + 1}: "
                    f"delta_stddev={delta_stddev:.3e} <= stop_tol={stop_delta_stddev:.1e}; rejected"
                )
                break

            # Accept and apply this systematic
            U_sysrem[:, system, chunk] = a
            corrected_flux[:, this_one] = trial_flux
            corrected_error[:, this_one] = np.sqrt(corrected_error[:, this_one]**2 + sigma_syserr**2)
            stddev_prev = stddev_next
            n_systematics_used[chunk] += 1
            sysrem_component_accepted[system, chunk] = True
            if n_systematics_used[chunk] <= min_sys_chunk:
                reason = "forced minimum"
            else:
                reason = f"delta_stddev > stop_tol={stop_delta_stddev:.1e}"
            print(
                f"  SYSREM chunk {chunk + 1}/{chunks}, component {system + 1}: "
                f"delta_stddev={delta_stddev:.3e}; accepted ({reason})"
            )

        print(
            f"SYSREM chunk {chunk + 1}/{chunks}: used {n_systematics_used[chunk]} "
            f"of min/max {min_sys_chunk}/{n_sys_chunk} systematics"
        )

    # Trim trailing all-NaN columns if discovery mode stopped early.
    used_max = max(n_systematics_used) if n_systematics_used else 0
    if used_max > 0:
        U_sysrem = U_sysrem[:, :used_max, :]
    else:
        U_sysrem = U_sysrem[:, :0, :]

    if return_diagnostics:
        diagnostics = {
            "sysrem_stddev_before": sysrem_stddev_before,
            "sysrem_stddev_after": sysrem_stddev_after,
            "sysrem_delta_stddev": sysrem_delta_stddev,
            "sysrem_component_attempted": sysrem_component_attempted,
            "sysrem_component_accepted": sysrem_component_accepted,
            "sysrem_min_systematics": np.asarray(min_systematics, dtype=int),
            "sysrem_max_systematics": np.asarray(max_systematics, dtype=int),
            "sysrem_stop_delta_stddev": np.asarray(stop_delta_stddev, dtype=float),
        }
        return corrected_flux, corrected_error, U_sysrem, no_tellurics, diagnostics

    return corrected_flux, corrected_error, U_sysrem, no_tellurics


# ==============================================================================
# PHASE BINNING UTILITIES
# ==============================================================================

def get_phase_bin_mask(
    phase: np.ndarray,
    bin_name: str,
    params: dict,
) -> np.ndarray:
    """Get boolean mask for exposures in a given phase bin.
    
    Args:
        phase: Orbital phase array (0 = mid-transit)
        bin_name: One of 'T12', 'T23', 'T34', or 'full'
        params: Planet parameters dict containing 'duration', 'period', and optionally 'tau'
    
    Returns:
        Boolean mask array, True for exposures in the specified bin.
    """
    if bin_name == "full":
        # Full transit: T1 to T4
        contacts = compute_contact_phases(params)
        return (phase >= contacts["T1"]) & (phase <= contacts["T4"])

    boundaries = get_phase_boundaries(params)
    lo, hi = boundaries[bin_name]
    return (phase >= lo) & (phase <= hi)


def filter_data_by_phase(
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    bin_name: str,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter time-series data to specific phase bin.
    
    Args:
        data: Flux/spectra array, shape (n_spectra, npix)
        sigma: Uncertainty array, shape (n_spectra, npix)
        phase: Orbital phase array, shape (n_spectra,)
        bin_name: Phase bin name ('T12', 'T23', 'T34', or 'full')
        params: Planet parameters dict
    
    Returns:
        Tuple of (filtered_data, filtered_sigma, filtered_phase)
    """
    mask = get_phase_bin_mask(phase, bin_name, params)
    return data[mask], sigma[mask], phase[mask]


def get_phase_bin_indices(
    phase: np.ndarray,
    params: dict,
) -> dict[str, np.ndarray]:
    """Get indices for all phase bins.
    
    Args:
        phase: Orbital phase array (0 = mid-transit)
        params: Planet parameters dict
    
    Returns:
        Dict mapping bin_name -> array of indices for that bin.
    """
    result = {}
    for bin_name in PHASE_BINS:
        mask = get_phase_bin_mask(phase, bin_name, params)
        result[bin_name] = np.where(mask)[0]
    return result


def summarize_phase_coverage(
    phase: np.ndarray,
    params: dict,
) -> dict[str, dict]:
    """Summarize phase coverage for each transit bin.
    
    Args:
        phase: Orbital phase array
        params: Planet parameters dict
    
    Returns:
        Dict with bin statistics (count, phase range, etc.)
    """
    contacts = compute_contact_phases(params)
    bin_indices = get_phase_bin_indices(phase, params)
    
    summary = {
        "contacts": contacts,
        "bins": {}
    }
    
    for bin_name, indices in bin_indices.items():
        if len(indices) > 0:
            bin_phases = phase[indices]
            summary["bins"][bin_name] = {
                "count": len(indices),
                "phase_min": float(np.min(bin_phases)),
                "phase_max": float(np.max(bin_phases)),
                "indices": indices.tolist(),
            }
        else:
            summary["bins"][bin_name] = {
                "count": 0,
                "phase_min": None,
                "phase_max": None,
                "indices": [],
            }
    
    # Also compute total in-transit coverage
    full_mask = get_phase_bin_mask(phase, "full", params)
    summary["total_in_transit"] = int(np.sum(full_mask))
    summary["total_exposures"] = len(phase)
    
    return summary


# ==============================================================================
# EPOCH-SPECIFIC CORRECTIONS
# ==============================================================================

_INTRODUCED_SHIFTS_MPS = {
    "20210501": 6000.0,
    "20210518": 3500.0,
    "20190425": 464500.0,
    "20190504": 6300.0,
    "20190515": 506000.0,
    "20190622": -54300.0,
    "20190623": -334000.0,
    "20190625": 97800.0,
    "20210303": -174600.0,
    "20220208": -141300.0,
    "20210628": -57200.0,
    "20211031": -94200.0,
    "20220929": -38600.0,
    "20221202": -96100.0,
    "20230327": -23900.0,
    "20180703": -61800.0,
    "20230430": -19900.0,
    "20220925": -117200.0,
    "20230615": -32400.0,
    "20231023": -97000.0,
    "20231106": -84700.0,
    "20241126": -105800.0,
    "20251002": -89100.0,
    "20240114": -112100.0,
    "20220926": -65000.0,
    "20240312": -75200.0,
}

_LEGACY_DATASET_VELOCITY_OFFSET_MPS = {
    ("KELT-20b", "20190504", "red"): 3223.4,
}

_PLANET_MOLECFIT_VELOCITY_OFFSET_MPS = {
    "WASP-76b": 0.0,
    "TOI-1431b": 24903.0,
    "TOI-1518b": 11170.0,
}

_MOLECFIT_CORRECTED_PREFIX = "SCIENCE_TELLURIC_CORR_"
_MOLECFIT_FITS_SUFFIX = ".fits"
_MAX_PIXEL_SPACING_RATIO_DEVIATION = 1.0e-3
_PEPSI_VELOCITY_SUM_TOLERANCE_MPS = 100.0
_SPEED_OF_LIGHT_MPS = 299792458.0
_SPEED_OF_LIGHT_KMS = _SPEED_OF_LIGHT_MPS / 1000.0
_SUPPORTED_WAVELENGTH_FRAMES = frozenset({"native", "barycentric"})
_REMOVED_VELOCITY_HISTORY_RE = re.compile(
    r"Radial\s+velocity\s+([A-Z0-9_+\-]+)\s*=\s*"
    r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)\s*m/s\s+is\s+removed",
    re.IGNORECASE,
)


def _wavelength_angstrom(data, column_config: dict) -> np.ndarray:
    """Read a configured FITS wavelength column and return Angstroms."""
    wavelength = np.asarray(data[column_config["wave"]], dtype=float).copy()
    unit = str(column_config["wave_unit"]).lower()
    if unit == "micron":
        wavelength *= 10000.0
    elif unit != "angstrom":
        raise ValueError(f"Unsupported PEPSI wavelength unit: {unit!r}")
    return wavelength


def _air_wavelength_to_vacuum(wavelength_air: np.ndarray) -> np.ndarray:
    """Convert valid air wavelengths in Angstroms to vacuum."""
    wavelength_vacuum = np.asarray(wavelength_air, dtype=float).copy()
    valid = np.isfinite(wavelength_vacuum) & (wavelength_vacuum > 0.0)
    wavelength_vacuum[valid] = np.asarray(
        air_to_vac(wavelength_vacuum[valid]),
        dtype=float,
    )
    return wavelength_vacuum


def _molecfit_raw_product_name(molecfit_path: str | os.PathLike[str]) -> str:
    """Return the original PEPSI product name encoded in a Molecfit output."""
    name = Path(molecfit_path).name
    if not name.startswith(_MOLECFIT_CORRECTED_PREFIX) or not name.endswith(
        _MOLECFIT_FITS_SUFFIX
    ):
        raise ValueError(
            "Could not identify the original PEPSI exposure from Molecfit file "
            f"{name!r}."
        )
    raw_name = name[
        len(_MOLECFIT_CORRECTED_PREFIX) : -len(_MOLECFIT_FITS_SUFFIX)
    ]
    if not raw_name:
        raise ValueError(f"Molecfit filename {name!r} has no source product name.")
    return raw_name


def _resolve_molecfit_raw_product(
    molecfit_path: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
) -> Path | None:
    """Locate the exact raw PEPSI product paired with a Molecfit output, if present."""
    molecfit_path = Path(molecfit_path)
    data_root = Path(data_dir)
    raw_name = _molecfit_raw_product_name(molecfit_path)

    direct_candidates = (
        molecfit_path.parent / raw_name,
        molecfit_path.parent.parent / raw_name,
        data_root / raw_name,
    )
    matches: dict[str, Path] = {}
    for candidate in direct_candidates:
        if candidate.is_file():
            matches[str(candidate.resolve())] = candidate
    if not matches and data_root.is_dir():
        for candidate in data_root.rglob(raw_name):
            if candidate.is_file():
                matches[str(candidate.resolve())] = candidate

    if not matches:
        return None
    if len(matches) > 1:
        locations = ", ".join(sorted(matches))
        raise RuntimeError(
            f"Found multiple possible raw products for '{molecfit_path.name}': "
            f"{locations}"
        )
    return next(iter(matches.values()))


def _finite_header_velocity_mps(header, key: str) -> float | None:
    """Return a finite FITS velocity value in m/s, or None when unavailable."""
    if key not in header:
        return None
    try:
        value = float(header[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _history_mentions(header, key: str) -> bool:
    """Return whether a PEPSI HISTORY card says a velocity term was used."""
    history = header.get("HISTORY", ())
    if isinstance(history, str):
        history = (history,)
    return any(key in str(entry) for entry in history)


def _velocity_history_entries(header) -> tuple[str, ...]:
    """Return normalized PEPSI HISTORY entries describing removed velocities."""
    history = header.get("HISTORY", ())
    if isinstance(history, str):
        history = (history,)
    return tuple(
        str(entry).strip()
        for entry in history
        if _REMOVED_VELOCITY_HISTORY_RE.search(str(entry))
    )


def parse_pepsi_velocity_frame(header) -> dict[str, object]:
    """Decompose the velocity terms already removed from a PEPSI wavelength grid.

    The returned observer and stellar components describe the transformation
    recorded by the PEPSI FITS HISTORY cards.  They are deliberately separate:
    a barycentric wavelength retains the stellar velocity while removing the
    observatory/solar-system-barycenter velocity.
    """
    entries = _velocity_history_entries(header)
    if not entries:
        raise ValueError(
            "PEPSI FITS header does not document a removed velocity term in HISTORY."
        )

    removed: list[tuple[str, float]] = []
    for entry in entries:
        match = _REMOVED_VELOCITY_HISTORY_RE.search(entry)
        if match is None:
            continue
        removed.append((match.group(1).upper(), float(match.group(2))))
    labels = {label for label, _value in removed}

    observer_header = _finite_header_velocity_mps(header, "SSBVEL")
    observer_header_key = "SSBVEL"
    if observer_header is None:
        observer_header = _finite_header_velocity_mps(header, "OBSVEL")
        observer_header_key = "OBSVEL"
    stellar_header = _finite_header_velocity_mps(header, "RADVEL")

    def history_value(*requested_labels: str) -> float | None:
        requested = set(requested_labels)
        values = [value for label, value in removed if label in requested]
        if not values:
            return None
        return float(np.sum(values))

    observer_removed = 0.0
    stellar_removed = 0.0
    observer_documented = False
    stellar_documented = False

    standalone_observer = history_value("SSBVEL", "OBSVEL")
    if standalone_observer is not None:
        observer_removed = standalone_observer
        observer_documented = True

    standalone_stellar = history_value("RADVEL")
    if standalone_stellar is not None:
        stellar_removed = standalone_stellar
        stellar_documented = True

    combined_observer_stellar = history_value("OBSVEL+RADVEL")
    if combined_observer_stellar is not None:
        if observer_header is None and stellar_header is None:
            raise ValueError(
                "OBSVEL+RADVEL was removed, but neither component is available "
                "in the PEPSI FITS header."
            )
        if observer_header is None:
            observer_header = combined_observer_stellar - float(stellar_header)
            observer_header_key = "OBSVEL(derived)"
        if stellar_header is None:
            stellar_header = combined_observer_stellar - float(observer_header)
        observer_removed = float(observer_header)
        stellar_removed = float(stellar_header)
        observer_documented = True
        stellar_documented = True
        reconstructed = observer_removed + stellar_removed
        if not np.isclose(
            reconstructed,
            combined_observer_stellar,
            rtol=0.0,
            atol=_PEPSI_VELOCITY_SUM_TOLERANCE_MPS,
        ):
            raise ValueError(
                "PEPSI OBSVEL+RADVEL HISTORY value is inconsistent with its "
                f"components: {combined_observer_stellar:.3f} != "
                f"{reconstructed:.3f} m/s."
            )

    combined_sst = history_value("SSTVEL")
    if combined_sst is not None:
        if observer_header is None and stellar_header is None:
            raise ValueError(
                "SSTVEL was removed, but neither SSBVEL/OBSVEL nor RADVEL is "
                "available in the PEPSI FITS header."
            )
        if observer_header is None:
            observer_header = combined_sst - float(stellar_header)
            observer_header_key = "observer_velocity(derived)"
        if stellar_header is None:
            stellar_header = combined_sst - float(observer_header)
        observer_removed = float(observer_header)
        stellar_removed = float(stellar_header)
        observer_documented = True
        stellar_documented = True
        reconstructed = observer_removed + stellar_removed
        if not np.isclose(
            reconstructed,
            combined_sst,
            rtol=0.0,
            atol=_PEPSI_VELOCITY_SUM_TOLERANCE_MPS,
        ):
            raise ValueError(
                "PEPSI SSTVEL HISTORY value is inconsistent with its components: "
                f"{combined_sst:.3f} != {reconstructed:.3f} m/s."
            )

    if not observer_documented:
        raise ValueError(
            "PEPSI FITS HISTORY does not document removal of OBSVEL, SSBVEL, "
            "OBSVEL+RADVEL, or SSTVEL; a barycentric grid cannot be reconstructed."
        )

    instrument_removed = history_value("CCFOFF") or 0.0
    recognized = {
        "OBSVEL",
        "SSBVEL",
        "RADVEL",
        "OBSVEL+RADVEL",
        "SSTVEL",
        "CCFOFF",
    }
    unknown = sorted(labels - recognized)
    if unknown:
        raise ValueError(
            "Unsupported PEPSI removed-velocity HISTORY term(s): "
            + ", ".join(unknown)
        )

    return {
        "observer_velocity_removed_mps": float(observer_removed),
        "observer_velocity_header_key": observer_header_key,
        "stellar_velocity_removed_mps": float(stellar_removed),
        "stellar_velocity_was_removed": bool(stellar_documented),
        "instrument_velocity_removed_mps": float(instrument_removed),
        "total_velocity_removed_mps": float(sum(value for _label, value in removed)),
        "velocity_history_recipe": tuple(label for label, _value in removed),
        "velocity_history_entries": entries,
    }


def _doppler_factor_velocity_mps(velocity_mps: float) -> float:
    """Return the relativistic wavelength factor for a velocity in m/s."""
    beta = float(velocity_mps) / _SPEED_OF_LIGHT_MPS
    if not np.isfinite(beta) or abs(beta) >= 1.0:
        raise ValueError("Velocity must be finite and smaller than the speed of light.")
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def _raw_pipeline_wavelength_to_barycentric(
    wavelength_vacuum: np.ndarray,
    frame_info: dict[str, object],
) -> np.ndarray:
    """Restore only the stellar term removed from a raw PEPSI wavelength grid."""
    stellar_removed_mps = float(frame_info["stellar_velocity_removed_mps"])
    return np.asarray(wavelength_vacuum, dtype=float) * _doppler_factor_velocity_mps(
        stellar_removed_mps
    )


def _molecfit_velocity_correction_mps(
    header,
    *,
    observation_epoch: str,
    planet_name: str,
    arm: str,
) -> tuple[float, tuple[str, ...]]:
    """Return the velocity correction used by the original PEPSI loader."""
    total_velocity_mps = _INTRODUCED_SHIFTS_MPS.get(observation_epoch, 0.0)
    components: list[str] = []
    if total_velocity_mps != 0.0:
        components.append(f"introduced_shift[{observation_epoch}]")

    if _history_mentions(header, "RADVEL"):
        radial_velocity = _finite_header_velocity_mps(header, "RADVEL")
        if radial_velocity is not None:
            total_velocity_mps += radial_velocity
            components.append("RADVEL(HISTORY)")

    for key in ("OBSVEL", "SSBVEL"):
        velocity = _finite_header_velocity_mps(header, key)
        if velocity is not None:
            total_velocity_mps += velocity
            components.append(key)

    legacy_offset = _LEGACY_DATASET_VELOCITY_OFFSET_MPS.get(
        (planet_name, observation_epoch, arm),
        0.0,
    )
    if legacy_offset != 0.0:
        total_velocity_mps += legacy_offset
        components.append("legacy_dataset_offset")

    planet_offset = _PLANET_MOLECFIT_VELOCITY_OFFSET_MPS.get(planet_name, 0.0)
    if planet_offset != 0.0:
        total_velocity_mps += planet_offset
        components.append("planet_molecfit_offset")

    return total_velocity_mps, tuple(components)


def _shift_wavelength_velocity_mps(
    wavelength: np.ndarray,
    correction_velocity_mps: float,
) -> np.ndarray:
    """Apply the repository's wavelength Doppler convention."""
    beta = float(correction_velocity_mps) / _SPEED_OF_LIGHT_MPS
    return np.asarray(wavelength, dtype=float) / (1.0 - beta)


def _shift_wavelength_to_stellar_rest(
    wavelength: np.ndarray,
    residual_velocity_kms: float,
) -> np.ndarray:
    """Remove a measured residual RV using positive velocity for recession."""
    beta = float(residual_velocity_kms) / _SPEED_OF_LIGHT_KMS
    if not np.isfinite(beta) or abs(beta) >= 1.0:
        raise ValueError("stellar_rest_velocity_kms must be finite and smaller than c.")
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    return np.asarray(wavelength, dtype=float) / doppler_factor


def _validate_one_to_one_wavelength_ordering(
    reference_wave: np.ndarray,
    candidate_wave: np.ndarray,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    """Verify that two wavelength arrays retain the same one-to-one pixel order."""
    reference_wave = np.asarray(reference_wave, dtype=float)
    candidate_wave = np.asarray(candidate_wave, dtype=float)
    if reference_wave.ndim != 1 or candidate_wave.ndim != 1:
        raise ValueError(
            f"{reference_label} and {candidate_label} wavelengths must be 1D."
        )
    if reference_wave.size != candidate_wave.size:
        raise ValueError(
            f"{reference_label} and {candidate_label} pixel counts differ: "
            f"{reference_wave.size} != {candidate_wave.size}."
        )
    if reference_wave.size < 2:
        raise ValueError("PEPSI wavelength arrays must contain at least two pixels.")
    if not np.all(np.isfinite(reference_wave)) or not np.all(
        np.isfinite(candidate_wave)
    ):
        raise ValueError(
            f"{reference_label} and {candidate_label} wavelengths must be finite."
        )

    reference_step = np.diff(reference_wave)
    candidate_step = np.diff(candidate_wave)
    same_increasing_order = np.all(reference_step > 0.0) and np.all(
        candidate_step > 0.0
    )
    same_decreasing_order = np.all(reference_step < 0.0) and np.all(
        candidate_step < 0.0
    )
    if not (same_increasing_order or same_decreasing_order):
        raise ValueError(
            f"{reference_label} and {candidate_label} do not have the same "
            "strictly monotonic pixel ordering."
        )

    spacing_ratio = candidate_step / reference_step
    median_ratio = float(np.median(spacing_ratio))
    if not np.isfinite(median_ratio) or median_ratio <= 0.0:
        raise ValueError(
            f"{reference_label} and {candidate_label} have incompatible pixel spacing."
        )
    maximum_relative_deviation = float(
        np.max(np.abs(spacing_ratio / median_ratio - 1.0))
    )
    if maximum_relative_deviation > _MAX_PIXEL_SPACING_RATIO_DEVIATION:
        raise ValueError(
            f"{candidate_label} does not preserve the pixel-spacing structure of "
            f"{reference_label}; maximum relative spacing-ratio deviation is "
            f"{maximum_relative_deviation:.3e}."
        )


def _canonical_molecfit_wavelength(
    *,
    molecfit_path: str | os.PathLike[str],
    molecfit_data,
    molecfit_header,
    observation_epoch: str,
    planet_name: str,
    arm: str,
    data_dir: str | os.PathLike[str],
    raw_column_config: dict,
    molecfit_column_config: dict,
    jd_header_key: str,
) -> tuple[np.ndarray, Path | None, float, tuple[str, ...]]:
    """Restore a Molecfit grid with the original PEPSI loader correction.

    When the original PEPSI exposure is available, it is used to verify the
    one-to-one flux-pixel mapping. It does not determine the velocity
    correction.
    """
    raw_path = _resolve_molecfit_raw_product(molecfit_path, data_dir)
    molecfit_wave = _wavelength_angstrom(
        molecfit_data,
        molecfit_column_config,
    )
    correction_mps, correction_keys = _molecfit_velocity_correction_mps(
        molecfit_header,
        observation_epoch=observation_epoch,
        planet_name=planet_name,
        arm=arm,
    )
    canonical_wave = _shift_wavelength_velocity_mps(molecfit_wave, correction_mps)

    if raw_path is not None:
        with fits.open(raw_path) as raw_hdu:
            raw_data = raw_hdu[1].data
            raw_header = raw_hdu[0].header.copy()
            raw_air_wave = _wavelength_angstrom(raw_data, raw_column_config)
            raw_flux = np.asarray(
                raw_data[raw_column_config["flux"]],
                dtype=float,
            ).copy()

        if raw_air_wave.size != len(molecfit_data):
            raise ValueError(
                f"Raw product '{raw_path.name}' and Molecfit output "
                f"'{Path(molecfit_path).name}' have different pixel counts: "
                f"{raw_air_wave.size} != {len(molecfit_data)}."
            )

        if jd_header_key in raw_header and jd_header_key in molecfit_header:
            raw_jd = float(raw_header[jd_header_key])
            molecfit_jd = float(molecfit_header[jd_header_key])
            if not np.isclose(raw_jd, molecfit_jd, rtol=0.0, atol=1.0e-5):
                raise ValueError(
                    f"Raw product '{raw_path.name}' and Molecfit output "
                    f"'{Path(molecfit_path).name}' have inconsistent {jd_header_key}: "
                    f"{raw_jd} != {molecfit_jd}."
                )

        _validate_one_to_one_wavelength_ordering(
            _air_wavelength_to_vacuum(raw_air_wave),
            molecfit_wave,
            reference_label=f"vacuum-converted raw grid '{raw_path.name}'",
            candidate_label=f"Molecfit output grid '{Path(molecfit_path).name}'",
        )

        molecfit_input_path = raw_path.with_name(raw_path.name + ".fits")
        if molecfit_input_path.is_file():
            with fits.open(molecfit_input_path) as input_hdu:
                input_data = input_hdu[1].data
                input_wave = _wavelength_angstrom(
                    input_data,
                    molecfit_column_config,
                )
                input_flux = np.asarray(
                    input_data[molecfit_column_config["flux"]],
                    dtype=float,
                ).copy()
            _validate_one_to_one_wavelength_ordering(
                canonical_wave,
                input_wave,
                reference_label=f"corrected Molecfit grid '{Path(molecfit_path).name}'",
                candidate_label=f"Molecfit input grid '{molecfit_input_path.name}'",
            )
            if input_flux.shape != raw_flux.shape or not np.allclose(
                input_flux,
                raw_flux,
                rtol=1.0e-7,
                atol=1.0e-10,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Molecfit input '{molecfit_input_path.name}' does not preserve "
                    f"the flux-pixel ordering of raw product '{raw_path.name}'."
                )

    return canonical_wave, raw_path, correction_mps, correction_keys


def _barycentric_molecfit_wavelength(
    *,
    molecfit_path: str | os.PathLike[str],
    molecfit_data,
    molecfit_header,
    observation_epoch: str,
    data_dir: str | os.PathLike[str],
    raw_column_config: dict,
    molecfit_column_config: dict,
    jd_header_key: str,
) -> tuple[np.ndarray, Path | None, dict[str, object]]:
    """Return a barycentric vacuum grid for a Molecfit-corrected exposure.

    Molecfit flux values retain the input pixel ordering.  When possible, the
    paired raw PEPSI wavelength vector is therefore authoritative.  Legacy
    Molecfit-only products are reconstructed from their header velocity
    provenance without applying target-specific empirical offsets.
    """
    raw_path = _resolve_molecfit_raw_product(molecfit_path, data_dir)
    molecfit_wave = _wavelength_angstrom(molecfit_data, molecfit_column_config)

    if raw_path is not None:
        with fits.open(raw_path) as raw_hdu:
            raw_data = raw_hdu[1].data
            raw_header = raw_hdu[0].header.copy()
            raw_air_wave = _wavelength_angstrom(raw_data, raw_column_config)

        if raw_air_wave.size != len(molecfit_data):
            raise ValueError(
                f"Raw product '{raw_path.name}' and Molecfit output "
                f"'{Path(molecfit_path).name}' have different pixel counts: "
                f"{raw_air_wave.size} != {len(molecfit_data)}."
            )
        if jd_header_key in raw_header and jd_header_key in molecfit_header:
            raw_jd = float(raw_header[jd_header_key])
            molecfit_jd = float(molecfit_header[jd_header_key])
            if not np.isclose(raw_jd, molecfit_jd, rtol=0.0, atol=1.0e-5):
                raise ValueError(
                    f"Raw product '{raw_path.name}' and Molecfit output "
                    f"'{Path(molecfit_path).name}' have inconsistent {jd_header_key}: "
                    f"{raw_jd} != {molecfit_jd}."
                )

        raw_vacuum_wave = _air_wavelength_to_vacuum(raw_air_wave)
        _validate_one_to_one_wavelength_ordering(
            raw_vacuum_wave,
            molecfit_wave,
            reference_label=f"vacuum-converted raw grid '{raw_path.name}'",
            candidate_label=f"Molecfit output grid '{Path(molecfit_path).name}'",
        )
        frame_info = parse_pepsi_velocity_frame(raw_header)
        frame_info = {
            **frame_info,
            "synthetic_molecfit_shift_undone_mps": 0.0,
            "wavelength_frame_source": "paired_raw_product",
        }
        return (
            _raw_pipeline_wavelength_to_barycentric(raw_vacuum_wave, frame_info),
            raw_path,
            frame_info,
        )

    frame_info = parse_pepsi_velocity_frame(molecfit_header)
    synthetic_shift_mps = float(_INTRODUCED_SHIFTS_MPS.get(observation_epoch, 0.0))
    barycentric_wave = np.asarray(molecfit_wave, dtype=float)
    # Undo only mechanical product-generation shifts.  The target-specific
    # legacy and planet offsets are intentionally excluded from an absolute
    # stellar-velocity measurement.
    barycentric_wave /= _doppler_factor_velocity_mps(synthetic_shift_mps)
    barycentric_wave /= _doppler_factor_velocity_mps(
        float(frame_info["observer_velocity_removed_mps"])
    )
    barycentric_wave /= _doppler_factor_velocity_mps(
        float(frame_info["instrument_velocity_removed_mps"])
    )
    frame_info = {
        **frame_info,
        "synthetic_molecfit_shift_undone_mps": synthetic_shift_mps,
        "wavelength_frame_source": "molecfit_header_reconstruction",
    }
    return barycentric_wave, None, frame_info


def _paired_snr_filtered_selection(
    selection: ScienceExposureSelection,
    *,
    arm: str,
    observation_epoch: str,
    planet_name: str,
    data_mode: str,
    data_dir: str | os.PathLike[str],
) -> ScienceExposureSelection:
    """Return this arm's selection after the shared paired-S/N hard cut."""

    if arm not in {"blue", "red"}:
        return selection
    companion_arm = "red" if arm == "blue" else "blue"
    companion_selection = None
    companion_product_choices = (True, False) if companion_arm == "red" else (False,)
    for companion_molecfit in companion_product_choices:
        companion_patterns = get_data_patterns(
            observation_epoch,
            planet_name,
            mode=companion_arm,
            do_molecfit=companion_molecfit,
            data_dir=data_dir,
        )
        candidate = select_science_exposures(
            companion_patterns,
            planet_name=planet_name,
            data_mode=data_mode,
            observation_epoch=observation_epoch,
            arm=companion_arm,
            do_molecfit=companion_molecfit,
        )
        if candidate.discovered_files:
            companion_selection = candidate
            break
    if companion_selection is None:
        return selection

    if arm == "blue":
        blue_selection, _ = apply_paired_snr_quality_rule(
            selection,
            companion_selection,
        )
        return blue_selection
    _, red_selection = apply_paired_snr_quality_rule(
        companion_selection,
        selection,
    )
    return red_selection


def get_pepsi_data(
    arm: str,
    observation_epoch: str,
    planet_name: str,
    do_molecfit: bool = config.DEFAULT_USE_MOLECFIT,
    data_dir: str | os.PathLike[str] | None = None,
    regrid: bool = False,
    subtract_median: bool = False,
    run_sysrem: bool = False,
    *,
    wavelength_frame: str = "native",
    stellar_rest_velocity_kms: float | None = None,
    sysrem_stop_tol: float = config.DEFAULT_SYSREM_STOP_TOL,
    data_mode: str = "transmission",
    edge_trim_widths_A: tuple[float, float] | None = None,
) -> tuple[np.ndarray, ...] | tuple[tuple[np.ndarray, ...], dict] | None:
    """Load and preprocess spectroscopic data from configured instrument.

    Args:
        arm: Spectrograph arm ('red' or 'blue')
        observation_epoch: Observation date (YYYYMMDD)
        planet_name: Planet name
        do_molecfit: Use Molecfit-corrected flux/error. ``native`` retains the
            historical Molecfit velocity-offset recipe; ``barycentric`` uses
            the paired raw grid or the documented header-only reconstruction.
        data_dir: Epoch-specific raw exposure directory
        regrid: Regrid all spectra to common wavelength grid
        subtract_median: Subtract median spectrum (stellar line removal)
        run_sysrem: Run SYSREM systematics removal
        wavelength_frame: Return the historical loader grid (``native``) or
            reconstruct a vacuum barycentric grid (``barycentric``)
        sysrem_stop_tol: Minimum sigma-improvement required to keep a SYSREM component
        data_mode: Data family ('transmission' or 'emission') for preprocessing context
        stellar_rest_velocity_kms: Measured residual stellar velocity to remove
            from every wavelength array before regridding or SYSREM
        edge_trim_widths_A: Explicit accepted left/right arm-edge trims in
            Angstroms. No trim is discovered implicitly.

    Returns:
        ``(result, extras)``, where ``result`` is the tuple
        ``(wave, flux, error, jd_utc, snr, exptime, airmass, n_spectra, npix)``.
        ``extras`` always includes any header ``JD-TDB`` values so the caller can
        validate the canonical UTC-to-BJD_TDB conversion. It also contains
        preprocessing artifacts when those steps are enabled.
    """
    if data_dir is None:
        data_dir = config_utils.get_raw_hrs_dir(
            planet=planet_name,
            epoch=observation_epoch,
            mode=data_mode,
        )
    wavelength_frame = str(wavelength_frame).strip().lower()
    if wavelength_frame not in _SUPPORTED_WAVELENGTH_FRAMES:
        raise ValueError(
            f"Unsupported wavelength_frame={wavelength_frame!r}; expected one of "
            f"{sorted(_SUPPORTED_WAVELENGTH_FRAMES)}."
        )

    # Get config for this instrument
    header_keys = get_header_keys()
    col_cfg = get_fits_columns(molecfit=do_molecfit)
    raw_col_cfg = get_fits_columns(molecfit=False)

    # Get file patterns from instrument config
    patterns = get_data_patterns(
        observation_epoch, planet_name, mode=arm, do_molecfit=do_molecfit, data_dir=data_dir
    )

    exposure_selection = select_science_exposures(
        patterns,
        planet_name=planet_name,
        data_mode=data_mode,
        observation_epoch=observation_epoch,
        arm=arm,
        do_molecfit=do_molecfit,
    )
    exposure_selection = _paired_snr_filtered_selection(
        exposure_selection,
        arm=arm,
        observation_epoch=observation_epoch,
        planet_name=planet_name,
        data_mode=data_mode,
        data_dir=data_dir,
    )
    matched_pattern = exposure_selection.matched_pattern
    if not exposure_selection.discovered_files:
        print(f'No files found for {observation_epoch}_{planet_name} ({arm}) in {data_dir}')
        return None

    spectra_files = [str(path) for path in exposure_selection.usable_files]
    configured_excluded_files = [
        path.name for path in exposure_selection.configured_excluded_files
    ]
    if configured_excluded_files:
        print(
            "Excluded known-bad exposure(s) before preprocessing: "
            + ", ".join(configured_excluded_files)
        )
    snr_quality_excluded_files = [
        path.name for path in exposure_selection.snr_quality_excluded_files
    ]
    if snr_quality_excluded_files:
        print(
            "Excluded paired q_snr < "
            f"{config.HRS_PAIRED_SNR_Q_THRESHOLD:g} exposure(s) before preprocessing: "
            + ", ".join(snr_quality_excluded_files)
        )
    for check in exposure_selection.fits_object_checks:
        if check.accepted:
            continue
        print(
            "Excluded FITS OBJECT mismatch/unverifiable exposure before "
            f"preprocessing: {check.path.name} "
            f"(OBJECT={check.observed_name!r}; reason={check.rejection_reason})."
        )
    excluded_exposure_files = [
        path.name for path in exposure_selection.excluded_files
    ]
    if not spectra_files:
        print(
            f"No usable files remain for {observation_epoch}_{planet_name} "
            f"({arm}) after exposure-level QC exclusions."
        )
        return None

    n_spectra = len(spectra_files)
    input_exposure_files = [Path(path).name for path in spectra_files]
    # Surface which PEPSI product family fed this run so provenance is never
    # ambiguous:
    #   .dxt.*  dual-beam coadd (preferred)
    #   .sxt.*  single-beam coadd (used when dual-beam product unavailable)
    #   .sxs.*  per-readout sub-exposure (last-resort fallback)
    if matched_pattern and ".dxs." in matched_pattern:
        product_grade = "per-readout, dual-beam (.dxs)"
    elif matched_pattern and ".sxs." in matched_pattern:
        product_grade = "sub-exposure (.sxs)"
    elif matched_pattern and ".sxt." in matched_pattern:
        product_grade = "coadded, single-beam (.sxt)"
    else:
        product_grade = "coadded, dual-beam (.dxt)"
    print(f"Found {n_spectra} spectra [{product_grade}]")

    i = 0
    jd, snr_spectra, exptime = np.zeros(n_spectra), np.zeros(n_spectra), np.zeros(n_spectra)
    calendar_midpoint_jd_utc = np.full(n_spectra, np.nan, dtype=float)
    header_bjd_tdb = np.full(n_spectra, np.nan, dtype=float)
    wavelength_velocity_correction_mps = np.zeros(n_spectra, dtype=float)
    wavelength_velocity_components: list[tuple[str, ...]] = [()] * n_spectra
    observer_velocity_removed_mps = np.full(n_spectra, np.nan, dtype=float)
    stellar_velocity_removed_mps = np.full(n_spectra, np.nan, dtype=float)
    instrument_velocity_removed_mps = np.full(n_spectra, np.nan, dtype=float)
    synthetic_molecfit_shift_undone_mps = np.zeros(n_spectra, dtype=float)
    velocity_history_recipe: list[tuple[str, ...]] = [()] * n_spectra
    velocity_history_entries: list[tuple[str, ...]] = [()] * n_spectra
    velocity_frame_parse_error: list[str | None] = [None] * n_spectra
    wavelength_frame_source: list[str] = ["native"] * n_spectra
    airmass = np.zeros(n_spectra)

    for spectrum in spectra_files:
        with fits.open(spectrum) as hdu:
            data = hdu[1].data.copy()
            header = hdu[0].header.copy()

        # Get column names from instrument config
        flux_tag, error_tag = col_cfg["flux"], col_cfg["error"]
        loaded_wave = _wavelength_angstrom(data, col_cfg)
        frame_info: dict[str, object] | None = None
        if do_molecfit:
            if wavelength_frame == "barycentric":
                loaded_wave, raw_path, frame_info = _barycentric_molecfit_wavelength(
                    molecfit_path=spectrum,
                    molecfit_data=data,
                    molecfit_header=header,
                    observation_epoch=observation_epoch,
                    data_dir=data_dir,
                    raw_column_config=raw_col_cfg,
                    molecfit_column_config=col_cfg,
                    jd_header_key=header_keys["jd"],
                )
                if i == 0:
                    print(
                        "Barycentric Molecfit wavelength reconstruction: "
                        f"{frame_info['wavelength_frame_source']} "
                        f"(observer={float(frame_info['observer_velocity_removed_mps']) / 1000.0:+.6f}, "
                        f"stellar={float(frame_info['stellar_velocity_removed_mps']) / 1000.0:+.6f} km/s)."
                    )
            else:
                loaded_wave, raw_path, rest_correction_mps, rest_correction_keys = (
                    _canonical_molecfit_wavelength(
                        molecfit_path=spectrum,
                        molecfit_data=data,
                        molecfit_header=header,
                        observation_epoch=observation_epoch,
                        planet_name=planet_name,
                        arm=arm,
                        data_dir=data_dir,
                        raw_column_config=raw_col_cfg,
                        molecfit_column_config=col_cfg,
                        jd_header_key=header_keys["jd"],
                    )
                )
                wavelength_velocity_correction_mps[i] = rest_correction_mps
                wavelength_velocity_components[i] = rest_correction_keys
                if i == 0:
                    keys = ", ".join(rest_correction_keys) or "no nonzero terms"
                    validation = (
                        f" Raw product '{raw_path.name}' validates the pixel mapping."
                        if raw_path is not None
                        else " Original raw product unavailable; using FITS-header reconstruction."
                    )
                    print(
                        "Molecfit velocity restoration: original PEPSI recipe using "
                        f"{keys} "
                        f"(correction={rest_correction_mps:.3f} m/s)."
                        f"{validation}"
                    )
        else:
            loaded_wave = _air_wavelength_to_vacuum(loaded_wave)
            try:
                frame_info = parse_pepsi_velocity_frame(header)
            except ValueError as exc:
                if wavelength_frame == "barycentric":
                    raise ValueError(f"{Path(spectrum).name}: {exc}") from exc
                velocity_frame_parse_error[i] = str(exc)
            if wavelength_frame == "barycentric" and frame_info is not None:
                loaded_wave = _raw_pipeline_wavelength_to_barycentric(
                    loaded_wave,
                    frame_info,
                )
                frame_info = {
                    **frame_info,
                    "synthetic_molecfit_shift_undone_mps": 0.0,
                    "wavelength_frame_source": "raw_pipeline_history",
                }
                if i == 0:
                    print(
                        "Barycentric raw wavelength reconstruction: FITS HISTORY "
                        f"(observer={float(frame_info['observer_velocity_removed_mps']) / 1000.0:+.6f}, "
                        f"stellar={float(frame_info['stellar_velocity_removed_mps']) / 1000.0:+.6f} km/s)."
                    )

        if frame_info is None and velocity_frame_parse_error[i] is None:
            try:
                frame_info = parse_pepsi_velocity_frame(header)
            except ValueError as exc:
                velocity_frame_parse_error[i] = str(exc)
        if frame_info is not None:
            observer_velocity_removed_mps[i] = float(
                frame_info["observer_velocity_removed_mps"]
            )
            stellar_velocity_removed_mps[i] = float(
                frame_info["stellar_velocity_removed_mps"]
            )
            instrument_velocity_removed_mps[i] = float(
                frame_info["instrument_velocity_removed_mps"]
            )
            synthetic_molecfit_shift_undone_mps[i] = float(
                frame_info.get("synthetic_molecfit_shift_undone_mps", 0.0)
            )
            velocity_history_recipe[i] = tuple(
                str(item) for item in frame_info["velocity_history_recipe"]
            )
            velocity_history_entries[i] = tuple(
                str(item) for item in frame_info["velocity_history_entries"]
            )
            wavelength_frame_source[i] = str(
                frame_info.get(
                    "wavelength_frame_source",
                    "native_molecfit" if do_molecfit else "native_raw_pipeline",
                )
            )
        if stellar_rest_velocity_kms is not None:
            loaded_wave = _shift_wavelength_to_stellar_rest(
                loaded_wave,
                stellar_rest_velocity_kms,
            )

        if i == 0:
            npix = loaded_wave.size
            wave = np.zeros((n_spectra, npix))
            fluxin = np.zeros((n_spectra, npix))
            errorin = np.zeros((n_spectra, npix))

        # Handle inconsistent pixel numbers
        npixhere = loaded_wave.size
        if npixhere >= npix:
            wave[i, :] = loaded_wave[0:npix]
            fluxin[i, :] = data[flux_tag][0:npix]
            errorin[i, :] = data[error_tag][0:npix]
        else:
            wave[i, 0:npixhere] = loaded_wave
            fluxin[i, 0:npixhere] = data[flux_tag]
            errorin[i, 0:npixhere] = data[error_tag]

        # Raw files have variance, need sqrt for uncertainty
        if col_cfg.get("error") == "Var":
            errorin[i, :] = np.sqrt(errorin[i, :])

        snr_key = header_keys["snr"]
        snr_spectra[i] = header.get(snr_key, np.nan)

        exptime_key = header_keys["exptime"]
        exptime_val = header[exptime_key]
        if isinstance(exptime_val, str):
            exptime_strings = exptime_val.split(':')
            exptime[i] = (
                float(exptime_strings[0]) * 3600.
                + float(exptime_strings[1]) * 60.
                + float(exptime_strings[2])
            )
        else:
            exptime[i] = float(exptime_val)

        input_time_key = header_keys["jd"]
        declared_time_scale = str(
            header.get(header_keys.get("timesys", "TIMESYS"), "")
        ).strip().lower()
        if declared_time_scale and declared_time_scale not in {"ut", "utc"}:
            raise ValueError(
                f"PEPSI {input_time_key} is configured as UTC, but "
                f"'{Path(spectrum).name}' declares TIMESYS={declared_time_scale!r}."
            )
        jd[i] = float(header[input_time_key])  # UTC mid-exposure by PEPSI contract
        calendar_midpoint = _calendar_mid_exposure_jd_utc(header, exptime[i])
        if calendar_midpoint is not None:
            calendar_midpoint_jd_utc[i] = calendar_midpoint
        header_bjd_tdb[i] = float(
            header.get(header_keys.get("bjd_tdb", "JD-TDB"), np.nan)
        )

        airmass_key = header_keys["airmass"]
        airmass[i] = header[airmass_key]

        i += 1

    input_time_provenance: dict[str, object] = {
        "source": input_time_key,
        "reconstructed": False,
        "reason": None,
        "original_keyword": input_time_key,
        "scale": PEPSI_INPUT_TIME_SCALE,
        "reference": "topocenter",
        "exposure_reference": "midpoint",
    }
    if n_spectra > 1 and float(np.ptp(jd)) == 0.0:
        constant_header_jd = float(jd[0])
        complete_calendar_times = bool(
            np.all(np.isfinite(calendar_midpoint_jd_utc))
        )
        varying_calendar_times = bool(
            complete_calendar_times
            and float(np.ptp(calendar_midpoint_jd_utc)) > 0.0
        )
        if not varying_calendar_times:
            missing_count = int(
                np.count_nonzero(~np.isfinite(calendar_midpoint_jd_utc))
            )
            raise ValueError(
                f"All {n_spectra} exposures have identical {input_time_key}="
                f"{jd[0]:.9f}, and a varying UTC midpoint cannot be reconstructed "
                "from DATE-OBS + UT-OBS + EXPTIME/2 "
                f"({missing_count} incomplete or unparseable calendar timestamps)."
            )
        jd = calendar_midpoint_jd_utc.copy()
        reconstructed_source = "DATE-OBS+UT-OBS+EXPTIME/2"
        input_time_provenance.update(
            {
                "source": reconstructed_source,
                "reconstructed": True,
                "reason": f"constant_{input_time_key}_across_multiple_exposures",
                "date_keyword": "DATE-OBS",
                "time_keyword": "UT-OBS",
                "exposure_keyword": header_keys["exptime"],
            }
        )
        print(
            f"Warning: all {n_spectra} exposures share {input_time_key}="
            f"{constant_header_jd:.9f}; using UTC midpoints reconstructed "
            f"from {reconstructed_source}."
        )

    n_missing_snr = int(np.count_nonzero(~np.isfinite(snr_spectra)))
    if n_missing_snr:
        print(
            f"Warning: {n_missing_snr}/{n_spectra} spectra have no finite "
            f"{header_keys['snr']} header; saving those snr.npy entries as NaN."
        )

    # ====================
    # Preprocessing pipeline
    # ====================

    # Step 1: Regrid to common wavelength (before sorting, fixes sub-pixel drift)
    if regrid:
        print("Regridding spectra to common wavelength grid...")
        wave_common, fluxin, errorin = regrid_to_common_wavelength(wave, fluxin, errorin)
        original_npix = wave_common.size
        wave_common, fluxin, errorin, _valid_regrid_columns = keep_finite_common_columns(
            wave_common,
            fluxin,
            errorin,
        )
        dropped_npix = original_npix - wave_common.size
        if dropped_npix:
            print(
                f"  Dropped {dropped_npix} unsupported common-grid columns after regridding."
            )
        # Replace 2D wave array with 1D common grid (broadcast back for compatibility)
        npix = wave_common.size
        wave = np.broadcast_to(wave_common[np.newaxis, :], (n_spectra, npix)).copy()

    # Step 2: Sort by time
    obs_order = np.argsort(jd)
    jd = jd[obs_order]
    header_bjd_tdb = header_bjd_tdb[obs_order]
    wavelength_velocity_correction_mps = wavelength_velocity_correction_mps[obs_order]
    wavelength_velocity_components = [
        wavelength_velocity_components[index] for index in obs_order
    ]
    observer_velocity_removed_mps = observer_velocity_removed_mps[obs_order]
    stellar_velocity_removed_mps = stellar_velocity_removed_mps[obs_order]
    instrument_velocity_removed_mps = instrument_velocity_removed_mps[obs_order]
    synthetic_molecfit_shift_undone_mps = (
        synthetic_molecfit_shift_undone_mps[obs_order]
    )
    velocity_history_recipe = [velocity_history_recipe[index] for index in obs_order]
    velocity_history_entries = [velocity_history_entries[index] for index in obs_order]
    velocity_frame_parse_error = [
        velocity_frame_parse_error[index] for index in obs_order
    ]
    wavelength_frame_source = [wavelength_frame_source[index] for index in obs_order]
    input_exposure_files = [input_exposure_files[index] for index in obs_order]
    snr_spectra = snr_spectra[obs_order]
    exptime = exptime[obs_order]
    airmass = airmass[obs_order]
    wave = wave[obs_order, :]
    fluxin = fluxin[obs_order, :]
    errorin = errorin[obs_order, :]

    # Step 3: Subtract median spectrum (stellar line removal)
    median_flux = None
    if subtract_median:
        print("Subtracting median spectrum (stellar line removal)...")
        fluxin, errorin, median_flux = subtract_median_spectrum(fluxin, errorin)

    pre_sysrem_flux = fluxin.copy()
    pre_sysrem_error = errorin.copy()

    # Step 4: SYSREM systematics removal
    U_sysrem = None
    no_tellurics = None
    n_systematics_used = None
    sysrem_diagnostics = None
    if run_sysrem:
        sysrem_max_systematics = get_sysrem_max_systematics(arm)
        sysrem_min_systematics = get_sysrem_min_systematics(arm)
        print(
            f"Running SYSREM (adaptive) with min_systematics={sysrem_min_systematics}, "
            f"max_systematics={sysrem_max_systematics}, "
            f"stop_tol={sysrem_stop_tol:.1e}..."
        )
        wave_1d = wave[0, :]  # Use first spectrum's wavelength grid
        sysrem_result = do_sysrem(
            wave_1d,
            fluxin,
            errorin,
            arm,
            airmass,
            do_molecfit=do_molecfit,
            stop_delta_stddev=sysrem_stop_tol,
            return_diagnostics=True,
            planet_name=planet_name,
            data_mode=data_mode,
            observation_epoch=observation_epoch,
            edge_trim_widths_A=edge_trim_widths_A,
        )
        fluxin, errorin, U_sysrem, no_tellurics, sysrem_diagnostics = sysrem_result
        n_systematics_used = [int(np.sum(np.isfinite(U_sysrem[0, :, i]))) for i in range(U_sysrem.shape[2])]

    result = (wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix)

    final_wavelength_frame = (
        'stellar_rest' if stellar_rest_velocity_kms is not None else wavelength_frame
    )
    extras = {
        'header_bjd_tdb': header_bjd_tdb,
        'input_time_keyword': input_time_provenance['source'],
        'input_time_provenance': input_time_provenance,
        'input_time_scale': PEPSI_INPUT_TIME_SCALE,
        'input_time_reference': 'topocenter',
        'input_exposure_reference': 'midpoint',
        'input_exposure_files': input_exposure_files,
        'excluded_exposure_files': excluded_exposure_files,
        'science_exposure_selection': exposure_selection,
        'science_exposure_selection_metadata': exposure_selection.metadata(),
        'header_bjd_tdb_keyword': header_keys.get('bjd_tdb', 'JD-TDB'),
        'wavelength_medium': 'vacuum',
        'wavelength_frame': final_wavelength_frame,
        'wavelength_intermediate_frame': wavelength_frame,
        'wavelength_frame_source': wavelength_frame_source,
        'observer_velocity_removed_kms': observer_velocity_removed_mps / 1000.0,
        'stellar_velocity_removed_kms': stellar_velocity_removed_mps / 1000.0,
        'instrument_velocity_removed_kms': instrument_velocity_removed_mps / 1000.0,
        'synthetic_molecfit_shift_undone_kms': (
            synthetic_molecfit_shift_undone_mps / 1000.0
        ),
        'velocity_history_recipe': velocity_history_recipe,
        'velocity_history_entries': velocity_history_entries,
        'velocity_frame_parse_error': velocity_frame_parse_error,
        'stellar_rest_velocity_kms': stellar_rest_velocity_kms,
        'wavelength_velocity_recipe': (
            'barycentric_from_pepsi_history'
            if wavelength_frame == 'barycentric'
            else ('original_pepsi_molecfit' if do_molecfit else 'none')
        ),
        'wavelength_velocity_correction_mps': wavelength_velocity_correction_mps,
        'wavelength_velocity_components': wavelength_velocity_components,
        'median_flux': median_flux,
        'pre_sysrem_flux': pre_sysrem_flux,
        'pre_sysrem_error': pre_sysrem_error,
        'U_sysrem': U_sysrem,
        'no_tellurics': no_tellurics,
        'n_systematics_used': n_systematics_used,
    }
    if sysrem_diagnostics is not None:
        extras.update(sysrem_diagnostics)
    return result, extras


def get_bjd_tdb(
    jd_utc: np.ndarray,
    RA: str,
    Dec: str,
    *,
    header_bjd_tdb: np.ndarray | None = None,
    input_time_keyword: str = "JD-OBS",
    validation_tolerance_seconds: float = PEPSI_BJD_TDB_VALIDATION_TOLERANCE_SECONDS,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Convert topocentric UTC Julian dates to barycentric BJD_TDB.

    PEPSI ``JD-OBS`` values normally provide UTC mid-exposure times. The loader
    may instead pass strictly reconstructed calendar midpoints when ``JD-OBS``
    is constant across multiple exposures. When a product also supplies
    ``JD-TDB``, it is treated as an independent validation value, not as an
    alternate input path.
    """
    jd_utc = np.asarray(jd_utc, dtype=float)
    if not np.all(np.isfinite(jd_utc)):
        raise ValueError("UTC Julian dates must all be finite.")
    if validation_tolerance_seconds <= 0.0:
        raise ValueError("BJD_TDB validation tolerance must be positive.")

    observatory_location = EarthLocation.of_site(OBSERVATORY)
    observed_times = Time(
        jd_utc,
        format='jd',
        scale=PEPSI_INPUT_TIME_SCALE,
        location=observatory_location,
    )
    coordinates = SkyCoord(RA + ' ' + Dec, frame='icrs', unit=(u.hourangle, u.deg))
    ltt_bary = observed_times.light_travel_time(coordinates)
    bjd_tdb = np.asarray((observed_times.tdb + ltt_bary).jd, dtype=float)

    diagnostics: dict[str, float | int | str | None] = {
        "input_keyword": str(input_time_keyword),
        "input_scale": PEPSI_INPUT_TIME_SCALE,
        "input_reference": "topocenter",
        "input_exposure_reference": "midpoint",
        "canonical_time_standard": "BJD_TDB",
        "canonical_scale": "tdb",
        "canonical_reference": "barycenter",
        "header_validation_keyword": "JD-TDB",
        "header_validation_tolerance_seconds": float(validation_tolerance_seconds),
        "header_validation_count": 0,
        "header_validation_max_abs_residual_seconds": None,
    }
    if header_bjd_tdb is not None:
        header_bjd_tdb = np.asarray(header_bjd_tdb, dtype=float)
        if header_bjd_tdb.shape != bjd_tdb.shape:
            raise ValueError(
                "Header JD-TDB shape does not match the input UTC time shape: "
                f"{header_bjd_tdb.shape} != {bjd_tdb.shape}."
            )
        validated = np.isfinite(header_bjd_tdb)
        n_validated = int(np.count_nonzero(validated))
        diagnostics["header_validation_count"] = n_validated
        if n_validated:
            residual_seconds = (
                bjd_tdb[validated] - header_bjd_tdb[validated]
            ) * 86400.0
            max_abs_residual = float(np.max(np.abs(residual_seconds)))
            diagnostics["header_validation_max_abs_residual_seconds"] = max_abs_residual
            if max_abs_residual > validation_tolerance_seconds:
                raise ValueError(
                    "Computed BJD_TDB disagrees with the PEPSI JD-TDB header: "
                    f"max |residual|={max_abs_residual:.6f} s exceeds "
                    f"{validation_tolerance_seconds:.6f} s."
                )
            print(
                f"Validated computed BJD_TDB against JD-TDB for {n_validated} "
                f"exposures (max |residual|={max_abs_residual:.6f} s)."
            )

    if return_diagnostics:
        return bjd_tdb, diagnostics
    return bjd_tdb


def get_ephemeris_epoch_bjd_tdb(
    epoch: float,
    epoch_scale: str,
    epoch_reference: str,
) -> float:
    """Normalize a barycentric ephemeris epoch to the TDB timescale."""
    reference = str(epoch_reference).strip().lower().replace("_", "-")
    if reference not in {"barycenter", "barycentric", "solar-system-barycenter"}:
        raise ValueError(
            "Ephemeris epoch_reference must explicitly identify the barycenter; "
            f"got {epoch_reference!r}."
        )
    scale = str(epoch_scale).strip().lower()
    if scale not in {"tdb", "tt", "tai", "utc"}:
        raise ValueError(
            "Unsupported or missing ephemeris epoch_scale. Expected one of "
            f"tdb/tt/tai/utc, got {epoch_scale!r}."
        )
    epoch_time = Time(float(epoch), format="jd", scale=scale)
    return float(epoch_time.tdb.jd)


def get_orbital_phase(
    bjd_tdb: np.ndarray,
    epoch_bjd_tdb: float,
    period: float,
) -> np.ndarray:
    """Calculate phase from canonical BJD_TDB observation and epoch values."""
    bjd_tdb = np.asarray(bjd_tdb, dtype=float)
    if not np.all(np.isfinite(bjd_tdb)):
        raise ValueError("BJD_TDB values must all be finite.")
    if not np.isfinite(epoch_bjd_tdb):
        raise ValueError("The BJD_TDB ephemeris epoch must be finite.")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("The orbital period must be finite and positive.")

    orbital_phase = (bjd_tdb - epoch_bjd_tdb) / period
    orbital_phase -= np.round(np.mean(orbital_phase))

    return orbital_phase


def build_out_of_transit_residuals(
    flux: np.ndarray,
    error: np.ndarray,
    out_transit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide every exposure by a shared out-of-transit master and subtract one."""
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    out_transit = np.asarray(out_transit, dtype=bool)
    n_out = int(np.count_nonzero(out_transit))

    master_out = np.median(flux[out_transit], axis=0)
    # Large-sample variance of the median is pi/2 times the variance of the
    # mean. This remains a diagonal approximation to the shared-master error.
    master_error = (
        np.sqrt(np.pi / 2.0)
        * np.sqrt(np.sum(error[out_transit] ** 2, axis=0))
        / n_out
    )
    valid_master = (
        np.isfinite(master_out)
        & (master_out != 0.0)
        & np.isfinite(master_error)
        & (master_error >= 0.0)
    )
    residual = np.full_like(flux, np.nan, dtype=float)
    residual_error = np.full_like(error, np.inf, dtype=float)
    residual[:, valid_master] = (
        flux[:, valid_master] / master_out[None, valid_master] - 1.0
    )
    residual_error[:, valid_master] = np.sqrt(
        (error[:, valid_master] / master_out[None, valid_master]) ** 2
        + (
            flux[:, valid_master]
            * master_error[None, valid_master]
            / master_out[None, valid_master] ** 2
        )
        ** 2
    )
    return residual, residual_error


def calculate_transmission_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    jd: np.ndarray,
    transit_params: dict,
    RA: str,
    Dec: str,
    Kp_kms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a planet-rest-frame differential transmission spectrum.

    The returned spectrum is the inverse-variance coadd of active-transit
    fractional flux residuals (T23 normally, T14 for a grazing transit),

        flux / median(flux_out_of_transit) - 1,

    after shifting each exposure into the planet rest frame using the fixed
    orbital semi-amplitude ``Kp_kms``. Absorption is therefore negative.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    jd = np.asarray(jd, dtype=float)

    wave_grid = wave

    order = np.argsort(wave_grid)
    wave_grid = wave_grid[order]
    flux = flux[:, order]
    error = error[:, order]

    bjd_tdb = get_bjd_tdb(jd, RA, Dec)
    epoch_bjd_tdb = get_ephemeris_epoch_bjd_tdb(
        transit_params['T0'],
        transit_params.get('T0_scale', transit_params.get('epoch_scale')),
        transit_params.get('T0_reference', transit_params.get('epoch_reference')),
    )
    orbital_phase = get_orbital_phase(
        bjd_tdb,
        epoch_bjd_tdb,
        transit_params['period'],
    )

    contacts = compute_contact_phases(transit_params)
    active_transit = active_transit_mask(orbital_phase, transit_params)
    active_interval = (
        "T14_grazing"
        if bool(transit_params.get("grazing_transit", False))
        else "T23"
    )
    out_transit = (orbital_phase < contacts["T1"]) | (orbital_phase > contacts["T4"])

    print(f"Active transit ({active_interval}): {np.sum(active_transit)} exposures")
    print(f"Out-of-transit: {np.sum(out_transit)} exposures")
    print(f"Orbital phase range: {orbital_phase.min():.4f} to {orbital_phase.max():.4f}")

    residual, residual_error = build_out_of_transit_residuals(
        flux,
        error,
        out_transit,
    )

    active_phase = orbital_phase[active_transit]
    planet_velocity = planet_radial_velocity_kms(
        active_phase,
        kp_kms=float(Kp_kms),
        eccentricity=float(transit_params.get("eccentricity", 0.0)),
        omega_deg=transit_params.get("omega"),
    )
    active_residual = residual[active_transit]
    active_error = residual_error[active_transit]

    shifted_residual = np.full_like(active_residual, np.nan)
    shifted_error = np.full_like(active_error, np.inf)
    for i, velocity_kms in enumerate(planet_velocity):
        beta = velocity_kms / _SPEED_OF_LIGHT_KMS
        doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
        observed_wavelength = wave_grid * doppler_factor
        shifted_residual[i] = np.interp(
            observed_wavelength,
            wave_grid,
            active_residual[i],
            left=np.nan,
            right=np.nan,
        )
        shifted_error[i] = np.interp(
            observed_wavelength,
            wave_grid,
            active_error[i],
            left=np.inf,
            right=np.inf,
        )

    valid_shifted = (
        np.isfinite(shifted_residual)
        & np.isfinite(shifted_error)
        & (shifted_error > 0.0)
    )
    full_coverage = np.all(valid_shifted, axis=0)
    if not np.any(full_coverage):
        raise ValueError(
            f"No wavelength columns retain coverage from every {active_interval} exposure "
            "after shifting into the planet rest frame."
        )

    wave_grid = wave_grid[full_coverage]
    shifted_residual = shifted_residual[:, full_coverage]
    shifted_error = shifted_error[:, full_coverage]
    weights = 1.0 / shifted_error**2
    weight_sum = np.sum(weights, axis=0)
    spectrum = np.sum(weights * shifted_residual, axis=0) / weight_sum
    spectrum_error = np.sqrt(1.0 / weight_sum)

    return (
        wave_grid,
        spectrum,
        spectrum_error,
        orbital_phase,
        active_transit,
        out_transit,
    )


def bin_spectrum(
    wave: np.ndarray, flux: np.ndarray, error: np.ndarray, bin_size: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance bin a 1D spectrum."""
    npix = len(wave)
    n_bins = npix // bin_size

    wave_binned = np.zeros(n_bins)
    flux_binned = np.zeros(n_bins)
    error_binned = np.zeros(n_bins)

    for i in range(n_bins):
        idx_start = i * bin_size
        idx_end = (i + 1) * bin_size

        wave_chunk = wave[idx_start:idx_end]
        flux_chunk = flux[idx_start:idx_end]
        error_chunk = error[idx_start:idx_end]
        valid = (
            np.isfinite(wave_chunk)
            & np.isfinite(flux_chunk)
            & np.isfinite(error_chunk)
            & (error_chunk > 0.0)
        )
        if not np.any(valid):
            wave_binned[i] = np.nan
            flux_binned[i] = np.nan
            error_binned[i] = np.inf
            continue
        weights = 1.0 / error_chunk[valid] ** 2
        weight_sum = np.sum(weights)
        wave_binned[i] = np.sum(weights * wave_chunk[valid]) / weight_sum
        flux_binned[i] = np.sum(weights * flux_chunk[valid]) / weight_sum
        error_binned[i] = np.sqrt(1.0 / weight_sum)

    return wave_binned, flux_binned, error_binned


def collapsed_transmission_dir(
    *,
    planet: str,
    epoch: str,
    arm: str,
) -> Path:
    """Return the standard directory for a collapsed transmission product."""
    return config_utils.get_collapsed_transmission_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
    )


def _load_transmission_collapse_source(
    data_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    expected = {
        "wavelength": data_dir / "wavelength.npy",
        "data": data_dir / "data.npy",
        "sigma": data_dir / "sigma.npy",
        "phase": data_dir / "phase.npy",
    }
    missing = [path.name for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is missing transmission collapse-source files: "
            f"{', '.join(missing)}."
        )
    metadata_path = data_dir / "timeseries_prep.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{data_dir} is missing timeseries_prep.json."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {metadata_path}: {exc}") from exc
    if metadata.get("product_kind") != "collapse-source":
        raise ValueError(
            f"{data_dir} is not a collapse-source product."
        )
    if metadata.get("phase_bin") != "all":
        raise ValueError(
            f"{data_dir} must be prepared with phase_bin='all'."
        )
    if metadata.get("out_of_transit_master_division") is not True:
        raise ValueError(
            f"{data_dir} was not prepared using out-of-transit master division."
        )

    wavelength = np.asarray(np.load(expected["wavelength"]), dtype=float)
    data = np.asarray(np.load(expected["data"]), dtype=float)
    sigma = np.asarray(np.load(expected["sigma"]), dtype=float)
    phase = np.asarray(np.load(expected["phase"]), dtype=float)
    return wavelength, data, sigma, phase, metadata


def _load_required_lsd_shadow_source_model(
    data_dir: Path,
    *,
    metadata: dict,
    wavelength: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Load the mandatory shared-basis LSD model for a collapse source."""

    shadow = metadata.get("fixed_doppler_shadow")
    if not isinstance(shadow, dict) or not bool(shadow.get("enabled", False)):
        raise ValueError(
            f"{data_dir} has no enabled shared-basis LSD Doppler shadow. Run "
            "python -m spectroscopy.doppler_shadow after preparing both source products."
        )
    if int(shadow.get("schema_version", -1)) != FIXED_LSD_SHADOW_SCHEMA_VERSION:
        raise ValueError(
            f"{data_dir} has unsupported fixed LSD shadow schema "
            f"{shadow.get('schema_version')!r}; expected "
            f"{FIXED_LSD_SHADOW_SCHEMA_VERSION}."
        )
    if str(shadow.get("method")) != FIXED_LSD_SHADOW_METHOD:
        raise ValueError(
            f"{data_dir} does not use the required {FIXED_LSD_SHADOW_METHOD} method."
        )
    model_name = str(shadow.get("source_model_file", ""))
    model_path = (data_dir / model_name).resolve()
    if not model_name or model_path.parent != data_dir.resolve():
        raise ValueError("The fixed LSD source model must live in its source directory.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Required fixed LSD source model is missing: {model_path}")
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha256 != str(shadow.get("source_model_sha256", "")):
        raise ValueError(f"Fixed LSD source-model hash mismatch for {model_path}.")
    model = np.asarray(np.load(model_path), dtype=float)
    expected_shape = (phase.size, wavelength.size)
    if model.shape != expected_shape or np.any(~np.isfinite(model)):
        raise ValueError(
            f"{model_path} must be finite with shape {expected_shape}; got {model.shape}."
        )
    required_provenance = (
        "artifact_file",
        "artifact_sha256",
        "artifact_metadata_file",
        "template_sha256",
    )
    missing = [name for name in required_provenance if not shadow.get(name)]
    if missing:
        raise ValueError(
            f"{data_dir} fixed LSD provenance is missing: {', '.join(missing)}."
        )
    return model, dict(shadow)


def collapse_transmission_epoch_arm(
    *,
    planet: str,
    ephemeris: str,
    shadow_source: str = "Recommended",
    epoch: str,
    arm: str,
    kp_kms: float,
    bin_size: int = 1,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Build one SYSREM-aware, active-transit 1D spectrum."""
    from dataio.collapse_emission_timeseries_to_1d import (
        COLLAPSE_COVERAGE_POLICY,
        apply_emission_collapse_operator,
        build_emission_collapse_operator,
        load_frozen_sysrem_arrays,
    )

    params = config_utils.resolve_parameter_domains(
        planet=planet,
        timing_source=ephemeris,
        shadow_source=shadow_source,
    )
    if source_dir is None:
        source_dir = config_utils.get_collapse_source_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode="transmission",
        )
    else:
        source_dir = Path(source_dir)
    if output_dir is None:
        output_dir = collapsed_transmission_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
        )
    else:
        output_dir = Path(output_dir)
    wavelength, data, sigma, phase, source_metadata = (
        _load_transmission_collapse_source(source_dir)
    )
    fixed_source_model, fixed_shadow_metadata = (
        _load_required_lsd_shadow_source_model(
            source_dir,
            metadata=source_metadata,
            wavelength=wavelength,
            phase=phase,
        )
    )
    active_transit = active_transit_mask(phase, params)
    active_interval = (
        "T14_grazing" if bool(params.get("grazing_transit", False)) else "T23"
    )
    selected_indices = np.flatnonzero(active_transit)
    if selected_indices.size == 0:
        raise ValueError(
            f"No exposures fall inside the active {active_interval} interval."
        )

    selected_phase = phase[selected_indices]
    selected_sigma = sigma[selected_indices]
    fixed = build_emission_collapse_operator(
        wavelength,
        selected_sigma,
        selected_phase,
        kp_kms=kp_kms,
        eccentricity=float(params.get("eccentricity", 0.0)),
        omega_deg=params.get("omega"),
        bin_size=bin_size,
    )
    wavelength_1d, spectrum_1d, uncertainty_1d = (
        apply_emission_collapse_operator(
            data[selected_indices],
            fixed,
        )
    )
    frozen_sysrem = load_frozen_sysrem_arrays(source_dir)
    if source_metadata.get("run_sysrem") is True and frozen_sysrem is None:
        raise FileNotFoundError(
            f"{source_dir} declares run_sysrem=true but U_sysrem.npz is missing."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "wavelength_transmission.npy", wavelength_1d)
    np.save(output_dir / "spectrum_transmission.npy", spectrum_1d)
    np.save(
        output_dir / "uncertainty_transmission.npy",
        uncertainty_1d,
    )
    operator_name = "transmission_collapse_operator.npz"
    np.savez_compressed(
        output_dir / operator_name,
        schema_version=np.asarray(
            TRANSMISSION_COLLAPSE_SCHEMA_VERSION,
            dtype=np.int32,
        ),
        source_wavelength=wavelength,
        source_phase=phase,
        fixed_source_model=fixed_source_model,
        active_exposure_mask=active_transit.astype(float),
        active_transit_interval=np.asarray(active_interval),
        selected_exposure_indices=selected_indices.astype(np.int32),
        shift_left_indices=fixed["shift_left_indices"],
        shift_fractions=fixed["shift_fractions"],
        coadd_weights=fixed["coadd_weights"],
        bin_indices=fixed["bin_indices"],
        bin_weights=fixed["bin_weights"],
        output_wavelength=fixed["output_wavelength"],
        covered_source_indices=fixed["covered_source_indices"],
        n_source_wavelengths=fixed["n_source_wavelengths"],
        n_covered_wavelengths=fixed["n_covered_wavelengths"],
        n_dropped_out_of_bounds=fixed["n_dropped_out_of_bounds"],
        n_dropped_gap_crossing=fixed["n_dropped_gap_crossing"],
        max_native_gap_factor=fixed["max_native_gap_factor"],
        max_native_gap_angstrom=fixed["max_native_gap_angstrom"],
        coverage_policy=np.asarray(COLLAPSE_COVERAGE_POLICY),
        kp_reference_kms=np.asarray(kp_kms, dtype=float),
        planet_velocity_kms=fixed["planet_velocity_kms"],
        eccentricity=np.asarray(params.get("eccentricity", 0.0), dtype=float),
        omega_planet_deg=np.asarray(params.get("omega", np.nan), dtype=float),
        velocity_offset_reference_kms=np.asarray(0.0, dtype=float),
        has_sysrem=np.asarray(frozen_sysrem is not None, dtype=bool),
        **({} if frozen_sysrem is None else frozen_sysrem),
    )
    metadata = {
        "schema_version": TRANSMISSION_COLLAPSE_SCHEMA_VERSION,
        "product_kind": "collapsed_transmission_spectrum",
        "observable_kind": "continuum_removed_negative_transmission_flux",
        "model_preprocessing": (
            "fixed_shared_basis_lsd_shadow_then_"
            f"{active_interval.lower()}_visibility_then_frozen_per_pixel_sysrem_"
            "then_planet_frame_"
            "inverse_variance_coadd_then_subtract_inverse_variance_"
            "weighted_constant"
        ),
        "active_transit_interval": active_interval,
        "planet": planet,
        "ephemeris": ephemeris,
        "shadow_source": shadow_source,
        "parameter_resolution": params.get("parameter_resolution"),
        "epoch": epoch,
        "arm": arm,
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
        "source_data_dir": str(source_dir),
        "source_run_sysrem": source_metadata.get("run_sysrem"),
        "fixed_doppler_shadow": fixed_shadow_metadata,
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
        "collapse_operator_file": operator_name,
        "wavelength_coverage_policy": COLLAPSE_COVERAGE_POLICY,
        "n_source_wavelengths": int(wavelength.size),
        "n_covered_unbinned_wavelengths": int(
            fixed["n_covered_wavelengths"]
        ),
        "n_dropped_wavelengths": int(
            wavelength.size - int(fixed["n_covered_wavelengths"])
        ),
        "n_dropped_out_of_bounds": int(
            fixed["n_dropped_out_of_bounds"]
        ),
        "n_dropped_gap_crossing": int(
            fixed["n_dropped_gap_crossing"]
        ),
        "retained_unbinned_wavelength_fraction": float(
            int(fixed["n_covered_wavelengths"]) / wavelength.size
        ),
        "max_native_gap_factor": float(fixed["max_native_gap_factor"]),
        "max_native_gap_angstrom": float(
            fixed["max_native_gap_angstrom"]
        ),
        "shift_fraction_min": float(np.min(fixed["shift_fractions"])),
        "shift_fraction_max": float(np.max(fixed["shift_fractions"])),
        "n_output_wavelengths": int(wavelength_1d.size),
    }
    (output_dir / "collapse_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse dedicated transmission source cubes into SYSREM-aware "
            "active-transit 1D spectra."
        )
    )
    parser.add_argument("--planet", required=True)
    parser.add_argument("--ephemeris", default=EPHEMERIS)
    parser.add_argument("--shadow-source", default="Recommended")
    parser.add_argument("--epoch", nargs="+", required=True)
    parser.add_argument(
        "--arm",
        choices=["red", "blue", "full"],
        default=config.DEFAULT_DATA_ARM,
    )
    parser.add_argument("--kp-kms", type=float, default=None)
    parser.add_argument("--bin-size", type=int, default=1)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    params = config_utils.resolve_parameter_domains(
        planet=args.planet,
        timing_source=args.ephemeris,
        shadow_source=args.shadow_source,
    )
    kp_kms = params.get("Kp") if args.kp_kms is None else args.kp_kms
    arms = config.FULL_ARM_MEMBERS if args.arm == "full" else (args.arm,)
    for epoch in args.epoch:
        for arm in arms:
            result = collapse_transmission_epoch_arm(
                planet=args.planet,
                ephemeris=args.ephemeris,
                shadow_source=args.shadow_source,
                epoch=str(epoch),
                arm=arm,
                kp_kms=float(kp_kms),
                bin_size=int(args.bin_size),
            )
            print(
                f"READY {epoch}/{arm}/{result['active_transit_interval']}: "
                f"{result['n_selected_exposures']} exposures, "
                f"{result['n_output_wavelengths']} wavelengths"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
