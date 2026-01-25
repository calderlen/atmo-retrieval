#!/usr/bin/env python
"""Prepare spectroscopic data for retrieval.

This script:
1. Loads raw FITS files from the configured instrument
2. Calculates transmission spectrum (in-transit / out-of-transit)
3. Bins to desired resolution
4. Saves wavelength, spectrum, and uncertainty as .npy files

Usage:
    python preprocess.py --epoch 20250601 --planet KELT-20b --arm red
"""

import argparse
import os
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

from config.instrument_config import (
    OBSERVATORY,
    get_data_patterns,
    get_header_keys,
    get_fits_columns,
    get_resolution,
    TELLURIC_REGIONS,
)
from config.planets_config import PHASE_BINS


def compute_contact_phases(params: dict) -> dict[str, float]:
    period = params["period"]
    duration = params["duration"]
    half_dur_phase = (duration / period) / 2

    if "tau" not in params:
        raise ValueError("'tau' (ingress/egress duration) required in params")

    tau = params["tau"]
    if tau != tau:
        raise ValueError("'tau' is NaN")

    tau_phase = tau / period
    return {
        "T1": -half_dur_phase,
        "T2": -half_dur_phase + tau_phase,
        "T3": half_dur_phase - tau_phase,
        "T4": half_dur_phase,
    }


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid all spectra to the first spectrum's wavelength grid.

    Corrects for sub-pixel drift between exposures by interpolating all spectra
    onto a common wavelength grid (the first spectrum's grid).

    Args:
        wave: Wavelength arrays, shape (n_spectra, npix)
        flux: Flux arrays, shape (n_spectra, npix)
        error: Error arrays, shape (n_spectra, npix)

    Returns:
        common_wave: Common wavelength grid, shape (npix,)
        flux: Regridded flux, shape (n_spectra, npix)
        error: Regridded error, shape (n_spectra, npix)
    """
    common_wave = wave[0, :]
    n_spectra = wave.shape[0]

    for i in range(1, n_spectra):
        flux[i, :] = np.interp(common_wave, wave[i, :], flux[i, :])
        error[i, :] = np.interp(common_wave, wave[i, :], error[i, :])

    return common_wave, flux, error


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


def do_sysrem(
    wave: np.ndarray,
    residual_flux: np.ndarray,
    residual_error: np.ndarray,
    arm: str,
    airmass: np.ndarray,
    n_systematics: list[int],
    niter: int = 10,
    do_molecfit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run SYSREM with separate treatment for telluric and non-telluric regions.

    The first systematic is initialized with airmass (physically motivated for
    telluric residuals), subsequent systematics start with unity.

    Args:
        wave: Common wavelength grid, shape (npix,)
        residual_flux: Median-subtracted flux, shape (n_spectra, npix)
        residual_error: Uncertainties, shape (n_spectra, npix)
        arm: Spectrograph arm ('red' or 'blue')
        airmass: Per-exposure airmass values, shape (n_spectra,)
        n_systematics: Number of systematics to remove.
            For red arm: [n_sys_nontelluric, n_sys_telluric]
            For blue arm: [n_sys]
        niter: Number of iterations per systematic (default: 10)
        do_molecfit: If True, mask deep telluric regions (default: True)

    Returns:
        corrected_flux: Systematics-corrected flux, shape (n_spectra, npix)
        corrected_error: Propagated uncertainties, shape (n_spectra, npix)
        U_sysrem: Systematic vectors, shape (n_spectra, max_n_sys, n_chunks)
        no_tellurics: Indices of non-telluric pixels
    """
    n_spectra = residual_flux.shape[0]

    # Work with copies to avoid modifying input
    corrected_flux = residual_flux.copy()
    corrected_error = residual_error.copy()

    # Get telluric regions for this arm
    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})

    if arm == "red":
        # Build telluric mask from wavelength ranges
        telluric_mask = np.zeros(len(wave), dtype=bool)
        for wmin, wmax in telluric_config["telluric"]:
            telluric_mask |= (wave > wmin) & (wave <= wmax)

        no_tellurics = np.where(~telluric_mask)[0]
        has_tellurics = np.where(telluric_mask)[0]

        # Deep mask: set flux to 0 and error to 1 (effectively ignore)
        if do_molecfit:
            for wmin, wmax in telluric_config["deep_mask"]:
                deep_mask = (wave >= wmin) & (wave < wmax)
                corrected_flux[:, deep_mask] = 0.0
                corrected_error[:, deep_mask] = 1.0

        chunks = 2  # Non-telluric and telluric
        chunk_indices = [no_tellurics, has_tellurics]
    else:
        # Blue arm: no tellurics, single chunk
        no_tellurics = np.arange(len(wave))
        chunks = 1
        chunk_indices = [no_tellurics]

    # Ensure n_systematics has correct length
    if len(n_systematics) < chunks:
        n_systematics = n_systematics + [n_systematics[-1]] * (chunks - len(n_systematics))

    max_n_sys = max(n_systematics)
    U_sysrem = np.ones((n_spectra, max_n_sys, chunks))

    for chunk in range(chunks):
        this_one = chunk_indices[chunk]
        if len(this_one) == 0:
            continue

        npixhere = len(this_one)
        n_sys_chunk = n_systematics[chunk]

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
                    saoa = np.where(a != 0, sigma_a / np.abs(a), 0.0)
                    eof = np.where(
                        corrected_flux[:, pix_idx] != 0,
                        corrected_error[:, pix_idx] / np.abs(corrected_flux[:, pix_idx]),
                        0.0
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
                    scoc = np.where(c != 0, sigma_c / np.abs(c), 0.0)
                    eof = np.where(
                        corrected_flux[ep, pix_indices] != 0,
                        corrected_error[ep, pix_indices] / np.abs(corrected_flux[ep, pix_indices]),
                        0.0
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

            # Store systematic vector
            U_sysrem[:, system, chunk] = a

            # Remove systematic: flux -= a * c, propagate errors
            for s in range(npixhere):
                pix_idx = this_one[s]
                syserr = a * c[s]

                # Error on systematic term
                sigma_syserr = np.where(
                    (a != 0) & (c[s] != 0),
                    np.abs(syserr) * np.sqrt((sigma_a / np.abs(a))**2 + (sigma_c[s] / np.abs(c[s]))**2),
                    0.0
                )

                corrected_flux[:, pix_idx] -= syserr
                corrected_error[:, pix_idx] = np.sqrt(
                    corrected_error[:, pix_idx]**2 + sigma_syserr**2
                )

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
    
    if bin_name not in PHASE_BINS:
        raise ValueError(f"Unknown phase bin: {bin_name}. Available: {list(PHASE_BINS.keys())} or 'full'")
    
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
    contacts = get_contact_phases(params)
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

#TODO: figure out what these are and why they are here

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
    "20240114": -112100.0,
    "20220926": -65000.0,
    "20240312": -75200.0,
}


def _get_barycentric_velocity_mps(
    header, velocity_keys: tuple[str, ...] = ("RADVEL", "OBSVEL", "SSBVEL")
) -> tuple[float, list[str]]:
    """Extract barycentric velocity from FITS header."""
    total_velocity = 0.0
    used_keys = []
    for key in velocity_keys:
        if key in header:
            try:
                total_velocity += float(header[key])
                used_keys.append(key)
            except (TypeError, ValueError):
                continue
    return total_velocity, used_keys



def _get_introduced_shift_mps(observation_epoch: str) -> float:
    """Get epoch-specific wavelength shift correction."""
    return _INTRODUCED_SHIFTS_MPS.get(observation_epoch, 0.0)


def get_pepsi_data(
    arm: str,
    observation_epoch: str,
    planet_name: str,
    do_molecfit: bool = True,
    data_dir: str = "input/raw",
    barycentric_correction: bool = False,
    apply_introduced_shift: bool = True,
    regrid: bool = False,
    subtract_median: bool = False,
    run_sysrem: bool = False,
    n_systematics: list[int] | None = None,
    remove_doppler_shadow: bool = False,
    shadow_params: dict | None = None,
) -> tuple[np.ndarray, ...] | None:
    """Load and preprocess spectroscopic data from configured instrument.

    Args:
        arm: Spectrograph arm ('red' or 'blue')
        observation_epoch: Observation date (YYYYMMDD)
        planet_name: Planet name
        do_molecfit: Use molecfit-corrected files
        data_dir: Base data directory
        barycentric_correction: Apply barycentric velocity correction
        apply_introduced_shift: Apply epoch-specific wavelength shift
        regrid: Regrid all spectra to common wavelength grid
        subtract_median: Subtract median spectrum (stellar line removal)
        run_sysrem: Run SYSREM systematics removal
        n_systematics: Number of SYSREM iterations [nontelluric, telluric] or [n]
        remove_doppler_shadow: Remove Doppler shadow (RM effect)
        shadow_params: Dict with 'phase', 'planet_params', 'stellar_params' for shadow removal

    Returns:
        Tuple of arrays: (wave, flux, error, jd, snr, exptime, airmass, n_spectra, npix)
        Additional keys in dict if preprocessing applied
    """
    ckms = 2.9979e5

    # Get config for this instrument
    header_keys = get_header_keys()
    col_cfg = get_fits_columns(molecfit=do_molecfit)

    # Get file patterns from instrument config
    patterns = get_data_patterns(
        observation_epoch, planet_name, mode=arm, do_molecfit=do_molecfit, data_dir=data_dir
    )

    spectra_files = []
    for pattern in patterns:
        spectra_files = glob(pattern, recursive=True)
        if spectra_files:
            break

    if not spectra_files:
        print(f'No files found for {observation_epoch}_{planet_name} ({arm}) in {data_dir}')
        return None

    n_spectra = len(spectra_files)
    print(f"Found {n_spectra} spectra")

    i = 0
    jd, snr_spectra, exptime = np.zeros(n_spectra), np.zeros(n_spectra), np.zeros(n_spectra)
    airmass = np.zeros(n_spectra)

    for spectrum in spectra_files:
        hdu = fits.open(spectrum)
        data, header = hdu[1].data, hdu[0].header

        # Get column names from instrument config
        wave_tag, flux_tag, error_tag = col_cfg["wave"], col_cfg["flux"], col_cfg["error"]

        if i == 0:
            npix = len(data[wave_tag])
            wave = np.zeros((n_spectra, npix))
            fluxin = np.zeros((n_spectra, npix))
            errorin = np.zeros((n_spectra, npix))

        # Handle inconsistent pixel numbers
        npixhere = len(data[wave_tag])
        if npixhere >= npix:
            wave[i, :] = data[wave_tag][0:npix]
            fluxin[i, :] = data[flux_tag][0:npix]
            errorin[i, :] = data[error_tag][0:npix]
        else:
            wave[i, 0:npixhere] = data[wave_tag]
            fluxin[i, 0:npixhere] = data[flux_tag]
            errorin[i, 0:npixhere] = data[error_tag]

        # Raw files have variance, need sqrt for uncertainty
        if col_cfg.get("error") == "Var":
            errorin[i, :] = np.sqrt(errorin[i, :])

        # Convert wavelength units if needed
        if col_cfg["wave_unit"] == "micron":
            wave[i, :] *= 10000.0  # microns -> Angstroms

        introduced_shift = 0.0
        if do_molecfit and apply_introduced_shift:
            introduced_shift = _get_introduced_shift_mps(observation_epoch)

        total_velocity = introduced_shift
        used_keys = []
        if barycentric_correction:
            bary_velocity, used_keys = _get_barycentric_velocity_mps(header)
            total_velocity += bary_velocity

        if introduced_shift != 0.0 or used_keys:
            doppler_shift = 1.0 / (1.0 - total_velocity / 1000.0 / ckms)
            wave[i, :] *= doppler_shift
            if i == 0:
                parts = []
                if introduced_shift != 0.0:
                    parts.append(f"introduced_shift={introduced_shift:.1f} m/s")
                if used_keys:
                    used = ", ".join(used_keys)
                    parts.append(f"{used} sum added")
                detail = "; ".join(parts) if parts else "no components"
                print(f"Velocity correction: {detail} (total={total_velocity:.3f} m/s)")
        elif barycentric_correction and i == 0:
            print("Velocity correction: no RADVEL/OBSVEL/SSBVEL found; skipping")

        jd[i] = header[header_keys["jd"]]  # mid-exposure time
        try:
            snr_spectra[i] = header[header_keys["snr"]]
        except KeyError:
            snr_spectra[i] = np.percentile(fluxin[i, :] / errorin[i, :], 90)

        exptime_val = header[header_keys["exptime"]]
        if isinstance(exptime_val, str):
            exptime_strings = exptime_val.split(':')
            exptime[i] = float(exptime_strings[0]) * 3600. + float(exptime_strings[1]) * 60. + float(exptime_strings[2])
        else:
            exptime[i] = float(exptime_val)
        airmass[i] = header[header_keys["airmass"]]
        hdu.close()
        i += 1

    # ====================
    # Preprocessing pipeline
    # ====================

    # Step 1: Regrid to common wavelength (before sorting, fixes sub-pixel drift)
    if regrid:
        print("Regridding spectra to common wavelength grid...")
        wave_common, fluxin, errorin = regrid_to_common_wavelength(wave, fluxin, errorin)
        # Replace 2D wave array with 1D common grid (broadcast back for compatibility)
        wave = np.broadcast_to(wave_common[np.newaxis, :], (n_spectra, npix)).copy()

    # Step 2: Sort by time
    obs_order = np.argsort(jd)
    jd = jd[obs_order]
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

    # Step 4: SYSREM systematics removal
    U_sysrem = None
    no_tellurics = None
    if run_sysrem:
        if n_systematics is None:
            # Default: 5 iterations for both regions
            n_systematics = [5, 5] if arm == "red" else [5]
        print(f"Running SYSREM with n_systematics={n_systematics}...")
        wave_1d = wave[0, :]  # Use first spectrum's wavelength grid
        fluxin, errorin, U_sysrem, no_tellurics = do_sysrem(
            wave_1d, fluxin, errorin, arm, airmass, n_systematics, do_molecfit=do_molecfit
        )

    # Step 5: Doppler shadow removal
    shadow_model = None
    shadow_fit_info = None
    if remove_doppler_shadow and shadow_params is not None:
        print("Removing Doppler shadow (Rossiter-McLaughlin effect)...")
        from dataio.horus import remove_doppler_shadow as _remove_shadow
        wave_1d = wave[0, :] if wave.ndim == 2 else wave
        fluxin, shadow_model, shadow_fit_info = _remove_shadow(
            fluxin, wave_1d,
            shadow_params['phase'],
            shadow_params['planet_params'],
            shadow_params['stellar_params'],
        )

    result = (wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix)

    # Optionally return preprocessing artifacts as dict
    if subtract_median or run_sysrem or remove_doppler_shadow:
        extras = {
            'median_flux': median_flux,
            'U_sysrem': U_sysrem,
            'no_tellurics': no_tellurics,
            'shadow_model': shadow_model,
            'shadow_fit_info': shadow_fit_info,
        }
        return result, extras

    return result


def get_orbital_phase(
    jd: np.ndarray, epoch: float, period: float, RA: str, Dec: str
) -> np.ndarray:
    """Calculate orbital phase with light travel time correction."""
    observatory_location = EarthLocation.of_site(OBSERVATORY)

    observed_times = Time(jd, format='jd', location=lbt_coordinates)

    coordinates = SkyCoord(RA + ' ' + Dec, frame='icrs', unit=(u.hourangle, u.deg))

    ltt_bary = observed_times.light_travel_time(coordinates)

    bary_times = observed_times + ltt_bary

    orbital_phase = (bary_times.value - epoch) / period
    orbital_phase -= np.round(np.mean(orbital_phase))

    return orbital_phase


def calculate_transmission_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    jd: np.ndarray,
    transit_params: dict,
    RA: str,
    Dec: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate transmission spectrum from in-transit and out-of-transit exposures."""
    # Calculate orbital phase with light travel time correction
    orbital_phase = get_orbital_phase(
        jd,
        transit_params['T0'],
        transit_params['period'],
        RA,
        Dec
    )

    # Identify in-transit and out-of-transit exposures
    half_duration = transit_params['duration'] / 2.0

    in_transit = np.abs(orbital_phase) <= half_duration / transit_params['period']
    out_transit = ~in_transit

    print(f"In-transit: {np.sum(in_transit)} exposures")
    print(f"Out-of-transit: {np.sum(out_transit)} exposures")
    print(f"Orbital phase range: {orbital_phase.min():.4f} to {orbital_phase.max():.4f}")

    if np.sum(in_transit) == 0 or np.sum(out_transit) == 0:
        raise ValueError("Need both in-transit and out-of-transit exposures!")

    # Use median wavelength grid (they should all be similar)
    wave_grid = np.median(wave, axis=0)

    # Calculate transmission spectrum: in-transit / out-of-transit
    flux_in = np.median(flux[in_transit], axis=0)
    flux_out = np.median(flux[out_transit], axis=0)

    # Error propagation
    error_in = np.sqrt(np.sum(error[in_transit]**2, axis=0)) / np.sum(in_transit)
    error_out = np.sqrt(np.sum(error[out_transit]**2, axis=0)) / np.sum(out_transit)

    # Transmission = F_in / F_out (relative depth)
    # This gives the transit depth as a function of wavelength
    transmission = flux_in / flux_out
    transmission_err = transmission * np.sqrt((error_in / flux_in)**2 + (error_out / flux_out)**2)

    return wave_grid, transmission, transmission_err, orbital_phase, in_transit, out_transit


def bin_spectrum(
    wave: np.ndarray, flux: np.ndarray, error: np.ndarray, bin_size: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin spectrum to lower resolution."""
    npix = len(wave)
    n_bins = npix // bin_size

    wave_binned = np.zeros(n_bins)
    flux_binned = np.zeros(n_bins)
    error_binned = np.zeros(n_bins)

    for i in range(n_bins):
        idx_start = i * bin_size
        idx_end = (i + 1) * bin_size

        wave_binned[i] = np.mean(wave[idx_start:idx_end])
        flux_binned[i] = np.mean(flux[idx_start:idx_end])
        error_binned[i] = np.sqrt(np.sum(error[idx_start:idx_end]**2)) / bin_size

    return wave_binned, flux_binned, error_binned
def main():
    parser = argparse.ArgumentParser(description='Prepare PEPSI data for retrieval')
    parser.add_argument('--epoch', type=str, required=True, help='Observation epoch (YYYYMMDD)')
    parser.add_argument('--planet', type=str, default='KELT-20b', help='Planet name')
    parser.add_argument('--arm', type=str, choices=['red', 'blue'], default='red', help='Spectrograph arm')
    parser.add_argument('--molecfit', action='store_true', default=True, help='Use molecfit-corrected data')
    parser.add_argument('--no-molecfit', action='store_false', dest='molecfit', help='Use uncorrected data')
    parser.add_argument('--data-dir', type=str, default='input/raw', help='Raw data directory')
    parser.add_argument('--barycorr', action='store_true', default=False,
                        help='Apply barycentric correction to wavelength grid')
    parser.add_argument('--no-barycorr', action='store_false', dest='barycorr',
                        help='Disable barycentric correction')
    parser.add_argument('--introduced-shift', action='store_true', default=True,
                        help='Apply epoch-specific Molecfit shift (default)')
    parser.add_argument('--no-introduced-shift', action='store_false', dest='introduced_shift',
                        help='Disable epoch-specific Molecfit shift')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: input/spectra/{planet}/{epoch}/{arm})')
    parser.add_argument('--bin-size', type=int, default=50, help='Spectral binning (pixels)')
    parser.add_argument('--t0', type=float, default=None, help='Transit midpoint (BJD_TDB)')
    parser.add_argument('--period', type=float, default=None, help='Orbital period (days)')
    parser.add_argument('--duration', type=float, default=None, help='Transit duration (days)')
    parser.add_argument('--ra', type=str, default=None, help='Right ascension (e.g., "19h38m38.74s")')
    parser.add_argument('--dec', type=str, default=None, help='Declination (e.g., "+31d13m09.12s")')
    parser.add_argument('--calculate-transmission', action='store_true',
                        help='Calculate transmission spectrum (requires --t0, --period, --duration, --ra, --dec)')

    args = parser.parse_args()

    print(f"\nLoading PEPSI {args.arm} arm data for {args.planet} ({args.epoch})...")
    result = get_pepsi_data(
        args.arm,
        args.epoch,
        args.planet,
        args.molecfit,
        args.data_dir,
        barycentric_correction=args.barycorr,
        apply_introduced_shift=args.introduced_shift,
    )

    if result is None:
        print("Failed to load data!")
        return 1

    wave, flux, error, jd, snr, exptime, airmass, n_spectra, npix = result

    print(f"Loaded {n_spectra} spectra with {npix} pixels each")
    print(f"Wavelength range: {wave.min():.1f} - {wave.max():.1f} Angstroms")
    print(f"JD range: {jd.min():.4f} - {jd.max():.4f}")

    # Calculate transmission spectrum if parameters provided
    if args.calculate_transmission:
        if not all([args.t0, args.period, args.duration, args.ra, args.dec]):
            print("ERROR: --calculate-transmission requires --t0, --period, --duration, --ra, and --dec")
            return 1

        print("\nCalculating transmission spectrum...")
        transit_params = {
            'T0': args.t0,
            'period': args.period,
            'duration': args.duration,
        }

        wave_combined, flux_combined, error_combined, orbital_phase, in_transit, out_transit = calculate_transmission_spectrum(
            wave, flux, error, jd, transit_params, args.ra, args.dec
        )

        print(f"Successfully calculated transmission spectrum")
        print(f"Used {np.sum(in_transit)} in-transit and {np.sum(out_transit)} out-of-transit exposures")

    else:
        print("\nWARNING: Not calculating transmission spectrum (use --calculate-transmission)")
        print("Saving median combined spectrum instead...")

        # Use median wavelength and median flux
        wave_combined = np.median(wave, axis=0)
        flux_combined = np.median(flux, axis=0)
        error_combined = np.sqrt(np.sum(error**2, axis=0)) / n_spectra

    # Bin to lower resolution
    if args.bin_size > 1:
        print(f"\nBinning by factor of {args.bin_size}...")
        wave_binned, flux_binned, error_binned = bin_spectrum(
            wave_combined, flux_combined, error_combined, args.bin_size
        )
    else:
        wave_binned, flux_binned, error_binned = wave_combined, flux_combined, error_combined

    # Keep wavelength in Angstroms (no conversion)
    wave_angstrom = wave_binned

    print(f"Final spectrum: {len(wave_angstrom)} points")
    print(f"Wavelength range: {wave_angstrom.min():.1f} - {wave_angstrom.max():.1f} Angstroms")

    # Setup output directory
    if args.output_dir is None:
        planet_dir = args.planet.lower().replace('-', '')
        args.output_dir = f'input/spectra/{planet_dir}/{args.epoch}/{args.arm}'

    os.makedirs(args.output_dir, exist_ok=True)

    # Save as .npy files (wavelength in Angstroms)
    np.save(f"{args.output_dir}/wavelength_transmission.npy", wave_angstrom)
    np.save(f"{args.output_dir}/spectrum_transmission.npy", flux_binned)
    np.save(f"{args.output_dir}/uncertainty_transmission.npy", error_binned)

    print(f"\nSaved files to {args.output_dir}:")
    print(f"  - wavelength_transmission.npy ({len(wave_angstrom)} points, Angstroms)")
    print(f"  - spectrum_transmission.npy")
    print(f"  - uncertainty_transmission.npy")

    if not args.calculate_transmission:
        print("\n⚠️  NOTE: This saved the COMBINED spectrum, not transmission spectrum!")
        print("   Use --calculate-transmission with transit parameters to get Rp/Rs.")
    else:
        print("\n✓ Transmission spectrum saved successfully!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
