#!/usr/bin/env python
"""Prepare PEPSI transmission spectrum data for retrieval.

This script:
1. Loads raw PEPSI FITS files
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
    data_dir: str = "input",
    barycentric_correction: bool = False,
    apply_introduced_shift: bool = True,
) -> tuple[np.ndarray, ...] | None:
    """Load PEPSI spectroscopic data."""
    ckms = 2.9979e5

    # Get data
    if arm == 'blue':
        arm_file = 'pepsib'
    elif arm == 'red':
        arm_file = 'pepsir'
    else:
        raise ValueError(f"arm must be 'blue' or 'red', got {arm}")

    year = float(observation_epoch[0:4])
    pepsi_exts = ["nor", "avr"]
    if year >= 2024:
        pepsi_exts.insert(0, "bwl")

    patterns = []
    if do_molecfit:
        for ext in pepsi_exts:
            patterns.append(
                f'{data_dir}/{observation_epoch}_{planet_name}/molecfit_weak/SCIENCE_TELLURIC_CORR_{arm_file}*.dxt.{ext}.fits'
            )
            patterns.append(
                f'{data_dir}/{observation_epoch}_{planet_name}/**/SCIENCE_TELLURIC_CORR_{arm_file}*.dxt.{ext}.fits'
            )
    else:
        for ext in pepsi_exts:
            patterns.append(
                f'{data_dir}/{observation_epoch}_{planet_name}/{arm_file}*.dxt.{ext}'
            )
            patterns.append(
                f'{data_dir}/{observation_epoch}_{planet_name}/**/{arm_file}*.dxt.{ext}'
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

        if do_molecfit:
            wave_tag, flux_tag, error_tag = 'lambda', 'flux', 'error'
        else:
            wave_tag, flux_tag, error_tag = 'Arg', 'Fun', 'Var'

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

        # Molecfit utilities already handle variance->uncertainty
        if not do_molecfit:
            errorin[i, :] = np.sqrt(errorin[i, :])

        if do_molecfit:
            wave[i, :] *= 10000.  # microns -> Angstroms

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

        jd[i] = header['JD-OBS']  # mid-exposure time
        try:
            snr_spectra[i] = header['SNR']
        except KeyError:
            snr_spectra[i] = np.percentile(fluxin[i, :] / errorin[i, :], 90)

        exptime_val = header['EXPTIME']
        if isinstance(exptime_val, str):
            exptime_strings = exptime_val.split(':')
            exptime[i] = float(exptime_strings[0]) * 3600. + float(exptime_strings[1]) * 60. + float(exptime_strings[2])
        else:
            exptime[i] = float(exptime_val)
        airmass[i] = header['AIRMASS']
        hdu.close()
        i += 1

    # Sort by time
    obs_order = np.argsort(jd)
    jd = jd[obs_order]
    snr_spectra = snr_spectra[obs_order]
    exptime = exptime[obs_order]
    airmass = airmass[obs_order]
    wave = wave[obs_order, :]
    fluxin = fluxin[obs_order, :]
    errorin = errorin[obs_order, :]

    return wave, fluxin, errorin, jd, snr_spectra, exptime, airmass, n_spectra, npix


def get_orbital_phase(
    jd: np.ndarray, epoch: float, period: float, RA: str, Dec: str
) -> np.ndarray:
    """Calculate orbital phase with light travel time correction."""
    lbt_coordinates = EarthLocation.of_site('lbt')

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
    parser.add_argument('--data-dir', type=str, default='input', help='Data directory')
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
