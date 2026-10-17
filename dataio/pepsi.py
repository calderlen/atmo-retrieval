"""PEPSI exposure loading, wavelength-frame reconstruction, and common-grid regridding."""

from __future__ import annotations

import os
import re
import numpy as np
import config
import config_utils
from datetime import datetime
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from exojax.database.core_atom.io import air_to_vac
from config_utils import get_data_patterns, get_header_keys, get_fits_columns
from dataio.exposure_selection import (
    ScienceExposureSelection,
    apply_paired_snr_quality_rule,
    select_science_exposures,
)
from dataio.timing import PEPSI_INPUT_TIME_SCALE
from dataio.spectral_preprocessing import (
    do_sysrem,
    get_sysrem_max_systematics,
    get_sysrem_min_systematics,
    subtract_median_spectrum,
)


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
        max_gap = float(max_native_gap_factor) * float(np.median(spacing))
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


_SPEED_OF_LIGHT_MPS = 299792458.0


_SPEED_OF_LIGHT_KMS = _SPEED_OF_LIGHT_MPS / 1000.0


_REMOVED_VELOCITY_HISTORY_RE = re.compile(
    r"Radial\s+velocity\s+([A-Z0-9_+\-]+)\s*=\s*"
    r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)\s*m/s\s+is\s+removed",
    re.IGNORECASE,
)


def _wavelength_angstrom(data, column_config: dict) -> np.ndarray:
    """Read the configured wavelength column in Angstroms."""
    wavelength = np.asarray(data[column_config["wave"]], dtype=float).copy()
    wavelength *= {"micron": 10000.0, "angstrom": 1.0}[str(column_config["wave_unit"]).lower()]
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
    raw_name = name[
        len(_MOLECFIT_CORRECTED_PREFIX) : -len(_MOLECFIT_FITS_SUFFIX)
    ]
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
    return next(iter(matches.values()))


def _finite_header_velocity_mps(header, key: str) -> float | None:
    return float(header[key]) if key in header else None


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

    removed: list[tuple[str, float]] = []
    for entry in entries:
        match = _REMOVED_VELOCITY_HISTORY_RE.search(entry)
        if match is None:
            continue
        removed.append((match.group(1).upper(), float(match.group(2))))

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

    observer_removed = None
    stellar_removed = 0.0
    stellar_documented = False

    standalone_observer = history_value("SSBVEL", "OBSVEL")
    if standalone_observer is not None:
        observer_removed = standalone_observer

    standalone_stellar = history_value("RADVEL")
    if standalone_stellar is not None:
        stellar_removed = standalone_stellar
        stellar_documented = True

    combined_observer_stellar = history_value("OBSVEL+RADVEL")
    if combined_observer_stellar is not None:
        if observer_header is None:
            observer_header = combined_observer_stellar - float(stellar_header)
            observer_header_key = "OBSVEL(derived)"
        if stellar_header is None:
            stellar_header = combined_observer_stellar - float(observer_header)
        observer_removed = float(observer_header)
        stellar_removed = float(stellar_header)
        stellar_documented = True

    combined_sst = history_value("SSTVEL")
    if combined_sst is not None:
        if observer_header is None:
            observer_header = combined_sst - float(stellar_header)
            observer_header_key = "observer_velocity(derived)"
        if stellar_header is None:
            stellar_header = combined_sst - float(observer_header)
        observer_removed = float(observer_header)
        stellar_removed = float(stellar_header)
        stellar_documented = True


    instrument_removed = history_value("CCFOFF") or 0.0

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
    doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
    return np.asarray(wavelength, dtype=float) / doppler_factor


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
    """Restore the Molecfit grid with the original loader correction."""
    raw_path = _resolve_molecfit_raw_product(molecfit_path, data_dir)
    molecfit_wave = _wavelength_angstrom(molecfit_data, molecfit_column_config)
    correction_mps, correction_keys = _molecfit_velocity_correction_mps(
        molecfit_header,
        observation_epoch=observation_epoch,
        planet_name=planet_name,
        arm=arm,
    )
    return (
        _shift_wavelength_velocity_mps(molecfit_wave, correction_mps),
        raw_path,
        correction_mps,
        correction_keys,
    )


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


        raw_vacuum_wave = _air_wavelength_to_vacuum(raw_air_wave)
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
    excluded_exposure_files = [
        path.name for path in exposure_selection.excluded_files
    ]

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
                    raw_source = (
                        f" Raw product '{raw_path.name}' is paired with this exposure."
                        if raw_path is not None
                        else " Original raw product unavailable; using FITS-header reconstruction."
                    )
                    print(
                        "Molecfit velocity restoration: original PEPSI recipe using "
                        f"{keys} "
                        f"(correction={rest_correction_mps:.3f} m/s)."
                        f"{raw_source}"
                    )
        else:
            loaded_wave = _air_wavelength_to_vacuum(loaded_wave)
            frame_info = parse_pepsi_velocity_frame(header)
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
            frame_info = parse_pepsi_velocity_frame(header)
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
