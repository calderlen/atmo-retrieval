"""Canonical BJD_TDB timing, orbital phases, and transit phase selections."""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from config import OBSERVATORY, PHASE_BINS


PEPSI_INPUT_TIME_SCALE = "utc"

PEPSI_BJD_TDB_VALIDATION_TOLERANCE_SECONDS = 0.1


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


def get_phase_bin_mask(phase: np.ndarray, bin_name: str, params: dict) -> np.ndarray:
    """Select T12, T23, T34, or full-transit exposures; phase zero is mid-transit."""
    if bin_name == "full":
        # Full transit: T1 to T4
        contacts = compute_contact_phases(params)
        return (phase >= contacts["T1"]) & (phase <= contacts["T4"])

    boundaries = get_phase_boundaries(params)
    lo, hi = boundaries[bin_name]
    return (phase >= lo) & (phase <= hi)


def get_phase_bin_indices(phase: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """Map configured phase-bin names to exposure indices; phase zero is mid-transit."""
    result = {}
    for bin_name in PHASE_BINS:
        mask = get_phase_bin_mask(phase, bin_name, params)
        result[bin_name] = np.where(mask)[0]
    return result


def summarize_phase_coverage(phase: np.ndarray, params: dict) -> dict[str, dict]:
    """Summarize exposure counts and phase limits for each transit bin."""
    contacts = compute_contact_phases(params)
    bin_indices = get_phase_bin_indices(phase, params)

    summary = {"contacts": contacts, "bins": {}}

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
        validated = np.isfinite(header_bjd_tdb)
        n_validated = int(np.count_nonzero(validated))
        diagnostics["header_validation_count"] = n_validated
        if n_validated:
            residual_seconds = (
                bjd_tdb[validated] - header_bjd_tdb[validated]
            ) * 86400.0
            max_abs_residual = float(np.max(np.abs(residual_seconds)))
            diagnostics["header_validation_max_abs_residual_seconds"] = max_abs_residual
            print(
                f"Compared computed BJD_TDB with JD-TDB for {n_validated} "
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
    scale = str(epoch_scale).strip().lower()
    epoch_time = Time(float(epoch), format="jd", scale=scale)
    return float(epoch_time.tdb.jd)


def get_orbital_phase(
    bjd_tdb: np.ndarray,
    epoch_bjd_tdb: float,
    period: float,
) -> np.ndarray:
    """Calculate phase from canonical BJD_TDB observation and epoch values."""
    bjd_tdb = np.asarray(bjd_tdb, dtype=float)

    orbital_phase = (bjd_tdb - epoch_bjd_tdb) / period
    orbital_phase -= np.round(np.mean(orbital_phase))

    return orbital_phase
