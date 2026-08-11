"""Authoritative raw-exposure selection for HRS science workflows.

File-pattern discovery, product-family fallback, and exposure-level QC must be
resolved once and identically for every science consumer.  This module keeps
that contract deliberately small: one immutable result object and one resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from glob import glob
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.time import Time

import config


SCIENCE_EXPOSURE_SELECTION_SCHEMA_VERSION = 3
SCIENCE_EXPOSURE_SELECTION_POLICY = "science_exposure_selection_v3"
SNR_QUALITY_POLICY = "paired_header_snr_q_v1"
_PEPSI_EXPOSURE_ID_RE = re.compile(r"(pepsi[br]\.\d{8}\.\d+)", re.IGNORECASE)


def _planet_key(planet_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(planet_name).lower())


def canonical_exposure_id(path_or_name: str | Path) -> str:
    """Return the product-independent PEPSI exposure identifier when possible."""

    name = Path(path_or_name).name
    match = _PEPSI_EXPOSURE_ID_RE.search(name)
    return match.group(1).lower() if match is not None else name.lower()


def normalize_fits_object_name(value: str) -> str:
    """Normalize harmless FITS OBJECT spelling differences for comparison."""

    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _configured_exposure_ids(
    *,
    planet_name: str,
    data_mode: str,
    observation_epoch: str,
    arm: str,
) -> tuple[str, ...]:
    key = (
        _planet_key(planet_name),
        str(data_mode).strip().lower(),
        str(observation_epoch),
        str(arm).strip().lower(),
    )
    configured = list(
        getattr(config, "HRS_EXCLUDED_EXPOSURE_IDS", {}).get(key, ())
    )
    # Accept the former exact-filename configuration during migration.  Values
    # are canonicalized to exposure IDs so raw and derived products are treated
    # as the same science exposure.
    configured.extend(
        getattr(config, "HRS_EXCLUDED_EXPOSURE_FILES", {}).get(key, ())
    )
    return tuple(
        dict.fromkeys(canonical_exposure_id(value) for value in configured)
    )


def _configured_nominal_object_names(planet_name: str) -> tuple[str, ...]:
    planet_key = _planet_key(planet_name)
    configured = tuple(
        str(value).strip()
        for value in getattr(config, "HRS_NOMINAL_FITS_OBJECT_NAMES", {}).get(
            planet_key,
            (),
        )
        if str(value).strip()
    )
    if not configured:
        raise ValueError(
            f"No nominal FITS OBJECT name is configured for HRS target "
            f"{planet_name!r} ({planet_key})."
        )
    normalized = tuple(normalize_fits_object_name(value) for value in configured)
    if any(not value for value in normalized):
        raise ValueError(
            f"HRS target {planet_name!r} has an empty nominal FITS OBJECT name."
        )
    return configured


@dataclass(frozen=True)
class FitsObjectCheck:
    """FITS OBJECT validation result for one candidate science file."""

    path: Path
    observed_name: str | None
    normalized_name: str | None
    accepted: bool
    rejection_reason: str | None

    def metadata(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "observed_name": self.observed_name,
            "normalized_name": self.normalized_name,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


def _check_fits_object(
    path: Path,
    *,
    accepted_normalized_names: frozenset[str],
) -> FitsObjectCheck:
    try:
        header = fits.getheader(path, 0)
    except (OSError, ValueError) as exc:
        return FitsObjectCheck(
            path=path,
            observed_name=None,
            normalized_name=None,
            accepted=False,
            rejection_reason=(
                f"unreadable_fits_header:{type(exc).__name__}:{exc}"
            ),
        )
    raw_value = header.get("OBJECT")
    observed_name = None if raw_value is None else str(raw_value).strip()
    if not observed_name:
        return FitsObjectCheck(
            path=path,
            observed_name=observed_name,
            normalized_name=None,
            accepted=False,
            rejection_reason="missing_fits_object",
        )
    normalized_name = normalize_fits_object_name(observed_name)
    accepted = normalized_name in accepted_normalized_names
    return FitsObjectCheck(
        path=path,
        observed_name=observed_name,
        normalized_name=normalized_name,
        accepted=accepted,
        rejection_reason=None if accepted else "fits_object_mismatch",
    )


@dataclass(frozen=True)
class SnrQualityCheck:
    """One exposure's deterministic paired-arm header-S/N decision."""

    path: Path
    snr: float
    exptime_s: float
    airmass: float
    midpoint_jd_utc: float
    expected_snr: float
    q_snr: float
    companion_path: Path | None
    companion_q_snr: float
    pair_dt_s: float
    excluded: bool
    reason: str

    def metadata(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "file": self.path.name,
            "snr": finite_or_none(self.snr),
            "exptime_s": finite_or_none(self.exptime_s),
            "airmass": finite_or_none(self.airmass),
            "midpoint_jd_utc": finite_or_none(self.midpoint_jd_utc),
            "expected_snr": finite_or_none(self.expected_snr),
            "q_snr": finite_or_none(self.q_snr),
            "companion_file": (
                None if self.companion_path is None else self.companion_path.name
            ),
            "companion_q_snr": finite_or_none(self.companion_q_snr),
            "pair_dt_s": finite_or_none(self.pair_dt_s),
            "excluded": bool(self.excluded),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScienceExposureSelection:
    """The complete, inspectable result of raw science-exposure selection."""

    planet_key: str
    data_mode: str
    observation_epoch: str
    arm: str
    do_molecfit: bool
    matched_pattern: str | None
    discovered_files: tuple[Path, ...]
    superseded_files: tuple[Path, ...]
    configured_exposure_ids: tuple[str, ...]
    nominal_fits_object_names: tuple[str, ...]
    configured_excluded_files: tuple[Path, ...]
    fits_object_checks: tuple[FitsObjectCheck, ...]
    fits_object_rejected_files: tuple[Path, ...]
    excluded_files: tuple[Path, ...]
    usable_files: tuple[Path, ...]
    snr_quality_checks: tuple[SnrQualityCheck, ...] = ()
    snr_quality_excluded_files: tuple[Path, ...] = ()
    snr_quality_applied: bool = False

    def metadata(self) -> dict[str, object]:
        """Return a JSON-safe summary for prepared-product metadata."""

        return {
            "schema_version": SCIENCE_EXPOSURE_SELECTION_SCHEMA_VERSION,
            "policy": SCIENCE_EXPOSURE_SELECTION_POLICY,
            "planet_key": self.planet_key,
            "data_mode": self.data_mode,
            "observation_epoch": self.observation_epoch,
            "arm": self.arm,
            "do_molecfit": self.do_molecfit,
            "matched_pattern": self.matched_pattern,
            "n_discovered_files": len(self.discovered_files),
            "n_superseded_files": len(self.superseded_files),
            "n_configured_excluded_files": len(self.configured_excluded_files),
            "n_fits_object_rejected_files": len(
                self.fits_object_rejected_files
            ),
            "n_snr_quality_excluded_files": len(
                self.snr_quality_excluded_files
            ),
            "n_excluded_files": len(self.excluded_files),
            "n_usable_files": len(self.usable_files),
            "configured_exposure_ids": list(self.configured_exposure_ids),
            "nominal_fits_object_names": list(self.nominal_fits_object_names),
            "discovered_files": [path.name for path in self.discovered_files],
            "superseded_files": [path.name for path in self.superseded_files],
            "configured_excluded_files": [
                path.name for path in self.configured_excluded_files
            ],
            "fits_object_checks": [
                check.metadata() for check in self.fits_object_checks
            ],
            "fits_object_rejected_files": [
                path.name for path in self.fits_object_rejected_files
            ],
            "snr_quality": {
                "policy": SNR_QUALITY_POLICY,
                "applied": bool(self.snr_quality_applied),
                "hard_threshold": float(config.HRS_PAIRED_SNR_Q_THRESHOLD),
                "q_definition": (
                    "snr_over_sqrt_exptime_divided_by_robust_"
                    "epoch_arm_time_airmass_expectation"
                ),
                "requires_both_arms_below_threshold": True,
                "missing_or_nonfinite_snr": "pass",
                "fallback_estimator": None,
                "checks": [
                    check.metadata() for check in self.snr_quality_checks
                ],
            },
            "snr_quality_excluded_files": [
                path.name for path in self.snr_quality_excluded_files
            ],
            "excluded_files": [path.name for path in self.excluded_files],
            "excluded_exposure_ids": [
                canonical_exposure_id(path) for path in self.excluded_files
            ],
            "usable_files": [path.name for path in self.usable_files],
        }


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _exptime_seconds(value: object) -> float:
    if isinstance(value, str):
        fields = value.split(":")
        if len(fields) == 3:
            try:
                return (
                    float(fields[0]) * 3600.0
                    + float(fields[1]) * 60.0
                    + float(fields[2])
                )
            except ValueError:
                return float("nan")
    return _safe_float(value)


def _calendar_midpoint_jd_utc(header, exptime_s: float) -> float:
    date_obs = str(header.get("DATE-OBS", "")).strip()
    ut_obs = str(header.get("UT-OBS", "")).strip()
    if not date_obs or not ut_obs or not np.isfinite(exptime_s) or exptime_s <= 0:
        return float("nan")
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
        return float(Time(start_utc, scale="utc").jd + 0.5 * exptime_s / 86400.0)
    return float("nan")


def _header_snr_rows(files: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        header = fits.getheader(path, 0)
        exptime_s = _exptime_seconds(header.get("EXPTIME"))
        rows.append(
            {
                "path": path,
                "snr": _safe_float(header.get("SNR")),
                "exptime_s": exptime_s,
                "airmass": _safe_float(header.get("AIRMASS")),
                "jd_header": _safe_float(header.get("JD-OBS")),
                "jd_calendar": _calendar_midpoint_jd_utc(header, exptime_s),
            }
        )

    header_jd = np.asarray([row["jd_header"] for row in rows], dtype=float)
    calendar_jd = np.asarray([row["jd_calendar"] for row in rows], dtype=float)
    use_calendar = bool(
        len(rows) > 1
        and np.all(np.isfinite(header_jd))
        and np.ptp(header_jd) == 0.0
        and np.all(np.isfinite(calendar_jd))
        and np.ptp(calendar_jd) > 0.0
    )
    for index, row in enumerate(rows):
        row["jd"] = float(calendar_jd[index] if use_calendar else header_jd[index])
    return rows


def _robust_expected_snr_rate(
    jd: np.ndarray,
    airmass: np.ndarray,
    snr_rate: np.ndarray,
) -> np.ndarray:
    """Fit the compact robust epoch/arm S/N baseline used by the audit."""

    finite = np.isfinite(jd) & np.isfinite(snr_rate) & (snr_rate > 0.0)
    expected = np.full(snr_rate.shape, np.nan, dtype=float)
    if np.count_nonzero(finite) == 0:
        return expected
    if np.count_nonzero(finite) < 5:
        expected[finite] = np.nanmedian(snr_rate[finite])
        return expected

    time = jd[finite]
    time_span = float(np.ptp(time))
    scaled_time = (
        np.zeros_like(time)
        if time_span <= 0.0
        else 2.0 * (time - np.min(time)) / time_span - 1.0
    )
    columns = [np.ones_like(time), scaled_time, scaled_time**2]
    finite_airmass = airmass[finite]
    if np.all(np.isfinite(finite_airmass)) and np.ptp(finite_airmass) > 0.02:
        scaled_airmass = (
            finite_airmass - np.median(finite_airmass)
        ) / max(float(np.std(finite_airmass)), 1.0e-8)
        trial = np.column_stack([*columns, scaled_airmass])
        if np.linalg.matrix_rank(trial) == trial.shape[1]:
            columns.append(scaled_airmass)

    design = np.column_stack(columns)
    log_rate = np.log(snr_rate[finite])
    weights = np.ones_like(log_rate)
    beta = np.linalg.lstsq(design, log_rate, rcond=None)[0]
    for _ in range(30):
        previous = beta.copy()
        root_weights = np.sqrt(weights)
        beta = np.linalg.lstsq(
            design * root_weights[:, None],
            log_rate * root_weights,
            rcond=None,
        )[0]
        residual = log_rate - design @ beta
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        if not np.isfinite(scale) or scale < 1.0e-5:
            break
        normalized = np.abs((residual - center) / (1.5 * scale))
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        if np.max(np.abs(beta - previous)) < 1.0e-10:
            break
    expected[finite] = np.exp(design @ beta)
    return expected


def _add_snr_quality(rows: list[dict[str, object]]) -> None:
    snr = np.asarray([row["snr"] for row in rows], dtype=float)
    exptime = np.asarray([row["exptime_s"] for row in rows], dtype=float)
    jd = np.asarray([row["jd"] for row in rows], dtype=float)
    airmass = np.asarray([row["airmass"] for row in rows], dtype=float)
    valid_rate = (
        np.isfinite(snr)
        & np.isfinite(exptime)
        & (snr > 0.0)
        & (exptime > 0.0)
    )
    snr_rate = np.full(snr.shape, np.nan, dtype=float)
    snr_rate[valid_rate] = snr[valid_rate] / np.sqrt(exptime[valid_rate])
    expected_rate = _robust_expected_snr_rate(jd, airmass, snr_rate)
    q_snr = np.full(snr.shape, np.nan, dtype=float)
    valid_q = np.isfinite(snr_rate) & np.isfinite(expected_rate) & (expected_rate > 0)
    q_snr[valid_q] = snr_rate[valid_q] / expected_rate[valid_q]
    for index, row in enumerate(rows):
        row["expected_snr"] = float(
            expected_rate[index] * np.sqrt(exptime[index])
            if np.isfinite(expected_rate[index]) and np.isfinite(exptime[index])
            else np.nan
        )
        row["q_snr"] = float(q_snr[index])


def _paired_indices(
    blue_rows: Sequence[dict[str, object]],
    red_rows: Sequence[dict[str, object]],
) -> tuple[tuple[int, int, float], ...]:
    candidates: list[tuple[float, int, int]] = []
    for blue_index, blue in enumerate(blue_rows):
        for red_index, red in enumerate(red_rows):
            blue_jd = float(blue["jd"])
            red_jd = float(red["jd"])
            if not np.isfinite(blue_jd) or not np.isfinite(red_jd):
                continue
            delta_s = abs(blue_jd - red_jd) * 86400.0
            tolerance_s = max(
                60.0,
                0.75
                * (
                    float(blue["exptime_s"])
                    + float(red["exptime_s"])
                ),
            )
            if (
                np.isfinite(delta_s)
                and np.isfinite(tolerance_s)
                and delta_s <= tolerance_s
            ):
                candidates.append((float(delta_s), blue_index, red_index))
    paired_blue: set[int] = set()
    paired_red: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for delta_s, blue_index, red_index in sorted(candidates):
        if blue_index in paired_blue or red_index in paired_red:
            continue
        paired_blue.add(blue_index)
        paired_red.add(red_index)
        pairs.append((blue_index, red_index, delta_s))
    return tuple(pairs)


def _quality_checks(
    rows: Sequence[dict[str, object]],
    companions: dict[int, tuple[dict[str, object], float]],
    *,
    excluded_indices: frozenset[int],
) -> tuple[SnrQualityCheck, ...]:
    checks: list[SnrQualityCheck] = []
    for index, row in enumerate(rows):
        companion, delta_s = companions.get(index, (None, float("nan")))
        q_snr = float(row["q_snr"])
        companion_q = (
            float("nan") if companion is None else float(companion["q_snr"])
        )
        if index in excluded_indices:
            reason = "paired_below_hard_threshold"
        elif not np.isfinite(q_snr):
            reason = "missing_or_nonfinite_snr_pass"
        elif companion is None:
            reason = "unpaired_pass"
        elif not np.isfinite(companion_q):
            reason = "companion_missing_or_nonfinite_snr_pass"
        else:
            reason = "threshold_pass"
        checks.append(
            SnrQualityCheck(
                path=Path(row["path"]),
                snr=float(row["snr"]),
                exptime_s=float(row["exptime_s"]),
                airmass=float(row["airmass"]),
                midpoint_jd_utc=float(row["jd"]),
                expected_snr=float(row["expected_snr"]),
                q_snr=q_snr,
                companion_path=(
                    None if companion is None else Path(companion["path"])
                ),
                companion_q_snr=companion_q,
                pair_dt_s=float(delta_s),
                excluded=index in excluded_indices,
                reason=reason,
            )
        )
    return tuple(checks)


def _with_snr_quality_checks(
    selection: ScienceExposureSelection,
    checks: tuple[SnrQualityCheck, ...],
) -> ScienceExposureSelection:
    quality_excluded = tuple(check.path for check in checks if check.excluded)
    excluded_set = set(quality_excluded)
    return replace(
        selection,
        snr_quality_checks=checks,
        snr_quality_excluded_files=quality_excluded,
        snr_quality_applied=True,
        excluded_files=tuple(
            dict.fromkeys([*selection.excluded_files, *quality_excluded])
        ),
        usable_files=tuple(
            path for path in selection.usable_files if path not in excluded_set
        ),
    )


def apply_paired_snr_quality_rule(
    blue_selection: ScienceExposureSelection,
    red_selection: ScienceExposureSelection,
) -> tuple[ScienceExposureSelection, ScienceExposureSelection]:
    """Apply the sole automatic S/N cut: both paired arms must have q < 0.4."""

    identity_fields = ("planet_key", "data_mode", "observation_epoch")
    if any(
        getattr(blue_selection, field) != getattr(red_selection, field)
        for field in identity_fields
    ):
        raise ValueError("Blue/red S/N selections do not describe the same observation.")
    if blue_selection.arm != "blue" or red_selection.arm != "red":
        raise ValueError("Paired S/N quality requires blue and red selections.")

    blue_rows = _header_snr_rows(blue_selection.usable_files)
    red_rows = _header_snr_rows(red_selection.usable_files)
    _add_snr_quality(blue_rows)
    _add_snr_quality(red_rows)
    blue_companions: dict[int, tuple[dict[str, object], float]] = {}
    red_companions: dict[int, tuple[dict[str, object], float]] = {}
    excluded_blue: set[int] = set()
    excluded_red: set[int] = set()
    threshold = float(config.HRS_PAIRED_SNR_Q_THRESHOLD)
    for blue_index, red_index, delta_s in _paired_indices(blue_rows, red_rows):
        blue = blue_rows[blue_index]
        red = red_rows[red_index]
        blue_companions[blue_index] = (red, delta_s)
        red_companions[red_index] = (blue, delta_s)
        blue_q = float(blue["q_snr"])
        red_q = float(red["q_snr"])
        if (
            np.isfinite(blue_q)
            and np.isfinite(red_q)
            and blue_q < threshold
            and red_q < threshold
        ):
            excluded_blue.add(blue_index)
            excluded_red.add(red_index)

    blue_checks = _quality_checks(
        blue_rows,
        blue_companions,
        excluded_indices=frozenset(excluded_blue),
    )
    red_checks = _quality_checks(
        red_rows,
        red_companions,
        excluded_indices=frozenset(excluded_red),
    )
    return (
        _with_snr_quality_checks(blue_selection, blue_checks),
        _with_snr_quality_checks(red_selection, red_checks),
    )


def select_science_exposures(
    patterns: Sequence[str],
    *,
    planet_name: str,
    data_mode: str,
    observation_epoch: str,
    arm: str,
    do_molecfit: bool,
) -> ScienceExposureSelection:
    """Resolve the exact files every HRS science consumer must use."""

    matched_pattern: str | None = None
    discovered_files: tuple[Path, ...] = ()
    superseded_files: tuple[Path, ...] = ()
    product_files: tuple[Path, ...] = ()

    for pattern in patterns:
        discovered = tuple(
            Path(path) for path in sorted(glob(pattern, recursive=True))
        )
        if not discovered:
            continue
        matched_pattern = str(pattern)
        discovered_files = discovered
        product_files = discovered
        if do_molecfit:
            current = tuple(
                path
                for path in discovered
                if not any(part.endswith("_old") for part in path.parts)
            )
            if current:
                product_files = current
                current_set = set(current)
                superseded_files = tuple(
                    path for path in discovered if path not in current_set
                )
        break

    configured_ids = _configured_exposure_ids(
        planet_name=planet_name,
        data_mode=data_mode,
        observation_epoch=observation_epoch,
        arm=arm,
    )
    nominal_object_names = _configured_nominal_object_names(planet_name)
    accepted_normalized_names = frozenset(
        normalize_fits_object_name(value) for value in nominal_object_names
    )
    excluded_id_set = frozenset(configured_ids)
    configured_excluded_files = tuple(
        path
        for path in product_files
        if canonical_exposure_id(path) in excluded_id_set
    )
    configured_excluded_set = set(configured_excluded_files)
    header_candidates = tuple(
        path for path in product_files if path not in configured_excluded_set
    )
    fits_object_checks = tuple(
        _check_fits_object(
            path,
            accepted_normalized_names=accepted_normalized_names,
        )
        for path in header_candidates
    )
    fits_object_rejected_files = tuple(
        check.path for check in fits_object_checks if not check.accepted
    )
    excluded_set = configured_excluded_set | set(fits_object_rejected_files)
    excluded_files = tuple(path for path in product_files if path in excluded_set)
    usable_files = tuple(path for path in product_files if path not in excluded_set)

    return ScienceExposureSelection(
        planet_key=_planet_key(planet_name),
        data_mode=str(data_mode).strip().lower(),
        observation_epoch=str(observation_epoch),
        arm=str(arm).strip().lower(),
        do_molecfit=bool(do_molecfit),
        matched_pattern=matched_pattern,
        discovered_files=discovered_files,
        superseded_files=superseded_files,
        configured_exposure_ids=configured_ids,
        nominal_fits_object_names=nominal_object_names,
        configured_excluded_files=configured_excluded_files,
        fits_object_checks=fits_object_checks,
        fits_object_rejected_files=fits_object_rejected_files,
        excluded_files=excluded_files,
        usable_files=usable_files,
    )
