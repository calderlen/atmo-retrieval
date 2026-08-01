#!/usr/bin/env python
"""Prepare retrieval-ready time-series products from PEPSI exposure folders.

This module converts raw/reduced PEPSI exposure directories such as
``input/hrs/transmission/raw/kelt20b/20250601`` into the `.npy` bundle consumed by the
time-series retrieval path:

- ``wavelength.npy`` (1D wavelength grid in Angstroms)
- ``data.npy`` (2D exposure x wavelength matrix)
- ``sigma.npy`` (2D uncertainty matrix)
- ``phase.npy`` (1D orbital phase array, mid-transit at 0)
- ``bjd_tdb.npy`` (1D canonical barycentric mid-exposure times)

Optional auxiliary products are also written when available, including
``jd.npy`` (raw UTC ``JD-OBS``), ``snr.npy``, ``exptime.npy``, ``airmass.npy``,
and the frozen full-exposure SYSREM operator with per-pixel uncertainties. When SYSREM is
enabled, ``pre_sysrem_data.npy`` and ``pre_sysrem_sigma.npy`` preserve the
selected spectra just before SYSREM for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import config
import config_utils
from config import EPHEMERIS, FULL_ARM_MEMBERS, PHASE_BINS
from config_utils import get_params
from dataio.collapse_transmission_timeseries_to_1d import (
    arm_edge_trim_metadata,
    build_out_of_transit_residuals,
    compute_contact_phases,
    do_sysrem,
    get_bjd_tdb,
    get_ephemeris_epoch_bjd_tdb,
    get_orbital_phase,
    get_pepsi_data,
    get_phase_bin_mask,
    get_sysrem_deep_mask,
    get_sysrem_chunk_indices,
    get_sysrem_ignore_mask,
)
from dataio.horus import remove_doppler_shadow


def _output_dir_for(
    planet: str,
    epoch: str,
    arm: str,
    product_kind: str,
) -> Path:
    if product_kind == "collapse-source":
        return config_utils.get_collapse_source_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode="transmission",
        )
    return config_utils.get_timeseries_data_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
        mode="transmission",
    )


def _raw_input_dir_for(planet: str, epoch: str) -> Path:
    return config_utils.get_raw_hrs_dir(planet=planet, epoch=epoch, mode="transmission")


def _nearest_transit_midpoint(
    bjd_tdb: np.ndarray,
    reference_epoch_bjd_tdb: float,
    period: float,
) -> float:
    obs_mid = 0.5 * (float(np.min(bjd_tdb)) + float(np.max(bjd_tdb)))
    n_orbits = round((obs_mid - reference_epoch_bjd_tdb) / period)
    return float(reference_epoch_bjd_tdb + n_orbits * period)


def _planet_config(planet: str, ephemeris: str) -> dict[str, Any]:
    return get_params(planet, ephemeris)


def _unwrap_result(result: Any) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def _load_single_arm(
    arm: str,
    epoch: str,
    planet: str,
    *,
    prefer_molecfit: bool,
    regrid: bool,
    subtract_median: bool,
    run_sysrem: bool,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    result = get_pepsi_data(
        arm=arm,
        observation_epoch=epoch,
        planet_name=planet,
        do_molecfit=prefer_molecfit,
        data_dir=_raw_input_dir_for(planet, epoch),
        regrid=regrid,
        subtract_median=subtract_median,
        run_sysrem=run_sysrem,
        data_mode="transmission",
    )
    if result is None and prefer_molecfit:
        print(f"  No molecfit files for {arm}; retrying with raw files.")
        result = get_pepsi_data(
            arm=arm,
            observation_epoch=epoch,
            planet_name=planet,
            do_molecfit=False,
            data_dir=_raw_input_dir_for(planet, epoch),
            regrid=regrid,
            subtract_median=subtract_median,
            run_sysrem=run_sysrem,
            data_mode="transmission",
        )
    if result is None:
        raise FileNotFoundError(
            f"Could not load {arm}-arm PEPSI data for {planet} {epoch} from "
            f"{_raw_input_dir_for(planet, epoch)}."
        )
    return _unwrap_result(result)


def _load_data(
    *,
    arm: str,
    epoch: str,
    planet: str,
    molecfit: bool,
    regrid: bool,
    subtract_median: bool,
    run_sysrem: bool,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    if arm == "full":
        raise ValueError(
            "_load_data() is per-arm; 'full' must be expanded into its "
            "constituent arms by the caller."
        )

    prefer_molecfit = molecfit
    if arm == "blue":
        prefer_molecfit = False

    return _load_single_arm(
        arm,
        epoch,
        planet,
        prefer_molecfit=prefer_molecfit,
        regrid=regrid,
        subtract_median=subtract_median,
        run_sysrem=run_sysrem,
    )


def _phase_selection_mask(
    phase: np.ndarray,
    *,
    phase_bin: str,
    planet_params: dict[str, Any],
) -> np.ndarray:
    if phase_bin == "all":
        return np.ones_like(phase, dtype=bool)
    return get_phase_bin_mask(phase, phase_bin, planet_params)


def _sanitize_columns(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    *,
    arm: str | None = None,
    planet: str | None = None,
    mode: str | None = None,
    epoch: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    column_indices = _valid_sorted_column_indices(
        wavelength,
        data,
        sigma,
        arm=arm,
        planet=planet,
        mode=mode,
        epoch=epoch,
    )
    wavelength = np.asarray(wavelength, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return wavelength[column_indices], data[:, column_indices], sigma[:, column_indices]


def _valid_sorted_column_indices(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    *,
    arm: str | None = None,
    planet: str | None = None,
    mode: str | None = None,
    epoch: str | None = None,
) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if wavelength.ndim != 1:
        raise ValueError(f"Expected 1D wavelength grid, got shape {wavelength.shape}.")
    if data.ndim != 2 or sigma.ndim != 2:
        raise ValueError(f"Expected 2D data/sigma matrices, got {data.shape=} and {sigma.shape=}.")
    if data.shape != sigma.shape:
        raise ValueError(f"data shape {data.shape} does not match sigma shape {sigma.shape}.")
    if data.shape[1] != wavelength.size:
        raise ValueError(
            f"Spectral axis mismatch: data.shape[1]={data.shape[1]} "
            f"but wavelength.size={wavelength.size}."
        )

    valid = np.isfinite(wavelength) & (wavelength > 0.0)
    valid &= np.all(np.isfinite(data), axis=0)
    valid &= np.all(np.isfinite(sigma), axis=0)
    valid &= np.all(sigma > 0.0, axis=0)
    if arm is not None:
        valid &= ~get_sysrem_ignore_mask(
            wavelength,
            arm,
            planet=planet,
            mode=mode,
            epoch=epoch,
        )

    if not np.any(valid):
        raise ValueError("No valid spectral columns remain after masking.")

    valid_indices = np.flatnonzero(valid)
    sort_idx = np.argsort(wavelength[valid_indices])
    return valid_indices[sort_idx]


def _shadow_status(
    *,
    applied: bool,
    skip_reason: str | None = None,
    scaling: float | None = None,
) -> dict[str, Any]:
    return {
        "applied": bool(applied),
        "skip_reason": skip_reason,
        "scaling": scaling,
    }


def _is_missing_numeric(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return True


def _build_shadow_inputs(
    planet_cfg: dict[str, Any],
    phase: np.ndarray,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1 or phase.size == 0 or not np.all(np.isfinite(phase)):
        reason = "invalid_phase_array"
        print(f"Warning: skipping Doppler shadow removal ({reason}).")
        return None, _shadow_status(applied=False, skip_reason=reason)

    planet_field_names = ("rp_rs", "b", "lambda_angle", "a_rs", "period")
    stellar_field_map = {
        "vsini": "v_sini_star",
        "gamma1": "gamma1",
        "gamma2": "gamma2",
    }

    missing: list[str] = []
    planet_params: dict[str, float] = {}
    for field_name in planet_field_names:
        value = planet_cfg.get(field_name)
        if _is_missing_numeric(value):
            missing.append(field_name)
        else:
            planet_params[field_name] = float(value)

    stellar_params: dict[str, float] = {}
    for out_name, cfg_name in stellar_field_map.items():
        value = planet_cfg.get(cfg_name)
        if _is_missing_numeric(value):
            missing.append(cfg_name)
        else:
            stellar_params[out_name] = float(value)

    if missing:
        reason = f"missing_or_invalid_shadow_params: {', '.join(missing)}"
        print(f"Warning: skipping Doppler shadow removal ({reason}).")
        return None, _shadow_status(applied=False, skip_reason=reason)

    return {
        "phase": phase,
        "planet_params": planet_params,
        "stellar_params": stellar_params,
    }, _shadow_status(applied=False)


def _apply_default_doppler_shadow(
    data: np.ndarray,
    wavelength: np.ndarray,
    phase: np.ndarray,
    *,
    planet_cfg: dict[str, Any],
    subtract_median: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = np.asarray(data, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if not subtract_median:
        reason = "subtract_median_disabled"
        print("Warning: skipping Doppler shadow removal because --no-subtract-median was used.")
        return data, _shadow_status(applied=False, skip_reason=reason)

    shadow_inputs, status = _build_shadow_inputs(planet_cfg, phase)
    if shadow_inputs is None:
        return data, status

    print("Applying Doppler shadow removal to retrieval-prep cube...")
    corrected_data, _shadow_model, fit_info = remove_doppler_shadow(
        data,
        wavelength,
        shadow_inputs["phase"],
        shadow_inputs["planet_params"],
        shadow_inputs["stellar_params"],
    )
    scaling = fit_info.get("scaling")
    scaling_value = None if _is_missing_numeric(scaling) else float(scaling)
    if scaling_value is not None:
        print(f"  Doppler shadow scaling: {scaling_value:.6g}")
    return np.asarray(corrected_data, dtype=float), _shadow_status(
        applied=True,
        scaling=scaling_value,
    )


def _chunk_labels_from_indices(
    n_wave: int,
    chunk_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    labels = np.full(n_wave, -1, dtype=int)
    for chunk_id, indices in enumerate(chunk_indices):
        labels[np.asarray(indices, dtype=int)] = chunk_id
    if np.any(labels < 0):
        missing = int(np.sum(labels < 0))
        raise ValueError(f"{missing} wavelength columns were not assigned to any SYSREM chunk.")
    return labels


def _sysrem_basis_counts(U_full: np.ndarray) -> np.ndarray:
    U_full = np.asarray(U_full, dtype=float)
    if U_full.ndim == 2:
        U_full = U_full[:, :, np.newaxis]
    if U_full.ndim != 3:
        raise ValueError(f"Unsupported U_sysrem shape: {U_full.shape}")

    counts = []
    for chunk in range(U_full.shape[2]):
        counts.append(int(np.sum(np.any(np.isfinite(U_full[:, :, chunk]), axis=0))))
    return np.asarray(counts, dtype=int)


def _sysrem_diagnostics_for_save(extras: dict[str, Any]) -> dict[str, np.ndarray]:
    keys = (
        "sysrem_stddev_before",
        "sysrem_stddev_after",
        "sysrem_delta_stddev",
        "sysrem_component_attempted",
        "sysrem_component_accepted",
        "sysrem_min_systematics",
        "sysrem_max_systematics",
        "sysrem_stop_delta_stddev",
    )
    return {
        key: np.asarray(extras[key])
        for key in keys
        if extras.get(key) is not None
    }


def _save_metadata(
    output_dir: Path,
    *,
    planet: str,
    ephemeris: str,
    epoch: str,
    arm: str,
    phase_bin: str,
    t0: float,
    phase: np.ndarray,
    source_phase: np.ndarray,
    selected_exposure_indices: np.ndarray,
    jd: np.ndarray,
    bjd_tdb: np.ndarray,
    time_metadata: dict[str, Any],
    contacts: dict[str, float],
    subtract_median: bool,
    run_sysrem: bool,
    regrid: bool,
    doppler_shadow_status: dict[str, Any],
    arm_edge_trim: dict[str, float | int],
    spectral_column_masking: dict[str, Any],
    product_kind: str,
    out_of_transit_master_division: bool,
) -> None:
    contacts_serialized: dict[str, float] = {}
    for k, v in contacts.items():
        contacts_serialized[k] = float(v)
    model_preprocessing_steps = ["active_exposure_mask"]
    if subtract_median:
        model_preprocessing_steps.append("time_median_subtraction")
    if run_sysrem:
        model_preprocessing_steps.append("frozen_per_pixel_sysrem")
    model_preprocessing_steps.append("exposure_selection")
    metadata = {
        "planet": planet,
        "product_kind": product_kind,
        "ephemeris": ephemeris,
        "epoch": epoch,
        "arm": arm,
        "phase_bin": phase_bin,
        "t0_bjd": float(t0),  # Backward-compatible alias; now explicitly BJD_TDB.
        "t0_bjd_tdb": float(t0),
        "n_exposures": int(phase.size),
        "n_source_exposures": int(source_phase.size),
        "selected_exposure_indices": np.asarray(
            selected_exposure_indices,
            dtype=int,
        ).tolist(),
        "timeseries_operator_file": "timeseries_operator.npz",
        "model_preprocessing": "_then_".join(model_preprocessing_steps),
        "phase_min": float(np.min(phase)),
        "phase_max": float(np.max(phase)),
        "jd_min": float(np.min(jd)),
        "jd_max": float(np.max(jd)),
        "bjd_tdb_min": float(np.min(bjd_tdb)),
        "bjd_tdb_max": float(np.max(bjd_tdb)),
        "time": time_metadata,
        "contacts": contacts_serialized,
        "regrid": bool(regrid),
        "subtract_median": bool(subtract_median),
        "run_sysrem": bool(run_sysrem),
        "out_of_transit_master_division": bool(
            out_of_transit_master_division
        ),
        "doppler_shadow_applied": bool(doppler_shadow_status.get("applied", False)),
        "doppler_shadow_skip_reason": doppler_shadow_status.get("skip_reason"),
        "doppler_shadow_scaling": doppler_shadow_status.get("scaling"),
        "arm_edge_trim": arm_edge_trim,
        "spectral_column_masking": spectral_column_masking,
    }
    (output_dir / "timeseries_prep.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare retrieval-ready time-series products from PEPSI exposures.")
    parser.add_argument("--epoch", type=str, required=True, help="Observation epoch (YYYYMMDD)")
    parser.add_argument(
        "--planet",
        type=str,
        default=config.DEFAULT_DATA_PLANET,
        help="Planet name",
    )
    parser.add_argument(
        "--ephemeris",
        type=str,
        default=EPHEMERIS,
        help="Ephemeris key from config (default: %(default)s)",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["red", "blue", "full"],
        default=config.DEFAULT_DATA_ARM,
        help="Spectrograph arm",
    )
    parser.add_argument(
        "--phase-bin",
        type=str,
        choices=["all", "full", *PHASE_BINS.keys()],
        default="full",
        help="Which exposures to keep in the exported cube (default: full in-transit)",
    )
    parser.add_argument(
        "--product-kind",
        choices=["timeseries", "collapse-source"],
        default="timeseries",
        help=(
            "Write a phase-selected retrieval cube or the all-exposure "
            "out-of-transit-referenced cube used to build a collapsed 1D "
            "spectrum (default: timeseries)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory "
            "(default: the product-kind subdirectory under "
            "input/hrs/transmission/<planet>/<epoch>/<arm>). "
            "Not allowed with --arm full, since red and blue are written separately."
        ),
    )
    parser.add_argument(
        "--molecfit",
        action="store_true",
        default=config.DEFAULT_USE_MOLECFIT,
        help="Prefer molecfit-corrected files",
    )
    parser.add_argument(
        "--no-molecfit",
        action="store_false",
        dest="molecfit",
        help="Use uncorrected files",
    )
    parser.add_argument(
        "--regrid",
        action="store_true",
        default=True,
        help="Regrid all exposures to a common wavelength grid (default: on)",
    )
    parser.add_argument(
        "--no-regrid",
        action="store_false",
        dest="regrid",
        help="Keep native per-exposure wavelength grids (not recommended for retrieval export)",
    )
    parser.add_argument(
        "--subtract-median",
        action="store_true",
        default=True,
        help="Subtract the median spectrum before export (default: on)",
    )
    parser.add_argument(
        "--no-subtract-median",
        action="store_false",
        dest="subtract_median",
        help="Export spectra without median subtraction",
    )
    parser.add_argument(
        "--run-sysrem",
        action="store_true",
        help="Run chunk-aware SYSREM and export retrieval SYSREM auxiliaries",
    )
    return parser


def _process_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    planet_cfg: dict[str, Any],
    output_dir: Path,
) -> None:
    period = planet_cfg["period"]
    ra = planet_cfg["RA"]
    dec = planet_cfg["Dec"]
    reference_epoch = planet_cfg["epoch"]
    reference_epoch_bjd_tdb = get_ephemeris_epoch_bjd_tdb(
        reference_epoch,
        planet_cfg.get("epoch_scale"),
        planet_cfg.get("epoch_reference"),
    )

    print(f"\nLoading PEPSI {arm} data for {args.planet} ({args.epoch})...")
    collapse_source = args.product_kind == "collapse-source"
    result, extras = _load_data(
        arm=arm,
        epoch=args.epoch,
        planet=args.planet,
        molecfit=args.molecfit,
        regrid=args.regrid,
        subtract_median=False if collapse_source else args.subtract_median,
        run_sysrem=False if collapse_source else args.run_sysrem,
    )

    wave, data, sigma, jd, snr, exptime, airmass, n_spectra, npix = result
    print(f"Loaded {n_spectra} exposures with {npix} pixels each before selection.")

    bjd_tdb, time_metadata = get_bjd_tdb(
        np.asarray(jd),
        ra,
        dec,
        header_bjd_tdb=extras.get("header_bjd_tdb"),
        return_diagnostics=True,
    )
    time_metadata.update(
        {
            "raw_file": "jd.npy",
            "canonical_file": "bjd_tdb.npy",
            "ephemeris_epoch": float(reference_epoch),
            "ephemeris_epoch_scale": str(planet_cfg["epoch_scale"]).lower(),
            "ephemeris_epoch_reference": planet_cfg["epoch_reference"],
            "ephemeris_epoch_bjd_tdb": reference_epoch_bjd_tdb,
        }
    )
    t0 = _nearest_transit_midpoint(bjd_tdb, reference_epoch_bjd_tdb, period)
    phase = np.asarray(get_orbital_phase(bjd_tdb, t0, period), dtype=float)
    wave_1d_full = np.asarray(wave[0] if np.asarray(wave).ndim == 2 else wave, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if collapse_source:
        contacts = compute_contact_phases(planet_cfg)
        out_transit = (
            (phase < contacts["T1"])
            | (phase > contacts["T4"])
        )
        full_transit = (
            (phase >= contacts["T2"])
            & (phase <= contacts["T3"])
        )
        source_rows = out_transit | full_transit
        phase = phase[source_rows]
        jd = np.asarray(jd)[source_rows]
        bjd_tdb = np.asarray(bjd_tdb)[source_rows]
        snr = np.asarray(snr)[source_rows]
        exptime = np.asarray(exptime)[source_rows]
        airmass = np.asarray(airmass)[source_rows]
        data = data[source_rows]
        sigma = sigma[source_rows]
        out_transit = out_transit[source_rows]
        data, sigma = build_out_of_transit_residuals(
            data,
            sigma,
            out_transit,
        )
        extras = {
            "pre_sysrem_flux": np.asarray(data, dtype=float).copy(),
            "pre_sysrem_error": np.asarray(sigma, dtype=float).copy(),
        }
        if args.run_sysrem:
            sysrem_result = do_sysrem(
                wave_1d_full,
                data,
                sigma,
                arm,
                np.asarray(airmass, dtype=float),
                do_molecfit=bool(args.molecfit and arm == "red"),
                stop_delta_stddev=config.DEFAULT_SYSREM_STOP_TOL,
                return_diagnostics=True,
                planet_name=args.planet,
                data_mode="transmission",
                observation_epoch=args.epoch,
            )
            data, sigma, U_sysrem, no_tellurics, diagnostics = sysrem_result
            extras.update(
                {
                    "U_sysrem": U_sysrem,
                    "no_tellurics": no_tellurics,
                    **diagnostics,
                }
            )
        doppler_shadow_status = {
            "applied": False,
            "skip_reason": "disabled_for_collapsed_1d_source",
            "scaling": None,
        }
    else:
        data, doppler_shadow_status = _apply_default_doppler_shadow(
            data,
            wave_1d_full,
            phase,
            planet_cfg=planet_cfg,
            subtract_median=args.subtract_median,
        )
    source_phase = np.asarray(phase, dtype=float).copy()
    source_bjd_tdb = np.asarray(bjd_tdb, dtype=float).copy()
    source_data = np.asarray(data, dtype=float).copy()
    source_sigma = np.asarray(sigma, dtype=float).copy()
    contacts = compute_contact_phases(planet_cfg)
    if collapse_source:
        # The dedicated collapse source intentionally omits ingress and
        # egress, so its active model rows are the retained T23 rows.
        active_exposure_mask = (
            (source_phase >= contacts["T2"])
            & (source_phase <= contacts["T3"])
        ).astype(float)
    else:
        # Ordinary preparation derives its temporal median and SYSREM basis
        # from the complete exposure sequence. Preserve the pre-existing T14
        # model support while adding zero-valued out-of-transit source rows;
        # likelihood-row selection is replayed only after filtering.
        active_exposure_mask = (
            (source_phase >= contacts["T1"])
            & (source_phase <= contacts["T4"])
        ).astype(float)
    selection = _phase_selection_mask(
        phase,
        phase_bin=args.phase_bin,
        planet_params=planet_cfg,
    )
    if not np.any(selection):
        raise ValueError(
            f"No exposures selected for phase_bin={args.phase_bin} (arm={arm})."
        )
    selected_exposure_indices = np.flatnonzero(selection).astype(int)

    phase = np.asarray(phase)[selection]
    jd = np.asarray(jd)[selection]
    bjd_tdb = np.asarray(bjd_tdb)[selection]
    snr = np.asarray(snr)[selection]
    exptime = np.asarray(exptime)[selection]
    airmass = np.asarray(airmass)[selection]
    data = np.asarray(data)[selection]
    sigma = np.asarray(sigma)[selection]
    pre_sysrem_data = extras.get("pre_sysrem_flux")
    pre_sysrem_sigma = extras.get("pre_sysrem_error")
    if pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        pre_sysrem_data = np.asarray(pre_sysrem_data, dtype=float)[selection]
        pre_sysrem_sigma = np.asarray(pre_sysrem_sigma, dtype=float)[selection]

    wave_1d = np.asarray(wave[0] if np.asarray(wave).ndim == 2 else wave)
    raw_n_columns = int(wave_1d.size)
    sysrem_ignore_mask = get_sysrem_ignore_mask(
        wave_1d,
        arm,
        planet=args.planet,
        mode="transmission",
        epoch=args.epoch,
    )
    deep_telluric_mask = get_sysrem_deep_mask(wave_1d, arm)
    edge_trim_info = arm_edge_trim_metadata(
        wave_1d,
        arm,
        planet=args.planet,
        mode="transmission",
        epoch=args.epoch,
    )
    column_indices = _valid_sorted_column_indices(
        wave_1d,
        source_data,
        source_sigma,
        arm=arm,
        planet=args.planet,
        mode="transmission",
        epoch=args.epoch,
    )
    wave_1d = np.asarray(wave_1d, dtype=float)[column_indices]
    data = data[:, column_indices]
    sigma = sigma[:, column_indices]
    source_sigma = source_sigma[:, column_indices]
    spectral_column_masking = {
        "n_input_columns": raw_n_columns,
        "n_output_columns": int(wave_1d.size),
        "n_dropped_columns": int(raw_n_columns - wave_1d.size),
        "n_sysrem_ignore_columns_dropped": int(np.count_nonzero(sysrem_ignore_mask)),
        "n_deep_telluric_columns_dropped": int(np.count_nonzero(deep_telluric_mask)),
        "mask_components": [
            "deep_telluric",
            "telluric_region_edges",
            "arm_edges",
        ],
    }
    if pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        pre_sysrem_data = pre_sysrem_data[:, column_indices]
        pre_sysrem_sigma = pre_sysrem_sigma[:, column_indices]

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "wavelength.npy", wave_1d)
    np.save(output_dir / "data.npy", data)
    np.save(output_dir / "sigma.npy", sigma)
    np.save(output_dir / "phase.npy", phase)
    np.save(output_dir / "jd.npy", jd)
    np.save(output_dir / "bjd_tdb.npy", bjd_tdb)
    np.save(output_dir / "snr.npy", snr)
    np.save(output_dir / "exptime.npy", exptime)
    np.save(output_dir / "airmass.npy", airmass)
    np.savez_compressed(
        output_dir / "timeseries_operator.npz",
        schema_version=np.asarray(1, dtype=np.int32),
        source_wavelength=wave_1d,
        source_phase=source_phase,
        source_bjd_tdb=source_bjd_tdb,
        active_exposure_mask=active_exposure_mask,
        selected_exposure_indices=selected_exposure_indices,
        subtract_time_median=np.asarray(
            False if collapse_source else args.subtract_median,
            dtype=bool,
        ),
        has_sysrem=np.asarray(args.run_sysrem, dtype=bool),
    )
    if args.run_sysrem and pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        np.save(output_dir / "pre_sysrem_data.npy", pre_sysrem_data)
        np.save(output_dir / "pre_sysrem_sigma.npy", pre_sysrem_sigma)

    if args.run_sysrem:
        U_full = extras.get("U_sysrem")
        if U_full is None:
            raise ValueError("SYSREM requested but U_sysrem was not returned by preprocessing.")
        U_full = np.asarray(U_full)
        if U_full.shape[0] != source_phase.size:
            raise ValueError(
                "SYSREM basis exposure axis does not match the frozen source "
                f"sequence: {U_full.shape[0]} versus {source_phase.size}."
            )
        chunk_names, chunk_indices, _ = get_sysrem_chunk_indices(wave_1d, arm)
        chunk_labels = _chunk_labels_from_indices(wave_1d.size, chunk_indices)
        basis_counts = _sysrem_basis_counts(U_full)
        sysrem_diagnostics = _sysrem_diagnostics_for_save(extras)
        np.savez_compressed(
            output_dir / "U_sysrem.npz",
            U_sysrem=U_full,
            chunk_labels=chunk_labels,
            basis_counts=basis_counts,
            projection_sigma=source_sigma,
            chunk_names=np.asarray(chunk_names, dtype="U32"),
            **sysrem_diagnostics,
        )
        print(
            "  Saved chunked SYSREM bundle: "
            f"{len(chunk_names)} chunks, basis counts={basis_counts.tolist()}"
        )
        if "sysrem_delta_stddev" in sysrem_diagnostics:
            print("  Saved SYSREM component diagnostics: sysrem_delta_stddev and acceptance masks")
    else:
        # Conditional products from an earlier SYSREM run must not survive
        # beside a newly exported non-SYSREM cube.
        for filename in (
            "U_sysrem.npz",
            "pre_sysrem_data.npy",
            "pre_sysrem_sigma.npy",
        ):
            (output_dir / filename).unlink(missing_ok=True)

    contacts = compute_contact_phases(planet_cfg)
    _save_metadata(
        output_dir,
        planet=args.planet,
        ephemeris=args.ephemeris,
        epoch=args.epoch,
        arm=arm,
        phase_bin=args.phase_bin,
        t0=t0,
        phase=phase,
        source_phase=source_phase,
        selected_exposure_indices=selected_exposure_indices,
        jd=jd,
        bjd_tdb=bjd_tdb,
        time_metadata=time_metadata,
        contacts=contacts,
        subtract_median=False if collapse_source else args.subtract_median,
        run_sysrem=args.run_sysrem,
        regrid=args.regrid,
        doppler_shadow_status=doppler_shadow_status,
        arm_edge_trim=edge_trim_info,
        spectral_column_masking=spectral_column_masking,
        product_kind=args.product_kind,
        out_of_transit_master_division=collapse_source,
    )

    product_label = (
        "transmission collapse-source"
        if collapse_source
        else "retrieval-ready time-series"
    )
    print(f"\nSaved {product_label} products (arm={arm}):")
    print(f"  Output dir: {output_dir}")
    print(f"  wavelength.npy: {wave_1d.shape}")
    print(f"  data.npy: {data.shape}")
    print(f"  sigma.npy: {sigma.shape}")
    print(f"  phase.npy: {phase.shape} ({args.phase_bin})")
    print(
        "  timeseries_operator.npz: "
        f"{source_phase.size} source rows -> {phase.size} likelihood rows"
    )
    print("  jd.npy: raw UTC JD-OBS; bjd_tdb.npy: canonical BJD_TDB")
    print(
        f"  Phase range: {float(np.min(phase)):.5f} to {float(np.max(phase)):.5f}; "
        f"wavelength range: {float(np.min(wave_1d)):.1f} to {float(np.max(wave_1d)):.1f} A"
    )
    if edge_trim_info["n_trimmed_columns"]:
        print(
            "  Applied arm-edge trim: "
            f"left={edge_trim_info['left_trim_A']:.1f} A, "
            f"right={edge_trim_info['right_trim_A']:.1f} A, "
            f"columns={edge_trim_info['n_trimmed_columns']}"
        )
    if spectral_column_masking["n_sysrem_ignore_columns_dropped"]:
        print(
            "  Dropped SYSREM-ignore columns before export: "
            f"{spectral_column_masking['n_sysrem_ignore_columns_dropped']} "
            f"(deep telluric={spectral_column_masking['n_deep_telluric_columns_dropped']})"
        )
    if args.run_sysrem:
        print("  Saved chunk-aware SYSREM auxiliaries: U_sysrem.npz")
        if pre_sysrem_data is not None and pre_sysrem_sigma is not None:
            print("  Saved pre-SYSREM diagnostic arrays: pre_sysrem_data.npy, pre_sysrem_sigma.npy")


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if not args.regrid:
        raise ValueError(
            "Retrieval-ready time-series export requires a common wavelength grid; "
            "leave --regrid enabled."
        )
    if args.product_kind == "collapse-source" and args.phase_bin != "all":
        raise ValueError(
            "--product-kind collapse-source requires --phase-bin all so the "
            "out-of-transit reference and full SYSREM basis are preserved."
        )

    if args.arm == "full":
        arms_to_run: tuple[str, ...] = FULL_ARM_MEMBERS
        if args.output_dir:
            raise ValueError(
                "--output-dir is not supported with --arm full because red and blue are "
                "written to separate directories. Run each arm explicitly with an "
                "--output-dir, or drop --output-dir to use the default per-arm paths."
            )
    else:
        arms_to_run = (args.arm,)

    planet_cfg = _planet_config(args.planet, args.ephemeris)
    missing = []
    for name in ("period", "duration", "RA", "Dec", "epoch", "tau"):
        value = planet_cfg.get(name)
        if value is None or value != value:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Missing required planet parameters for {args.planet}: {', '.join(missing)}."
        )

    for arm in arms_to_run:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = _output_dir_for(
                args.planet,
                args.epoch,
                arm,
                args.product_kind,
            )
        _process_arm(
            arm=arm,
            args=args,
            planet_cfg=planet_cfg,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
