#!/usr/bin/env python
"""Prepare retrieval-ready time-series products from PEPSI exposure folders.

This module converts raw/reduced PEPSI exposure directories such as
``input/hrs/transmission/raw/kelt20b/20250601`` into the `.npy` bundle consumed by the
time-series retrieval path:

- ``wavelength.npy`` (1D wavelength grid in Angstroms)
- ``data.npy`` (2D exposure x wavelength matrix)
- ``sigma.npy`` (2D uncertainty matrix)
- ``phase.npy`` (1D orbital phase array, mid-transit at 0)

Optional auxiliary products are also written when available, including
``jd.npy``, ``snr.npy``, ``exptime.npy``, ``airmass.npy``, and SYSREM
approximations compatible with the current retrieval loader. When SYSREM is
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
    compute_contact_phases,
    get_orbital_phase,
    get_pepsi_data,
    get_phase_bin_mask,
    get_sysrem_deep_mask,
    get_sysrem_chunk_indices,
    get_sysrem_ignore_mask,
)
from dataio.horus import remove_doppler_shadow


def _output_dir_for(planet: str, epoch: str, arm: str) -> Path:
    return config_utils.get_data_dir(planet=planet, epoch=epoch, arm=arm, mode="transmission")


def _raw_input_dir_for(planet: str, epoch: str) -> Path:
    return config_utils.get_raw_hrs_dir(planet=planet, epoch=epoch, mode="transmission")


def _nearest_transit_midpoint(jd: np.ndarray, reference_epoch: float, period: float) -> float:
    obs_mid = 0.5 * (float(np.min(jd)) + float(np.max(jd)))
    n_orbits = round((obs_mid - reference_epoch) / period)
    return float(reference_epoch + n_orbits * period)


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
    barycorr: bool,
    introduced_shift: bool,
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
        barycentric_correction=barycorr,
        apply_introduced_shift=introduced_shift if prefer_molecfit else False,
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
            barycentric_correction=barycorr,
            apply_introduced_shift=False,
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
    barycorr: bool,
    introduced_shift: bool,
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
    apply_introduced_shift = introduced_shift
    if arm == "blue":
        prefer_molecfit = False
        apply_introduced_shift = False

    return _load_single_arm(
        arm,
        epoch,
        planet,
        prefer_molecfit=prefer_molecfit,
        barycorr=barycorr,
        introduced_shift=apply_introduced_shift,
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


def _sysrem_vdiag_from_sigma(sigma: np.ndarray) -> np.ndarray:
    exposure_sigma = np.sqrt(np.mean(np.square(sigma), axis=1))
    return 1.0 / np.clip(exposure_sigma, config.F32_FLOOR_RECIP, None)


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


def _sysrem_chunk_vdiag_from_sigma(
    sigma: np.ndarray,
    chunk_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    v_diag = []
    for indices in chunk_indices:
        chunk_sigma = sigma[:, np.asarray(indices, dtype=int)]
        if chunk_sigma.shape[1] == 0:
            v_diag.append(np.ones((sigma.shape[0],), dtype=float))
        else:
            v_diag.append(_sysrem_vdiag_from_sigma(chunk_sigma))
    return np.asarray(v_diag, dtype=float)


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
    jd: np.ndarray,
    contacts: dict[str, float],
    subtract_median: bool,
    run_sysrem: bool,
    regrid: bool,
    doppler_shadow_status: dict[str, Any],
    arm_edge_trim: dict[str, float | int],
    spectral_column_masking: dict[str, Any],
) -> None:
    contacts_serialized: dict[str, float] = {}
    for k, v in contacts.items():
        contacts_serialized[k] = float(v)
    metadata = {
        "planet": planet,
        "ephemeris": ephemeris,
        "epoch": epoch,
        "arm": arm,
        "phase_bin": phase_bin,
        "t0_bjd": float(t0),
        "n_exposures": int(phase.size),
        "phase_min": float(np.min(phase)),
        "phase_max": float(np.max(phase)),
        "jd_min": float(np.min(jd)),
        "jd_max": float(np.max(jd)),
        "contacts": contacts_serialized,
        "regrid": bool(regrid),
        "subtract_median": bool(subtract_median),
        "run_sysrem": bool(run_sysrem),
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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory "
            "(default: input/hrs/transmission/<planet>/<epoch>/<arm>). "
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
        "--barycorr",
        action="store_true",
        default=config.DEFAULT_BARYCORR,
        help="Apply barycentric correction",
    )
    parser.add_argument(
        "--no-barycorr",
        action="store_false",
        dest="barycorr",
        help="Disable barycentric correction",
    )
    parser.add_argument(
        "--introduced-shift",
        action="store_true",
        default=config.DEFAULT_INTRODUCED_SHIFT,
        help="Apply epoch-specific Molecfit shift correction",
    )
    parser.add_argument(
        "--no-introduced-shift",
        action="store_false",
        dest="introduced_shift",
        help="Disable epoch-specific Molecfit shift correction",
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

    print(f"\nLoading PEPSI {arm} data for {args.planet} ({args.epoch})...")
    result, extras = _load_data(
        arm=arm,
        epoch=args.epoch,
        planet=args.planet,
        molecfit=args.molecfit,
        barycorr=args.barycorr,
        introduced_shift=args.introduced_shift,
        regrid=args.regrid,
        subtract_median=args.subtract_median,
        run_sysrem=args.run_sysrem,
    )

    wave, data, sigma, jd, snr, exptime, airmass, n_spectra, npix = result
    print(f"Loaded {n_spectra} exposures with {npix} pixels each before selection.")

    t0 = _nearest_transit_midpoint(np.asarray(jd), reference_epoch, period)
    phase = np.asarray(get_orbital_phase(np.asarray(jd), t0, period, ra, dec), dtype=float)
    wave_1d_full = np.asarray(wave[0] if np.asarray(wave).ndim == 2 else wave, dtype=float)
    data = np.asarray(data, dtype=float)
    data, doppler_shadow_status = _apply_default_doppler_shadow(
        data,
        wave_1d_full,
        phase,
        planet_cfg=planet_cfg,
        subtract_median=args.subtract_median,
    )
    selection = _phase_selection_mask(
        phase,
        phase_bin=args.phase_bin,
        planet_params=planet_cfg,
    )
    if not np.any(selection):
        raise ValueError(
            f"No exposures selected for phase_bin={args.phase_bin} (arm={arm})."
        )

    phase = np.asarray(phase)[selection]
    jd = np.asarray(jd)[selection]
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
        data,
        sigma,
        arm=arm,
        planet=args.planet,
        mode="transmission",
        epoch=args.epoch,
    )
    wave_1d = np.asarray(wave_1d, dtype=float)[column_indices]
    data = data[:, column_indices]
    sigma = sigma[:, column_indices]
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
    np.save(output_dir / "snr.npy", snr)
    np.save(output_dir / "exptime.npy", exptime)
    np.save(output_dir / "airmass.npy", airmass)
    if args.run_sysrem and pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        np.save(output_dir / "pre_sysrem_data.npy", pre_sysrem_data)
        np.save(output_dir / "pre_sysrem_sigma.npy", pre_sysrem_sigma)

    if args.run_sysrem:
        U_full = extras.get("U_sysrem")
        if U_full is None:
            raise ValueError("SYSREM requested but U_sysrem was not returned by preprocessing.")
        U_full = np.asarray(U_full)[selection]
        chunk_names, chunk_indices, _ = get_sysrem_chunk_indices(wave_1d, arm)
        chunk_labels = _chunk_labels_from_indices(wave_1d.size, chunk_indices)
        basis_counts = _sysrem_basis_counts(U_full)
        V_chunk_diag = _sysrem_chunk_vdiag_from_sigma(sigma, chunk_indices)
        sysrem_diagnostics = _sysrem_diagnostics_for_save(extras)
        np.savez(
            output_dir / "U_sysrem.npz",
            U_sysrem=U_full,
            chunk_labels=chunk_labels,
            basis_counts=basis_counts,
            V_chunk_diag=V_chunk_diag,
            chunk_names=np.asarray(chunk_names, dtype="U32"),
            **sysrem_diagnostics,
        )
        print(
            "  Saved chunked SYSREM bundle: "
            f"{len(chunk_names)} chunks, basis counts={basis_counts.tolist()}"
        )
        if "sysrem_delta_stddev" in sysrem_diagnostics:
            print("  Saved SYSREM component diagnostics: sysrem_delta_stddev and acceptance masks")

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
        jd=jd,
        contacts=contacts,
        subtract_median=args.subtract_median,
        run_sysrem=args.run_sysrem,
        regrid=args.regrid,
        doppler_shadow_status=doppler_shadow_status,
        arm_edge_trim=edge_trim_info,
        spectral_column_masking=spectral_column_masking,
    )

    print(f"\nSaved retrieval-ready time-series products (arm={arm}):")
    print(f"  Output dir: {output_dir}")
    print(f"  wavelength.npy: {wave_1d.shape}")
    print(f"  data.npy: {data.shape}")
    print(f"  sigma.npy: {sigma.shape}")
    print(f"  phase.npy: {phase.shape} ({args.phase_bin})")
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
            output_dir = _output_dir_for(args.planet, args.epoch, arm)
        _process_arm(
            arm=arm,
            args=args,
            planet_cfg=planet_cfg,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
