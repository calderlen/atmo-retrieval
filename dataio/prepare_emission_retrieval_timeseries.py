#!/usr/bin/env python
"""Prepare retrieval-ready emission time-series products from PEPSI exposures.

This module converts raw/reduced PEPSI emission exposure directories such as
``input/hrs/emission/raw/kelt20b/20230430`` into the `.npy` bundle consumed by
the time-series retrieval path:

- ``wavelength.npy`` (1D wavelength grid in Angstroms)
- ``data.npy`` (2D exposure x wavelength matrix)
- ``sigma.npy`` (2D uncertainty matrix)
- ``phase.npy`` (1D orbital phase array, transit-centered convention)
- ``bjd_tdb.npy`` (1D canonical barycentric mid-exposure times)

Optional auxiliary products are also written when available, including
``jd.npy`` (UTC exposure midpoint), ``snr.npy``, ``exptime.npy``, ``airmass.npy``,
and the frozen full-exposure SYSREM operator with per-pixel uncertainties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import config
from config import EPHEMERIS, FULL_ARM_MEMBERS
from config_utils import get_params, resolve_transit_midpoint
from dataio.collapse_emission_timeseries_to_1d import (
    describe_emission_selection,
    emission_selection_mask,
)
from dataio.spectral_preprocessing import arm_edge_trim_metadata, get_sysrem_deep_mask, get_sysrem_ignore_mask
from dataio.timing import get_bjd_tdb, get_ephemeris_epoch_bjd_tdb, get_orbital_phase
from dataio.edge_trim_manifest import load_accepted_edge_trim_widths
from dataio.hrs_preparation import (
    load_hrs_arm,
    output_dir_for,
    requested_stellar_velocity_correction,
    save_sysrem_products,
    valid_sorted_column_indices,
)
from dataio.wavelength_frame_contract import build_wavelength_frame_contract


EMISSION_PHASE_BINS = {
    "all": "all exposures",
    "eclipse": "narrow window centered on orbital phase 0.5",
    "dayside": "half orbit centered on orbital phase 0.5",
    "nightside": "half orbit centered on orbital phase 0.0",
    "pre_eclipse": "out-of-eclipse dayside phases before secondary eclipse",
    "post_eclipse": "out-of-eclipse dayside phases after secondary eclipse",
}


def _phase_mod_1(phase: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(phase, dtype=float), 1.0)


def _circular_phase_distance(phase: np.ndarray, center: float) -> np.ndarray:
    phase_01 = _phase_mod_1(phase)
    delta = np.abs(phase_01 - float(center))
    return np.minimum(delta, 1.0 - delta)


def _eclipse_half_width_phase(planet_params: dict[str, Any]) -> float:
    return float(planet_params["duration"]) / (2.0 * float(planet_params["period"]))


def _phase_bin_definition(phase_bin: str, planet_params: dict[str, Any]) -> str:
    if phase_bin == "all":
        return EMISSION_PHASE_BINS["all"]
    if phase_bin == "eclipse":
        return (
            "circular phase distance to 0.5 <= duration / (2 * period) "
            f"(half-width={_eclipse_half_width_phase(planet_params):.6f})"
        )
    if phase_bin == "dayside":
        return "circular phase distance to 0.5 <= 0.25"
    if phase_bin == "nightside":
        return "circular phase distance to 0.0 <= 0.25"
    if phase_bin == "pre_eclipse":
        return describe_emission_selection("pre_eclipse", planet_params)
    if phase_bin == "post_eclipse":
        return describe_emission_selection("post_eclipse", planet_params)


def emission_phase_selection_mask(
    phase: np.ndarray,
    *,
    phase_bin: str,
    planet_params: dict[str, Any],
) -> np.ndarray:
    phase_01 = _phase_mod_1(phase)

    if phase_bin == "all":
        return np.ones_like(phase_01, dtype=bool)
    if phase_bin == "eclipse":
        return _circular_phase_distance(phase_01, 0.5) <= (_eclipse_half_width_phase(planet_params) + 1.0e-12)
    if phase_bin == "dayside":
        return _circular_phase_distance(phase_01, 0.5) <= 0.25 + 1.0e-12
    if phase_bin == "nightside":
        return _circular_phase_distance(phase_01, 0.0) <= 0.25 + 1.0e-12
    if phase_bin == "pre_eclipse":
        return emission_selection_mask(phase_01, "pre_eclipse", planet_params)
    if phase_bin == "post_eclipse":
        return emission_selection_mask(phase_01, "post_eclipse", planet_params)



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
    subtract_median: bool,
    run_sysrem: bool,
    regrid: bool,
    planet_params: dict[str, Any],
    arm_edge_trim: dict[str, float | int],
    spectral_column_masking: dict[str, Any],
    product_kind: str,
    stellar_velocity: dict[str, Any],
    wavelength_frame_contract: dict[str, Any],
    input_exposure_files: list[str],
    excluded_exposure_files: list[str],
    science_exposure_selection: dict[str, Any],
) -> None:
    phase_01 = _phase_mod_1(phase)
    model_preprocessing_steps = ["active_exposure_mask"]
    if subtract_median:
        model_preprocessing_steps.append("time_median_subtraction")
    if run_sysrem:
        model_preprocessing_steps.append("frozen_per_pixel_sysrem")
    model_preprocessing_steps.append("exposure_selection")
    metadata: dict[str, Any] = {
        "mode": "emission",
        "product_kind": product_kind,
        "planet": planet,
        "ephemeris": ephemeris,
        "epoch": epoch,
        "arm": arm,
        "phase_bin": phase_bin,
        "phase_bin_definition": _phase_bin_definition(phase_bin, planet_params),
        "phase_convention": "orbital_transit_zero",
        "t0_bjd": float(t0),  # Backward-compatible alias; now explicitly BJD_TDB.
        "t0_bjd_tdb": float(t0),
        "eclipse_center_phase": 0.5,
        "n_exposures": int(phase.size),
        "n_source_exposures": int(source_phase.size),
        "input_exposure_files": list(input_exposure_files),
        "excluded_exposure_files": list(excluded_exposure_files),
        "science_exposure_selection": science_exposure_selection,
        "selected_exposure_indices": np.asarray(
            selected_exposure_indices,
            dtype=int,
        ).tolist(),
        "timeseries_operator_file": "timeseries_operator.npz",
        "model_preprocessing": "_then_".join(model_preprocessing_steps),
        "phase_min": float(np.min(phase)),
        "phase_max": float(np.max(phase)),
        "phase_mod1_min": float(np.min(phase_01)),
        "phase_mod1_max": float(np.max(phase_01)),
        "jd_min": float(np.min(jd)),
        "jd_max": float(np.max(jd)),
        "bjd_tdb_min": float(np.min(bjd_tdb)),
        "bjd_tdb_max": float(np.max(bjd_tdb)),
        "time": time_metadata,
        "regrid": bool(regrid),
        "subtract_median": bool(subtract_median),
        "run_sysrem": bool(run_sysrem),
        "arm_edge_trim": arm_edge_trim,
        "spectral_column_masking": spectral_column_masking,
        "stellar_velocity": stellar_velocity,
        "wavelength_medium": wavelength_frame_contract["wavelength_medium"],
        "wavelength_frame": wavelength_frame_contract["wavelength_frame"],
        "wavelength_frame_contract": wavelength_frame_contract,
    }
    duration = planet_params.get("duration")
    if duration is not None and np.isfinite(float(duration)):
        metadata["eclipse_half_width_phase"] = _eclipse_half_width_phase(planet_params)

    (output_dir / "timeseries_prep.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare retrieval-ready emission time-series products from PEPSI exposures.")
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
        choices=sorted(EMISSION_PHASE_BINS.keys()),
        default="all",
        help="Which orbital phase selection to keep in the exported cube (default: all)",
    )
    parser.add_argument(
        "--product-kind",
        choices=["timeseries", "collapse-source"],
        default="timeseries",
        help=(
            "Write a phase-selected retrieval cube or the all-exposure source "
            "cube used to build collapsed 1D spectra (default: timeseries)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory "
            "(default: the product-kind subdirectory under "
            "input/hrs/emission/<planet>/<epoch>/<arm>). "
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
    parser.add_argument(
        "--edge-trim-manifest",
        type=Path,
        default=None,
        help=(
            "Apply this dataset's exact widths from an accepted adaptive schema-v4 "
            "calibration manifest"
        ),
    )
    parser.add_argument(
        "--apply-stellar-rest",
        action="store_true",
        help=(
            "Apply an accepted stellar_velocity_lsd.json correction. By default, "
            "products remain in the barycentric frame."
        ),
    )
    return parser


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
            "saved SYSREM basis retains every source exposure."
        )

    if args.arm == "full" and args.output_dir:
        raise ValueError(
            "--output-dir is not supported with --arm full because red and blue are "
            "written to separate directories. Run each arm explicitly with an "
            "--output-dir, or drop --output-dir to use the default per-arm paths."
        )

    planet_cfg = get_params(args.planet, args.ephemeris)
    period = planet_cfg["period"]
    ra = planet_cfg["RA"]
    dec = planet_cfg["Dec"]
    reference_epoch = planet_cfg["epoch"]

    if args.arm == "full":
        arms_to_run: tuple[str, ...] = FULL_ARM_MEMBERS
    else:
        arms_to_run = (args.arm,)

    for arm in arms_to_run:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = output_dir_for(
                mode="emission",
                planet=args.planet,
                epoch=args.epoch,
                arm=arm,
                product_kind=args.product_kind,
            )
        prepare_arm(
            arm=arm,
            args=args,
            planet_cfg=planet_cfg,
            reference_epoch=float(reference_epoch),
            period=float(period),
            ra=str(ra),
            dec=str(dec),
            output_dir=output_dir,
        )

    return 0


def prepare_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    planet_cfg: dict[str, Any],
    reference_epoch: float,
    period: float,
    ra: str,
    dec: str,
    output_dir: Path,
) -> None:
    """Prepare and persist one emission arm using resolved CLI arguments."""

    print(f"\nLoading PEPSI {arm} emission data for {args.planet} ({args.epoch})...")
    edge_trim_widths_A = None
    edge_trim_source = None
    if args.edge_trim_manifest is not None:
        selected_manifest, edge_trim_widths_A = load_accepted_edge_trim_widths(
            args.edge_trim_manifest,
            planet=args.planet,
            mode="emission",
            epoch=args.epoch,
            arm=arm,
        )
        edge_trim_source = str(selected_manifest)
    stellar_rest_velocity_kms, stellar_velocity = (
        requested_stellar_velocity_correction(
            enabled=args.apply_stellar_rest,
            mode="emission",
            planet=args.planet,
            epoch=args.epoch,
            arm=arm,
        )
    )
    result, extras = load_hrs_arm(
        mode="emission",
        arm=arm,
        epoch=args.epoch,
        planet=args.planet,
        molecfit=args.molecfit,
        regrid=args.regrid,
        subtract_median=args.subtract_median,
        run_sysrem=args.run_sysrem,
        stellar_rest_velocity_kms=stellar_rest_velocity_kms,
        edge_trim_widths_A=edge_trim_widths_A,
    )

    wave, data, sigma, jd, snr, exptime, airmass, n_spectra, npix = result
    print(f"Loaded {n_spectra} exposures with {npix} pixels each before selection.")

    reference_epoch_bjd_tdb = get_ephemeris_epoch_bjd_tdb(
        reference_epoch,
        planet_cfg.get("epoch_scale"),
        planet_cfg.get("epoch_reference"),
    )
    bjd_tdb, time_metadata = get_bjd_tdb(
        np.asarray(jd),
        ra,
        dec,
        header_bjd_tdb=extras.get("header_bjd_tdb"),
        input_time_keyword=str(extras.get("input_time_keyword", "JD-OBS")),
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
            "input_time_provenance": dict(
                extras.get("input_time_provenance", {})
            ),
        }
    )
    t0 = resolve_transit_midpoint(
        bjd_tdb,
        planet_cfg,
        reference_epoch_bjd_tdb=reference_epoch_bjd_tdb,
        observation_epoch=args.epoch,
    )
    time_metadata.update(
        {
            "timing_model": str(planet_cfg.get("timing_model", "linear")),
            "resolved_mid_transit_bjd_tdb": t0,
        }
    )
    phase = np.asarray(get_orbital_phase(bjd_tdb, t0, period), dtype=float)
    source_phase = np.asarray(phase, dtype=float).copy()
    source_bjd_tdb = np.asarray(bjd_tdb, dtype=float).copy()
    source_data = np.asarray(data, dtype=float).copy()
    source_sigma = np.asarray(sigma, dtype=float).copy()
    active_exposure_mask = np.ones_like(source_phase, dtype=float)
    selection = emission_phase_selection_mask(
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
    data = np.asarray(data, dtype=float)[selection]
    sigma = np.asarray(sigma, dtype=float)[selection]

    wave_1d = np.asarray(wave[0] if np.asarray(wave).ndim == 2 else wave)
    raw_n_columns = int(wave_1d.size)
    sysrem_ignore_mask = get_sysrem_ignore_mask(
        wave_1d,
        arm,
        explicit_edge_trim_widths_A=edge_trim_widths_A,
    )
    deep_telluric_mask = get_sysrem_deep_mask(wave_1d, arm)
    edge_trim_info = arm_edge_trim_metadata(
        wave_1d,
        arm,
        planet=args.planet,
        mode="emission",
        epoch=args.epoch,
        explicit_widths_A=edge_trim_widths_A,
        source=edge_trim_source,
    )
    column_indices = valid_sorted_column_indices(
        wave_1d,
        source_data,
        source_sigma,
        arm=arm,
        edge_trim_widths_A=edge_trim_widths_A,
    )
    wave_1d = np.asarray(wave_1d, dtype=float)[column_indices]
    source_sigma = source_sigma[:, column_indices]
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
        subtract_time_median=np.asarray(args.subtract_median, dtype=bool),
        has_sysrem=np.asarray(args.run_sysrem, dtype=bool),
    )

    save_sysrem_products(
        output_dir,
        wavelength=wave_1d,
        arm=arm,
        n_source_exposures=source_phase.size,
        source_sigma=source_sigma,
        extras=extras,
        run_sysrem=args.run_sysrem,
    )

    wavelength_frame_contract = build_wavelength_frame_contract(
        extras,
        n_source_exposures=source_phase.size,
        stellar_velocity=stellar_velocity,
    )
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
        subtract_median=args.subtract_median,
        run_sysrem=args.run_sysrem,
        regrid=args.regrid,
        planet_params=planet_cfg,
        arm_edge_trim=edge_trim_info,
        spectral_column_masking=spectral_column_masking,
        product_kind=args.product_kind,
        stellar_velocity=stellar_velocity,
        wavelength_frame_contract=wavelength_frame_contract,
        input_exposure_files=list(extras.get("input_exposure_files", ())),
        excluded_exposure_files=list(extras.get("excluded_exposure_files", ())),
        science_exposure_selection=dict(
            extras.get("science_exposure_selection_metadata", {})
        ),
    )

    phase_01 = _phase_mod_1(phase)
    product_label = (
        "emission collapse-source"
        if args.product_kind == "collapse-source"
        else "retrieval-ready emission time-series"
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
    print(
        "  jd.npy: UTC midpoint from "
        f"{time_metadata['input_keyword']}; bjd_tdb.npy: canonical BJD_TDB"
    )
    print(
        f"  Phase range: {float(np.min(phase)):.5f} to {float(np.max(phase)):.5f}; "
        f"phase(mod1): {float(np.min(phase_01)):.5f} to {float(np.max(phase_01)):.5f}; "
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


if __name__ == "__main__":
    raise SystemExit(main())
