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
``jd.npy`` (UTC exposure midpoint), ``snr.npy``, ``exptime.npy``, ``airmass.npy``,
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
from config import EPHEMERIS, FULL_ARM_MEMBERS, PHASE_BINS
from config_utils import get_params, resolve_transit_midpoint
from dataio.timing import active_transit_mask, compute_contact_phases, get_bjd_tdb, get_ephemeris_epoch_bjd_tdb, get_orbital_phase, get_phase_bin_mask
from dataio.spectral_preprocessing import arm_edge_trim_metadata, build_out_of_transit_residuals, do_sysrem, get_sysrem_deep_mask, get_sysrem_ignore_mask
from dataio.edge_trim_manifest import load_accepted_edge_trim_widths
from dataio.hrs_preparation import (
    load_hrs_arm,
    output_dir_for,
    requested_stellar_velocity_correction,
    save_sysrem_products,
    valid_sorted_column_indices,
)
from dataio.wavelength_frame_contract import (
    build_wavelength_frame_contract,
    subset_loader_frame_extras,
)


def transmission_phase_selection_mask(
    phase: np.ndarray,
    *,
    phase_bin: str,
    planet_params: dict[str, Any],
) -> np.ndarray:
    if phase_bin == "all":
        return np.ones_like(phase, dtype=bool)
    return get_phase_bin_mask(phase, phase_bin, planet_params)


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
    active_transit_interval: str,
    jd: np.ndarray,
    bjd_tdb: np.ndarray,
    time_metadata: dict[str, Any],
    contacts: dict[str, float],
    subtract_median: bool,
    run_sysrem: bool,
    regrid: bool,
    arm_edge_trim: dict[str, float | int],
    spectral_column_masking: dict[str, Any],
    product_kind: str,
    out_of_transit_master_division: bool,
    stellar_velocity: dict[str, Any],
    wavelength_frame_contract: dict[str, Any],
    input_exposure_files: list[str],
    excluded_exposure_files: list[str],
    science_exposure_selection: dict[str, Any],
) -> None:
    contacts_serialized: dict[str, float | None] = {}
    for k, v in contacts.items():
        contacts_serialized[k] = float(v) if np.isfinite(v) else None
    model_preprocessing_steps = [
        "fixed_shared_basis_lsd_shadow",
        "active_exposure_mask",
    ]
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
        "input_exposure_files": list(input_exposure_files),
        "excluded_exposure_files": list(excluded_exposure_files),
        "science_exposure_selection": science_exposure_selection,
        "selected_exposure_indices": np.asarray(
            selected_exposure_indices,
            dtype=int,
        ).tolist(),
        "active_transit_interval": str(active_transit_interval),
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
        "fixed_doppler_shadow": {
            "schema_version": 1,
            "enabled": False,
            "required": True,
            "status": "pending_shared_basis_lsd_fit",
        },
        "arm_edge_trim": arm_edge_trim,
        "spectral_column_masking": spectral_column_masking,
        "stellar_velocity": stellar_velocity,
        "wavelength_medium": wavelength_frame_contract["wavelength_medium"],
        "wavelength_frame": wavelength_frame_contract["wavelength_frame"],
        "wavelength_frame_contract": wavelength_frame_contract,
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


def prepare_arm(
    *,
    arm: str,
    args: argparse.Namespace,
    planet_cfg: dict[str, Any],
    output_dir: Path,
) -> None:
    """Prepare and persist one transmission arm using resolved CLI arguments."""

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
    edge_trim_widths_A = None
    edge_trim_source = None
    if args.edge_trim_manifest is not None:
        selected_manifest, edge_trim_widths_A = load_accepted_edge_trim_widths(
            args.edge_trim_manifest,
            planet=args.planet,
            mode="transmission",
            epoch=args.epoch,
            arm=arm,
        )
        edge_trim_source = str(selected_manifest)
    stellar_rest_velocity_kms, stellar_velocity = (
        requested_stellar_velocity_correction(
            enabled=args.apply_stellar_rest,
            mode="transmission",
            planet=args.planet,
            epoch=args.epoch,
            arm=arm,
        )
    )
    collapse_source = args.product_kind == "collapse-source"
    result, extras = load_hrs_arm(
        mode="transmission",
        arm=arm,
        epoch=args.epoch,
        planet=args.planet,
        molecfit=args.molecfit,
        regrid=args.regrid,
        subtract_median=False if collapse_source else args.subtract_median,
        run_sysrem=False if collapse_source else args.run_sysrem,
        stellar_rest_velocity_kms=stellar_rest_velocity_kms,
        edge_trim_widths_A=edge_trim_widths_A,
    )

    wave, data, sigma, jd, snr, exptime, airmass, n_spectra, npix = result
    print(f"Loaded {n_spectra} exposures with {npix} pixels each before selection.")

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
    wave_1d_full = np.asarray(wave[0] if np.asarray(wave).ndim == 2 else wave, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if collapse_source:
        contacts = compute_contact_phases(planet_cfg)
        active_transit = active_transit_mask(phase, planet_cfg)
        out_transit = (
            (phase < contacts["T1"])
            | (phase > contacts["T4"])
        )
        source_rows = out_transit | active_transit
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
        extras = subset_loader_frame_extras(extras, source_rows)
        extras.update({
            "pre_sysrem_flux": np.asarray(data, dtype=float).copy(),
            "pre_sysrem_error": np.asarray(sigma, dtype=float).copy(),
        })
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
                edge_trim_widths_A=edge_trim_widths_A,
            )
            data, sigma, U_sysrem, no_tellurics, diagnostics = sysrem_result
            extras.update(
                {
                    "U_sysrem": U_sysrem,
                    "no_tellurics": no_tellurics,
                    **diagnostics,
                }
            )
    source_phase = np.asarray(phase, dtype=float).copy()
    source_bjd_tdb = np.asarray(bjd_tdb, dtype=float).copy()
    source_data = np.asarray(data, dtype=float).copy()
    source_sigma = np.asarray(sigma, dtype=float).copy()
    contacts = compute_contact_phases(planet_cfg)
    grazing_transit = bool(planet_cfg.get("grazing_transit", False))
    if collapse_source:
        # The dedicated collapse source intentionally omits ingress and
        # egress for ordinary transits. Grazing systems have no T23 interval,
        # so their active model rows retain the observed T14 interval instead.
        active_exposure_mask = active_transit_mask(
            source_phase,
            planet_cfg,
        ).astype(float)
        active_transit_interval = "T14_grazing" if grazing_transit else "T23"
    else:
        # Ordinary preparation derives its temporal median and SYSREM basis
        # from the complete exposure sequence. Preserve the pre-existing T14
        # model support while adding zero-valued out-of-transit source rows;
        # likelihood-row selection is replayed only after filtering.
        active_exposure_mask = (
            (source_phase >= contacts["T1"])
            & (source_phase <= contacts["T4"])
        ).astype(float)
        active_transit_interval = "T14_grazing" if grazing_transit else "T14"
    selection = transmission_phase_selection_mask(
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
        explicit_edge_trim_widths_A=edge_trim_widths_A,
    )
    deep_telluric_mask = get_sysrem_deep_mask(wave_1d, arm)
    edge_trim_info = arm_edge_trim_metadata(
        wave_1d,
        arm,
        planet=args.planet,
        mode="transmission",
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
    # A newly prepared wavelength/exposure grid invalidates any previously
    # projected fixed shadow cube.  The LSD fitter recreates it and updates the
    # metadata only after exact alignment checks.
    (output_dir / "shadow_source_model.npy").unlink(missing_ok=True)

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
        active_transit_interval=np.asarray(active_transit_interval),
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

    save_sysrem_products(
        output_dir,
        wavelength=wave_1d,
        arm=arm,
        n_source_exposures=source_phase.size,
        source_sigma=source_sigma,
        extras=extras,
        run_sysrem=args.run_sysrem,
    )

    contacts = compute_contact_phases(planet_cfg)
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
        active_transit_interval=active_transit_interval,
        jd=jd,
        bjd_tdb=bjd_tdb,
        time_metadata=time_metadata,
        contacts=contacts,
        subtract_median=False if collapse_source else args.subtract_median,
        run_sysrem=args.run_sysrem,
        regrid=args.regrid,
        arm_edge_trim=edge_trim_info,
        spectral_column_masking=spectral_column_masking,
        product_kind=args.product_kind,
        out_of_transit_master_division=collapse_source,
        stellar_velocity=stellar_velocity,
        wavelength_frame_contract=wavelength_frame_contract,
        input_exposure_files=list(extras.get("input_exposure_files", ())),
        excluded_exposure_files=list(extras.get("excluded_exposure_files", ())),
        science_exposure_selection=dict(
            extras.get("science_exposure_selection_metadata", {})
        ),
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
    print(
        "  jd.npy: UTC midpoint from "
        f"{time_metadata['input_keyword']}; bjd_tdb.npy: canonical BJD_TDB"
    )
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

    planet_cfg = get_params(args.planet, args.ephemeris)
    for arm in arms_to_run:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = output_dir_for(
                mode="transmission",
                planet=args.planet,
                epoch=args.epoch,
                arm=arm,
                product_kind=args.product_kind,
            )
        prepare_arm(
            arm=arm,
            args=args,
            planet_cfg=planet_cfg,
            output_dir=output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
