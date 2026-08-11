#!/usr/bin/env python
"""Measure PEPSI stellar velocities with a frozen PySME LSD template."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "atmo_retrieval_matplotlib"),
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
from plotting.style import configure_matplotlib, save_figure_pdf

matplotlib.use("Agg")
configure_matplotlib()
import matplotlib.pyplot as plt
import numpy as np

import config
import config_utils
from dataio.collapse_transmission_timeseries_to_1d import (
    compute_contact_phases,
    get_arm_edge_trim_mask,
    get_bjd_tdb,
    get_ephemeris_epoch_bjd_tdb,
    get_orbital_phase,
    get_pepsi_data,
    get_sysrem_ignore_mask,
)
from dataio.edge_trim_manifest import ACCEPTED_STATUS, load_accepted_edge_trim_manifest
from dataio.stellar_lsd import (
    LSD_EDGE_TRIM_POLICY,
    LSD_METHOD,
    LSD_VELOCITY_STEP_KMS,
    STELLAR_VELOCITY_RESULT_SCHEMA_VERSION,
    SYSTEMIC_VELOCITY_ARM,
    SYSTEMIC_VELOCITY_ARM_POLICY,
    TEMPLATE_MATRIX_RCOND,
    WAVELENGTH_MEDIUM,
    fit_circular_stellar_velocity,
    load_stellar_template,
    mask_regions,
    measure_lsd_exposures,
    relativistic_velocity_difference_kms,
    summarize_velocity,
    validate_quadratic_limb_darkening,
    velocity_to_doppler_factor,
)


BALMER_VACUUM_ANGSTROM = (4862.683, 6564.614)
BALMER_EXCLUSION_KMS = 600.0
DEFAULT_VELOCITY_SPAN_KMS = 180.0
DEFAULT_RESOLVING_POWER = 130000.0
MAX_PROFILE_FRACTIONAL_RMS = 0.10
MAX_PROFILE_ASYMMETRY = 0.15
DEFAULT_SYSTEMATIC_FLOOR_KMS = 0.0


def _planet_slug(planet: str) -> str:
    return planet.strip().lower().replace("-", "").replace(" ", "")


def _raw_dir(planet: str, mode: str, epoch: str) -> Path:
    return config_utils.get_raw_hrs_dir(planet=planet, epoch=epoch, mode=mode)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_edge_trim_calibration(
    *,
    calibration_root: Path,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
) -> dict[str, Any]:
    """Load one accepted edge trim with no zero-trim or older-run fallback."""
    manifest_path, manifest, rows = load_accepted_edge_trim_manifest(
        calibration_root,
        planet=planet,
        mode=mode,
        required_datasets=((epoch, arm),),
    )
    row = rows[(str(epoch), str(arm))]
    left_trim_A = float(row["left_trim_A"])
    right_trim_A = float(row["right_trim_A"])
    if (
        not np.isfinite(left_trim_A)
        or not np.isfinite(right_trim_A)
        or left_trim_A < 0.0
        or right_trim_A < 0.0
    ):
        raise ValueError(
            f"{manifest_path}: invalid accepted edge trim for {mode} {epoch} {arm}."
        )
    return {
        "policy": LSD_EDGE_TRIM_POLICY,
        "status": ACCEPTED_STATUS,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_generated_utc": str(manifest["generated_utc"]),
        "manifest_schema_version": int(manifest["schema_version"]),
        "left_trim_A": left_trim_A,
        "right_trim_A": right_trim_A,
        "calibrated_raw_min_A": float(row["raw_min_A"]),
        "calibrated_raw_max_A": float(row["raw_max_A"]),
        "calibrated_keep_min_A": float(row["keep_min_A"]),
        "calibrated_keep_max_A": float(row["keep_max_A"]),
    }


def _result_path(planet: str, mode: str, epoch: str) -> Path:
    arm_dir = config_utils.get_data_dir(
        planet=planet,
        mode=mode,
        epoch=epoch,
        arm="blue",
    )
    return arm_dir.parent / "stellar_velocity_lsd.json"


def _load_arm(
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    if arm != SYSTEMIC_VELOCITY_ARM:
        raise ValueError(
            f"Systemic-velocity measurements require arm={SYSTEMIC_VELOCITY_ARM!r}; "
            f"received {arm!r}."
        )
    raw_dir = _raw_dir(planet, mode, epoch)
    loaded = get_pepsi_data(
        arm=arm,
        observation_epoch=epoch,
        planet_name=planet,
        do_molecfit=False,
        data_dir=raw_dir,
        regrid=False,
        subtract_median=False,
        run_sysrem=False,
        wavelength_frame="barycentric",
        data_mode=mode,
    )
    if loaded is None:
        raise FileNotFoundError(f"No PEPSI {arm} spectra found in {raw_dir}.")
    result, extras = loaded
    return result, extras


def _phases(
    jd_utc: np.ndarray,
    extras: dict[str, Any],
    params: dict[str, Any],
    *,
    observation_epoch: str,
) -> tuple[np.ndarray, np.ndarray]:
    bjd_tdb = get_bjd_tdb(
        jd_utc,
        str(params["RA"]),
        str(params["Dec"]),
        header_bjd_tdb=extras.get("header_bjd_tdb"),
    )
    epoch_bjd_tdb = get_ephemeris_epoch_bjd_tdb(
        float(params["epoch"]),
        str(params.get("epoch_scale")),
        str(params.get("epoch_reference")),
    )
    midpoint_bjd_tdb = config_utils.resolve_transit_midpoint(
        bjd_tdb,
        params,
        reference_epoch_bjd_tdb=epoch_bjd_tdb,
        observation_epoch=observation_epoch,
    )
    phase = get_orbital_phase(
        bjd_tdb,
        midpoint_bjd_tdb,
        float(params["period"]),
    )
    return np.asarray(bjd_tdb, dtype=float), np.asarray(phase, dtype=float)


def _lsd_pixel_masks(
    wavelength: np.ndarray,
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
    edge_trim_widths_A: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float)
    masks = np.zeros(wavelength.shape, dtype=bool)
    edge_masks = np.zeros(wavelength.shape, dtype=bool)
    balmer_regions = [
        (
            line * float(velocity_to_doppler_factor(-BALMER_EXCLUSION_KMS)),
            line * float(velocity_to_doppler_factor(BALMER_EXCLUSION_KMS)),
        )
        for line in BALMER_VACUUM_ANGSTROM
    ]
    for index, row in enumerate(wavelength):
        valid_wavelength = np.isfinite(row) & (row > 0.0)
        if np.any(valid_wavelength):
            valid_row = row[valid_wavelength]
            masks[index, valid_wavelength] = get_sysrem_ignore_mask(
                valid_row,
                arm,
                explicit_edge_trim_widths_A=edge_trim_widths_A,
            )
            edge_masks[index, valid_wavelength] = get_arm_edge_trim_mask(
                valid_row,
                arm,
                explicit_widths_A=edge_trim_widths_A,
            )
        masks[index] |= mask_regions(row, balmer_regions)
    return masks, edge_masks


def _measure_arm(
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
    params: dict[str, Any],
    template: dict[str, Any],
    velocity_grid: np.ndarray,
    vsini_kms: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    resolving_power: float,
    include_in_transit: bool,
    max_profile_fractional_rms: float,
    max_profile_asymmetry: float,
    edge_trim_calibration: dict[str, Any],
    diagnostics_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"\nMeasuring {planet} {mode} {epoch} {arm}...")
    loaded, extras = _load_arm(
        planet=planet,
        mode=mode,
        epoch=epoch,
        arm=arm,
    )
    wavelength, flux, error, jd, _snr, _exptime, _airmass, n_exposures, _npix = loaded
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    if extras.get("wavelength_medium") != WAVELENGTH_MEDIUM:
        raise ValueError(
            f"PEPSI loader returned wavelength_medium={extras.get('wavelength_medium')!r}; "
            f"expected {WAVELENGTH_MEDIUM!r}."
        )
    if extras.get("wavelength_frame") != "barycentric":
        raise ValueError(
            f"PEPSI loader returned wavelength_frame={extras.get('wavelength_frame')!r}; "
            "the mean stellar velocity requires a barycentric grid."
        )
    bjd_tdb, phase = _phases(
        np.asarray(jd, dtype=float),
        extras,
        params,
        observation_epoch=epoch,
    )
    if mode == "transmission" and not include_in_transit:
        contacts = compute_contact_phases(params)
        selected_by_observation = (phase < contacts["T1"]) | (
            phase > contacts["T4"]
        )
        selection_name = "out_of_transit"
    else:
        selected_by_observation = np.ones(n_exposures, dtype=bool)
        selection_name = "all_exposures"

    edge_trim_widths_A = (
        float(edge_trim_calibration["left_trim_A"]),
        float(edge_trim_calibration["right_trim_A"]),
    )
    pixel_masks, edge_trim_pixel_masks = _lsd_pixel_masks(
        wavelength,
        planet=planet,
        mode=mode,
        epoch=epoch,
        arm=arm,
        edge_trim_widths_A=edge_trim_widths_A,
    )
    measured = measure_lsd_exposures(
        wavelength,
        flux,
        error,
        template_wavelength=template["wavelength"],
        template_flux=template["flux"],
        velocity_grid=velocity_grid,
        vsini_kms=vsini_kms,
        limb_darkening_u1=limb_darkening_u1,
        limb_darkening_u2=limb_darkening_u2,
        resolving_power=resolving_power,
        pixel_masks=pixel_masks,
    )
    profile_qc_passed = (
        np.isfinite(measured["centroid_kms"])
        & np.isfinite(measured["centroid_err_kms"])
        & (measured["centroid_err_kms"] > 0.0)
        & np.isfinite(measured["fractional_model_rms"])
        & (
            measured["fractional_model_rms"]
            <= float(max_profile_fractional_rms)
        )
        & np.isfinite(measured["profile_asymmetry"])
        & (measured["profile_asymmetry"] <= float(max_profile_asymmetry))
    )
    selected = selected_by_observation & profile_qc_passed
    arm_measurement_valid = bool(np.count_nonzero(selected) >= 2)

    stellar_removed_kms = np.asarray(
        extras["stellar_velocity_removed_kms"],
        dtype=float,
    )
    observer_removed_kms = np.asarray(
        extras["observer_velocity_removed_kms"],
        dtype=float,
    )
    instrument_removed_kms = np.asarray(
        extras["instrument_velocity_removed_kms"],
        dtype=float,
    )
    lsd_residual_kms = relativistic_velocity_difference_kms(
        measured["centroid_kms"],
        stellar_removed_kms,
    )
    if arm_measurement_valid:
        summary = summarize_velocity(
            measured["centroid_kms"],
            measured["centroid_err_kms"],
            selected,
        )
        residual_summary = summarize_velocity(
            lsd_residual_kms,
            measured["centroid_err_kms"],
            selected,
        )
    else:
        summary = {
            "residual_velocity_kms": None,
            "residual_velocity_err_kms": None,
            "formal_weighted_mean_err_kms": None,
            "exposure_rms_kms": None,
            "n_exposures_used": int(np.count_nonzero(selected)),
            "used_exposure_indices": np.flatnonzero(selected).astype(int).tolist(),
        }
        residual_summary = dict(summary)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    profile_path = diagnostics_dir / f"{mode}_{epoch}_{arm}_profiles.npz"
    np.savez_compressed(
        profile_path,
        velocity_kms=measured["velocity_kms"],
        profiles=measured["profiles"],
        profile_uncertainties=measured["profile_uncertainties"],
        profile_models=measured["profile_models"],
        amplitude=measured["amplitude"],
        offset=measured["offset"],
        centroid_kms=measured["centroid_kms"],
        centroid_err_kms=measured["centroid_err_kms"],
        systemic_velocity_kms=measured["centroid_kms"],
        lsd_residual_velocity_kms=lsd_residual_kms,
        pipeline_removed_stellar_velocity_kms=stellar_removed_kms,
        observer_velocity_removed_kms=observer_removed_kms,
        instrument_velocity_removed_kms=instrument_removed_kms,
        reduced_chi2=measured["reduced_chi2"],
        fractional_model_rms=measured["fractional_model_rms"],
        profile_asymmetry=measured["profile_asymmetry"],
        profile_qc_passed=profile_qc_passed,
        selected_by_observation=selected_by_observation,
        phase=phase,
        bjd_tdb=bjd_tdb,
        selected=selected,
        effective_rank=measured["effective_rank"],
        n_pixels=measured["n_pixels"],
        normal_matrix_condition_number=measured[
            "normal_matrix_condition_number"
        ],
        normal_matrix_rcond=measured["normal_matrix_rcond"],
        pixel_mask=pixel_masks,
        edge_trim_pixel_mask=edge_trim_pixel_masks,
        edge_trim_widths_A=np.asarray(edge_trim_widths_A, dtype=float),
    )
    profile_qc_plot_path = diagnostics_dir / f"{mode}_{epoch}_{arm}_profile_qc.pdf"
    _plot_arm_profile_qc(
        measured=measured,
        profile_qc_passed=profile_qc_passed,
        maximum_fractional_model_rms=float(max_profile_fractional_rms),
        maximum_profile_asymmetry=float(max_profile_asymmetry),
        title=f"{planet} | {mode} {epoch} | {arm}",
        output_path=profile_qc_plot_path,
    )

    edge_trim_result = {
        **edge_trim_calibration,
        "n_edge_trimmed_pixels_total": int(np.count_nonzero(edge_trim_pixel_masks)),
        "median_edge_trimmed_pixels_per_exposure": float(
            np.median(np.count_nonzero(edge_trim_pixel_masks, axis=1))
        ),
    }
    result: dict[str, Any] = {
        "arm": arm,
        "input_product": "normalized_raw",
        "wavelength_frame": "barycentric",
        "science_exposure_selection": dict(
            extras.get("science_exposure_selection_metadata", {})
        ),
        "edge_trim_policy": LSD_EDGE_TRIM_POLICY,
        "edge_trim_calibration": edge_trim_result,
        "selection": selection_name,
        "arm_measurement_valid": arm_measurement_valid,
        "arm_systemic_velocity_kms": summary["residual_velocity_kms"],
        "arm_systemic_velocity_err_kms": summary["residual_velocity_err_kms"],
        "arm_lsd_residual_velocity_kms": residual_summary["residual_velocity_kms"],
        "arm_lsd_residual_velocity_err_kms": residual_summary[
            "residual_velocity_err_kms"
        ],
        "pipeline_removed_stellar_velocity_kms": float(
            np.nanmedian(stellar_removed_kms)
        ),
        "observer_velocity_removed_kms": float(
            np.nanmedian(observer_removed_kms)
        ),
        "instrument_velocity_removed_kms": float(
            np.nanmedian(instrument_removed_kms)
        ),
        "velocity_history_recipe": sorted(
            {"+".join(recipe) for recipe in extras["velocity_history_recipe"]}
        ),
        "wavelength_frame_source": sorted(set(extras["wavelength_frame_source"])),
        "formal_weighted_mean_err_kms": summary["formal_weighted_mean_err_kms"],
        "exposure_rms_kms": summary["exposure_rms_kms"],
        "n_exposures": int(n_exposures),
        "n_exposures_selected_by_observation": int(
            np.count_nonzero(selected_by_observation)
        ),
        "n_exposures_used": summary["n_exposures_used"],
        "n_profile_fits_succeeded": int(
            np.count_nonzero(np.isfinite(measured["centroid_kms"]))
        ),
        "median_fractional_model_rms": float(
            np.nanmedian(measured["fractional_model_rms"])
        ),
        "median_profile_asymmetry": float(
            np.nanmedian(measured["profile_asymmetry"])
        ),
        "used_exposure_indices": summary["used_exposure_indices"],
        "profile_file": str(profile_path.resolve()),
        "profile_qc_plot": str(profile_qc_plot_path.resolve()),
    }
    arrays = {
        "velocity_kms": measured["velocity_kms"],
        "profiles": measured["profiles"],
        "profile_models": measured["profile_models"],
        "centroid_kms": measured["centroid_kms"],
        "centroid_err_kms": measured["centroid_err_kms"],
        "lsd_residual_velocity_kms": lsd_residual_kms,
        "pipeline_removed_stellar_velocity_kms": stellar_removed_kms,
        "profile_qc_passed": profile_qc_passed,
        "fractional_model_rms": measured["fractional_model_rms"],
        "profile_asymmetry": measured["profile_asymmetry"],
        "pixel_mask": pixel_masks,
        "edge_trim_pixel_mask": edge_trim_pixel_masks,
        "bjd_tdb": bjd_tdb,
        "phase": phase,
        "selected": selected,
        "selected_by_observation": selected_by_observation,
    }
    if arm_measurement_valid:
        print(
            f"  {arm}: barycentric RV={summary['residual_velocity_kms']:+.3f} +/- "
            f"{summary['residual_velocity_err_kms']:.3f} km/s "
            f"({summary['n_exposures_used']}/{n_exposures} profiles pass QC; "
            f"RMS={summary['exposure_rms_kms']:.3f})"
        )
    else:
        print(
            f"  {arm}: rejected; {summary['n_exposures_used']}/{n_exposures} "
            "profiles pass observation and shape QC."
        )
    return result, arrays


def _validate_blue_arm(
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Accept a dataset only from a valid blue-arm measurement."""
    unexpected_arms = sorted(set(arms) - {SYSTEMIC_VELOCITY_ARM})
    if unexpected_arms:
        raise ValueError(
            "Systemic-velocity measurements are blue-only; unexpected arm results: "
            + ", ".join(unexpected_arms)
        )
    blue = arms.get(SYSTEMIC_VELOCITY_ARM)
    if blue is None or blue.get("arm_measurement_valid") is not True:
        return {
            "blue_arm_validation_passed": False,
            "status": "blue arm must have at least two QC-passing profiles",
            "epoch_systemic_velocity_kms": None,
            "epoch_systemic_velocity_err_kms": None,
        }
    return {
        "blue_arm_validation_passed": True,
        "status": "blue arm accepted",
        "epoch_systemic_velocity_kms": blue["arm_systemic_velocity_kms"],
        "epoch_systemic_velocity_err_kms": blue["arm_systemic_velocity_err_kms"],
    }


def _blue_only_exposures(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return blue-arm rows and reject any attempt to supply another arm."""
    blue_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("arm") != SYSTEMIC_VELOCITY_ARM:
            raise ValueError(
                "Systemic-velocity joint inputs must contain blue-arm rows only; "
                f"received arm={row.get('arm')!r}."
            )
        output = dict(row)
        output["arms_used"] = SYSTEMIC_VELOCITY_ARM
        output["n_arms"] = 1
        output["systemic_velocity_arm_policy"] = SYSTEMIC_VELOCITY_ARM_POLICY
        blue_rows.append(output)
    return sorted(blue_rows, key=lambda row: row["bjd_tdb"])


def _write_dataset_result(dataset: dict[str, Any]) -> None:
    path = _result_path(dataset["planet"], dataset["mode"], dataset["epoch"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved {path}")


def _plot_arm_profile_qc(
    *,
    measured: dict[str, Any],
    profile_qc_passed: np.ndarray,
    maximum_fractional_model_rms: float,
    maximum_profile_asymmetry: float,
    title: str,
    output_path: Path,
) -> None:
    """Plot the profile shapes and numerical quantities used by arm-level QC."""
    passed = np.asarray(profile_qc_passed, dtype=bool)
    finite_profiles = np.all(np.isfinite(measured["profiles"]), axis=1)
    groups = (
        (passed & finite_profiles, "QC pass", "#2a8f5b"),
        ((~passed) & finite_profiles, "QC reject", "#c43c35"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    velocity = measured["velocity_kms"]
    for selected, label, color in groups:
        if not np.any(selected):
            continue
        axes[0, 0].plot(
            velocity,
            np.nanmedian(measured["profiles"][selected], axis=0),
            color=color,
            lw=1.5,
            label=f"{label} profile (n={np.count_nonzero(selected)})",
        )
        axes[0, 0].plot(
            velocity,
            np.nanmedian(measured["profile_models"][selected], axis=0),
            color=color,
            lw=1.2,
            ls="--",
            label=f"{label} quadratic rotation+Gaussian fit",
        )

    exposure = np.arange(passed.size)
    point_colors = np.where(passed, "#2a8f5b", "#c43c35")
    axes[0, 1].scatter(
        exposure,
        measured["fractional_model_rms"],
        c=point_colors,
        s=20,
    )
    axes[0, 1].axhline(maximum_fractional_model_rms, color="black", ls="--", lw=1)
    axes[1, 0].scatter(
        exposure,
        measured["profile_asymmetry"],
        c=point_colors,
        s=20,
    )
    axes[1, 0].axhline(maximum_profile_asymmetry, color="black", ls="--", lw=1)
    axes[1, 1].scatter(
        exposure,
        measured["effective_rank"],
        c=point_colors,
        s=20,
    )

    axes[0, 0].set(
        xlabel="Barycentric velocity (km/s)",
        ylabel="LSD absorption",
        title="Median recovered profile and fitted model",
    )
    axes[0, 1].set(
        xlabel="Exposure index",
        ylabel="RMS / fitted amplitude",
        title="Rotational-profile residual",
    )
    axes[1, 0].set(
        xlabel="Exposure index",
        ylabel="Mirrored-profile asymmetry",
        title="Profile asymmetry",
    )
    axes[1, 1].set(
        xlabel="Exposure index",
        ylabel="Retained matrix modes",
        title="Template-deconvolution effective rank",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{title} | {np.count_nonzero(passed)}/{passed.size} profiles pass QC",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pdf(fig, output_path, dpi=180)
    plt.close(fig)


def _plot_dataset(
    dataset: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    if set(arrays) - {SYSTEMIC_VELOCITY_ARM}:
        raise ValueError("Systemic-velocity dataset plots accept blue-arm arrays only.")
    if SYSTEMIC_VELOCITY_ARM in arrays:
        values = arrays[SYSTEMIC_VELOCITY_ARM]
        selected = values["selected"] & np.isfinite(values["centroid_kms"])
        if np.any(selected):
            color = "#2474b5"
            mean_profile = np.nanmedian(values["profiles"][selected], axis=0)
            model = np.nanmedian(values["profile_models"][selected], axis=0)
            axes[0].plot(
                values["velocity_kms"],
                mean_profile,
                color=color,
                label="blue LSD",
            )
            axes[0].plot(
                values["velocity_kms"],
                model,
                color=color,
                ls="--",
                alpha=0.8,
            )
            axes[1].errorbar(
                values["phase"][selected],
                values["centroid_kms"][selected],
                yerr=values["centroid_err_kms"][selected],
                fmt="o",
                ms=4,
                capsize=2,
                color=color,
                label="blue",
            )
            excluded = (~values["selected"]) & np.isfinite(values["centroid_kms"])
            if np.any(excluded):
                axes[1].scatter(
                    values["phase"][excluded],
                    values["centroid_kms"][excluded],
                    facecolors="none",
                    edgecolors=color,
                    s=24,
                    alpha=0.6,
                )

    if dataset.get("epoch_systemic_velocity_kms") is not None:
        axes[1].axhline(
            dataset["epoch_systemic_velocity_kms"],
            color="black",
            lw=1.2,
            label="blue-arm epoch RV",
        )
    axes[0].axvline(0.0, color="0.5", lw=0.8)
    axes[0].set(xlabel="Velocity (km/s)", ylabel="LSD absorption", title="Mean selected profiles")
    axes[1].set(
        xlabel="Orbital phase",
        ylabel="Barycentric stellar RV (km/s)",
        title="Exposure velocities",
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    status = dataset["status"]
    if dataset["blue_arm_validation_passed"]:
        status += (
            f"; epoch RV={dataset['epoch_systemic_velocity_kms']:+.3f} +/- "
            f"{dataset['epoch_systemic_velocity_err_kms']:.3f} km/s"
        )
    fig.suptitle(
        f"{dataset['planet']} | {dataset['mode']} {dataset['epoch']} | {status}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pdf(fig, output_path, dpi=180)
    plt.close(fig)


def _format_velocity(value: float | None, error: float | None = None) -> str:
    if value is None:
        return "--"
    if error is None:
        return f"{value:+.3f}"
    return f"{value:+.3f} +/- {error:.3f}"


def _measurement_rows(
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    selected = np.asarray(arrays["selected"], dtype=bool)
    finite_fit = (
        np.isfinite(arrays["centroid_kms"])
        & np.isfinite(arrays["centroid_err_kms"])
        & (arrays["centroid_err_kms"] > 0.0)
    )
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(finite_fit):
        rows.append(
            {
                "planet": planet,
                "mode": mode,
                "epoch": epoch,
                "arm": arm,
                "exposure_index": int(index),
                "bjd_tdb": float(arrays["bjd_tdb"][index]),
                "orbital_phase": float(arrays["phase"][index]),
                "pipeline_removed_stellar_velocity_kms": float(
                    arrays["pipeline_removed_stellar_velocity_kms"][index]
                ),
                "lsd_residual_velocity_kms": float(
                    arrays["lsd_residual_velocity_kms"][index]
                ),
                "stellar_rv_barycentric_kms": float(arrays["centroid_kms"][index]),
                "stellar_rv_stat_err_kms": float(
                    arrays["centroid_err_kms"][index]
                ),
                "fractional_model_rms": float(
                    arrays["fractional_model_rms"][index]
                ),
                "profile_asymmetry": float(arrays["profile_asymmetry"][index]),
                "profile_qc_passed": bool(arrays["profile_qc_passed"][index]),
                "selected_by_observation": bool(
                    arrays["selected_by_observation"][index]
                ),
                "selected_for_rv": bool(selected[index]),
            }
        )
    return rows


def _write_summary(
    datasets: list[dict[str, Any]],
    joint_result: dict[str, Any],
    blue_measurement_rows: list[dict[str, Any]],
    joint_measurement_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stellar_velocity_results.csv"
    fieldnames = [
        "planet",
        "mode",
        "epoch",
        "blue_systemic_velocity_kms",
        "blue_error_kms",
        "epoch_systemic_velocity_kms",
        "epoch_systemic_velocity_err_kms",
        "blue_arm_validation_passed",
        "accepted_for_stellar_rest",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            blue = dataset["arms"].get("blue", {})
            writer.writerow(
                {
                    "planet": dataset["planet"],
                    "mode": dataset["mode"],
                    "epoch": dataset["epoch"],
                    "blue_systemic_velocity_kms": blue.get(
                        "arm_systemic_velocity_kms"
                    ),
                    "blue_error_kms": blue.get("arm_systemic_velocity_err_kms"),
                    "epoch_systemic_velocity_kms": dataset.get(
                        "epoch_systemic_velocity_kms"
                    ),
                    "epoch_systemic_velocity_err_kms": dataset.get(
                        "epoch_systemic_velocity_err_kms"
                    ),
                    "blue_arm_validation_passed": dataset[
                        "blue_arm_validation_passed"
                    ],
                    "accepted_for_stellar_rest": dataset[
                        "accepted_for_stellar_rest"
                    ],
                    "status": dataset["status"],
                }
            )

    markdown = [
        "# Stellar LSD velocity results",
        "",
        "",
        f"Joint circular fit: gamma = {joint_result['mean_stellar_velocity_kms']:+.3f} "
        f"+/- {joint_result['systemic_velocity_err_kms']:.3f} km/s "
        f"(statistical {joint_result['mean_stellar_velocity_stat_err_kms']:.3f}, "
        f"systematic floor {joint_result['systematic_error_floor_kms']:.3f}); "
        f"K_star = {joint_result['stellar_rv_semiamplitude_kms']:+.3f} +/- "
        f"{joint_result['stellar_rv_semiamplitude_err_kms']:.3f} km/s; "
        f"jitter = {joint_result['rv_jitter_kms']:.3f} km/s.",
        "",
        "Systemic-velocity arm policy: **blue only, with no red-arm fallback**.",
        "",
        "| Dataset | Blue barycentric RV (km/s) | Status |",
        "|---|---:|---|",
    ]
    for dataset in datasets:
        blue = dataset["arms"].get("blue", {})
        markdown.append(
            "| "
            f"{dataset['mode']} {dataset['epoch']} | "
            f"{_format_velocity(blue.get('arm_systemic_velocity_kms'), blue.get('arm_systemic_velocity_err_kms'))} | "
            f"{dataset['status']} |"
        )
    (output_dir / "stellar_velocity_results.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    fig_height = max(3.5, 0.65 * len(datasets) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height), constrained_layout=True)
    y = np.arange(len(datasets), dtype=float)
    blue_label_written = False
    for index, dataset in enumerate(datasets):
        if dataset["blue_arm_validation_passed"]:
            ax.errorbar(
                dataset["epoch_systemic_velocity_kms"],
                y[index],
                xerr=dataset["epoch_systemic_velocity_err_kms"],
                fmt="o",
                ms=5,
                capsize=3,
                color="#2474b5",
                label="blue arm" if not blue_label_written else None,
            )
            blue_label_written = True
    ax.axvline(
        joint_result["mean_stellar_velocity_kms"],
        color="black",
        lw=1.0,
        ls=":",
        label="joint gamma",
    )
    ax.set_yticks(y, [f"{item['mode']} {item['epoch']}" for item in datasets])
    ax.invert_yaxis()
    ax.set_xlabel("Barycentric stellar velocity (km/s)")
    ax.set_title(f"{datasets[0]['planet']} LSD stellar velocity by dataset")
    ax.legend(frameon=False)
    results_plot_path = save_figure_pdf(
        fig, output_dir / "stellar_velocity_results.pdf", dpi=180
    )
    plt.close(fig)

    exposure_path = output_dir / "stellar_velocity_blue_exposures.csv"
    with exposure_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in blue_measurement_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(blue_measurement_rows)

    joint_input_path = output_dir / "stellar_velocity_joint_inputs.csv"
    with joint_input_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in joint_measurement_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joint_measurement_rows)

    summary_payload = {
        "schema_version": STELLAR_VELOCITY_RESULT_SCHEMA_VERSION,
        "method": LSD_METHOD,
        "planet": datasets[0]["planet"],
        "ephemeris": datasets[0]["ephemeris"],
        "reference_frame": "PEPSI barycentric vacuum",
        "wavelength_frame": "barycentric",
        "wavelength_medium": WAVELENGTH_MEDIUM,
        "template_path": datasets[0]["template_path"],
        "template_sha256": datasets[0]["template_sha256"],
        "template_metadata_path": datasets[0]["template_metadata_path"],
        "template_metadata_sha256": datasets[0]["template_metadata_sha256"],
        "template_parameters": datasets[0]["template_parameters"],
        "n_template_pixels": datasets[0]["n_template_pixels"],
        "template_matrix_rcond": datasets[0]["template_matrix_rcond"],
        "resolving_power": datasets[0]["resolving_power"],
        "limb_darkening_law": datasets[0]["limb_darkening_law"],
        "limb_darkening_u1": datasets[0]["limb_darkening_u1"],
        "limb_darkening_u2": datasets[0]["limb_darkening_u2"],
        "systemic_velocity_arm": SYSTEMIC_VELOCITY_ARM,
        "systemic_velocity_arm_policy": SYSTEMIC_VELOCITY_ARM_POLICY,
        "edge_trim_policy": LSD_EDGE_TRIM_POLICY,
        "edge_trim_calibrations": [
            {
                "mode": dataset["mode"],
                "epoch": dataset["epoch"],
                **dataset["edge_trim_calibration"],
            }
            for dataset in datasets
        ],
        "science_exposure_selections": [
            {
                "mode": dataset["mode"],
                "epoch": dataset["epoch"],
                **dataset["science_exposure_selection"],
            }
            for dataset in datasets
        ],
        "orbit_model": "gamma + K_star * sin(2*pi*phase); circular orbit",
        "systemic_velocity_kms": joint_result["mean_stellar_velocity_kms"],
        "systemic_velocity_stat_err_kms": joint_result[
            "mean_stellar_velocity_stat_err_kms"
        ],
        "mean_stellar_velocity_kms": joint_result["mean_stellar_velocity_kms"],
        "mean_stellar_velocity_stat_err_kms": joint_result[
            "mean_stellar_velocity_stat_err_kms"
        ],
        "systemic_velocity_err_kms": joint_result["systemic_velocity_err_kms"],
        "systematic_error_floor_kms": joint_result["systematic_error_floor_kms"],
        "stellar_rv_semiamplitude_kms": joint_result[
            "stellar_rv_semiamplitude_kms"
        ],
        "stellar_rv_semiamplitude_err_kms": joint_result[
            "stellar_rv_semiamplitude_err_kms"
        ],
        "rv_jitter_kms": joint_result["rv_jitter_kms"],
        "chi2": joint_result["chi2"],
        "dof": joint_result["dof"],
        "reduced_chi2": joint_result["reduced_chi2"],
        "n_measurements": joint_result["n_measurements"],
        "accepted_for_stellar_rest": True,
        "exposure_table": str(exposure_path.resolve()),
        "joint_input_table": str(joint_input_path.resolve()),
    }
    summary_path = output_dir / "systemic_velocity_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    if blue_measurement_rows:
        ax.errorbar(
            [row["orbital_phase"] for row in blue_measurement_rows],
            [row["stellar_rv_barycentric_kms"] for row in blue_measurement_rows],
            yerr=[row["stellar_rv_stat_err_kms"] for row in blue_measurement_rows],
            fmt="o",
            ms=3,
            alpha=0.7,
            color="#2474b5",
            label="blue-arm measurements",
        )
    ax.errorbar(
        [row["orbital_phase"] for row in joint_measurement_rows],
        [row["stellar_rv_barycentric_kms"] for row in joint_measurement_rows],
        yerr=[row["stellar_rv_stat_err_kms"] for row in joint_measurement_rows],
        fmt="D",
        ms=3,
        alpha=0.9,
        color="black",
        label="joint-fit input",
    )
    phase_grid = np.linspace(
        min(row["orbital_phase"] for row in joint_measurement_rows),
        max(row["orbital_phase"] for row in joint_measurement_rows),
        500,
    )
    model_grid = (
        joint_result["mean_stellar_velocity_kms"]
        + joint_result["stellar_rv_semiamplitude_kms"]
        * np.sin(2.0 * np.pi * phase_grid)
    )
    ax.plot(phase_grid, model_grid, color="black", lw=1.5, label="joint circular fit")
    ax.set(
        xlabel="Orbital phase",
        ylabel="Barycentric stellar RV (km/s)",
        title=(
            f"{datasets[0]['planet']} mean stellar velocity: "
            f"{joint_result['mean_stellar_velocity_kms']:+.3f} +/- "
            f"{joint_result['systemic_velocity_err_kms']:.3f} km/s"
        ),
    )
    ax.legend(frameon=False)
    fit_plot_path = save_figure_pdf(
        fig, output_dir / "systemic_velocity_fit.pdf", dpi=180
    )
    plt.close(fig)
    print(f"\nResults table: {csv_path}")
    print(f"Results display: {results_plot_path}")
    print(f"Mean stellar velocity: {summary_path}")
    print(f"Joint fit display: {fit_plot_path}")


def _discover_datasets(
    *,
    planet: str,
    modes: tuple[str, ...],
    requested_epochs: set[str] | None,
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for mode in modes:
        root = config_utils.get_raw_hrs_dir(planet=planet, mode=mode)
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or len(path.name) != 8 or not path.name.isdigit():
                continue
            if requested_epochs is not None and path.name not in requested_epochs:
                continue
            found.append((mode, path.name))
    return found


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one blue-arm stellar-rest correction per PEPSI dataset "
            "with no red-arm fallback, using a frozen PySME spectrum and a "
            "fixed quadratic-limb-darkened rotation-plus-Gaussian profile."
        )
    )
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default="Duck24")
    parser.add_argument(
        "--mode",
        choices=("transmission", "emission", "both"),
        default="both",
    )
    parser.add_argument(
        "--epoch",
        action="append",
        help="Epoch to measure; repeat as needed. Omit to discover every raw epoch.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Frozen intrinsic PySME .npz spectrum with its .json provenance sidecar.",
    )
    parser.add_argument(
        "--edge-trim-calibration-root",
        type=Path,
        required=True,
        help=(
            "Root containing accepted edge-trim calibration manifests. The newest "
            "run for every requested mode must be accepted and contain every "
            "requested epoch; zero-trim and older-run fallbacks are disabled."
        ),
    )
    parser.add_argument("--vsini-kms", type=float, default=None)
    parser.add_argument(
        "--limb-darkening-u1",
        type=float,
        default=None,
        help="Quadratic limb-darkening coefficient u1 (default: configured gamma1).",
    )
    parser.add_argument(
        "--limb-darkening-u2",
        type=float,
        default=None,
        help="Quadratic limb-darkening coefficient u2 (default: configured gamma2).",
    )
    parser.add_argument(
        "--resolving-power",
        type=float,
        default=DEFAULT_RESOLVING_POWER,
        help="PEPSI resolving power used for the fixed instrumental Gaussian.",
    )
    parser.add_argument(
        "--include-in-transit",
        action="store_true",
        help=(
            "Include transmission spectra affected by the Rossiter-McLaughlin "
            "profile distortion. The default follows the LSD RV method and "
            "uses only phases outside first/fourth contact."
        ),
    )
    parser.add_argument(
        "--max-profile-fractional-rms",
        type=float,
        default=MAX_PROFILE_FRACTIONAL_RMS,
        help="Reject a rotational-profile fit above this RMS/profile-depth ratio.",
    )
    parser.add_argument(
        "--max-profile-asymmetry",
        type=float,
        default=MAX_PROFILE_ASYMMETRY,
        help="Reject a rotational-profile fit above this mirrored-profile metric.",
    )
    parser.add_argument(
        "--systematic-error-floor-kms",
        type=float,
        default=DEFAULT_SYSTEMATIC_FLOOR_KMS,
        help=(
            "Optional correlated floor added in quadrature to the final gamma "
            "uncertainty (default: 0; no imposed floor)."
        ),
    )
    parser.add_argument("--velocity-span-kms", type=float, default=DEFAULT_VELOCITY_SPAN_KMS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diagnostic plots/tables directory (default: diagnostics/stellar_velocity/<planet>).",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    params = config_utils.get_params(args.planet, args.ephemeris)
    vsini = float(args.vsini_kms if args.vsini_kms is not None else params.get("v_sini_star", np.nan))
    limb_darkening_u1 = float(
        args.limb_darkening_u1
        if args.limb_darkening_u1 is not None
        else params.get("gamma1", np.nan)
    )
    limb_darkening_u2 = float(
        args.limb_darkening_u2
        if args.limb_darkening_u2 is not None
        else params.get("gamma2", np.nan)
    )
    if not np.isfinite(vsini) or vsini <= 0.0:
        raise ValueError("A finite positive --vsini-kms or configured v_sini_star is required.")
    validate_quadratic_limb_darkening(limb_darkening_u1, limb_darkening_u2)
    if args.velocity_span_kms <= vsini:
        raise ValueError("--velocity-span-kms must be larger than v sin(i).")
    if args.resolving_power <= 0.0:
        raise ValueError("--resolving-power must be positive.")
    if args.max_profile_fractional_rms <= 0.0:
        raise ValueError("--max-profile-fractional-rms must be positive.")
    if args.max_profile_asymmetry <= 0.0:
        raise ValueError("--max-profile-asymmetry must be positive.")
    if args.systematic_error_floor_kms < 0.0:
        raise ValueError("--systematic-error-floor-kms must be non-negative.")

    template = load_stellar_template(args.template)
    velocity_grid = np.arange(
        -args.velocity_span_kms,
        args.velocity_span_kms + 0.5 * LSD_VELOCITY_STEP_KMS,
        LSD_VELOCITY_STEP_KMS,
    )
    modes = ("transmission", "emission") if args.mode == "both" else (args.mode,)
    datasets_to_run = _discover_datasets(
        planet=args.planet,
        modes=modes,
        requested_epochs=None if args.epoch is None else set(args.epoch),
    )
    if not datasets_to_run:
        raise FileNotFoundError("No matching raw PEPSI datasets were found.")
    edge_trim_calibrations: dict[tuple[str, str], dict[str, Any]] = {}
    for mode, epoch in datasets_to_run:
        try:
            edge_trim_calibrations[(mode, epoch)] = _load_edge_trim_calibration(
                calibration_root=args.edge_trim_calibration_root,
                planet=args.planet,
                mode=mode,
                epoch=epoch,
                arm=SYSTEMIC_VELOCITY_ARM,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"No acceptance-grade edge trim is available for "
                f"{args.planet} {mode} {epoch} {SYSTEMIC_VELOCITY_ARM}; "
                "untrimmed LSD fallback is disabled."
            ) from exc
    output_dir = args.output_dir or (
        config.PROJECT_ROOT / "diagnostics" / "stellar_velocity" / _planet_slug(args.planet)
    )

    datasets: list[dict[str, Any]] = []
    blue_measurements: list[dict[str, Any]] = []
    validated_blue_measurements: list[dict[str, Any]] = []
    validated_dataset_keys: set[tuple[str, str]] = set()
    for mode, epoch in datasets_to_run:
        edge_trim_calibration = edge_trim_calibrations[(mode, epoch)]
        print(
            f"\nEdge trim {mode} {epoch} blue: "
            f"left={edge_trim_calibration['left_trim_A']:.1f} A, "
            f"right={edge_trim_calibration['right_trim_A']:.1f} A "
            f"from {edge_trim_calibration['manifest_path']}"
        )
        arm_results: dict[str, dict[str, Any]] = {}
        arm_arrays: dict[str, dict[str, np.ndarray]] = {}
        dataset_blue_measurements: list[dict[str, Any]] = []
        try:
            result, arrays = _measure_arm(
                planet=args.planet,
                mode=mode,
                epoch=epoch,
                arm=SYSTEMIC_VELOCITY_ARM,
                params=params,
                template=template,
                velocity_grid=velocity_grid,
                vsini_kms=vsini,
                limb_darkening_u1=limb_darkening_u1,
                limb_darkening_u2=limb_darkening_u2,
                resolving_power=float(args.resolving_power),
                include_in_transit=args.include_in_transit,
                max_profile_fractional_rms=args.max_profile_fractional_rms,
                max_profile_asymmetry=args.max_profile_asymmetry,
                edge_trim_calibration=edge_trim_calibration,
                diagnostics_dir=output_dir,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  FAILED {mode} {epoch} blue: {exc}")
        else:
            arm_results[SYSTEMIC_VELOCITY_ARM] = result
            arm_arrays[SYSTEMIC_VELOCITY_ARM] = arrays
            dataset_blue_measurements = _measurement_rows(
                planet=args.planet,
                mode=mode,
                epoch=epoch,
                arm=SYSTEMIC_VELOCITY_ARM,
                arrays=arrays,
            )

        validation = _validate_blue_arm(arm_results)
        dataset_joint_measurements = _blue_only_exposures(
            [row for row in dataset_blue_measurements if row["selected_for_rv"]],
        )
        for row in dataset_joint_measurements:
            row["dataset_blue_arm_validation_passed"] = validation[
                "blue_arm_validation_passed"
            ]
        blue_measurements.extend(dataset_blue_measurements)
        if validation["blue_arm_validation_passed"]:
            validated_dataset_keys.add((mode, epoch))
            validated_blue_measurements.extend(dataset_joint_measurements)
        dataset = {
            "schema_version": STELLAR_VELOCITY_RESULT_SCHEMA_VERSION,
            "method": LSD_METHOD,
            "planet": args.planet,
            "mode": mode,
            "epoch": epoch,
            "ephemeris": args.ephemeris,
            "wavelength_medium": WAVELENGTH_MEDIUM,
            "wavelength_frame": "barycentric",
            "input_frame": "PEPSI barycentric vacuum",
            "vsini_kms": vsini,
            "limb_darkening_law": "quadratic",
            "limb_darkening_u1": limb_darkening_u1,
            "limb_darkening_u2": limb_darkening_u2,
            "systemic_velocity_arm": SYSTEMIC_VELOCITY_ARM,
            "systemic_velocity_arm_policy": SYSTEMIC_VELOCITY_ARM_POLICY,
            "edge_trim_policy": LSD_EDGE_TRIM_POLICY,
            "edge_trim_calibration": (
                arm_results.get(SYSTEMIC_VELOCITY_ARM, {}).get(
                    "edge_trim_calibration",
                    edge_trim_calibration,
                )
            ),
            "resolving_power": float(args.resolving_power),
            "velocity_span_kms": [float(velocity_grid[0]), float(velocity_grid[-1])],
            "velocity_step_kms": float(velocity_grid[1] - velocity_grid[0]),
            "template_path": template["path"],
            "template_sha256": template["sha256"],
            "template_metadata_path": template["metadata_path"],
            "template_metadata_sha256": template["metadata_sha256"],
            "template_parameters": template["metadata"],
            "n_template_pixels": int(template["wavelength"].size),
            "template_matrix_rcond": TEMPLATE_MATRIX_RCOND,
            "profile_qc": {
                "maximum_fractional_model_rms": args.max_profile_fractional_rms,
                "maximum_profile_asymmetry": args.max_profile_asymmetry,
            },
            "exposure_selection": (
                "out_of_transit_only"
                if mode == "transmission" and not args.include_in_transit
                else "all_exposures"
            ),
            "science_exposure_selection": arm_results.get(
                SYSTEMIC_VELOCITY_ARM,
                {},
            ).get("science_exposure_selection", {}),
            "arms": arm_results,
            "n_blue_exposures_for_joint_fit": len(dataset_joint_measurements),
            "correction_convention": (
                "positive velocity means recession; stellar_rest_wavelength = "
                "barycentric_wavelength / sqrt((1+beta)/(1-beta))"
            ),
            **validation,
        }
        datasets.append(dataset)
        if arm_arrays:
            _plot_dataset(dataset, arm_arrays, output_dir / f"{mode}_{epoch}.pdf")
        print(f"  Dataset status: {dataset['status']}")

    if len(validated_dataset_keys) < 2:
        raise ValueError(
            "At least two datasets must have valid blue-arm measurements; "
            "red-arm and invalid-dataset fallbacks are disabled."
        )
    if len(validated_blue_measurements) < 3:
        raise ValueError(
            "At least three QC-passing blue-arm exposure RVs are required; "
            "red-arm and invalid-dataset fallbacks are disabled."
        )
    joint_measurements = validated_blue_measurements
    joint_result = fit_circular_stellar_velocity(
        np.asarray([row["orbital_phase"] for row in joint_measurements], dtype=float),
        np.asarray(
            [row["stellar_rv_barycentric_kms"] for row in joint_measurements],
            dtype=float,
        ),
        np.asarray(
            [row["stellar_rv_stat_err_kms"] for row in joint_measurements],
            dtype=float,
        ),
    )
    systematic_error_floor_kms = float(args.systematic_error_floor_kms)
    joint_result["systematic_error_floor_kms"] = systematic_error_floor_kms
    joint_result["systemic_velocity_err_kms"] = float(
        np.hypot(
            joint_result["mean_stellar_velocity_stat_err_kms"],
            systematic_error_floor_kms,
        )
    )
    print(
        "\nJoint circular stellar RV fit: "
        f"gamma={joint_result['mean_stellar_velocity_kms']:+.4f} +/- "
        f"{joint_result['systemic_velocity_err_kms']:.4f} km/s "
        f"(stat={joint_result['mean_stellar_velocity_stat_err_kms']:.4f}, "
        f"floor={systematic_error_floor_kms:.4f}), "
        f"K_star={joint_result['stellar_rv_semiamplitude_kms']:+.4f} +/- "
        f"{joint_result['stellar_rv_semiamplitude_err_kms']:.4f} km/s, "
        f"jitter={joint_result['rv_jitter_kms']:.4f} km/s."
    )
    for dataset in datasets:
        dataset_accepted = bool(dataset["blue_arm_validation_passed"])
        dataset.update(
            {
                "accepted_for_stellar_rest": dataset_accepted,
                "orbit_model": "gamma + K_star * sin(2*pi*phase); circular orbit",
                "systemic_velocity_kms": joint_result[
                    "mean_stellar_velocity_kms"
                ],
                "systemic_velocity_stat_err_kms": joint_result[
                    "mean_stellar_velocity_stat_err_kms"
                ],
                "systemic_velocity_err_kms": joint_result[
                    "systemic_velocity_err_kms"
                ],
                "systematic_error_floor_kms": systematic_error_floor_kms,
                "mean_stellar_velocity_kms": joint_result[
                    "mean_stellar_velocity_kms"
                ],
                "mean_stellar_velocity_stat_err_kms": joint_result[
                    "mean_stellar_velocity_stat_err_kms"
                ],
                "stellar_rv_semiamplitude_kms": joint_result[
                    "stellar_rv_semiamplitude_kms"
                ],
                "stellar_rv_semiamplitude_err_kms": joint_result[
                    "stellar_rv_semiamplitude_err_kms"
                ],
                "rv_jitter_kms": joint_result["rv_jitter_kms"],
                "n_joint_rv_measurements": joint_result["n_measurements"],
                "joint_fit_uses_only_validated_blue_datasets": True,
            }
        )
        _write_dataset_result(dataset)

    _write_summary(
        datasets,
        joint_result,
        blue_measurements,
        joint_measurements,
        output_dir,
    )
    return 0 if all(item["accepted_for_stellar_rest"] for item in datasets) else 2


if __name__ == "__main__":
    raise SystemExit(main())
