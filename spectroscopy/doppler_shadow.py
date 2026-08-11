"""Fit and install an in-transit Doppler shadow in PySME LSD profile space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "atmo_retrieval_matplotlib"),
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import config_utils
from dataio.collapse_transmission_timeseries_to_1d import (
    compute_contact_phases,
    get_bjd_tdb,
    get_ephemeris_epoch_bjd_tdb,
    get_orbital_phase,
    get_pepsi_data,
    get_sysrem_ignore_mask,
)
from dataio.lsd_doppler_shadow import (
    FIXED_LSD_SHADOW_METHOD,
    FIXED_LSD_SHADOW_SCHEMA_VERSION,
    compute_doppler_shadow_track,
    compute_local_doppler_shadow_profiles,
)
from dataio.orbital_velocity import planet_radial_velocity_kms
from dataio.stellar_lsd import (
    LSD_VELOCITY_STEP_KMS,
    WAVELENGTH_MEDIUM,
    extract_lsd_profiles,
    fit_rotational_profile,
    load_stellar_template,
    mask_regions,
    relativistic_velocity_difference_kms,
    synthesize_spectrum_from_lsd_profile,
    validate_quadratic_limb_darkening,
    velocity_to_doppler_factor,
)
from plotting.style import configure_matplotlib, save_figure_pdf

configure_matplotlib()


ARTIFACT_SCHEMA_VERSION = 2
DEFAULT_RESOLVING_POWER = 130000.0
DEFAULT_VELOCITY_SPAN_KMS = 180.0
DEFAULT_LOCAL_SIGMA_KMS = 8.0
BALMER_VACUUM_ANGSTROM = (4862.683, 6564.614)
BALMER_EXCLUSION_KMS = 600.0


@dataclass
class DopplerShadowFitConfig:
    """Explicit programmatic configuration for one Doppler-shadow fit."""

    epoch: str
    planet: str = "KELT-20b"
    ephemeris: str = "Duck24"
    shadow_source: str = "Recommended"
    arm: str = "both"
    template: Path | None = None
    prepared_root: Path | None = None
    diagnostic_root: Path | None = None
    vsini_kms: float | None = None
    lambda_angle_deg: float | None = None
    fit_lambda_angle: bool = False
    limb_darkening_u1: float | None = None
    limb_darkening_u2: float | None = None
    resolving_power: float = DEFAULT_RESOLVING_POWER
    velocity_span_kms: float = DEFAULT_VELOCITY_SPAN_KMS
    initial_local_sigma_kms: float = DEFAULT_LOCAL_SIGMA_KMS
    shadow_exclusion_kms: float = 20.0
    planet_exclusion_kms: float = 15.0

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "DopplerShadowFitConfig":
        return cls(**vars(args))


def _planet_slug(planet: str) -> str:
    return planet.strip().lower().replace("-", "").replace(" ", "")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_prepared_operator(
    data_dir: Path,
    arm: str,
    *,
    product_kind: str,
) -> dict[str, Any]:
    metadata_path = data_dir / "timeseries_prep.json"
    operator_path = data_dir / "timeseries_operator.npz"
    if not metadata_path.is_file() or not operator_path.is_file():
        raise FileNotFoundError(
            f"{data_dir} must first be regenerated with timeseries_prep.json and "
            "timeseries_operator.npz before fitting a fixed Doppler shadow."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        str(metadata.get("arm")) != arm
        or str(metadata.get("product_kind")) != product_kind
    ):
        raise ValueError(
            f"{metadata_path} is not the requested {arm} {product_kind} product."
        )
    with np.load(operator_path, allow_pickle=False) as archive:
        required = {
            "source_wavelength",
            "source_phase",
            "source_bjd_tdb",
            "active_exposure_mask",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{operator_path} is missing: {', '.join(missing)}")
        operator = {name: np.asarray(archive[name]) for name in archive.files}

    wavelength = np.asarray(operator["source_wavelength"], dtype=float)
    phase = np.asarray(operator["source_phase"], dtype=float)
    bjd_tdb = np.asarray(operator["source_bjd_tdb"], dtype=float)
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError(f"{operator_path} source wavelength must be finite and increasing.")
    if phase.ndim != 1 or bjd_tdb.shape != phase.shape:
        raise ValueError(f"{operator_path} source phase and time arrays must match.")
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "operator_path": operator_path,
        "source_wavelength": wavelength,
        "source_phase": phase,
        "source_bjd_tdb": bjd_tdb,
    }


def _load_raw_arm(
    *,
    planet: str,
    epoch: str,
    arm: str,
    params: dict[str, Any],
    do_molecfit: bool,
) -> dict[str, Any]:
    raw_dir = config_utils.get_raw_hrs_dir(
        planet=planet,
        epoch=epoch,
        mode="transmission",
    )
    loaded = get_pepsi_data(
        arm=arm,
        observation_epoch=epoch,
        planet_name=planet,
        do_molecfit=bool(do_molecfit),
        data_dir=raw_dir,
        regrid=False,
        subtract_median=False,
        run_sysrem=False,
        wavelength_frame="barycentric",
        data_mode="transmission",
    )
    if loaded is None:
        raise FileNotFoundError(f"No raw PEPSI {arm} spectra found in {raw_dir}.")
    result, extras = loaded
    wavelength, flux, error, jd, _snr, _exptime, _airmass, _n_exp, _n_pix = result
    if extras.get("wavelength_medium") != WAVELENGTH_MEDIUM:
        raise ValueError("The raw LSD fit requires vacuum wavelengths.")
    if extras.get("wavelength_frame") != "barycentric":
        raise ValueError("The raw LSD fit requires the barycentric wavelength reconstruction.")

    jd = np.asarray(jd, dtype=float)
    bjd_tdb = get_bjd_tdb(
        jd,
        str(params["RA"]),
        str(params["Dec"]),
        header_bjd_tdb=extras.get("header_bjd_tdb"),
    )
    reference_epoch = get_ephemeris_epoch_bjd_tdb(
        float(params["epoch"]),
        str(params.get("epoch_scale")),
        str(params.get("epoch_reference")),
    )
    midpoint = config_utils.resolve_transit_midpoint(
        np.asarray(bjd_tdb, dtype=float),
        params,
        reference_epoch_bjd_tdb=reference_epoch,
        observation_epoch=epoch,
    )
    phase = get_orbital_phase(
        np.asarray(bjd_tdb, dtype=float),
        midpoint,
        float(params["period"]),
    )
    return {
        "wavelength": np.asarray(wavelength, dtype=float),
        "flux": np.asarray(flux, dtype=float),
        "error": np.asarray(error, dtype=float),
        "jd": jd,
        "bjd_tdb": np.asarray(bjd_tdb, dtype=float),
        "phase": np.asarray(phase, dtype=float),
        "extras": extras,
    }


def _prepared_uses_molecfit(metadata: dict[str, Any]) -> bool:
    """Recover the frozen flux-product choice from frame provenance."""

    contract = metadata.get("wavelength_frame_contract")
    per_exposure = contract.get("per_exposure", []) if isinstance(contract, dict) else []
    products = {
        str(row.get("flux_product"))
        for row in per_exposure
        if isinstance(row, dict) and row.get("flux_product") is not None
    }
    if products == {"molecfit"}:
        return True
    if products == {"raw_pepsi"}:
        return False
    raise ValueError(
        "Prepared frame provenance must identify one uniform flux product; "
        f"found {sorted(products)!r}."
    )


def _pixel_masks(
    wavelength: np.ndarray,
    *,
    planet: str,
    epoch: str,
    arm: str,
    prepared_metadata: dict[str, Any],
) -> np.ndarray:
    edge_trim = prepared_metadata.get("arm_edge_trim")
    if not isinstance(edge_trim, dict):
        raise ValueError("Prepared metadata is missing arm_edge_trim provenance.")
    left_trim = float(edge_trim.get("left_trim_A", np.nan))
    right_trim = float(edge_trim.get("right_trim_A", np.nan))
    if not np.isfinite(left_trim) or not np.isfinite(right_trim):
        raise ValueError("Prepared arm-edge trim widths must be finite.")

    masks = np.zeros(wavelength.shape, dtype=bool)
    balmer_regions = [
        (
            line * float(velocity_to_doppler_factor(-BALMER_EXCLUSION_KMS)),
            line * float(velocity_to_doppler_factor(BALMER_EXCLUSION_KMS)),
        )
        for line in BALMER_VACUUM_ANGSTROM
    ]
    for index, row in enumerate(wavelength):
        valid = np.isfinite(row) & (row > 0.0)
        masks[index] |= ~valid
        if np.any(valid):
            masks[index, valid] |= get_sysrem_ignore_mask(
                row[valid],
                arm,
                explicit_edge_trim_widths_A=(left_trim, right_trim),
            )
        masks[index] |= mask_regions(row, balmer_regions)
    return masks


def _match_source_exposures(
    raw_bjd_tdb: np.ndarray,
    source_bjd_tdb: np.ndarray,
    *,
    tolerance_day: float = 1.0e-8,
) -> np.ndarray:
    raw = np.asarray(raw_bjd_tdb, dtype=float)
    source = np.asarray(source_bjd_tdb, dtype=float)
    indices = np.empty(source.size, dtype=int)
    for index, time in enumerate(source):
        match = int(np.argmin(np.abs(raw - time)))
        if abs(float(raw[match] - time)) > tolerance_day:
            raise ValueError(
                "Raw LSD exposures do not align with the frozen source sequence; "
                f"nearest BJD_TDB differs by {abs(float(raw[match] - time)):.3g} day."
            )
        indices[index] = match
    if np.unique(indices).size != indices.size:
        raise ValueError("Frozen source times matched duplicate raw exposures.")
    return indices


def _weighted_master(
    profiles: np.ndarray,
    uncertainties: np.ndarray,
    out_of_transit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_rows = (
        np.asarray(out_of_transit, dtype=bool)
        & np.all(np.isfinite(profiles), axis=1)
        & np.all(np.isfinite(uncertainties) & (uncertainties > 0.0), axis=1)
    )
    if np.count_nonzero(valid_rows) < 3:
        raise ValueError("At least three successful out-of-transit LSD profiles are required.")
    weights = 1.0 / uncertainties[valid_rows] ** 2
    weight_sum = np.sum(weights, axis=0)
    master = np.sum(weights * profiles[valid_rows], axis=0) / weight_sum
    master_uncertainty = np.sqrt(1.0 / weight_sum)
    return master, master_uncertainty, valid_rows


def _master_residuals(
    velocity_kms: np.ndarray,
    profiles: np.ndarray,
    uncertainties: np.ndarray,
    master: np.ndarray,
    master_uncertainty: np.ndarray,
    *,
    master_centroid_kms: float,
    track_velocity_kms: np.ndarray | None,
    in_transit: np.ndarray,
    planet_velocity_kms: np.ndarray,
    vsini_kms: float,
    shadow_exclusion_kms: float,
    planet_exclusion_kms: float,
) -> dict[str, np.ndarray]:
    n_exposures = profiles.shape[0]
    reference_models = np.full_like(profiles, np.nan)
    residuals = np.full_like(profiles, np.nan)
    residual_errors = np.full_like(profiles, np.nan)
    scales = np.full(n_exposures, np.nan)
    offsets = np.full(n_exposures, np.nan)
    relative_velocity = velocity_kms - float(master_centroid_kms)
    # Keep enough continuum to estimate each exposure's scale even for a
    # slowly rotating, instrumentally unresolved star.  A v sin(i)+25 window
    # is too narrow once the shadow and planet neighborhoods are excluded.
    reference_half_width_kms = max(float(vsini_kms) + 25.0, 60.0)

    for exposure in range(n_exposures):
        valid = (
            np.isfinite(profiles[exposure])
            & np.isfinite(uncertainties[exposure])
            & (uncertainties[exposure] > 0.0)
            & (np.abs(relative_velocity) <= reference_half_width_kms)
        )
        if in_transit[exposure] and track_velocity_kms is not None:
            valid &= np.abs(
                relative_velocity - track_velocity_kms[exposure]
            ) > float(shadow_exclusion_kms)
        if in_transit[exposure]:
            valid &= np.abs(
                relative_velocity - planet_velocity_kms[exposure]
            ) > float(planet_exclusion_kms)
        if np.count_nonzero(valid) < 10:
            continue
        design = np.column_stack([master[valid], np.ones(np.count_nonzero(valid))])
        weighted_design = design / uncertainties[exposure, valid, np.newaxis]
        weighted_profile = profiles[exposure, valid] / uncertainties[exposure, valid]
        scale, offset = np.linalg.lstsq(weighted_design, weighted_profile, rcond=None)[0]
        reference = float(scale) * master + float(offset)
        reference_models[exposure] = reference
        residuals[exposure] = profiles[exposure] - reference
        residual_errors[exposure] = np.sqrt(
            uncertainties[exposure] ** 2
            + (float(scale) * master_uncertainty) ** 2
        )
        scales[exposure] = float(scale)
        offsets[exposure] = float(offset)

    return {
        "reference_models": reference_models,
        "residuals": residuals,
        "residual_uncertainties": residual_errors,
        "master_scale": scales,
        "master_offset": offsets,
    }


def _fit_shadow(
    relative_velocity_kms: np.ndarray,
    residuals: np.ndarray,
    uncertainties: np.ndarray,
    *,
    track_velocity_kms: np.ndarray,
    occulted_weight: np.ndarray,
    in_transit: np.ndarray,
    planet_velocity_kms: np.ndarray,
    vsini_kms: float,
    initial_sigma_kms: float,
    planet_exclusion_kms: float | None,
) -> dict[str, Any]:
    velocity = np.asarray(relative_velocity_kms, dtype=float)
    fit_mask = (
        np.asarray(in_transit, dtype=bool)[:, np.newaxis]
        & np.isfinite(residuals)
        & np.isfinite(uncertainties)
        & (uncertainties > 0.0)
        & (np.abs(velocity)[np.newaxis, :] <= float(vsini_kms) + 25.0)
    )
    if planet_exclusion_kms is not None:
        fit_mask &= (
            np.abs(
                velocity[np.newaxis, :] - planet_velocity_kms[:, np.newaxis]
            )
            > float(planet_exclusion_kms)
        )
    if np.count_nonzero(fit_mask) < 30:
        raise ValueError("Too few finite in-transit LSD bins remain for shadow fitting.")

    initial_template = compute_local_doppler_shadow_profiles(
        velocity,
        track_velocity_kms,
        occulted_weight,
        local_sigma_kms=float(initial_sigma_kms),
    )
    weighted_template = initial_template[fit_mask] / uncertainties[fit_mask]
    weighted_residuals = residuals[fit_mask] / uncertainties[fit_mask]
    denominator = float(np.sum(weighted_template**2))
    amplitude0 = (
        float(np.sum(weighted_residuals * weighted_template) / denominator)
        if denominator > 0.0
        else -0.01
    )

    def objective(parameters: np.ndarray) -> np.ndarray:
        amplitude, velocity_offset, log_sigma = parameters
        template = compute_local_doppler_shadow_profiles(
            velocity,
            track_velocity_kms,
            occulted_weight,
            local_sigma_kms=float(np.exp(log_sigma)),
            velocity_offset_kms=float(velocity_offset),
        )
        return (
            residuals[fit_mask] - float(amplitude) * template[fit_mask]
        ) / uncertainties[fit_mask]

    optimized = least_squares(
        objective,
        x0=np.asarray([amplitude0, 0.0, np.log(initial_sigma_kms)], dtype=float),
        bounds=(
            np.asarray([-np.inf, -20.0, np.log(1.0)]),
            np.asarray([np.inf, 20.0, np.log(30.0)]),
        ),
        method="trf",
        max_nfev=300,
    )
    if not optimized.success:
        raise RuntimeError(f"Doppler-shadow fit failed: {optimized.message}")
    amplitude, velocity_offset, log_sigma = optimized.x
    local_sigma = float(np.exp(log_sigma))
    template = compute_local_doppler_shadow_profiles(
        velocity,
        track_velocity_kms,
        occulted_weight,
        local_sigma_kms=local_sigma,
        velocity_offset_kms=float(velocity_offset),
    )
    model = float(amplitude) * template
    normalized_residual = ((residuals - model) / uncertainties)[fit_mask]
    chi2 = float(np.sum(normalized_residual**2))
    dof = max(1, int(normalized_residual.size - 3))
    covariance = np.linalg.pinv(optimized.jac.T @ optimized.jac) * (chi2 / dof)
    parameter_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "amplitude": float(amplitude),
        "amplitude_err": float(parameter_error[0]),
        "velocity_offset_kms": float(velocity_offset),
        "velocity_offset_err_kms": float(parameter_error[1]),
        "local_sigma_kms": local_sigma,
        "local_sigma_err_kms": float(local_sigma * parameter_error[2]),
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "n_fit_bins": int(np.count_nonzero(fit_mask)),
        "model": model,
        "fit_mask": fit_mask,
    }


def _fit_shadow_with_lambda(
    relative_velocity_kms: np.ndarray,
    residuals: np.ndarray,
    uncertainties: np.ndarray,
    *,
    phase: np.ndarray,
    planet_velocity_kms: np.ndarray,
    vsini_kms: float,
    b: float,
    rp_rs: float,
    a_rs: float,
    period: float,
    gamma1: float,
    gamma2: float,
    eccentricity: float,
    omega_deg: float | None,
    initial_lambda_angle_deg: float,
    initial_sigma_kms: float,
    planet_exclusion_kms: float | None,
) -> dict[str, Any]:
    """Fit lambda jointly when no trusted fixed obliquity is available."""

    velocity = np.asarray(relative_velocity_kms, dtype=float)
    phase = np.asarray(phase, dtype=float)
    initial_track = compute_doppler_shadow_track(
        phase,
        vsini=float(vsini_kms),
        lambda_angle=float(initial_lambda_angle_deg),
        b=float(b),
        rp_rs=float(rp_rs),
        a_rs=float(a_rs),
        period=float(period),
        gamma1=float(gamma1),
        gamma2=float(gamma2),
        eccentricity=float(eccentricity),
        omega_deg=omega_deg,
    )
    fit_mask = (
        np.asarray(initial_track["in_transit"], dtype=bool)[:, np.newaxis]
        & np.isfinite(residuals)
        & np.isfinite(uncertainties)
        & (uncertainties > 0.0)
        & (np.abs(velocity)[np.newaxis, :] <= float(vsini_kms) + 25.0)
    )
    if planet_exclusion_kms is not None:
        fit_mask &= (
            np.abs(
                velocity[np.newaxis, :] - planet_velocity_kms[:, np.newaxis]
            )
            > float(planet_exclusion_kms)
        )
    if np.count_nonzero(fit_mask) < 30:
        raise ValueError("Too few finite in-transit LSD bins remain for lambda fitting.")

    def template_for(parameters: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        _amplitude, velocity_offset, log_sigma, lambda_angle = parameters
        track = compute_doppler_shadow_track(
            phase,
            vsini=float(vsini_kms),
            lambda_angle=float(lambda_angle),
            b=float(b),
            rp_rs=float(rp_rs),
            a_rs=float(a_rs),
            period=float(period),
            gamma1=float(gamma1),
            gamma2=float(gamma2),
            eccentricity=float(eccentricity),
            omega_deg=omega_deg,
        )
        template = compute_local_doppler_shadow_profiles(
            velocity,
            track["velocity_kms"],
            track["occulted_weight"],
            local_sigma_kms=float(np.exp(log_sigma)),
            velocity_offset_kms=float(velocity_offset),
        )
        return template, track

    def objective(parameters: np.ndarray) -> np.ndarray:
        amplitude = float(parameters[0])
        template, _track = template_for(parameters)
        return (residuals[fit_mask] - amplitude * template[fit_mask]) / uncertainties[
            fit_mask
        ]

    start_angles = list(
        dict.fromkeys(
            float(value)
            for value in (
                initial_lambda_angle_deg,
                -150.0,
                -90.0,
                -30.0,
                30.0,
                90.0,
                150.0,
            )
        )
    )
    solutions = []
    for lambda_start in start_angles:
        initial_parameters = np.asarray(
            [-0.01, 0.0, np.log(initial_sigma_kms), lambda_start],
            dtype=float,
        )
        initial_template, _track = template_for(initial_parameters)
        inverse_variance = np.zeros_like(uncertainties)
        inverse_variance[fit_mask] = 1.0 / uncertainties[fit_mask] ** 2
        denominator = float(np.sum(inverse_variance * initial_template**2))
        if denominator > 0.0:
            initial_parameters[0] = float(
                np.sum(inverse_variance * residuals * initial_template) / denominator
            )
        solution = least_squares(
            objective,
            x0=initial_parameters,
            bounds=(
                np.asarray([-np.inf, -20.0, np.log(1.0), -180.0]),
                np.asarray([np.inf, 20.0, np.log(30.0), 180.0]),
            ),
            method="trf",
            max_nfev=600,
        )
        if solution.success:
            solutions.append(solution)
    if not solutions:
        raise RuntimeError("All multistart joint-lambda Doppler-shadow fits failed.")
    optimized = min(solutions, key=lambda item: float(np.sum(item.fun**2)))
    amplitude, velocity_offset, log_sigma, lambda_angle = optimized.x
    template, track = template_for(optimized.x)
    model = float(amplitude) * template
    normalized_residual = ((residuals - model) / uncertainties)[fit_mask]
    chi2 = float(np.sum(normalized_residual**2))
    dof = max(1, int(normalized_residual.size - 4))
    covariance = np.linalg.pinv(optimized.jac.T @ optimized.jac) * (chi2 / dof)
    parameter_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "amplitude": float(amplitude),
        "amplitude_err": float(parameter_error[0]),
        "velocity_offset_kms": float(velocity_offset),
        "velocity_offset_err_kms": float(parameter_error[1]),
        "local_sigma_kms": float(np.exp(log_sigma)),
        "local_sigma_err_kms": float(np.exp(log_sigma) * parameter_error[2]),
        "lambda_angle_deg": float(lambda_angle),
        "lambda_angle_err_deg": float(parameter_error[3]),
        "lambda_angle_mode": "fitted_from_lsd_shadow",
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "n_fit_bins": int(np.count_nonzero(fit_mask)),
        "model": model,
        "fit_mask": fit_mask,
        "track": track,
    }


def _projection_velocity_grid(
    velocity_grid: np.ndarray,
    prepared_metadata: dict[str, Any],
) -> tuple[np.ndarray, float | None]:
    frame = str(prepared_metadata.get("wavelength_frame", "")).strip().lower()
    if frame == "barycentric":
        return np.asarray(velocity_grid, dtype=float), None
    if frame != "stellar_rest":
        raise ValueError(f"Unsupported prepared wavelength frame {frame!r}.")
    contract = prepared_metadata.get("wavelength_frame_contract")
    correction = contract.get("stellar_rest_correction", {}) if isinstance(contract, dict) else {}
    removed_velocity = float(correction.get("applied_velocity_kms", np.nan))
    if not np.isfinite(removed_velocity):
        raise ValueError("Stellar-rest metadata is missing its applied velocity.")
    return (
        relativistic_velocity_difference_kms(velocity_grid, removed_velocity),
        removed_velocity,
    )


def _project_fixed_source_model(
    *,
    prepared: dict[str, Any],
    raw: dict[str, Any],
    velocity_grid: np.ndarray,
    fitted_shadow_profiles: np.ndarray,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Project one fitted LSD shadow onto an exact prepared source grid."""

    source_order = _match_source_exposures(
        raw["bjd_tdb"],
        prepared["source_bjd_tdb"],
    )
    if not np.allclose(
        raw["phase"][source_order],
        prepared["source_phase"],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "Raw and prepared source phases disagree after BJD matching for "
            f"{prepared['metadata_path']}."
        )
    projection_velocity, removed_velocity = _projection_velocity_grid(
        velocity_grid,
        prepared["metadata"],
    )
    source_model = synthesize_spectrum_from_lsd_profile(
        prepared["source_wavelength"],
        template["wavelength"],
        template["flux"],
        projection_velocity,
        np.asarray(fitted_shadow_profiles, dtype=float)[source_order],
    )
    expected_shape = (
        prepared["source_phase"].size,
        prepared["source_wavelength"].size,
    )
    if source_model.shape != expected_shape or np.any(~np.isfinite(source_model)):
        raise ValueError(
            "Projected LSD shadow has an invalid shape or non-finite values: "
            f"{source_model.shape}, expected {expected_shape}."
        )
    source_model_path = prepared["metadata_path"].parent / "shadow_source_model.npy"
    np.save(source_model_path, np.asarray(source_model, dtype=np.float64))
    return {
        "source_exposure_indices": source_order,
        "source_model_path": source_model_path,
        "source_model_sha256": _sha256_file(source_model_path),
        "source_model_shape": list(source_model.shape),
        "stellar_rest_velocity_removed_kms": removed_velocity,
    }


def _enable_prepared_lsd_shadow(
    *,
    prepared: dict[str, Any],
    artifact_path: Path,
    artifact_json_path: Path,
    artifact_sha256: str,
    template_sha256: str,
    projection: dict[str, Any],
) -> None:
    """Persist the mandatory shared-basis LSD contract for one source product."""

    data_dir = prepared["metadata_path"].parent
    metadata = dict(prepared["metadata"])
    metadata["fixed_doppler_shadow"] = {
        "schema_version": FIXED_LSD_SHADOW_SCHEMA_VERSION,
        "enabled": True,
        "required": True,
        "method": FIXED_LSD_SHADOW_METHOD,
        "source_model_file": projection["source_model_path"].name,
        "source_model_sha256": projection["source_model_sha256"],
        "artifact_file": os.path.relpath(artifact_path, start=data_dir),
        "artifact_sha256": artifact_sha256,
        "artifact_metadata_file": os.path.relpath(
            artifact_json_path,
            start=data_dir,
        ),
        "template_sha256": template_sha256,
    }
    model_preprocessing = str(metadata["model_preprocessing"])
    if not model_preprocessing.startswith("fixed_shared_basis_lsd_shadow_then_"):
        model_preprocessing = (
            "fixed_shared_basis_lsd_shadow_then_" + model_preprocessing
        )
    metadata["model_preprocessing"] = model_preprocessing
    prepared["metadata_path"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _color_limit(array: np.ndarray) -> float:
    finite = np.abs(np.asarray(array)[np.isfinite(array)])
    return float(np.percentile(finite, 99.0)) if finite.size else 1.0


def _save_diagnostic_pdf(
    path: Path,
    *,
    phase: np.ndarray,
    relative_velocity: np.ndarray,
    profiles: np.ndarray,
    master: np.ndarray,
    residuals: np.ndarray,
    shadow_model: np.ndarray,
    track_velocity: np.ndarray,
    velocity_offset: float,
    planet_velocity: np.ndarray,
    contacts: dict[str, float],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.2),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    maps = (
        profiles - master[np.newaxis, :],
        residuals,
        residuals - shadow_model,
    )
    titles = (
        "LSD profiles minus OOT master",
        "Master-scaled LSD residuals",
        "Residuals after fitted shadow",
    )
    common_limit = max(_color_limit(values) for values in maps)
    in_transit = (phase >= float(contacts["T1"])) & (
        phase <= float(contacts["T4"])
    )
    track_plot = np.where(
        in_transit,
        track_velocity + velocity_offset,
        np.nan,
    )
    planet_plot = np.where(in_transit, planet_velocity, np.nan)
    for axis, values, title in zip(axes, maps, titles):
        mesh = axis.pcolormesh(
            relative_velocity,
            phase,
            values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-common_limit,
            vmax=common_limit,
            rasterized=True,
        )
        axis.plot(track_plot, phase, color="black", lw=1.2, label="RM track")
        axis.plot(planet_plot, phase, color="gold", lw=1.0, ls="--", label="planet mask")
        for contact in ("T1", "T2", "T3", "T4"):
            axis.axhline(float(contacts[contact]), color="0.35", lw=0.6, ls=":")
        axis.set_title(title)
        axis.set_xlabel("Velocity relative to stellar LSD centroid [km/s]")
        figure.colorbar(mesh, ax=axis, pad=0.01, fraction=0.045)
    axes[0].set_ylabel("Orbital phase")
    axes[1].legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pdf(figure, path)
    plt.close(figure)


def _fit_arm(
    *,
    args: argparse.Namespace,
    arm: str,
    params: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    if args.prepared_root is None:
        timeseries_dir = config_utils.get_timeseries_data_dir(
            planet=args.planet,
            mode="transmission",
            epoch=args.epoch,
            arm=arm,
        )
        collapse_source_dir = config_utils.get_collapse_source_dir(
            planet=args.planet,
            mode="transmission",
            epoch=args.epoch,
            arm=arm,
        )
    else:
        prepared_root = args.prepared_root.resolve() / arm
        timeseries_dir = prepared_root / "timeseries"
        collapse_source_dir = prepared_root / "collapse_source"
    prepared_timeseries = _load_prepared_operator(
        timeseries_dir,
        arm,
        product_kind="timeseries",
    )
    prepared_collapse_source = _load_prepared_operator(
        collapse_source_dir,
        arm,
        product_kind="collapse-source",
    )

    raw = _load_raw_arm(
        planet=args.planet,
        epoch=args.epoch,
        arm=arm,
        params=params,
        do_molecfit=_prepared_uses_molecfit(prepared_timeseries["metadata"]),
    )
    velocity_grid = np.arange(
        -float(args.velocity_span_kms),
        float(args.velocity_span_kms) + 0.5 * LSD_VELOCITY_STEP_KMS,
        LSD_VELOCITY_STEP_KMS,
        dtype=float,
    )
    pixel_masks = _pixel_masks(
        raw["wavelength"],
        planet=args.planet,
        epoch=args.epoch,
        arm=arm,
        prepared_metadata=prepared_timeseries["metadata"],
    )
    print(f"  Extracting {arm} LSD profiles for {raw['flux'].shape[0]} exposures...")
    extracted = extract_lsd_profiles(
        raw["wavelength"],
        raw["flux"],
        raw["error"],
        template_wavelength=template["wavelength"],
        template_flux=template["flux"],
        velocity_grid=velocity_grid,
        pixel_masks=pixel_masks,
        shared_basis=True,
    )
    profiles = np.asarray(extracted["profiles"], dtype=float)
    profile_uncertainties = np.asarray(extracted["profile_uncertainties"], dtype=float)
    if not bool(extracted["shared_basis_used"]):
        raise RuntimeError("Fresh Doppler-shadow extraction did not use a shared basis.")
    shared_rank = int(extracted["shared_effective_rank"])
    exposure_ranks = np.asarray(extracted["effective_rank"], dtype=int)
    if not np.all(exposure_ranks == shared_rank):
        raise RuntimeError("Not every exposure used the declared shared LSD rank.")
    projected_condition = np.asarray(
        extracted["projected_condition_number"], dtype=float
    )
    if np.any(~np.isfinite(projected_condition)):
        raise RuntimeError("Shared-basis projected condition numbers must be finite.")
    print(
        f"  {arm}: shared LSD rank={shared_rank} across {profiles.shape[0]} "
        f"exposures; projected condition range="
        f"{np.min(projected_condition):.3f}-{np.max(projected_condition):.3f}"
    )
    profile_source = {
        "kind": "fresh_raw_pysme_lsd_extraction",
        "solver": "shared_nightly_basis_v1",
    }
    source_order = _match_source_exposures(
        raw["bjd_tdb"],
        prepared_timeseries["source_bjd_tdb"],
    )
    if not np.allclose(
        raw["phase"][source_order],
        prepared_timeseries["source_phase"],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("Raw and frozen-operator phases disagree after BJD matching.")
    contacts = compute_contact_phases(params)
    out_of_transit = (raw["phase"] < contacts["T1"]) | (raw["phase"] > contacts["T4"])
    master, master_uncertainty, master_rows = _weighted_master(
        profiles,
        profile_uncertainties,
        out_of_transit,
    )
    master_fit = fit_rotational_profile(
        velocity_grid,
        master,
        master_uncertainty,
        vsini_kms=float(args.vsini_kms),
        limb_darkening_u1=float(args.limb_darkening_u1),
        limb_darkening_u2=float(args.limb_darkening_u2),
        resolving_power=float(args.resolving_power),
    )
    master_centroid = float(master_fit["centroid_kms"])
    track = compute_doppler_shadow_track(
        raw["phase"],
        vsini=float(args.vsini_kms),
        lambda_angle=float(args.lambda_angle_deg),
        b=float(params["b"]),
        rp_rs=float(params["rp_rs"]),
        a_rs=float(params["a_rs"]),
        period=float(params["period"]),
        gamma1=float(args.limb_darkening_u1),
        gamma2=float(args.limb_darkening_u2),
        eccentricity=float(params.get("eccentricity", 0.0)),
        omega_deg=params.get("omega"),
    )
    planet_velocity = planet_radial_velocity_kms(
        raw["phase"],
        kp_kms=float(params["Kp"]),
        eccentricity=float(params.get("eccentricity", 0.0)),
        omega_deg=params.get("omega"),
    )
    residual_bundle = _master_residuals(
        velocity_grid,
        profiles,
        profile_uncertainties,
        master,
        master_uncertainty,
        master_centroid_kms=master_centroid,
        track_velocity_kms=(
            None if args.fit_lambda_angle else track["velocity_kms"]
        ),
        in_transit=track["in_transit"],
        planet_velocity_kms=planet_velocity,
        vsini_kms=float(args.vsini_kms),
        shadow_exclusion_kms=float(args.shadow_exclusion_kms),
        planet_exclusion_kms=float(args.planet_exclusion_kms),
    )
    relative_velocity = velocity_grid - master_centroid
    if args.fit_lambda_angle:
        lambda_fit_kwargs = {
            "phase": raw["phase"],
            "planet_velocity_kms": planet_velocity,
            "vsini_kms": float(args.vsini_kms),
            "b": float(params["b"]),
            "rp_rs": float(params["rp_rs"]),
            "a_rs": float(params["a_rs"]),
            "period": float(params["period"]),
            "gamma1": float(args.limb_darkening_u1),
            "gamma2": float(args.limb_darkening_u2),
            "eccentricity": float(params.get("eccentricity", 0.0)),
            "omega_deg": params.get("omega"),
            "initial_lambda_angle_deg": float(args.lambda_angle_deg),
            "initial_sigma_kms": float(args.initial_local_sigma_kms),
        }
        masked_fit = _fit_shadow_with_lambda(
            relative_velocity,
            residual_bundle["residuals"],
            residual_bundle["residual_uncertainties"],
            planet_exclusion_kms=float(args.planet_exclusion_kms),
            **lambda_fit_kwargs,
        )
        unmasked_fit = _fit_shadow_with_lambda(
            relative_velocity,
            residual_bundle["residuals"],
            residual_bundle["residual_uncertainties"],
            planet_exclusion_kms=None,
            **lambda_fit_kwargs,
        )
        track = masked_fit["track"]
    else:
        masked_fit = _fit_shadow(
            relative_velocity,
            residual_bundle["residuals"],
            residual_bundle["residual_uncertainties"],
            track_velocity_kms=track["velocity_kms"],
            occulted_weight=track["occulted_weight"],
            in_transit=track["in_transit"],
            planet_velocity_kms=planet_velocity,
            vsini_kms=float(args.vsini_kms),
            initial_sigma_kms=float(args.initial_local_sigma_kms),
            planet_exclusion_kms=float(args.planet_exclusion_kms),
        )
        unmasked_fit = _fit_shadow(
            relative_velocity,
            residual_bundle["residuals"],
            residual_bundle["residual_uncertainties"],
            track_velocity_kms=track["velocity_kms"],
            occulted_weight=track["occulted_weight"],
            in_transit=track["in_transit"],
            planet_velocity_kms=planet_velocity,
            vsini_kms=float(args.vsini_kms),
            initial_sigma_kms=float(args.initial_local_sigma_kms),
            planet_exclusion_kms=None,
        )
    if masked_fit["amplitude"] >= 0.0:
        raise ValueError(
            f"The fitted {arm} LSD amplitude is {masked_fit['amplitude']:.6g}, not the "
            "negative occulted-light sign required for a physical Doppler shadow."
        )

    diagnostic_root = (
        REPOSITORY_ROOT / "diagnostics" / "doppler_shadow"
        if args.diagnostic_root is None
        else args.diagnostic_root.resolve()
    )
    diagnostic_dir = diagnostic_root / _planet_slug(args.planet) / args.epoch
    pdf_path = diagnostic_dir / f"{arm}.pdf"
    _save_diagnostic_pdf(
        pdf_path,
        phase=raw["phase"],
        relative_velocity=relative_velocity,
        profiles=profiles,
        master=master,
        residuals=residual_bundle["residuals"],
        shadow_model=masked_fit["model"],
        track_velocity=track["velocity_kms"],
        velocity_offset=masked_fit["velocity_offset_kms"],
        planet_velocity=planet_velocity,
        contacts=contacts,
    )

    artifact_path = timeseries_dir / "doppler_shadow_lsd.npz"
    artifact_json_path = artifact_path.with_suffix(".json")
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
        velocity_kms=velocity_grid,
        velocity_relative_to_master_kms=relative_velocity,
        phase=raw["phase"],
        bjd_tdb=raw["bjd_tdb"],
        profiles=profiles,
        profile_uncertainties=profile_uncertainties,
        extraction_reduced_chi2=np.asarray(extracted["reduced_chi2"], dtype=float),
        extraction_effective_rank=np.asarray(extracted["effective_rank"], dtype=np.int32),
        extraction_n_pixels=np.asarray(extracted["n_pixels"], dtype=np.int64),
        extraction_failures=np.asarray(
            ["" if value is None else value for value in extracted["failures"]],
            dtype="U512",
        ),
        shared_basis_used=np.asarray(extracted["shared_basis_used"], dtype=bool),
        shared_effective_rank=np.asarray(
            extracted["shared_effective_rank"], dtype=np.int32
        ),
        shared_eigenvalue_ratios=np.asarray(
            extracted["shared_eigenvalue_ratios"], dtype=float
        ),
        shared_velocity_basis=np.asarray(
            extracted["shared_velocity_basis"], dtype=float
        ),
        projected_condition_number=np.asarray(
            extracted["projected_condition_number"], dtype=float
        ),
        template_matrix_rcond=np.asarray(
            extracted["template_matrix_rcond"], dtype=float
        ),
        out_of_transit_master=master,
        out_of_transit_master_uncertainty=master_uncertainty,
        out_of_transit_master_rows=master_rows,
        master_reference_models=residual_bundle["reference_models"],
        master_scale=residual_bundle["master_scale"],
        master_offset=residual_bundle["master_offset"],
        residuals=residual_bundle["residuals"],
        residual_uncertainties=residual_bundle["residual_uncertainties"],
        physical_track_velocity_kms=track["velocity_kms"],
        physical_track_weight=track["occulted_weight"],
        planet_velocity_kms=planet_velocity,
        lambda_angle_deg=np.asarray(
            masked_fit.get("lambda_angle_deg", args.lambda_angle_deg),
            dtype=float,
        ),
        lambda_angle_fitted=np.asarray(args.fit_lambda_angle, dtype=bool),
        planet_masked_fit_mask=masked_fit["fit_mask"],
        fitted_shadow_model=masked_fit["model"],
        residuals_after_shadow=residual_bundle["residuals"] - masked_fit["model"],
        unmasked_fitted_shadow_model=unmasked_fit["model"],
        source_exposure_indices=source_order,
    )

    fit_summary = {
        "masked": {
            key: value
            for key, value in masked_fit.items()
            if key not in {"model", "fit_mask", "track"}
        },
        "unmasked_control": {
            key: value
            for key, value in unmasked_fit.items()
            if key not in {"model", "fit_mask", "track"}
        },
    }
    resolved_lambda_angle = float(
        masked_fit.get("lambda_angle_deg", args.lambda_angle_deg)
    )
    projected_condition = np.asarray(
        extracted["projected_condition_number"], dtype=float
    )
    shared_rank = int(extracted["shared_effective_rank"])
    exposure_ranks = np.asarray(extracted["effective_rank"], dtype=int)
    artifact_metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "method": FIXED_LSD_SHADOW_METHOD,
        "planet": args.planet,
        "epoch": args.epoch,
        "arm": arm,
        "wavelength_medium": WAVELENGTH_MEDIUM,
        "lsd_wavelength_frame": "barycentric",
        "prepared_wavelength_frame": prepared_timeseries["metadata"].get(
            "wavelength_frame"
        ),
        "profile_source": profile_source,
        "lsd_solver": {
            "method": "shared_nightly_basis_v1",
            "basis_source": "mean_normalized_normal_matrix",
            "shared_rank": shared_rank,
            "rcond": float(extracted["template_matrix_rcond"]),
            "all_exposures_use_shared_basis": bool(
                extracted["shared_basis_used"]
                and np.all(exposure_ranks == shared_rank)
            ),
            "condition_number_min": float(np.min(projected_condition)),
            "condition_number_max": float(np.max(projected_condition)),
        },
        "template": {
            "path": template["path"],
            "sha256": template["sha256"],
            "metadata_path": template["metadata_path"],
            "metadata_sha256": template["metadata_sha256"],
        },
        "parameter_resolution": params.get("parameter_resolution"),
        "geometry": {
            "vsini_kms": float(args.vsini_kms),
            "lambda_angle_deg": resolved_lambda_angle,
            "lambda_angle_mode": (
                "fitted_from_lsd_shadow"
                if args.fit_lambda_angle
                else "fixed_from_shadow_source"
            ),
            "lambda_angle_err_deg": masked_fit.get("lambda_angle_err_deg"),
            "impact_parameter": float(params["b"]),
            "rp_rs": float(params["rp_rs"]),
            "a_rs": float(params["a_rs"]),
            "period_day": float(params["period"]),
            "eccentricity": float(params.get("eccentricity", 0.0)),
            "omega_planet_deg": params.get("omega"),
            "kp_kms": float(params["Kp"]),
            "kp_source": params.get("Kp_source"),
            "kp_is_derived": bool(params.get("Kp_is_derived", False)),
            "limb_darkening_u1": float(args.limb_darkening_u1),
            "limb_darkening_u2": float(args.limb_darkening_u2),
            "resolving_power": float(args.resolving_power),
        },
        "fit": fit_summary,
        "master_centroid_kms": master_centroid,
        "n_exposures": int(raw["phase"].size),
        "n_successful_profiles": int(np.count_nonzero(np.all(np.isfinite(profiles), axis=1))),
        "n_out_of_transit_master_profiles": int(np.count_nonzero(master_rows)),
        "planet_mask_half_width_kms": float(args.planet_exclusion_kms),
        "artifact_file": artifact_path.name,
        "artifact_sha256": _sha256_file(artifact_path),
        "diagnostic_pdf": str(pdf_path.resolve()),
        "standard_pipeline_model": True,
    }
    projections: dict[str, dict[str, Any]] = {}
    for product_name, prepared_product in (
        ("timeseries", prepared_timeseries),
        ("collapse_source", prepared_collapse_source),
    ):
        projection = _project_fixed_source_model(
            prepared=prepared_product,
            raw=raw,
            velocity_grid=velocity_grid,
            fitted_shadow_profiles=masked_fit["model"],
            template=template,
        )
        projections[product_name] = projection
        artifact_metadata.setdefault("prepared_products", {})[product_name] = {
            "product_kind": prepared_product["metadata"]["product_kind"],
            "source_phase_sha256": _array_sha256(prepared_product["source_phase"]),
            "source_bjd_tdb_sha256": _array_sha256(
                prepared_product["source_bjd_tdb"]
            ),
            "source_wavelength_sha256": _array_sha256(
                prepared_product["source_wavelength"]
            ),
            "source_model_file": str(projection["source_model_path"].resolve()),
            "source_model_sha256": projection["source_model_sha256"],
            "source_model_shape": projection["source_model_shape"],
            "stellar_rest_velocity_removed_kms": projection[
                "stellar_rest_velocity_removed_kms"
            ],
        }

    artifact_json_path.write_text(
        json.dumps(artifact_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for product_name, prepared_product in (
        ("timeseries", prepared_timeseries),
        ("collapse_source", prepared_collapse_source),
    ):
        _enable_prepared_lsd_shadow(
            prepared=prepared_product,
            artifact_path=artifact_path,
            artifact_json_path=artifact_json_path,
            artifact_sha256=artifact_metadata["artifact_sha256"],
            template_sha256=template["sha256"],
            projection=projections[product_name],
        )

    print(
        f"  {arm}: amplitude={masked_fit['amplitude']:.6g} +/- "
        f"{masked_fit['amplitude_err']:.3g}, offset="
        f"{masked_fit['velocity_offset_kms']:.3f} km/s, sigma="
        f"{masked_fit['local_sigma_kms']:.3f} km/s"
    )
    if args.fit_lambda_angle:
        print(
            "  Fitted projected obliquity: "
            f"lambda={masked_fit['lambda_angle_deg']:.3f} +/- "
            f"{masked_fit['lambda_angle_err_deg']:.3f} deg"
        )
    if masked_fit["reduced_chi2"] > 3.0:
        print(
            f"  Warning: reduced chi2={masked_fit['reduced_chi2']:.2f}; "
            "formal fit errors are not trustworthy until residual structure is resolved."
        )
    print(f"  Saved LSD artifact: {artifact_path}")
    print(f"  Saved diagnostic: {pdf_path}")
    print(
        "  Enabled mandatory LSD source models: "
        f"{timeseries_dir / 'shadow_source_model.npy'}; "
        f"{collapse_source_dir / 'shadow_source_model.npy'}"
    )
    return artifact_metadata


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract in-transit PySME LSD residuals, fit one physical RM track, "
            "and install the required fixed shadow on both prepared source grids."
        )
    )
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default="Duck24")
    parser.add_argument(
        "--shadow-source",
        default="Recommended",
        help=(
            "Independent config block for stellar profile, geometry, obliquity, "
            "and Kp (default: Recommended). --ephemeris remains the timing source."
        ),
    )
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--arm", choices=("blue", "red", "both"), default="both")
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help=(
            "Isolated <planet>/<epoch> root containing each arm's timeseries "
            "and collapse_source directories. "
            "Defaults to the canonical prepared tree."
        ),
    )
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=None,
        help="Optional noncanonical root for diagnostic PDFs.",
    )
    parser.add_argument("--vsini-kms", type=float, default=None)
    parser.add_argument("--lambda-angle-deg", type=float, default=None)
    parser.add_argument(
        "--fit-lambda-angle",
        action="store_true",
        help=(
            "Fit projected obliquity jointly with the LSD shadow. This is "
            "enabled automatically when the selected shadow source has no lambda."
        ),
    )
    parser.add_argument("--limb-darkening-u1", type=float, default=None)
    parser.add_argument("--limb-darkening-u2", type=float, default=None)
    parser.add_argument("--resolving-power", type=float, default=DEFAULT_RESOLVING_POWER)
    parser.add_argument("--velocity-span-kms", type=float, default=DEFAULT_VELOCITY_SPAN_KMS)
    parser.add_argument("--initial-local-sigma-kms", type=float, default=DEFAULT_LOCAL_SIGMA_KMS)
    parser.add_argument("--shadow-exclusion-kms", type=float, default=20.0)
    parser.add_argument("--planet-exclusion-kms", type=float, default=15.0)
    return parser


def fit_doppler_shadow(args: DopplerShadowFitConfig) -> int:
    """Fit and install the required LSD shadow for the requested arm(s)."""

    params = config_utils.resolve_parameter_domains(
        planet=args.planet,
        timing_source=args.ephemeris,
        shadow_source=args.shadow_source,
    )
    args.vsini_kms = float(
        params.get("v_sini_star", np.nan) if args.vsini_kms is None else args.vsini_kms
    )
    configured_lambda_value = params.get("lambda_angle", np.nan)
    try:
        configured_lambda = float(configured_lambda_value)
    except (TypeError, ValueError):
        configured_lambda = np.nan
    if args.lambda_angle_deg is not None:
        configured_lambda = float(args.lambda_angle_deg)
    if not np.isfinite(configured_lambda):
        args.fit_lambda_angle = True
        configured_lambda = 0.0
        print(
            "No published/configured projected obliquity is available; "
            "fitting lambda from the LSD Doppler shadow."
        )
    args.lambda_angle_deg = configured_lambda
    args.limb_darkening_u1 = float(
        params.get("gamma1", np.nan)
        if args.limb_darkening_u1 is None
        else args.limb_darkening_u1
    )
    args.limb_darkening_u2 = float(
        params.get("gamma2", np.nan)
        if args.limb_darkening_u2 is None
        else args.limb_darkening_u2
    )
    required_geometry = {
        "v_sini_star": args.vsini_kms,
        "gamma1": args.limb_darkening_u1,
        "gamma2": args.limb_darkening_u2,
        "b": params.get("b"),
        "rp_rs": params.get("rp_rs"),
        "a_rs": params.get("a_rs"),
        "period": params.get("period"),
        "Kp": params.get("Kp"),
    }
    if float(params.get("eccentricity", 0.0)) != 0.0:
        required_geometry["omega_planet"] = params.get("omega")
    if not args.fit_lambda_angle:
        required_geometry["lambda_angle"] = args.lambda_angle_deg
    missing = [
        name
        for name, value in required_geometry.items()
        if value is None or not np.isfinite(float(value))
    ]
    if missing:
        raise ValueError(f"Missing finite fixed shadow geometry: {', '.join(missing)}")
    validate_quadratic_limb_darkening(args.limb_darkening_u1, args.limb_darkening_u2)
    if args.vsini_kms <= 0.0 or args.velocity_span_kms <= args.vsini_kms:
        raise ValueError("Velocity span must be wider than a finite positive v sin(i).")
    for name in (
        "resolving_power",
        "initial_local_sigma_kms",
        "shadow_exclusion_kms",
        "planet_exclusion_kms",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    template_path = args.template
    if template_path is None:
        template_path = (
            REPOSITORY_ROOT
            / "reference"
            / "stellar_lsd_templates"
            / f"{_planet_slug(args.planet)}_pysme_lte_vacuum.npz"
        )
    template = load_stellar_template(template_path)
    arms = ("blue", "red") if args.arm == "both" else (args.arm,)
    print(
        f"Fitting {args.planet} {args.epoch} Doppler shadow with "
        f"{template_path.name} ({', '.join(arms)})"
    )
    for arm in arms:
        _fit_arm(args=args, arm=arm, params=params, template=template)
    return 0


def main() -> int:
    return fit_doppler_shadow(
        DopplerShadowFitConfig.from_namespace(create_parser().parse_args())
    )


if __name__ == "__main__":
    raise SystemExit(main())
