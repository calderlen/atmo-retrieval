"""PySME-template deconvolution stellar-velocity measurements.

The public functions in this module intentionally implement one estimator:
an uncertainty-weighted broadening profile deconvolved from a fixed intrinsic
PySME spectrum, followed by a quadratic-limb-darkened rotation plus
instrumental-Gaussian fit. Positive velocity means recession.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, minimize_scalar

from dataio.exposure_selection import (
    SCIENCE_EXPOSURE_SELECTION_POLICY,
    SCIENCE_EXPOSURE_SELECTION_SCHEMA_VERSION,
)


C_KMS = 299792.458
LSD_METHOD = "pysme_spectrum_lsd_blue_arm_quadratic_rotational_gaussian_profile"
WAVELENGTH_MEDIUM = "vacuum"
STELLAR_VELOCITY_RESULT_SCHEMA_VERSION = 7
STELLAR_TEMPLATE_SCHEMA_VERSION = 1
TEMPLATE_KIND = "pysme_intrinsic_spectrum"
TEMPLATE_MATRIX_RCOND = 1.0e-2
TEMPLATE_MATRIX_CHUNK_SIZE = 2048
LSD_VELOCITY_STEP_KMS = 3.0
SYSTEMIC_VELOCITY_ARM = "blue"
SYSTEMIC_VELOCITY_ARM_POLICY = "blue_only_no_fallback"
LSD_EDGE_TRIM_POLICY = "accepted_calibration_manifest_no_fallback"


def validate_production_lsd_velocity_grid(
    velocity_grid: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    """Enforce the production LSD velocity-grid contract."""
    velocity_grid = np.asarray(velocity_grid, dtype=float)
    if velocity_grid.ndim != 1 or velocity_grid.size < 2:
        raise ValueError(f"{source} must contain at least two velocity bins.")
    spacing = np.diff(velocity_grid)
    if (
        not np.all(np.isfinite(velocity_grid))
        or not np.all(spacing > 0.0)
        or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{source} must be finite, increasing, and uniformly spaced.")
    step = float(spacing[0])
    if not np.isclose(step, LSD_VELOCITY_STEP_KMS, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{source} uses {step:g} km/s bins; the production LSD contract "
            f"requires {LSD_VELOCITY_STEP_KMS:g} km/s bins. Regenerate the "
            "LSD profiles instead of reusing this grid."
        )
    return velocity_grid


def _validate_template_metadata(metadata: dict[str, Any], *, source: Path) -> None:
    """Validate the scientific frame and broadening contract of a PySME template."""
    required_metadata = {
        "schema_version",
        "template_kind",
        "wavelength_medium",
        "wavelength_frame",
        "continuum_normalized",
        "radial_velocity_kms",
        "vsini_kms",
        "vmacro_kms",
        "pysme_version",
        "linelist_sha256",
    }
    missing = sorted(required_metadata - set(metadata))
    if missing:
        raise ValueError(f"{source} is missing required field(s): {', '.join(missing)}")
    if metadata["schema_version"] != STELLAR_TEMPLATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported stellar-template schema {metadata['schema_version']!r}; "
            f"expected {STELLAR_TEMPLATE_SCHEMA_VERSION}."
        )
    if metadata["template_kind"] != TEMPLATE_KIND:
        raise ValueError(f"Unsupported stellar-template kind: {metadata['template_kind']!r}")
    if str(metadata["wavelength_medium"]).strip().lower() != WAVELENGTH_MEDIUM:
        raise ValueError("The PySME stellar template must use vacuum wavelengths.")
    if str(metadata["wavelength_frame"]).strip().lower() != "stellar_rest":
        raise ValueError("The PySME stellar template must be in the stellar rest frame.")
    if metadata["continuum_normalized"] is not True:
        raise ValueError("The PySME stellar template must be continuum normalized.")
    for field in ("radial_velocity_kms", "vsini_kms", "vmacro_kms"):
        value = float(metadata[field])
        if not np.isfinite(value) or abs(value) > 1.0e-12:
            raise ValueError(f"The intrinsic PySME template requires {field}=0; found {value}.")


def load_stellar_template(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a frozen intrinsic PySME spectrum."""
    path = Path(path)
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise FileNotFoundError(f"PySME stellar template not found: {path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PySME stellar-template provenance is missing: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_template_metadata(metadata, source=metadata_path)

    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required_arrays = {"wavelength_vacuum_angstrom", "normalized_flux"}
        if not required_arrays.issubset(names):
            raise ValueError(
                f"{path} must contain arrays: {', '.join(sorted(required_arrays))}."
            )
        wavelength = np.asarray(archive["wavelength_vacuum_angstrom"], dtype=float)
        flux = np.asarray(archive["normalized_flux"], dtype=float)

    if wavelength.ndim != 1 or flux.shape != wavelength.shape or wavelength.size < 100:
        raise ValueError("PySME template wavelength and flux must be matching 1D arrays.")
    if (
        not np.all(np.isfinite(wavelength))
        or not np.all(wavelength > 0.0)
        or not np.all(np.diff(wavelength) > 0.0)
    ):
        raise ValueError("PySME template wavelengths must be finite, positive, and increasing.")
    if not np.all(np.isfinite(flux)):
        raise ValueError("PySME template flux must be finite.")
    depression = 1.0 - flux
    if float(np.nanmax(depression)) <= 1.0e-4:
        raise ValueError("PySME template contains no measurable absorption features.")

    return {
        "wavelength": wavelength,
        "flux": flux,
        "depression": depression,
        "metadata": metadata,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }


def velocity_to_doppler_factor(velocity_kms: float | np.ndarray) -> np.ndarray:
    """Return the relativistic wavelength factor for a radial velocity."""
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("Radial velocity magnitude must be smaller than c.")
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def shift_to_stellar_rest(
    wavelength: np.ndarray,
    residual_velocity_kms: float,
) -> np.ndarray:
    """Remove a measured stellar residual velocity from a wavelength array."""
    if not np.isfinite(residual_velocity_kms):
        raise ValueError("The stellar residual velocity must be finite.")
    return np.asarray(wavelength, dtype=float) / velocity_to_doppler_factor(
        residual_velocity_kms
    )


def relativistic_velocity_difference_kms(
    velocity_kms: float | np.ndarray,
    removed_velocity_kms: float | np.ndarray,
) -> np.ndarray:
    """Return the residual whose Doppler factor completes a removed velocity."""
    velocity = np.asarray(velocity_kms, dtype=float)
    removed = np.asarray(removed_velocity_kms, dtype=float)
    rapidity = np.arctanh(velocity / C_KMS) - np.arctanh(removed / C_KMS)
    return C_KMS * np.tanh(rapidity)


def build_lsd_normal_system(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    velocity_grid: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one uncertainty-weighted template-deconvolution system."""
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    template_wavelength = np.asarray(template_wavelength, dtype=float)
    template_flux = np.asarray(template_flux, dtype=float)
    velocity_grid = np.asarray(velocity_grid, dtype=float)

    if wavelength.ndim != 1 or flux.shape != wavelength.shape or uncertainty.shape != wavelength.shape:
        raise ValueError("wavelength, flux, and uncertainty must be matching 1D arrays.")
    if (
        template_wavelength.ndim != 1
        or template_flux.shape != template_wavelength.shape
        or template_wavelength.size < 100
    ):
        raise ValueError("template_wavelength and template_flux must be matching 1D arrays.")
    if not np.all(np.diff(template_wavelength) > 0.0):
        raise ValueError("template_wavelength must be strictly increasing.")
    if velocity_grid.ndim != 1 or velocity_grid.size < 5:
        raise ValueError("velocity_grid must be a 1D array with at least five bins.")
    spacing = np.diff(velocity_grid)
    if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("velocity_grid must be strictly increasing and uniformly spaced.")

    valid = (
        np.isfinite(wavelength)
        & (wavelength > 0.0)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != wavelength.shape:
            raise ValueError("pixel_mask shape must match wavelength.")
        valid &= ~pixel_mask
    if np.count_nonzero(valid) < velocity_grid.size:
        raise ValueError("Too few valid spectral pixels for the requested LSD grid.")

    doppler_factors = velocity_to_doppler_factor(velocity_grid)
    template_low = float(template_wavelength[0])
    template_high = float(template_wavelength[-1])
    valid &= wavelength / float(np.max(doppler_factors)) >= template_low
    valid &= wavelength / float(np.min(doppler_factors)) <= template_high
    if np.count_nonzero(valid) < velocity_grid.size:
        raise ValueError("Too few pixels have complete PySME-template velocity coverage.")

    wave = wavelength[valid]
    observed_depression = 1.0 - flux[valid]
    sigma = uncertainty[valid]
    order = np.argsort(wave)
    wave = wave[order]
    observed_depression = observed_depression[order]
    sigma = sigma[order]
    template_depression = 1.0 - template_flux
    n_velocity = velocity_grid.size
    normal = np.zeros((n_velocity, n_velocity), dtype=float)
    right_hand_side = np.zeros(n_velocity, dtype=float)
    weighted_data_norm = 0.0
    n_constrained_pixels = 0
    for start in range(0, wave.size, TEMPLATE_MATRIX_CHUNK_SIZE):
        stop = min(wave.size, start + TEMPLATE_MATRIX_CHUNK_SIZE)
        rest_wavelength = wave[start:stop, np.newaxis] / doppler_factors[np.newaxis, :]
        design = np.interp(
            rest_wavelength.ravel(),
            template_wavelength,
            template_depression,
        ).reshape(rest_wavelength.shape)
        constrained = np.max(np.abs(design), axis=1) > 1.0e-8
        if not np.any(constrained):
            continue
        design = design[constrained]
        chunk_sigma = sigma[start:stop][constrained]
        chunk_data = observed_depression[start:stop][constrained]
        weighted_design = design / chunk_sigma[:, np.newaxis]
        weighted_data = chunk_data / chunk_sigma
        normal += weighted_design.T @ weighted_design
        right_hand_side += weighted_design.T @ weighted_data
        weighted_data_norm += float(weighted_data @ weighted_data)
        n_constrained_pixels += int(np.count_nonzero(constrained))

    if n_constrained_pixels <= n_velocity:
        raise ValueError("Template LSD system is not over-constrained after spectral masking.")

    return {
        "normal_matrix": normal,
        "right_hand_side": right_hand_side,
        "weighted_data_norm": weighted_data_norm,
        "n_pixels": n_constrained_pixels,
    }


def build_shared_lsd_basis(
    normal_matrices: np.ndarray,
    *,
    rcond: float,
) -> dict[str, Any]:
    """Choose one retained velocity-profile basis for an exposure sequence."""
    matrices = np.asarray(normal_matrices, dtype=float)
    rcond = float(rcond)
    if (
        matrices.ndim != 3
        or matrices.shape[0] < 1
        or matrices.shape[1] != matrices.shape[2]
    ):
        raise ValueError("normal_matrices must be exposure x velocity x velocity.")
    if np.any(~np.isfinite(matrices)):
        raise ValueError("Shared LSD normal matrices must be finite.")
    if not np.isfinite(rcond) or not 0.0 < rcond < 1.0:
        raise ValueError("Shared LSD rcond must be finite and between zero and one.")

    normalized_matrices = np.empty_like(matrices)
    for index, matrix in enumerate(matrices):
        symmetric = 0.5 * (matrix + matrix.T)
        maximum_eigenvalue = float(np.max(np.linalg.eigvalsh(symmetric)))
        if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0.0:
            raise ValueError(
                f"Exposure {index} has a singular LSD normal matrix."
            )
        normalized_matrices[index] = symmetric / maximum_eigenvalue

    shared_matrix = np.mean(normalized_matrices, axis=0)
    shared_matrix = 0.5 * (shared_matrix + shared_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(shared_matrix)
    maximum_eigenvalue = float(np.max(eigenvalues))
    if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0.0:
        raise ValueError("The shared LSD normal matrix is singular.")
    eigenvalue_ratios = eigenvalues / maximum_eigenvalue
    retained = eigenvalue_ratios > rcond
    effective_rank = int(np.count_nonzero(retained))
    if effective_rank < 5:
        raise ValueError("Shared LSD design matrix has insufficient numerical rank.")

    return {
        "basis": eigenvectors[:, retained],
        "eigenvalues": eigenvalues,
        "eigenvalue_ratios": eigenvalue_ratios,
        "effective_rank": effective_rank,
        "rcond": rcond,
    }


def solve_lsd_in_basis(
    system: dict[str, Any],
    shared_basis: np.ndarray,
) -> dict[str, Any]:
    """Solve one LSD system in a fixed exposure-sequence basis."""
    normal = np.asarray(system["normal_matrix"], dtype=float)
    right_hand_side = np.asarray(system["right_hand_side"], dtype=float)
    basis = np.asarray(shared_basis, dtype=float)
    if normal.ndim != 2 or normal.shape[0] != normal.shape[1]:
        raise ValueError("LSD normal_matrix must be square.")
    if right_hand_side.shape != (normal.shape[0],):
        raise ValueError("LSD right_hand_side must match the normal matrix.")
    if basis.ndim != 2 or basis.shape[0] != normal.shape[0] or basis.shape[1] < 1:
        raise ValueError("shared_basis must be velocity x retained mode.")
    if (
        np.any(~np.isfinite(normal))
        or np.any(~np.isfinite(right_hand_side))
        or np.any(~np.isfinite(basis))
    ):
        raise ValueError("Shared-basis LSD inputs must be finite.")

    projected_normal = basis.T @ normal @ basis
    projected_normal = 0.5 * (projected_normal + projected_normal.T)
    projected_eigenvalues = np.linalg.eigvalsh(projected_normal)
    minimum_eigenvalue = float(np.min(projected_eigenvalues))
    maximum_eigenvalue = float(np.max(projected_eigenvalues))
    if (
        not np.isfinite(minimum_eigenvalue)
        or not np.isfinite(maximum_eigenvalue)
        or minimum_eigenvalue <= 0.0
    ):
        raise ValueError("Projected shared-basis LSD system is singular.")

    projected_right_hand_side = basis.T @ right_hand_side
    coefficients = np.linalg.solve(projected_normal, projected_right_hand_side)
    projected_covariance = np.linalg.inv(projected_normal)
    profile = basis @ coefficients
    covariance = basis @ projected_covariance @ basis.T
    profile_uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not np.all(np.isfinite(profile_uncertainty)) or np.any(profile_uncertainty <= 0.0):
        raise ValueError("Could not derive finite shared-basis LSD uncertainties.")

    weighted_data_norm = float(system["weighted_data_norm"])
    chi2 = float(
        max(
            0.0,
            weighted_data_norm
            - 2.0 * profile @ right_hand_side
            + profile @ normal @ profile,
        )
    )
    effective_rank = int(basis.shape[1])
    n_pixels = int(system["n_pixels"])
    dof = max(1, n_pixels - effective_rank)
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    return {
        "profile": profile,
        "uncertainty": profile_uncertainty,
        "effective_rank": effective_rank,
        "n_pixels": n_pixels,
        "chi2": chi2,
        "dof": dof,
        "projected_condition_number": condition_number,
    }


def extract_lsd_profile(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    velocity_grid: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Deconvolve one uncertainty-weighted broadening profile from a template."""
    velocity_grid = np.asarray(velocity_grid, dtype=float)
    system = build_lsd_normal_system(
        wavelength,
        flux,
        uncertainty,
        template_wavelength,
        template_flux,
        velocity_grid,
        pixel_mask=pixel_mask,
    )
    normal = np.asarray(system["normal_matrix"], dtype=float)
    right_hand_side = np.asarray(system["right_hand_side"], dtype=float)
    weighted_data_norm = float(system["weighted_data_norm"])
    n_constrained_pixels = int(system["n_pixels"])
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    maximum_eigenvalue = float(np.max(eigenvalues))
    if not np.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0.0:
        raise ValueError("Template LSD normal matrix is singular.")
    retained = eigenvalues > TEMPLATE_MATRIX_RCOND * maximum_eigenvalue
    effective_rank = int(np.count_nonzero(retained))
    if effective_rank < 5:
        raise ValueError("Template LSD design matrix has insufficient numerical rank.")
    retained_vectors = eigenvectors[:, retained]
    inverse_eigenvalues = 1.0 / eigenvalues[retained]
    covariance = (retained_vectors * inverse_eigenvalues) @ retained_vectors.T
    profile = covariance @ right_hand_side
    chi2 = float(
        max(
            0.0,
            weighted_data_norm
            - 2.0 * profile @ right_hand_side
            + profile @ normal @ profile,
        )
    )
    dof = max(1, int(n_constrained_pixels - effective_rank))
    profile_uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not np.all(np.isfinite(profile_uncertainty)) or np.any(profile_uncertainty <= 0.0):
        raise ValueError("Could not derive finite LSD profile uncertainties.")

    return {
        "velocity_kms": velocity_grid.copy(),
        "profile": profile,
        "uncertainty": profile_uncertainty,
        "effective_rank": effective_rank,
        "n_pixels": n_constrained_pixels,
        "chi2": chi2,
        "dof": dof,
        "normal_matrix_condition_number": float(
            maximum_eigenvalue / float(np.min(eigenvalues[retained]))
        ),
        "normal_matrix_rcond": TEMPLATE_MATRIX_RCOND,
    }


def extract_lsd_profiles(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    *,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    velocity_grid: np.ndarray,
    pixel_masks: np.ndarray | None = None,
    shared_basis: bool = False,
) -> dict[str, Any]:
    """Extract an LSD profile for each exposure without fitting its centroid."""
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if wavelength.ndim == 1:
        wavelength = np.broadcast_to(wavelength, flux.shape)
    if wavelength.shape != flux.shape or uncertainty.shape != flux.shape or flux.ndim != 2:
        raise ValueError("wavelength, flux, and uncertainty must share an exposure x pixel shape.")
    if pixel_masks is not None:
        pixel_masks = np.asarray(pixel_masks, dtype=bool)
        if pixel_masks.ndim == 1:
            pixel_masks = np.broadcast_to(pixel_masks, flux.shape)
        if pixel_masks.shape != flux.shape:
            raise ValueError("pixel_masks must match the spectral-array shape.")

    n_exposures = flux.shape[0]
    n_velocity = np.asarray(velocity_grid).size
    profiles = np.full((n_exposures, n_velocity), np.nan, dtype=float)
    profile_errors = np.full_like(profiles, np.nan)
    effective_rank = np.zeros(n_exposures, dtype=int)
    n_pixels = np.zeros(n_exposures, dtype=int)
    reduced_chi2 = np.full(n_exposures, np.nan, dtype=float)
    failures: list[str | None] = [None] * n_exposures
    projected_condition_number = np.full(n_exposures, np.nan, dtype=float)

    if shared_basis:
        systems: list[dict[str, Any]] = []
        for index in range(n_exposures):
            try:
                system = build_lsd_normal_system(
                    wavelength[index],
                    flux[index],
                    uncertainty[index],
                    template_wavelength,
                    template_flux,
                    velocity_grid,
                    pixel_mask=None if pixel_masks is None else pixel_masks[index],
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                raise ValueError(
                    f"Exposure {index} cannot contribute to the shared LSD basis: {exc}"
                ) from exc
            systems.append(system)

        shared = build_shared_lsd_basis(
            np.stack([system["normal_matrix"] for system in systems]),
            rcond=TEMPLATE_MATRIX_RCOND,
        )
        for index, system in enumerate(systems):
            try:
                extracted = solve_lsd_in_basis(system, shared["basis"])
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                raise ValueError(
                    f"Exposure {index} failed in the shared LSD basis: {exc}"
                ) from exc
            profiles[index] = extracted["profile"]
            profile_errors[index] = extracted["uncertainty"]
            effective_rank[index] = extracted["effective_rank"]
            n_pixels[index] = extracted["n_pixels"]
            reduced_chi2[index] = extracted["chi2"] / extracted["dof"]
            projected_condition_number[index] = extracted[
                "projected_condition_number"
            ]

        return {
            "velocity_kms": np.asarray(velocity_grid, dtype=float),
            "profiles": profiles,
            "profile_uncertainties": profile_errors,
            "effective_rank": effective_rank,
            "n_pixels": n_pixels,
            "reduced_chi2": reduced_chi2,
            "failures": failures,
            "shared_basis_used": True,
            "shared_effective_rank": int(shared["effective_rank"]),
            "shared_eigenvalue_ratios": np.asarray(
                shared["eigenvalue_ratios"], dtype=float
            ),
            "shared_velocity_basis": np.asarray(shared["basis"], dtype=float),
            "projected_condition_number": projected_condition_number,
            "template_matrix_rcond": float(shared["rcond"]),
        }

    for index in range(n_exposures):
        try:
            extracted = extract_lsd_profile(
                wavelength[index],
                flux[index],
                uncertainty[index],
                template_wavelength,
                template_flux,
                velocity_grid,
                pixel_mask=None if pixel_masks is None else pixel_masks[index],
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures[index] = str(exc)
            continue
        profiles[index] = extracted["profile"]
        profile_errors[index] = extracted["uncertainty"]
        effective_rank[index] = extracted["effective_rank"]
        n_pixels[index] = extracted["n_pixels"]
        reduced_chi2[index] = extracted["chi2"] / extracted["dof"]

    return {
        "velocity_kms": np.asarray(velocity_grid, dtype=float),
        "profiles": profiles,
        "profile_uncertainties": profile_errors,
        "effective_rank": effective_rank,
        "n_pixels": n_pixels,
        "reduced_chi2": reduced_chi2,
        "failures": failures,
        "shared_basis_used": False,
        "shared_effective_rank": 0,
        "shared_eigenvalue_ratios": np.empty(0, dtype=float),
        "shared_velocity_basis": np.empty((n_velocity, 0), dtype=float),
        "projected_condition_number": projected_condition_number,
        "template_matrix_rcond": float(TEMPLATE_MATRIX_RCOND),
    }


def synthesize_spectrum_from_lsd_profile(
    wavelength: np.ndarray,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    velocity_grid: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    """Project one or more LSD residual profiles into pixel-space flux.

    ``extract_lsd_profile`` solves ``depression = design @ profile``.  This
    inverse operation returns the corresponding normalized-flux perturbation,
    ``-design @ profile``.  A negative occulted-light profile therefore becomes
    the positive line bump expected for a Doppler shadow.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    template_wavelength = np.asarray(template_wavelength, dtype=float)
    template_flux = np.asarray(template_flux, dtype=float)
    velocity_grid = np.asarray(velocity_grid, dtype=float)
    profiles = np.asarray(profile, dtype=float)
    profile_was_1d = profiles.ndim == 1
    if profile_was_1d:
        profiles = profiles[np.newaxis, :]

    if wavelength.ndim not in {1, 2}:
        raise ValueError("wavelength must be a 1D grid or exposure x pixel array.")
    if (
        template_wavelength.ndim != 1
        or template_flux.shape != template_wavelength.shape
        or not np.all(np.diff(template_wavelength) > 0.0)
    ):
        raise ValueError("template wavelength and flux must be matching, increasing 1D arrays.")
    if velocity_grid.ndim != 1 or profiles.ndim != 2 or profiles.shape[1] != velocity_grid.size:
        raise ValueError("profile must end with the velocity_grid dimension.")
    if not np.all(np.diff(velocity_grid) > 0.0):
        raise ValueError("velocity_grid must be strictly increasing.")
    if np.any(~np.isfinite(profiles)):
        raise ValueError("LSD profiles must be finite before pixel-space synthesis.")
    if wavelength.ndim == 2 and wavelength.shape[0] != profiles.shape[0]:
        raise ValueError("Exposure-dependent wavelengths must match the profile exposure axis.")
    if profile_was_1d and wavelength.ndim == 2:
        raise ValueError("A 2D wavelength array requires one LSD profile per exposure.")

    doppler_factors = velocity_to_doppler_factor(velocity_grid)
    template_depression = 1.0 - template_flux
    n_pixels = wavelength.shape[-1]
    synthesized = np.zeros((profiles.shape[0], n_pixels), dtype=float)
    active_profiles = np.any(profiles != 0.0, axis=1)

    if wavelength.ndim == 1:
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("wavelength must be finite and positive.")
        for start in range(0, n_pixels, TEMPLATE_MATRIX_CHUNK_SIZE):
            stop = min(n_pixels, start + TEMPLATE_MATRIX_CHUNK_SIZE)
            rest_wavelength = wavelength[start:stop, np.newaxis] / doppler_factors[np.newaxis, :]
            design = np.interp(
                rest_wavelength.ravel(),
                template_wavelength,
                template_depression,
                left=0.0,
                right=0.0,
            ).reshape(rest_wavelength.shape)
            if np.any(active_profiles):
                synthesized[active_profiles, start:stop] = -(
                    profiles[active_profiles] @ design.T
                )
    else:
        for exposure in range(profiles.shape[0]):
            if not active_profiles[exposure]:
                continue
            wave = wavelength[exposure]
            if np.any(~np.isfinite(wave)) or np.any(wave <= 0.0):
                raise ValueError("wavelength must be finite and positive.")
            for start in range(0, n_pixels, TEMPLATE_MATRIX_CHUNK_SIZE):
                stop = min(n_pixels, start + TEMPLATE_MATRIX_CHUNK_SIZE)
                rest_wavelength = wave[start:stop, np.newaxis] / doppler_factors[np.newaxis, :]
                design = np.interp(
                    rest_wavelength.ravel(),
                    template_wavelength,
                    template_depression,
                    left=0.0,
                    right=0.0,
                ).reshape(rest_wavelength.shape)
                synthesized[exposure, start:stop] = -(design @ profiles[exposure])

    return synthesized[0] if profile_was_1d else synthesized


def validate_quadratic_limb_darkening(
    limb_darkening_u1: float,
    limb_darkening_u2: float,
) -> tuple[float, float]:
    """Validate a non-negative quadratic specific-intensity law on the disk."""
    u1 = float(limb_darkening_u1)
    u2 = float(limb_darkening_u2)
    if not np.isfinite(u1) or not np.isfinite(u2):
        raise ValueError("Quadratic limb-darkening coefficients u1 and u2 must be finite.")

    candidate_t = [0.0, 1.0]
    if u2 != 0.0:
        stationary_t = -u1 / (2.0 * u2)
        if 0.0 < stationary_t < 1.0:
            candidate_t.append(stationary_t)
    intensity = np.asarray(
        [1.0 - u1 * t - u2 * t**2 for t in candidate_t],
        dtype=float,
    )
    if float(np.min(intensity)) < 0.0:
        raise ValueError(
            "Quadratic limb darkening must keep "
            "I(mu) = 1 - u1*(1-mu) - u2*(1-mu)^2 non-negative "
            "for 0 <= mu <= 1."
        )
    return u1, u2


def rotational_gaussian_profile(
    velocity_kms: np.ndarray,
    centroid_kms: float,
    amplitude: float,
    offset: float,
    *,
    vsini_kms: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    resolving_power: float,
) -> np.ndarray:
    """Quadratic-limb-darkened rotation convolved with the instrumental Gaussian."""
    if not np.isfinite(vsini_kms) or vsini_kms <= 0.0:
        raise ValueError("vsini_kms must be finite and positive.")
    u1, u2 = validate_quadratic_limb_darkening(
        limb_darkening_u1,
        limb_darkening_u2,
    )
    if not np.isfinite(resolving_power) or resolving_power <= 0.0:
        raise ValueError("resolving_power must be finite and positive.")
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    spacing = np.diff(velocity_kms)
    if spacing.size == 0 or not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("velocity_kms must be strictly increasing and uniformly spaced.")
    instrumental_fwhm_kms = C_KMS / float(resolving_power)
    instrumental_sigma_kms = instrumental_fwhm_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    center_value = (
        2.0 * (1.0 - u1 - u2)
        + 0.5 * np.pi * (u1 + 2.0 * u2)
        - (4.0 / 3.0) * u2
    )
    if not np.isfinite(center_value) or center_value <= 0.0:
        raise ValueError("Quadratic limb darkening produced a non-positive disk-center chord.")

    def rotation_chord(x: np.ndarray) -> np.ndarray:
        inside = np.abs(x) < 1.0
        chord = np.zeros_like(x)
        one_minus_x2 = np.maximum(0.0, 1.0 - x[inside] ** 2)
        root_one_minus_x2 = np.sqrt(one_minus_x2)
        numerator = (
            2.0 * (1.0 - u1 - u2) * root_one_minus_x2
            + 0.5 * np.pi * (u1 + 2.0 * u2) * one_minus_x2
            - (4.0 / 3.0) * u2 * root_one_minus_x2**3
        )
        chord[inside] = np.maximum(numerator, 0.0) / center_value
        return chord

    native_step = float(spacing[0])
    if float(vsini_kms) >= native_step:
        shape = rotation_chord(
            (velocity_kms - float(centroid_kms)) / float(vsini_kms)
        )
        shape = gaussian_filter1d(
            shape,
            sigma=instrumental_sigma_kms / native_step,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )
    else:
        # Construct an internal kernel when rotation is narrower than one LSD
        # bin. Direct native-grid sampling can otherwise be identically zero.
        internal_step = min(native_step / 8.0, float(vsini_kms) / 16.0)
        extent = float(vsini_kms) + 6.0 * instrumental_sigma_kms + native_step
        kernel_velocity = np.arange(
            -extent,
            extent + 0.5 * internal_step,
            internal_step,
            dtype=float,
        )
        kernel = rotation_chord(kernel_velocity / float(vsini_kms))
        kernel = gaussian_filter1d(
            kernel,
            sigma=instrumental_sigma_kms / internal_step,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )
        kernel_peak = float(np.max(kernel))
        if kernel_peak <= 0.0:
            raise ValueError("Could not construct the rotational-Gaussian profile.")
        kernel /= kernel_peak
        shape = np.interp(
            velocity_kms - float(centroid_kms),
            kernel_velocity,
            kernel,
            left=0.0,
            right=0.0,
        )

    peak = float(np.max(shape))
    if peak <= 0.0:
        raise ValueError("Could not construct the rotational-Gaussian profile.")
    shape /= peak
    return float(offset) + float(amplitude) * shape


def fit_rotational_profile(
    velocity_kms: np.ndarray,
    profile: np.ndarray,
    uncertainty: np.ndarray,
    *,
    vsini_kms: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    resolving_power: float,
) -> dict[str, Any]:
    """Fit centroid, amplitude, and offset for one broadening profile."""
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    profile = np.asarray(profile, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    valid = (
        np.isfinite(velocity_kms)
        & np.isfinite(profile)
        & np.isfinite(uncertainty)
        & (uncertainty > 0.0)
    )
    velocity = velocity_kms[valid]
    observed = profile[valid]
    sigma = uncertainty[valid]
    if velocity.size < 10:
        raise ValueError("Too few finite LSD bins for rotational-profile fitting.")

    edge = np.abs(velocity) >= 0.85 * np.max(np.abs(velocity))
    offset0 = float(np.median(observed[edge])) if np.any(edge) else float(np.median(observed))
    positive = np.maximum(observed - offset0, 0.0)
    amplitude0 = float(np.max(positive))
    if not np.isfinite(amplitude0) or amplitude0 <= 0.0:
        raise ValueError("The LSD profile has no positive absorption signal to fit.")
    centroid0 = (
        float(np.sum(velocity * positive) / np.sum(positive))
        if np.sum(positive) > 0.0
        else 0.0
    )

    center_min = float(np.min(velocity) + vsini_kms)
    center_max = float(np.max(velocity) - vsini_kms)
    if center_min >= center_max:
        raise ValueError("The LSD velocity span must be wider than twice v sin(i).")
    centroid0 = float(np.clip(centroid0, center_min, center_max))

    def model(v: np.ndarray, centroid: float, amplitude: float, offset: float) -> np.ndarray:
        return rotational_gaussian_profile(
            v,
            centroid,
            amplitude,
            offset,
            vsini_kms=vsini_kms,
            limb_darkening_u1=limb_darkening_u1,
            limb_darkening_u2=limb_darkening_u2,
            resolving_power=resolving_power,
        )

    parameters, covariance = curve_fit(
        model,
        velocity,
        observed,
        p0=(centroid0, amplitude0, offset0),
        sigma=sigma,
        absolute_sigma=True,
        bounds=(
            (center_min, 0.0, -np.inf),
            (center_max, np.inf, np.inf),
        ),
        maxfev=20000,
    )
    centroid, amplitude, offset = (float(value) for value in parameters)
    boundary_tolerance = float(np.median(np.diff(velocity)))
    if (
        centroid <= center_min + boundary_tolerance
        or centroid >= center_max - boundary_tolerance
    ):
        raise ValueError("The rotational-profile centroid converged at its fit boundary.")
    parameter_uncertainty = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    fitted = model(velocity_kms, centroid, amplitude, offset)
    fitted_valid = model(velocity, centroid, amplitude, offset)
    unscaled_residual = observed - fitted_valid
    residual = unscaled_residual / sigma
    fractional_model_rms = float(
        np.sqrt(np.mean(unscaled_residual**2)) / max(abs(amplitude), np.finfo(float).eps)
    )

    velocity_step = float(np.median(np.diff(velocity)))
    symmetric_offsets = np.arange(
        velocity_step,
        max(velocity_step, 0.95 * float(vsini_kms)),
        velocity_step,
    )
    if symmetric_offsets.size:
        left = np.interp(centroid - symmetric_offsets, velocity, observed) - offset
        right = np.interp(centroid + symmetric_offsets, velocity, observed) - offset
        denominator = float(np.sum(np.abs(left) + np.abs(right)))
        profile_asymmetry = (
            float(2.0 * np.sum(np.abs(left - right)) / denominator)
            if denominator > 0.0
            else np.inf
        )
    else:
        profile_asymmetry = np.inf
    return {
        "centroid_kms": centroid,
        "centroid_err_kms": float(parameter_uncertainty[0]),
        "amplitude": amplitude,
        "amplitude_err": float(parameter_uncertainty[1]),
        "offset": offset,
        "offset_err": float(parameter_uncertainty[2]),
        "model": fitted,
        "chi2": float(np.sum(residual**2)),
        "dof": max(1, int(velocity.size - 3)),
        "reduced_chi2": float(np.sum(residual**2) / max(1, velocity.size - 3)),
        "fractional_model_rms": fractional_model_rms,
        "profile_asymmetry": profile_asymmetry,
    }


def measure_lsd_exposures(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    *,
    template_wavelength: np.ndarray,
    template_flux: np.ndarray,
    velocity_grid: np.ndarray,
    vsini_kms: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    resolving_power: float,
    pixel_masks: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract and fit an LSD profile for every exposure."""
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if wavelength.ndim == 1:
        wavelength = np.broadcast_to(wavelength, flux.shape)
    if wavelength.shape != flux.shape or uncertainty.shape != flux.shape or flux.ndim != 2:
        raise ValueError("wavelength, flux, and uncertainty must share an exposure x pixel shape.")
    if pixel_masks is not None:
        pixel_masks = np.asarray(pixel_masks, dtype=bool)
        if pixel_masks.ndim == 1:
            pixel_masks = np.broadcast_to(pixel_masks, flux.shape)
        if pixel_masks.shape != flux.shape:
            raise ValueError("pixel_masks must match the spectral-array shape.")

    n_exposures = flux.shape[0]
    n_velocity = np.asarray(velocity_grid).size
    profiles = np.full((n_exposures, n_velocity), np.nan)
    profile_errors = np.full_like(profiles, np.nan)
    profile_models = np.full_like(profiles, np.nan)
    centroids = np.full(n_exposures, np.nan)
    centroid_errors = np.full(n_exposures, np.nan)
    amplitudes = np.full(n_exposures, np.nan)
    offsets = np.full(n_exposures, np.nan)
    reduced_chi2 = np.full(n_exposures, np.nan)
    fractional_model_rms = np.full(n_exposures, np.nan)
    profile_asymmetry = np.full(n_exposures, np.nan)
    effective_rank = np.zeros(n_exposures, dtype=int)
    n_pixels = np.zeros(n_exposures, dtype=int)
    normal_matrix_condition_number = np.full(n_exposures, np.nan)
    normal_matrix_rcond = np.full(n_exposures, np.nan)
    failures: list[str | None] = [None] * n_exposures

    for index in range(n_exposures):
        try:
            extracted = extract_lsd_profile(
                wavelength[index],
                flux[index],
                uncertainty[index],
                template_wavelength,
                template_flux,
                velocity_grid,
                pixel_mask=None if pixel_masks is None else pixel_masks[index],
            )
            fitted = fit_rotational_profile(
                extracted["velocity_kms"],
                extracted["profile"],
                extracted["uncertainty"],
                vsini_kms=vsini_kms,
                limb_darkening_u1=limb_darkening_u1,
                limb_darkening_u2=limb_darkening_u2,
                resolving_power=resolving_power,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures[index] = str(exc)
            continue
        profiles[index] = extracted["profile"]
        profile_errors[index] = extracted["uncertainty"]
        profile_models[index] = fitted["model"]
        centroids[index] = fitted["centroid_kms"]
        centroid_errors[index] = fitted["centroid_err_kms"]
        amplitudes[index] = fitted["amplitude"]
        offsets[index] = fitted["offset"]
        reduced_chi2[index] = fitted["reduced_chi2"]
        fractional_model_rms[index] = fitted["fractional_model_rms"]
        profile_asymmetry[index] = fitted["profile_asymmetry"]
        effective_rank[index] = extracted["effective_rank"]
        n_pixels[index] = extracted["n_pixels"]
        normal_matrix_condition_number[index] = extracted[
            "normal_matrix_condition_number"
        ]
        normal_matrix_rcond[index] = extracted["normal_matrix_rcond"]

    return {
        "velocity_kms": np.asarray(velocity_grid, dtype=float),
        "profiles": profiles,
        "profile_uncertainties": profile_errors,
        "profile_models": profile_models,
        "centroid_kms": centroids,
        "centroid_err_kms": centroid_errors,
        "amplitude": amplitudes,
        "offset": offsets,
        "reduced_chi2": reduced_chi2,
        "fractional_model_rms": fractional_model_rms,
        "profile_asymmetry": profile_asymmetry,
        "effective_rank": effective_rank,
        "n_pixels": n_pixels,
        "normal_matrix_condition_number": normal_matrix_condition_number,
        "normal_matrix_rcond": normal_matrix_rcond,
        "failures": failures,
    }


def fit_circular_stellar_velocity(
    phase: np.ndarray,
    velocity_kms: np.ndarray,
    uncertainty_kms: np.ndarray,
) -> dict[str, Any]:
    """Fit ``gamma + K_star sin(2 pi phase)`` with one shared RV jitter.

    For Gaussian errors and flat priors, the weighted linear solution is the
    same point estimator used by a two-parameter MCMC.  Profiling one positive
    jitter term keeps underestimated LSD formal errors from producing a
    misleadingly precise mean velocity.
    """
    phase = np.asarray(phase, dtype=float)
    velocity_kms = np.asarray(velocity_kms, dtype=float)
    uncertainty_kms = np.asarray(uncertainty_kms, dtype=float)
    valid = (
        np.isfinite(phase)
        & np.isfinite(velocity_kms)
        & np.isfinite(uncertainty_kms)
        & (uncertainty_kms > 0.0)
    )
    phase = phase[valid]
    velocity = velocity_kms[valid]
    uncertainty = uncertainty_kms[valid]
    if velocity.size < 3:
        raise ValueError("At least three finite exposure RVs are required.")

    design = np.column_stack(
        [np.ones(velocity.size, dtype=float), np.sin(2.0 * np.pi * phase)]
    )
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("The exposure phases do not constrain a circular RV model.")

    def solve(jitter_kms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        variance = uncertainty**2 + float(jitter_kms) ** 2
        inverse_variance = 1.0 / variance
        normal = design.T @ (inverse_variance[:, np.newaxis] * design)
        covariance = np.linalg.inv(normal)
        parameters = covariance @ (design.T @ (inverse_variance * velocity))
        residual = velocity - design @ parameters
        negative_log_likelihood = 0.5 * float(
            np.sum(residual**2 / variance + np.log(variance))
        )
        return parameters, covariance, residual, negative_log_likelihood

    scatter = float(np.std(velocity))
    jitter_upper = max(1.0, 5.0 * scatter, 5.0 * float(np.median(uncertainty)))
    optimized = minimize_scalar(
        lambda jitter: solve(float(jitter))[3],
        bounds=(0.0, jitter_upper),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    candidates = [(0.0, solve(0.0))]
    if optimized.success:
        jitter = float(optimized.x)
        candidates.append((jitter, solve(jitter)))
    jitter, (parameters, covariance, residual, _nll) = min(
        candidates,
        key=lambda item: item[1][3],
    )
    gamma, semi_amplitude = (float(value) for value in parameters)
    parameter_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    effective_variance = uncertainty**2 + jitter**2
    chi2 = float(np.sum(residual**2 / effective_variance))
    dof = max(1, int(velocity.size - 2))
    return {
        "mean_stellar_velocity_kms": gamma,
        "mean_stellar_velocity_stat_err_kms": float(parameter_error[0]),
        "stellar_rv_semiamplitude_kms": semi_amplitude,
        "stellar_rv_semiamplitude_err_kms": float(parameter_error[1]),
        "rv_jitter_kms": float(jitter),
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof),
        "n_measurements": int(velocity.size),
        "model_velocity_kms": design @ parameters,
        "residual_kms": residual,
        "valid_input_mask": valid,
    }


def summarize_velocity(
    centroid_kms: np.ndarray,
    centroid_err_kms: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    """Summarize selected exposure centroids with a weighted mean and RMS."""
    centroid_kms = np.asarray(centroid_kms, dtype=float)
    centroid_err_kms = np.asarray(centroid_err_kms, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    valid = (
        selected
        & np.isfinite(centroid_kms)
        & np.isfinite(centroid_err_kms)
        & (centroid_err_kms > 0.0)
    )
    n_used = int(np.count_nonzero(valid))
    if n_used == 0:
        raise ValueError("No successful selected-exposure RV measurements.")
    values = centroid_kms[valid]
    errors = centroid_err_kms[valid]
    weights = 1.0 / errors**2
    mean = float(np.sum(weights * values) / np.sum(weights))
    formal_error = float(np.sqrt(1.0 / np.sum(weights)))
    rms = float(np.sqrt(np.mean((values - mean) ** 2))) if n_used > 1 else 0.0
    scatter_error = rms / np.sqrt(n_used) if n_used > 1 else 0.0
    adopted_error = max(formal_error, scatter_error)
    return {
        "residual_velocity_kms": mean,
        "residual_velocity_err_kms": adopted_error,
        "formal_weighted_mean_err_kms": formal_error,
        "exposure_rms_kms": rms,
        "n_exposures_used": n_used,
        "used_exposure_indices": np.flatnonzero(valid).astype(int).tolist(),
    }


def load_stellar_velocity_result(
    path: str | Path,
    *,
    planet: str | None = None,
    mode: str | None = None,
    epoch: str | None = None,
) -> dict[str, Any]:
    """Load and validate an accepted dataset velocity result."""
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {"planet": planet, "mode": mode, "epoch": epoch}
    for key, value in expected.items():
        if value is not None and str(result.get(key)) != str(value):
            raise ValueError(
                f"Stellar-velocity result {path} has {key}={result.get(key)!r}; "
                f"expected {value!r}."
            )
    if result.get("method") != LSD_METHOD:
        raise ValueError(f"Unsupported stellar-velocity method in {path}.")
    if result.get("schema_version") != STELLAR_VELOCITY_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Stellar-velocity result {path} uses schema_version="
            f"{result.get('schema_version')!r}; expected "
            f"{STELLAR_VELOCITY_RESULT_SCHEMA_VERSION}. Re-run the LSD measurement."
        )
    if result.get("wavelength_medium") != WAVELENGTH_MEDIUM:
        raise ValueError(f"Stellar-velocity result {path} is not in vacuum wavelengths.")
    if result.get("wavelength_frame") != "barycentric":
        raise ValueError(
            f"Stellar-velocity result {path} is not based on barycentric wavelengths."
        )
    if result.get("limb_darkening_law") != "quadratic":
        raise ValueError(
            f"Stellar-velocity result {path} does not use quadratic limb darkening."
        )
    validate_quadratic_limb_darkening(
        result.get("limb_darkening_u1", np.nan),
        result.get("limb_darkening_u2", np.nan),
    )
    if result.get("systemic_velocity_arm") != SYSTEMIC_VELOCITY_ARM:
        raise ValueError(
            f"Stellar-velocity result {path} is not based exclusively on the blue arm."
        )
    if result.get("systemic_velocity_arm_policy") != SYSTEMIC_VELOCITY_ARM_POLICY:
        raise ValueError(
            f"Stellar-velocity result {path} does not enforce the blue-only, "
            "no-fallback arm policy."
        )
    if result.get("edge_trim_policy") != LSD_EDGE_TRIM_POLICY:
        raise ValueError(
            f"Stellar-velocity result {path} does not use an explicitly validated "
            "edge-trim calibration manifest."
        )
    edge_trim = result.get("edge_trim_calibration")
    if not isinstance(edge_trim, dict):
        raise ValueError(
            f"Stellar-velocity result {path} has no edge-trim calibration provenance."
        )
    if edge_trim.get("status") != "accepted_post_sysrem":
        raise ValueError(
            f"Stellar-velocity result {path} does not use an accepted edge trim."
        )
    manifest_sha256 = edge_trim.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in manifest_sha256.lower()
        )
    ):
        raise ValueError(
            f"Stellar-velocity result {path} has invalid edge-trim manifest provenance."
        )
    for field in ("left_trim_A", "right_trim_A"):
        value = float(edge_trim.get(field, np.nan))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Stellar-velocity result {path} has invalid edge-trim field {field}."
            )
    exposure_selection = result.get("science_exposure_selection")
    if not isinstance(exposure_selection, dict):
        raise ValueError(
            f"Stellar-velocity result {path} has no science-exposure selection "
            "provenance. Re-run the LSD measurement."
        )
    selection_contract = (
        exposure_selection.get("schema_version"),
        exposure_selection.get("policy"),
    )
    supported_selection_contracts = {
        (2, "science_exposure_selection_v2"),
        (
            SCIENCE_EXPOSURE_SELECTION_SCHEMA_VERSION,
            SCIENCE_EXPOSURE_SELECTION_POLICY,
        ),
    }
    if selection_contract not in supported_selection_contracts:
        raise ValueError(
            f"Stellar-velocity result {path} uses unsupported science-exposure "
            "selection metadata. Re-run the LSD measurement."
        )
    expected_selection = {
        "data_mode": str(result.get("mode", "")).strip().lower(),
        "observation_epoch": str(result.get("epoch", "")),
        "arm": SYSTEMIC_VELOCITY_ARM,
    }
    for field, expected_value in expected_selection.items():
        if str(exposure_selection.get(field, "")) != expected_value:
            raise ValueError(
                f"Stellar-velocity result {path} has science-exposure selection "
                f"{field}={exposure_selection.get(field)!r}; expected "
                f"{expected_value!r}."
            )
    nominal_names = exposure_selection.get("nominal_fits_object_names")
    if not isinstance(nominal_names, list) or not nominal_names:
        raise ValueError(
            f"Stellar-velocity result {path} has no nominal FITS OBJECT "
            "provenance. Re-run the LSD measurement."
        )
    blue_arm = result.get("arms", {}).get(SYSTEMIC_VELOCITY_ARM, {})
    if int(exposure_selection.get("n_usable_files", -1)) != int(
        blue_arm.get("n_exposures", -2)
    ):
        raise ValueError(
            f"Stellar-velocity result {path} has inconsistent usable-exposure "
            "counts between selection provenance and the blue-arm result."
        )
    template_sha256 = result.get("template_sha256")
    if (
        not isinstance(template_sha256, str)
        or len(template_sha256) != 64
        or any(character not in "0123456789abcdef" for character in template_sha256.lower())
    ):
        raise ValueError(f"Stellar-velocity result {path} has invalid template provenance.")
    template_metadata = result.get("template_parameters")
    if not isinstance(template_metadata, dict):
        raise ValueError(f"Stellar-velocity result {path} has no template parameters.")
    _validate_template_metadata(template_metadata, source=path)
    if result.get("accepted_for_stellar_rest") is not True:
        raise ValueError(f"Stellar-velocity result {path} is not accepted for stellar-rest use.")
    velocity = float(result.get("systemic_velocity_kms", np.nan))
    uncertainty = float(result.get("systemic_velocity_err_kms", np.nan))
    if not np.isfinite(velocity) or not np.isfinite(uncertainty) or uncertainty <= 0.0:
        raise ValueError(f"Stellar-velocity result {path} has invalid velocity fields.")
    return result


def mask_regions(
    wavelength: np.ndarray,
    regions: Iterable[tuple[float, float]],
) -> np.ndarray:
    """Return a pixel mask for inclusive wavelength intervals."""
    wavelength = np.asarray(wavelength, dtype=float)
    masked = np.zeros(wavelength.shape, dtype=bool)
    for lower, upper in regions:
        masked |= (wavelength >= float(lower)) & (wavelength <= float(upper))
    return masked
