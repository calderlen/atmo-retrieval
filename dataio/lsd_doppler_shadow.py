"""Geometry and source-model contract for shared-basis LSD Doppler shadows."""

from __future__ import annotations

import numpy as np

from dataio.orbital_velocity import planet_orbital_state_from_transit_phase


FIXED_LSD_SHADOW_SCHEMA_VERSION = 1
FIXED_LSD_SHADOW_METHOD = "shared_basis_pysme_lsd_physical_rm_track_v1"


def compute_planet_positions(
    phase: np.ndarray,
    a_rs: float,
    b: float,
    lambda_angle: float,
    period: float,
    eccentricity: float = 0.0,
    periarg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sky-plane planet position in the stellar-rotation frame."""

    phase = np.asarray(phase, dtype=float)
    if eccentricity == 0.0:
        chord_position = a_rs * np.sin(2.0 * np.pi * phase)
    else:
        state = planet_orbital_state_from_transit_phase(
            phase,
            eccentricity=float(eccentricity),
            omega_deg=float(np.degrees(periarg)),
        )
        conjunction_angle = state["true_anomaly_rad"] + float(periarg)
        separation_rs = float(a_rs) * state["separation_over_semimajor_axis"]
        chord_position = -separation_rs * np.cos(conjunction_angle)

    x_planet = chord_position * np.cos(lambda_angle) + b * np.sin(lambda_angle)
    y_planet = b * np.cos(lambda_angle) - chord_position * np.sin(lambda_angle)
    return x_planet, y_planet


def compute_doppler_shadow_track(
    phase: np.ndarray,
    *,
    vsini: float,
    lambda_angle: float,
    b: float,
    rp_rs: float,
    a_rs: float,
    period: float,
    gamma1: float,
    gamma2: float,
    eccentricity: float = 0.0,
    omega_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """Return the RM velocity track and relative occulted-light weight."""

    phase = np.asarray(phase, dtype=float)
    values = np.asarray(
        [vsini, lambda_angle, b, rp_rs, a_rs, period, gamma1, gamma2, eccentricity],
        dtype=float,
    )
    if phase.ndim != 1 or np.any(~np.isfinite(phase)) or np.any(~np.isfinite(values)):
        raise ValueError("Doppler-shadow phases and geometry must be finite.")
    if vsini <= 0.0 or rp_rs <= 0.0 or a_rs <= 0.0 or period <= 0.0:
        raise ValueError("vsini, rp_rs, a_rs, and period must be positive.")
    if float(eccentricity) != 0.0 and (
        omega_deg is None or not np.isfinite(float(omega_deg))
    ):
        raise ValueError("An eccentric Doppler-shadow track requires finite omega_deg.")

    x_planet, y_planet = compute_planet_positions(
        phase,
        float(a_rs),
        float(b),
        np.radians(float(lambda_angle)),
        float(period),
        eccentricity=float(eccentricity),
        periarg=(
            0.0
            if float(eccentricity) == 0.0 and omega_deg is None
            else np.radians(float(omega_deg))
        ),
    )
    separation = np.sqrt(x_planet**2 + y_planet**2)

    overlap_area = np.zeros_like(separation)
    fully_inside = separation <= 1.0 - float(rp_rs)
    overlap_area[fully_inside] = np.pi * float(rp_rs) ** 2
    partial = (
        (separation > abs(1.0 - float(rp_rs)))
        & (separation < 1.0 + float(rp_rs))
    )
    if np.any(partial):
        distance = separation[partial]
        planet_radius = float(rp_rs)
        stellar_term = np.clip(
            (distance**2 + 1.0 - planet_radius**2) / (2.0 * distance),
            -1.0,
            1.0,
        )
        planet_term = np.clip(
            (distance**2 + planet_radius**2 - 1.0)
            / (2.0 * distance * planet_radius),
            -1.0,
            1.0,
        )
        radical = np.maximum(
            0.0,
            (-distance + 1.0 + planet_radius)
            * (distance + 1.0 - planet_radius)
            * (distance - 1.0 + planet_radius)
            * (distance + 1.0 + planet_radius),
        )
        overlap_area[partial] = (
            np.arccos(stellar_term)
            + planet_radius**2 * np.arccos(planet_term)
            - 0.5 * np.sqrt(radical)
        )

    mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(separation, 1.0) ** 2))
    limb_intensity = (
        1.0
        - float(gamma1) * (1.0 - mu)
        - float(gamma2) * (1.0 - mu) ** 2
    )
    limb_intensity = np.maximum(limb_intensity, 0.0)
    occulted_weight = overlap_area * limb_intensity
    maximum_weight = float(np.max(occulted_weight)) if occulted_weight.size else 0.0
    if maximum_weight > 0.0:
        occulted_weight = occulted_weight / maximum_weight

    return {
        "x_stellar_radii": x_planet,
        "y_stellar_radii": y_planet,
        "velocity_kms": float(vsini) * x_planet,
        "occulted_weight": occulted_weight,
        "in_transit": overlap_area > 0.0,
    }


def compute_local_doppler_shadow_profiles(
    velocity_grid: np.ndarray,
    track_velocity_kms: np.ndarray,
    occulted_weight: np.ndarray,
    *,
    local_sigma_kms: float,
    velocity_offset_kms: float = 0.0,
) -> np.ndarray:
    """Evaluate Gaussian local profiles along a physical RM track."""

    velocity_grid = np.asarray(velocity_grid, dtype=float)
    track_velocity_kms = np.asarray(track_velocity_kms, dtype=float)
    occulted_weight = np.asarray(occulted_weight, dtype=float)
    if velocity_grid.ndim != 1 or not np.all(np.diff(velocity_grid) > 0.0):
        raise ValueError("velocity_grid must be a strictly increasing 1D array.")
    if track_velocity_kms.ndim != 1 or occulted_weight.shape != track_velocity_kms.shape:
        raise ValueError("Track velocity and occulted weight must be matching 1D arrays.")
    if not np.isfinite(local_sigma_kms) or local_sigma_kms <= 0.0:
        raise ValueError("local_sigma_kms must be finite and positive.")
    center = track_velocity_kms + float(velocity_offset_kms)
    gaussian = np.exp(
        -0.5
        * (
            (velocity_grid[np.newaxis, :] - center[:, np.newaxis])
            / float(local_sigma_kms)
        )
        ** 2
    )
    return occulted_weight[:, np.newaxis] * gaussian
