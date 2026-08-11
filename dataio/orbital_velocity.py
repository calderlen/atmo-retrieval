"""Keplerian orbital velocities using transit-centered phase conventions."""

from __future__ import annotations

import numpy as np


def _validate_eccentric_geometry(
    eccentricity: float,
    omega_deg: float | None,
) -> tuple[float, float]:
    eccentricity = float(eccentricity)
    if not np.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must be finite with 0 <= e < 1.")
    if eccentricity == 0.0:
        return eccentricity, 0.0 if omega_deg is None else float(omega_deg)
    if omega_deg is None or not np.isfinite(float(omega_deg)):
        raise ValueError(
            "A non-circular planet velocity requires a finite planetary "
            "argument of periastron omega_deg."
        )
    return eccentricity, float(omega_deg)


def _eccentric_anomaly_from_true_anomaly(
    true_anomaly: np.ndarray | float,
    eccentricity: float,
) -> np.ndarray:
    true_anomaly = np.asarray(true_anomaly, dtype=float)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )


def _solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(50):
        residual = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        step = residual / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= step
        if float(np.max(np.abs(step), initial=0.0)) < 1.0e-13:
            return eccentric_anomaly
    raise RuntimeError("Kepler equation did not converge within 50 iterations.")


def planet_orbital_state_from_transit_phase(
    phase: np.ndarray,
    *,
    eccentricity: float = 0.0,
    omega_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """Return true anomaly and separation for phase zero at mid-transit.

    ``omega_deg`` is the planet's argument of periastron. Inferior conjunction
    is represented by ``nu + omega = pi/2``. This is exact for an edge-on orbit
    and is the transit-centered convention historically used by this pipeline.
    """

    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1 or np.any(~np.isfinite(phase)):
        raise ValueError("phase must be a finite one-dimensional array.")
    eccentricity, omega_deg = _validate_eccentric_geometry(
        eccentricity,
        omega_deg,
    )
    omega = np.radians(omega_deg)
    transit_true_anomaly = 0.5 * np.pi - omega
    transit_eccentric_anomaly = _eccentric_anomaly_from_true_anomaly(
        transit_true_anomaly,
        eccentricity,
    )
    transit_mean_anomaly = (
        transit_eccentric_anomaly
        - eccentricity * np.sin(transit_eccentric_anomaly)
    )
    mean_anomaly = transit_mean_anomaly + 2.0 * np.pi * phase
    eccentric_anomaly = _solve_kepler(mean_anomaly, eccentricity)
    true_anomaly = np.arctan2(
        np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - eccentricity,
    )
    separation_over_semimajor_axis = (
        1.0 - eccentricity * np.cos(eccentric_anomaly)
    )
    return {
        "true_anomaly_rad": true_anomaly,
        "separation_over_semimajor_axis": separation_over_semimajor_axis,
        "omega_rad": np.full_like(true_anomaly, omega),
    }


def planet_radial_velocity_kms(
    phase: np.ndarray,
    *,
    kp_kms: float,
    eccentricity: float = 0.0,
    omega_deg: float | None = None,
) -> np.ndarray:
    """Return the planet barycentric line-of-sight velocity in km/s.

    The sign is chosen to preserve the historical circular convention exactly:
    ``v = Kp*sin(2*pi*phase)``. For eccentric systems this becomes
    ``-Kp*(cos(nu + omega) + e*cos(omega))``.
    """

    kp_kms = float(kp_kms)
    if not np.isfinite(kp_kms) or kp_kms <= 0.0:
        raise ValueError("kp_kms must be finite and positive.")
    state = planet_orbital_state_from_transit_phase(
        phase,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
    )
    omega = state["omega_rad"]
    return -kp_kms * (
        np.cos(state["true_anomaly_rad"] + omega)
        + float(eccentricity) * np.cos(omega)
    )
