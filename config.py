"""Ultra-Hot Jupiter atmospheric retrieval configuration.

This single module contains constants and static data for atmospheric retrieval
configuration. Helper APIs that derive paths, lookups, or runtime mutations live
in ``config_utils.py``.
"""

from __future__ import annotations

import os
from math import nan
from pathlib import Path

import numpy as np

# ==============================================================================
# PLANET SYSTEM PARAMETERS
# ==============================================================================

# Active planet and ephemeris (can be overridden via CLI)
PLANET = "KELT-20b"
EPHEMERIS = "Duck24"

# Fallback defaults for Doppler shadow utilities
DEFAULT_PERIOD_DAY = 1.0

# ==============================================================================
# Phase Bin Definitions
# ==============================================================================

# Standard transit contact points:
# T1 = first contact (start of ingress)
# T2 = second contact (end of ingress, start of full transit)
# T3 = third contact (end of full transit, start of egress)
# T4 = fourth contact (end of egress)

PHASE_BINS = {
    "T12": "ingress",       # T1 to T2 (first to second contact)
    "T23": "full_transit",  # T2 to T3 (second to third contact)
    "T34": "egress",        # T3 to T4 (third to fourth contact)
}

# Planet parameters dictionary
# Format: PLANETS[planet_name][ephemeris_source]
PLANETS = {
    "KELT-20b": {
        "Duck24": {
            # Ephemeris
            "period": 3.47410151,        # days
            "period_err": 0.00000012,
            "epoch": 2459757.811176,     # BJD_TDB
            "epoch_err": 0.000019,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 0.147565,        # days (T14)
            "duration_err": 0.000092,
            "tau": 0.02007,              # days (ingress/egress duration)
            "tau_err": 0.00011,
            # Orbital parameters
            "inclination": 86.12,        # degrees
            "inclination_err": 0.28,
            "a": 0.0542,                 # AU
            "eccentricity": 0.0,
            # Stellar parameters
            "M_star": 1.76,              # M_Sun
            "M_star_err": 0.19,
            "R_star": 1.565,             # R_Sun
            "R_star_err": 0.06,
            "T_star": 8720,              # K
            "logg_star": 4.290,
            "logg_star_err": 0.02,
            "Fe_H": -0.29,               # metallicity
            "v_sini_star": 117.4,        # km/s
            "v_sini_star_err": 2.9,
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": 0.0,         # deg (spin-orbit angle, ~aligned)
            "lambda_angle_err": 5.0,     # deg (estimated)
            # Limb darkening coefficients (quadratic law, ATLAS models for Teff=8720K, logg=4.3)
            # Computed for PEPSI bandpass using PyLDTk or similar
            "gamma1": 0.35,              # linear limb darkening coeff
            "gamma2": 0.25,              # quadratic limb darkening coeff
            # Derived transit geometry (for Doppler shadow model)
            # a_rs = a[AU] * 215.03 / R_star[R_Sun] = 0.0542 * 215.03 / 1.565 = 7.45
            "a_rs": 7.45,                # semi-major axis in stellar radii
            # b = a_rs * cos(i) = 7.45 * cos(86.12°) = 0.50
            "b": 0.50,                   # impact parameter
            # rp_rs from Lund+17: tau = 0.02007 days, gives Rp/Rs ~ 0.11
            "rp_rs": 0.111,              # planet-to-star radius ratio
            # Planetary parameters
            "M_p": 3.382,                # M_J (upper limit)
            "M_p_err": 0.13,
            "M_p_upper_3sigma": 3.382,
            "R_p": 1.741,                # R_J
            "R_p_err": 0.07,
            "T_eq": 2262,                # K
            "Tirr_mean": 2862,           # K (Guillot irradiation-temperature prior mean)
            "Tirr_std": 24,              # K
            "Kp": 169.0,                 # km/s
            "Kp_err": 6.1,
            "Kp_low": nan,  # TODO: look up or compute
            "Kp_high": nan,  # TODO: look up or compute
            "RV_abs": -22.78,            # km/s (systemic velocity)
            "RV_abs_err": 0.11,
            # Atmospheric parameters
            "kappa_IR": 0.04,            # infrared opacity
            "gamma": 30,                 # ratio of optical to IR opacities
            "P0": 1.0,                   # reference pressure (bar)
            "X_H2": 0.7496,              # H2 mass fraction
            "X_He": 0.2504,              # He mass fraction
            "VMR_H_minus": 1e-9,         # H- volume mixing ratio
            # Coordinates
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
        "Singh24": {
            # Ephemeris
            "period": 3.4741039,
            "period_err": 0.0000040,
            "epoch": 2459406.927174,
            "epoch_err": 0.000024,
            "duration": 0.1475,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 86.03,
            "inclination_err": 0.05,
            "a": 0.0542,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.76,
            "M_star_err": 0.19,
            "R_star": 1.60,
            "R_star_err": 0.06,
            "T_star": 8980,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 3.382,
            "M_p_err": 0.13,
            "M_p_upper_3sigma": 3.382,
            "R_p": 1.741,
            "R_p_err": 0.07,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
        "Lund17": {
            # Ephemeris
            "period": 3.4741085,
            "period_err": 0.0000019,
            "epoch": 2457503.120049,
            "epoch_err": 0.000190,
            "duration": 0.14898,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 86.12,
            "inclination_err": 0.28,
            "a": 0.0542,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.89,
            "M_star_err": 0.06,
            "R_star": 1.60,
            "R_star_err": 0.06,
            "T_star": 8980,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 3.382,
            "M_p_err": 0.13,
            "M_p_upper_3sigma": 3.382,
            "R_p": 1.735,
            "R_p_err": 0.07,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "19h38m38.74s",
            "Dec": "+31d13m09.12s",
        },
    },
    "WASP-76b": {
        "West16": {
            # Ephemeris
            "period": 1.809886,
            "period_err": 0.000001,
            "epoch": 2456107.85507,
            "epoch_err": 0.00034,
            "duration": 3.694 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 88.0,
            "inclination_err": 1.6,
            "a": 0.033,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.46,
            "M_star_err": 0.07,
            "R_star": 1.73,
            "R_star_err": 0.04,
            "T_star": 6329,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 0.92,
            "M_p_err": 0.03,
            "R_p": 1.83,
            "R_p_err": 0.06,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "01h46m31.90s",
            "Dec": "+02d42m01.40s",
        },
    },
    "KELT-9b": {
        "Gaudi17": {
            # Ephemeris (Gaudi et al. 2017; NASA Exoplanet Archive contact timing)
            "period": 1.4811235,
            "period_err": 0.0000011,
            "epoch": 2457095.68572,
            "epoch_err": 0.00014,
            "duration": 3.9158 / 24.0,
            "duration_err": 0.0115 / 24.0,
            "tau": 0.3164 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            # Orbital parameters
            "inclination": 86.79,
            "inclination_err": 0.25,
            "a": 0.03462,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 2.11,
            "M_star_err": 0.78,
            "R_star": 2.362,
            "R_star_err": 0.075,
            "T_star": 10170,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry (same reference as ephemeris)
            "a_rs": 3.1530,
            "b": 0.17700,
            "rp_rs": 0.08228,
            # Planetary parameters
            "M_p": 2.17,
            "M_p_err": 0.56,
            "R_p": 1.891,
            "R_p_err": 0.061,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "20h31m26.38s",
            "Dec": "+39d56m20.10s",
        },
        "Kokori23": {
            # Ephemeris (Kokori et al. 2023; contact timing)
            "period": 1.48111874,
            "period_err": 0.00000014,
            "epoch": 2458955.970923,
            "epoch_err": 0.000050,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 3.8541 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.3114 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 86.790,
            "inclination_err": nan,  # TODO: look up
            "a": 0.03462,
            "eccentricity": nan,  # TODO: look up
            "M_star": 2.11,
            "M_star_err": 0.78,
            "R_star": 2.362,
            "R_star_err": 0.075,
            "T_star": 10170,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 3.2000,
            "b": 0.17919,
            "rp_rs": 0.08228,
            "M_p": 2.17,
            "M_p_err": 0.56,
            "R_p": 1.891,
            "R_p_err": 0.061,
            "T_eq": nan,  # TODO: look up
            "Kp": 253.89,  # km/s; derived from 2*pi*a*sin(i)/P
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "20h31m26.38s",
            "Dec": "+39d56m20.10s",
        },
        "Wong20": {
            # Ephemeris (Wong et al. 2020; contact timing)
            "period": 1.4811235,
            "period_err": 0.0000011,
            "epoch": 2458711.586270,
            "epoch_err": 0.00025,
            "duration": 3.8751 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.2977 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 87.600,
            "inclination_err": nan,  # TODO: look up
            "a": 0.03462,
            "eccentricity": nan,  # TODO: look up
            "M_star": 2.11,
            "M_star_err": 0.78,
            "R_star": 2.362,
            "R_star_err": 0.075,
            "T_star": 10170,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 3.1910,
            "b": 0.13400,
            "rp_rs": 0.07900,
            "M_p": 2.17,
            "M_p_err": 0.56,
            "R_p": 1.891,
            "R_p_err": 0.061,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "20h31m26.38s",
            "Dec": "+39d56m20.10s",
        },
    },
    "WASP-12b": {
        "Ivshina22": {
            # Ephemeris
            "period": 1.091419108,
            "period_err": 5.5e-08,
            "epoch": 2457010.512173,
            "epoch_err": 7e-05,
            "duration": 3.0408 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 83.3,
            "inclination_err": 1.1,
            "a": 0.0234,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.38,
            "M_star_err": 0.18,
            "R_star": 1.619,
            "R_star_err": 0.065,
            "T_star": 6300,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 1.39,
            "M_p_err": 0.12,
            "R_p": 1.937,
            "R_p_err": 0.064,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "06h30m32.79s",
            "Dec": "+29d40m20.16s",
        },
    },
    "WASP-33b": {
        "Ivshina22": {
            # Ephemeris (Ivshina & Winn 2022, BJD_TDB)
            "period": 1.21987070,
            "period_err": 0.00000038,
            "epoch": 2456217.48738,
            "epoch_err": 0.00039,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 2.854 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 86.63,
            "inclination_err": 0.03,
            "a": 0.02558,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.495,
            "M_star_err": 0.031,
            "R_star": 1.509,
            "R_star_err": 0.043,
            "T_star": 7430,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 2.093,
            "M_p_err": 0.139,
            "R_p": 1.593,
            "R_p_err": 0.054,
            "T_eq": nan,  # TODO: look up
            "Kp": 227.73,  # km/s; derived from 2*pi*a*sin(i)/P
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "02h26m51.06s",
            "Dec": "+37d33m01.60s",
        },
        "Chakrabarty19": {
            # Ephemeris (Chakrabarty & Sengupta 2019; geometry + ingress from contact model)
            # Tc aligned with Ivshina22 epoch here; Chakrabarty table gave offsets relative to Tc only.
            "period": 1.21987000,
            "period_err": nan,  # TODO: look up
            "epoch": 2454163.22367,
            "epoch_err": 0.00022,
            "duration": 2.8540 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.3110 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 86.630,
            "inclination_err": nan,  # TODO: look up
            "a": 0.02558,
            "eccentricity": nan,  # TODO: look up
            "M_star": 1.495,
            "M_star_err": 0.031,
            "R_star": 1.509,
            "R_star_err": 0.043,
            "T_star": 7430,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 3.5710,
            "b": 0.21000,
            "rp_rs": 0.11180,
            "M_p": 2.093,
            "M_p_err": 0.139,
            "R_p": 1.593,
            "R_p_err": 0.054,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "02h26m51.06s",
            "Dec": "+37d33m01.60s",
        },
        "CollierCameron10": {
            # Ephemeris (Collier Cameron et al. 2010); Tc in HJD per original table — use with care vs BJD.
            "period": 1.21986690,
            "period_err": nan,  # TODO: look up
            "epoch": 2454163.223730,
            "epoch_err": 0.00030,
            "duration": 2.7371 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.2753 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 87.670,
            "inclination_err": nan,  # TODO: look up
            "a": 0.02558,
            "eccentricity": nan,  # TODO: look up
            "M_star": 1.495,
            "M_star_err": 0.031,
            "R_star": 1.509,
            "R_star_err": 0.043,
            "T_star": 7430,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 3.7879,
            "b": 0.15500,
            "rp_rs": 0.10660,
            "M_p": 2.093,
            "M_p_err": 0.139,
            "R_p": 1.593,
            "R_p_err": 0.054,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "02h26m51.06s",
            "Dec": "+37d33m01.60s",
        },
    },
    "WASP-18b": {
        "Cortes-Zuleta20": {
            # Ephemeris
            "period": 0.94145223,
            "period_err": 0.00000024,
            "epoch": 2456740.80560,
            "epoch_err": 0.00019,
            "duration": 2.21 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 83.5,
            "inclination_err": 2.0,
            "a": 0.02047,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.294,
            "M_star_err": 0.063,
            "R_star": 1.23,
            "R_star_err": 0.05,
            "T_star": 6400,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 10.20,
            "M_p_err": 0.35,
            "R_p": 1.240,
            "R_p_err": 0.079,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "01h37m25.07s",
            "Dec": "-45d40m40.06s",
        },
    },
    "WASP-189b": {
        "Anderson18": {
            # Ephemeris
            "period": 2.7240308,
            "period_err": 0.0000028,
            "epoch": 2458926.5416960,
            "epoch_err": 0.0000650,
            "duration": 4.3336 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.3721 / 24.0,  # Lendl 2020 ingress T12 (same Tc block as this ephemeris)
            "tau_err": nan,  # TODO: compute or look up
            # Orbital parameters
            "inclination": 84.03,
            "inclination_err": 0.14,
            "a": 0.05053,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 2.030,
            "M_star_err": 0.066,
            "R_star": 2.36,
            "R_star_err": 0.030,
            "T_star": 8000,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry (Lendl 2020 contact row; consistent with tau above)
            "a_rs": 4.6000,
            "b": 0.47800,
            "rp_rs": 0.07045,
            # Planetary parameters
            "M_p": 1.99,
            "M_p_err": 0.16,
            "R_p": 1.619,
            "R_p_err": 0.021,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "15h02m44.82s",
            "Dec": "-03d01m53.35s",
        },
        "Lendl20": {
            # Ephemeris (Lendl et al. 2020; BJD-TT in NASA Exoplanet Archive)
            "period": 2.72403300,
            "period_err": nan,  # TODO: look up
            "epoch": 2458926.541696,
            "epoch_err": 0.000065,
            "epoch_scale": "tt",
            "epoch_reference": "barycenter",
            "duration": 4.3336 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.3721 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 84.03,
            "inclination_err": 0.14,
            "a": 0.05053,
            "eccentricity": nan,  # TODO: look up
            "M_star": 2.030,
            "M_star_err": 0.066,
            "R_star": 2.36,
            "R_star_err": 0.030,
            "T_star": 8000,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 4.6000,
            "b": 0.47800,
            "rp_rs": 0.07045,
            "M_p": 1.99,
            "M_p_err": 0.16,
            "R_p": 1.619,
            "R_p_err": 0.021,
            "T_eq": nan,  # TODO: look up
            "Kp": 200.71,  # km/s; derived from 2*pi*a*sin(i)/P
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "15h02m44.82s",
            "Dec": "-03d01m53.35s",
        },
        "Deline22": {
            # Ephemeris (Deline et al. 2022; contact timing)
            "period": 2.72403500,
            "period_err": nan,  # TODO: look up
            "epoch": 2459021.882937,
            "epoch_err": 0.000048,
            "duration": 4.4917 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.3591 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 84.580,
            "inclination_err": nan,  # TODO: look up
            "a": 0.05053,
            "eccentricity": nan,  # TODO: look up
            "M_star": 2.030,
            "M_star_err": 0.066,
            "R_star": 2.36,
            "R_star_err": 0.030,
            "T_star": 8000,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 4.5870,
            "b": 0.43300,
            "rp_rs": 0.06958,
            "M_p": 1.99,
            "M_p_err": 0.16,
            "R_p": 1.619,
            "R_p_err": 0.021,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "15h02m44.82s",
            "Dec": "-03d01m53.35s",
        },
    },
    "MASCARA-1b": {
        "Talens17": {
            # Ephemeris
            "period": 2.14877381,
            "period_err": 0.00000088,
            "epoch": 2458833.488151,
            "epoch_err": 0.000092,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 4.226 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 88.45,
            "inclination_err": 0.17,
            "a": 0.04034,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.900,
            "M_star_err": 0.068,
            "R_star": 2.082,
            "R_star_err": 0.038,
            "T_star": 7554,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 3.7,
            "M_p_err": 0.9,
            "R_p": 1.597,
            "R_p_err": 0.037,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "21h10m12.37s",
            "Dec": "+10d44m20.03s",
        },
    },
    "TOI-1431b": {
        "Addison21": {
            # Ephemeris
            "period": 2.650237,
            "period_err": 0.000003,
            "epoch": 2458739.17737,
            "epoch_err": 0.00007,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 2.489 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: look up
            "tau_err": float('nan'),  # TODO: look up
            # Orbital parameters
            "inclination": 80.13,
            "inclination_err": 0.13,
            "a": 0.046,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.90,
            "M_star_err": 0.10,
            "R_star": 1.92,
            "R_star_err": 0.07,
            "T_star": 7690,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry
            "a_rs": float('nan'),  # TODO: look up
            "b": float('nan'),  # TODO: look up
            "rp_rs": float('nan'),  # TODO: look up
            # Planetary parameters
            "M_p": 3.12,
            "M_p_err": 0.18,
            "R_p": 1.49,
            "R_p_err": 0.05,
            "T_eq": nan,  # TODO: look up
            "Kp": 186.03,  # km/s; derived from 2*pi*a*sin(i)/P
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "Ks_expected": 294.1,  # m/s
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "21h04m48.89s",
            "Dec": "+55d35m16.88s",
        },
    },
    "TOI-1518b": {
        "Cabot21": {
            # Ephemeris (Cabot et al. 2021); contact table lists T14 calc, no T12 — tau left unset
            "period": 1.902603,
            "period_err": 0.000011,
            "epoch": 2458787.049255,
            "epoch_err": 0.000094,
            "duration": 2.1744 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: compute from contact times or look up
            "tau_err": float('nan'),  # TODO: compute or look up
            # Orbital parameters
            "inclination": 77.84,
            "inclination_err": 0.26,
            "a": 0.0389,
            "eccentricity": nan,  # TODO: look up
            # Stellar parameters
            "M_star": 1.79,
            "M_star_err": 0.26,
            "R_star": 1.95,
            "R_star_err": 0.08,
            "T_star": 7300,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            # Spin-orbit alignment (Doppler shadow)
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            # Limb darkening coefficients
            "gamma1": float('nan'),  # TODO: look up
            "gamma2": float('nan'),  # TODO: look up
            # Transit geometry (Cabot et al. 2021 contact row)
            "a_rs": 4.2910,
            "b": 0.90360,
            "rp_rs": 0.09880,
            # Planetary parameters
            "M_p": 2.3,
            "M_p_err": 2.3,
            "R_p": 1.875,
            "R_p_err": 0.053,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            # Atmospheric parameters
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            # Coordinates
            "RA": "23h29m04.20s",
            "Dec": "+67d02m05.30s",
        },
        "Simonnin25": {
            # Ephemeris (Simonnin et al. 2025; contact timing)
            "period": 1.90261131,
            "period_err": 0.00000043,
            "epoch": 2459983.791942,
            "epoch_err": 0.000066,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 2.3950 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": 0.8552 / 24.0,
            "tau_err": nan,  # TODO: compute or look up
            "inclination": 77.626,
            "inclination_err": nan,  # TODO: look up
            "a": 0.0389,
            "eccentricity": nan,  # TODO: look up
            "M_star": 1.79,
            "M_star_err": 0.26,
            "R_star": 1.95,
            "R_star_err": 0.08,
            "T_star": 7300,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 4.1090,
            "b": 0.88060,
            "rp_rs": 0.09939,
            "M_p": 2.3,
            "M_p_err": 2.3,
            "R_p": 1.875,
            "R_p_err": 0.053,
            "T_eq": nan,  # TODO: look up
            "Kp": 217.26,  # km/s; derived from 2*pi*a*sin(i)/P
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "23h29m04.20s",
            "Dec": "+67d02m05.30s",
        },
        "Kokori23": {
            # Ephemeris (Kokori et al. 2023); contact table has no ingress — tau unset
            "period": 1.90261440,
            "period_err": 0.0000016,
            "epoch": 2458806.075406,
            "epoch_err": 0.000096,
            "duration": 2.1731 / 24.0,
            "duration_err": nan,  # TODO: look up
            "tau": float('nan'),  # TODO: compute from contact times or look up
            "tau_err": float('nan'),  # TODO: compute or look up
            "inclination": 77.840,
            "inclination_err": 0.26,
            "a": 0.0389,
            "eccentricity": nan,  # TODO: look up
            "M_star": 1.79,
            "M_star_err": 0.26,
            "R_star": 1.95,
            "R_star_err": 0.08,
            "T_star": 7300,
            "logg_star": nan,  # TODO: look up
            "logg_star_err": nan,  # TODO: look up
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": 4.2910,
            "b": 0.90387,
            "rp_rs": 0.09880,
            "M_p": 2.3,
            "M_p_err": 2.3,
            "R_p": 1.875,
            "R_p_err": 0.053,
            "T_eq": nan,  # TODO: look up
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "23h29m04.20s",
            "Dec": "+67d02m05.30s",
        },
    },
    "TOI-1413.01": {
        "ExoFOP": {
            # TOI-1413.01 candidate (ExoFOP / NASA Exoplanet Archive); many fields incomplete
            "profile_kind": "source_ephemeris",
            "ephemeris_source": "ExoFOP",
            "timing_model": "linear",
            "timing_source": "ExoFOP",
            "period": 6.11821285129936,
            "period_err": 0.00085346506,
            "epoch": 2459829.69897600,
            "epoch_err": 0.00158291,
            "epoch_scale": "tdb",
            "epoch_reference": "barycenter",
            "duration": 0.667144093941722 / 24.0,
            "duration_err": 0.17293526 / 24.0,
            "tau": float('nan'),  # TODO: compute from contact times or look up
            "tau_err": float('nan'),  # TODO: compute or look up
            "inclination": nan,  # TODO: look up or compute
            "inclination_err": nan,  # TODO: look up
            "a": nan,  # TODO: look up or compute
            "eccentricity": nan,  # TODO: look up
            "M_star": 0.945,
            "M_star_err": 0.122,
            "R_star": 0.8901090,
            "R_star_err": 0.052,
            "T_star": 5427,
            "logg_star": 4.51461,
            "logg_star_err": 0.085,
            "Fe_H": nan,  # TODO: look up
            "v_sini_star": nan,  # TODO: look up
            "v_sini_star_err": nan,  # TODO: look up
            "lambda_angle": float('nan'),  # TODO: look up
            "lambda_angle_err": float('nan'),  # TODO: look up
            "gamma1": float('nan'),  # TODO: look up or compute
            "gamma2": float('nan'),  # TODO: look up or compute
            "a_rs": float('nan'),  # TODO: compute or look up
            "b": float('nan'),  # TODO: compute or look up
            "rp_rs": float('nan'),  # TODO: compute or look up
            "M_p": nan,  # TODO: look up
            "M_p_err": nan,  # TODO: look up
            "R_p": 0.549,
            "R_p_err": 0.02,
            "T_eq": 891.0,
            "Kp": nan,  # TODO: look up
            "Kp_err": nan,  # TODO: look up
            "RV_abs": nan,  # TODO: look up
            "RV_abs_err": nan,  # TODO: look up
            "kappa_IR": nan,  # TODO: look up
            "gamma": nan,  # TODO: look up
            "P0": nan,  # TODO: look up
            "X_H2": nan,  # TODO: look up
            "X_He": nan,  # TODO: look up
            "VMR_H_minus": nan,  # TODO: look up
            "RA": "22h13m00.76s",
            "Dec": "+37d37m39.27s",
        },
    },
}

# ==============================================================================
# RECOMMENDED COMPOSITE SYSTEM PROFILES
# ==============================================================================
#
# These flat profiles intentionally combine the most useful timing, transit-
# geometry, and stellar-line-profile measurements.  The older publication-
# keyed profiles above remain available for reproducibility.  Runtime callers
# should request ephemeris="Recommended" for the composite values below.

PLANETS["KELT-20b"]["Recommended"] = {
    **PLANETS["KELT-20b"]["Duck24"],
    "timing_model": "linear",
    "timing_source": "Lenhart25",
    "period": 3.47410151,
    "period_err": 0.00000012,
    "epoch": 2459757.811176,
    "epoch_err": 0.000019,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 3.54156 / 24.0,
    "duration_err": 0.002195 / 24.0,
    "tau": 0.48168 / 24.0,
    "tau_err": 0.00264 / 24.0,
    "geometry_source": "Singh24; lambda from Lund17",
    "inclination": 86.03,
    "inclination_err": 0.05,
    "a": 0.0542,
    "a_rs": 7.4579,
    "b": 0.515,
    "rp_rs": 0.11572,
    "eccentricity": 0.0,
    "lambda_angle": 3.4,
    "lambda_angle_err": 2.1,
    "rotation_source": "Singh24",
    "v_sini_star": 117.4,
    "v_sini_star_err": 2.9,
    "limb_darkening_source": "Singh24 q1/q2 conversion",
    "gamma1": 0.3265,
    "gamma2": 0.1805,
}

PLANETS["WASP-76b"]["Recommended"] = {
    **PLANETS["WASP-76b"]["West16"],
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 1.80988044,
    "period_err": 0.00000026,
    "epoch": 2459313.154379,
    "epoch_err": 0.000058,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 3.79 / 24.0,
    "duration_err": nan,
    "tau": 0.380 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Demangeon24 transit geometry",
    "geometry_source": "Demangeon24; lambda from Ehrenreich20",
    "inclination": 89.620,
    "a": 0.033,
    "a_rs": 4.1088,
    "b": 0.027,
    "rp_rs": 0.109284,
    "eccentricity": 0.00180,
    "omega": 51.0,
    "lambda_angle": 61.28,
    "lambda_angle_err": 6.335,
    "rotation_source": "ESPRESSO high-resolution transit analysis",
    "v_sini_star": 2.33,
    "v_sini_star_err": 0.36,
    "limb_darkening_source": "ESPRESSO high-resolution transit analysis",
    "gamma1": 0.393,
    "gamma2": 0.219,
}

PLANETS["KELT-9b"]["Recommended"] = {
    **PLANETS["KELT-9b"]["Kokori23"],
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 1.48111897,
    "period_err": 0.00000013,
    "epoch": 2459074.460549,
    "epoch_err": 0.000050,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 3.8541 / 24.0,
    "duration_err": nan,
    "tau": 0.3114 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Gaudi17 reference transit geometry",
    "geometry_source": "Gaudi17; lambda from later tomography",
    "inclination": 86.79,
    "inclination_err": 0.25,
    "a": 0.03462,
    "a_rs": 3.20,
    "b": 0.179,
    "rp_rs": 0.08228,
    "eccentricity": 0.0,
    "lambda_angle": -84.8,
    "lambda_angle_err": 1.4,
    "rotation_source": "Borsa19",
    "v_sini_star": 111.4,
    "v_sini_star_err": 1.3,
    "limb_darkening_source": "Jones22",
    "gamma1": 0.2541,
    "gamma2": 0.320,
}

PLANETS["WASP-12b"]["Recommended"] = {
    **PLANETS["WASP-12b"]["Ivshina22"],
    "timing_model": "quadratic",
    "timing_source": "Wong22Decay",
    "period": 1.091419370,
    "period_err": 0.000000020,
    "epoch": 2457103.283654,
    "epoch_err": 0.000032,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "dperiod_depoch": -1.031e-9,
    "dperiod_depoch_err": 0.033e-9,
    "duration": 3.0408 / 24.0,
    "duration_err": nan,
    "tau": 0.365 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Wong22 transit geometry",
    "geometry_source": "Wong22",
    "inclination": 83.54,
    "inclination_err": 0.74,
    "a": 0.0234,
    "a_rs": 3.061,
    "b": 0.344,
    "rp_rs": 0.11600,
    "eccentricity": 0.0,
    "lambda_angle": 59.0,
    "lambda_angle_err": 17.5,
    "rotation_source": "Wong22",
    "v_sini_star": 1.6,
    "v_sini_star_err": 1.5,
    "limb_darkening_source": "Wong22",
    "gamma1": 0.24,
    "gamma2": 0.31,
}

PLANETS["WASP-33b"]["Recommended"] = {
    **PLANETS["WASP-33b"]["Ivshina22"],
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 1.21987081,
    "period_err": 0.00000011,
    "epoch": 2458518.16266,
    "epoch_err": 0.00014,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 2.854 / 24.0,
    "duration_err": nan,
    "tau": 0.306 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Smith25 transit geometry",
    "geometry_source": "Smith25",
    "inclination": 88.96,
    "a": 0.02558,
    "a_rs": 3.512,
    "b": 0.064,
    "rp_rs": 0.10696,
    "eccentricity": 0.0,
    "lambda_angle": -112.0,
    "lambda_angle_err": nan,
    "rotation_source": "Smith25",
    "v_sini_star": 85.64,
    "v_sini_star_err": 0.13,
    "limb_darkening_source": "Smith25 u-plus/u-minus conversion",
    "gamma1": 0.2385,
    "gamma2": 0.4985,
}

PLANETS["WASP-18b"]["Recommended"] = {
    **PLANETS["WASP-18b"]["Cortes-Zuleta20"],
    "timing_model": "linear",
    "timing_source": "SalguneswaranNediyedath26",
    "period": 0.94145252,
    "period_err": 0.000000011,
    "epoch": 2460933.096346,
    "epoch_err": 0.000022,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 2.1790 / 24.0,
    "duration_err": 0.0017 / 24.0,
    "tau": 0.225 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Deline25 transit geometry",
    "geometry_source": "Deline25",
    "inclination": 84.08,
    "inclination_err": 0.17,
    "a": 0.02041,
    "a_rs": 3.493,
    "b": 0.361,
    "rp_rs": 0.09757,
    "eccentricity": 0.00852,
    "omega": 261.9,
    "lambda_angle": 4.0,
    "lambda_angle_err": 5.0,
    "rotation_source": "Deline25",
    "v_sini_star": 10.9,
    "v_sini_star_err": 0.7,
    "limb_darkening_source": "Deline25",
    "gamma1": 0.357,
    "gamma2": 0.229,
}

PLANETS["WASP-189b"]["Recommended"] = {
    **PLANETS["WASP-189b"]["Lendl20"],
    "timing_model": "linear",
    "timing_source": "Yumoto26",
    "period": 2.72403141,
    "period_err": 0.00000035,
    "epoch": 2456706.45616,
    "epoch_err": 0.00031,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 4.3336 / 24.0,
    "duration_err": 0.0056 / 24.0,
    "tau": 0.3721 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Lendl20 transit geometry",
    "geometry_source": "Lendl20; lambda from Prinoth24",
    "inclination": 84.03,
    "inclination_err": 0.14,
    "a": 0.05053,
    "a_rs": 4.600,
    "b": 0.478,
    "rp_rs": 0.07045,
    "eccentricity": 0.0,
    "lambda_angle": 90.07,
    "lambda_angle_err": 0.24,
    "rotation_source": "Prinoth24",
    "v_sini_star": 95.05,
    "v_sini_star_err": 0.55,
    "limb_darkening_source": "Prinoth24",
    "gamma1": 0.414,
    "gamma2": 0.155,
}

PLANETS["MASCARA-1b"]["Recommended"] = {
    **PLANETS["MASCARA-1b"]["Talens17"],
    "timing_model": "linear",
    "timing_source": "Yumoto26",
    "period": 2.14876998,
    "period_err": 0.00000013,
    "epoch": 2458833.488125,
    "epoch_err": 0.000079,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 4.226 / 24.0,
    "duration_err": 0.0105 / 24.0,
    "tau": 0.318 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Hooton22 transit geometry",
    "geometry_source": "Hooton22",
    "inclination": 88.45,
    "inclination_err": 0.17,
    "a": 0.040352,
    "a_rs": 4.1676,
    "b": 0.113,
    "rp_rs": 0.07884,
    "eccentricity": 0.00034,
    "omega": -16.0,
    "lambda_angle": -69.2,
    "lambda_angle_err": 3.25,
    "rotation_source": "Hooton22",
    "v_sini_star": 101.7,
    "v_sini_star_err": 3.85,
    "limb_darkening_source": "Hooton22 q1/q2 conversion",
    "gamma1": 0.3918,
    "gamma2": 0.0919,
}

PLANETS["TOI-1431b"]["Recommended"] = {
    **PLANETS["TOI-1431b"]["Addison21"],
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 2.65023153,
    "period_err": 0.00000038,
    "epoch": 2459558.098917,
    "epoch_err": 0.000068,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 2.49 / 24.0,
    "duration_err": nan,
    "tau": 0.714 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Addison21 transit geometry",
    "geometry_source": "Addison21; lambda from Stangret21",
    "inclination": 80.13,
    "inclination_err": 0.13,
    "a": 0.046,
    "a_rs": 5.15,
    "b": 0.881,
    "rp_rs": 0.07955,
    "eccentricity": 0.0022,
    "omega": 108.0,
    "lambda_angle": -155.0,
    "lambda_angle_err": 15.0,
    "rotation_source": "Addison21",
    "v_sini_star": 6.0,
    "v_sini_star_err": 0.2,
    "limb_darkening_source": "Addison21",
    "gamma1": 0.13,
    "gamma2": 0.33,
}

PLANETS["TOI-1518b"]["Recommended"] = {
    **PLANETS["TOI-1518b"]["Simonnin25"],
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 1.90261185,
    "period_err": 0.00000024,
    "epoch": 2459616.588037,
    "epoch_err": 0.000053,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 2.3950 / 24.0,
    "duration_err": 0.00805 / 24.0,
    "tau": 0.8552 / 24.0,
    "tau_err": nan,
    "tau_derived_from": "Simonnin25 transit geometry",
    "geometry_source": "Simonnin25; lambda from PIRANGA III",
    "inclination": 77.626,
    "inclination_err": 0.097,
    "a": 0.03712,
    "a_rs": 4.109,
    "b": 0.8806,
    "rp_rs": 0.09939,
    "eccentricity": 0.0,
    "lambda_angle": -113.56,
    "lambda_angle_err": 0.54,
    "rotation_source": "PIRANGA III",
    "v_sini_star": 74.4,
    "v_sini_star_err": 2.3,
    "limb_darkening_source": "PIRANGA III q1/q2 conversion",
    "gamma1": 0.4426,
    "gamma2": 0.2736,
}

PLANETS["TOI-1413.01"]["Recommended"] = {
    **PLANETS["TOI-1413.01"]["ExoFOP"],
    "timing_model": "linear",
    "timing_source": "ExoFOP",
    "period": 6.1182128513,
    "period_err": 0.0008534651,
    "epoch": 2459829.698976,
    "epoch_err": 0.00158291,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.66714 / 24.0,
    "duration_err": 0.17294 / 24.0,
    "tau": nan,
    "tau_err": nan,
    "geometry_source": "unavailable",
    "v_sini_star": nan,
    "v_sini_star_err": nan,
    "gamma1": nan,
    "gamma2": nan,
}

PLANETS["HAT-P-11b"] = {
    "Recommended": {
        "timing_model": "linear",
        "timing_source": "Basilicata24",
        "period": 4.887802443,
        "period_err": 0.000000034,
        "epoch": 2454957.8132067,
        "epoch_err": 0.0000053,
        "epoch_scale": "tdb",
        "epoch_reference": "barycenter",
        "duration": 2.35562 / 24.0,
        "duration_err": nan,
        "tau": 0.138 / 24.0,
        "tau_err": nan,
        "tau_derived_from": "Basilicata24 eccentric transit geometry",
        "geometry_source": "Basilicata24",
        "inclination": 89.027,
        "inclination_err": 0.068,
        "a": 0.05254,
        "a_rs": 15.05,
        "b": 0.227,
        "rp_rs": 0.058993,
        "eccentricity": 0.2577,
        "omega": 192.0,
        "lambda_angle": 103.0,
        "lambda_angle_err": 18.0,
        "M_star": 0.809,
        "M_star_err": 0.020,
        "R_star": 0.683,
        "R_star_err": 0.009,
        "T_star": 4780,
        "logg_star": 4.59,
        "logg_star_err": nan,
        "Fe_H": 0.31,
        "rotation_source": "Basilicata24",
        "v_sini_star": 0.670,
        "v_sini_star_err": 0.095,
        "limb_darkening_source": "Basilicata24 q1/q2 conversion",
        "gamma1": 0.6560,
        "gamma2": 0.0255,
        "M_p": 0.0736,
        "M_p_err": nan,
        "R_p": 0.389,
        "R_p_err": nan,
        "T_eq": nan,
        "Kp": nan,
        "Kp_err": nan,
        "RV_abs": nan,
        "RV_abs_err": nan,
        "kappa_IR": nan,
        "gamma": nan,
        "P0": nan,
        "X_H2": nan,
        "X_He": nan,
        "VMR_H_minus": nan,
        "RA": "19h50m50.44s",
        "Dec": "+48d04m54.71s",
    },
}

PLANETS["KELT-5b"] = {
    "Recommended": {
        "timing_model": "linear",
        "timing_source": "ExoFOP",
        "period": 3.0197806980,
        "period_err": 0.0001354653,
        "epoch": 2459853.583682,
        "epoch_err": 0.00070863,
        "epoch_scale": "tdb",
        "epoch_reference": "barycenter",
        "duration": 3.58968 / 24.0,
        "duration_err": 0.04127 / 24.0,
        "tau": nan,
        "tau_err": nan,
        "geometry_source": "unavailable",
        "inclination": nan,
        "inclination_err": nan,
        "a": nan,
        "a_rs": nan,
        "b": nan,
        "rp_rs": 0.10962,
        "rp_rs_derived_from": "ExoFOP transit depth",
        "eccentricity": nan,
        "lambda_angle": nan,
        "lambda_angle_err": nan,
        "M_star": nan,
        "M_star_err": nan,
        "R_star": nan,
        "R_star_err": nan,
        "T_star": 7128,
        "logg_star": 4.24,
        "logg_star_err": nan,
        "Fe_H": 0.0,
        "v_sini_star": nan,
        "v_sini_star_err": nan,
        "gamma1": nan,
        "gamma2": nan,
        "M_p": nan,
        "M_p_err": nan,
        "R_p": nan,
        "R_p_err": nan,
        "T_eq": nan,
        "Kp": nan,
        "Kp_err": nan,
        "RV_abs": nan,
        "RV_abs_err": nan,
        "kappa_IR": nan,
        "gamma": nan,
        "P0": nan,
        "X_H2": nan,
        "X_He": nan,
        "VMR_H_minus": nan,
        "RA": "23h26m19.13s",
        "Dec": "+38d33m10.71s",
    },
}

PLANETS["TOI-1789b"] = {
    "Recommended": {
        "timing_model": "linear",
        "timing_source": "Kokori25",
        "period": 3.2087107,
        "period_err": 0.0000035,
        "epoch": 2459470.47469,
        "epoch_err": 0.00031,
        "epoch_scale": "tdb",
        "epoch_reference": "barycenter",
        "duration": 2.29 / 24.0,
        "duration_err": nan,
        "tau": nan,
        "tau_err": nan,
        "grazing_transit": True,
        "geometry_source": "Khandelwal21/22",
        "inclination": 78.41,
        "inclination_err": 0.47,
        "a": 0.04882,
        "a_rs": 4.83,
        "b": 0.972,
        "rp_rs": 0.0661,
        "eccentricity": 0.0,
        "lambda_angle": nan,
        "lambda_angle_err": nan,
        "M_star": 1.507,
        "M_star_err": 0.059,
        "R_star": 2.168,
        "R_star_err": 0.036,
        "T_star": 5991,
        "logg_star": 3.943,
        "logg_star_err": nan,
        "Fe_H": 0.373,
        "rotation_source": "Khandelwal21/22",
        "v_sini_star": 7.0,
        "v_sini_star_err": 0.5,
        "limb_darkening_source": "Khandelwal21/22",
        "gamma1": 0.263,
        "gamma2": 0.284,
        "M_p": 0.70,
        "M_p_err": nan,
        "R_p": 1.44,
        "R_p_err": nan,
        "T_eq": 1927.0,
        "Kp": nan,
        "Kp_err": nan,
        "RV_abs": nan,
        "RV_abs_err": nan,
        "kappa_IR": nan,
        "gamma": nan,
        "P0": nan,
        "X_H2": nan,
        "X_He": nan,
        "VMR_H_minus": nan,
        "RA": "09h30m58.42s",
        "Dec": "+26d32m23.98s",
    },
}

PLANETS["V1298 Tau b"] = {
    "Recommended": {
        "timing_model": "ttv_table",
        "timing_source": "Livingston26TTV",
        "period": 24.140006,
        "period_err": 0.000017,
        "epoch": 2458298.209601,
        "epoch_err": 0.000585,
        "epoch_scale": "tdb",
        "epoch_reference": "barycenter",
        # Johnson22 measured the transit observed by the 2019 PEPSI sequence.
        "transit_midpoints_bjd_tdb": {"20191024": 2458781.083656},
        "duration": 6.45 / 24.0,
        "duration_err": 0.055 / 24.0,
        "tau": 0.6185965 / 24.0,
        "tau_err": nan,
        "tau_derived_from": "Johnson22 a_rs/b/inclination/rp_rs, circular contact formula",
        "geometry_source": "Johnson22",
        "inclination": 88.759,
        "inclination_err": 0.077,
        "a": 0.1685,
        "a_rs": 26.06,
        "b": 0.564,
        "rp_rs": 0.071998,
        # Adopt a circular orbit for spectral trail and Doppler-shadow work.
        # Preserve the previous small-e value because omega is unavailable.
        "eccentricity": 0.0,
        "eccentricity_reported": 0.0079,
        "eccentricity_model": "adopted_circular",
        "eccentricity_adoption_reason": (
            "e=0.0079 is negligible for this spectral analysis and omega is unavailable"
        ),
        "lambda_angle": 4.0,
        "lambda_angle_err": 8.5,
        "M_star": 1.10,
        "M_star_err": 0.05,
        "R_star": 1.32,
        "R_star_err": 0.05,
        "T_star": 4970,
        "logg_star": nan,
        "logg_star_err": nan,
        "Fe_H": nan,
        "rotation_source": "Johnson22",
        "v_sini_star": 24.77,
        "v_sini_star_err": 0.19,
        "limb_darkening_source": "Johnson22",
        "gamma1": 0.54,
        "gamma2": -0.03,
        "M_p": 0.0412,
        "M_p_err": nan,
        "R_p": 0.840,
        "R_p_err": nan,
        "T_eq": 666.0,
        "Kp": nan,
        "Kp_err": nan,
        "RV_abs": nan,
        "RV_abs_err": nan,
        "kappa_IR": nan,
        "gamma": nan,
        "P0": nan,
        "X_H2": nan,
        "X_He": nan,
        "VMR_H_minus": nan,
        "RA": "04h05m19.60s",
        "Dec": "+20d09m25.31s",
    },
}

# Source-keyed versions of every newly supplied timing solution. Each
# dictionary below explicitly owns its parameters; none is reconstructed
# from a Recommended profile. Mixed-source fields retain their own
# timing/geometry/rotation/limb-darkening provenance.
PLANETS["KELT-20b"]["Lenhart25"] = {
    "period": 3.47410151,
    "period_err": 1.2e-07,
    "epoch": 2459757.811176,
    "epoch_err": 1.9e-05,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.147565,
    "duration_err": 9.145833333333332e-05,
    "tau": 0.02007,
    "tau_err": 0.00011,
    "inclination": 86.03,
    "inclination_err": 0.05,
    "a": 0.0542,
    "eccentricity": 0.0,
    "M_star": 1.76,
    "M_star_err": 0.19,
    "R_star": 1.565,
    "R_star_err": 0.06,
    "T_star": 8720,
    "logg_star": 4.29,
    "logg_star_err": 0.02,
    "Fe_H": -0.29,
    "v_sini_star": 117.4,
    "v_sini_star_err": 2.9,
    "lambda_angle": 3.4,
    "lambda_angle_err": 2.1,
    "gamma1": 0.3265,
    "gamma2": 0.1805,
    "a_rs": 7.4579,
    "b": 0.515,
    "rp_rs": 0.11572,
    "M_p": 3.382,
    "M_p_err": 0.13,
    "M_p_upper_3sigma": 3.382,
    "R_p": 1.741,
    "R_p_err": 0.07,
    "T_eq": 2262,
    "Tirr_mean": 2862,
    "Tirr_std": 24,
    "Kp": 169.0,
    "Kp_err": 6.1,
    "Kp_low": nan,
    "Kp_high": nan,
    "RV_abs": -22.78,
    "RV_abs_err": 0.11,
    "kappa_IR": 0.04,
    "gamma": 30,
    "P0": 1.0,
    "X_H2": 0.7496,
    "X_He": 0.2504,
    "VMR_H_minus": 1e-09,
    "RA": "19h38m38.74s",
    "Dec": "+31d13m09.12s",
    "timing_model": "linear",
    "timing_source": "Lenhart25",
    "geometry_source": "Singh24; lambda from Lund17",
    "rotation_source": "Singh24",
    "limb_darkening_source": "Singh24 q1/q2 conversion",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Lenhart25",
}

PLANETS["WASP-76b"]["Kokori25"] = {
    "period": 1.80988044,
    "period_err": 2.6e-07,
    "epoch": 2459313.154379,
    "epoch_err": 5.8e-05,
    "duration": 0.15791666666666668,
    "duration_err": nan,
    "tau": 0.015833333333333335,
    "tau_err": nan,
    "inclination": 89.62,
    "inclination_err": 1.6,
    "a": 0.033,
    "eccentricity": 0.0018,
    "M_star": 1.46,
    "M_star_err": 0.07,
    "R_star": 1.73,
    "R_star_err": 0.04,
    "T_star": 6329,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 2.33,
    "v_sini_star_err": 0.36,
    "lambda_angle": 61.28,
    "lambda_angle_err": 6.335,
    "gamma1": 0.393,
    "gamma2": 0.219,
    "a_rs": 4.1088,
    "b": 0.027,
    "rp_rs": 0.109284,
    "M_p": 0.92,
    "M_p_err": 0.03,
    "R_p": 1.83,
    "R_p_err": 0.06,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "01h46m31.90s",
    "Dec": "+02d42m01.40s",
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "tau_derived_from": "Demangeon24 transit geometry",
    "geometry_source": "Demangeon24; lambda from Ehrenreich20",
    "omega": 51.0,
    "rotation_source": "ESPRESSO high-resolution transit analysis",
    "limb_darkening_source": "ESPRESSO high-resolution transit analysis",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["KELT-9b"]["Kokori25"] = {
    "period": 1.48111897,
    "period_err": 1.3e-07,
    "epoch": 2459074.460549,
    "epoch_err": 5e-05,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.1605875,
    "duration_err": nan,
    "tau": 0.012975,
    "tau_err": nan,
    "inclination": 86.79,
    "inclination_err": 0.25,
    "a": 0.03462,
    "eccentricity": 0.0,
    "M_star": 2.11,
    "M_star_err": 0.78,
    "R_star": 2.362,
    "R_star_err": 0.075,
    "T_star": 10170,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 111.4,
    "v_sini_star_err": 1.3,
    "lambda_angle": -84.8,
    "lambda_angle_err": 1.4,
    "gamma1": 0.2541,
    "gamma2": 0.32,
    "a_rs": 3.2,
    "b": 0.179,
    "rp_rs": 0.08228,
    "M_p": 2.17,
    "M_p_err": 0.56,
    "R_p": 1.891,
    "R_p_err": 0.061,
    "T_eq": nan,
    "Kp": 253.89,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "20h31m26.38s",
    "Dec": "+39d56m20.10s",
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "tau_derived_from": "Gaudi17 reference transit geometry",
    "geometry_source": "Gaudi17; lambda from later tomography",
    "rotation_source": "Borsa19",
    "limb_darkening_source": "Jones22",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["WASP-12b"]["Wong22Decay"] = {
    "period": 1.09141937,
    "period_err": 2e-08,
    "epoch": 2457103.283654,
    "epoch_err": 3.2e-05,
    "duration": 0.1267,
    "duration_err": nan,
    "tau": 0.015208333333333332,
    "tau_err": nan,
    "inclination": 83.54,
    "inclination_err": 0.74,
    "a": 0.0234,
    "eccentricity": 0.0,
    "M_star": 1.38,
    "M_star_err": 0.18,
    "R_star": 1.619,
    "R_star_err": 0.065,
    "T_star": 6300,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 1.6,
    "v_sini_star_err": 1.5,
    "lambda_angle": 59.0,
    "lambda_angle_err": 17.5,
    "gamma1": 0.24,
    "gamma2": 0.31,
    "a_rs": 3.061,
    "b": 0.344,
    "rp_rs": 0.116,
    "M_p": 1.39,
    "M_p_err": 0.12,
    "R_p": 1.937,
    "R_p_err": 0.064,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "06h30m32.79s",
    "Dec": "+29d40m20.16s",
    "timing_model": "quadratic",
    "timing_source": "Wong22Decay",
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "dperiod_depoch": -1.031e-09,
    "dperiod_depoch_err": 3.3e-11,
    "tau_derived_from": "Wong22 transit geometry",
    "geometry_source": "Wong22",
    "rotation_source": "Wong22",
    "limb_darkening_source": "Wong22",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Wong22Decay",
}

PLANETS["WASP-33b"]["Kokori25"] = {
    "period": 1.21987081,
    "period_err": 1.1e-07,
    "epoch": 2458518.16266,
    "epoch_err": 0.00014,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.11891666666666667,
    "duration_err": nan,
    "tau": 0.01275,
    "tau_err": nan,
    "inclination": 88.96,
    "inclination_err": 0.03,
    "a": 0.02558,
    "eccentricity": 0.0,
    "M_star": 1.495,
    "M_star_err": 0.031,
    "R_star": 1.509,
    "R_star_err": 0.043,
    "T_star": 7430,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 85.64,
    "v_sini_star_err": 0.13,
    "lambda_angle": -112.0,
    "lambda_angle_err": nan,
    "gamma1": 0.2385,
    "gamma2": 0.4985,
    "a_rs": 3.512,
    "b": 0.064,
    "rp_rs": 0.10696,
    "M_p": 2.093,
    "M_p_err": 0.139,
    "R_p": 1.593,
    "R_p_err": 0.054,
    "T_eq": nan,
    "Kp": 227.73,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "02h26m51.06s",
    "Dec": "+37d33m01.60s",
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "tau_derived_from": "Smith25 transit geometry",
    "geometry_source": "Smith25",
    "rotation_source": "Smith25",
    "limb_darkening_source": "Smith25 u-plus/u-minus conversion",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["WASP-18b"]["SalguneswaranNediyedath26"] = {
    "period": 0.94145252,
    "period_err": 1.1e-08,
    "epoch": 2460933.096346,
    "epoch_err": 2.2e-05,
    "duration": 0.09079166666666666,
    "duration_err": 7.083333333333332e-05,
    "tau": 0.009375,
    "tau_err": nan,
    "inclination": 84.08,
    "inclination_err": 0.17,
    "a": 0.02041,
    "eccentricity": 0.00852,
    "M_star": 1.294,
    "M_star_err": 0.063,
    "R_star": 1.23,
    "R_star_err": 0.05,
    "T_star": 6400,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 10.9,
    "v_sini_star_err": 0.7,
    "lambda_angle": 4.0,
    "lambda_angle_err": 5.0,
    "gamma1": 0.357,
    "gamma2": 0.229,
    "a_rs": 3.493,
    "b": 0.361,
    "rp_rs": 0.09757,
    "M_p": 10.2,
    "M_p_err": 0.35,
    "R_p": 1.24,
    "R_p_err": 0.079,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "01h37m25.07s",
    "Dec": "-45d40m40.06s",
    "timing_model": "linear",
    "timing_source": "SalguneswaranNediyedath26",
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "tau_derived_from": "Deline25 transit geometry",
    "geometry_source": "Deline25",
    "omega": 261.9,
    "rotation_source": "Deline25",
    "limb_darkening_source": "Deline25",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "SalguneswaranNediyedath26",
}

PLANETS["WASP-189b"]["Yumoto26"] = {
    "period": 2.72403141,
    "period_err": 3.5e-07,
    "epoch": 2456706.45616,
    "epoch_err": 0.00031,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.18056666666666665,
    "duration_err": 0.00023333333333333333,
    "tau": 0.015504166666666666,
    "tau_err": nan,
    "inclination": 84.03,
    "inclination_err": 0.14,
    "a": 0.05053,
    "eccentricity": 0.0,
    "M_star": 2.03,
    "M_star_err": 0.066,
    "R_star": 2.36,
    "R_star_err": 0.03,
    "T_star": 8000,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 95.05,
    "v_sini_star_err": 0.55,
    "lambda_angle": 90.07,
    "lambda_angle_err": 0.24,
    "gamma1": 0.414,
    "gamma2": 0.155,
    "a_rs": 4.6,
    "b": 0.478,
    "rp_rs": 0.07045,
    "M_p": 1.99,
    "M_p_err": 0.16,
    "R_p": 1.619,
    "R_p_err": 0.021,
    "T_eq": nan,
    "Kp": 200.71,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "15h02m44.82s",
    "Dec": "-03d01m53.35s",
    "timing_model": "linear",
    "timing_source": "Yumoto26",
    "tau_derived_from": "Lendl20 transit geometry",
    "geometry_source": "Lendl20; lambda from Prinoth24",
    "rotation_source": "Prinoth24",
    "limb_darkening_source": "Prinoth24",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Yumoto26",
}

PLANETS["MASCARA-1b"]["Yumoto26"] = {
    "period": 2.14876998,
    "period_err": 1.3e-07,
    "epoch": 2458833.488125,
    "epoch_err": 7.9e-05,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.17608333333333334,
    "duration_err": 0.0004375,
    "tau": 0.01325,
    "tau_err": nan,
    "inclination": 88.45,
    "inclination_err": 0.17,
    "a": 0.040352,
    "eccentricity": 0.00034,
    "M_star": 1.9,
    "M_star_err": 0.068,
    "R_star": 2.082,
    "R_star_err": 0.038,
    "T_star": 7554,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 101.7,
    "v_sini_star_err": 3.85,
    "lambda_angle": -69.2,
    "lambda_angle_err": 3.25,
    "gamma1": 0.3918,
    "gamma2": 0.0919,
    "a_rs": 4.1676,
    "b": 0.113,
    "rp_rs": 0.07884,
    "M_p": 3.7,
    "M_p_err": 0.9,
    "R_p": 1.597,
    "R_p_err": 0.037,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "21h10m12.37s",
    "Dec": "+10d44m20.03s",
    "timing_model": "linear",
    "timing_source": "Yumoto26",
    "tau_derived_from": "Hooton22 transit geometry",
    "geometry_source": "Hooton22",
    "omega": -16.0,
    "rotation_source": "Hooton22",
    "limb_darkening_source": "Hooton22 q1/q2 conversion",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Yumoto26",
}

PLANETS["TOI-1431b"]["Kokori25"] = {
    "period": 2.65023153,
    "period_err": 3.8e-07,
    "epoch": 2459558.098917,
    "epoch_err": 6.8e-05,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.10375000000000001,
    "duration_err": nan,
    "tau": 0.02975,
    "tau_err": nan,
    "inclination": 80.13,
    "inclination_err": 0.13,
    "a": 0.046,
    "eccentricity": 0.0022,
    "M_star": 1.9,
    "M_star_err": 0.1,
    "R_star": 1.92,
    "R_star_err": 0.07,
    "T_star": 7690,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 6.0,
    "v_sini_star_err": 0.2,
    "lambda_angle": -155.0,
    "lambda_angle_err": 15.0,
    "gamma1": 0.13,
    "gamma2": 0.33,
    "a_rs": 5.15,
    "b": 0.881,
    "rp_rs": 0.07955,
    "M_p": 3.12,
    "M_p_err": 0.18,
    "R_p": 1.49,
    "R_p_err": 0.05,
    "T_eq": nan,
    "Kp": 186.03,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "Ks_expected": 294.1,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "21h04m48.89s",
    "Dec": "+55d35m16.88s",
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "tau_derived_from": "Addison21 transit geometry",
    "geometry_source": "Addison21; lambda from Stangret21",
    "omega": 108.0,
    "rotation_source": "Addison21",
    "limb_darkening_source": "Addison21",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["TOI-1518b"]["Kokori25"] = {
    "period": 1.90261185,
    "period_err": 2.4e-07,
    "epoch": 2459616.588037,
    "epoch_err": 5.3e-05,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.09979166666666667,
    "duration_err": 0.00033541666666666664,
    "tau": 0.03563333333333333,
    "tau_err": nan,
    "inclination": 77.626,
    "inclination_err": 0.097,
    "a": 0.03712,
    "eccentricity": 0.0,
    "M_star": 1.79,
    "M_star_err": 0.26,
    "R_star": 1.95,
    "R_star_err": 0.08,
    "T_star": 7300,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "v_sini_star": 74.4,
    "v_sini_star_err": 2.3,
    "lambda_angle": -113.56,
    "lambda_angle_err": 0.54,
    "gamma1": 0.4426,
    "gamma2": 0.2736,
    "a_rs": 4.109,
    "b": 0.8806,
    "rp_rs": 0.09939,
    "M_p": 2.3,
    "M_p_err": 2.3,
    "R_p": 1.875,
    "R_p_err": 0.053,
    "T_eq": nan,
    "Kp": 217.26,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "23h29m04.20s",
    "Dec": "+67d02m05.30s",
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "tau_derived_from": "Simonnin25 transit geometry",
    "geometry_source": "Simonnin25; lambda from PIRANGA III",
    "rotation_source": "PIRANGA III",
    "limb_darkening_source": "PIRANGA III q1/q2 conversion",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["HAT-P-11b"]["Basilicata24"] = {
    "timing_model": "linear",
    "timing_source": "Basilicata24",
    "period": 4.887802443,
    "period_err": 3.4e-08,
    "epoch": 2454957.8132067,
    "epoch_err": 5.3e-06,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.09815083333333334,
    "duration_err": nan,
    "tau": 0.005750000000000001,
    "tau_err": nan,
    "tau_derived_from": "Basilicata24 eccentric transit geometry",
    "geometry_source": "Basilicata24",
    "inclination": 89.027,
    "inclination_err": 0.068,
    "a": 0.05254,
    "a_rs": 15.05,
    "b": 0.227,
    "rp_rs": 0.058993,
    "eccentricity": 0.2577,
    "omega": 192.0,
    "lambda_angle": 103.0,
    "lambda_angle_err": 18.0,
    "M_star": 0.809,
    "M_star_err": 0.02,
    "R_star": 0.683,
    "R_star_err": 0.009,
    "T_star": 4780,
    "logg_star": 4.59,
    "logg_star_err": nan,
    "Fe_H": 0.31,
    "rotation_source": "Basilicata24",
    "v_sini_star": 0.67,
    "v_sini_star_err": 0.095,
    "limb_darkening_source": "Basilicata24 q1/q2 conversion",
    "gamma1": 0.656,
    "gamma2": 0.0255,
    "M_p": 0.0736,
    "M_p_err": nan,
    "R_p": 0.389,
    "R_p_err": nan,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "19h50m50.44s",
    "Dec": "+48d04m54.71s",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Basilicata24",
}

PLANETS["KELT-5b"]["ExoFOP"] = {
    "timing_model": "linear",
    "timing_source": "ExoFOP",
    "period": 3.019780698,
    "period_err": 0.0001354653,
    "epoch": 2459853.583682,
    "epoch_err": 0.00070863,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.14957,
    "duration_err": 0.0017195833333333334,
    "tau": nan,
    "tau_err": nan,
    "geometry_source": "unavailable",
    "inclination": nan,
    "inclination_err": nan,
    "a": nan,
    "a_rs": nan,
    "b": nan,
    "rp_rs": 0.10962,
    "rp_rs_derived_from": "ExoFOP transit depth",
    "eccentricity": nan,
    "lambda_angle": nan,
    "lambda_angle_err": nan,
    "M_star": nan,
    "M_star_err": nan,
    "R_star": nan,
    "R_star_err": nan,
    "T_star": 7128,
    "logg_star": 4.24,
    "logg_star_err": nan,
    "Fe_H": 0.0,
    "v_sini_star": nan,
    "v_sini_star_err": nan,
    "gamma1": nan,
    "gamma2": nan,
    "M_p": nan,
    "M_p_err": nan,
    "R_p": nan,
    "R_p_err": nan,
    "T_eq": nan,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "23h26m19.13s",
    "Dec": "+38d33m10.71s",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "ExoFOP",
}

PLANETS["TOI-1789b"]["Kokori25"] = {
    "timing_model": "linear",
    "timing_source": "Kokori25",
    "period": 3.2087107,
    "period_err": 3.5e-06,
    "epoch": 2459470.47469,
    "epoch_err": 0.00031,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "duration": 0.09541666666666666,
    "duration_err": nan,
    "tau": nan,
    "tau_err": nan,
    "grazing_transit": True,
    "geometry_source": "Khandelwal21/22",
    "inclination": 78.41,
    "inclination_err": 0.47,
    "a": 0.04882,
    "a_rs": 4.83,
    "b": 0.972,
    "rp_rs": 0.0661,
    "eccentricity": 0.0,
    "lambda_angle": nan,
    "lambda_angle_err": nan,
    "M_star": 1.507,
    "M_star_err": 0.059,
    "R_star": 2.168,
    "R_star_err": 0.036,
    "T_star": 5991,
    "logg_star": 3.943,
    "logg_star_err": nan,
    "Fe_H": 0.373,
    "rotation_source": "Khandelwal21/22",
    "v_sini_star": 7.0,
    "v_sini_star_err": 0.5,
    "limb_darkening_source": "Khandelwal21/22",
    "gamma1": 0.263,
    "gamma2": 0.284,
    "M_p": 0.7,
    "M_p_err": nan,
    "R_p": 1.44,
    "R_p_err": nan,
    "T_eq": 1927.0,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "09h30m58.42s",
    "Dec": "+26d32m23.98s",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Kokori25",
}

PLANETS["V1298 Tau b"]["Livingston26TTV"] = {
    "timing_model": "ttv_table",
    "timing_source": "Livingston26TTV",
    "period": 24.140006,
    "period_err": 1.7e-05,
    "epoch": 2458298.209601,
    "epoch_err": 0.000585,
    "epoch_scale": "tdb",
    "epoch_reference": "barycenter",
    "transit_midpoints_bjd_tdb": {"20191024": 2458781.083656},
    "duration": 0.26875,
    "duration_err": 0.0022916666666666667,
    "tau": 0.025774854166666666,
    "tau_err": nan,
    "tau_derived_from": "Johnson22 a_rs/b/inclination/rp_rs, circular contact formula",
    "geometry_source": "Johnson22",
    "inclination": 88.759,
    "inclination_err": 0.077,
    "a": 0.1685,
    "a_rs": 26.06,
    "b": 0.564,
    "rp_rs": 0.071998,
    # Adopt a circular orbit for spectral trail and Doppler-shadow work.
    # Preserve the previous small-e value because omega is unavailable.
    "eccentricity": 0.0,
    "eccentricity_reported": 0.0079,
    "eccentricity_model": "adopted_circular",
    "eccentricity_adoption_reason": (
        "e=0.0079 is negligible for this spectral analysis and omega is unavailable"
    ),
    "lambda_angle": 4.0,
    "lambda_angle_err": 8.5,
    "M_star": 1.1,
    "M_star_err": 0.05,
    "R_star": 1.32,
    "R_star_err": 0.05,
    "T_star": 4970,
    "logg_star": nan,
    "logg_star_err": nan,
    "Fe_H": nan,
    "rotation_source": "Johnson22",
    "v_sini_star": 24.77,
    "v_sini_star_err": 0.19,
    "limb_darkening_source": "Johnson22",
    "gamma1": 0.54,
    "gamma2": -0.03,
    "M_p": 0.0412,
    "M_p_err": nan,
    "R_p": 0.84,
    "R_p_err": nan,
    "T_eq": 666.0,
    "Kp": nan,
    "Kp_err": nan,
    "RV_abs": nan,
    "RV_abs_err": nan,
    "kappa_IR": nan,
    "gamma": nan,
    "P0": nan,
    "X_H2": nan,
    "X_He": nan,
    "VMR_H_minus": nan,
    "RA": "04h05m19.60s",
    "Dec": "+20d09m25.31s",
    "profile_kind": "source_ephemeris",
    "ephemeris_source": "Livingston26TTV",
}

for _planet_profiles in PLANETS.values():
    if "Recommended" in _planet_profiles:
        _planet_profiles["Recommended"]["profile_kind"] = "recommended_composite"
del _planet_profiles

# ==============================================================================
# INSTRUMENT CONFIGURATION
# ==============================================================================

# ==============================================================================
# ACTIVE SELECTION (global state, can be modified at runtime)
# ==============================================================================

OBSERVATORY = "lbt"
INSTRUMENT = "PEPSI"
OBSERVING_MODE = "full"
RESOLUTION_MODE = "hr"  # Options: "standard" (R=50k), "hr" (R=130k), "uhr" (R=270k)


# ==============================================================================
# INSTRUMENT DATABASE
# ==============================================================================

# Common PEPSI header keys (shared across all modes)
_PEPSI_HEADER_KEYS = {
    "jd": "JD-OBS",          # UTC mid-exposure Julian Date
    "bjd_tdb": "JD-TDB",     # Optional barycentric TDB validation value
    "timesys": "TIMESYS",    # Optional declared JD-OBS timescale
    "snr": "SNR",            # Signal-to-noise ratio
    "exptime": "EXPTIME",    # Exposure time
    "airmass": "AIRMASS",    # Airmass
    "radvel": "RADVEL",      # Radial velocity correction
    "obsvel": "OBSVEL",      # Observatory velocity
    "ssbvel": "SSBVEL",      # Solar system barycenter velocity
}

# Common PEPSI FITS columns (shared across all modes)
_PEPSI_FITS_COLUMNS = {
    "molecfit": {
        "wave": "lambda",
        "flux": "flux",
        "error": "error",
        "wave_unit": "micron",  # molecfit outputs in microns
    },
    "raw": {
        "wave": "Arg",
        "flux": "Fun",
        "error": "Var",  # Note: this is variance, needs sqrt
        "wave_unit": "angstrom",
    },
}


# TODO: maybe hardcoding the telluric regions is a bad idea
# ==============================================================================
# TELLURIC REGIONS (wavelength ranges in Angstroms)
# ==============================================================================

# From Lenhart et al. 2026 Table 2 (PEPSI observations)
TELLURIC_REGIONS: dict[str, dict[str, list[tuple[float, float]]]] = {
    "red": {
        # Regions with >1% line depth in adjacent telluric lines
        "telluric": [
            (6278, 6328),   # O2 B-band wing
            (6459, 6527),   # H2O
            (6867, 6867.5), # O2 B-band edge
            (6930, 7168),   # H2O + O2 A-band
            (7312, 7500),   # H2O
        ],
        # Deep absorption - mask if molecfit was used (set flux=0, err=1)
        "deep_mask": [
            (6867.5, 6930),  # O2 B-band core
            (7168, 7312),    # Deep H2O
        ],
    },
    "blue": {
        # Blue arm lacks significant tellurics (Smette et al. 2015)
        "telluric": [],
        "deep_mask": [],
    },
}


INSTRUMENTS: dict[str, dict[str, dict]] = {
    "lbt": {
        "PEPSI": {
            "resolution": 130000,  # Default to HR mode
            "resolution_modes": {
                "standard": 50000,   # 300 µm fiber
                "hr": 130000,        # 200 µm fiber (High Resolution)
                "uhr": 270000,       # 100 µm fiber (Ultra-High Resolution)
            },
            "header_keys": _PEPSI_HEADER_KEYS,
            "fits_columns": _PEPSI_FITS_COLUMNS,
            "data_pattern_family": "pepsi",
            "modes": {
                "blue": {
                    "range": (4752, 5425),
                    "file_prefix": "pepsib",
                },
                "red": {
                    "range": (6231, 7427),
                    "file_prefix": "pepsir",
                },
                "green": {
                    "range": (4760, 6570),  # CD3+CD4 approximate
                    "file_prefix": "pepsig",
                },
                "full": {
                    "range": (4752, 7427),  # Both arms combined
                    "file_prefix": None,    # No single file prefix for combined
                },
            },
        },
    },
}

# ==============================================================================
# RADIATIVE TRANSFER MODEL PARAMETERS
# ==============================================================================

# ==============================================================================
# RETRIEVAL MODE
# ==============================================================================

RETRIEVAL_MODE = "transmission"  # Options: "transmission", "emission"

# Mode-specific default P-T profiles
TRANSMISSION_PT_PROFILE_DEFAULT = "isothermal"
EMISSION_PT_PROFILE_DEFAULT = "guillot"

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 10 # number of atmospheric layers (runtime profiles below override)

# Mode-specific pressure ranges [bar]
TRANSMISSION_PRESSURE_TOP = 1e-8
TRANSMISSION_PRESSURE_BTM = 1e0
# Adopt the catalog-informed transmission reference radius at the RT lower
# boundary.  This is an explicit modeling convention, not a direct
# observational determination of R_1bar.  Keep this tied to the configured
# transmission boundary so changing the grid cannot silently change the
# radius-pressure meaning.
TRANSMISSION_REFERENCE_PRESSURE_BAR = 1.0
# Cover the upper line-forming atmosphere and deep continuum anchoring used by
# the KELT-20b emission analysis. This is the default for all emission runs;
# transmission retains its independent pressure domain above.
EMISSION_PRESSURE_TOP = 1e-6
EMISSION_PRESSURE_BTM = 1e2

# Temperature range [K]
# Sets the common supported domain for PreMODIT, ART, and FastChem. Profiles
# outside this interval are rejected; clipping is only used to keep JAX's
# evaluation of a rejected proposal numerically safe.
# [1500, 5500] gives a PreModit robust range of 1451.74 - 5825.62 K (dE=875, Tref, Twt
# chosen by the LUT), covering Guillot upper-atmosphere draws (observed up to ~5472 K
# under the current priors) while only increasing the LBD+xsmatrix scratch tensor by
# ~14% over the historical [1500, 4500] setting. Widening the cold edge below 1500 K
# requires a smaller dE and runs past the 10 GB GPU budget.
T_LOW = 1500.0
T_HIGH = 5500.0

# Guillot profile defaults and bounds
TINT_FIXED = 100.0
# LOG_KAPPA_IR_BOUNDS: log10(kappa_IR [cm^2/g]). kappa_IR is the Rosseland-mean
# IR opacity of the atmosphere. Hot-Jupiter retrieval literature (Guillot 2010,
# Line et al. 2013, Molliere et al. 2015) places this in 1e-3 - 1e-1 cm^2/g for
# solar-composition atmospheres. The previous (-4, 0) range extended four orders
# of magnitude wider than physical on both ends, contributing to the
# upper-atmosphere temperature runaway (tau = kappa_IR * P / g).
LOG_KAPPA_IR_BOUNDS = (-3.0, -1.0)
# LOG_GAMMA_BOUNDS: log10(gamma) where gamma = kappa_V / kappa_IR. Physically
# plausible hot-Jupiter values span roughly 0.1 - 3 (Guillot 2010 Fig. 4,
# Fortney et al. 2008, Line+ 2013), with gamma > 10 indicating an extreme
# stratospheric absorber. The previous (0, 2) range allowed gamma up to 100,
# which at Tirr = 5500 K drives Guillot's top-of-atmosphere T beyond 14,000 K
# (T_top^4 ~ (3/4) Tirr^4 gamma/sqrt(3)) - well outside PreModit's robust range
# and FastChem's tabulated grid, producing NaN cross sections / VMRs and a
# NaN logL. (-1, 1) covers the physical range while keeping the bulk of prior
# mass inside [T_LOW, T_HIGH]; the model-validity check rejects the gamma > 3
# tail where Guillot would overshoot.
LOG_GAMMA_BOUNDS = (-1.0, 1.0)

DEFAULT_KP = 169.0  # planet radial velocity semi-amplitude [km/s]
DEFAULT_KP_ERR = 20.0 
DEFAULT_RV_ABS = 0.0 # absolute stellar systemic velocity metadata [km/s]; not a model shift
DEFAULT_RV_ABS_ERR = 1.0 # metadata uncertainty [km/s]
DEFAULT_TSTAR = 6000.0 # stellar temperature [K] 
DEFAULT_RP_ERR = 0.1 # planet radius error (relative to Rp/Rs)
DEFAULT_MP_ERR = 0.1 # planet mass error (relative to Mp/Ms)
DEFAULT_RSTAR_ERR = 0.1 # stellar radius error (relative to Rstar)

# Posterior reconstruction defaults
DEFAULT_POSTERIOR_RP = 1.5 # Maximum Rp/Rs for posterior reconstruction
DEFAULT_POSTERIOR_MP = 1.0 # Maximum Mp/Ms for posterior reconstruction

# Pipeline behavior defaults
APPLY_SYSREM_DEFAULT = True # Apply the stored SYSREM distortion operator to the model before the Gaussian likelihood. Requires U and V from data preprocessing.

# ==============================================================================
# SPECTRAL GRID PARAMETERS
# ==============================================================================

N_SPECTRAL_POINTS = 500000
#N_SPECTRAL_POINTS = 50000
WAV_MIN_OFFSET = 100  # Angstroms
WAV_MAX_OFFSET = 100  # Angstroms

# preMODIT parameters
# Line-wing truncation (relative to grid spacing). Set to None to use the default.
PREMODIT_CUTWING = None

# ==============================================================================
# CLOUD/HAZE PARAMETERS
# ==============================================================================

CLOUD_WIDTH = 1.0 / 20.0  # Cloud width in log10(P)
CLOUD_INTEGRATED_TAU = 30.0 

# ==============================================================================
# DATABASE, DATA, AND OUTPUT PATHS
# ==============================================================================

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================

# Root of the project.
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input"
INPUT_DIR.mkdir(exist_ok=True)

REFERENCE_DIR = PROJECT_ROOT / "reference"
REFERENCE_DIR.mkdir(exist_ok=True)

REFERENCE_BANDPASS_DIR = REFERENCE_DIR / "bandpasses"
REFERENCE_BANDPASS_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_ABUNDANCE_DIR = REFERENCE_DIR / "abundances"
REFERENCE_ABUNDANCE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PHOENIX_CACHE_DIR = Path(os.environ.get("PHOENIX_CACHE_DIR") or CACHE_DIR / "phoenix")
PHOENIX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPA_CACHE_DIR = CACHE_DIR / "opacity"
OPA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DATABASE PATHS
# ==============================================================================

DB_ROOT_DIR = PROJECT_ROOT / "db"
DB_ROOT_DIR.mkdir(exist_ok=True)

# Molecular databases (override with env vars if set)
DB_HITEMP = Path(os.environ.get("HITEMP_DIR") or DB_ROOT_DIR / "hitemp")
DB_EXOMOL = Path(os.environ.get("EXOMOL_DIR") or DB_ROOT_DIR / "exomol")
DB_EXOATOM = Path(os.environ.get("EXOATOM_DIR") or DB_ROOT_DIR / "exoatom")
DB_KURUCZ = Path(os.environ.get("KURUCZ_DIR") or DB_ROOT_DIR / "kurucz")
DB_VALD = Path(os.environ.get("VALD_DIR") or DB_ROOT_DIR / "vald")
DB_CIA = Path(os.environ.get("CIA_DIR") or DB_ROOT_DIR / "cia")

for db_dir in (DB_HITEMP, DB_EXOMOL, DB_EXOATOM, DB_KURUCZ, DB_VALD, DB_CIA):
    db_dir.mkdir(parents=True, exist_ok=True)

# CIA paths
CIA_PATHS = {
    "H2H2": DB_CIA / "H2-H2_2011.cia",
    "H2He": DB_CIA / "H2-He_2011.cia",
}

# Molecular line lists (HITEMP)
MOLPATH_HITEMP = {
    "H2O": DB_HITEMP / "H2O",
    "CO": DB_HITEMP / "CO",
    "CO2": DB_HITEMP / "CO2",
    "OH": DB_HITEMP / "OH",
    "NO": DB_HITEMP / "NO",
}

# Molecular line lists (ExoMol)
MOLPATH_EXOMOL = {
    "CH4": DB_EXOMOL / "CH4/12C-1H4/10to10",
    "NH3": DB_EXOMOL / "NH3/14N-1H3/CoYuTe",
    "HCN": DB_EXOMOL / "HCN/1H-12C-14N/Harris",
    "C2H2": DB_EXOMOL / "C2H2/12C2-1H2/aCeTY",
    "C2H4": DB_EXOMOL / "C2H4/12C2-1H4/MaYTY",
    "H2S": DB_EXOMOL / "H2S/1H2-32S/AYT2",
    "SO": DB_EXOMOL / "SO/32S-16O/ExoMol",
    "SO2": DB_EXOMOL / "SO2/32S-16O2/ExoAmes",
    "SiO": DB_EXOMOL / "SiO/28Si-16O/SiOUVenIR",
    "TiH": DB_EXOMOL / "TiH/48Ti-1H/TiH",
    "MgH": DB_EXOMOL / "MgH/24Mg-1H/MgH",
    "AlH": DB_EXOMOL / "AlH/27Al-1H/AlH",
    "SiH": DB_EXOMOL / "SiH/28Si-1H/SiH",
    "NaH": DB_EXOMOL / "NaH/23Na-1H/NaH",
    "KH": DB_EXOMOL / "KH/39K-1H/KH",
    "TiO": DB_EXOMOL / "TiO/48Ti-16O/Toto",
    "VO": DB_EXOMOL / "VO/51V-16O/VOMYT",
    "FeH": DB_EXOMOL / "FeH/56Fe-1H/MoLLIST",
    "CaH": DB_EXOMOL / "CaH/40Ca-1H/XAB",
    "CrH": DB_EXOMOL / "CrH/52Cr-1H/MoLLIST",
    "AlO": DB_EXOMOL / "AlO/27Al-16O/ATP",
}

# Atomic line lists (Kurucz/VALD)
# Format: "Element_I" for neutral, "Element_II" for singly ionized
# Key names match spectroscopic notation (e.g., "Fe I", "Fe II")
ATOMIC_SPECIES = {
    # Neutral atoms (ionization = 0)
    "Al I": {"element": "Al", "ionization": 0},
    "B I": {"element": "B", "ionization": 0},
    "Ba I": {"element": "Ba", "ionization": 0},
    "Be I": {"element": "Be", "ionization": 0},
    "Ca I": {"element": "Ca", "ionization": 0},
    "Co I": {"element": "Co", "ionization": 0},
    "Cr I": {"element": "Cr", "ionization": 0},
    "Cs I": {"element": "Cs", "ionization": 0},
    "Cu I": {"element": "Cu", "ionization": 0},
    "Fe I": {"element": "Fe", "ionization": 0},
    "Ga I": {"element": "Ga", "ionization": 0},
    "Ge I": {"element": "Ge", "ionization": 0},
    "Hf I": {"element": "Hf", "ionization": 0},
    "In I": {"element": "In", "ionization": 0},
    "Ir I": {"element": "Ir", "ionization": 0},
    "K I": {"element": "K", "ionization": 0},
    "Li I": {"element": "Li", "ionization": 0},
    "Mg I": {"element": "Mg", "ionization": 0},
    "Mn I": {"element": "Mn", "ionization": 0},
    "Mo I": {"element": "Mo", "ionization": 0},
    "Na I": {"element": "Na", "ionization": 0},
    "Nb I": {"element": "Nb", "ionization": 0},
    "Ni I": {"element": "Ni", "ionization": 0},
    "Os I": {"element": "Os", "ionization": 0},
    "Pb I": {"element": "Pb", "ionization": 0},
    "Pd I": {"element": "Pd", "ionization": 0},
    "Rb I": {"element": "Rb", "ionization": 0},
    "Rh I": {"element": "Rh", "ionization": 0},
    "Ru I": {"element": "Ru", "ionization": 0},
    "Sc I": {"element": "Sc", "ionization": 0},
    "Si I": {"element": "Si", "ionization": 0},
    "Sn I": {"element": "Sn", "ionization": 0},
    "Sr I": {"element": "Sr", "ionization": 0},
    "Ti I": {"element": "Ti", "ionization": 0},
    "Tl I": {"element": "Tl", "ionization": 0},
    "V I": {"element": "V", "ionization": 0},
    "W I": {"element": "W", "ionization": 0},
    "Y I": {"element": "Y", "ionization": 0},
    "Zn I": {"element": "Zn", "ionization": 0},
    "Zr I": {"element": "Zr", "ionization": 0},
    # Singly ionized atoms (ionization = 1)
    "Ba II": {"element": "Ba", "ionization": 1},
    "Ca II": {"element": "Ca", "ionization": 1},
    "Cr II": {"element": "Cr", "ionization": 1},
    "Fe II": {"element": "Fe", "ionization": 1},
    "Mg II": {"element": "Mg", "ionization": 1},
    "Sc II": {"element": "Sc", "ionization": 1},
    "Sr II": {"element": "Sr", "ionization": 1},
    "Ti II": {"element": "Ti", "ionization": 1},
    "Y II": {"element": "Y", "ionization": 1},
}

# ==============================================================================
# DATA PATHS
# ==============================================================================

FULL_ARM_MEMBERS: tuple[str, ...] = ("red", "blue")

# Explicit exceptions to the usual dual-arm PEPSI observation contract.
# Keys are (retrieval mode, planet, observing epoch); unlisted observations
# use both members of FULL_ARM_MEMBERS.
HRS_OBSERVATION_ARMS: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("transmission", "KELT-5b", "20171015"): ("blue",),
    ("transmission", "TOI-1789b", "20240312"): ("red",),
    ("transmission", "V1298 Tau b", "20191024"): ("blue",),
}

_DEFAULT_PLANET_SLUG = PLANET.strip().lower().replace("-", "").replace(" ", "")

RAW_HRS_DIR = INPUT_DIR / "hrs" / "transmission" / "raw" / _DEFAULT_PLANET_SLUG
LOWRES_DIR = INPUT_DIR / "lrs" / RETRIEVAL_MODE / _DEFAULT_PLANET_SLUG
PHOT_DIR = INPUT_DIR / "phot" / RETRIEVAL_MODE / _DEFAULT_PLANET_SLUG

if OBSERVING_MODE == "full":
    DATA_DIR = None
    TRANSMISSION_DATA = None
    EMISSION_DATA = None
else:
    DATA_DIR = INPUT_DIR / "hrs" / RETRIEVAL_MODE / _DEFAULT_PLANET_SLUG / OBSERVING_MODE
    TRANSMISSION_DATA = {
        "wavelength": DATA_DIR / "wavelength_transmission.npy",
        "spectrum": DATA_DIR / "spectrum_transmission.npy",
        "uncertainty": DATA_DIR / "uncertainty_transmission.npy",
    }
    EMISSION_DATA = {
        "wavelength": DATA_DIR / "wavelength_emission.npy",
        "spectrum": DATA_DIR / "spectrum_emission.npy",
        "uncertainty": DATA_DIR / "uncertainty_emission.npy",
    }

del _DEFAULT_PLANET_SLUG

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Default output directory for phase-binned runs
DEFAULT_PHASE_BINNED_OUTPUT_DIR = PROJECT_ROOT / "output" / "phase_binned"

# Default output directory (lazy - will be set by CLI or on first use)
DIR_SAVE = None  # Set by CLI via get_output_dir()

# Opacity loading/saving
OPA_LOAD = True
OPA_SAVE = False

# Atomic database preferences
# Kurucz: auto-downloaded from kurucz.harvard.edu
# VALD: requires manual download from vald.astro.uu.se (place in db/vald/)
USE_KURUCZ = True
USE_VALD = True

# ==============================================================================
# DEFAULT SPECIES SELECTION
# ==============================================================================
# Species detected in literature (used by default unless --all-species is set)
# Based on high-resolution detections from multiple instruments (PEPSI, HARPS-N,
# CARMENES, EXPRES, FIES) - see literature compilation table.
#
# To use all available species instead of this subset, pass --all-species to CLI.
# To override with a custom list, use --atoms "Fe I,Na I" or --molecules "H2O,CO".

DEFAULT_SPECIES = {
    "atoms": [
        "Na I",   # Detected in multiple studies (PEPSI, HARPS-N, CARMENES, EXPRES)
        "Mg I",   # Detected (EXPRES)
        "Ca II",  # Detected (CARMENES, HARPS-N)
        "Cr I",   # Detected (PEPSI)
        "Cr II",  # Detected (EXPRES)
        "Fe I",   # Strong detection in most studies
        "Fe II",  # Strong detection in most studies
    ],
    "molecules": [
        "FeH",    # Detected (CARMENES)
    ],
}

# Set to True to use DEFAULT_SPECIES by default, False to use all available species
USE_DEFAULT_SPECIES = True

# ==============================================================================
# INFERENCE PARAMETERS
# ==============================================================================

# ==============================================================================
# SVI PARAMETERS
# ==============================================================================

SVI_NUM_STEPS = 2000
SVI_LEARNING_RATE = 0.001
SVI_LR_DECAY_STEPS = None
SVI_LR_DECAY_RATE = None

# ==============================================================================
# MCMC PARAMETERS
# ==============================================================================

MCMC_NUM_WARMUP = 2000
MCMC_NUM_SAMPLES = 2000
MCMC_MAX_TREE_DEPTH = 10

# Parallel chains
MCMC_NUM_CHAINS = 2
MCMC_CHAIN_METHOD = "parallel"
MCMC_REQUIRE_GPU_PER_CHAIN = False
# TODO: if MCMC_NUM_CHAINS = 4 w/o parallel gpus then 4 chains will run sequentially, which is not ideal. see how t correctly make this code run in parallel on GPUs, then change this parameter before a run



# ==============================================================================
# INFERENCE BEHAVIOR DEFAULTS
# ==============================================================================

INIT_TO_MEDIAN_SAMPLES = 100

# Quick mode defaults
QUICK_SVI_STEPS = 100
QUICK_MCMC_WARMUP = 100
QUICK_MCMC_SAMPLES = 100
QUICK_MCMC_CHAINS = 1

# ==============================================================================
# CHEMISTRY PARAMETERS
# ==============================================================================

# ==============================================================================
# VMR PRIOR RANGES
# ==============================================================================

# Logarithmic VMR prior bounds for trace species
LOG_VMR_MIN = -12.0  # Minimum log10(VMR)
LOG_VMR_MAX = -2.0   # Maximum log10(VMR)

# ==============================================================================
# BULK COMPOSITION
# ==============================================================================

# H2/He number ratio (solar ~10-11, hot Jupiters often use ~6-7)
H2_HE_RATIO = 6.0

# ==============================================================================
# FREE CHEMISTRY PROFILE PARAMETERIZATION
# ==============================================================================

# Number of nodes for altitude-dependent VMR profiles
N_VMR_NODES = 5

# ==============================================================================
# EQUILIBRIUM CHEMISTRY
# ==============================================================================

# Metallicity [M/H] prior range (log10 relative to solar)
METALLICITY_RANGE = (-2.0, 3.0)

# C/O ratio prior range (solar ~ 0.55)
CO_RATIO_RANGE = (0.1, 2.0)

# Solar elemental abundance table (Asplund 2020; log epsilon format)
SOLAR_ABUNDANCE_FILE = "reference/abundances/asplund_2020_extended.dat"

# ==============================================================================
# FASTCHEM GRID PARAMETERS
# ==============================================================================

FASTCHEM_N_TEMP = 50
FASTCHEM_N_PRESSURE = 50
FASTCHEM_T_MIN = T_LOW
FASTCHEM_T_MAX = T_HIGH
FASTCHEM_CACHE_DIR = "cache/fastchem"
FASTCHEM_DATA_DIR = None  # None = use pyfastchem defaults
FASTCHEM_PARAMETER_FILE = None  # Path to FastChem parameters.dat

# Chemistry solver selection
CHEMISTRY_MODEL_DEFAULT = "constant"

# Hybrid FastChem grid settings (NUTS-safe via JAX interpolation)
FASTCHEM_HYBRID_CONTINUUM_SPECIES = ("H-", "e-", "H")
FASTCHEM_HYBRID_N_METALLICITY = 17
FASTCHEM_HYBRID_N_CO_RATIO = 17
FASTCHEM_HYBRID_METALLICITY_RANGE = METALLICITY_RANGE
FASTCHEM_HYBRID_CO_RATIO_RANGE = CO_RATIO_RANGE

# ==============================================================================
# DISEQUILIBRIUM CHEMISTRY
# ==============================================================================

# Eddy diffusion coefficient Kzz [cm^2/s] prior range
LOG_KZZ_RANGE = (6.0, 12.0)

# Quench pressure range [bar]
LOG_QUENCH_P_RANGE = (-6.0, 2.0)

# ==============================================================================
# NUMERICAL GUARD CONSTANTS
# ==============================================================================

# Machine epsilon for float32 around 1.0. Used for relative comparisons/tolerances.
F32_EPS = float(np.finfo(np.float32).eps)
F32_FLOOR_RECIP = 1.0e-30 # Safe floor for linear reciprocals in float32 code.
F32_FLOOR_RECIPSQ = 1.0e-18 # Larger floor for expressions that square the reciprocal, e.g. 1 / sigma^2.
F32_GRAVITY_FLOOR = 1.0e-20 # Safe floor for gravity-like denominators in P-T profile code.
F32_LENGTHSCALE_FLOOR = 1.0e-12 # Safe floor for GP lengthscales in standardized coordinate units.
F32_STDDEV_FLOOR = 1.0e-12 # Small stabilizer for standard deviation-like scale terms.
F64_FLOOR = 1.0e-300 # float64 underflow guard.
TRACE_SPECIES_FLOOR = 1.0e-30 # Semantic floor for absent/trace chemistry species profiles.
# These are numerical limits of the current ExoJAX atmospheric geometry, not
# claims about the physically possible MMW of a gas. Fully ionized gas can
# have MMW below one.
MMW_RT_MIN = 1.0
MMW_RT_MAX = 50.0
# PreMODIT evaluates in float32. Accept at most one float32 epsilon relative to
# the same term's positive peak; widening this guard requires an empirical
# audit of the actual opacity databases rather than an arbitrary guess.
PREMODIT_NEGATIVE_ROUNDOFF_RTOL = F32_EPS

# ==============================================================================
# DATA PREPARATION DEFAULTS
# ==============================================================================

# Data prep defaults
DEFAULT_DATA_PLANET = PLANET
DEFAULT_DATA_ARM = OBSERVING_MODE
DEFAULT_USE_MOLECFIT = True
DEFAULT_RAW_DATA_DIR = "input/hrs/transmission/raw"

# Explicit exposure-level science-QC exclusions. Keys use normalized planet
# names and identify the data family, epoch, and arm. Values are stable PEPSI
# exposure IDs rather than product filenames, so the same observation is
# excluded from raw, Molecfit, calibration, SYSREM, LSD, and retrieval paths.
# Files remain on disk for provenance.
HRS_EXCLUDED_EXPOSURE_IDS = {
    ("kelt20b", "transmission", "20190504", "blue"): (
        "pepsib.20190504.038",
    ),
    # Blue-arm counterparts of the persistent red ingress-profile mismatch.
    # PEPSI exposure numbering is arm-specific, so these are the time-matched
    # companions of red exposures 041--044 rather than identically numbered.
    ("kelt20b", "transmission", "20250601", "blue"): (
        "pepsib.20250601.062",
        "pepsib.20250601.063",
        "pepsib.20250601.064",
        "pepsib.20250601.065",
    ),
    # Persistent red-arm ingress-profile mismatch. These exposures do not
    # match the post-transit LSD master and are excluded from every science
    # consumer through the shared exposure-selection contract.
    ("kelt20b", "transmission", "20250601", "red"): (
        "pepsir.20250601.041",
        "pepsir.20250601.042",
        "pepsir.20250601.043",
        "pepsir.20250601.044",
    ),
}

# Automatic exposure-level S/N QC. An exposure is removed only when its
# time-matched blue and red measurements both fall below this normalized
# header-S/N ratio. Missing/nonfinite S/N values pass this rule.
HRS_PAIRED_SNR_Q_THRESHOLD = 0.4

# Accepted FITS OBJECT labels for raw HRS science targets. Comparisons ignore
# capitalization, whitespace, and punctuation, but aliases are never inferred
# from the contents of an observing night. Every configured raw HRS target must
# have an explicit entry here before its exposures can enter science processing.
HRS_NOMINAL_FITS_OBJECT_NAMES = {
    "hatp11b": ("HAT-P-11",),
    "kelt5b": ("TYC 3230-1174-1",),
    "kelt9b": ("Kelt-9",),
    "kelt20b": ("KELT-20",),
    "mascara1b": ("Mascara-1",),
    "toi1431b": ("TOI 1431",),
    "toi1518b": ("TOI 1518",),
    "toi1789b": ("TOI 1789",),
    "v1298taub": ("V1298 Tau",),
    "wasp33b": ("WASP 33",),
    "wasp189b": ("WASP 189",),
}

# Data loading defaults
# Default to time-series input so the main CLI and phase-binned path work without
# extra flags. Use --data-format spectrum for collapsed retrieval products.
DEFAULT_DATA_FORMAT = "timeseries"

# Binning defaults
DEFAULT_BIN_SIZE = 50

# Doppler shadow fitting defaults
DEFAULT_FIT_PARAM_FALLBACK = 1.0

# Misc utility defaults
DEFAULT_BIN_INFO_COUNT = 0
DEFAULT_TRACKER_MAX_USED = 0.0

# SYSREM defaults
DEFAULT_SYSREM_MAX_SYSTEMATICS_RED = [10, 10]
DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER = [10]
DEFAULT_SYSREM_MIN_SYSTEMATICS_RED = [1, 1]
DEFAULT_SYSREM_MIN_SYSTEMATICS_OTHER = [1]
DEFAULT_SYSREM_STOP_TOL = 1e-4
DEFAULT_REGRID_MAX_NATIVE_GAP_FACTOR = 5.0
DEFAULT_TELLURIC_EDGE_MASK_WIDTH_A = 3.0

# ==============================================================================
# PHOTOMETRY DEFAULTS
# ==============================================================================

TESS_BANDPASS_URL = "https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv"
TESS_BANDPASS_PATH = REFERENCE_BANDPASS_DIR / "tess-response-function-v2.0.csv"

# Explicit starting values for the supported headless TESS transit fits.
TESS_TRANSIT_CASES = {
    "KELT-20b": {
        "target": "KELT-20", "period_d": 3.4741039, "t0_btjd": 1698.210775,
        "transit_duration_d": 3.54 / 24.0, "radius_ratio_guess": 0.116,
        "impact_guess": 0.60,
    },
    "KELT-9b": {
        "target": "KELT-9", "period_d": 1.481124, "t0_btjd": 2000.410541,
        "transit_duration_d": 0.163158, "radius_ratio_guess": 0.0804,
        "impact_guess": 0.18,
    },
    "MASCARA-1b": {
        "target": "MASCARA-1", "period_d": 2.148774, "t0_btjd": 1998.943734,
        "transit_duration_d": 0.176083, "radius_ratio_guess": 0.0771,
        "impact_guess": 0.11,
    },
    "TOI-1431b": {
        "target": "TOI-1431", "period_d": 2.650237, "t0_btjd": 1998.900596,
        "transit_duration_d": 0.103708, "radius_ratio_guess": 0.0780,
        "impact_guess": 0.88,
    },
    "TOI-1518b": {
        "target": "TOI-1518", "period_d": 1.902603, "t0_btjd": 2000.140791,
        "transit_duration_d": 0.098542, "radius_ratio_guess": 0.0966,
        "impact_guess": 0.90,
    },
    "WASP-12b": {
        "target": "WASP-12", "period_d": 1.091419, "t0_btjd": 2000.169207,
        "transit_duration_d": 0.126700, "radius_ratio_guess": 0.1202,
        "impact_guess": 0.36,
    },
    "WASP-189b": {
        "target": "WASP-189", "period_d": 2.724031, "t0_btjd": 2000.090528,
        "transit_duration_d": 0.180567, "radius_ratio_guess": 0.0689,
        "impact_guess": 0.48,
    },
    "WASP-18b": {
        "target": "WASP-18", "period_d": 0.941452, "t0_btjd": 2000.290952,
        "transit_duration_d": 0.092083, "radius_ratio_guess": 0.1013,
        "impact_guess": 0.41,
    },
    "WASP-33b": {
        "target": "WASP-33", "period_d": 1.219870, "t0_btjd": 2000.008220,
        "transit_duration_d": 0.118917, "radius_ratio_guess": 0.1060,
        "impact_guess": 0.21,
    },
    "WASP-76b": {
        "target": "WASP-76", "period_d": 1.809886, "t0_btjd": 2000.052898,
        "transit_duration_d": 0.153917, "radius_ratio_guess": 0.1063,
        "impact_guess": 0.14,
    },
}

# Physical constants in SI units used by broadband reflection calculations.
AU_M = 1.495978707e11

# ==============================================================================
# RUNTIME PROFILES
# ==============================================================================

CONFIG_PROFILE_ENVVAR = "ATMO_CONFIG_PROFILE"
DEFAULT_RUNTIME_PROFILE = "desktop"


CONFIG_PROFILES = {
    "desktop": {
        "description": "Lower-memory local defaults for desktop and laptop runs.",
        "overrides": {
            # NLAYER scales linearly across most GPU memory components: the PreModit
            # xsmatrix scratch tensor, the per-layer dtau array, the chord geometric
            # matrix, and (crucially) the reverse-mode gradient tape through those
            # tensors during SVI init's value_and_grad pass. On a 10 GB GPU with the
            # [1500, 5500] K PreModit range and 4 atomic species at 50k nu_grid points,
            # NLAYER=20 peaked above the ~7.8 GB free budget and OOM'd in the backward
            # pass. NLAYER=10 is the standard transmission-retrieval choice (see
            # petitRADTRANS, POSEIDON, CHIMERA defaults) and preserves enough vertical
            # resolution for a smooth Guillot profile in a ~20-data-point retrieval.
            "NLAYER": 10,
            "N_SPECTRAL_POINTS": 50_000,
            "FASTCHEM_N_TEMP": 50,
            "FASTCHEM_N_PRESSURE": 50,
            "FASTCHEM_HYBRID_N_METALLICITY": 17,
            "FASTCHEM_HYBRID_N_CO_RATIO": 17,
            "MCMC_NUM_CHAINS": 2,
        },
    },
    "hpc": {
        "description": "Higher-fidelity defaults for cluster or large-memory GPU runs.",
        "overrides": {
            "NLAYER": 100,
            "N_SPECTRAL_POINTS": 250_000,
            "FASTCHEM_N_TEMP": 100,
            "FASTCHEM_N_PRESSURE": 100,
            "FASTCHEM_HYBRID_N_METALLICITY": 25,
            "FASTCHEM_HYBRID_N_CO_RATIO": 25,
            "MCMC_NUM_CHAINS": 4,
        },
    },
}

_active_runtime_profile = DEFAULT_RUNTIME_PROFILE

_profile_name = os.environ.get(CONFIG_PROFILE_ENVVAR) or DEFAULT_RUNTIME_PROFILE
_normalized_profile_name = str(_profile_name).strip().lower()
_runtime_profile = CONFIG_PROFILES[_normalized_profile_name]
for _name, _value in _runtime_profile["overrides"].items():
    globals()[_name] = _value
_active_runtime_profile = _normalized_profile_name

del _profile_name, _normalized_profile_name, _runtime_profile, _name, _value
