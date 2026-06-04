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
DEFAULT_LAMBDA_ANGLE = 0.0
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
            # Ephemeris
            "period": 1.219870,
            "period_err": 0.000001,
            "epoch": 2454163.22367,
            "epoch_err": 0.00022,
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
            "Kp": nan,  # TODO: look up
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
    "TOI-1413b": {
        "ExoFOP": {
            # TOI-1413.01 candidate (ExoFOP / NASA Exoplanet Archive); many fields incomplete
            "period": 6.11821285129936,
            "period_err": 0.00085346506,
            "epoch": 2459829.69897600,
            "epoch_err": 0.00158291,
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
    "jd": "JD-OBS",          # Mid-exposure Julian Date
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

# Default P-T profile
PT_PROFILE_DEFAULT = "guillot"  # Options: "guillot", "isothermal", "free`"

# ==============================================================================
# ATMOSPHERIC RT PARAMETERS
# ==============================================================================

DIFFMODE = 0
NLAYER = 10 # number of atmospheric layers (runtime profiles below override)

# Pressure range [bar]
PRESSURE_TOP = 1e-8 
PRESSURE_BTM = 1e0

# Temperature range [K]
# Sets PreModit auto_trange, ART layer T clipping, and FastChem T clipping.
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
# mass inside [T_LOW, T_HIGH]; the clip in _sample_atmosphere_state still
# catches the gamma > 3 tail where Guillot would overshoot.
LOG_GAMMA_BOUNDS = (-1.0, 1.0)

DEFAULT_KP = 169.0  # planet radial velocity semi-amplitude [km/s]
DEFAULT_KP_ERR = 20.0 
DEFAULT_RV_ABS = 0.0 # absolute RV shift [km/s]
DEFAULT_RV_ABS_ERR = 1.0
DEFAULT_TSTAR = 6000.0 # stellar temperature [K] 
DEFAULT_RP_ERR = 0.1 # planet radius error (relative to Rp/Rs)
DEFAULT_MP_ERR = 0.1 # planet mass error (relative to Mp/Ms)
DEFAULT_RSTAR_ERR = 0.1 # stellar radius error (relative to Rstar)

# Posterior reconstruction defaults
DEFAULT_POSTERIOR_RP = 1.5 # Maximum Rp/Rs for posterior reconstruction
DEFAULT_POSTERIOR_MP = 1.0 # Maximum Mp/Ms for posterior reconstruction

# Pipeline behavior defaults
SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT = True # Whether to subtract per-exposure mean from model and data before computing likelihood. Should be True for CCF-like likelihoods, but can be False for full-spectrum Gaussian likelihoods.
APPLY_SYSREM_DEFAULT = True # Whether to apply SysRem-like filtering to model and data before computing likelihood. Should be True for CCF-like likelihoods, but can be False for full-spectrum Gaussian likelihoods. Requires U and V from data preprocessing.

# Phase modeling defaults
DEFAULT_PHASE_MODE = "global" 

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
# TELLURIC LINE MODELING
# ==============================================================================

ENABLE_TELLURICS = True

TELLURIC_PWV = 5.0  # Precipitable water vapor [mm]
TELLURIC_AIRMASS = 1.2  # Typical airmass

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
FASTCHEM_T_MIN = 500.0
FASTCHEM_T_MAX = 5000.0
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

# ==============================================================================
# DATA PREPARATION DEFAULTS
# ==============================================================================

# Data prep defaults
DEFAULT_DATA_PLANET = PLANET
DEFAULT_DATA_ARM = OBSERVING_MODE
DEFAULT_USE_MOLECFIT = True
DEFAULT_RAW_DATA_DIR = "input/hrs/transmission/raw"
DEFAULT_BARYCORR = False
DEFAULT_INTRODUCED_SHIFT = True

# Data loading defaults
# Default to time-series input so the main CLI and phase-binned path work without
# extra flags. Use --data-format spectrum for collapsed retrieval products.
DEFAULT_DATA_FORMAT = "timeseries"

# Binning defaults
DEFAULT_BIN_SIZE = 50

# Doppler shadow fitting defaults
DEFAULT_SHADOW_SCALING = 1.0
DEFAULT_FIT_PARAM_FALLBACK = 1.0

# Wavelength shift defaults
DEFAULT_INTRODUCED_SHIFT_MPS = 0.0

# Misc utility defaults
DEFAULT_BIN_INFO_COUNT = 0
DEFAULT_TRACKER_MAX_USED = 0.0

# SYSREM defaults
DEFAULT_SYSREM_MAX_SYSTEMATICS_RED = [10, 10]
DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER = [10]
DEFAULT_SYSREM_STOP_TOL = 1e-4

# ==============================================================================
# PHOTOMETRY DEFAULTS
# ==============================================================================

TESS_BANDPASS_URL = "https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv"
TESS_BANDPASS_PATH = REFERENCE_BANDPASS_DIR / "tess-response-function-v2.0.csv"

# Physical constants in SI units used by broadband reflection calculations.
AU_M = 1.495978707e11

# ==============================================================================
# TELLURIC DEFAULTS
# ==============================================================================

TELLURIC_SPECIES_DEFAULT = "H2O"
TELLURIC_N_GRID = 2 ** 15
TELLURIC_T_RANGE = (150.0, 300.0)
TELLURIC_MARGIN_CM1 = 10.0
TELLURIC_VRMAX = 10.0

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
