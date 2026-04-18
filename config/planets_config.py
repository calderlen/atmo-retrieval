"""
Planet system parameters from published literature.
"""

from math import nan

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


def get_params(planet: str | None = None, ephemeris: str | None = None) -> dict:
    """Get planet parameters for the specified planet and ephemeris."""
    planet = planet or PLANET
    ephemeris = ephemeris or EPHEMERIS
    return PLANETS[planet][ephemeris]


def list_planets() -> list[str]:
    """List all available planets."""
    return list(PLANETS.keys())


def list_ephemerides(planet: str | None = None) -> list[str]:
    """List available ephemerides for a planet."""
    planet = planet or PLANET
    return list(PLANETS[planet].keys())
