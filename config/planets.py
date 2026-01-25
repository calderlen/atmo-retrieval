"""
Planet system parameters from published literature.
"""

from math import nan

# Active planet and ephemeris (can be overridden via CLI)
PLANET = "KELT-20b"
EPHEMERIS = "Duck24"

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
            "duration": 0.147565,        # days
            "duration_err": 0.000092,
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
            # Planetary parameters
            "M_p": 3.382,                # M_J (upper limit)
            "M_p_err": 0.13,
            "R_p": 1.741,                # R_J
            "R_p_err": 0.07,
            "T_eq": 2262,                # K
            "Kp": 169.0,                 # km/s
            "Kp_err": 6.1,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 86.03,
            "inclination_err": 0.05,
            "a": 0.0542,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.76,
            "M_star_err": 0.19,
            "R_star": 1.60,
            "R_star_err": 0.06,
            "T_star": 8980,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 3.382,
            "M_p_err": 0.13,
            "R_p": 1.741,
            "R_p_err": 0.07,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 86.12,
            "inclination_err": 0.28,
            "a": 0.0542,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.89,
            "M_star_err": 0.06,
            "R_star": 1.60,
            "R_star_err": 0.06,
            "T_star": 8980,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 3.382,
            "M_p_err": 0.13,
            "R_p": 1.735,
            "R_p_err": 0.07,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 88.0,
            "inclination_err": 1.6,
            "a": 0.033,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.46,
            "M_star_err": 0.07,
            "R_star": 1.73,
            "R_star_err": 0.04,
            "T_star": 6329,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 0.92,
            "M_p_err": 0.03,
            "R_p": 1.83,
            "R_p_err": 0.06,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
            "RA": "01h46m31.90s",
            "Dec": "+02d42m01.40s",
        },
    },
    "KELT-9b": {
        "Gaudi17": {
            # Ephemeris
            "period": 1.4811235,
            "period_err": 0.0000011,
            "epoch": 2457095.68572,
            "epoch_err": 0.00014,
            "duration": 3.9158 / 24.0,
            "duration_err": nan,
            # Orbital parameters
            "inclination": 86.79,
            "inclination_err": 0.25,
            "a": 0.03462,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 2.11,
            "M_star_err": 0.78,
            "R_star": 2.362,
            "R_star_err": 0.075,
            "T_star": 10170,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 2.17,
            "M_p_err": 0.56,
            "R_p": 1.891,
            "R_p_err": 0.061,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 83.3,
            "inclination_err": 1.1,
            "a": 0.0234,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.38,
            "M_star_err": 0.18,
            "R_star": 1.619,
            "R_star_err": 0.065,
            "T_star": 6300,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 1.39,
            "M_p_err": 0.12,
            "R_p": 1.937,
            "R_p_err": 0.064,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 86.63,
            "inclination_err": 0.03,
            "a": 0.02558,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.495,
            "M_star_err": 0.031,
            "R_star": 1.509,
            "R_star_err": 0.043,
            "T_star": 7430,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 2.093,
            "M_p_err": 0.139,
            "R_p": 1.593,
            "R_p_err": 0.054,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 83.5,
            "inclination_err": 2.0,
            "a": 0.02047,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.294,
            "M_star_err": 0.063,
            "R_star": 1.23,
            "R_star_err": 0.05,
            "T_star": 6400,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 10.20,
            "M_p_err": 0.35,
            "R_p": 1.240,
            "R_p_err": 0.079,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 84.03,
            "inclination_err": 0.14,
            "a": 0.05053,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 2.030,
            "M_star_err": 0.066,
            "R_star": 2.36,
            "R_star_err": 0.030,
            "T_star": 8000,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 1.99,
            "M_p_err": 0.16,
            "R_p": 1.619,
            "R_p_err": 0.021,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 88.45,
            "inclination_err": 0.17,
            "a": 0.04034,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.900,
            "M_star_err": 0.068,
            "R_star": 2.082,
            "R_star_err": 0.038,
            "T_star": 7554,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 3.7,
            "M_p_err": 0.9,
            "R_p": 1.597,
            "R_p_err": 0.037,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
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
            "duration_err": nan,
            # Orbital parameters
            "inclination": 80.13,
            "inclination_err": 0.13,
            "a": 0.046,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.90,
            "M_star_err": 0.10,
            "R_star": 1.92,
            "R_star_err": 0.07,
            "T_star": 7690,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 3.12,
            "M_p_err": 0.18,
            "R_p": 1.49,
            "R_p_err": 0.05,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            "Ks_expected": 294.1,  # m/s
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
            "RA": "21h04m48.89s",
            "Dec": "+55d35m16.88s",
        },
    },
    "TOI-1518b": {
        "Cabot21": {
            # Ephemeris
            "period": 1.902603,
            "period_err": 0.000011,
            "epoch": 2458787.049255,
            "epoch_err": 0.000094,
            "duration": 2.365 / 24.0,
            "duration_err": nan,
            # Orbital parameters
            "inclination": 77.84,
            "inclination_err": 0.26,
            "a": 0.0389,
            "eccentricity": nan,
            # Stellar parameters
            "M_star": 1.79,
            "M_star_err": 0.26,
            "R_star": 1.95,
            "R_star_err": 0.08,
            "T_star": 7300,
            "logg_star": nan,
            "logg_star_err": nan,
            "Fe_H": nan,
            "v_sini_star": nan,
            "v_sini_star_err": nan,
            # Planetary parameters
            "M_p": 2.3,
            "M_p_err": 2.3,
            "R_p": 1.875,
            "R_p_err": 0.053,
            "T_eq": nan,
            "Kp": nan,
            "Kp_err": nan,
            "RV_abs": nan,
            "RV_abs_err": nan,
            # Atmospheric parameters
            "kappa_IR": nan,
            "gamma": nan,
            "P0": nan,
            "X_H2": nan,
            "X_He": nan,
            "VMR_H_minus": nan,
            # Coordinates
            "RA": "23h29m04.20s",
            "Dec": "+67d02m05.30s",
        },
    },
}


def get_params(planet: str | None = None, ephemeris: str | None = None) -> dict:
    """Get planet parameters for the specified planet and ephemeris."""
    planet = planet or PLANET
    ephemeris = ephemeris or EPHEMERIS

    if planet not in PLANETS:
        raise ValueError(f"Unknown planet: {planet}. Available: {list(PLANETS.keys())}")

    if ephemeris not in PLANETS[planet]:
        available = list(PLANETS[planet].keys())
        raise ValueError(f"Unknown ephemeris '{ephemeris}' for {planet}. Available: {available}")

    return PLANETS[planet][ephemeris]


def list_planets() -> list[str]:
    """List all available planets."""
    return list(PLANETS.keys())


def list_ephemerides(planet: str | None = None) -> list[str]:
    """List available ephemerides for a planet."""
    planet = planet or PLANET
    if planet not in PLANETS:
        raise ValueError(f"Unknown planet: {planet}")
    return list(PLANETS[planet].keys())
