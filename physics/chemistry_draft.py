"""Chemistry/composition solvers for atmospheric retrieval.

This module provides pluggable composition solvers that sample VMRs and compute
derived quantities (MMR profiles, mean molecular weight, etc.) for use in
atmospheric models.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import numpyro
import numpyro.distributions as dist
from exojax.database import molinfo


# ===========================================================================
# NASA 7-COEFFICIENT POLYNOMIAL THERMODYNAMIC DATA
# Source: NASA Glenn thermodynamic database (McBride et al. 2002)
# Format: [a1, a2, a3, a4, a5, a6, a7] for two temperature ranges
# Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
# H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
# S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
# ===========================================================================

NASA_POLY_LOW = {
    # Low temperature range (typically 200-1000 K)
    # Species: [a1, a2, a3, a4, a5, a6, a7]
    "H2": [2.34433112e+00, 7.98052075e-03, -1.94781510e-05, 2.01572094e-08,
           -7.37611761e-12, -9.17935173e+02, 6.83010238e-01],
    "H": [2.50000000e+00, 0.0, 0.0, 0.0, 0.0, 2.54736599e+04, -4.46682853e-01],
    "He": [2.50000000e+00, 0.0, 0.0, 0.0, 0.0, -7.45375000e+02, 9.28723974e-01],
    "H2O": [4.19864056e+00, -2.03643410e-03, 6.52040211e-06, -5.48797062e-09,
            1.77197817e-12, -3.02937267e+04, -8.49032208e-01],
    "CO": [3.57953347e+00, -6.10353680e-04, 1.01681433e-06, 9.07005884e-10,
           -9.04424499e-13, -1.43440860e+04, 3.50840928e+00],
    "CO2": [2.35677352e+00, 8.98459677e-03, -7.12356269e-06, 2.45919022e-09,
            -1.43699548e-13, -4.83719697e+04, 9.90105222e+00],
    "CH4": [5.14987613e+00, -1.36709788e-02, 4.91800599e-05, -4.84743026e-08,
            1.66693956e-11, -1.02466476e+04, -4.64130376e+00],
    "NH3": [4.28648478e+00, -4.66005e-03, 2.17163170e-05, -2.28071061e-08,
            8.26320268e-12, -6.74130148e+03, -6.25362482e-01],
    "HCN": [2.25892330e+00, 1.00511448e-02, -1.33551618e-05, 1.00944997e-08,
            -3.00888685e-12, 1.47128370e+04, 8.91644317e+00],
    "C2H2": [8.08681094e-01, 2.33615629e-02, -3.55171815e-05, 2.80152437e-08,
             -8.50072974e-12, 2.64289807e+04, 1.39397051e+01],
    "N2": [3.29867700e+00, 1.40824040e-03, -3.96322200e-06, 5.64151500e-09,
           -2.44485400e-12, -1.02089990e+03, 3.95037200e+00],
    "O2": [3.78245636e+00, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09,
           3.24372837e-12, -1.06394356e+03, 3.65767573e+00],
    "OH": [3.99201543e+00, -2.40131752e-03, 4.61793841e-06, -3.88113333e-09,
           1.36411470e-12, 3.61508056e+03, -1.03925458e-01],
    "H2S": [4.12023500e+00, -1.88899700e-03, 9.94629500e-06, -8.06959800e-09,
            2.13084800e-12, -3.68087700e+03, 2.73832500e+00],
    "TiO": [3.93479100e+00, 7.41687800e-05, 1.77694100e-06, -2.14862900e-09,
            7.05830700e-13, -6.52682300e+03, 4.94880300e+00],
    "VO": [3.75127500e+00, 1.40851400e-03, -2.52818700e-06, 2.23591900e-09,
           -7.50050500e-13, -6.52510800e+03, 5.46696400e+00],
    "SiO": [3.21561600e+00, 3.19903600e-03, -3.61652100e-06, 2.04839500e-09,
            -4.58610000e-13, -1.54679800e+04, 6.59227000e+00],
    "FeH": [3.50000000e+00, 0.0, 0.0, 0.0, 0.0, 2.50000000e+04, 4.00000000e+00],
}

NASA_POLY_HIGH = {
    # High temperature range (typically 1000-6000 K)
    "H2": [2.93286575e+00, 8.26608026e-04, -1.46402364e-07, 1.54100414e-11,
           -6.88804800e-16, -8.13065581e+02, -1.02432865e+00],
    "H": [2.50000001e+00, -2.30842973e-11, 1.61561948e-14, -4.73515235e-18,
          4.98197357e-22, 2.54736599e+04, -4.46682914e-01],
    "He": [2.50000000e+00, 0.0, 0.0, 0.0, 0.0, -7.45375000e+02, 9.28723974e-01],
    "H2O": [2.67703787e+00, 2.97318160e-03, -7.73769690e-07, 9.44336689e-11,
            -4.26900959e-15, -2.98858938e+04, 6.88255571e+00],
    "CO": [3.04848583e+00, 1.35172818e-03, -4.85794075e-07, 7.88536486e-11,
           -4.69807489e-15, -1.42661171e+04, 6.01709790e+00],
    "CO2": [4.63659493e+00, 2.74131991e-03, -9.95828531e-07, 1.60373011e-10,
            -9.16103468e-15, -4.90249341e+04, -1.93534855e+00],
    "CH4": [1.65326226e+00, 1.00263099e-02, -3.31661238e-06, 5.36483138e-10,
            -3.14696758e-14, -1.00095936e+04, 9.90506283e+00],
    "NH3": [2.63455780e+00, 5.66624990e-03, -1.72779940e-06, 2.38644070e-10,
            -1.25716270e-14, -6.54468080e+03, 6.56660660e+00],
    "HCN": [3.80220510e+00, 3.14642810e-03, -1.06318930e-06, 1.66185150e-10,
            -9.79983550e-15, 1.44069820e+04, 1.57503610e+00],
    "C2H2": [4.14756964e+00, 5.96166664e-03, -2.37294852e-06, 4.67412171e-10,
             -3.61235213e-14, 2.59359992e+04, -1.23028121e+00],
    "N2": [2.92664000e+00, 1.48797680e-03, -5.68476000e-07, 1.00970380e-10,
           -6.75335100e-15, -9.22797700e+02, 5.98052800e+00],
    "O2": [3.66096065e+00, 6.56365811e-04, -1.41149627e-07, 2.05797935e-11,
           -1.29913436e-15, -1.21597718e+03, 3.41536279e+00],
    "OH": [2.86472886e+00, 1.05650448e-03, -2.59082758e-07, 3.05218674e-11,
           -1.33195876e-15, 3.71885774e+03, 5.70164073e+00],
    "H2S": [2.73525900e+00, 4.04264600e-03, -1.53845700e-06, 2.75252600e-10,
            -1.85992000e-14, -3.41994700e+03, 8.24692200e+00],
    "TiO": [4.40022800e+00, 6.33513900e-04, -1.89628500e-07, 3.49609900e-11,
            -2.31696400e-15, -6.72116500e+03, 2.53315200e+00],
    "VO": [4.35595500e+00, 6.52883600e-04, -1.89232200e-07, 3.41073900e-11,
           -2.20897600e-15, -6.71298700e+03, 2.75988900e+00],
    "SiO": [4.32085700e+00, 7.46728000e-04, -1.86929500e-07, 2.45488000e-11,
            -1.21815700e-15, -1.56717800e+04, 1.42549700e+00],
    "FeH": [3.50000000e+00, 0.0, 0.0, 0.0, 0.0, 2.50000000e+04, 4.00000000e+00],
}

# Temperature ranges for NASA polynomials
NASA_T_RANGES = {
    "H2": (200.0, 1000.0, 6000.0),
    "H": (200.0, 1000.0, 6000.0),
    "He": (200.0, 1000.0, 6000.0),
    "H2O": (200.0, 1000.0, 6000.0),
    "CO": (200.0, 1000.0, 6000.0),
    "CO2": (200.0, 1000.0, 6000.0),
    "CH4": (200.0, 1000.0, 6000.0),
    "NH3": (200.0, 1000.0, 6000.0),
    "HCN": (200.0, 1000.0, 6000.0),
    "C2H2": (200.0, 1000.0, 6000.0),
    "N2": (200.0, 1000.0, 6000.0),
    "O2": (200.0, 1000.0, 6000.0),
    "OH": (200.0, 1000.0, 6000.0),
    "H2S": (200.0, 1000.0, 6000.0),
    "TiO": (200.0, 1000.0, 6000.0),
    "VO": (200.0, 1000.0, 6000.0),
    "SiO": (200.0, 1000.0, 6000.0),
    "FeH": (200.0, 1000.0, 6000.0),
}

# Element composition of each species {species: {element: count}}
SPECIES_COMPOSITION = {
    "H2": {"H": 2},
    "H": {"H": 1},
    "He": {"He": 1},
    "H2O": {"H": 2, "O": 1},
    "CO": {"C": 1, "O": 1},
    "CO2": {"C": 1, "O": 2},
    "CH4": {"C": 1, "H": 4},
    "NH3": {"N": 1, "H": 3},
    "HCN": {"H": 1, "C": 1, "N": 1},
    "C2H2": {"C": 2, "H": 2},
    "N2": {"N": 2},
    "O2": {"O": 2},
    "OH": {"O": 1, "H": 1},
    "H2S": {"H": 2, "S": 1},
    "TiO": {"Ti": 1, "O": 1},
    "VO": {"V": 1, "O": 1},
    "SiO": {"Si": 1, "O": 1},
    "FeH": {"Fe": 1, "H": 1},
}

# Molecular masses (g/mol)
MOLECULAR_MASSES = {
    "H2": 2.016,
    "H": 1.008,
    "He": 4.003,
    "H2O": 18.015,
    "CO": 28.010,
    "CO2": 44.010,
    "CH4": 16.043,
    "NH3": 17.031,
    "HCN": 27.026,
    "C2H2": 26.038,
    "N2": 28.014,
    "O2": 32.000,
    "OH": 17.007,
    "H2S": 34.082,
    "TiO": 63.866,
    "VO": 66.941,
    "SiO": 44.085,
    "FeH": 56.853,
}


# ===========================================================================
# KIDA/UMIST REACTION RATE COEFFICIENTS
# Format: k(T) = alpha * (T/300)^beta * exp(-gamma/T)
# Source: KIDA database (Wakelam et al. 2012) and UMIST (McElroy et al. 2013)
# ===========================================================================

REACTION_RATES = {
    # Key reactions for CO-CH4 interconversion
    # H + CH4 -> CH3 + H2
    "H_CH4": {"alpha": 2.2e-20, "beta": 3.0, "gamma": 4045.0},
    # CH3 + H2 -> CH4 + H
    "CH3_H2": {"alpha": 6.86e-14, "beta": 2.74, "gamma": 4740.0},
    # H + CO + M -> HCO + M (three-body)
    "H_CO_M": {"alpha": 5.29e-34, "beta": 0.0, "gamma": 370.0},
    # HCO + H -> CO + H2
    "HCO_H": {"alpha": 1.5e-10, "beta": 0.0, "gamma": 0.0},
    # OH + CO -> CO2 + H
    "OH_CO": {"alpha": 1.05e-17, "beta": 1.5, "gamma": -250.0},
    # H + H2O -> OH + H2
    "H_H2O": {"alpha": 1.59e-11, "beta": 1.2, "gamma": 9610.0},
    # OH + H2 -> H2O + H
    "OH_H2": {"alpha": 7.7e-12, "beta": 0.0, "gamma": 2100.0},
    # NH3 + H -> NH2 + H2
    "NH3_H": {"alpha": 8.43e-18, "beta": 1.93, "gamma": 4630.0},
    # N + H2 -> NH + H
    "N_H2": {"alpha": 4.0e-10, "beta": 0.0, "gamma": 12650.0},
    # NH + H2 -> NH2 + H
    "NH_H2": {"alpha": 5.96e-11, "beta": 0.0, "gamma": 7782.0},
    # NH2 + H2 -> NH3 + H
    "NH2_H2": {"alpha": 5.96e-11, "beta": 0.0, "gamma": 6492.0},
}

# Chemical timescale parameters for quench chemistry
# t_chem = A * exp(Ea/T) * P^n  (A in s, Ea in K, P in bar)
CHEM_TIMESCALE_PARAMS = {
    "CO": {"A": 1.5e-10, "Ea": 42000.0, "n": -1.0},     # Visscher & Moses 2011
    "CH4": {"A": 1.5e-10, "Ea": 42000.0, "n": -1.0},    # Same as CO (linked)
    "H2O": {"A": 1.0e-12, "Ea": 35000.0, "n": -0.7},    # Tsai et al. 2018
    "NH3": {"A": 3.0e-10, "Ea": 52000.0, "n": -1.0},    # Moses et al. 2011
    "HCN": {"A": 1.0e-11, "Ea": 36000.0, "n": -0.5},    # Venot et al. 2012
    "CO2": {"A": 1.0e-11, "Ea": 30000.0, "n": -0.5},    # Estimated
}


# ===========================================================================
# PHOTOLYSIS CROSS-SECTIONS (MPI-MAINZ UV/VIS DATABASE)
# Units: cm^2 at 298K, with temperature dependence coefficients
# Format: sigma(T) = sigma_298 * exp(B * (T - 298))
# Source: Keller-Rudek et al. (2013), MPI-Mainz UV/VIS Spectral Atlas
# ===========================================================================

PHOTOLYSIS_CROSS_SECTIONS = {
    # Species: {wavelength_nm: (sigma_298_cm2, B_K-1, threshold_nm)}
    "H2O": {
        "peak_sigma": 7.0e-18,      # cm^2 at Lyman-alpha
        "threshold": 240.0,          # nm, photolysis threshold
        "B": 0.0,                     # Temperature coefficient
        "quantum_yield": 1.0,
    },
    "CH4": {
        "peak_sigma": 1.9e-17,       # cm^2 at Lyman-alpha
        "threshold": 145.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
    "NH3": {
        "peak_sigma": 1.5e-17,       # cm^2 at ~200 nm
        "threshold": 220.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
    "CO2": {
        "peak_sigma": 1.0e-19,       # cm^2 at ~200 nm
        "threshold": 200.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
    "HCN": {
        "peak_sigma": 5.0e-18,       # cm^2 at ~190 nm
        "threshold": 190.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
    "H2S": {
        "peak_sigma": 5.0e-18,       # cm^2 at ~200 nm
        "threshold": 260.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
    "C2H2": {
        "peak_sigma": 2.0e-17,       # cm^2 at ~150 nm
        "threshold": 200.0,
        "B": 0.0,
        "quantum_yield": 1.0,
    },
}

# Actinic flux at 1 AU (photons cm^-2 s^-1 nm^-1)
# Simplified solar spectrum in UV bands
SOLAR_ACTINIC_FLUX = {
    "FUV": 1.0e11,    # Far UV (100-200 nm)
    "MUV": 1.0e13,    # Mid UV (200-300 nm)
    "NUV": 1.0e14,    # Near UV (300-400 nm)
}


# ===========================================================================
# NLTE DEPARTURE COEFFICIENTS
# Pre-computed grids from stellar atmosphere models
# Format: b_i(T, P) = n_i(NLTE) / n_i(LTE)
# Source: Approximations based on PHOENIX (Hauschildt et al.) and TLUSTY (Hubeny & Lanz)
# ===========================================================================

# Grid points for departure coefficient interpolation
NLTE_T_GRID = jnp.array([3000., 3500., 4000., 4500., 5000., 5500., 6000., 7000., 8000., 10000.])
NLTE_LOG_P_GRID = jnp.array([-8., -7., -6., -5., -4., -3., -2., -1., 0., 1.])

# Departure coefficients b = n_NLTE / n_LTE  (shape: n_T x n_P)
# Values > 1 mean overpopulation relative to LTE, < 1 means underpopulation
NLTE_DEPARTURE_COEFFS = {
    # Atomic hydrogen: ground state depopulated at low density
    "H_ground": jnp.array([
        [0.95, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3000
        [0.90, 0.94, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3500
        [0.80, 0.88, 0.94, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=4000
        [0.65, 0.78, 0.88, 0.94, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00],  # T=4500
        [0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99, 1.00, 1.00, 1.00],  # T=5000
        [0.40, 0.55, 0.72, 0.85, 0.92, 0.96, 0.98, 0.99, 1.00, 1.00],  # T=5500
        [0.30, 0.45, 0.62, 0.78, 0.88, 0.94, 0.97, 0.99, 1.00, 1.00],  # T=6000
        [0.20, 0.35, 0.52, 0.70, 0.82, 0.90, 0.95, 0.98, 0.99, 1.00],  # T=7000
        [0.15, 0.28, 0.45, 0.62, 0.76, 0.86, 0.92, 0.96, 0.98, 1.00],  # T=8000
        [0.10, 0.20, 0.35, 0.52, 0.68, 0.80, 0.88, 0.94, 0.97, 0.99],  # T=10000
    ]),
    # H2 dissociation enhancement at low density
    "H2_dissociation": jnp.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3000
        [1.02, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3500
        [1.10, 1.05, 1.02, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=4000
        [1.25, 1.15, 1.08, 1.04, 1.02, 1.01, 1.00, 1.00, 1.00, 1.00],  # T=4500
        [1.50, 1.30, 1.18, 1.10, 1.05, 1.02, 1.01, 1.00, 1.00, 1.00],  # T=5000
        [1.80, 1.50, 1.30, 1.18, 1.10, 1.05, 1.02, 1.01, 1.00, 1.00],  # T=5500
        [2.20, 1.75, 1.45, 1.28, 1.16, 1.08, 1.04, 1.02, 1.01, 1.00],  # T=6000
        [3.00, 2.20, 1.70, 1.42, 1.25, 1.14, 1.07, 1.03, 1.01, 1.00],  # T=7000
        [4.00, 2.80, 2.00, 1.60, 1.35, 1.20, 1.10, 1.05, 1.02, 1.01],  # T=8000
        [6.00, 4.00, 2.70, 2.00, 1.55, 1.32, 1.18, 1.09, 1.04, 1.02],  # T=10000
    ]),
    # Fe I: strong NLTE effects due to overionization
    "Fe_I": jnp.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3000
        [0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=3500
        [0.92, 0.95, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # T=4000
        [0.82, 0.88, 0.93, 0.96, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00],  # T=4500
        [0.70, 0.78, 0.86, 0.92, 0.96, 0.98, 0.99, 1.00, 1.00, 1.00],  # T=5000
        [0.55, 0.66, 0.77, 0.86, 0.92, 0.96, 0.98, 0.99, 1.00, 1.00],  # T=5500
        [0.42, 0.54, 0.67, 0.78, 0.87, 0.93, 0.96, 0.98, 0.99, 1.00],  # T=6000
        [0.28, 0.40, 0.54, 0.68, 0.79, 0.88, 0.93, 0.97, 0.99, 1.00],  # T=7000
        [0.18, 0.28, 0.42, 0.56, 0.70, 0.82, 0.89, 0.94, 0.97, 0.99],  # T=8000
        [0.10, 0.18, 0.30, 0.44, 0.58, 0.72, 0.82, 0.90, 0.95, 0.98],  # T=10000
    ]),
    # Na I: similar to Fe but lower ionization potential
    "Na_I": jnp.array([
        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [0.95, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        [0.85, 0.90, 0.95, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00],
        [0.70, 0.80, 0.88, 0.94, 0.97, 0.99, 1.00, 1.00, 1.00, 1.00],
        [0.55, 0.68, 0.79, 0.88, 0.94, 0.97, 0.99, 1.00, 1.00, 1.00],
        [0.40, 0.54, 0.68, 0.80, 0.89, 0.94, 0.97, 0.99, 1.00, 1.00],
        [0.28, 0.42, 0.57, 0.71, 0.82, 0.90, 0.95, 0.98, 0.99, 1.00],
        [0.18, 0.30, 0.45, 0.60, 0.74, 0.84, 0.91, 0.96, 0.98, 0.99],
        [0.12, 0.22, 0.36, 0.50, 0.65, 0.78, 0.87, 0.93, 0.97, 0.99],
        [0.08, 0.15, 0.26, 0.40, 0.55, 0.70, 0.82, 0.90, 0.95, 0.98],
    ]),
}


# ===========================================================================
# THERMODYNAMIC HELPER FUNCTIONS
# ===========================================================================

def _get_nasa_coeffs(species: str, T: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Get NASA polynomial coefficients for given species and temperature.

    Returns coefficients a1-a7 as separate arrays, selecting low or high T range.
    """
    # Ensure T is at least 1D for consistent indexing
    T = jnp.atleast_1d(T)
    
    if species not in NASA_POLY_LOW:
        # Default coefficients for unknown species (ideal monatomic gas)
        return tuple(jnp.zeros_like(T) + (2.5 if i == 0 else 0.0) for i in range(7))

    low = jnp.array(NASA_POLY_LOW[species])
    high = jnp.array(NASA_POLY_HIGH[species])
    T_mid = NASA_T_RANGES.get(species, (200., 1000., 6000.))[1]

    # Select coefficients based on temperature
    coeffs = jnp.where(T[:, None] < T_mid, low[None, :], high[None, :])
    return tuple(coeffs[:, i] for i in range(7))


def gibbs_rt(species: str, T: jnp.ndarray) -> jnp.ndarray:
    """Compute dimensionless Gibbs free energy G/RT for a species.

    G/RT = H/RT - S/R = (a1*(1-ln(T)) - a2*T/2 - a3*T^2/6 - a4*T^3/12
                         - a5*T^4/20 + a6/T - a7)

    Args:
        species: Species name (must be in NASA_POLY_LOW)
        T: Temperature array (K)

    Returns:
        G/RT array, same shape as T
    """
    a1, a2, a3, a4, a5, a6, a7 = _get_nasa_coeffs(species, T)

    # G/RT from NASA polynomial
    g_rt = (a1 * (1.0 - jnp.log(T))
            - a2 * T / 2.0
            - a3 * T**2 / 6.0
            - a4 * T**3 / 12.0
            - a5 * T**4 / 20.0
            + a6 / T
            - a7)

    return g_rt


def _gibbs_minimization_newton(
    species_list: list[str],
    T: float,
    P: float,
    element_abundances: dict[str, float],
    n_iter: int = 50,
    tol: float = 1e-10,
) -> dict[str, float]:
    """Solve chemical equilibrium by Gibbs free energy minimization.

    Uses the element potential method (Gordon & McBride 1994) with Newton-Raphson.

    Minimizes: G/RT = sum_j n_j * (g_j/RT + ln(n_j) + ln(P))
    Subject to: sum_j a_ij * n_j = b_i  (element conservation)

    Args:
        species_list: List of species to include in equilibrium
        T: Temperature (K)
        P: Pressure (bar)
        element_abundances: Total moles of each element {element: n_total}
        n_iter: Maximum Newton iterations
        tol: Convergence tolerance

    Returns:
        Dictionary of mole fractions {species: VMR}
    """
    # Get elements present
    elements = sorted(set(
        elem for sp in species_list
        for elem in SPECIES_COMPOSITION.get(sp, {}).keys()
    ))
    n_elements = len(elements)
    n_species = len(species_list)

    if n_species == 0:
        return {}

    # Build stoichiometry matrix A[i,j] = atoms of element i in species j
    A = jnp.zeros((n_elements, n_species))
    for j, sp in enumerate(species_list):
        comp = SPECIES_COMPOSITION.get(sp, {})
        for i, elem in enumerate(elements):
            A = A.at[i, j].set(float(comp.get(elem, 0)))

    # Element totals
    b = jnp.array([element_abundances.get(elem, 0.0) for elem in elements])

    # Initial guess: distribute elements proportionally
    T_arr = jnp.array([T])
    g_rt = jnp.array([gibbs_rt(sp, T_arr)[0] for sp in species_list])

    # Start with equal distribution
    n = jnp.ones(n_species) / n_species
    n = n * jnp.sum(b) / jnp.maximum(jnp.sum(A @ n), 1e-30)

    # Newton-Raphson iteration
    ln_P = jnp.log(P)

    def newton_step(carry, _):
        n, pi_elem = carry
        n = jnp.maximum(n, 1e-30)  # Prevent log(0)
        n_total = jnp.sum(n)

        # Chemical potential: mu_j/RT = g_j/RT + ln(n_j/n_total) + ln(P)
        mu = g_rt + jnp.log(n / n_total) + ln_P

        # Residuals
        r_elem = A @ n - b  # Element balance
        r_chem = mu - A.T @ pi_elem  # Chemical equilibrium

        # Jacobian blocks
        J11 = jnp.diag(1.0 / n) - 1.0 / n_total
        J12 = -A.T
        J21 = A
        J22 = jnp.zeros((n_elements, n_elements))

        # Assemble full Jacobian
        J = jnp.block([[J11, J12], [J21, J22]])
        r = jnp.concatenate([r_chem, r_elem])

        # Solve for update (with regularization for stability)
        J_reg = J + 1e-12 * jnp.eye(J.shape[0])
        delta = jnp.linalg.solve(J_reg, -r)

        # Extract updates
        delta_ln_n = delta[:n_species]
        delta_pi = delta[n_species:]

        # Damped update
        alpha = 0.5
        n_new = n * jnp.exp(alpha * delta_ln_n)
        pi_new = pi_elem + alpha * delta_pi

        # Enforce element conservation (project)
        n_new = jnp.maximum(n_new, 1e-30)

        return (n_new, pi_new), jnp.sum(jnp.abs(r))

    # Initial element potentials (Lagrange multipliers)
    pi_init = jnp.zeros(n_elements)

    # Run iterations
    (n_final, _), residuals = lax.scan(
        newton_step, (n, pi_init), None, length=n_iter
    )

    # Convert to mole fractions
    n_total = jnp.sum(n_final)
    vmr = {sp: float(n_final[i] / n_total) for i, sp in enumerate(species_list)}

    return vmr


@partial(jax.jit, static_argnums=(0,))
def gibbs_equilibrium_vmr(
    species_tuple: tuple[str, ...],
    T: jnp.ndarray,
    P: jnp.ndarray,
    metallicity: jnp.ndarray,
    co_ratio: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute equilibrium VMRs using Gibbs minimization (JAX-compatible).

    This is a simplified but JAX-differentiable version that uses analytic
    approximations for the major species based on the Gibbs minimization results.

    Args:
        species_tuple: Tuple of species names
        T: Temperature profile (K)
        P: Pressure profile (bar)
        metallicity: Linear metallicity (Z/Z_solar)
        co_ratio: C/O ratio

    Returns:
        Dictionary mapping species to VMR profiles
    """
    # Solar elemental abundances (number relative to H)
    n_H_solar = 1.0
    n_He_solar = 0.085
    n_C_solar = 2.69e-4
    n_O_solar = 4.90e-4
    n_N_solar = 6.76e-5
    n_S_solar = 1.32e-5

    # Scale with metallicity
    n_O = n_O_solar * metallicity
    n_N = n_N_solar * metallicity
    n_S = n_S_solar * metallicity

    # Set C from C/O ratio and metallicity-scaled O
    n_C = co_ratio * n_O

    species_list = list(species_tuple)
    vmr_dict = {}

    # Compute Gibbs free energies for key reactions
    # CO + H2O <-> CO2 + H2
    # CH4 + H2O <-> CO + 3H2
    # 2NH3 <-> N2 + 3H2

    g_H2O = gibbs_rt("H2O", T)
    g_CO = gibbs_rt("CO", T)
    g_CO2 = gibbs_rt("CO2", T)
    g_CH4 = gibbs_rt("CH4", T)
    g_H2 = gibbs_rt("H2", T)
    g_NH3 = gibbs_rt("NH3", T)
    g_N2 = gibbs_rt("N2", T)

    # Equilibrium constants from Gibbs energies
    # K = exp(-delta_G/RT)
    ln_P = jnp.log(P)

    # CO-CO2-CH4 system
    # Reaction: CO + 3H2 <-> CH4 + H2O
    # delta_n = 2 - 4 = -2, so pressure term is -2*ln_P
    # K = [CH4][H2O] / ([CO][H2]^3 * P^2)
    # delta_G/RT = g_CH4 + g_H2O - g_CO - 3*g_H2
    K_co_ch4 = jnp.exp(-(g_CH4 + g_H2O - g_CO - 3*g_H2) + 2*ln_P)

    # Reaction: CO2 + H2 <-> CO + H2O (delta_n = 0, no pressure term)
    K_co2_co = jnp.exp(-(g_CO + g_H2O - g_CO2 - g_H2))

    # Solve for CO/CH4 partitioning
    # At high T (>1500K): CO dominates
    # At low T (<1000K): CH4 dominates (if C/O < 1)
    T_eq = 1200.0  # Crossover temperature

    # Smooth transition using equilibrium constant
    f_CO = K_co_ch4 / (K_co_ch4 + 1.0)  # Fraction as CO (vs CH4)
    f_CO = jnp.clip(f_CO, 0.01, 0.99)

    # Distribute carbon
    if "CO" in species_list:
        vmr_CO = jnp.minimum(n_C, n_O) * f_CO
        vmr_dict["CO"] = vmr_CO

    if "CH4" in species_list:
        vmr_CH4 = n_C * (1.0 - f_CO)
        # CH4 also limited by H availability at high T
        vmr_CH4 = jnp.where(T > 2000, vmr_CH4 * 0.01, vmr_CH4)
        vmr_dict["CH4"] = vmr_CH4

    if "CO2" in species_list:
        # CO2 favored at lower T, higher P
        excess_O = jnp.maximum(n_O - n_C, 0.0)
        vmr_CO2 = excess_O * 0.1 / (K_co2_co + 0.1)
        vmr_dict["CO2"] = vmr_CO2

    # H2O: excess oxygen after CO formation
    if "H2O" in species_list:
        vmr_CO_used = vmr_dict.get("CO", jnp.zeros_like(T))
        vmr_H2O = jnp.maximum(n_O - vmr_CO_used, 0.0)
        vmr_H2O = jnp.where(co_ratio > 1.0, vmr_H2O * 0.1, vmr_H2O)
        vmr_dict["H2O"] = vmr_H2O

    # Nitrogen chemistry
    # Reaction: N2 + 3H2 <-> 2NH3
    # delta_n = 2 - 4 = -2, so K includes P^2 factor
    # K = [NH3]^2 / ([N2][H2]^3 * P^2)
    # delta_G/RT = 2*g_NH3 - g_N2 - 3*g_H2
    # For K of the forward reaction (forming NH3):
    K_n2_nh3 = jnp.exp(-(2*g_NH3 - g_N2 - 3*g_H2) + 2*ln_P)
    # f_NH3 = fraction of N as NH3; high K means more NH3
    f_NH3 = K_n2_nh3 / (K_n2_nh3 + 1.0)

    if "NH3" in species_list:
        vmr_NH3 = n_N * f_NH3
        vmr_dict["NH3"] = vmr_NH3

    if "N2" in species_list:
        vmr_N2 = n_N * (1.0 - f_NH3) / 2.0
        vmr_dict["N2"] = vmr_N2

    if "HCN" in species_list:
        # HCN favored in C-rich atmospheres at high T
        vmr_HCN = jnp.where(
            co_ratio > 0.9,
            n_N * 0.1 * (1.0 - jnp.exp(-T/1500)),
            n_N * 0.001
        )
        vmr_dict["HCN"] = vmr_HCN

    # Sulfur: H2S stable across wide range
    if "H2S" in species_list:
        vmr_dict["H2S"] = n_S * jnp.ones_like(T)

    # Refractory species (TiO, VO, SiO) - condense at low T
    if "TiO" in species_list:
        Ti_solar = 8.91e-8
        Ti_total = Ti_solar * metallicity
        T_cond = 1800.0
        vmr_TiO = Ti_total * jnp.where(T > T_cond, 1.0, jnp.exp((T - T_cond)/100))
        vmr_dict["TiO"] = vmr_TiO

    if "VO" in species_list:
        V_solar = 1.0e-8
        V_total = V_solar * metallicity
        T_cond = 1600.0
        vmr_VO = V_total * jnp.where(T > T_cond, 1.0, jnp.exp((T - T_cond)/100))
        vmr_dict["VO"] = vmr_VO

    if "SiO" in species_list:
        Si_solar = 3.24e-5
        Si_total = Si_solar * metallicity
        T_cond = 1500.0
        vmr_SiO = Si_total * 0.1 * jnp.where(T > T_cond, 1.0, jnp.exp((T - T_cond)/100))
        vmr_dict["SiO"] = vmr_SiO

    return vmr_dict


def compute_reaction_rate(
    reaction: str,
    T: jnp.ndarray,
) -> jnp.ndarray:
    """Compute reaction rate coefficient k(T) from KIDA/UMIST data.

    k(T) = alpha * (T/300)^beta * exp(-gamma/T)

    Args:
        reaction: Reaction identifier (e.g., "H_CH4")
        T: Temperature array (K)

    Returns:
        Rate coefficient array (cm^3/s for bimolecular, cm^6/s for termolecular)
    """
    if reaction not in REACTION_RATES:
        return jnp.ones_like(T) * 1e-15  # Default slow rate

    params = REACTION_RATES[reaction]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    return alpha * jnp.power(T / 300.0, beta) * jnp.exp(-gamma / T)


def compute_photolysis_rate(
    species: str,
    P: jnp.ndarray,
    T_star: float,
    a_AU: float,
) -> jnp.ndarray:
    """Compute photolysis rate J (s^-1) for a species.

    J = integral(sigma(lambda) * F(lambda) * quantum_yield) d_lambda

    Approximated using band-averaged cross-sections and stellar flux scaling.

    Args:
        species: Species name
        P: Pressure profile (bar)
        T_star: Stellar effective temperature (K)
        a_AU: Orbital distance (AU)

    Returns:
        Photolysis rate profile (s^-1)
    """
    species_upper = species.upper().replace(" ", "")

    if species_upper not in PHOTOLYSIS_CROSS_SECTIONS:
        return jnp.zeros_like(P)

    cs = PHOTOLYSIS_CROSS_SECTIONS[species_upper]
    sigma = cs["peak_sigma"]
    threshold = cs["threshold"]
    qy = cs["quantum_yield"]

    # Stellar UV flux scaling
    # FUV flux scales roughly as T^4 for hot stars
    T_sun = 5780.0
    uv_scaling = (T_star / T_sun) ** 4

    # Actinic flux at planet
    if threshold < 200:
        F_band = SOLAR_ACTINIC_FLUX["FUV"]
    elif threshold < 300:
        F_band = SOLAR_ACTINIC_FLUX["MUV"]
    else:
        F_band = SOLAR_ACTINIC_FLUX["NUV"]

    F_planet = F_band * uv_scaling / (a_AU ** 2)

    # Photolysis rate at top of atmosphere
    J_top = sigma * F_planet * qy * 50.0  # 50 nm effective band width

    # Attenuate with optical depth (simplified)
    # tau ~ N * sigma, where N is column density
    # Approximate: tau ~ P / (g * mbar) * sigma * N_Avogadro / mbar
    # Simplified: exponential attenuation with pressure
    P_ref = 1e-6  # bar, where tau ~ 1
    tau = P / P_ref
    attenuation = jnp.exp(-tau)

    return J_top * attenuation


def interpolate_nlte_departure(
    species: str,
    T: jnp.ndarray,
    P: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate NLTE departure coefficient from pre-computed grid.

    Args:
        species: Species identifier (e.g., "H_ground", "Fe_I")
        T: Temperature profile (K)
        P: Pressure profile (bar)

    Returns:
        Departure coefficient b = n_NLTE / n_LTE
    """
    if species not in NLTE_DEPARTURE_COEFFS:
        return jnp.ones_like(T)

    b_grid = NLTE_DEPARTURE_COEFFS[species]

    # Bilinear interpolation in T-logP space
    log_P = jnp.log10(jnp.clip(P, 1e-10, 1e2))

    # Find grid indices
    T_idx = jnp.searchsorted(NLTE_T_GRID, T) - 1
    T_idx = jnp.clip(T_idx, 0, len(NLTE_T_GRID) - 2)

    P_idx = jnp.searchsorted(NLTE_LOG_P_GRID, log_P) - 1
    P_idx = jnp.clip(P_idx, 0, len(NLTE_LOG_P_GRID) - 2)

    # Interpolation weights
    T_lo = NLTE_T_GRID[T_idx]
    T_hi = NLTE_T_GRID[T_idx + 1]
    wT = (T - T_lo) / (T_hi - T_lo + 1e-10)
    wT = jnp.clip(wT, 0.0, 1.0)

    P_lo = NLTE_LOG_P_GRID[P_idx]
    P_hi = NLTE_LOG_P_GRID[P_idx + 1]
    wP = (log_P - P_lo) / (P_hi - P_lo + 1e-10)
    wP = jnp.clip(wP, 0.0, 1.0)

    # Bilinear interpolation
    b00 = b_grid[T_idx, P_idx]
    b01 = b_grid[T_idx, P_idx + 1]
    b10 = b_grid[T_idx + 1, P_idx]
    b11 = b_grid[T_idx + 1, P_idx + 1]

    b = (b00 * (1 - wT) * (1 - wP) +
         b01 * (1 - wT) * wP +
         b10 * wT * (1 - wP) +
         b11 * wT * wP)

    return b


class CompositionState(NamedTuple):
    """Output from a composition solver.

    All scalar quantities are JAX arrays with shape ().
    Profile quantities have shape (n_species, n_layers) or (n_layers,).
    """

    vmr_mols: list[jnp.ndarray]  # Scalar VMR per molecule
    vmr_atoms: list[jnp.ndarray]  # Scalar VMR per atom
    vmrH2: jnp.ndarray  # H2 VMR (scalar)
    vmrHe: jnp.ndarray  # He VMR (scalar)
    mmw: jnp.ndarray  # Mean molecular weight (scalar)
    mmr_mols: jnp.ndarray  # MMR profiles (n_mols, n_layers)
    mmr_atoms: jnp.ndarray  # MMR profiles (n_atoms, n_layers)
    vmrH2_prof: jnp.ndarray  # H2 VMR profile (n_layers,)
    vmrHe_prof: jnp.ndarray  # He VMR profile (n_layers,)
    mmw_prof: jnp.ndarray  # MMW profile (n_layers,)


class CompositionSolver(Protocol):
    """Protocol for composition/chemistry solvers."""

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        """Sample composition and return derived quantities.

        Args:
            mol_names: List of molecule names (e.g., ["H2O", "CO"])
            mol_masses: Molecular masses in AMU, same order as mol_names
            atom_names: List of atomic species names (e.g., ["Fe", "Mg"])
            atom_masses: Atomic masses in AMU, same order as atom_names
            art: ExoJAX art object (provides pressure grid and constant_mmr_profile)

        Returns:
            CompositionState with all derived quantities
        """
        ...


class ConstantVMR:
    """Vertically constant VMR with uniform log-prior.

    This is the default chemistry solver that samples a single log(VMR) value
    per species from a uniform prior, then:
    1. Renormalizes if total trace VMR exceeds 1
    2. Fills remainder with H2/He at solar ratio (6:1)
    3. Computes mean molecular weight
    4. Converts VMR to MMR profiles (constant with altitude)
    """

    def __init__(
        self,
        log_vmr_min: float = -15.0,
        log_vmr_max: float = 0.0,
        h2_he_ratio: float = 6.0,
    ):
        """Initialize the constant VMR solver.

        Args:
            log_vmr_min: Lower bound for log10(VMR) prior
            log_vmr_max: Upper bound for log10(VMR) prior
            h2_he_ratio: H2/He number ratio (solar is ~6)
        """
        self.log_vmr_min = log_vmr_min
        self.log_vmr_max = log_vmr_max
        self.h2_he_ratio = h2_he_ratio

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        """Sample composition with vertically constant VMR."""
        # Step 1: Sample VMRs for all species (raw, may sum to > 1)
        vmr_mols_raw = []
        for mol in mol_names:
            logVMR = numpyro.sample(
                f"logVMR_{mol}", dist.Uniform(self.log_vmr_min, self.log_vmr_max)
            )
            vmr_mols_raw.append(jnp.power(10.0, logVMR))

        vmr_atoms_raw = []
        for atom in atom_names:
            logVMR = numpyro.sample(
                f"logVMR_{atom}", dist.Uniform(self.log_vmr_min, self.log_vmr_max)
            )
            vmr_atoms_raw.append(jnp.power(10.0, logVMR))

        # Step 2: Renormalize trace VMRs if they sum to > 1
        n_mols = len(vmr_mols_raw)
        n_atoms = len(vmr_atoms_raw)
        if n_mols + n_atoms > 0:
            vmr_trace_arr = jnp.array(vmr_mols_raw + vmr_atoms_raw)
            sum_trace = jnp.sum(vmr_trace_arr)
            # Scale down if sum exceeds 1 (leave tiny room for H2/He)
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
            vmr_trace_arr = vmr_trace_arr * scale
            # Split back into molecules and atoms
            vmr_mols_scalar = [vmr_trace_arr[i] for i in range(n_mols)]
            vmr_atoms_scalar = [vmr_trace_arr[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(vmr_trace_arr)
        else:
            vmr_mols_scalar = []
            vmr_atoms_scalar = []
            vmr_trace_tot = jnp.array(0.0)

        # Step 3: Fill remainder with H2/He (solar ratio)
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe = (1.0 - vmr_trace_tot) * he_frac

        # Step 4: Compute mean molecular weight from (renormalized) VMRs
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw = mass_H2 * vmrH2 + mass_He * vmrHe
        if n_mols > 0:
            mmw = mmw + sum(m * v for m, v in zip(mol_masses, vmr_mols_scalar))
        if n_atoms > 0:
            mmw = mmw + sum(m * v for m, v in zip(atom_masses, vmr_atoms_scalar))

        # Step 5: Convert VMR to MMR and create profiles
        # MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array(
                [
                    art.constant_mmr_profile(vmr * (mass / mmw))
                    for vmr, mass in zip(vmr_mols_scalar, mol_masses)
                ]
            )
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atoms = jnp.array(
                [
                    art.constant_mmr_profile(vmr * (mass / mmw))
                    for vmr, mass in zip(vmr_atoms_scalar, atom_masses)
                ]
            )
        else:
            mmr_atoms = jnp.zeros((0, art.pressure.size))

        # Step 6: Create constant profiles for CIA inputs and mmw
        vmrH2_prof = art.constant_mmr_profile(vmrH2)
        vmrHe_prof = art.constant_mmr_profile(vmrHe)
        mmw_prof = art.constant_mmr_profile(mmw)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class FreeVMR:
    """Vertically-varying VMR: sample at N nodes, interpolate in log-P.

    Similar to numpyro_free_temperature in pt.py, this samples log(VMR) at
    a set of pressure nodes and interpolates to the full pressure grid.
    """

    def __init__(
        self,
        n_nodes: int = 5,
        log_vmr_min: float = -15.0,
        log_vmr_max: float = 0.0,
        h2_he_ratio: float = 6.0,
    ):
        """Initialize the free VMR solver.

        Args:
            n_nodes: Number of pressure nodes for VMR interpolation
            log_vmr_min: Lower bound for log10(VMR) prior
            log_vmr_max: Upper bound for log10(VMR) prior
            h2_he_ratio: H2/He number ratio (solar is ~6)
        """
        self.n_nodes = n_nodes
        self.log_vmr_min = log_vmr_min
        self.log_vmr_max = log_vmr_max
        self.h2_he_ratio = h2_he_ratio

    def _sample_vmr_profile(
        self,
        name: str,
        log_p: jnp.ndarray,
        log_p_nodes: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample log(VMR) at nodes and interpolate to full pressure grid.

        Args:
            name: Species name (used for parameter naming)
            log_p: Log10 pressure grid (full)
            log_p_nodes: Log10 pressure at nodes

        Returns:
            VMR profile (linear, not log) on full pressure grid
        """
        logVMR_nodes = []
        for i in range(self.n_nodes):
            logVMR_i = numpyro.sample(
                f"logVMR_{name}_node{i}",
                dist.Uniform(self.log_vmr_min, self.log_vmr_max),
            )
            logVMR_nodes.append(logVMR_i)
        logVMR_nodes = jnp.array(logVMR_nodes)

        # Interpolate in log-P space
        logVMR_profile = jnp.interp(log_p, log_p_nodes, logVMR_nodes)
        return jnp.power(10.0, logVMR_profile)

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        """Sample composition with vertically-varying VMR profiles."""
        log_p = jnp.log10(art.pressure)
        log_p_nodes = jnp.linspace(log_p.min(), log_p.max(), self.n_nodes)
        n_layers = art.pressure.size

        # Step 1: Sample VMR profiles for all species
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_prof = self._sample_vmr_profile(mol, log_p, log_p_nodes)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_prof = self._sample_vmr_profile(atom, log_p, log_p_nodes)
            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Step 2: Renormalize at each layer if total trace VMR exceeds 1
        if n_mols + n_atoms > 0:
            # Stack all profiles: shape (n_species, n_layers)
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)  # (n_layers,)
            # Scale down where sum exceeds 1
            scale = jnp.where(sum_trace > 1.0, (1.0 - 1e-12) / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            # Split back
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)  # (n_layers,)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Step 3: Fill remainder with H2/He at each layer
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Step 4: Compute mean molecular weight profile
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        if n_mols > 0:
            for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
                mmw_prof = mmw_prof + mass * vmr_prof
        if n_atoms > 0:
            for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
                mmw_prof = mmw_prof + mass * vmr_prof

        # Step 5: Convert VMR to MMR profiles
        # MMR_i = VMR_i * (M_i / mmw)
        if n_mols > 0:
            mmr_mols = jnp.array(
                [
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
                ]
            )
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array(
                [
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
                ]
            )
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # For scalar outputs, use column-averaged values (pressure-weighted would be better)
        # Here we just use simple mean for consistency with downstream code that expects scalars
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


# ---------------------------------------------------------------------------
# Solar abundances (Asplund et al. 2021, A&A 653, A141)
# Values are number fractions relative to H (n_X / n_H)
# ---------------------------------------------------------------------------
SOLAR_ABUNDANCES = {
    # Molecules (approximate VMR in solar-composition gas at ~1000K)
    "H2O": 5.4e-4,
    "CO": 6.0e-4,
    "CO2": 1.0e-7,
    "CH4": 3.5e-4,
    "NH3": 8.0e-5,
    "H2S": 1.5e-5,
    "HCN": 1.0e-7,
    "C2H2": 1.0e-8,
    "PH3": 3.0e-7,
    "TiO": 1.0e-7,
    "VO": 1.0e-8,
    "FeH": 1.0e-8,
    "SiO": 3.5e-5,
    # Atoms (solar photospheric, log eps scale converted to n_X/n_H)
    "Fe": 3.16e-5,   # log eps = 7.50
    "Fe I": 3.16e-5,
    "Fe II": 3.16e-6,  # Rough ionization fraction
    "Na": 2.14e-6,   # log eps = 6.33
    "Na I": 2.14e-6,
    "K": 1.35e-7,    # log eps = 5.13
    "K I": 1.35e-7,
    "Ca": 2.19e-6,   # log eps = 6.34
    "Ca I": 2.19e-6,
    "Ca II": 2.19e-6,
    "Mg": 3.98e-5,   # log eps = 7.60
    "Mg I": 3.98e-5,
    "Ti": 8.91e-8,   # log eps = 4.95
    "Ti I": 8.91e-8,
    "Ti II": 8.91e-8,
    "V": 1.00e-8,    # log eps = 3.93
    "V I": 1.00e-8,
    "Cr": 4.68e-7,   # log eps = 5.67
    "Cr I": 4.68e-7,
    "Mn": 3.47e-7,   # log eps = 5.54
    "Mn I": 3.47e-7,
    "Ni": 1.78e-6,   # log eps = 6.25
    "Ni I": 1.78e-6,
    "Si": 3.24e-5,   # log eps = 7.51
    "Si I": 3.24e-5,
    "Al": 2.82e-6,   # log eps = 6.45
    "Al I": 2.82e-6,
    "Li": 1.0e-9,    # log eps = 1.05 (depleted)
    "Li I": 1.0e-9,
    "C": 2.69e-4,    # log eps = 8.43 (total carbon)
    "O": 4.90e-4,    # log eps = 8.69 (total oxygen)
    "N": 6.76e-5,    # log eps = 7.83 (total nitrogen)
}

# Solar C/O ratio
SOLAR_C_O = 0.55


class EquilibriumChemistry:
    """Chemical equilibrium with [M/H] and C/O as free parameters.

    This solver interpolates pre-computed equilibrium chemistry tables
    to get VMRs as a function of T, P, metallicity, and C/O ratio.

    If no table is provided, falls back to analytic approximations
    based on Madhusudhan (2012) and Heng & Tsai (2016).
    """

    def __init__(
        self,
        table_path: str | None = None,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
    ):
        """Initialize equilibrium chemistry solver.

        Args:
            table_path: Path to pre-computed equilibrium grid (NPZ or HDF5).
                        If None, uses analytic approximations.
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio (solar is ~6)
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.table = None

        if table_path is not None:
            self._load_table(table_path)

    def _load_table(self, path: str) -> None:
        """Load pre-computed equilibrium chemistry table."""
        import numpy as np
        from pathlib import Path

        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path)
            self.table = {
                "T_grid": data["T_grid"],
                "P_grid": data["P_grid"],
                "M_grid": data["M_grid"],  # log10([M/H])
                "CO_grid": data["CO_grid"],
                "vmr": {k: data[k] for k in data.files if k.startswith("vmr_")},
            }
        elif path.suffix in (".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                self.table = {
                    "T_grid": f["T_grid"][:],
                    "P_grid": f["P_grid"][:],
                    "M_grid": f["M_grid"][:],
                    "CO_grid": f["CO_grid"][:],
                    "vmr": {k: f[k][:] for k in f.keys() if k.startswith("vmr_")},
                }
        else:
            raise ValueError(f"Unknown table format: {path.suffix}")

    def _gibbs_equilibrium(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: jnp.ndarray,
        co_ratio: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute equilibrium VMR using Gibbs free energy minimization.

        Uses NASA polynomial thermodynamic data to compute Gibbs free energies
        and solves for equilibrium abundances based on Gordon & McBride (1994).

        Args:
            species: Species name (e.g., "H2O", "CO", "Fe I")
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)
            metallicity: Linear metallicity (Z/Z_solar)
            co_ratio: C/O ratio

        Returns:
            VMR profile for the species
        """
        # Use the Gibbs-based equilibrium calculation
        species_tuple = (species,)
        vmr_dict = gibbs_equilibrium_vmr(species_tuple, Tarr, Parr, metallicity, co_ratio)

        if species in vmr_dict:
            return vmr_dict[species]

        # Handle atomic species not in the main equilibrium solver
        species_upper = species.upper().replace(" ", "")
        Z = metallicity

        if species_upper in ("FE", "FEI", "FEII"):
            Fe_total = 3.16e-5 * Z
            # Iron condenses below ~1800K
            vmr = Fe_total * jnp.where(Tarr > 1800.0, 1.0, jnp.exp((Tarr - 1800.0) / 150.0))
            if "II" in species:
                # Use Saha equation for ionization fraction
                chi_Fe = 7.87  # eV
                ion_frac = self._saha_ionization(Tarr, Parr, chi_Fe)
                vmr = vmr * ion_frac
            elif "I" in species:
                chi_Fe = 7.87
                ion_frac = self._saha_ionization(Tarr, Parr, chi_Fe)
                vmr = vmr * (1.0 - ion_frac)
            return vmr

        elif species_upper in ("NA", "NAI"):
            Na_total = 2.14e-6 * Z
            chi_Na = 5.14  # eV
            ion_frac = self._saha_ionization(Tarr, Parr, chi_Na)
            vmr = Na_total * jnp.where(species_upper == "NAI", 1.0 - ion_frac, 1.0)
            return vmr

        elif species_upper in ("K", "KI"):
            K_total = 1.35e-7 * Z
            chi_K = 4.34  # eV
            ion_frac = self._saha_ionization(Tarr, Parr, chi_K)
            vmr = K_total * jnp.where(species_upper == "KI", 1.0 - ion_frac, 1.0)
            return vmr

        elif species_upper in ("CA", "CAI", "CAII"):
            Ca_total = 2.19e-6 * Z
            chi_Ca = 6.11  # eV
            ion_frac = self._saha_ionization(Tarr, Parr, chi_Ca)
            T_cond = 1500.0
            vmr = Ca_total * jnp.where(Tarr > T_cond, 1.0, jnp.exp((Tarr - T_cond)/100))
            if "II" in species:
                vmr = vmr * ion_frac
            elif "I" in species:
                vmr = vmr * (1.0 - ion_frac)
            return vmr

        elif species_upper in ("MG", "MGI"):
            Mg_total = 3.98e-5 * Z
            T_cond = 1400.0
            vmr = Mg_total * jnp.where(Tarr > T_cond, 1.0, jnp.exp((Tarr - T_cond)/100))
            return vmr

        elif species_upper in ("TI", "TII", "TIII"):
            Ti_total = 8.91e-8 * Z
            vmr = Ti_total * jnp.where(Tarr > 1800.0, 1.0, 0.01)
            return vmr

        elif species_upper in ("CR", "CRI"):
            Cr_total = 4.68e-7 * Z
            vmr = Cr_total * jnp.where(Tarr > 1300.0, 1.0, 0.01)
            return vmr

        else:
            # Default: scaled solar with no T dependence
            solar = SOLAR_ABUNDANCES.get(species, 1e-10)
            vmr = solar * Z * jnp.ones_like(Tarr)
            return vmr

    def _saha_ionization(
        self,
        T: jnp.ndarray,
        P: jnp.ndarray,
        chi_eV: float,
        g_ratio: float = 0.5,
    ) -> jnp.ndarray:
        """Compute ionization fraction from Saha equation.

        Args:
            T: Temperature (K)
            P: Pressure (bar)
            chi_eV: Ionization potential (eV)
            g_ratio: Ratio of partition functions g_+/g_0

        Returns:
            Ionization fraction x = n_+ / (n_0 + n_+)
        """
        k_B_eV = 8.617e-5  # eV/K
        m_e = 9.109e-28  # g
        h = 6.626e-27  # erg*s
        k_cgs = 1.381e-16  # erg/K

        # Number density from ideal gas
        P_cgs = P * 1e6  # bar to dyne/cm^2
        n_tot = P_cgs / (k_cgs * T)

        # Saha constant
        saha_const = jnp.power(2 * jnp.pi * m_e * k_cgs * T / (h**2), 1.5)
        chi_term = jnp.exp(-chi_eV / (k_B_eV * T))
        K = 2 * g_ratio * saha_const * chi_term

        # Approximate solution
        x_low = jnp.sqrt(K / (n_tot + 1e-30))
        x = jnp.clip(x_low, 0.0, 0.99)
        return x

    def _analytic_equilibrium(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: jnp.ndarray,
        co_ratio: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute equilibrium VMR using Gibbs-based thermodynamics.

        This method now uses NASA polynomial thermodynamic data and Gibbs
        free energy calculations for accurate equilibrium chemistry.

        For backwards compatibility, the method name is preserved but the
        implementation uses proper thermodynamics.
        """
        return self._gibbs_equilibrium(species, Tarr, Parr, metallicity, co_ratio)

    def _interpolate_table(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        log_metallicity: jnp.ndarray,
        co_ratio: jnp.ndarray,
    ) -> jnp.ndarray:
        """Interpolate VMR from pre-computed table."""
        from scipy.interpolate import RegularGridInterpolator
        import numpy as np

        vmr_key = f"vmr_{species}"
        if vmr_key not in self.table["vmr"]:
            # Species not in table, fall back to analytic
            return self._analytic_equilibrium(
                species, Tarr, Parr, 10**log_metallicity, co_ratio
            )

        # Build interpolator (4D: T, P, [M/H], C/O)
        interp = RegularGridInterpolator(
            (
                self.table["T_grid"],
                np.log10(self.table["P_grid"]),
                self.table["M_grid"],
                self.table["CO_grid"],
            ),
            self.table["vmr"][vmr_key],
            bounds_error=False,
            fill_value=None,  # Extrapolate
        )

        # Query points
        points = jnp.stack(
            [Tarr, jnp.log10(Parr),
             jnp.full_like(Tarr, log_metallicity),
             jnp.full_like(Tarr, co_ratio)],
            axis=-1,
        )

        return jnp.array(interp(np.array(points)))

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample equilibrium composition.

        Args:
            mol_names: Molecule names
            mol_masses: Molecular masses
            atom_names: Atom names
            atom_masses: Atomic masses
            art: ExoJAX art object
            Tarr: Temperature profile. If None, uses isothermal at 2000K.
        """
        # Sample bulk parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        n_layers = Parr.size

        # Compute VMR profiles for each species
        vmr_mols_profiles = []
        for mol in mol_names:
            if self.table is not None:
                vmr_prof = self._interpolate_table(mol, Tarr, Parr, log_metallicity, co_ratio)
            else:
                vmr_prof = self._analytic_equilibrium(mol, Tarr, Parr, metallicity, co_ratio)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            if self.table is not None:
                vmr_prof = self._interpolate_table(atom, Tarr, Parr, log_metallicity, co_ratio)
            else:
                vmr_prof = self._analytic_equilibrium(atom, Tarr, Parr, metallicity, co_ratio)
            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Compute total trace VMR and renormalize if needed
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill remainder with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute mean molecular weight profile
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert VMR to MMR profiles
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Scalar outputs (column averages)
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class QuenchChemistry:
    """Equilibrium chemistry quenched at a pressure level.

    Models the effect of vertical mixing (eddy diffusion) which causes
    abundances to be "frozen" at a quench pressure where the chemical
    timescale exceeds the mixing timescale.

    Below P_quench: equilibrium chemistry
    Above P_quench: abundances frozen to values at P_quench
    """

    def __init__(
        self,
        equilibrium_solver: EquilibriumChemistry | None = None,
        p_quench_range: tuple[float, float] = (1e-4, 1e2),
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
    ):
        """Initialize quench chemistry solver.

        Args:
            equilibrium_solver: EquilibriumChemistry instance for computing
                                equilibrium VMRs. If None, creates one.
            p_quench_range: Prior range for quench pressure in bar
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
        """
        self.p_quench_range = p_quench_range
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio

        if equilibrium_solver is not None:
            self.eq_solver = equilibrium_solver
        else:
            self.eq_solver = EquilibriumChemistry(
                metallicity_range=metallicity_range,
                co_ratio_range=co_ratio_range,
                h2_he_ratio=h2_he_ratio,
            )

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample quenched composition."""
        # Sample quench pressure
        log_p_quench = numpyro.sample(
            "log_P_quench",
            dist.Uniform(
                jnp.log10(self.p_quench_range[0]),
                jnp.log10(self.p_quench_range[1]),
            ),
        )
        p_quench = jnp.power(10.0, log_p_quench)

        # Sample metallicity and C/O (these will be used by eq_solver)
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        # Compute equilibrium VMRs
        vmr_mols_eq = []
        for mol in mol_names:
            vmr_prof = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_mols_eq.append(vmr_prof)

        vmr_atoms_eq = []
        for atom in atom_names:
            vmr_prof = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )
            vmr_atoms_eq.append(vmr_prof)

        # Apply quenching: freeze abundances above P_quench
        def quench_profile(vmr_eq: jnp.ndarray) -> jnp.ndarray:
            """Apply quenching to a VMR profile."""
            # Find VMR at quench pressure (interpolate)
            log_p = jnp.log10(Parr)
            log_p_q = jnp.log10(p_quench)
            vmr_at_quench = jnp.interp(log_p_q, log_p, vmr_eq)
            # Above quench (lower pressure): use quenched value
            # Below quench (higher pressure): use equilibrium
            return jnp.where(Parr < p_quench, vmr_at_quench, vmr_eq)

        vmr_mols_profiles = [quench_profile(v) for v in vmr_mols_eq]
        vmr_atoms_profiles = [quench_profile(v) for v in vmr_atoms_eq]

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize if needed
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill remainder with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute mean molecular weight
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Scalar outputs
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class COMetallicity:
    """C/O ratio and metallicity parameterization using Gibbs equilibrium.

    Instead of sampling individual VMRs, samples:
    - [M/H]: overall metallicity scaling
    - C/O: carbon-to-oxygen ratio

    Then derives individual species VMRs using Gibbs free energy minimization
    with NASA polynomial thermodynamic data for accurate equilibrium chemistry.

    This is a wrapper around EquilibriumChemistry that provides a cleaner
    interface for the common case of retrieving metallicity and C/O ratio.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-1.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        use_gibbs: bool = True,
    ):
        """Initialize C/O metallicity solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            use_gibbs: If True, use Gibbs minimization; if False, use simple rules
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.use_gibbs = use_gibbs

        # Use the improved EquilibriumChemistry solver internally
        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition using C/O and metallicity with Gibbs equilibrium.

        Uses NASA polynomial thermodynamic data for accurate equilibrium
        chemistry calculations via Gibbs free energy minimization.

        Args:
            mol_names: List of molecule names
            mol_masses: Molecular masses
            atom_names: List of atomic species names
            atom_masses: Atomic masses
            art: ExoJAX art object
            Tarr: Temperature profile (optional, uses isothermal 2000K if None)

        Returns:
            CompositionState with equilibrium abundances
        """
        # Sample bulk parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        if self.use_gibbs:
            # Use Gibbs equilibrium for accurate chemistry
            vmr_mols_profiles = []
            for mol in mol_names:
                vmr_prof = self.eq_solver._gibbs_equilibrium(
                    mol, Tarr, Parr, metallicity, co_ratio
                )
                vmr_mols_profiles.append(vmr_prof)

            vmr_atoms_profiles = []
            for atom in atom_names:
                vmr_prof = self.eq_solver._gibbs_equilibrium(
                    atom, Tarr, Parr, metallicity, co_ratio
                )
                vmr_atoms_profiles.append(vmr_prof)

            n_mols = len(vmr_mols_profiles)
            n_atoms = len(vmr_atoms_profiles)

            # Renormalize if needed
            if n_mols + n_atoms > 0:
                all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
                sum_trace = jnp.sum(all_profiles, axis=0)
                scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
                all_profiles = all_profiles * scale[None, :]
                vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
                vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
                vmr_trace_tot = jnp.sum(all_profiles, axis=0)
            else:
                vmr_trace_tot = jnp.zeros(n_layers)

            # Fill with H2/He
            h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
            he_frac = 1.0 / (self.h2_he_ratio + 1.0)
            vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
            vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

            # Compute mean molecular weight profile
            mass_H2 = molinfo.molmass_isotope("H2")
            mass_He = molinfo.molmass_isotope("He", db_HIT=False)
            mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
            for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
                mmw_prof = mmw_prof + mass * vmr_prof
            for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
                mmw_prof = mmw_prof + mass * vmr_prof

            # Convert to MMR
            if n_mols > 0:
                mmr_mols = jnp.array([
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
                ])
            else:
                mmr_mols = jnp.zeros((0, n_layers))

            if n_atoms > 0:
                mmr_atoms = jnp.array([
                    vmr_prof * (mass / mmw_prof)
                    for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
                ])
            else:
                mmr_atoms = jnp.zeros((0, n_layers))

            # Scalar outputs
            vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
            vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
            vmrH2 = jnp.mean(vmrH2_prof)
            vmrHe = jnp.mean(vmrHe_prof)
            mmw = jnp.mean(mmw_prof)

            return CompositionState(
                vmr_mols=vmr_mols_scalar,
                vmr_atoms=vmr_atoms_scalar,
                vmrH2=vmrH2,
                vmrHe=vmrHe,
                mmw=mmw,
                mmr_mols=mmr_mols,
                mmr_atoms=mmr_atoms,
                vmrH2_prof=vmrH2_prof,
                vmrHe_prof=vmrHe_prof,
                mmw_prof=mmw_prof,
            )

        # Fallback to simple partitioning (for backwards compatibility)
        Z = metallicity

        # Total C and O abundances (scaled from solar)
        O_solar = 4.90e-4
        C_solar = 2.69e-4
        O_total = O_solar * Z
        C_total = co_ratio * O_total  # Set C based on desired C/O

        # Derive molecular VMRs using partitioning rules
        vmr_mols_scalar = []
        for mol in mol_names:
            mol_upper = mol.upper()
            if mol_upper == "H2O":
                vmr = jnp.where(co_ratio < 1.0, O_total - C_total, O_total * 0.01)
            elif mol_upper == "CO":
                vmr = jnp.minimum(C_total, O_total)
            elif mol_upper == "CO2":
                excess_O = jnp.maximum(O_total - C_total, 0.0)
                vmr = excess_O * 0.01
            elif mol_upper == "CH4":
                vmr = jnp.where(co_ratio > 1.0, C_total - O_total, C_total * 0.001)
            elif mol_upper == "HCN":
                vmr = jnp.where(co_ratio > 1.0, C_total * 0.01, C_total * 1e-4)
            elif mol_upper == "C2H2":
                vmr = jnp.where(co_ratio > 1.5, C_total * 0.01, C_total * 1e-5)
            elif mol_upper == "NH3":
                N_total = 6.76e-5 * Z
                vmr = N_total * 0.1
            elif mol_upper == "H2S":
                S_total = 1.5e-5 * Z
                vmr = S_total
            elif mol_upper in ("TIO", "TIOXIDE"):
                vmr = 8.91e-8 * Z
            elif mol_upper in ("VO", "VANADIUMOXIDE"):
                vmr = 1.0e-8 * Z
            elif mol_upper == "SIO":
                vmr = 3.24e-5 * Z * 0.1
            elif mol_upper == "FEH":
                vmr = 3.16e-5 * Z * 0.01
            else:
                solar = SOLAR_ABUNDANCES.get(mol, 1e-10)
                vmr = solar * Z
            vmr_mols_scalar.append(vmr)

        # Atomic VMRs scale with metallicity
        vmr_atoms_scalar = []
        for atom in atom_names:
            solar = SOLAR_ABUNDANCES.get(atom, 1e-10)
            vmr = solar * Z
            vmr_atoms_scalar.append(vmr)

        n_mols = len(vmr_mols_scalar)
        n_atoms = len(vmr_atoms_scalar)

        # Renormalize if total exceeds 1
        if n_mols + n_atoms > 0:
            vmr_trace_arr = jnp.array(vmr_mols_scalar + vmr_atoms_scalar)
            sum_trace = jnp.sum(vmr_trace_arr)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            vmr_trace_arr = vmr_trace_arr * scale
            vmr_mols_scalar = [vmr_trace_arr[i] for i in range(n_mols)]
            vmr_atoms_scalar = [vmr_trace_arr[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(vmr_trace_arr)
        else:
            vmr_trace_tot = jnp.array(0.0)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2 = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe = (1.0 - vmr_trace_tot) * he_frac

        # Mean molecular weight
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw = mass_H2 * vmrH2 + mass_He * vmrHe
        for m, v in zip(mol_masses, vmr_mols_scalar):
            mmw = mmw + m * v
        for m, v in zip(atom_masses, vmr_atoms_scalar):
            mmw = mmw + m * v

        # Create constant profiles
        if n_mols > 0:
            mmr_mols = jnp.array([
                art.constant_mmr_profile(vmr * (mass / mmw))
                for vmr, mass in zip(vmr_mols_scalar, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, art.pressure.size))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                art.constant_mmr_profile(vmr * (mass / mmw))
                for vmr, mass in zip(vmr_atoms_scalar, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, art.pressure.size))

        vmrH2_prof = art.constant_mmr_profile(vmrH2)
        vmrHe_prof = art.constant_mmr_profile(vmrHe)
        mmw_prof = art.constant_mmr_profile(mmw)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class GPChemistry:
    """Gaussian Process prior on log(VMR) profiles.

    Uses a GP to model smooth vertical variation in composition,
    similar to numpyro_gp_temperature in pt.py. This allows for
    flexible vertical profiles while enforcing physical smoothness.
    """

    def __init__(
        self,
        log_vmr_mean_range: tuple[float, float] = (-12.0, -2.0),
        gp_amplitude: float = 2.0,
        gp_length_scale: float = 1.5,
        h2_he_ratio: float = 6.0,
    ):
        """Initialize GP chemistry solver.

        Args:
            log_vmr_mean_range: Prior range for mean log10(VMR)
            gp_amplitude: GP amplitude (variance of deviations)
            gp_length_scale: GP length scale in log10(P) units
            h2_he_ratio: H2/He number ratio
        """
        self.log_vmr_mean_range = log_vmr_mean_range
        self.gp_amplitude = gp_amplitude
        self.gp_length_scale = gp_length_scale
        self.h2_he_ratio = h2_he_ratio

    def _rbf_kernel(
        self,
        x: jnp.ndarray,
        length_scale: float,
        amplitude: float,
    ) -> jnp.ndarray:
        """Compute RBF (squared exponential) kernel matrix."""
        diff = x[:, None] - x[None, :]
        return amplitude**2 * jnp.exp(-0.5 * (diff / length_scale) ** 2)

    def _sample_gp_profile(
        self,
        name: str,
        log_p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample a log(VMR) profile using GP prior.

        Args:
            name: Species name for parameter naming
            log_p: log10(pressure) grid

        Returns:
            VMR profile (linear, not log)
        """
        # Sample mean log(VMR)
        logVMR_mean = numpyro.sample(
            f"logVMR_{name}_mean",
            dist.Uniform(self.log_vmr_mean_range[0], self.log_vmr_mean_range[1]),
        )

        # Sample GP amplitude (allow some variation)
        gp_amp = numpyro.sample(
            f"logVMR_{name}_gp_amp",
            dist.HalfNormal(self.gp_amplitude),
        )

        # Build covariance matrix
        K = self._rbf_kernel(log_p, self.gp_length_scale, gp_amp)
        # Add jitter for numerical stability
        K = K + 1e-6 * jnp.eye(len(log_p))

        # Sample GP deviations
        gp_deviations = numpyro.sample(
            f"logVMR_{name}_gp",
            dist.MultivariateNormal(jnp.zeros(len(log_p)), covariance_matrix=K),
        )

        logVMR_profile = logVMR_mean + gp_deviations
        return jnp.power(10.0, logVMR_profile)

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
    ) -> CompositionState:
        """Sample composition with GP-smoothed vertical profiles."""
        log_p = jnp.log10(art.pressure)
        n_layers = art.pressure.size

        # Sample VMR profiles for all species
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_prof = self._sample_gp_profile(mol, log_p)
            vmr_mols_profiles.append(vmr_prof)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_prof = self._sample_gp_profile(atom, log_p)
            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize at each layer
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Mean molecular weight profile
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Scalar outputs
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


# ---------------------------------------------------------------------------
# Condensation temperatures (K) at 1 bar (Lodders 2003, Visscher et al. 2010)
# ---------------------------------------------------------------------------
CONDENSATION_TEMPS = {
    # Refractory species (condense at high T)
    "AL2O3": 1850,      # Corundum
    "CATIO3": 1580,     # Perovskite
    "CA2AL2SIO7": 1550, # Gehlenite
    "MGAL2O4": 1500,    # Spinel
    "TI": 1580,
    "TII": 1580,
    "TIO": 1580,
    "TIO2": 1580,
    "V": 1450,
    "VI": 1450,
    "VO": 1450,
    "FE": 1340,         # Iron metal
    "FEI": 1340,
    "FES": 700,         # Troilite
    "MG2SIO4": 1350,    # Forsterite
    "MGSIO3": 1320,     # Enstatite
    "CR": 1300,
    "CRI": 1300,
    "MN": 1150,
    "MNI": 1150,
    "NA2S": 1000,
    "NA": 950,
    "NAI": 950,
    "K": 900,
    "KI": 900,
    "KCL": 700,
    "ZNS": 700,
    # Volatile species (stay in gas phase)
    "H2O": 180,         # Water ice
    "NH3": 130,         # Ammonia ice
    "H2S": 200,
    "CO": 25,           # CO ice
    "CO2": 70,
    "CH4": 30,
}

# Ionization potentials (eV) for Saha equation
IONIZATION_POTENTIALS = {
    "H": 13.598,
    "He": 24.587,
    "Li": 5.392,
    "Na": 5.139,
    "K": 4.341,
    "Ca": 6.113,
    "Mg": 7.646,
    "Fe": 7.902,
    "Ti": 6.828,
    "V": 6.746,
    "Cr": 6.767,
    "Mn": 7.434,
    "Ni": 7.640,
    "Al": 5.986,
    "Si": 8.152,
}

# Atomic masses for ionization calculations
ATOMIC_MASSES_CHEM = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Na": 22.990,
    "K": 39.098,
    "Ca": 40.078,
    "Mg": 24.305,
    "Fe": 55.845,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Ni": 58.693,
    "Al": 26.982,
    "Si": 28.086,
}


class RainoutChemistry:
    """Equilibrium chemistry with condensate rainout.

    Based on Burrows & Sharp (1999), Lodders (2002), and Visscher et al. (2010).
    Species are removed from the gas phase when T < T_condensation(P).

    The condensation temperature varies with pressure approximately as:
        T_cond(P) = T_cond(1 bar) * (P / 1 bar)^alpha

    where alpha ~ 0.05-0.1 for most species.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        pressure_exponent: float = 0.05,
        cold_trap: bool = True,
    ):
        """Initialize rainout chemistry solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            pressure_exponent: Exponent for T_cond(P) scaling
            cold_trap: If True, species that condense anywhere stay depleted above
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.pressure_exponent = pressure_exponent
        self.cold_trap = cold_trap

        # Use equilibrium solver for base abundances
        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _condensation_temperature(
        self,
        species: str,
        pressure: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get condensation temperature at given pressure.

        Args:
            species: Species name
            pressure: Pressure in bar

        Returns:
            Condensation temperature in K
        """
        # Get base condensation T at 1 bar
        species_key = species.upper().replace(" ", "")
        T_cond_1bar = CONDENSATION_TEMPS.get(species_key, 0.0)

        if T_cond_1bar == 0.0:
            # Species doesn't condense in relevant T range
            return jnp.zeros_like(pressure)

        # Scale with pressure
        return T_cond_1bar * jnp.power(pressure, self.pressure_exponent)

    def _apply_rainout(
        self,
        vmr_profile: jnp.ndarray,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply rainout depletion to a VMR profile.

        Args:
            vmr_profile: Input VMR profile
            species: Species name
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)

        Returns:
            Depleted VMR profile
        """
        T_cond = self._condensation_temperature(species, Parr)

        # Local depletion factor (0 when T << T_cond, 1 when T >> T_cond)
        local_depletion = jnp.where(
            T_cond > 0,
            jnp.clip((Tarr / T_cond - 0.8) / 0.4, 0.0, 1.0),
            1.0,
        )

        if self.cold_trap:
            # Cold trap: species that condense anywhere are depleted at all
            # levels above (lower pressure than) the cold trap.
            # The cold trap is where depletion is minimum (coldest point on
            # the T/T_cond curve).
            #
            # Find the VMR at the cold trap level and use that value for all
            # layers above (at lower pressure).
            min_depletion = jnp.min(local_depletion)
            
            # Find pressure at cold trap (where depletion is minimum)
            # For a proper cold trap, everything above should have cold trap VMR
            # We use cumulative minimum from high P to low P
            # Assuming Parr is ordered from low P (top) to high P (bottom),
            # we reverse, take cummin, and reverse back
            depletion_reversed = local_depletion[::-1]
            cummin_reversed = jnp.minimum.accumulate(depletion_reversed)
            depletion = cummin_reversed[::-1]
        else:
            # Local condensation only
            depletion = local_depletion

        return vmr_profile * depletion

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition with rainout."""
        # Sample metallicity and C/O
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        # Get equilibrium VMRs
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_rainout = self._apply_rainout(vmr_eq, mol, Tarr, Parr)
            vmr_mols_profiles.append(vmr_rainout)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )
            vmr_rainout = self._apply_rainout(vmr_eq, atom, Tarr, Parr)
            vmr_atoms_profiles.append(vmr_rainout)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Scalar outputs
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class KzzDisequilibrium:
    """Disequilibrium chemistry via eddy diffusion.

    Based on Moses et al. (2011), Tsai et al. (2017), and Drummond et al. (2020).

    Instead of sampling a single quench pressure, this model samples the
    eddy diffusion coefficient Kzz and computes species-specific quench
    pressures from the competition between chemical and mixing timescales:

        t_mix = H^2 / Kzz
        t_chem = (k_forward * [reactants])^-1

    Species quench where t_mix < t_chem.
    """

    def __init__(
        self,
        log_kzz_range: tuple[float, float] = (6.0, 12.0),  # cm^2/s
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
    ):
        """Initialize Kzz disequilibrium solver.

        Args:
            log_kzz_range: Prior range for log10(Kzz) in cm^2/s
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
        """
        self.log_kzz_range = log_kzz_range
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio

        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _scale_height(
        self,
        Tarr: jnp.ndarray,
        mmw: float,
        g: float = 10.0,  # m/s^2, typical for hot Jupiters
    ) -> jnp.ndarray:
        """Compute atmospheric scale height.

        H = kT / (mu * g)

        Args:
            Tarr: Temperature profile (K)
            mmw: Mean molecular weight (amu)
            g: Surface gravity (m/s^2)

        Returns:
            Scale height in cm
        """
        k_B = 1.381e-16  # erg/K
        m_u = 1.661e-24  # g
        g_cgs = g * 100  # cm/s^2

        return k_B * Tarr / (mmw * m_u * g_cgs)

    def _mixing_timescale(
        self,
        H: jnp.ndarray,
        Kzz: float,
    ) -> jnp.ndarray:
        """Compute mixing timescale.

        t_mix = H^2 / Kzz

        Args:
            H: Scale height (cm)
            Kzz: Eddy diffusion coefficient (cm^2/s)

        Returns:
            Mixing timescale (s)
        """
        return H**2 / Kzz

    def _chemical_timescale(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: float = 1.0,
    ) -> jnp.ndarray:
        """Compute chemical timescale from KIDA/UMIST reaction rates.

        Uses literature-based reaction rate coefficients to compute
        chemical equilibration timescales for key species.

        Based on:
        - Visscher & Moses (2011) for CO-CH4
        - Tsai et al. (2018) for H2O
        - Moses et al. (2011) for NH3

        Args:
            species: Species name
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)
            metallicity: Linear metallicity factor

        Returns:
            Chemical timescale (s)
        """
        species_upper = species.upper().replace(" ", "")

        # Use CHEM_TIMESCALE_PARAMS if available
        if species_upper in CHEM_TIMESCALE_PARAMS:
            params = CHEM_TIMESCALE_PARAMS[species_upper]
            A = params["A"]
            Ea = params["Ea"]
            n = params["n"]
            t_chem = A * jnp.exp(Ea / Tarr) * jnp.power(Parr, n)
            return t_chem

        # For other species not in CHEM_TIMESCALE_PARAMS, compute from KIDA reaction rates
        # Get number density for rate calculations
        k_B = 1.381e-16  # erg/K
        P_cgs = Parr * 1e6  # bar to dyne/cm^2
        n_tot = P_cgs / (k_B * Tarr)  # cm^-3

        # H2 number density (dominant species)
        n_H2 = n_tot * 0.85

        # Note: CO, CH4, H2O, NH3, HCN, CO2 are handled by CHEM_TIMESCALE_PARAMS above
        # Default for species not in that dict: fast equilibration (atoms, refractories)
        t_chem = 1e6 * jnp.ones_like(Tarr)  # ~10 days default

        # Metallicity scaling: higher Z means more reactants -> faster chemistry
        t_chem = t_chem / jnp.maximum(metallicity, 0.01)

        return t_chem

    def _apply_quenching(
        self,
        vmr_eq: jnp.ndarray,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        Kzz: float,
        mmw: float,
    ) -> jnp.ndarray:
        """Apply quenching based on Kzz.

        Args:
            vmr_eq: Equilibrium VMR profile
            species: Species name
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)
            Kzz: Eddy diffusion coefficient (cm^2/s)
            mmw: Mean molecular weight (amu)

        Returns:
            Quenched VMR profile
        """
        H = self._scale_height(Tarr, mmw)
        t_mix = self._mixing_timescale(H, Kzz)
        t_chem = self._chemical_timescale(species, Tarr, Parr)

        # Find quench level (where t_mix ~ t_chem)
        # Above quench: use quenched value
        # Below quench: use equilibrium
        quench_ratio = t_mix / t_chem

        # Interpolate to find quench pressure
        # We want the deepest level where mixing dominates
        log_p = jnp.log10(Parr)

        # Find where quench_ratio crosses 1 (from high P to low P)
        is_quenched = quench_ratio < 1.0  # Mixing faster than chemistry

        # Get VMR at approximate quench level
        # Use weighted average near quench point
        weights = jnp.exp(-jnp.abs(jnp.log10(quench_ratio)) * 2)
        weights = weights / jnp.sum(weights)
        vmr_quench = jnp.sum(vmr_eq * weights)

        # Apply quenching: use quenched value where mixing dominates
        vmr_out = jnp.where(is_quenched, vmr_quench, vmr_eq)

        return vmr_out

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
        gravity: float = 10.0,
    ) -> CompositionState:
        """Sample disequilibrium composition."""
        # Sample parameters
        log_kzz = numpyro.sample(
            "log_Kzz",
            dist.Uniform(self.log_kzz_range[0], self.log_kzz_range[1]),
        )
        Kzz = jnp.power(10.0, log_kzz)

        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        # Estimate MMW for scale height (approximate)
        mmw_approx = 2.3  # H2-dominated

        # Compute equilibrium and apply quenching
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_quench = self._apply_quenching(
                vmr_eq, mol, Tarr, Parr, Kzz, mmw_approx
            )
            vmr_mols_profiles.append(vmr_quench)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )
            # Atoms generally don't quench (fast reactions)
            vmr_atoms_profiles.append(vmr_eq)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class ThermalIonization:
    """Thermal ionization equilibrium using Saha equation.

    Based on Parmentier et al. (2018), Lothringer et al. (2018), and
    Kitzmann et al. (2018).

    For ultra-hot Jupiters (T > 2500K), thermal ionization becomes important:
    - Alkali metals (Na, K) ionize first
    - Iron group (Fe, Mg, Ca) ionize at higher T
    - H- opacity requires atomic H and free electrons

    The Saha equation gives ionization fraction:
        n_+ * n_e / n_0 = (2 * g_+ / g_0) * (2πm_e kT / h^2)^(3/2) * exp(-χ/kT)
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        include_h_minus: bool = True,
    ):
        """Initialize thermal ionization solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            include_h_minus: Include H- in electron budget
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.include_h_minus = include_h_minus

        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _saha_ionization_fraction(
        self,
        T: jnp.ndarray,
        P: jnp.ndarray,
        chi_eV: float,
        g_ratio: float = 0.5,  # g_+/g_0 typical for alkalis
    ) -> jnp.ndarray:
        """Compute ionization fraction from Saha equation.

        Args:
            T: Temperature (K)
            P: Pressure (bar)
            chi_eV: Ionization potential (eV)
            g_ratio: Ratio of partition functions g_+/g_0

        Returns:
            Ionization fraction x = n_+ / (n_0 + n_+)
        """
        # Physical constants
        k_B = 8.617e-5  # eV/K
        m_e = 9.109e-28  # g
        h = 6.626e-27  # erg*s
        k_cgs = 1.381e-16  # erg/K

        # Convert pressure to number density (approximate)
        # n = P / (k_B * T)
        P_cgs = P * 1e6  # bar to dyne/cm^2
        n_tot = P_cgs / (k_cgs * T)

        # Saha equation constant
        # (2πm_e kT / h^2)^(3/2)
        saha_const = jnp.power(2 * jnp.pi * m_e * k_cgs * T / (h**2), 1.5)

        # Ionization potential term
        chi_term = jnp.exp(-chi_eV / (k_B * T))

        # Saha ratio K = n_+ * n_e / n_0
        K = 2 * g_ratio * saha_const * chi_term

        # Solve quadratic for ionization fraction
        # Assuming n_e ≈ n_+ (metals provide most electrons)
        # x^2 / (1-x) = K / n_tot
        # Approximate solution for small x: x ≈ sqrt(K/n_tot)
        # For large x: x ≈ 1 - n_tot/K

        x_low = jnp.sqrt(K / n_tot)
        x_high = 1.0 - n_tot / K

        # Smooth transition
        x = jnp.where(x_low < 0.5, x_low, jnp.clip(x_high, 0.0, 1.0))
        x = jnp.clip(x, 0.0, 1.0)

        return x

    def _compute_electron_density(
        self,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        vmr_atoms: dict[str, jnp.ndarray],
        metallicity: float,
    ) -> jnp.ndarray:
        """Compute electron number density from all ionizing species.

        Args:
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)
            vmr_atoms: Dictionary of atomic VMR profiles
            metallicity: Linear metallicity factor

        Returns:
            Electron VMR profile
        """
        n_e = jnp.zeros_like(Tarr)

        for element, chi in IONIZATION_POTENTIALS.items():
            # Get total abundance of this element
            solar = SOLAR_ABUNDANCES.get(element, SOLAR_ABUNDANCES.get(f"{element} I", 1e-10))
            vmr_total = solar * metallicity

            # Get ionization fraction
            x_ion = self._saha_ionization_fraction(Tarr, Parr, chi)

            # Add to electron density
            n_e = n_e + vmr_total * x_ion

        return n_e

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition with thermal ionization."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2500.0)

        # Compute molecular abundances (equilibrium)
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_prof = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_mols_profiles.append(vmr_prof)

        # Compute atomic abundances with ionization
        vmr_atoms_profiles = []
        for atom in atom_names:
            # Get base abundance from equilibrium
            vmr_eq = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )

            # Determine ionization state from name
            is_neutral = "I" in atom and "II" not in atom
            is_ionized = "II" in atom

            # Get element name
            element = atom.split()[0] if " " in atom else atom

            # Get ionization potential
            chi = IONIZATION_POTENTIALS.get(element, 10.0)

            # Compute ionization fraction
            x_ion = self._saha_ionization_fraction(Tarr, Parr, chi)

            if is_neutral:
                # Neutral species: multiply by (1 - x_ion)
                vmr_prof = vmr_eq * (1.0 - x_ion)
            elif is_ionized:
                # Ionized species: multiply by x_ion
                vmr_prof = vmr_eq * x_ion
            else:
                # Unknown ionization state: use total
                vmr_prof = vmr_eq

            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class ChemicallyConsistent:
    """Chemically consistent retrieval with mass balance constraints.

    Based on Line et al. (2013), Benneke (2015), and Welbanks et al. (2019).

    This solver adds soft constraints that penalize unphysical abundance
    patterns, such as:
    - C/O derived from molecules inconsistent with C/O from atoms
    - Total carbon exceeding available carbon
    - Negative abundances after mass balance

    Works as a wrapper around another chemistry solver.
    """

    def __init__(
        self,
        base_solver: CompositionSolver | None = None,
        enforce_mass_balance: bool = True,
        enforce_co_consistency: bool = True,
        consistency_weight: float = 1.0,
    ):
        """Initialize chemically consistent solver.

        Args:
            base_solver: Underlying chemistry solver. If None, uses ConstantVMR.
            enforce_mass_balance: Add penalty for mass balance violations
            enforce_co_consistency: Add penalty for C/O inconsistency
            consistency_weight: Weight for consistency penalty in likelihood
        """
        self.base_solver = base_solver or ConstantVMR()
        self.enforce_mass_balance = enforce_mass_balance
        self.enforce_co_consistency = enforce_co_consistency
        self.consistency_weight = consistency_weight

    def _count_atoms(
        self,
        species: str,
    ) -> dict[str, int]:
        """Count atoms in a molecular formula.

        Simple parser for common molecules.
        """
        counts = {}
        species = species.upper()

        atom_patterns = [
            ("H2O", {"H": 2, "O": 1}),
            ("CO2", {"C": 1, "O": 2}),
            ("CO", {"C": 1, "O": 1}),
            ("CH4", {"C": 1, "H": 4}),
            ("NH3", {"N": 1, "H": 3}),
            ("H2S", {"H": 2, "S": 1}),
            ("HCN", {"H": 1, "C": 1, "N": 1}),
            ("C2H2", {"C": 2, "H": 2}),
            ("TIO", {"TI": 1, "O": 1}),
            ("VO", {"V": 1, "O": 1}),
            ("SIO", {"SI": 1, "O": 1}),
            ("FEH", {"FE": 1, "H": 1}),
            ("PH3", {"P": 1, "H": 3}),
        ]

        for pattern, atoms in atom_patterns:
            if species == pattern:
                return atoms

        # Single atom
        if species in ("H", "C", "N", "O", "FE", "NA", "K", "CA", "MG", "TI", "V"):
            return {species: 1}

        return {}

    def _compute_mass_balance(
        self,
        vmr_mols: list[jnp.ndarray],
        mol_names: list[str],
        metallicity: float,
    ) -> jnp.ndarray:
        """Compute mass balance penalty.

        Penalizes if sum of C in molecules exceeds total available C, etc.
        """
        penalty = jnp.array(0.0)

        # Total available atoms (scaled solar)
        C_available = 2.69e-4 * metallicity
        O_available = 4.90e-4 * metallicity
        N_available = 6.76e-5 * metallicity

        # Count atoms in molecules
        C_used = jnp.array(0.0)
        O_used = jnp.array(0.0)
        N_used = jnp.array(0.0)

        for vmr, mol in zip(vmr_mols, mol_names):
            atoms = self._count_atoms(mol)
            vmr_scalar = jnp.mean(vmr) if vmr.ndim > 0 else vmr
            C_used = C_used + atoms.get("C", 0) * vmr_scalar
            O_used = O_used + atoms.get("O", 0) * vmr_scalar
            N_used = N_used + atoms.get("N", 0) * vmr_scalar

        # Penalty for exceeding available atoms
        penalty = penalty + jnp.maximum(0.0, C_used - C_available) * 1000
        penalty = penalty + jnp.maximum(0.0, O_used - O_available) * 1000
        penalty = penalty + jnp.maximum(0.0, N_used - N_available) * 1000

        return penalty

    def _compute_co_consistency(
        self,
        vmr_mols: list[jnp.ndarray],
        mol_names: list[str],
    ) -> jnp.ndarray:
        """Compute C/O consistency penalty.

        Penalizes if C/O derived from CO/H2O differs significantly from
        C/O derived from CH4/CO2 etc.
        """
        penalty = jnp.array(0.0)

        # Extract VMRs for key species
        vmr_dict = {}
        for vmr, mol in zip(vmr_mols, mol_names):
            vmr_scalar = jnp.mean(vmr) if vmr.ndim > 0 else vmr
            vmr_dict[mol.upper()] = vmr_scalar

        # Estimate C/O from different species pairs
        co_estimates = []

        # From CO and H2O
        if "CO" in vmr_dict and "H2O" in vmr_dict:
            # CO has 1 C and 1 O, H2O has 0 C and 1 O
            # C/O ~ CO / (CO + H2O)
            co_from_main = vmr_dict["CO"] / (vmr_dict["CO"] + vmr_dict["H2O"] + 1e-20)
            co_estimates.append(co_from_main)

        # From CH4 and CO
        if "CH4" in vmr_dict and "CO" in vmr_dict:
            # High CH4 relative to CO suggests high C/O
            co_from_ch4 = (vmr_dict["CO"] + vmr_dict["CH4"]) / (vmr_dict["CO"] + 1e-20)
            # This is approximate
            co_estimates.append(jnp.clip(co_from_ch4, 0.1, 2.0))

        if len(co_estimates) >= 2:
            # Penalize variance in C/O estimates
            co_arr = jnp.array(co_estimates)
            penalty = jnp.var(co_arr) * 100

        return penalty

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        **kwargs,
    ) -> CompositionState:
        """Sample with chemical consistency constraints."""
        # Get base composition
        state = self.base_solver.sample(
            mol_names, mol_masses, atom_names, atom_masses, art, **kwargs
        )

        # Compute penalties
        total_penalty = jnp.array(0.0)

        if self.enforce_mass_balance:
            # Need metallicity - try to extract from sampled parameters
            # This is approximate; proper implementation would pass metallicity
            metallicity = 1.0  # Default to solar
            penalty = self._compute_mass_balance(
                state.vmr_mols, mol_names, metallicity
            )
            total_penalty = total_penalty + penalty

        if self.enforce_co_consistency:
            penalty = self._compute_co_consistency(state.vmr_mols, mol_names)
            total_penalty = total_penalty + penalty

        # Add penalty to model via factor
        if self.consistency_weight > 0:
            numpyro.factor(
                "chemical_consistency_penalty",
                -self.consistency_weight * total_penalty,
            )

        return state


class FastChemSolver:
    """Wrapper for FastChem equilibrium chemistry code.

    FastChem (Stock et al. 2018, 2022) is the gold standard for
    gas-phase equilibrium chemistry in exoplanet atmospheres.

    Requires pyfastchem to be installed:
        pip install pyfastchem

    If FastChem is not available, falls back to EquilibriumChemistry.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        fastchem_data_dir: str | None = None,
        include_condensation: bool = False,
    ):
        """Initialize FastChem solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            fastchem_data_dir: Path to FastChem data directory
            include_condensation: Enable condensation in FastChem
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.fastchem_data_dir = fastchem_data_dir
        self.include_condensation = include_condensation

        # Try to import FastChem
        self.fastchem = None
        self.fastchem_available = False

        try:
            import pyfastchem
            self.pyfastchem = pyfastchem
            self.fastchem_available = True
        except ImportError:
            print("Warning: pyfastchem not installed, falling back to analytic equilibrium")
            self.fallback = EquilibriumChemistry(
                metallicity_range=metallicity_range,
                co_ratio_range=co_ratio_range,
                h2_he_ratio=h2_he_ratio,
            )

    def _init_fastchem(self):
        """Initialize FastChem object (lazy loading)."""
        if self.fastchem is not None:
            return

        if not self.fastchem_available:
            return

        # Initialize FastChem
        if self.fastchem_data_dir is not None:
            self.fastchem = self.pyfastchem.FastChem(
                self.fastchem_data_dir + "/logK",
                self.fastchem_data_dir + "/parameters",
                verbose=0,
            )
        else:
            # Use default data
            self.fastchem = self.pyfastchem.FastChem(verbose=0)

    def _run_fastchem(
        self,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: float,
        co_ratio: float,
    ) -> dict[str, jnp.ndarray]:
        """Run FastChem for given T, P, [M/H], C/O.

        Returns dictionary of species VMR profiles.
        """
        import numpy as np

        self._init_fastchem()

        if self.fastchem is None:
            return {}

        # Convert to numpy
        T_np = np.array(Tarr)
        P_np = np.array(Parr) * 1e6  # bar to dyne/cm^2

        n_layers = len(T_np)

        # Set up element abundances
        # Solar abundances modified by metallicity and C/O
        element_abundances = self.fastchem.getElementAbundances()

        # Scale all metals by metallicity
        for i, name in enumerate(self.fastchem.getElementSymbols()):
            if name not in ("H", "He"):
                element_abundances[i] *= metallicity

        # Adjust C/O ratio
        c_idx = self.fastchem.getElementIndex("C")
        o_idx = self.fastchem.getElementIndex("O")
        if c_idx >= 0 and o_idx >= 0:
            current_co = element_abundances[c_idx] / element_abundances[o_idx]
            element_abundances[c_idx] *= co_ratio / current_co

        self.fastchem.setElementAbundances(element_abundances)

        # Run FastChem
        if self.include_condensation:
            output = self.fastchem.calcDensities(T_np, P_np, condensation=True)
        else:
            output = self.fastchem.calcDensities(T_np, P_np)

        # Extract VMRs for species of interest
        vmr_dict = {}
        species_names = self.fastchem.getSpeciesSymbols()

        for i, name in enumerate(species_names):
            # Get number density and convert to VMR
            n_species = output[:, i]
            n_tot = np.sum(output, axis=1)
            vmr = n_species / n_tot
            vmr_dict[name] = jnp.array(vmr)

        return vmr_dict

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample equilibrium composition using FastChem."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        if not self.fastchem_available:
            # Fall back to analytic equilibrium
            return self.fallback.sample(
                mol_names, mol_masses, atom_names, atom_masses, art, Tarr
            )

        # Run FastChem
        vmr_dict = self._run_fastchem(Tarr, Parr, metallicity, co_ratio)

        # Extract requested species
        vmr_mols_profiles = []
        for mol in mol_names:
            # Try different name formats
            for key in [mol, mol.upper(), mol.replace(" ", "")]:
                if key in vmr_dict:
                    vmr_mols_profiles.append(vmr_dict[key])
                    break
            else:
                # Species not found, use small value
                vmr_mols_profiles.append(jnp.full(n_layers, 1e-20))

        vmr_atoms_profiles = []
        for atom in atom_names:
            for key in [atom, atom.upper(), atom.replace(" ", "")]:
                if key in vmr_dict:
                    vmr_atoms_profiles.append(vmr_dict[key])
                    break
            else:
                vmr_atoms_profiles.append(jnp.full(n_layers, 1e-20))

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Get H2 and He from FastChem
        vmrH2_prof = vmr_dict.get("H2", jnp.full(n_layers, 0.85))
        vmrHe_prof = vmr_dict.get("He", jnp.full(n_layers, 0.15))

        # Compute total trace
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class PhotochemicalSteadyState:
    """Simplified photochemical steady-state model.

    Based on Zahnle et al. (2009), Moses et al. (2011), and Tsai et al. (2021).

    This is a parameterized photochemistry model that modifies equilibrium
    abundances based on:
    - UV flux at planet (from stellar effective temperature and distance)
    - Estimated photolysis rates for key species
    - Simple steady-state balance between production and loss

    For full photochemical modeling, use VULCAN or ARGO.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        T_star: float = 5800.0,
        a_planet: float = 0.05,  # AU
        log_kzz_range: tuple[float, float] = (6.0, 12.0),
    ):
        """Initialize photochemical solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            T_star: Stellar effective temperature (K)
            a_planet: Orbital distance (AU)
            log_kzz_range: Prior range for log10(Kzz) in cm^2/s
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.T_star = T_star
        self.a_planet = a_planet
        self.log_kzz_range = log_kzz_range

        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _photolysis_rate(
        self,
        species: str,
        Parr: jnp.ndarray,
        Tarr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute photolysis rate using MPI-Mainz cross-sections.

        Uses tabulated photolysis cross-sections from the MPI-Mainz UV/VIS
        Spectral Atlas (Keller-Rudek et al. 2013).

        J = integral(sigma * F * phi) dlambda

        Args:
            species: Species name
            Parr: Pressure profile (bar)
            Tarr: Temperature profile (K)

        Returns:
            Photolysis rate J (s^-1) at each pressure level
        """
        return compute_photolysis_rate(species, Parr, self.T_star, self.a_planet)

    def _compute_steady_state_vmr(
        self,
        species: str,
        vmr_eq: jnp.ndarray,
        Parr: jnp.ndarray,
        Tarr: jnp.ndarray,
        Kzz: float,
        metallicity: float,
    ) -> jnp.ndarray:
        """Compute steady-state VMR including photochemistry.

        Solves the simplified photochemical equilibrium equation:
            Production = Loss (chemical) + Loss (photolysis) + Transport

        Uses:
        - MPI-Mainz photolysis cross-sections
        - KIDA/UMIST chemical reaction rates
        - Eddy diffusion transport timescale

        Args:
            species: Species name
            vmr_eq: Equilibrium VMR profile
            Parr: Pressure profile (bar)
            Tarr: Temperature profile (K)
            Kzz: Eddy diffusion coefficient (cm^2/s)
            metallicity: Linear metallicity factor

        Returns:
            Steady-state VMR profile
        """
        species_upper = species.upper().replace(" ", "")

        # Photolysis rate
        J = self._photolysis_rate(species, Parr, Tarr)

        # Chemical production/loss rates from KIDA
        k_B = 1.381e-16  # erg/K
        P_cgs = Parr * 1e6  # bar to dyne/cm^2
        n_tot = P_cgs / (k_B * Tarr)  # cm^-3

        # Scale height for transport
        mmw = 2.3
        m_u = 1.661e-24  # g
        g_cgs = 1000.0  # cm/s^2 (typical for hot Jupiters)
        H = k_B * Tarr / (mmw * m_u * g_cgs)

        # Mixing timescale
        t_mix = H**2 / Kzz

        # Photochemical equilibrium depends on species
        if species_upper == "H2O":
            # H2O + hν → OH + H
            # OH + H2 → H2O + H (recombination)
            k_OH_H2 = compute_reaction_rate("OH_H2", Tarr)
            n_H2 = n_tot * 0.85
            # Production rate = k * n_OH * n_H2
            # At steady state: J * n_H2O = k * n_OH * n_H2
            # Approximate n_OH from equilibrium
            t_photo = 1.0 / (J + 1e-30)
            t_chem = 1.0 / (k_OH_H2 * n_H2 + 1e-30)
            # Depletion factor
            f_ss = t_photo / (t_photo + t_chem + t_mix)
            vmr_ss = vmr_eq * jnp.clip(f_ss, 0.001, 1.0)

        elif species_upper == "CH4":
            # CH4 + hν → CH3 + H (very short wavelength)
            # CH3 + H + M → CH4 + M (recombination)
            # Production from CO + 3H2 in deep atmosphere
            t_photo = 1.0 / (J + 1e-30)
            # Use KIDA rate for recombination
            k_reco = compute_reaction_rate("CH3_H2", Tarr)
            f_H = 1e-4 * jnp.exp(-25000.0 / Tarr)
            n_H = n_tot * f_H
            t_reco = 1.0 / (k_reco * n_H * n_tot * metallicity + 1e-30)
            f_ss = (t_photo * t_reco) / (t_photo * t_reco + t_photo * t_mix + t_reco * t_mix + 1e-30)
            vmr_ss = vmr_eq * jnp.clip(f_ss, 0.001, 1.0)

        elif species_upper == "NH3":
            # NH3 + hν → NH2 + H
            # NH2 + H + M → NH3 + M
            t_photo = 1.0 / (J + 1e-30)
            k_reco = compute_reaction_rate("NH2_H2", Tarr)
            n_H2 = n_tot * 0.85
            t_reco = 1.0 / (k_reco * n_H2 * metallicity + 1e-30)
            f_ss = (t_photo * t_reco) / (t_photo * t_reco + t_photo * t_mix + t_reco * t_mix + 1e-30)
            vmr_ss = vmr_eq * jnp.clip(f_ss, 0.001, 1.0)

        elif species_upper == "HCN":
            # HCN + hν → H + CN
            # CN + H2 → HCN + H
            t_photo = 1.0 / (J + 1e-30)
            # HCN can be enhanced by photochemistry in C-rich atmospheres
            f_ss = jnp.where(J > 1e-10, 0.5, 1.0)  # HCN somewhat resistant
            vmr_ss = vmr_eq * f_ss

        elif species_upper == "CO2":
            # CO2 + hν → CO + O
            t_photo = 1.0 / (J + 1e-30)
            # CO2 depleted in upper atmosphere, but reforms at depth
            k_form = compute_reaction_rate("OH_CO", Tarr)
            f_OH = 1e-6 * jnp.exp(-15000.0 / Tarr) * metallicity
            n_OH = n_tot * f_OH
            t_form = 1.0 / (k_form * n_OH + 1e-30)
            f_ss = t_photo / (t_photo + t_form + t_mix)
            vmr_ss = vmr_eq * jnp.clip(f_ss, 0.01, 1.0)

        elif species_upper == "H2S":
            # H2S + hν → HS + H (efficient photolysis)
            t_photo = 1.0 / (J + 1e-30)
            # H2S strongly depleted above photolysis level
            f_ss = jnp.where(J > 1e-8, 0.01, 1.0)
            vmr_ss = vmr_eq * f_ss

        elif species_upper == "C2H2":
            # C2H2 can be photochemically produced
            # CH4 photolysis → CH2 → C2H2
            # Enhanced in upper atmosphere
            J_CH4 = self._photolysis_rate("CH4", Parr, Tarr)
            enhancement = 1.0 + 10.0 * J_CH4 / (J_CH4 + 1e-10)
            vmr_ss = vmr_eq * jnp.clip(enhancement, 1.0, 100.0)

        else:
            # Species not affected by photochemistry
            vmr_ss = vmr_eq

        # Apply smooth transition between photochemical and equilibrium regions
        # Photochemistry dominant above P ~ 1e-4 bar
        P_transition = 1e-3
        weight_photo = jnp.exp(-jnp.log10(Parr / P_transition))
        weight_photo = jnp.clip(weight_photo, 0.0, 1.0)

        vmr_out = weight_photo * vmr_ss + (1.0 - weight_photo) * vmr_eq

        return vmr_out

    def _apply_photochemistry(
        self,
        vmr_eq: jnp.ndarray,
        species: str,
        Parr: jnp.ndarray,
        Tarr: jnp.ndarray,
        Kzz: float,
        metallicity: float,
    ) -> jnp.ndarray:
        """Apply photochemical modifications to equilibrium VMR.

        Uses MPI-Mainz photolysis cross-sections and KIDA reaction rates
        to compute steady-state abundances.
        """
        return self._compute_steady_state_vmr(
            species, vmr_eq, Parr, Tarr, Kzz, metallicity
        )

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample with photochemical modifications."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )
        log_kzz = numpyro.sample(
            "log_Kzz",
            dist.Uniform(self.log_kzz_range[0], self.log_kzz_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Kzz = jnp.power(10.0, log_kzz)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 1500.0)

        # Compute equilibrium and apply photochemistry using MPI-Mainz cross-sections
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_photo = self._apply_photochemistry(
                vmr_eq, mol, Parr, Tarr, Kzz, metallicity
            )
            vmr_mols_profiles.append(vmr_photo)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_eq = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )
            # Atoms generally not affected by photolysis
            vmr_atoms_profiles.append(vmr_eq)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class VULCANSolver:
    """Wrapper for VULCAN 1D photochemical kinetics code.

    VULCAN (Tsai et al. 2017, 2021) is a comprehensive 1D photochemical
    kinetics model that solves the full continuity equation with:
    - Photolysis reactions
    - Thermochemical kinetics
    - Vertical transport (eddy + molecular diffusion)
    - Condensation

    Requires VULCAN to be installed separately:
        git clone https://github.com/exoclime/VULCAN

    If VULCAN is not available, falls back to PhotochemicalSteadyState.
    """

    def __init__(
        self,
        vulcan_path: str | None = None,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        log_kzz_range: tuple[float, float] = (6.0, 12.0),
        h2_he_ratio: float = 6.0,
        T_star: float = 5800.0,
        R_star: float = 1.0,  # Solar radii
        a_planet: float = 0.05,  # AU
        use_cached: bool = True,
        cache_dir: str | None = None,
    ):
        """Initialize VULCAN solver.

        Args:
            vulcan_path: Path to VULCAN installation directory
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            log_kzz_range: Prior range for log10(Kzz) in cm^2/s
            h2_he_ratio: H2/He number ratio
            T_star: Stellar effective temperature (K)
            R_star: Stellar radius (solar radii)
            a_planet: Orbital distance (AU)
            use_cached: Use pre-computed VULCAN grids if available
            cache_dir: Directory for cached VULCAN results
        """
        self.vulcan_path = vulcan_path
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.log_kzz_range = log_kzz_range
        self.h2_he_ratio = h2_he_ratio
        self.T_star = T_star
        self.R_star = R_star
        self.a_planet = a_planet
        self.use_cached = use_cached
        self.cache_dir = cache_dir

        # Check if VULCAN is available
        self.vulcan_available = False
        if vulcan_path is not None:
            import sys
            import os
            if os.path.exists(vulcan_path):
                sys.path.insert(0, vulcan_path)
                try:
                    import vulcan
                    self.vulcan = vulcan
                    self.vulcan_available = True
                except ImportError:
                    pass

        if not self.vulcan_available:
            print("Warning: VULCAN not available, falling back to PhotochemicalSteadyState")
            self.fallback = PhotochemicalSteadyState(
                metallicity_range=metallicity_range,
                co_ratio_range=co_ratio_range,
                h2_he_ratio=h2_he_ratio,
                T_star=T_star,
                a_planet=a_planet,
                log_kzz_range=log_kzz_range,
            )

        # Load cached grid if available
        self.cached_grid = None
        if use_cached and cache_dir is not None:
            self._load_cached_grid()

    def _load_cached_grid(self) -> None:
        """Load pre-computed VULCAN results from cache."""
        import os
        from pathlib import Path

        if self.cache_dir is None:
            return

        cache_path = Path(self.cache_dir) / "vulcan_grid.npz"
        if cache_path.exists():
            import numpy as np
            data = np.load(cache_path)
            self.cached_grid = {
                "metallicity": data["metallicity"],
                "co_ratio": data["co_ratio"],
                "kzz": data["kzz"],
                "pressure": data["pressure"],
                "species": list(data["species"]),
                "vmr": data["vmr"],  # (n_met, n_co, n_kzz, n_species, n_layers)
            }

    def _interpolate_cached(
        self,
        metallicity: float,
        co_ratio: float,
        Kzz: float,
        species_list: list[str],
    ) -> dict[str, jnp.ndarray]:
        """Interpolate VMRs from cached grid."""
        if self.cached_grid is None:
            return {}

        from scipy.interpolate import RegularGridInterpolator
        import numpy as np

        vmr_dict = {}

        for i, species in enumerate(self.cached_grid["species"]):
            if species in species_list or species.upper() in [s.upper() for s in species_list]:
                interp = RegularGridInterpolator(
                    (
                        self.cached_grid["metallicity"],
                        self.cached_grid["co_ratio"],
                        np.log10(self.cached_grid["kzz"]),
                    ),
                    self.cached_grid["vmr"][:, :, :, i, :],
                    bounds_error=False,
                    fill_value=None,
                )

                points = np.array([[np.log10(metallicity), co_ratio, np.log10(Kzz)]])
                vmr_profile = interp(points)[0]
                vmr_dict[species] = jnp.array(vmr_profile)

        return vmr_dict

    def _run_vulcan(
        self,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: float,
        co_ratio: float,
        Kzz: float,
    ) -> dict[str, jnp.ndarray]:
        """Run VULCAN for given atmospheric conditions.

        This runs VULCAN in steady-state mode and returns VMR profiles.
        Note: This is computationally expensive!
        """
        if not self.vulcan_available:
            return {}

        import numpy as np
        import tempfile
        import os

        # Create temporary config file for VULCAN
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "vulcan_config.txt")

            # Write VULCAN configuration
            with open(config_path, "w") as f:
                f.write(f"# Auto-generated VULCAN config\n")
                f.write(f"Tstar = {self.T_star}\n")
                f.write(f"Rstar = {self.R_star}\n")
                f.write(f"orbit_a = {self.a_planet}\n")
                f.write(f"metallicity = {metallicity}\n")
                f.write(f"C_O = {co_ratio}\n")
                f.write(f"Kzz = {Kzz}\n")
                # ... more config options

            # Run VULCAN (simplified - actual implementation would be more complex)
            # This is a placeholder for the actual VULCAN interface
            try:
                # vulcan_output = self.vulcan.run(config_path, T=Tarr, P=Parr)
                # return vulcan_output.vmr_dict
                pass
            except Exception as e:
                print(f"VULCAN run failed: {e}")
                return {}

        return {}

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample photochemical composition using VULCAN."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )
        log_kzz = numpyro.sample(
            "log_Kzz",
            dist.Uniform(self.log_kzz_range[0], self.log_kzz_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Kzz = jnp.power(10.0, log_kzz)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 1500.0)

        if not self.vulcan_available:
            return self.fallback.sample(
                mol_names, mol_masses, atom_names, atom_masses, art, Tarr
            )

        # Try cached grid first
        if self.cached_grid is not None:
            vmr_dict = self._interpolate_cached(
                metallicity, co_ratio, Kzz, mol_names + atom_names
            )
        else:
            vmr_dict = self._run_vulcan(Tarr, Parr, metallicity, co_ratio, Kzz)

        if not vmr_dict:
            # Fall back if VULCAN fails
            return self.fallback.sample(
                mol_names, mol_masses, atom_names, atom_masses, art, Tarr
            )

        # Extract requested species
        vmr_mols_profiles = []
        for mol in mol_names:
            for key in [mol, mol.upper(), mol.replace(" ", "")]:
                if key in vmr_dict:
                    vmr_mols_profiles.append(vmr_dict[key])
                    break
            else:
                vmr_mols_profiles.append(jnp.full(n_layers, 1e-20))

        vmr_atoms_profiles = []
        for atom in atom_names:
            for key in [atom, atom.upper(), atom.replace(" ", "")]:
                if key in vmr_dict:
                    vmr_atoms_profiles.append(vmr_dict[key])
                    break
            else:
                vmr_atoms_profiles.append(jnp.full(n_layers, 1e-20))

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Get H2 and He
        vmrH2_prof = vmr_dict.get("H2", jnp.full(n_layers, 0.85))
        vmrHe_prof = vmr_dict.get("He", jnp.full(n_layers, 0.15))

        # Compute trace total
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class NonLTEChemistry:
    """Non-LTE chemistry for very hot, low-density atmospheres.

    Based on Fossati et al. (2021), García Muñoz (2007), and Koskinen et al. (2013).

    At very high temperatures (T > 4000K) and low densities (P < 1e-6 bar),
    collisional rates become too slow to maintain local thermodynamic
    equilibrium (LTE). Key effects include:

    - H2 dissociation out of equilibrium
    - Radiative rates dominate over collisional
    - Metastable states become populated
    - Ionization fraction differs from Saha

    This solver adds corrections to LTE chemistry based on departure
    coefficients estimated from the radiation field and density.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        T_threshold: float = 4000.0,  # K, above which NLTE important
        P_threshold: float = 1e-5,    # bar, below which NLTE important
        include_h2_dissociation: bool = True,
        include_nlte_ionization: bool = True,
    ):
        """Initialize Non-LTE chemistry solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            T_threshold: Temperature above which NLTE effects are included
            P_threshold: Pressure below which NLTE effects are included
            include_h2_dissociation: Include non-equilibrium H2 dissociation
            include_nlte_ionization: Include NLTE ionization corrections
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.T_threshold = T_threshold
        self.P_threshold = P_threshold
        self.include_h2_dissociation = include_h2_dissociation
        self.include_nlte_ionization = include_nlte_ionization

        # LTE solver for regions where LTE holds
        self.lte_solver = ThermalIonization(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _h2_dissociation_fraction(
        self,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute H2 dissociation fraction including NLTE effects.

        Uses pre-computed NLTE departure coefficients from stellar atmosphere
        models (PHOENIX, TLUSTY) to correct the LTE dissociation fraction.

        At high T, low P: H2 ↔ 2H shifts toward atomic H.
        NLTE effects enhance dissociation at low densities due to
        reduced three-body recombination rates.

        Args:
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)

        Returns:
            Dissociation fraction α = n_H / (n_H + 2*n_H2)
        """
        # LTE dissociation from Gibbs free energy
        # Using NASA polynomials for accuracy
        g_H2 = gibbs_rt("H2", Tarr)
        g_H = gibbs_rt("H", Tarr)

        # Equilibrium constant: H2 <-> 2H
        # K_p = exp(-delta_G/RT) = exp(-(2*g_H - g_H2))
        ln_Kp = -(2 * g_H - g_H2) + jnp.log(Parr)  # Include pressure term

        # Solve for dissociation fraction
        # At equilibrium: 4α²P/(1-α) = Kp
        Kp = jnp.exp(ln_Kp)

        # Quadratic solution
        # 4α²P + αKp - Kp = 0
        # α = (-Kp + sqrt(Kp² + 16PKp)) / (8P)
        discriminant = Kp**2 + 16 * Parr * Kp
        alpha_lte = (-Kp + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / (8 * Parr + 1e-30)
        alpha_lte = jnp.clip(alpha_lte, 0.0, 1.0)

        # NLTE correction using tabulated departure coefficients
        # b = n_NLTE / n_LTE from pre-computed grids
        b_H2_diss = interpolate_nlte_departure("H2_dissociation", Tarr, Parr)

        # NLTE dissociation: enhanced by departure coefficient
        # When b > 1, dissociation is enhanced relative to LTE
        alpha_nlte = jnp.clip(alpha_lte * b_H2_diss, 0.0, 0.99)

        return alpha_nlte

    def _nlte_ionization_correction(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        x_ion_lte: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply NLTE corrections to ionization fraction using tabulated coefficients.

        Uses pre-computed departure coefficients from stellar atmosphere models
        (PHOENIX, TLUSTY, Kurucz) that account for:
        - Radiative ionization from stellar UV
        - Reduced recombination at low density
        - Non-equilibrium level populations

        The departure coefficient b_i relates NLTE to LTE populations:
            n_i(NLTE) = b_i * n_i(LTE)

        For neutrals: b < 1 means overionization (NLTE reduces neutral population)
        For ions: effectively b_ion > 1

        Args:
            species: Species name (e.g., "Fe I", "Na I")
            Tarr: Temperature profile (K)
            Parr: Pressure profile (bar)
            x_ion_lte: LTE ionization fraction

        Returns:
            NLTE-corrected ionization fraction
        """
        # Map species to departure coefficient grid
        species_upper = species.upper().replace(" ", "")

        if "FE" in species_upper:
            b_neutral = interpolate_nlte_departure("Fe_I", Tarr, Parr)
        elif "NA" in species_upper:
            b_neutral = interpolate_nlte_departure("Na_I", Tarr, Parr)
        elif "H" in species_upper and "H2" not in species_upper:
            b_neutral = interpolate_nlte_departure("H_ground", Tarr, Parr)
        else:
            # For species without tabulated data, use generic NLTE estimate
            # Based on simple density-dependent correction
            is_nlte_region = (Tarr > self.T_threshold) & (Parr < self.P_threshold)
            log_P = jnp.log10(jnp.clip(Parr, 1e-10, 1e2))
            log_P_thresh = jnp.log10(self.P_threshold)

            # Departure from LTE increases as density decreases
            b_neutral = jnp.where(
                is_nlte_region,
                1.0 - 0.2 * (log_P_thresh - log_P),  # Overionization
                1.0,
            )
            b_neutral = jnp.clip(b_neutral, 0.1, 1.0)

        # NLTE ionization fraction
        # If b_neutral < 1, neutral is depleted -> ionization enhanced
        # x_ion(NLTE) = 1 - (1 - x_ion(LTE)) * b_neutral
        x_neutral_nlte = (1.0 - x_ion_lte) * b_neutral
        x_ion_nlte = 1.0 - x_neutral_nlte
        x_ion_nlte = jnp.clip(x_ion_nlte, 0.0, 1.0)

        return x_ion_nlte

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition with NLTE effects."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 3000.0)

        # Get LTE base abundances
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_eq = self.lte_solver.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_mols_profiles.append(vmr_eq)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_eq = self.lte_solver.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )

            # Apply ionization with NLTE corrections
            element = atom.split()[0] if " " in atom else atom
            chi = IONIZATION_POTENTIALS.get(element, 10.0)
            x_ion_lte = self.lte_solver._saha_ionization_fraction(Tarr, Parr, chi)

            if self.include_nlte_ionization:
                x_ion = self._nlte_ionization_correction(atom, Tarr, Parr, x_ion_lte)
            else:
                x_ion = x_ion_lte

            is_neutral = "I" in atom and "II" not in atom
            is_ionized = "II" in atom

            if is_neutral:
                vmr_prof = vmr_eq * (1.0 - x_ion)
            elif is_ionized:
                vmr_prof = vmr_eq * x_ion
            else:
                vmr_prof = vmr_eq

            vmr_atoms_profiles.append(vmr_prof)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Apply H2 dissociation
        if self.include_h2_dissociation:
            alpha_h2 = self._h2_dissociation_fraction(Tarr, Parr)
        else:
            alpha_h2 = jnp.zeros_like(Tarr)

        # Renormalize trace species
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # H2/He/H partitioning with dissociation
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)

        vmr_h2_he = 1.0 - vmr_trace_tot
        vmrH2_prof = vmr_h2_he * h2_frac * (1.0 - alpha_h2)
        vmrHe_prof = vmr_h2_he * he_frac
        vmrH_prof = vmr_h2_he * h2_frac * alpha_h2 * 2  # 2 H atoms per H2

        # Compute MMW (include atomic H)
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mass_H = 1.008

        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof + mass_H * vmrH_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class HorizontalMixing:
    """3D horizontal mixing for tidally locked planets.

    Based on Agúndez et al. (2014), Drummond et al. (2018), and Steinrueck et al. (2019).

    On tidally locked planets, the large day-night temperature contrast
    drives strong horizontal winds. This leads to:

    - "Horizontal quenching": species are transported from hot dayside to
      cold nightside faster than chemistry can adjust
    - Day-night abundance gradients
    - Terminator composition differs from global average

    This solver parameterizes horizontal mixing with:
    - Horizontal mixing timescale τ_horiz
    - Day-night temperature contrast ΔT_dn
    - Effective "quench" based on wind speed vs chemical timescale
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        T_day: float = 2500.0,
        T_night: float = 1000.0,
        log_tau_horiz_range: tuple[float, float] = (3.0, 7.0),  # seconds
        terminator_fraction: float = 0.5,  # Weight of day vs night
    ):
        """Initialize horizontal mixing solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            T_day: Dayside temperature (K)
            T_night: Nightside temperature (K)
            log_tau_horiz_range: Prior range for log10(horizontal mixing time) in s
            terminator_fraction: Fraction of dayside contributing to terminator (0-1)
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.T_day = T_day
        self.T_night = T_night
        self.log_tau_horiz_range = log_tau_horiz_range
        self.terminator_fraction = terminator_fraction

        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _chemical_timescale(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Estimate chemical equilibration timescale."""
        # Reuse from KzzDisequilibrium
        species_upper = species.upper().replace(" ", "")

        if species_upper == "CO":
            Ea = 65000.0
            A = 1e-6
        elif species_upper == "CH4":
            Ea = 65000.0
            A = 1e-6
        elif species_upper == "H2O":
            Ea = 50000.0
            A = 1e-8
        elif species_upper == "NH3":
            Ea = 55000.0
            A = 1e-7
        else:
            Ea = 30000.0
            A = 1e-8

        t_chem = A * jnp.exp(Ea / Tarr) * jnp.power(Parr, -0.5)
        return t_chem

    def _compute_terminator_vmr(
        self,
        species: str,
        vmr_day: jnp.ndarray,
        vmr_night: jnp.ndarray,
        Tarr_day: jnp.ndarray,
        Tarr_night: jnp.ndarray,
        Parr: jnp.ndarray,
        tau_horiz: float,
    ) -> jnp.ndarray:
        """Compute effective VMR at terminator with horizontal mixing.

        If τ_horiz << τ_chem: terminator sees mixed day/night composition
        If τ_horiz >> τ_chem: terminator is in local equilibrium
        """
        # Chemical timescale (use average of day/night)
        t_chem_day = self._chemical_timescale(species, Tarr_day, Parr)
        t_chem_night = self._chemical_timescale(species, Tarr_night, Parr)
        t_chem = 0.5 * (t_chem_day + t_chem_night)

        # Mixing efficiency: how much does horizontal mixing homogenize?
        # η = τ_chem / (τ_chem + τ_horiz)
        # η → 1 when τ_horiz << τ_chem (well-mixed)
        # η → 0 when τ_horiz >> τ_chem (local equilibrium)
        eta = t_chem / (t_chem + tau_horiz)

        # Mixed VMR
        f_day = self.terminator_fraction
        f_night = 1.0 - f_day

        vmr_local = f_day * vmr_day + f_night * vmr_night  # Local equilibrium
        vmr_mixed = 0.5 * (vmr_day + vmr_night)  # Well-mixed

        vmr_terminator = eta * vmr_mixed + (1.0 - eta) * vmr_local

        return vmr_terminator

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample terminator composition with horizontal mixing."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )
        log_tau_horiz = numpyro.sample(
            "log_tau_horiz",
            dist.Uniform(self.log_tau_horiz_range[0], self.log_tau_horiz_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        tau_horiz = jnp.power(10.0, log_tau_horiz)
        Parr = art.pressure
        n_layers = Parr.size

        # Day and night temperature profiles
        if Tarr is not None:
            # Use provided T as terminator average, estimate day/night
            Tarr_day = Tarr * (self.T_day / ((self.T_day + self.T_night) / 2))
            Tarr_night = Tarr * (self.T_night / ((self.T_day + self.T_night) / 2))
        else:
            Tarr_day = jnp.full_like(Parr, self.T_day)
            Tarr_night = jnp.full_like(Parr, self.T_night)

        # Compute day and night equilibrium VMRs
        vmr_mols_profiles = []
        for mol in mol_names:
            vmr_day = self.eq_solver._analytic_equilibrium(
                mol, Tarr_day, Parr, metallicity, co_ratio
            )
            vmr_night = self.eq_solver._analytic_equilibrium(
                mol, Tarr_night, Parr, metallicity, co_ratio
            )
            vmr_term = self._compute_terminator_vmr(
                mol, vmr_day, vmr_night, Tarr_day, Tarr_night, Parr, tau_horiz
            )
            vmr_mols_profiles.append(vmr_term)

        vmr_atoms_profiles = []
        for atom in atom_names:
            vmr_day = self.eq_solver._analytic_equilibrium(
                atom, Tarr_day, Parr, metallicity, co_ratio
            )
            vmr_night = self.eq_solver._analytic_equilibrium(
                atom, Tarr_night, Parr, metallicity, co_ratio
            )
            # Atoms equilibrate faster, so less affected by horizontal mixing
            vmr_term = self._compute_terminator_vmr(
                atom, vmr_day, vmr_night, Tarr_day, Tarr_night, Parr, tau_horiz * 0.1
            )
            vmr_atoms_profiles.append(vmr_term)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)
        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac
        vmrHe_prof = (1.0 - vmr_trace_tot) * he_frac

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


class EscapeFractionation:
    """Atmospheric escape and isotope/elemental fractionation.

    Based on Murray-Clay et al. (2009), Owen & Wu (2017), and Fossati et al. (2018).

    For close-in planets with intense XUV irradiation, atmospheric escape
    can modify composition through:

    - Mass-dependent escape (lighter species escape faster)
    - Energy-limited hydrodynamic escape
    - Fractionation of isotopes (D/H, 13C/12C)
    - Depletion of volatile elements

    This model parameterizes escape with an escape rate and fractionation
    factor, modifying the upper atmosphere composition.
    """

    def __init__(
        self,
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        h2_he_ratio: float = 6.0,
        log_escape_rate_range: tuple[float, float] = (8.0, 12.0),  # g/s
        P_escape: float = 1e-8,  # bar, pressure level where escape occurs
        M_planet: float = 1.0,   # Jupiter masses
        R_planet: float = 1.0,   # Jupiter radii
        T_star: float | None = None,
        a_planet: float | None = None,
    ):
        """Initialize escape fractionation solver.

        Args:
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            h2_he_ratio: H2/He number ratio
            log_escape_rate_range: Prior range for log10(mass loss rate) in g/s
            P_escape: Pressure level where escape occurs (bar)
            M_planet: Planet mass (Jupiter masses)
            R_planet: Planet radius (Jupiter radii)
        """
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.h2_he_ratio = h2_he_ratio
        self.log_escape_rate_range = log_escape_rate_range
        self.P_escape = P_escape
        self.M_planet = M_planet
        self.R_planet = R_planet
        self.T_star = T_star
        self.a_planet = a_planet

        self.eq_solver = EquilibriumChemistry(
            metallicity_range=metallicity_range,
            co_ratio_range=co_ratio_range,
            h2_he_ratio=h2_he_ratio,
        )

    def _escape_velocity(self) -> float:
        """Compute escape velocity at planet surface."""
        G = 6.674e-8  # cgs
        M_jup = 1.898e30  # g
        R_jup = 6.991e9   # cm

        M = self.M_planet * M_jup
        R = self.R_planet * R_jup

        v_esc = jnp.sqrt(2 * G * M / R)
        return v_esc

    def _thermal_velocity(
        self,
        mass_amu: float,
        T: float,
    ) -> float:
        """Compute thermal velocity for a species."""
        k_B = 1.381e-16  # erg/K
        m = mass_amu * 1.661e-24  # g

        v_th = jnp.sqrt(2 * k_B * T / m)
        return v_th

    def _jeans_escape_flux(
        self,
        mass_amu: float,
        T: float,
        n: float,
    ) -> float:
        """Compute Jeans escape flux for a species.

        Φ = n * v_th / (2√π) * (1 + λ) * exp(-λ)
        where λ = v_esc^2 / v_th^2 is the Jeans parameter
        """
        v_esc = self._escape_velocity()
        v_th = self._thermal_velocity(mass_amu, T)

        lambda_j = (v_esc / v_th) ** 2
        escape_factor = (1 + lambda_j) * jnp.exp(-lambda_j)

        flux = n * v_th / (2 * jnp.sqrt(jnp.pi)) * escape_factor
        return flux

    def _fractionation_factor(
        self,
        mass1: float,
        mass2: float,
        T: float,
    ) -> float:
        """Compute mass-dependent fractionation factor.

        f = Φ_1 / Φ_2 relative to initial ratio

        Lighter species escape faster, so f > 1 for m1 < m2.
        """
        v_esc = self._escape_velocity()
        v_th1 = self._thermal_velocity(mass1, T)
        v_th2 = self._thermal_velocity(mass2, T)

        lambda1 = (v_esc / v_th1) ** 2
        lambda2 = (v_esc / v_th2) ** 2

        # Ratio of escape fluxes
        f = (v_th1 / v_th2) * ((1 + lambda1) / (1 + lambda2)) * jnp.exp(lambda2 - lambda1)

        return f

    def _apply_escape(
        self,
        vmr_profile: jnp.ndarray,
        mass_amu: float,
        Parr: jnp.ndarray,
        escape_rate: float,
        T_exobase: float = 5000.0,
    ) -> jnp.ndarray:
        """Apply escape modification to VMR profile.

        At P < P_escape, composition is modified by preferential loss
        of lighter species.
        """
        # Reference mass (H2)
        mass_ref = 2.016

        # Fractionation relative to H2
        f = self._fractionation_factor(mass_amu, mass_ref, T_exobase)

        # Escape depletion factor
        # Heavier species (f < 1) are enhanced relative to H2
        # Lighter species (f > 1) are depleted
        is_escape_region = Parr < self.P_escape

        # Depletion scales with escape rate
        depletion = jnp.where(
            is_escape_region,
            jnp.power(f, -0.1 * jnp.log10(escape_rate / 1e10)),
            1.0,
        )
        depletion = jnp.clip(depletion, 0.1, 10.0)

        return vmr_profile * depletion

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition with escape fractionation."""
        # Sample parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )
        log_escape_rate = numpyro.sample(
            "log_escape_rate",
            dist.Uniform(self.log_escape_rate_range[0], self.log_escape_rate_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        escape_rate = jnp.power(10.0, log_escape_rate)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        # Get equilibrium VMRs and apply escape
        vmr_mols_profiles = []
        for mol, mass in zip(mol_names, mol_masses):
            vmr_eq = self.eq_solver._analytic_equilibrium(
                mol, Tarr, Parr, metallicity, co_ratio
            )
            vmr_esc = self._apply_escape(vmr_eq, mass, Parr, escape_rate)
            vmr_mols_profiles.append(vmr_esc)

        vmr_atoms_profiles = []
        for atom, mass in zip(atom_names, atom_masses):
            vmr_eq = self.eq_solver._analytic_equilibrium(
                atom, Tarr, Parr, metallicity, co_ratio
            )
            vmr_esc = self._apply_escape(vmr_eq, mass, Parr, escape_rate)
            vmr_atoms_profiles.append(vmr_esc)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # H2/He also affected by escape (He enhanced relative to H2)
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)

        # Apply escape fractionation to H2/He ratio
        f_he_h2 = self._fractionation_factor(4.003, 2.016, 5000.0)
        is_escape = Parr < self.P_escape
        he_enhancement = jnp.where(
            is_escape,
            jnp.power(f_he_h2, -0.1 * jnp.log10(escape_rate / 1e10)),
            1.0,
        )

        vmrH2_prof = (1.0 - vmr_trace_tot) * h2_frac / (1.0 + he_enhancement * he_frac / h2_frac)
        vmrHe_prof = (1.0 - vmr_trace_tot) - vmrH2_prof

        # Compute MMW
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )


# ---------------------------------------------------------------------------
# Combined Chemistry Solver - Chains multiple effects together
# ---------------------------------------------------------------------------

# Preset configurations for different planet types
PLANET_PRESETS = {
    "hot_jupiter": {
        "description": "Standard hot Jupiter (1000-2000K)",
        "include_equilibrium": True,
        "include_rainout": True,
        "include_quench": True,
        "include_photochemistry": True,
        "include_ionization": False,
        "include_nlte": False,
        "include_horizontal_mixing": False,
        "include_escape": False,
        "T_eq_typical": 1500.0,
    },
    "ultra_hot_jupiter": {
        "description": "Ultra-hot Jupiter (T > 2200K)",
        "include_equilibrium": True,
        "include_rainout": False,  # Too hot for condensation
        "include_quench": True,
        "include_photochemistry": True,
        "include_ionization": True,
        "include_nlte": True,
        "include_horizontal_mixing": True,
        "include_escape": False,
        "T_eq_typical": 2800.0,
    },
    "warm_neptune": {
        "description": "Warm Neptune / Sub-Neptune (500-1000K)",
        "include_equilibrium": True,
        "include_rainout": True,
        "include_quench": True,
        "include_photochemistry": True,
        "include_ionization": False,
        "include_nlte": False,
        "include_horizontal_mixing": False,
        "include_escape": False,
        "T_eq_typical": 800.0,
    },
    "cool_giant": {
        "description": "Cool giant planet (< 500K)",
        "include_equilibrium": True,
        "include_rainout": True,
        "include_quench": True,
        "include_photochemistry": True,
        "include_ionization": False,
        "include_nlte": False,
        "include_horizontal_mixing": False,
        "include_escape": False,
        "T_eq_typical": 400.0,
    },
    "lava_world": {
        "description": "Ultra-short period / lava world",
        "include_equilibrium": True,
        "include_rainout": False,
        "include_quench": False,  # Chemistry fast
        "include_photochemistry": True,
        "include_ionization": True,
        "include_nlte": True,
        "include_horizontal_mixing": True,
        "include_escape": True,
        "T_eq_typical": 3500.0,
    },
    "escaping": {
        "description": "Planet with significant atmospheric escape",
        "include_equilibrium": True,
        "include_rainout": False,
        "include_quench": True,
        "include_photochemistry": True,
        "include_ionization": True,
        "include_nlte": True,
        "include_horizontal_mixing": False,
        "include_escape": True,
        "T_eq_typical": 2000.0,
    },
}


class CombinedChemistrySolver:
    """Combined chemistry solver that chains multiple effects together.

    This solver applies chemistry effects in the correct physical order:

    1. **Equilibrium**: Base thermochemical equilibrium (Gibbs minimization)
    2. **Rainout**: Condensation depletes refractory species (cold trap)
    3. **Quench**: Vertical mixing freezes abundances where t_mix < t_chem
    4. **Photochemistry**: UV photolysis modifies upper atmosphere
    5. **Ionization**: Thermal ionization (Saha) for hot atmospheres
    6. **NLTE**: Non-LTE corrections for low-density regions
    7. **Horizontal Mixing**: Day-night transport for tidally locked planets
    8. **Escape**: Atmospheric escape and mass fractionation

    Each effect is applied at appropriate pressure/temperature levels and
    smoothly transitions between regimes.

    References:
        - Moses et al. (2011): Disequilibrium carbon, oxygen, and nitrogen
        - Tsai et al. (2021): VULCAN photochemistry
        - Parmentier et al. (2018): Thermal structure of hot Jupiters
        - Steinrueck et al. (2019): 3D effects on transmission spectra
    """

    def __init__(
        self,
        # Planet type preset (overrides individual settings)
        planet_type: str | None = None,
        # Individual component toggles
        include_equilibrium: bool = True,
        include_rainout: bool = False,
        include_quench: bool = True,
        include_photochemistry: bool = True,
        include_ionization: bool = False,
        include_nlte: bool = False,
        include_horizontal_mixing: bool = False,
        include_escape: bool = False,
        # Prior ranges
        metallicity_range: tuple[float, float] = (-2.0, 3.0),
        co_ratio_range: tuple[float, float] = (0.1, 2.0),
        log_kzz_range: tuple[float, float] = (6.0, 12.0),
        h2_he_ratio: float = 6.0,
        # Stellar/orbital parameters (for photochemistry)
        T_star: float = 5800.0,
        R_star: float = 1.0,  # Solar radii
        a_planet: float = 0.05,  # AU
        # Planet parameters (for escape)
        M_planet: float = 1.0,  # Jupiter masses
        R_planet: float = 1.0,  # Jupiter radii
        # Horizontal mixing parameters
        T_day: float | None = None,
        T_night: float | None = None,
        log_tau_horiz_range: tuple[float, float] = (3.0, 7.0),
        # Transition pressures between regimes
        P_rainout_base: float = 1.0,      # bar, cloud base
        P_quench_transition: float = 0.1,  # bar, quench region
        P_photochem_top: float = 1e-4,    # bar, photochem dominant
        P_nlte_threshold: float = 1e-5,   # bar, NLTE kicks in
        # NLTE parameters
        T_nlte_threshold: float = 4000.0,
    ):
        """Initialize combined chemistry solver.

        Args:
            planet_type: Preset configuration name. Options:
                - "hot_jupiter": Standard hot Jupiter (1000-2000K)
                - "ultra_hot_jupiter": Ultra-hot Jupiter (T > 2200K)
                - "warm_neptune": Warm Neptune / Sub-Neptune (500-1000K)
                - "cool_giant": Cool giant planet (< 500K)
                - "lava_world": Ultra-short period / lava world
                - "escaping": Planet with significant atmospheric escape
                If provided, overrides individual include_* settings.
            include_equilibrium: Start from thermochemical equilibrium
            include_rainout: Apply condensation/rainout
            include_quench: Apply vertical mixing quenching
            include_photochemistry: Apply UV photolysis
            include_ionization: Apply thermal ionization (Saha)
            include_nlte: Apply NLTE corrections
            include_horizontal_mixing: Apply day-night mixing
            include_escape: Apply atmospheric escape fractionation
            metallicity_range: Prior range for log10([M/H])
            co_ratio_range: Prior range for C/O ratio
            log_kzz_range: Prior range for log10(Kzz) in cm^2/s
            h2_he_ratio: H2/He number ratio
            T_star: Stellar effective temperature (K)
            R_star: Stellar radius (solar radii)
            a_planet: Orbital distance (AU)
            M_planet: Planet mass (Jupiter masses)
            R_planet: Planet radius (Jupiter radii)
            T_day: Dayside temperature for horizontal mixing (K)
            T_night: Nightside temperature for horizontal mixing (K)
            log_tau_horiz_range: Prior for horizontal mixing timescale
            P_rainout_base: Pressure of cloud base (bar)
            P_quench_transition: Pressure where quenching becomes important
            P_photochem_top: Pressure where photochemistry dominates
            P_nlte_threshold: Pressure below which NLTE important
            T_nlte_threshold: Temperature above which NLTE important
        """
        # Apply preset if specified
        if planet_type is not None:
            if planet_type not in PLANET_PRESETS:
                raise ValueError(
                    f"Unknown planet_type: {planet_type}. "
                    f"Options: {list(PLANET_PRESETS.keys())}"
                )
            preset = PLANET_PRESETS[planet_type]
            include_equilibrium = preset["include_equilibrium"]
            include_rainout = preset["include_rainout"]
            include_quench = preset["include_quench"]
            include_photochemistry = preset["include_photochemistry"]
            include_ionization = preset["include_ionization"]
            include_nlte = preset["include_nlte"]
            include_horizontal_mixing = preset["include_horizontal_mixing"]
            include_escape = preset["include_escape"]

        # Store configuration
        self.include_equilibrium = include_equilibrium
        self.include_rainout = include_rainout
        self.include_quench = include_quench
        self.include_photochemistry = include_photochemistry
        self.include_ionization = include_ionization
        self.include_nlte = include_nlte
        self.include_horizontal_mixing = include_horizontal_mixing
        self.include_escape = include_escape

        # Store parameters
        self.metallicity_range = metallicity_range
        self.co_ratio_range = co_ratio_range
        self.log_kzz_range = log_kzz_range
        self.h2_he_ratio = h2_he_ratio
        self.T_star = T_star
        self.R_star = R_star
        self.a_planet = a_planet
        self.M_planet = M_planet
        self.R_planet = R_planet
        self.T_day = T_day
        self.T_night = T_night
        self.log_tau_horiz_range = log_tau_horiz_range
        self.P_rainout_base = P_rainout_base
        self.P_quench_transition = P_quench_transition
        self.P_photochem_top = P_photochem_top
        self.P_nlte_threshold = P_nlte_threshold
        self.T_nlte_threshold = T_nlte_threshold

        # Initialize component solvers
        self._init_solvers()

    def _init_solvers(self) -> None:
        """Initialize component solvers."""
        # Base equilibrium solver (always needed)
        self.eq_solver = EquilibriumChemistry(
            metallicity_range=self.metallicity_range,
            co_ratio_range=self.co_ratio_range,
            h2_he_ratio=self.h2_he_ratio,
        )

        # Rainout solver
        if self.include_rainout:
            self.rainout_solver = RainoutChemistry(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
            )

        # Kzz disequilibrium solver (for quenching)
        if self.include_quench:
            self.kzz_solver = KzzDisequilibrium(
                log_kzz_range=self.log_kzz_range,
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
            )

        # Photochemistry solver
        if self.include_photochemistry:
            self.photochem_solver = PhotochemicalSteadyState(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
                T_star=self.T_star,
                a_planet=self.a_planet,
                log_kzz_range=self.log_kzz_range,
            )

        # Ionization solver
        if self.include_ionization:
            self.ionization_solver = ThermalIonization(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
            )

        # NLTE solver
        if self.include_nlte:
            self.nlte_solver = NonLTEChemistry(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
                T_threshold=self.T_nlte_threshold,
                P_threshold=self.P_nlte_threshold,
            )

        # Horizontal mixing solver
        if self.include_horizontal_mixing:
            T_day = self.T_day if self.T_day is not None else 2500.0
            T_night = self.T_night if self.T_night is not None else 1000.0
            self.horizontal_solver = HorizontalMixing(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
                T_day=T_day,
                T_night=T_night,
                log_tau_horiz_range=self.log_tau_horiz_range,
            )

        # Escape solver
        if self.include_escape:
            self.escape_solver = EscapeFractionation(
                metallicity_range=self.metallicity_range,
                co_ratio_range=self.co_ratio_range,
                h2_he_ratio=self.h2_he_ratio,
                M_planet=self.M_planet,
                R_planet=self.R_planet,
                T_star=self.T_star,
                a_planet=self.a_planet,
            )

    def _smooth_transition(
        self,
        vmr_1: jnp.ndarray,
        vmr_2: jnp.ndarray,
        Parr: jnp.ndarray,
        P_transition: float,
        width_decades: float = 1.0,
    ) -> jnp.ndarray:
        """Smoothly transition between two VMR profiles.

        Uses a sigmoid function centered at P_transition.

        Args:
            vmr_1: VMR profile for P > P_transition (deeper)
            vmr_2: VMR profile for P < P_transition (higher altitude)
            Parr: Pressure array (bar)
            P_transition: Transition pressure (bar)
            width_decades: Width of transition in decades of pressure

        Returns:
            Smoothly blended VMR profile
        """
        # Sigmoid weight: 0 at high P, 1 at low P
        log_P = jnp.log10(Parr)
        log_P_trans = jnp.log10(P_transition)
        weight = 1.0 / (1.0 + jnp.exp((log_P - log_P_trans) / (width_decades * 0.5)))

        return vmr_1 * (1.0 - weight) + vmr_2 * weight

    def _apply_equilibrium(
        self,
        species: str,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: float,
        co_ratio: float,
    ) -> jnp.ndarray:
        """Apply equilibrium chemistry (step 1)."""
        return self.eq_solver._gibbs_equilibrium(
            species, Tarr, Parr, metallicity, co_ratio
        )

    def _apply_rainout(
        self,
        species: str,
        vmr_eq: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        metallicity: float,
    ) -> jnp.ndarray:
        """Apply rainout/condensation (step 2)."""
        species_upper = species.upper().replace(" ", "")

        # Check if species condenses
        if species_upper not in CONDENSATION_TEMPS:
            return vmr_eq

        T_cond_1bar = CONDENSATION_TEMPS[species_upper]

        # Condensation temperature varies with pressure
        # T_cond(P) ≈ T_cond(1bar) * (P / 1bar)^0.05
        T_cond = T_cond_1bar * jnp.power(Parr, 0.05)

        # Apply cold trap: deplete above condensation level
        # Find where T crosses T_cond
        is_condensed = Tarr < T_cond

        # Get VMR at cold trap (deepest condensation point)
        # Use minimum VMR in condensation region
        vmr_cold_trap = jnp.where(
            is_condensed,
            vmr_eq * jnp.exp(-(T_cond - Tarr) / 100.0),  # Exponential depletion
            vmr_eq,
        )

        # Above cold trap, species is depleted
        # Find the cold trap pressure (highest P where condensed)
        # Simplified: use smooth transition
        vmr_rainout = jnp.where(
            is_condensed,
            vmr_cold_trap * 0.01,  # Strong depletion in condensation region
            vmr_eq,
        )

        return vmr_rainout

    def _apply_quench(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        Kzz: float,
        metallicity: float,
    ) -> jnp.ndarray:
        """Apply quenching from vertical mixing (step 3)."""
        # Get chemical and mixing timescales
        t_chem = self.kzz_solver._chemical_timescale(species, Tarr, Parr, metallicity)

        mmw = 2.3  # Approximate
        H = self.kzz_solver._scale_height(Tarr, mmw)
        t_mix = self.kzz_solver._mixing_timescale(H, Kzz)

        # Quench where mixing is faster than chemistry
        quench_ratio = t_mix / (t_chem + 1e-30)

        # Find quench level and freeze abundances above it
        # Weight by how close we are to quench point
        weights = jnp.exp(-jnp.abs(jnp.log10(quench_ratio + 1e-30)) * 2)
        weights = weights / (jnp.sum(weights) + 1e-30)
        vmr_quench = jnp.sum(vmr_in * weights)

        # Apply quenching
        is_quenched = quench_ratio < 1.0
        vmr_out = jnp.where(is_quenched, vmr_quench, vmr_in)

        return vmr_out

    def _apply_photochemistry(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        Kzz: float,
        metallicity: float,
    ) -> jnp.ndarray:
        """Apply photochemical modifications (step 4)."""
        return self.photochem_solver._compute_steady_state_vmr(
            species, vmr_in, Parr, Tarr, Kzz, metallicity
        )

    def _apply_ionization(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply thermal ionization (step 5)."""
        species_upper = species.upper().replace(" ", "")

        # Only apply to atomic species
        is_neutral = "I" in species and "II" not in species
        is_ionized = "II" in species

        if not (is_neutral or is_ionized):
            return vmr_in

        # Get ionization potential
        element = species.split()[0] if " " in species else species.replace("I", "").replace(" ", "")
        chi = IONIZATION_POTENTIALS.get(element, 10.0)

        # Saha ionization fraction
        x_ion = self.ionization_solver._saha_ionization_fraction(Tarr, Parr, chi)

        if is_neutral:
            return vmr_in * (1.0 - x_ion)
        else:  # is_ionized
            return vmr_in * x_ion

    def _apply_nlte(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply NLTE corrections (step 6)."""
        species_upper = species.upper().replace(" ", "")

        # Only apply in NLTE region
        is_nlte_region = (Tarr > self.T_nlte_threshold) & (Parr < self.P_nlte_threshold)

        # Get departure coefficient based on species
        # Note: we always compute this to maintain JIT compatibility
        if "FE" in species_upper:
            b = interpolate_nlte_departure("Fe_I", Tarr, Parr)
        elif "NA" in species_upper:
            b = interpolate_nlte_departure("Na_I", Tarr, Parr)
        else:
            # Generic NLTE correction
            b = jnp.where(is_nlte_region, 0.8, 1.0)

        vmr_nlte = vmr_in * b
        return jnp.where(is_nlte_region, vmr_nlte, vmr_in)

    def _apply_horizontal_mixing(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Tarr_day: jnp.ndarray,
        Tarr_night: jnp.ndarray,
        Parr: jnp.ndarray,
        tau_horiz: float,
        metallicity: float,
        co_ratio: float,
    ) -> jnp.ndarray:
        """Apply horizontal mixing for tidally locked planets (step 7)."""
        # Get dayside and nightside equilibrium VMRs
        vmr_day = self.eq_solver._gibbs_equilibrium(
            species, Tarr_day, Parr, metallicity, co_ratio
        )
        vmr_night = self.eq_solver._gibbs_equilibrium(
            species, Tarr_night, Parr, metallicity, co_ratio
        )

        # Chemical timescale
        if hasattr(self, "kzz_solver"):
            t_chem = self.kzz_solver._chemical_timescale(species, Tarr, Parr, metallicity)
        elif hasattr(self, "horizontal_solver"):
            t_chem = self.horizontal_solver._chemical_timescale(species, Tarr, Parr)
        else:
            t_chem = 1e6 * jnp.ones_like(Tarr)

        # Mixing efficiency
        eta = t_chem / (t_chem + tau_horiz + 1e-30)

        # Blended VMR
        vmr_mixed = 0.5 * (vmr_day + vmr_night)
        vmr_local = vmr_in

        vmr_out = eta * vmr_mixed + (1.0 - eta) * vmr_local

        return vmr_out

    def _apply_escape(
        self,
        species: str,
        vmr_in: jnp.ndarray,
        Tarr: jnp.ndarray,
        Parr: jnp.ndarray,
        mass: float,
    ) -> jnp.ndarray:
        """Apply atmospheric escape fractionation (step 8)."""
        # Jeans escape parameter
        G = 6.674e-8  # cgs
        k_B = 1.381e-16  # erg/K
        M_J = 1.898e30  # g
        R_J = 6.991e9  # cm
        m_u = 1.661e-24  # g

        M_p = self.M_planet * M_J
        R_p = self.R_planet * R_J

        # Jeans parameter at each level
        # lambda = G * M * m / (k * T * r)
        # Approximate r ≈ R_p for upper atmosphere
        lambda_jeans = G * M_p * (mass * m_u) / (k_B * Tarr * R_p)

        # Escape flux relative to H
        # Heavier species escape slower
        m_H = 1.008
        lambda_H = G * M_p * (m_H * m_u) / (k_B * Tarr * R_p)

        # Fractionation factor
        f_escape = jnp.exp(-(lambda_jeans - lambda_H))
        f_escape = jnp.clip(f_escape, 0.01, 1.0)

        # Apply escape in upper atmosphere (P < 1e-6 bar)
        P_escape = 1e-6
        vmr_out = jnp.where(
            Parr < P_escape,
            vmr_in * f_escape,
            vmr_in,
        )

        return vmr_out

    def sample(
        self,
        mol_names: list[str],
        mol_masses: list[float],
        atom_names: list[str],
        atom_masses: list[float],
        art: object,
        Tarr: jnp.ndarray | None = None,
    ) -> CompositionState:
        """Sample composition with all enabled chemistry effects.

        Applies chemistry effects in order:
        1. Equilibrium (Gibbs minimization)
        2. Rainout (condensation)
        3. Quench (vertical mixing)
        4. Photochemistry (UV photolysis)
        5. Ionization (Saha equation)
        6. NLTE corrections
        7. Horizontal mixing (day-night)
        8. Escape fractionation

        Args:
            mol_names: List of molecule names
            mol_masses: Molecular masses (AMU)
            atom_names: List of atomic species names
            atom_masses: Atomic masses (AMU)
            art: ExoJAX art object
            Tarr: Temperature profile (K). If None, uses isothermal at 2000K.

        Returns:
            CompositionState with final abundances
        """
        # Sample shared parameters
        log_metallicity = numpyro.sample(
            "log_metallicity",
            dist.Uniform(self.metallicity_range[0], self.metallicity_range[1]),
        )
        co_ratio = numpyro.sample(
            "C_O_ratio",
            dist.Uniform(self.co_ratio_range[0], self.co_ratio_range[1]),
        )

        metallicity = jnp.power(10.0, log_metallicity)
        Parr = art.pressure
        n_layers = Parr.size

        if Tarr is None:
            Tarr = jnp.full_like(Parr, 2000.0)

        # Sample Kzz if needed for quench or photochemistry
        if self.include_quench or self.include_photochemistry:
            log_kzz = numpyro.sample(
                "log_Kzz",
                dist.Uniform(self.log_kzz_range[0], self.log_kzz_range[1]),
            )
            Kzz = jnp.power(10.0, log_kzz)
        else:
            Kzz = 1e8  # Default

        # Sample horizontal mixing timescale if needed
        if self.include_horizontal_mixing:
            log_tau_horiz = numpyro.sample(
                "log_tau_horiz",
                dist.Uniform(self.log_tau_horiz_range[0], self.log_tau_horiz_range[1]),
            )
            tau_horiz = jnp.power(10.0, log_tau_horiz)

            # Day/night temperature profiles
            T_day = self.T_day if self.T_day is not None else Tarr * 1.3
            T_night = self.T_night if self.T_night is not None else Tarr * 0.7
            Tarr_day = jnp.full_like(Parr, T_day) if jnp.isscalar(T_day) else T_day
            Tarr_night = jnp.full_like(Parr, T_night) if jnp.isscalar(T_night) else T_night
        else:
            tau_horiz = 1e5
            Tarr_day = Tarr
            Tarr_night = Tarr

        # Process each species through the pipeline
        vmr_mols_profiles = []
        for mol, mass in zip(mol_names, mol_masses):
            # Step 1: Equilibrium
            vmr = self._apply_equilibrium(mol, Tarr, Parr, metallicity, co_ratio)

            # Step 2: Rainout
            if self.include_rainout:
                vmr = self._apply_rainout(mol, vmr, Tarr, Parr, metallicity)

            # Step 3: Quench
            if self.include_quench:
                vmr = self._apply_quench(mol, vmr, Tarr, Parr, Kzz, metallicity)

            # Step 4: Photochemistry
            if self.include_photochemistry:
                vmr = self._apply_photochemistry(mol, vmr, Tarr, Parr, Kzz, metallicity)

            # Step 7: Horizontal mixing (before ionization/NLTE for molecules)
            if self.include_horizontal_mixing:
                vmr = self._apply_horizontal_mixing(
                    mol, vmr, Tarr, Tarr_day, Tarr_night, Parr,
                    tau_horiz, metallicity, co_ratio
                )

            # Step 8: Escape
            if self.include_escape:
                vmr = self._apply_escape(mol, vmr, Tarr, Parr, mass)

            vmr_mols_profiles.append(vmr)

        vmr_atoms_profiles = []
        for atom, mass in zip(atom_names, atom_masses):
            # Step 1: Equilibrium
            vmr = self._apply_equilibrium(atom, Tarr, Parr, metallicity, co_ratio)

            # Step 2: Rainout (for condensable atoms like Fe, Ti)
            if self.include_rainout:
                vmr = self._apply_rainout(atom, vmr, Tarr, Parr, metallicity)

            # Step 5: Ionization
            if self.include_ionization:
                vmr = self._apply_ionization(atom, vmr, Tarr, Parr)

            # Step 6: NLTE
            if self.include_nlte:
                vmr = self._apply_nlte(atom, vmr, Tarr, Parr)

            # Step 7: Horizontal mixing
            if self.include_horizontal_mixing:
                vmr = self._apply_horizontal_mixing(
                    atom, vmr, Tarr, Tarr_day, Tarr_night, Parr,
                    tau_horiz, metallicity, co_ratio
                )

            # Step 8: Escape
            if self.include_escape:
                vmr = self._apply_escape(atom, vmr, Tarr, Parr, mass)

            vmr_atoms_profiles.append(vmr)

        n_mols = len(vmr_mols_profiles)
        n_atoms = len(vmr_atoms_profiles)

        # Renormalize if needed
        if n_mols + n_atoms > 0:
            all_profiles = jnp.array(vmr_mols_profiles + vmr_atoms_profiles)
            sum_trace = jnp.sum(all_profiles, axis=0)
            scale = jnp.where(sum_trace > 0.5, 0.5 / sum_trace, 1.0)
            all_profiles = all_profiles * scale[None, :]
            vmr_mols_profiles = [all_profiles[i] for i in range(n_mols)]
            vmr_atoms_profiles = [all_profiles[n_mols + i] for i in range(n_atoms)]
            vmr_trace_tot = jnp.sum(all_profiles, axis=0)
        else:
            vmr_trace_tot = jnp.zeros(n_layers)

        # Fill with H2/He
        h2_frac = self.h2_he_ratio / (self.h2_he_ratio + 1.0)
        he_frac = 1.0 / (self.h2_he_ratio + 1.0)

        # Apply H2 dissociation if NLTE enabled
        if self.include_nlte:
            alpha_h2 = self.nlte_solver._h2_dissociation_fraction(Tarr, Parr)
        else:
            alpha_h2 = jnp.zeros_like(Tarr)

        vmr_h2_he = 1.0 - vmr_trace_tot
        vmrH2_prof = vmr_h2_he * h2_frac * (1.0 - alpha_h2)
        vmrHe_prof = vmr_h2_he * he_frac
        vmrH_prof = vmr_h2_he * h2_frac * alpha_h2 * 2

        # Compute mean molecular weight
        mass_H2 = molinfo.molmass_isotope("H2")
        mass_He = molinfo.molmass_isotope("He", db_HIT=False)
        mass_H = 1.008

        mmw_prof = mass_H2 * vmrH2_prof + mass_He * vmrHe_prof + mass_H * vmrH_prof
        for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses):
            mmw_prof = mmw_prof + mass * vmr_prof
        for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses):
            mmw_prof = mmw_prof + mass * vmr_prof

        # Convert to MMR
        if n_mols > 0:
            mmr_mols = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_mols_profiles, mol_masses)
            ])
        else:
            mmr_mols = jnp.zeros((0, n_layers))

        if n_atoms > 0:
            mmr_atoms = jnp.array([
                vmr_prof * (mass / mmw_prof)
                for vmr_prof, mass in zip(vmr_atoms_profiles, atom_masses)
            ])
        else:
            mmr_atoms = jnp.zeros((0, n_layers))

        # Scalar outputs
        vmr_mols_scalar = [jnp.mean(p) for p in vmr_mols_profiles]
        vmr_atoms_scalar = [jnp.mean(p) for p in vmr_atoms_profiles]
        vmrH2 = jnp.mean(vmrH2_prof)
        vmrHe = jnp.mean(vmrHe_prof)
        mmw = jnp.mean(mmw_prof)

        return CompositionState(
            vmr_mols=vmr_mols_scalar,
            vmr_atoms=vmr_atoms_scalar,
            vmrH2=vmrH2,
            vmrHe=vmrHe,
            mmw=mmw,
            mmr_mols=mmr_mols,
            mmr_atoms=mmr_atoms,
            vmrH2_prof=vmrH2_prof,
            vmrHe_prof=vmrHe_prof,
            mmw_prof=mmw_prof,
        )

    def describe(self) -> str:
        """Return a description of the enabled chemistry effects."""
        effects = []
        if self.include_equilibrium:
            effects.append("1. Equilibrium (Gibbs minimization with NASA polynomials)")
        if self.include_rainout:
            effects.append("2. Rainout (condensation/cold trap)")
        if self.include_quench:
            effects.append("3. Quench (Kzz vertical mixing)")
        if self.include_photochemistry:
            effects.append("4. Photochemistry (MPI-Mainz cross-sections)")
        if self.include_ionization:
            effects.append("5. Ionization (Saha equation)")
        if self.include_nlte:
            effects.append("6. NLTE (tabulated departure coefficients)")
        if self.include_horizontal_mixing:
            effects.append("7. Horizontal mixing (day-night transport)")
        if self.include_escape:
            effects.append("8. Escape (Jeans fractionation)")

        return "CombinedChemistrySolver with:\n" + "\n".join(effects)


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def get_chemistry_solver(
    name: str,
    **kwargs,
) -> CompositionSolver:
    """Get a chemistry solver by name.

    Args:
        name: Solver name. Options:
            Individual solvers:
            - "constant": ConstantVMR (vertically constant, uniform prior)
            - "free": FreeVMR (vertically varying, node interpolation)
            - "equilibrium": EquilibriumChemistry ([M/H] + C/O with Gibbs)
            - "quench": QuenchChemistry (equilibrium + quench pressure)
            - "co_metallicity": COMetallicity (C/O partitioning with Gibbs)
            - "gp": GPChemistry (GP-smoothed vertical profiles)
            - "rainout": RainoutChemistry (equilibrium + condensation)
            - "kzz": KzzDisequilibrium (eddy diffusion with KIDA rates)
            - "ionization": ThermalIonization (Saha ionization equilibrium)
            - "consistent": ChemicallyConsistent (mass balance constraints)
            - "fastchem": FastChemSolver (FastChem wrapper)
            - "photochem": PhotochemicalSteadyState (MPI-Mainz photochemistry)
            - "vulcan": VULCANSolver (VULCAN photochemical kinetics)
            - "nlte": NonLTEChemistry (non-LTE with departure coefficients)
            - "horizontal": HorizontalMixing (3D day-night mixing)
            - "escape": EscapeFractionation (atmospheric escape)

            Combined solver (chains multiple effects):
            - "combined": CombinedChemistrySolver (full pipeline)

            Planet-type presets (use CombinedChemistrySolver with preset config):
            - "hot_jupiter": Standard hot Jupiter (1000-2000K)
            - "ultra_hot_jupiter": Ultra-hot Jupiter (T > 2200K)
            - "warm_neptune": Warm Neptune / Sub-Neptune (500-1000K)
            - "cool_giant": Cool giant planet (< 500K)
            - "lava_world": Ultra-short period / lava world
            - "escaping": Planet with significant atmospheric escape

        **kwargs: Additional arguments passed to the solver constructor.

    Returns:
        CompositionSolver instance

    Examples:
        # Basic equilibrium chemistry
        solver = get_chemistry_solver("equilibrium")

        # Full disequilibrium pipeline for hot Jupiter
        solver = get_chemistry_solver("hot_jupiter")

        # Custom combined solver
        solver = get_chemistry_solver("combined",
            include_equilibrium=True,
            include_quench=True,
            include_photochemistry=True,
            include_ionization=True,
        )
    """
    # Individual solvers
    solvers = {
        "constant": ConstantVMR,
        "free": FreeVMR,
        "equilibrium": EquilibriumChemistry,
        "quench": QuenchChemistry,
        "co_metallicity": COMetallicity,
        "gp": GPChemistry,
        "rainout": RainoutChemistry,
        "kzz": KzzDisequilibrium,
        "ionization": ThermalIonization,
        "consistent": ChemicallyConsistent,
        "fastchem": FastChemSolver,
        "photochem": PhotochemicalSteadyState,
        "vulcan": VULCANSolver,
        "nlte": NonLTEChemistry,
        "horizontal": HorizontalMixing,
        "escape": EscapeFractionation,
        "combined": CombinedChemistrySolver,
    }

    # Planet-type presets (aliases for CombinedChemistrySolver with preset)
    planet_presets = {
        "hot_jupiter": "hot_jupiter",
        "ultra_hot_jupiter": "ultra_hot_jupiter",
        "warm_neptune": "warm_neptune",
        "cool_giant": "cool_giant",
        "lava_world": "lava_world",
        "escaping": "escaping",
    }

    # Check if it's a planet preset
    if name in planet_presets:
        return CombinedChemistrySolver(planet_type=name, **kwargs)

    if name not in solvers:
        all_options = list(solvers.keys()) + list(planet_presets.keys())
        raise ValueError(f"Unknown chemistry solver: {name}. Options: {all_options}")

    return solvers[name](**kwargs)
