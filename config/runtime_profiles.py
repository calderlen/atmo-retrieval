"""Named runtime profiles for machine-specific default settings.

These profiles intentionally target the largest VRAM/runtime drivers while
leaving science choices such as species selection to explicit CLI flags.
"""

CONFIG_PROFILE_ENVVAR = "ATMO_CONFIG_PROFILE"
DEFAULT_RUNTIME_PROFILE = "desktop"


CONFIG_PROFILES = {
    "desktop": {
        "description": "Lower-memory local defaults for desktop and laptop runs.",
        "overrides": {
            "NLAYER": 20,
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
