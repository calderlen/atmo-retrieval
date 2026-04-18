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
