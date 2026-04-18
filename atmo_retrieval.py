import argparse
import importlib.util
import math
import re
import sys
import os
import warnings

warnings.filterwarnings("once")

import config
from pipeline.retrieval import (
    make_bandpass_constraints_from_tbl,
    make_joint_spectrum_component_from_tbl,
    run_retrieval,
)
from pipeline.retrieval_binned import run_phase_binned_retrieval


TESS_BTJD_OFFSET = 2457000.0


def create_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    # Target selection
    target_group = parser.add_argument_group("Target")
    target_group.add_argument(
        "--planet",
        type=str,
        required=True,
        help="Target planet (required)"
    )
    target_group.add_argument(
        "--ephemeris",
        type=str,
        default=None,
        help="Ephemeris source (default: Duck24)"
    )

    # Retrieval mode
    parser.add_argument("--mode", type=str, choices=["transmission", "emission"], required=True, help="Retrieval mode (required)")

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, default=None, help="Path to custom config file (default: use config.py)")
    config_group.add_argument(
        "--profile",
        type=str,
        choices=config.list_runtime_profiles(),
        default=config.get_runtime_profile_name(),
        help=(
            "Named runtime profile for machine-specific defaults "
            f"(default: {config.get_runtime_profile_name()}, env: {config.CONFIG_PROFILE_ENVVAR})"
        ),
    )
    config_group.add_argument("--output", type=str, default=None, help="Output directory (default: output/{planet}/{ephemeris}/{mode})")

    # Data options
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--epoch", type=str, required=True, help="Observation epoch (YYYYMMDD) (required)")
    data_group.add_argument(
        "--joint-spectrum-tbl",
        type=str,
        action="append",
        default=None,
        help="Explicit NASA .tbl file to include as a low-resolution spectroscopic component",
    )
    data_group.add_argument(
        "--data-format",
        type=str,
        choices=["timeseries", "spectrum"],
        default=config.DEFAULT_DATA_FORMAT,
        help=f"Data format to load (default: {config.DEFAULT_DATA_FORMAT})",
    )
    data_group.add_argument("--wavelength-range", type=str, choices=["blue", "green", "red", "full"], default=None, help="Wavelength range mode (default: from config)")
    data_group.add_argument(
        "--resolution-mode",
        type=str,
        choices=["standard", "hr", "uhr"],
        default=None,
        help=f"Instrument resolution mode (default: {config.RESOLUTION_MODE})",
    )
    data_group.add_argument(
        "--bandpass-tbl",
        type=str,
        action="append",
        default=None,
        help=(
            "Explicit NASA .tbl file to include as one or more bandpass constraints "
            "(full path or relative to input/phot; canonical form: <mode>/<planet>/file.tbl)"
        ),
    )

    tess_group = parser.add_argument_group("TESS Transit Fit")
    tess_group.add_argument(
        "--fit-tess-transit",
        action="store_true",
        help=(
            "Fit raw TESS photometry with lightkurve+mlexo and inject the resulting "
            "broadband transit constraint into the retrieval"
        ),
    )
    tess_group.add_argument(
        "--tess-target",
        type=str,
        default=None,
        help="Target name to pass to lightkurve (default: reuse --planet)",
    )
    tess_group.add_argument(
        "--tess-sector",
        type=int,
        action="append",
        default=None,
        help="TESS sector to include; pass multiple times to include multiple sectors",
    )
    tess_group.add_argument(
        "--tess-exptime-s",
        type=int,
        default=120,
        help="Requested TESS cadence in seconds for lightkurve search (default: 120)",
    )
    tess_group.add_argument(
        "--tess-quality-bitmask",
        type=str,
        default="default",
        help=(
            "Lightkurve cadence quality mask for TESS downloads. "
            "Use one of none/default/hard/hardest or an integer bitmask "
            "(default: default)"
        ),
    )
    tess_group.add_argument(
        "--tess-flux-column",
        type=str,
        default="pdcsap_flux",
        help=(
            "Flux column to read from the downloaded TESS light curves "
            "(default: pdcsap_flux)"
        ),
    )
    tess_group.add_argument(
        "--tess-author",
        type=str,
        default="SPOC",
        help="TESS light curve author to request from lightkurve (default: SPOC)",
    )
    tess_group.add_argument(
        "--tess-mission",
        type=str,
        default="TESS",
        help="Mission label to pass to lightkurve (default: TESS)",
    )
    tess_group.add_argument(
        "--tess-period-d",
        type=float,
        default=None,
        help="Transit period in days (default: use active planet config)",
    )
    tess_group.add_argument(
        "--tess-t0-btjd",
        type=float,
        default=None,
        help="Transit mid-time in BTJD for the TESS fit",
    )
    tess_group.add_argument(
        "--tess-t0-bjd",
        type=float,
        default=None,
        help=(
            "Transit mid-time in BJD_TDB for the TESS fit. "
            "Converted internally to BTJD by subtracting 2457000.0"
        ),
    )
    tess_group.add_argument(
        "--tess-duration-d",
        type=float,
        default=None,
        help="Transit duration in days (default: use active planet config)",
    )
    tess_group.add_argument(
        "--tess-radius-ratio-guess",
        type=float,
        default=None,
        help="Initial Rp/R* guess for the transit fit (default: use active planet config)",
    )
    tess_group.add_argument(
        "--tess-impact-guess",
        type=float,
        default=None,
        help="Initial impact-parameter guess for the transit fit (default: use active planet config)",
    )
    tess_group.add_argument(
        "--tess-rho-star-solar-guess",
        type=float,
        default=None,
        help="Optional stellar-density prior mean in solar-density units",
    )
    tess_group.add_argument(
        "--tess-rho-star-solar-sigma",
        type=float,
        default=None,
        help="Optional stellar-density prior sigma in solar-density units",
    )
    tess_group.add_argument(
        "--tess-model-window-d",
        type=float,
        default=None,
        help="Half-width of the TESS fit window around transit in days",
    )
    tess_group.add_argument(
        "--tess-plot-window-d",
        type=float,
        default=None,
        help="Half-width of the stored phase plot window in days",
    )
    tess_group.add_argument(
        "--tess-flatten-window-length",
        type=int,
        default=None,
        help="Savitzky-Golay window length for light-curve flattening",
    )
    tess_group.add_argument(
        "--tess-outlier-sigma",
        type=float,
        default=None,
        help="Sigma threshold for outlier rejection in out-of-transit cadences",
    )
    tess_group.add_argument(
        "--tess-emcee-nwalkers-min",
        type=int,
        default=None,
        help="Minimum number of emcee walkers for the TESS fit",
    )
    tess_group.add_argument(
        "--tess-emcee-burnin-steps",
        type=int,
        default=None,
        help="Number of emcee burn-in steps for the TESS fit",
    )
    tess_group.add_argument(
        "--tess-emcee-production-steps",
        type=int,
        default=None,
        help="Number of emcee production steps for the TESS fit",
    )
    tess_group.add_argument(
        "--tess-emcee-thin",
        type=int,
        default=None,
        help="Thin factor for stored TESS emcee samples",
    )
    tess_group.add_argument(
        "--tess-emcee-use-pool",
        action="store_true",
        help="Enable multiprocessing for the TESS emcee run",
    )
    tess_group.add_argument(
        "--tess-mlexo-root",
        type=str,
        default=None,
        help="Path to a local mlexo checkout if it is not adjacent to this repo",
    )
    tess_group.add_argument(
        "--tess-observable",
        type=str,
        choices=["radius_ratio", "transit_depth"],
        default="transit_depth",
        help="Which fitted transit observable to pass into retrieval (default: transit_depth)",
    )
    tess_group.add_argument(
        "--tess-constraint-name",
        type=str,
        default="tess_transit",
        help="Name to assign to the injected TESS bandpass constraint",
    )
    tess_group.add_argument(
        "--tess-photon-weighted",
        action="store_true",
        help="Mark the generated TESS bandpass constraint as photon-weighted",
    )
    tess_group.add_argument(
        "--tess-bandpass-tbl-output",
        type=str,
        default=None,
        help=(
            "Optional path to also write the fitted TESS bandpass constraint as a "
            "NASA-style .tbl (canonical location: input/phot/<mode>/<planet>/...)"
        ),
    )

    # Inference parameters
    inference_group = parser.add_argument_group("Inference Parameters")
    inference_group.add_argument(
        "--svi-steps",
        type=int,
        default=None,
        help="Number of SVI steps (default: from config)"
    )
    inference_group.add_argument(
        "--svi-learning-rate",
        type=float,
        default=None,
        help="SVI learning rate (default: from config)"
    )
    inference_group.add_argument(
        "--svi-lr-decay-steps",
        type=int,
        default=None,
        help="Use exponential learning-rate decay with this many decay steps"
    )
    inference_group.add_argument(
        "--svi-lr-decay-rate",
        type=float,
        default=None,
        help="Use exponential learning-rate decay with this multiplicative decay rate"
    )
    inference_group.add_argument(
        "--no-svi-lr-decay",
        action="store_true",
        help="Disable SVI exponential learning-rate decay, even if enabled in config"
    )
    inference_group.add_argument(
        "--mcmc-warmup",
        type=int,
        default=None,
        help="Number of MCMC warmup steps (default: from config)"
    )
    inference_group.add_argument(
        "--mcmc-samples",
        type=int,
        default=None,
        help="Number of MCMC samples (default: from config)"
    )
    inference_group.add_argument(
        "--mcmc-chains",
        type=int,
        default=None,
        help="Number of MCMC chains (default: from config)"
    )
    inference_group.add_argument(
        "--mcmc-chain-method",
        type=str,
        choices=["parallel", "sequential", "vectorized"],
        default=None,
        help="NumPyro MCMC chain execution mode (default: from config)"
    )
    inference_group.add_argument(
        "--require-gpu-per-chain",
        action="store_true",
        help=(
            "Fail unless at least one GPU is visible for each requested MCMC "
            "chain when using parallel chain execution"
        )
    )
    inference_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (100 SVI steps, 100 MCMC samples)"
    )

    # Opacity options
    opacity_group = parser.add_argument_group("Opacity")
    opacity_group.add_argument(
        "--load-opacities",
        action="store_true",
        default=None,
        help="Load saved opacities (faster)"
    )
    opacity_group.add_argument(
        "--build-opacities",
        action="store_true",
        help="Force rebuild opacities from databases"
    )
    opacity_group.add_argument(
        "--save-opacities",
        action="store_true",
        help="Save computed opacities for future use"
    )

    # HITRAN credentials (for HITEMP downloads)
    hitran_group = parser.add_argument_group("HITRAN")
    hitran_group.add_argument(
        "--hitran-username",
        type=str,
        default=None,
        help="HITRAN username/email (or set HITRAN_USERNAME)"
    )
    hitran_group.add_argument(
        "--hitran-password",
        type=str,
        default=None,
        help="HITRAN password (or set HITRAN_PASSWORD)"
    )

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--pt-profile",
        type=str,
        choices=["guillot", "isothermal", "gradient", "madhu_seager", "free", "pspline", "gp"],
        default=config.PT_PROFILE_DEFAULT,
        help=f"P-T profile type (default: {config.PT_PROFILE_DEFAULT})"
    )
    model_group.add_argument(
        "--chemistry-model",
        type=str,
        choices=["constant", "free", "fastchem_hybrid_grid"],
        default=config.CHEMISTRY_MODEL_DEFAULT,
        help=(
            "Chemistry/composition model "
            f"(default: {config.CHEMISTRY_MODEL_DEFAULT})"
        ),
    )
    model_group.add_argument(
        "--fastchem-parameter-file",
        type=str,
        default=None,
        help="Path to FastChem parameters.dat (required for fastchem_hybrid_grid)",
    )
    model_group.add_argument(
        "--phoenix-spectrum-path",
        type=str,
        default=None,
        help=(
            "Optional local two-column ASCII PHOENIX stellar spectrum for emission mode "
            "(wavelength_A, stellar_surface_flux). If omitted, emission mode can "
            "auto-fetch PHOENIX spectra through chromatic-lightcurves."
        ),
    )
    model_group.add_argument(
        "--nlayer",
        type=int,
        default=None,
        help=f"Number of atmospheric layers (default: {config.NLAYER})",
    )
    model_group.add_argument(
        "--n-spectral-points",
        type=int,
        default=None,
        help=f"Number of spectral grid points (default: {config.N_SPECTRAL_POINTS})",
    )

    # Phase analysis options
    phase_group = parser.add_argument_group("Phase Analysis")
    phase_group.add_argument(
        "--phase-mode",
        type=str,
        choices=["global", "per_exposure", "linear"],
        default=config.DEFAULT_PHASE_MODE,
        help=f"How to model phase-dependent velocity offset dRV (default: {config.DEFAULT_PHASE_MODE})"
    )
    phase_group.add_argument(
        "--phase-bin",
        type=str,
        choices=["T12", "T23", "T34"],
        default=None,
        help="Run retrieval on specific phase bin only"
    )
    phase_group.add_argument(
        "--all-phase-bins",
        action="store_true",
        help="Run separate retrievals for all phase bins (T12, T23, T34)"
    )

    # Species selection (runtime overrides)
    species_group = parser.add_argument_group("Species Selection")
    species_group.add_argument(
        "--atoms",
        type=str,
        default=None,
        help='Comma-separated atomic species list (e.g., "Fe I,Fe II,Na I")'
    )
    species_group.add_argument(
        "--molecules",
        type=str,
        default=None,
        help='Comma-separated molecular species list (e.g., "H2O,CO,TiO")'
    )
    species_group.add_argument(
        "--no-molecules",
        action="store_true",
        help="Disable all molecular opacities (use atoms only)"
    )
    species_group.add_argument(
        "--no-atoms",
        action="store_true",
        help="Disable all atomic opacities (use molecules only)"
    )
    species_group.add_argument(
        "--all-species",
        action="store_true",
        help="Use all available species instead of the default detected subset"
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--skip-svi",
        action="store_true",
        help="Skip SVI warm-up, go straight to MCMC"
    )
    exec_group.add_argument(
        "--svi-only",
        action="store_true",
        help="Run only SVI, skip MCMC"
    )
    exec_group.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plotting (faster)"
    )
    exec_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Info
    parser.add_argument(
        "--version",
        action="version",
        version="KELT-20b Retrieval v1.0.0"
    )

    return parser


def load_custom_config(config_path):
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from: {config_path}")
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)

    return custom_config


def apply_custom_config(custom_config):
    for name in dir(custom_config):
        if name.isupper():
            config.set_runtime_config(name, getattr(custom_config, name))

    return config


def apply_cli_overrides(args):
    # Planet and ephemeris selection
    if args.planet:
        config.set_runtime_config("PLANET", args.planet)
    if args.ephemeris:
        config.set_runtime_config("EPHEMERIS", args.ephemeris)

    # Retrieval mode and observing mode must be synced before derived paths are built.
    if args.mode:
        config.set_runtime_config("RETRIEVAL_MODE", args.mode)
    if args.wavelength_range:
        config.set_runtime_config("OBSERVING_MODE", args.wavelength_range)
    if args.resolution_mode:
        config.set_runtime_config("RESOLUTION_MODE", args.resolution_mode)

    # Validate planet/ephemeris combination
    params = config.get_params()  # Will raise if invalid

    # Chemistry model
    if args.chemistry_model:
        config.set_runtime_config("CHEMISTRY_MODEL_DEFAULT", args.chemistry_model)
    if args.fastchem_parameter_file:
        config.set_runtime_config("FASTCHEM_PARAMETER_FILE", args.fastchem_parameter_file)
    if args.nlayer is not None:
        if args.nlayer < 1:
            raise ValueError("--nlayer must be >= 1.")
        config.set_runtime_config("NLAYER", args.nlayer)
    if args.n_spectral_points is not None:
        if args.n_spectral_points < 1:
            raise ValueError("--n-spectral-points must be >= 1.")
        config.set_runtime_config("N_SPECTRAL_POINTS", args.n_spectral_points)

    # Output directory (auto-set based on planet/ephemeris/mode)
    if args.output:
        config.set_runtime_config("DIR_SAVE", args.output)
    else:
        config.set_runtime_config("DIR_SAVE", config.get_output_dir())
    os.makedirs(config.DIR_SAVE, exist_ok=True)

    # HITRAN credentials (for HITEMP downloads)
    if args.hitran_username:
        os.environ["HITRAN_USERNAME"] = args.hitran_username
    if args.hitran_password:
        os.environ["HITRAN_PASSWORD"] = args.hitran_password

    # Data directory
    # NOTE: --wavelength-range full has no single on-disk DATA_DIR because the
    # red and blue arms are stored separately and loaded as two spectroscopic
    # components. Leave DATA_DIR unset in that case; run_retrieval() reads per-
    # arm directories via config.get_full_arm_data_dirs().
    if config.OBSERVING_MODE == "full":
        config.set_runtime_config("DATA_DIR", None)
        config.set_runtime_config("TRANSMISSION_DATA", None)
        config.set_runtime_config("EMISSION_DATA", None)
    else:
        config.set_runtime_config("DATA_DIR", config.get_data_dir(epoch=args.epoch))
        config.set_runtime_config("TRANSMISSION_DATA", config.get_transmission_paths(epoch=args.epoch))
        config.set_runtime_config("EMISSION_DATA", config.get_emission_paths(epoch=args.epoch))

    # Quick mode
    if args.quick:
        config.set_runtime_config("SVI_NUM_STEPS", config.QUICK_SVI_STEPS)
        config.set_runtime_config("MCMC_NUM_WARMUP", config.QUICK_MCMC_WARMUP)
        config.set_runtime_config("MCMC_NUM_SAMPLES", config.QUICK_MCMC_SAMPLES)
        config.set_runtime_config("MCMC_NUM_CHAINS", config.QUICK_MCMC_CHAINS)
        print(
            f"Quick mode: {config.QUICK_SVI_STEPS} SVI steps, "
            f"{config.QUICK_MCMC_SAMPLES} MCMC samples"
        )

    # Inference parameters
    if args.svi_steps is not None:
        config.set_runtime_config("SVI_NUM_STEPS", args.svi_steps)
    if args.svi_learning_rate is not None:
        if args.svi_learning_rate <= 0:
            raise ValueError("--svi-learning-rate must be > 0.")
        config.set_runtime_config("SVI_LEARNING_RATE", args.svi_learning_rate)
    if args.no_svi_lr_decay:
        config.set_runtime_config("SVI_LR_DECAY_STEPS", None)
        config.set_runtime_config("SVI_LR_DECAY_RATE", None)
    if args.svi_lr_decay_steps is not None:
        if args.svi_lr_decay_steps < 1:
            raise ValueError("--svi-lr-decay-steps must be >= 1.")
        config.set_runtime_config("SVI_LR_DECAY_STEPS", args.svi_lr_decay_steps)
    if args.svi_lr_decay_rate is not None:
        if not (0 < args.svi_lr_decay_rate < 1):
            raise ValueError("--svi-lr-decay-rate must be between 0 and 1.")
        config.set_runtime_config("SVI_LR_DECAY_RATE", args.svi_lr_decay_rate)
    if (config.SVI_LR_DECAY_STEPS is None) != (config.SVI_LR_DECAY_RATE is None):
        raise ValueError(
            "SVI exponential decay requires both SVI_LR_DECAY_STEPS and "
            "SVI_LR_DECAY_RATE to be set."
        )
    if args.mcmc_warmup is not None:
        config.set_runtime_config("MCMC_NUM_WARMUP", args.mcmc_warmup)
    if args.mcmc_samples is not None:
        config.set_runtime_config("MCMC_NUM_SAMPLES", args.mcmc_samples)
    if args.mcmc_chains is not None:
        config.set_runtime_config("MCMC_NUM_CHAINS", args.mcmc_chains)
    if args.mcmc_chain_method is not None:
        config.set_runtime_config("MCMC_CHAIN_METHOD", args.mcmc_chain_method)
    if args.require_gpu_per_chain:
        config.set_runtime_config("MCMC_REQUIRE_GPU_PER_CHAIN", True)
    if config.MCMC_REQUIRE_GPU_PER_CHAIN and config.MCMC_CHAIN_METHOD != "parallel":
        raise ValueError("--require-gpu-per-chain requires --mcmc-chain-method parallel.")

    # Opacity options
    if args.build_opacities:
        config.set_runtime_config("OPA_LOAD", False)
        config.set_runtime_config("OPA_SAVE", True)
    elif args.load_opacities:
        config.set_runtime_config("OPA_LOAD", True)
    if args.save_opacities:
        config.set_runtime_config("OPA_SAVE", True)

    # Species selection
    def _parse_csv(value: str) -> list[str]:
        parts = []
        for part in re.split(r"[,\n]+", value):
            stripped = part.strip()
            if stripped:
                parts.append(stripped)
        return parts

    # Apply default species filter unless --all-species or explicit selection
    use_defaults = (
        config.USE_DEFAULT_SPECIES
        and not args.all_species
        and not args.atoms
        and not args.molecules
        and not args.no_atoms
        and not args.no_molecules
    )

    if use_defaults:
        default_atoms = set(config.DEFAULT_SPECIES.get("atoms", []))
        default_mols = set(config.DEFAULT_SPECIES.get("molecules", []))
        default_atomic_species = {}
        for k, v in config.ATOMIC_SPECIES.items():
            if k in default_atoms:
                default_atomic_species[k] = v
        default_molpath_hitemp = {}
        for k, v in config.MOLPATH_HITEMP.items():
            if k in default_mols:
                default_molpath_hitemp[k] = v
        default_molpath_exomol = {}
        for k, v in config.MOLPATH_EXOMOL.items():
            if k in default_mols:
                default_molpath_exomol[k] = v
        config.set_runtime_config("ATOMIC_SPECIES", default_atomic_species)
        config.set_runtime_config("MOLPATH_HITEMP", default_molpath_hitemp)
        config.set_runtime_config("MOLPATH_EXOMOL", default_molpath_exomol)
        print(f"Using default detected species (pass --all-species for full set)")

    if args.no_molecules:
        config.set_runtime_config("MOLPATH_HITEMP", {})
        config.set_runtime_config("MOLPATH_EXOMOL", {})
        if args.molecules:
            print("Warning: --no-molecules overrides --molecules.")
    elif args.molecules:
        wanted = set(_parse_csv(args.molecules))
        mol_h = {}
        for k, v in config.MOLPATH_HITEMP.items():
            if k in wanted:
                mol_h[k] = v
        mol_e = {}
        for k, v in config.MOLPATH_EXOMOL.items():
            if k in wanted:
                mol_e[k] = v
        missing = wanted - set(mol_h.keys()) - set(mol_e.keys())
        if missing:
            print(f"Warning: Unknown molecules ignored: {', '.join(sorted(missing))}")
        config.set_runtime_config("MOLPATH_HITEMP", mol_h)
        config.set_runtime_config("MOLPATH_EXOMOL", mol_e)

    if args.no_atoms:
        config.set_runtime_config("ATOMIC_SPECIES", {})
        if args.atoms:
            print("Warning: --no-atoms overrides --atoms.")
    elif args.atoms:
        wanted = set(_parse_csv(args.atoms))
        atoms = {}
        for k, v in config.ATOMIC_SPECIES.items():
            if k in wanted:
                atoms[k] = v
        missing = wanted - set(atoms.keys())
        if missing:
            print(f"Warning: Unknown atoms ignored: {', '.join(sorted(missing))}")
        config.set_runtime_config("ATOMIC_SPECIES", atoms)

    return config


def _is_finite_number(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_tess_quality_bitmask(value):
    text = str(value).strip()
    if not text:
        raise ValueError("--tess-quality-bitmask cannot be empty.")
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    allowed = {"none", "default", "hard", "hardest"}
    if text.lower() not in allowed:
        raise ValueError(
            "--tess-quality-bitmask must be one of none/default/hard/hardest "
            "or an integer bitmask."
        )
    return text.lower()


def _require_tess_value(cli_value, fallback_value, *, cli_flag: str, config_key: str) -> float:
    if _is_finite_number(cli_value):
        return float(cli_value)
    if _is_finite_number(fallback_value):
        return float(fallback_value)
    raise ValueError(
        f"Missing {config_key} for TESS transit fit. Pass {cli_flag} or add a finite "
        f"{config_key!r} entry to the active planet config."
    )


def _resolve_tess_t0_btjd(args, params) -> float:
    if _is_finite_number(args.tess_t0_btjd):
        return float(args.tess_t0_btjd)
    if _is_finite_number(args.tess_t0_bjd):
        return float(args.tess_t0_bjd) - TESS_BTJD_OFFSET
    if _is_finite_number(params.get("epoch")):
        return float(params["epoch"]) - TESS_BTJD_OFFSET
    raise ValueError(
        "Missing transit mid-time for TESS transit fit. Pass --tess-t0-btjd, "
        "--tess-t0-bjd, or add a finite 'epoch' to the active planet config."
    )


def _build_tess_transit_fit_config(args, params, tess_module):
    if args.mode != "transmission":
        raise ValueError("--fit-tess-transit is only supported for transmission retrievals.")

    config_kwargs = {
        "target": args.tess_target or args.planet,
        "period_d": _require_tess_value(
            args.tess_period_d,
            params.get("period"),
            cli_flag="--tess-period-d",
            config_key="period",
        ),
        "t0_btjd": _resolve_tess_t0_btjd(args, params),
        "transit_duration_d": _require_tess_value(
            args.tess_duration_d,
            params.get("duration"),
            cli_flag="--tess-duration-d",
            config_key="duration",
        ),
        "radius_ratio_guess": _require_tess_value(
            args.tess_radius_ratio_guess,
            params.get("rp_rs"),
            cli_flag="--tess-radius-ratio-guess",
            config_key="rp_rs",
        ),
        "impact_guess": _require_tess_value(
            args.tess_impact_guess,
            params.get("b"),
            cli_flag="--tess-impact-guess",
            config_key="b",
        ),
        "mission": args.tess_mission,
        "author": args.tess_author,
        "exptime_s": int(args.tess_exptime_s),
        "quality_bitmask": _parse_tess_quality_bitmask(args.tess_quality_bitmask),
        "flux_column": str(args.tess_flux_column).strip(),
        "planet_name": args.planet,
        "reference": args.ephemeris or config.EPHEMERIS,
        "note": "Generated from raw TESS transit photometry via atmo_retrieval CLI",
    }
    if args.tess_sector:
        config_kwargs["sectors"] = tuple(int(sector) for sector in args.tess_sector)
    if args.tess_mlexo_root:
        config_kwargs["mlexo_root"] = args.tess_mlexo_root

    optional_numeric = {
        "rho_star_solar_guess": args.tess_rho_star_solar_guess,
        "rho_star_solar_sigma": args.tess_rho_star_solar_sigma,
        "model_window_d": args.tess_model_window_d,
        "plot_window_d": args.tess_plot_window_d,
        "flatten_window_length": args.tess_flatten_window_length,
        "outlier_sigma": args.tess_outlier_sigma,
        "emcee_nwalkers_min": args.tess_emcee_nwalkers_min,
        "emcee_burnin_steps": args.tess_emcee_burnin_steps,
        "emcee_production_steps": args.tess_emcee_production_steps,
        "emcee_thin": args.tess_emcee_thin,
    }
    for key, value in optional_numeric.items():
        if value is not None:
            config_kwargs[key] = value
    if args.tess_emcee_use_pool:
        config_kwargs["emcee_use_pool"] = True

    return tess_module.TessTransitFitConfig(**config_kwargs)


def _fit_tess_transit_constraint(args, params):
    tess_module = importlib.import_module("dataio.tess_photometry")
    fit_config = _build_tess_transit_fit_config(args, params, tess_module)

    sector_text = (
        ", ".join(str(sector) for sector in fit_config.sectors)
        if fit_config.sectors
        else "all available"
    )
    print("\nFitting raw TESS transit photometry for retrieval constraint...")
    print(f"  Target: {fit_config.target}")
    print(f"  Sectors: {sector_text}")
    print(f"  Cadence: {fit_config.exptime_s} s")
    print(f"  Quality bitmask: {fit_config.quality_bitmask}")
    print(f"  Flux column: {fit_config.flux_column}")
    print(f"  Observable export: {args.tess_observable}")

    result = tess_module.fit_tess_transit_to_bandpass_constraint(
        fit_config,
        observable=args.tess_observable,
        constraint_name=args.tess_constraint_name,
        photon_weighted=True if args.tess_photon_weighted else None,
        tbl_path=args.tess_bandpass_tbl_output,
    )
    constraint = result.bandpass_constraint

    print(
        "  Fitted constraint: "
        f"{constraint['observable']} = {constraint['value']:.8f} +/- {constraint['sigma']:.8f}"
    )
    if args.tess_bandpass_tbl_output:
        print(f"  Wrote TESS bandpass .tbl: {args.tess_bandpass_tbl_output}")

    return constraint


def print_config_summary(config, args):
    params = config.get_params()
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")
    print(f"  Period: {params['period']} days")
    print(f"  R_p: {params['R_p']} R_J")
    print(f"  T_star: {params['T_star']} K")

    print(f"\nMode: {config.RETRIEVAL_MODE.upper()}")
    print(f"Config profile: {config.get_runtime_profile_name()}")
    print(f"Chemistry model: {args.chemistry_model}")
    if args.chemistry_model == "fastchem_hybrid_grid":
        fc_file = args.fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
        print(f"FastChem parameter file: {fc_file}")
    print(f"Phase mode: {args.phase_mode}")
    if args.phase_bin:
        print(f"Phase bin: {args.phase_bin}")
    if args.all_phase_bins:
        print(f"Phase bins: T12, T23, T34 (all)")
    print(f"Output directory: {config.DIR_SAVE}")
    print(f"\nObserving mode: {config.OBSERVING_MODE}")
    wav_min, wav_max = config.get_wavelength_range()
    print(f"Wavelength range: {wav_min}-{wav_max} Angstroms")
    print(f"Resolution mode: {config.RESOLUTION_MODE}")
    print(f"Resolution: R = {config.get_resolution():,}")
    if args.fit_tess_transit:
        sectors = ", ".join(str(sector) for sector in (args.tess_sector or [])) or "all available"
        print(f"\nTESS transit fit:")
        print(f"  Lightkurve target: {args.tess_target or args.planet}")
        print(f"  Sectors: {sectors}")
        print(f"  Cadence: {args.tess_exptime_s} s")
        print(f"  Quality bitmask: {_parse_tess_quality_bitmask(args.tess_quality_bitmask)}")
        print(f"  Flux column: {args.tess_flux_column}")
        print(f"  Observable export: {args.tess_observable}")
        if args.tess_bandpass_tbl_output:
            print(f"  .tbl export: {args.tess_bandpass_tbl_output}")

    print(f"\nInference:")
    print(f"  SVI steps: {config.SVI_NUM_STEPS:,}")
    print(f"  SVI learning rate: {config.SVI_LEARNING_RATE}")
    if config.SVI_LR_DECAY_STEPS is not None and config.SVI_LR_DECAY_RATE is not None:
        print(
            "  SVI LR schedule: "
            f"exponential decay (steps={config.SVI_LR_DECAY_STEPS}, "
            f"rate={config.SVI_LR_DECAY_RATE})"
        )
    if not args.svi_only:
        print(f"  MCMC warmup: {config.MCMC_NUM_WARMUP:,}")
        print(f"  MCMC samples: {config.MCMC_NUM_SAMPLES:,}")
        print(f"  MCMC chains: {config.MCMC_NUM_CHAINS}")
        print(f"  MCMC chain method: {config.MCMC_CHAIN_METHOD}")
        if config.MCMC_REQUIRE_GPU_PER_CHAIN:
            print("  MCMC GPU policy: require >= 1 visible GPU per chain")
    print(f"  Vsys handling: fixed at systemic velocity = {params['RV_abs']} km/s")

    print(f"\nAtmosphere:")
    print(f"  Layers: {config.NLAYER}")
    print(f"  Pressure: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature: {config.T_LOW}-{config.T_HIGH} K")
    print(f"  Spectral grid points: {config.N_SPECTRAL_POINTS:,}")

    print(f"\nMolecules:")
    for mol in list(config.MOLPATH_HITEMP.keys()) + list(config.MOLPATH_EXOMOL.keys()):
        print(f"  • {mol}")

    print("="*70 + "\n")


def main():
    parser = create_parser()
    args = parser.parse_args()

    runtime_config = config
    runtime_config.apply_runtime_profile(args.profile)

    # Load config
    if args.config:
        print(f"Loading custom config: {args.config}")
        custom_config = load_custom_config(args.config)
        runtime_config = apply_custom_config(custom_config)
    # Apply CLI overrides
    runtime_config = apply_cli_overrides(args)

    # Print configuration summary
    print_config_summary(runtime_config, args)

    joint_spectra = []
    for tbl_path in args.joint_spectrum_tbl or []:
        joint_spectra.append(make_joint_spectrum_component_from_tbl(tbl_path))

    bandpass_constraints = []
    if args.fit_tess_transit:
        bandpass_constraints.append(_fit_tess_transit_constraint(args, runtime_config.get_params()))
    for tbl_path in args.bandpass_tbl or []:
        bandpass_constraints.extend(make_bandpass_constraints_from_tbl(tbl_path))

    # Run retrieval
    if args.all_phase_bins:
        run_phase_binned_retrieval(
            phase_bins=["T12", "T23", "T34"],
            mode=args.mode,
            epoch=args.epoch,
            data_format=args.data_format,
            skip_svi=args.skip_svi,
            svi_only=args.svi_only,
            no_plots=args.no_plots,
            pt_profile=args.pt_profile or runtime_config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
        )

    elif args.phase_bin:
        run_phase_binned_retrieval(
            phase_bins=[args.phase_bin],
            mode=args.mode,
            epoch=args.epoch,
            data_format=args.data_format,
            skip_svi=args.skip_svi,
            svi_only=args.svi_only,
            no_plots=args.no_plots,
            pt_profile=args.pt_profile or runtime_config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
        )

    elif args.mode == "transmission":
        run_retrieval(
            mode="transmission",
            epoch=args.epoch,
            data_format=args.data_format,
            skip_svi=args.skip_svi,
            svi_only=args.svi_only,
            no_plots=args.no_plots,
            pt_profile=args.pt_profile or runtime_config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
            phoenix_spectrum_path=args.phoenix_spectrum_path,
        )

    elif args.mode == "emission":
        run_retrieval(
            mode="emission",
            epoch=args.epoch,
            data_format=args.data_format,
            skip_svi=args.skip_svi,
            svi_only=args.svi_only,
            no_plots=args.no_plots,
            pt_profile=args.pt_profile or runtime_config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
            phoenix_spectrum_path=args.phoenix_spectrum_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
