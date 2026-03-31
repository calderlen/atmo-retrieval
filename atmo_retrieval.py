import argparse
import importlib.util
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
        "--bandpass-tbl",
        type=str,
        action="append",
        default=None,
        help="Explicit NASA .tbl file to include as one or more bandpass constraints",
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
        choices=["constant", "fastchem_hybrid_grid"],
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
    exec_group.add_argument(
        "--no-preallocate",
        action="store_true",
        help="Disable JAX GPU preallocation (slower but safer on memory)"
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
            setattr(config, name, getattr(custom_config, name))

    return config


def apply_cli_overrides(args):
    # Planet and ephemeris selection
    if args.planet:
        config.PLANET = args.planet
    if args.ephemeris:
        config.EPHEMERIS = args.ephemeris

    # Validate planet/ephemeris combination
    params = config.get_params()  # Will raise if invalid

    # Retrieval mode
    if args.mode:
        config.RETRIEVAL_MODE = args.mode

    # Chemistry model
    if args.chemistry_model:
        config.CHEMISTRY_MODEL_DEFAULT = args.chemistry_model
    if args.fastchem_parameter_file:
        config.FASTCHEM_PARAMETER_FILE = args.fastchem_parameter_file

    # Output directory (auto-set based on planet/ephemeris/mode)
    if args.output:
        config.DIR_SAVE = args.output
    else:
        config.DIR_SAVE = config.get_output_dir()
    os.makedirs(config.DIR_SAVE, exist_ok=True)

    # HITRAN credentials (for HITEMP downloads)
    if args.hitran_username:
        os.environ["HITRAN_USERNAME"] = args.hitran_username
    if args.hitran_password:
        os.environ["HITRAN_PASSWORD"] = args.hitran_password

    # Data directory
    config.DATA_DIR = config.get_data_dir(epoch=args.epoch)
    config.TRANSMISSION_DATA = config.get_transmission_paths(epoch=args.epoch)
    config.EMISSION_DATA = config.get_emission_paths(epoch=args.epoch)

    # Wavelength range / observing mode
    if args.wavelength_range:
        config.OBSERVING_MODE = args.wavelength_range

    # Quick mode
    if args.quick:
        config.SVI_NUM_STEPS = config.QUICK_SVI_STEPS
        config.MCMC_NUM_WARMUP = config.QUICK_MCMC_WARMUP
        config.MCMC_NUM_SAMPLES = config.QUICK_MCMC_SAMPLES
        config.MCMC_NUM_CHAINS = config.QUICK_MCMC_CHAINS
        print(
            f"🚀 Quick mode: {config.QUICK_SVI_STEPS} SVI steps, "
            f"{config.QUICK_MCMC_SAMPLES} MCMC samples"
        )

    # Inference parameters
    if args.svi_steps is not None:
        config.SVI_NUM_STEPS = args.svi_steps
    if args.mcmc_warmup is not None:
        config.MCMC_NUM_WARMUP = args.mcmc_warmup
    if args.mcmc_samples is not None:
        config.MCMC_NUM_SAMPLES = args.mcmc_samples
    if args.mcmc_chains is not None:
        config.MCMC_NUM_CHAINS = args.mcmc_chains

    # Opacity options
    if args.build_opacities:
        config.OPA_LOAD = False
        config.OPA_SAVE = True
    elif args.load_opacities:
        config.OPA_LOAD = True
    if args.save_opacities:
        config.OPA_SAVE = True

    # Species selection
    def _parse_csv(value: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"[,\n]+", value) if p.strip()]
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
        config.ATOMIC_SPECIES = {
            k: v for k, v in config.ATOMIC_SPECIES.items() if k in default_atoms
        }
        config.MOLPATH_HITEMP = {
            k: v for k, v in config.MOLPATH_HITEMP.items() if k in default_mols
        }
        config.MOLPATH_EXOMOL = {
            k: v for k, v in config.MOLPATH_EXOMOL.items() if k in default_mols
        }
        print(f"Using default detected species (pass --all-species for full set)")

    if args.no_molecules:
        config.MOLPATH_HITEMP = {}
        config.MOLPATH_EXOMOL = {}
        if args.molecules:
            print("Warning: --no-molecules overrides --molecules.")
    elif args.molecules:
        wanted = set(_parse_csv(args.molecules))
        mol_h = {k: v for k, v in config.MOLPATH_HITEMP.items() if k in wanted}
        mol_e = {k: v for k, v in config.MOLPATH_EXOMOL.items() if k in wanted}
        missing = wanted - set(mol_h.keys()) - set(mol_e.keys())
        if missing:
            print(f"Warning: Unknown molecules ignored: {', '.join(sorted(missing))}")
        config.MOLPATH_HITEMP = mol_h
        config.MOLPATH_EXOMOL = mol_e

    if args.no_atoms:
        config.ATOMIC_SPECIES = {}
        if args.atoms:
            print("Warning: --no-atoms overrides --atoms.")
    elif args.atoms:
        wanted = set(_parse_csv(args.atoms))
        atoms = {k: v for k, v in config.ATOMIC_SPECIES.items() if k in wanted}
        missing = wanted - set(atoms.keys())
        if missing:
            print(f"Warning: Unknown atoms ignored: {', '.join(sorted(missing))}")
        config.ATOMIC_SPECIES = atoms

    return config
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
    print(f"Resolution: R = {config.get_resolution():,}")

    print(f"\nInference:")
    print(f"  SVI steps: {config.SVI_NUM_STEPS:,}")
    if not args.svi_only:
        print(f"  MCMC warmup: {config.MCMC_NUM_WARMUP:,}")
        print(f"  MCMC samples: {config.MCMC_NUM_SAMPLES:,}")
        print(f"  MCMC chains: {config.MCMC_NUM_CHAINS}")

    print(f"\nAtmosphere:")
    print(f"  Layers: {config.NLAYER}")
    print(f"  Pressure: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature: {config.T_LOW}-{config.T_HIGH} K")

    print(f"\nMolecules:")
    for mol in list(config.MOLPATH_HITEMP.keys()) + list(config.MOLPATH_EXOMOL.keys()):
        print(f"  • {mol}")

    print("="*70 + "\n")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.no_preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    # Load config
    if args.config:
        print(f"Loading custom config: {args.config}")
        custom_config = load_custom_config(args.config)
        config = apply_custom_config(custom_config)
    # Apply CLI overrides
    config = apply_cli_overrides(args)

    # Print configuration summary
    print_config_summary(config, args)

    joint_spectra = []
    for tbl_path in args.joint_spectrum_tbl or []:
        joint_spectra.append(make_joint_spectrum_component_from_tbl(tbl_path))

    bandpass_constraints = []
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
            pt_profile=args.pt_profile or config.PT_PROFILE_DEFAULT,
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
            pt_profile=args.pt_profile or config.PT_PROFILE_DEFAULT,
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
            pt_profile=args.pt_profile or config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
        )

    elif args.mode == "emission":
        run_retrieval(
            mode="emission",
            epoch=args.epoch,
            data_format=args.data_format,
            skip_svi=args.skip_svi,
            svi_only=args.svi_only,
            no_plots=args.no_plots,
            pt_profile=args.pt_profile or config.PT_PROFILE_DEFAULT,
            phase_mode=args.phase_mode,
            chemistry_model=args.chemistry_model,
            fastchem_parameter_file=args.fastchem_parameter_file,
            seed=args.seed,
            joint_spectra=joint_spectra or None,
            bandpass_constraints=bandpass_constraints or None,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
