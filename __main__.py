import argparse
import sys
import os
from pathlib import Path


def create_parser():
    """Create argument parser for CLI."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    # Target selection
    target_group = parser.add_argument_group("Target")
    target_group.add_argument(
        "--planet",
        type=str,
        default=None,
        help="Target planet (default: KELT-20b). Use --list-planets to see options"
    )
    target_group.add_argument(
        "--ephemeris",
        type=str,
        default=None,
        help="Ephemeris source (default: Duck24). Use --list-ephemerides to see options"
    )
    target_group.add_argument(
        "--list-planets",
        action="store_true",
        help="List available planets and exit"
    )
    target_group.add_argument(
        "--list-ephemerides",
        action="store_true",
        help="List available ephemerides for the selected planet and exit"
    )

    # Retrieval mode
    parser.add_argument("--mode", type=str, choices=["transmission", "emission"])

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, default=None, help="Path to custom config file (default: use config.py)")
    config_group.add_argument("--output", type=str, default=None, help="Output directory (default: output/{planet}/{ephemeris}/{mode})")

    # Data options
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-dir", type=str, default=None, help="Override data directory path")
    data_group.add_argument("--wavelength-range", type=str, choices=["blue", "green", "red", "full"], default=None, help="Wavelength range mode (default: from config)")

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

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--temperature-profile",
        type=str,
        choices=["isothermal", "gradient", "madhu_seager", "free"],
        default=None,
        help="Temperature profile type (default: isothermal for transmission)"
    )
    model_group.add_argument(
        "--enable-tellurics",
        action="store_true",
        default=None,
        help="Enable telluric modeling"
    )
    model_group.add_argument(
        "--disable-tellurics",
        action="store_true",
        help="Disable telluric modeling"
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

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet output (errors only)"
    )
    output_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write output to log file"
    )

    # Info
    parser.add_argument(
        "--version",
        action="version",
        version="KELT-20b Retrieval v1.0.0"
    )

    return parser


def load_custom_config(config_path):
    """Load custom configuration file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)

    return custom_config


def apply_cli_overrides(args):
    """Apply command-line argument overrides to config."""
    import config

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

    # Output directory (auto-set based on planet/ephemeris/mode)
    if args.output:
        config.DIR_SAVE = args.output
    else:
        config.DIR_SAVE = config.get_output_dir()
    os.makedirs(config.DIR_SAVE, exist_ok=True)

    # Data directory
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    else:
        config.DATA_DIR = config.get_data_dir()
        config.TRANSMISSION_DATA = config.get_transmission_paths()
        config.EMISSION_DATA = config.get_emission_paths()

    # Wavelength range
    if args.wavelength_range:
        config.OBSERVING_MODE = args.wavelength_range
        config.WAV_MIN, config.WAV_MAX = config.WAVELENGTH_RANGES[args.wavelength_range]

    # Quick mode
    if args.quick:
        config.SVI_NUM_STEPS = 100
        config.MCMC_NUM_WARMUP = 100
        config.MCMC_NUM_SAMPLES = 100
        config.MCMC_NUM_CHAINS = 1
        print("🚀 Quick mode: 100 SVI steps, 100 MCMC samples")

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

    # Tellurics
    if args.enable_tellurics:
        config.ENABLE_TELLURICS = True
    elif args.disable_tellurics:
        config.ENABLE_TELLURICS = False

    return config


def setup_logging(args):
    """Setup logging based on verbosity."""
    import logging

    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]

    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      KELT-20b Ultra-Hot Jupiter Atmospheric Retrieval            ║
║                   ExoJAX + NumPyro                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config_summary(config, args):
    """Print configuration summary."""
    params = config.get_params()
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")
    print(f"  Period: {params['period']} days")
    print(f"  R_p: {params['R_p']} R_J")
    print(f"  T_star: {params['T_star']} K")

    print(f"\nMode: {config.RETRIEVAL_MODE.upper()}")
    print(f"Output directory: {config.DIR_SAVE}")
    print(f"\nObserving mode: {config.OBSERVING_MODE}")
    print(f"Wavelength range: {config.WAV_MIN}-{config.WAV_MAX} nm")
    print(f"Resolution: R = {config.RESOLUTION:,}")

    print(f"\nInference:")
    print(f"  SVI steps: {config.SVI_NUM_STEPS:,}")
    if not args.svi_only:
        print(f"  MCMC warmup: {config.MCMC_NUM_WARMUP:,}")
        print(f"  MCMC samples: {config.MCMC_NUM_SAMPLES:,}")
        print(f"  MCMC chains: {config.MCMC_NUM_CHAINS}")

    print(f"\nAtmosphere:")
    print(f"  Layers: {config.NLAYER}")
    print(f"  Pressure: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature: {config.TLOW}-{config.THIGH} K")

    print(f"\nMolecules:")
    for mol in list(config.MOLPATH_HITEMP.keys()) + list(config.MOLPATH_EXOMOL.keys()):
        print(f"  • {mol}")

    if config.ENABLE_TELLURICS:
        print(f"\nTelluric correction: ENABLED")

    print("="*70 + "\n")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle info commands first (before loading full config)
    if args.list_planets:
        import config
        print("\nAvailable planets:")
        for planet in config.list_planets():
            ephems = config.list_ephemerides(planet)
            print(f"  {planet}: {', '.join(ephems)}")
        return 0

    if args.list_ephemerides:
        import config
        planet = args.planet or config.PLANET
        print(f"\nAvailable ephemerides for {planet}:")
        for ephem in config.list_ephemerides(planet):
            print(f"  {ephem}")
        return 0

    # Setup logging
    logger = setup_logging(args)

    # Print banner
    if not args.quiet:
        print_banner()

    # Load config
    if args.config:
        logger.info(f"Loading custom config: {args.config}")
        config = load_custom_config(args.config)
    else:
        import config

    # Apply CLI overrides
    config = apply_cli_overrides(args)

    # Print configuration summary
    if not args.quiet:
        print_config_summary(config, args)

    # Run retrieval
    try:
        if args.mode == "transmission":
            from retrieval import run_transmission_retrieval

            logger.info("Starting transmission retrieval...")
            run_transmission_retrieval(
                skip_svi=args.skip_svi,
                svi_only=args.svi_only,
                no_plots=args.no_plots,
                temperature_profile=args.temperature_profile or "isothermal",
                seed=args.seed,
            )

        elif args.mode == "emission":
            from retrieval import run_emission_retrieval

            logger.info("Starting emission retrieval...")
            run_emission_retrieval(
                skip_svi=args.skip_svi,
                svi_only=args.svi_only,
                no_plots=args.no_plots,
                temperature_profile=args.temperature_profile or "madhu_seager",
                seed=args.seed,
            )

        elif args.mode == "combined":
            logger.error("Combined retrieval not yet implemented")
            sys.exit(1)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
