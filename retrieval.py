"""
KELT-20b Ultra-Hot Jupiter Retrieval - Main Script
===================================================

Orchestrates transmission/emission retrieval pipeline.

Can be run directly or via CLI (__main__.py).
"""

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

# ExoJAX imports
from exojax.rt import ArtTransPure, ArtEmisPure

# Local modules
import config
from data_loader import load_observed_spectrum, ResolutionInterpolator
from grid_setup import setup_wavenumber_grid, setup_spectral_operators
from opacity_setup import setup_cia_opacities, load_molecular_opacities
from transmission_model import create_transmission_model
from emission_model import create_emission_model
from inference import run_svi, run_mcmc, generate_predictions
from plotting import create_transmission_plots


def run_transmission_retrieval(
    skip_svi=False,
    svi_only=False,
    no_plots=False,
    temperature_profile="isothermal",
    seed=42,
):
    """
    Run transmission spectrum retrieval.

    Parameters
    ----------
    skip_svi : bool
        Skip SVI warm-up
    svi_only : bool
        Run only SVI, skip MCMC
    no_plots : bool
        Skip plotting
    temperature_profile : str
        Temperature profile type
    seed : int
        Random seed
    """

    print("="*70)
    print("KELT-20b Transmission Spectrum Retrieval")
    print("PEPSI/LBT")
    print("="*70)

    # Load observed data
    print("\n[1/8] Loading observed transmission spectrum...")
    wav_obs, rp_mean, rp_std, inst_nus = load_observed_spectrum(
        config.TRANSMISSION_DATA["wavelength"],
        config.TRANSMISSION_DATA["spectrum"],
        config.TRANSMISSION_DATA["uncertainty"],
    )
    print(f"  Loaded {len(wav_obs)} spectral points")
    print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} nm")

    # Setup instrumental resolution
    print("\n[2/8] Setting up instrumental resolution...")
    res_interp = ResolutionInterpolator(constant_R=config.PEPSI_RESOLUTION)
    Rinst = res_interp(np.mean(wav_obs))
    print(f"  PEPSI resolving power: R = {Rinst:.0f}")

    # Setup wavenumber grid
    print("\n[3/8] Building wavenumber grid...")
    nu_grid, wav_grid, res_high = setup_wavenumber_grid(
        config.WAV_MIN - config.WAV_MIN_OFFSET,
        config.WAV_MAX + config.WAV_MAX_OFFSET,
        config.N_SPECTRAL_POINTS,
        unit="nm",
    )

    sop_rot, sop_inst, beta_inst = setup_spectral_operators(
        nu_grid, Rinst, vsini_max=150.0, vrmax=500.0
    )
    print("  Spectral operators initialized")

    # Setup atmospheric RT
    print("\n[4/8] Initializing atmospheric RT...")
    art = ArtTransPure(
        pressure_top=config.PRESSURE_TOP,
        pressure_btm=config.PRESSURE_BTM,
        nlayer=config.NLAYER,
    )
    art.change_temperature_range(config.TLOW, config.THIGH)
    print(f"  {config.NLAYER} atmospheric layers")
    print(f"  Pressure range: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
    print(f"  Temperature range: {config.TLOW:.0f} - {config.THIGH:.0f} K")

    # Load opacities
    print("\n[5/8] Loading opacities...")

    # CIA
    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    print(f"  Loaded {len(opa_cias)} CIA sources")

    # Molecules
    opa_mols, molmass_arr = load_molecular_opacities(
        config.MOLPATH_HITEMP,
        config.MOLPATH_EXOMOL,
        nu_grid,
        config.OPA_LOAD,
        config.NDIV,
        config.DIFFMODE,
        config.TLOW,
        config.THIGH,
    )
    print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

    # Build forward model
    print(f"\n[6/8] Building transmission forward model ({temperature_profile})...")
    model_c = create_transmission_model(
        art=art,
        opa_mols=opa_mols,
        opa_cias=opa_cias,
        molmass_arr=molmass_arr,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        beta_inst=beta_inst,
        inst_nus=inst_nus,
        period_day=config.PERIOD_DAY,
        Mp_mean=config.MP_MEAN,
        Mp_std=config.MP_STD,
        Rstar_mean=config.RSTAR_MEAN,
        Rstar_std=config.RSTAR_STD,
        Tlow=config.TLOW,
        Thigh=config.THIGH,
        pressure_top=config.PRESSURE_TOP,
        pressure_btm=config.PRESSURE_BTM,
        nlayer=config.NLAYER,
        cloud_width=config.CLOUD_WIDTH,
        cloud_integrated_tau=config.CLOUD_INTEGRATED_TAU,
        temperature_profile=temperature_profile,
    )
    print("  Forward model created")

    # Run inference
    rng_key = random.PRNGKey(seed)

    # SVI
    svi_params = None
    svi_median = None
    svi_guide = None
    losses = None
    init_strategy = None

    if not skip_svi:
        print("\n[7/8] Running Stochastic Variational Inference...")
        print(f"  Steps: {config.SVI_NUM_STEPS}, LR: {config.SVI_LEARNING_RATE}")
        rng_key, rng_key_ = random.split(rng_key)

        svi_params, losses, init_strategy, svi_median, svi_guide = run_svi(
            model_c=model_c,
            rng_key=rng_key_,
            rp_mean=rp_mean,
            rp_std=rp_std,
            Mp_mean=config.MP_MEAN,
            Mp_std=config.MP_STD,
            Rstar_mean=config.RSTAR_MEAN,
            Rstar_std=config.RSTAR_STD,
            output_dir=config.DIR_SAVE,
            num_steps=config.SVI_NUM_STEPS,
            lr=config.SVI_LEARNING_RATE,
        )
        print(f"  Final SVI loss: {float(losses[-1]):.2f}")

    if svi_only:
        print("\n  SVI-only mode: skipping MCMC")
        print(f"\n  Results saved to: {config.DIR_SAVE}/")
        return

    # HMC-NUTS
    print("\n  Running HMC-NUTS sampling...")
    print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
    print(f"  Chains: {config.MCMC_NUM_CHAINS}")
    rng_key, rng_key_ = random.split(rng_key)

    mcmc, posterior_sample = run_mcmc(
        model_c=model_c,
        rng_key=rng_key_,
        rp_mean=rp_mean,
        rp_std=rp_std,
        init_strategy=init_strategy,
        output_dir=config.DIR_SAVE,
        num_warmup=config.MCMC_NUM_WARMUP,
        num_samples=config.MCMC_NUM_SAMPLES,
        max_tree_depth=config.MCMC_MAX_TREE_DEPTH,
        num_chains=config.MCMC_NUM_CHAINS,
    )

    # Predictions
    print("\n  Generating predictive spectrum...")
    rng_key, rng_key_ = random.split(rng_key)
    predictions = generate_predictions(
        model_c, rng_key_, posterior_sample, rp_std, config.DIR_SAVE
    )

    # Plotting
    if not no_plots:
        print("\n[8/8] Creating diagnostic plots...")

        # SVI prediction
        if svi_median is not None:
            rng_key, rng_plot = random.split(rng_key)
            svi_pred = Predictive(
                model_c, params=svi_median, num_samples=1, return_sites=["rp_mu"]
            )
            svi_mu = svi_pred(rng_plot, rp_mean=rp_mean, rp_std=rp_std)["rp_mu"][0]

            rng_key, rng_svi = random.split(rng_key)
            svi_samples = svi_guide[-1].sample_posterior(
                rng_svi,
                svi_params,
                rp_mean=rp_mean,
                rp_std=rp_std,
                sample_shape=(1000,),
            )
        else:
            svi_mu = None
            svi_samples = None

        create_transmission_plots(
            losses=losses if losses is not None else [],
            wav_obs=wav_obs,
            rp_mean=rp_mean,
            rp_std=rp_std,
            predictions=predictions,
            svi_mu=svi_mu,
            posterior_sample=posterior_sample,
            svi_samples=svi_samples,
            opa_mols=opa_mols,
            art=art,
            output_dir=config.DIR_SAVE,
        )
    else:
        print("\n[8/8] Skipping plots (--no-plots)")

    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {config.DIR_SAVE}/")
    print("="*70)


def run_emission_retrieval(
    skip_svi=False,
    svi_only=False,
    no_plots=False,
    temperature_profile="madhu_seager",
    seed=42,
):
    """
    Run emission spectrum retrieval.

    Parameters
    ----------
    skip_svi : bool
        Skip SVI warm-up
    svi_only : bool
        Run only SVI, skip MCMC
    no_plots : bool
        Skip plotting
    temperature_profile : str
        Temperature profile type
    seed : int
        Random seed
    """
    print("Emission retrieval not yet implemented.")
    print("Please use transmission mode for now.")
    # TODO: Implement similar to transmission but with ArtEmisPure


if __name__ == "__main__":
    # If run directly (not via CLI), use default settings
    print("Running with default settings.")
    print("For more options, use: python -m uhj_atmo_retrieval --help\n")

    # Select retrieval mode
    if config.RETRIEVAL_MODE == "transmission":
        run_transmission_retrieval()
    elif config.RETRIEVAL_MODE == "emission":
        run_emission_retrieval()
    elif config.RETRIEVAL_MODE == "combined":
        print("Combined retrieval not yet implemented.")
    else:
        raise ValueError(f"Unknown retrieval mode: {config.RETRIEVAL_MODE}")
