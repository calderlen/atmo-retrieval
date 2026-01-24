"""
WASP-39 b Transmission Spectrum Retrieval - Main Script
========================================================

Orchestrates the complete retrieval pipeline using modular components.

See Section 7.2 of https://arxiv.org/abs/2410.06900 for details.
"""

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

from numpyro.infer import Predictive

from exojax.rt import ArtTransPure

# Import local modules
import config
from data_loader import load_observed_spectrum, ResolutionInterpolator
from grid_setup import setup_wavenumber_grid, setup_spectral_operators
from opacity_setup import setup_cia_opacities, load_molecular_opacities
from forward_model import create_transmission_model
from inference import run_svi, run_mcmc, generate_predictions
from plotting import create_all_plots


def main():
    """Main retrieval pipeline."""

    print("="*70)
    print("WASP-39 b Transmission Spectrum Retrieval")
    print("ExoJAX + NumPyro HMC-NUTS")
    print("="*70)

    # -------------------------------------------------------------------------
    # 1. Load observed data
    # -------------------------------------------------------------------------
    print("\n[1/8] Loading observed spectrum...")
    wav_obs, rp_mean, rp_std, inst_nus = load_observed_spectrum(
        config.WAV_OBS_PATH,
        config.RP_MEAN_PATH,
        config.RP_STD_PATH,
    )
    print(f"  Loaded {len(wav_obs)} spectral points")
    print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} nm")

    # -------------------------------------------------------------------------
    # 2. Setup instrumental resolution
    # -------------------------------------------------------------------------
    print("\n[2/8] Setting up instrumental resolution...")
    res_interp = ResolutionInterpolator(config.RESOLUTION_CURVE_PATH)
    Rinst = res_interp(np.mean(wav_obs))
    print(f"  Mean resolving power: R ≈ {Rinst:.0f}")

    # -------------------------------------------------------------------------
    # 3. Setup wavenumber grid and spectral operators
    # -------------------------------------------------------------------------
    print("\n[3/8] Building wavenumber grid...")
    nu_grid, wav_grid, res_high = setup_wavenumber_grid(
        np.min(wav_obs) - config.WAV_MIN_OFFSET,
        np.max(wav_obs) + config.WAV_MAX_OFFSET,
        config.N_SPECTRAL_POINTS,
        unit="nm",
    )

    sop_rot, sop_inst, beta_inst = setup_spectral_operators(nu_grid, Rinst)
    print("  Spectral operators initialized")

    # -------------------------------------------------------------------------
    # 4. Setup atmospheric radiative transfer
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 5. Load opacities
    # -------------------------------------------------------------------------
    print("\n[5/8] Loading opacities...")

    # CIA opacities
    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    print(f"  Loaded {len(opa_cias)} CIA opacity sources")

    # Molecular opacities
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
    print(f"  Loaded {len(opa_mols)} molecular opacity sources")
    print(f"  Molecules: {list(opa_mols.keys())}")

    # -------------------------------------------------------------------------
    # 6. Build forward model
    # -------------------------------------------------------------------------
    print("\n[6/8] Building probabilistic forward model...")
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
    )
    print("  Forward model created")

    # -------------------------------------------------------------------------
    # 7. Run inference
    # -------------------------------------------------------------------------
    rng_key = random.PRNGKey(0)

    # SVI warm-up
    print("\n[7/8] Running Stochastic Variational Inference...")
    print(f"  Steps: {config.SVI_NUM_STEPS}, Learning rate: {config.SVI_LEARNING_RATE}")
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

    # HMC-NUTS
    print("\n  Running HMC-NUTS sampling...")
    print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
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
    )

    # Generate predictions
    print("\n  Generating predictive spectrum...")
    rng_key, rng_key_ = random.split(rng_key)
    predictions = generate_predictions(
        model_c, rng_key_, posterior_sample, rp_std, config.DIR_SAVE
    )

    # -------------------------------------------------------------------------
    # 8. Create diagnostic plots
    # -------------------------------------------------------------------------
    print("\n[8/8] Creating diagnostic plots...")

    # Generate SVI prediction
    rng_key, rng_plot = random.split(rng_key)
    svi_pred = Predictive(
        model_c, params=svi_median, num_samples=1, return_sites=["rp_mu"]
    )
    svi_mu = svi_pred(rng_plot, rp_mean=rp_mean, rp_std=rp_std)["rp_mu"][0]

    # Sample from SVI guide for corner plot
    rng_key, rng_svi = random.split(rng_key)
    svi_samples = svi_guide[-1].sample_posterior(
        rng_svi,
        svi_params,
        rp_mean=rp_mean,
        rp_std=rp_std,
        sample_shape=(1000,),
    )

    create_all_plots(
        losses=losses,
        wav_obs=wav_obs,
        rp_mean=rp_mean,
        rp_std=rp_std,
        predictions=predictions,
        svi_mu=svi_mu,
        posterior_sample=posterior_sample,
        svi_samples=svi_samples,
        opa_mols=opa_mols,
        output_dir=config.DIR_SAVE,
    )

    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {config.DIR_SAVE}/")
    print("="*70)


if __name__ == "__main__":
    main()
