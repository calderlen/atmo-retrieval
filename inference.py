"""
Inference Module
================

Stochastic Variational Inference (SVI) and HMC-NUTS sampling.
Generic for any retrieval model.
"""

import os
from contextlib import redirect_stdout
import numpy as np
import jax
import jax.numpy as jnp

from numpyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro import handlers
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList
from numpyro.infer.initialization import init_to_value


def create_prior_guide(Mp_mean, Mp_std, Rstar_mean, Rstar_std):
    """
    Create prior guide for Mp and Rs during SVI.

    Parameters
    ----------
    Mp_mean : float
        Planet mass prior mean [M_J]
    Mp_std : float
        Planet mass prior std [M_J]
    Rstar_mean : float
        Stellar radius prior mean [R_Sun]
    Rstar_std : float
        Stellar radius prior std [R_Sun]

    Returns
    -------
    prior_guide : callable
        Guide function
    """
    def prior_guide(rp_mean, rp_std):
        """Guide for Mp and Rs so they follow their priors during SVI."""
        Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0))
        Rs = numpyro.sample("Rs", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0))
        return {"Mp": Mp, "Rs": Rs}

    return prior_guide


def build_guide(model_c, Mp_mean, Mp_std, Rstar_mean, Rstar_std):
    """
    Construct AutoGuideList with separated priors for Mp/Rs.

    Parameters
    ----------
    model_c : callable
        NumPyro model
    Mp_mean : float
        Planet mass prior mean
    Mp_std : float
        Planet mass prior std
    Rstar_mean : float
        Stellar radius prior mean
    Rstar_std : float
        Stellar radius prior std

    Returns
    -------
    guide : AutoGuideList
        Guide for SVI
    """
    guide = AutoGuideList(model_c)
    prior_guide = create_prior_guide(Mp_mean, Mp_std, Rstar_mean, Rstar_std)
    guide.append(prior_guide)

    # Hide deterministic sites and prior sites from Auto guide
    model_hidden = handlers.block(model_c, hide=["Mp", "Rs", "rp_mu"])
    guide.append(AutoMultivariateNormal(model_hidden))

    return guide


def save_svi_outputs(params, losses, init_values, output_dir):
    """
    Save SVI results to disk.

    Parameters
    ----------
    params : dict
        SVI parameters
    losses : jnp.ndarray
        Loss values
    init_values : dict
        Initial values for HMC
    output_dir : str
        Output directory
    """
    params_cpu = {k: np.asarray(jax.device_get(v)) for k, v in params.items()}
    losses_cpu = np.asarray(jax.device_get(losses))
    init_cpu = {k: np.asarray(jax.device_get(v)) for k, v in init_values.items()}

    np.savez(os.path.join(output_dir, "svi_params.npz"), **params_cpu)
    np.save(os.path.join(output_dir, "svi_losses.npy"), losses_cpu)
    np.savez(os.path.join(output_dir, "svi_init_values.npz"), **init_cpu)

    print(f"SVI params saved to {output_dir}/svi_params.npz")
    print(f"SVI losses saved to {output_dir}/svi_losses.npy")
    print(f"SVI init values saved to {output_dir}/svi_init_values.npz")


def run_svi(
    model_c,
    rng_key,
    rp_mean,
    rp_std,
    Mp_mean,
    Mp_std,
    Rstar_mean,
    Rstar_std,
    output_dir,
    num_steps=1000,
    lr=0.005,
):
    """
    Run Stochastic Variational Inference.

    Parameters
    ----------
    model_c : callable
        NumPyro model
    rng_key : jax.random.PRNGKey
        Random key
    rp_mean : jnp.ndarray
        Observed spectrum mean
    rp_std : jnp.ndarray
        Observed spectrum uncertainty
    Mp_mean : float
        Planet mass prior mean
    Mp_std : float
        Planet mass prior std
    Rstar_mean : float
        Stellar radius prior mean
    Rstar_std : float
        Stellar radius prior std
    output_dir : str
        Output directory
    num_steps : int
        Number of SVI steps
    lr : float
        Learning rate

    Returns
    -------
    params : dict
        SVI parameters
    losses : jnp.ndarray
        Loss trajectory
    init_strategy : callable
        Initialization strategy for HMC
    svi_median : dict
        Median parameter values
    guide : AutoGuideList
        SVI guide
    """
    guide = build_guide(model_c, Mp_mean, Mp_std, Rstar_mean, Rstar_std)
    optimizer = optim.Adam(lr)
    svi = SVI(model_c, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(
        rng_key,
        num_steps,
        rp_mean=rp_mean,
        rp_std=rp_std,
    )

    params = svi_result.params
    losses = svi_result.losses

    # Median of AutoMVN in constrained space
    svi_median = guide[-1].median(params)
    # Anchor Mp and Rs to prior means for HMC
    svi_median.update({"Mp": Mp_mean, "Rs": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, output_dir)

    return params, losses, init_strategy, svi_median, guide


def run_mcmc(
    model_c,
    rng_key,
    rp_mean,
    rp_std,
    init_strategy,
    output_dir,
    num_warmup=1000,
    num_samples=1000,
    max_tree_depth=5,
    num_chains=1,
):
    """
    Run HMC-NUTS sampling.

    Parameters
    ----------
    model_c : callable
        NumPyro model
    rng_key : jax.random.PRNGKey
        Random key
    rp_mean : jnp.ndarray
        Observed spectrum mean
    rp_std : jnp.ndarray
        Observed spectrum uncertainty
    init_strategy : callable
        Initialization strategy
    output_dir : str
        Output directory
    num_warmup : int
        Warmup iterations
    num_samples : int
        Sampling iterations
    max_tree_depth : int
        Maximum tree depth for NUTS
    num_chains : int
        Number of parallel chains

    Returns
    -------
    mcmc : MCMC
        MCMC object with results
    posterior_sample : dict
        Posterior samples
    """
    kernel = NUTS(
        model_c,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, rp_mean=rp_mean, rp_std=rp_std)

    # Print and save summary
    mcmc.print_summary()
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()

    # Save posterior samples
    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)

    return mcmc, posterior_sample


def generate_predictions(model_c, rng_key, posterior_sample, rp_std, output_dir):
    """
    Generate predictive spectrum from posterior samples.

    Parameters
    ----------
    model_c : callable
        NumPyro model
    rng_key : jax.random.PRNGKey
        Random key
    posterior_sample : dict
        Posterior samples from MCMC
    rp_std : jnp.ndarray
        Observed spectrum uncertainty
    output_dir : str
        Output directory

    Returns
    -------
    predictions : dict
        Predictive samples
    """
    pred = Predictive(model_c, posterior_sample, return_sites=["rp"])
    predictions = pred(rng_key, rp_mean=None, rp_std=rp_std)

    jnp.save(os.path.join(output_dir, "rp_pred"), predictions["rp"])

    return predictions
