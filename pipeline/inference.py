import os
from typing import Callable
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


def create_prior_guide(
    Mp_mean: float,
    Mp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
) -> Callable:
    def prior_guide(data: jnp.ndarray, sigma: jnp.ndarray, phase: jnp.ndarray, **kwargs) -> dict:
        Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0))
        Rstar = numpyro.sample("Rstar", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0))
        return {"Mp": Mp, "Rstar": Rstar}

    return prior_guide


def build_guide(
    model_c: Callable,
    Mp_mean: float,
    Mp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
) -> AutoGuideList:
    guide = AutoGuideList(model_c)
    prior_guide = create_prior_guide(Mp_mean, Mp_std, Rstar_mean, Rstar_std)
    guide.append(prior_guide)

    model_hidden = handlers.block(model_c, hide=["Mp", "Rstar", "Rp"])
    guide.append(AutoMultivariateNormal(model_hidden))

    return guide


def save_svi_outputs(
    params: dict,
    losses: jnp.ndarray,
    init_values: dict,
    output_dir: str,
) -> None:
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
    model_c: Callable,
    rng_key: jax.Array,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    phase: jnp.ndarray,
    Mp_mean: float,
    Mp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
    output_dir: str,
    num_steps: int = 1000,
    lr: float = 0.005,
) -> tuple[dict, jnp.ndarray, Callable, dict, AutoGuideList]:
    guide = build_guide(model_c, Mp_mean, Mp_std, Rstar_mean, Rstar_std)
    optimizer = optim.Adam(lr)
    svi = SVI(model_c, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(
        rng_key,
        num_steps,
        data=data,
        sigma=sigma,
        phase=phase,
    )

    params = svi_result.params
    losses = svi_result.losses

    svi_median = guide[-1].median(params)
    svi_median.update({"Mp": Mp_mean, "Rstar": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, output_dir)

    return params, losses, init_strategy, svi_median, guide


def run_mcmc(
    model_c: Callable,
    rng_key: jax.Array,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    phase: jnp.ndarray,
    init_strategy: Callable,
    output_dir: str,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    max_tree_depth: int = 5,
    num_chains: int = 1,
) -> tuple[MCMC, dict]:
    kernel = NUTS(
        model_c,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, data=data, sigma=sigma, phase=phase)

    mcmc.print_summary()
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()

    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)

    return mcmc, posterior_sample


def generate_predictions(
    model_c: Callable,
    rng_key: jax.Array,
    posterior_sample: dict,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    phase: jnp.ndarray,
    output_dir: str,
) -> dict:
    pred = Predictive(model_c, posterior_sample, return_sites=["Rp"])
    predictions = pred(rng_key, data=data, sigma=sigma, phase=phase)

    jnp.save(os.path.join(output_dir, "rp_pred"), predictions["Rp"])

    return predictions
