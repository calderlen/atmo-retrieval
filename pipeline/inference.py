import os
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp

from numpyro.infer import SVI, Trace_ELBO
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
    def prior_guide(*args, **kwargs) -> dict:
        del args, kwargs
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

    def _hide_from_autoguide(site: dict) -> bool:
        if site["type"] != "sample":
            return True
        if site.get("is_observed", False):
            return True
        return site["name"] in {"Mp", "Rstar", "Rp"}

    model_hidden = handlers.block(model_c, hide_fn=_hide_from_autoguide)
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
    model_inputs: dict[str, object],
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
        **model_inputs,
    )

    params = svi_result.params
    losses = svi_result.losses

    svi_median = guide[-1].median(params)
    svi_median.update({"Mp": Mp_mean, "Rstar": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, output_dir)

    return params, losses, init_strategy, svi_median, guide
