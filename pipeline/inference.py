import os
from time import perf_counter
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_optimizers

from numpyro.infer import SVI, Trace_ELBO
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro import handlers
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList
from numpyro.infer.initialization import init_to_value, init_to_median


def create_prior_guide(
    Mp_mean: float,
    Mp_std: float,
    Mp_upper_3sigma: float | None,
    Rp_mean: float,
    Rp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
) -> Callable:
    def prior_guide(*args, **kwargs) -> dict:
        del args, kwargs
        if Mp_upper_3sigma is not None:
            Mp = numpyro.sample("Mp", dist.Uniform(0.5, Mp_upper_3sigma))
        else:
            Mp = numpyro.sample("Mp", dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0))
        Rp = numpyro.sample("Rp", dist.TruncatedNormal(Rp_mean, Rp_std, low=0.5))
        Rstar = numpyro.sample("Rstar", dist.TruncatedNormal(Rstar_mean, Rstar_std, low=0.0))
        return {"Mp": Mp, "Rp": Rp, "Rstar": Rstar}

    return prior_guide


def build_guide(
    model_c: Callable,
    Mp_mean: float,
    Mp_std: float,
    Mp_upper_3sigma: float | None,
    Rp_mean: float,
    Rp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
) -> AutoGuideList:
    guide = AutoGuideList(model_c)
    prior_guide = create_prior_guide(
        Mp_mean,
        Mp_std,
        Mp_upper_3sigma,
        Rp_mean,
        Rp_std,
        Rstar_mean,
        Rstar_std,
    )
    guide.append(prior_guide)

    def _hide_from_autoguide(site: dict) -> bool:
        if site["type"] != "sample":
            return True
        if site.get("is_observed", False):
            return True
        return site["name"] in {"Mp", "Rstar", "Rp"}

    model_hidden = handlers.block(model_c, hide_fn=_hide_from_autoguide)
    guide.append(AutoMultivariateNormal(model_hidden, init_loc_fn=init_to_median))

    return guide


def save_svi_outputs(
    params: dict,
    losses: jnp.ndarray,
    init_values: dict,
    output_dir: str,
) -> None:
    params_cpu = {}
    for k, v in params.items():
        params_cpu[k] = np.asarray(jax.device_get(v))
    losses_cpu = np.asarray(jax.device_get(losses))
    init_cpu = {}
    for k, v in init_values.items():
        init_cpu[k] = np.asarray(jax.device_get(v))

    np.savez(os.path.join(output_dir, "svi_params.npz"), **params_cpu)
    np.save(os.path.join(output_dir, "svi_losses.npy"), losses_cpu)
    np.savez(os.path.join(output_dir, "svi_init_values.npz"), **init_cpu)

    print(f"SVI params saved to {output_dir}/svi_params.npz", flush=True)
    print(f"SVI losses saved to {output_dir}/svi_losses.npy", flush=True)
    print(f"SVI init values saved to {output_dir}/svi_init_values.npz", flush=True)


def build_svi_optimizer(
    lr: float,
    decay_steps: int | None = None,
    decay_rate: float | None = None,
) -> optim._NumPyroOptim:
    if (decay_steps is None) != (decay_rate is None):
        raise ValueError("SVI learning-rate decay requires both decay_steps and decay_rate.")

    if decay_steps is not None and decay_rate is not None:
        schedule = jax_optimizers.exponential_decay(lr, decay_steps, decay_rate)
        return optim.Adam(schedule)

    return optim.Adam(lr)


def _default_svi_report_interval(num_steps: int) -> int:
    """Emit about twenty progress updates across the SVI run."""
    return max(1, num_steps // 20)


def _format_svi_loss(loss: float) -> str:
    if np.isnan(loss):
        return "nan"
    if np.isposinf(loss):
        return "inf"
    if np.isneginf(loss):
        return "-inf"
    return f"{loss:.6g}"


def _mean_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _min_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def _run_svi_with_logging(
    svi: SVI,
    rng_key: jax.Array,
    *,
    num_steps: int,
    model_inputs: dict[str, object],
    report_interval: int | None = None,
) -> tuple[dict, jnp.ndarray]:
    if num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")

    interval = max(1, int(report_interval or _default_svi_report_interval(num_steps)))
    start = perf_counter()

    print("  SVI: JAX compile starting; first update may take a while...", flush=True)
    initial_result = svi.run(
        rng_key,
        1,
        progress_bar=False,
        **model_inputs,
    )
    initial_losses = np.atleast_1d(
        np.asarray(jax.device_get(initial_result.losses), dtype=float)
    )
    loss_chunks: list[np.ndarray] = [initial_losses]
    steps_completed = 1
    last_result = initial_result
    elapsed = perf_counter() - start
    print(
        "  SVI progress: "
        f"{steps_completed}/{num_steps} steps ({100.0 * steps_completed / num_steps:.1f}%), "
        f"loss={_format_svi_loss(float(initial_losses[-1]))}, elapsed={elapsed:.1f}s",
        flush=True,
    )

    while steps_completed < num_steps:
        chunk_steps = min(interval, num_steps - steps_completed)
        chunk_start = steps_completed + 1
        last_result = svi.run(
            rng_key,
            chunk_steps,
            progress_bar=False,
            init_state=last_result.state,
            **model_inputs,
        )
        chunk_losses = np.atleast_1d(
            np.asarray(jax.device_get(last_result.losses), dtype=float)
        )
        loss_chunks.append(chunk_losses)
        steps_completed += chunk_steps
        elapsed = perf_counter() - start
        print(
            "  SVI progress: "
            f"{steps_completed}/{num_steps} steps ({100.0 * steps_completed / num_steps:.1f}%), "
            f"last loss={_format_svi_loss(float(chunk_losses[-1]))}, "
            f"avg loss [{chunk_start}-{steps_completed}]={_format_svi_loss(_mean_finite(chunk_losses))}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    losses_cpu = np.concatenate(loss_chunks, axis=0)
    total_elapsed = perf_counter() - start
    print(
        "  SVI warm-up complete: "
        f"final loss={_format_svi_loss(float(losses_cpu[-1]))}, "
        f"best loss={_format_svi_loss(_min_finite(losses_cpu))}, "
        f"elapsed={total_elapsed:.1f}s",
        flush=True,
    )

    return last_result.params, jnp.asarray(losses_cpu)


def run_svi(
    model_c: Callable,
    rng_key: jax.Array,
    model_inputs: dict[str, object],
    Mp_mean: float,
    Mp_std: float,
    Mp_upper_3sigma: float | None,
    Rp_mean: float,
    Rp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
    output_dir: str,
    num_steps: int = 1000,
    lr: float = 0.005,
    lr_decay_steps: int | None = None,
    lr_decay_rate: float | None = None,
) -> tuple[dict, jnp.ndarray, Callable, dict, AutoGuideList]:
    guide = build_guide(
        model_c,
        Mp_mean,
        Mp_std,
        Mp_upper_3sigma,
        Rp_mean,
        Rp_std,
        Rstar_mean,
        Rstar_std,
    )
    optimizer = build_svi_optimizer(lr, decay_steps=lr_decay_steps, decay_rate=lr_decay_rate)
    svi = SVI(model_c, guide, optimizer, loss=Trace_ELBO())
    params, losses = _run_svi_with_logging(
        svi,
        rng_key,
        num_steps=num_steps,
        model_inputs=model_inputs,
    )

    svi_median = guide[-1].median(params)
    Mp_init = Mp_upper_3sigma / 3.0 if Mp_upper_3sigma is not None else Mp_mean
    svi_median.update({"Mp": Mp_init, "Rp": Rp_mean, "Rstar": Rstar_mean})
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, output_dir)

    return params, losses, init_strategy, svi_median, guide
