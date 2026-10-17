from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import jax
from jax import random
from jax.example_libraries import optimizers as jax_optimizers
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList
from numpyro.infer.initialization import init_to_value, init_to_median
import numpyro.optim as optim

import config
from pipeline.mcmc_diagnostics import DEFAULT_MCMC_EXTRA_FIELDS


class _IndependentPriorGuide:
    """Keep selected shared-system guide sites equal to their model priors."""

    def __init__(
        self,
        distributions: dict[str, dist.Distribution],
        centers: dict[str, float],
    ) -> None:
        self.distributions = dict(distributions)
        self.centers = dict(centers)

    def __call__(self, *args, **kwargs) -> dict[str, jnp.ndarray]:
        del args, kwargs
        return {
            name: numpyro.sample(name, distribution)
            for name, distribution in self.distributions.items()
        }

    def sample_posterior(
        self,
        rng_key: jax.Array,
        params: dict,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> dict[str, jnp.ndarray]:
        del params
        if not self.distributions:
            return {}
        keys = jax.random.split(rng_key, len(self.distributions))
        return {
            name: distribution.sample(key, sample_shape=sample_shape)
            for (name, distribution), key in zip(self.distributions.items(), keys)
        }

    def median(self, params: dict) -> dict[str, jnp.ndarray]:
        del params
        return {name: jnp.asarray(value) for name, value in self.centers.items()}

    def quantiles(
        self,
        params: dict,
        quantiles: list[float] | jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        del params
        q = jnp.asarray(quantiles)
        return {
            name: distribution.icdf(q)
            for name, distribution in self.distributions.items()
        }


def create_prior_guide(
    Mp_mean: float,
    Mp_std: float,
    Mp_upper_3sigma: float | None,
    Rp_mean: float,
    Rp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
    prior_modes: dict[str, str] | None = None,
) -> _IndependentPriorGuide:
    prior_modes = dict(prior_modes or {})
    distributions: dict[str, dist.Distribution] = {}
    centers: dict[str, float] = {}
    if prior_modes.get("Mp") != "fixed":
        if prior_modes.get("Mp") == "upper_limit" or (
            "Mp" not in prior_modes and Mp_upper_3sigma is not None
        ):
            distributions["Mp"] = dist.Uniform(0.5, Mp_upper_3sigma)
            centers["Mp"] = Mp_upper_3sigma / 3.0
        else:
            distributions["Mp"] = dist.TruncatedNormal(Mp_mean, Mp_std, low=0.0)
            centers["Mp"] = Mp_mean
    if prior_modes.get("Rp", "normal") != "fixed":
        distributions["Rp"] = dist.TruncatedNormal(Rp_mean, Rp_std, low=0.5)
        centers["Rp"] = Rp_mean
    if prior_modes.get("Rstar", "normal") != "fixed":
        distributions["Rstar"] = dist.TruncatedNormal(
            Rstar_mean,
            Rstar_std,
            low=0.0,
        )
        centers["Rstar"] = Rstar_mean
    return _IndependentPriorGuide(distributions, centers)


def build_guide(
    model_c: Callable,
    Mp_mean: float,
    Mp_std: float,
    Mp_upper_3sigma: float | None,
    Rp_mean: float,
    Rp_std: float,
    Rstar_mean: float,
    Rstar_std: float,
    prior_modes: dict[str, str] | None = None,
) -> AutoGuideList:
    prior_modes = dict(prior_modes or {})
    guide = AutoGuideList(model_c, init_loc_fn=init_to_median)
    prior_guide = create_prior_guide(
        Mp_mean,
        Mp_std,
        Mp_upper_3sigma,
        Rp_mean,
        Rp_std,
        Rstar_mean,
        Rstar_std,
        prior_modes,
    )
    guide.append(prior_guide)

    def _hide_from_autoguide(site: dict) -> bool:
        if site["type"] != "sample":
            return True
        if site.get("is_observed", False):
            return True
        return (
            site["name"] in {"Mp", "Rstar", "Rp"}
            and prior_modes.get(
                site["name"],
                "upper_limit"
                if site["name"] == "Mp" and Mp_upper_3sigma is not None
                else "normal",
            )
            != "fixed"
        )

    # Blocking also hides sample messages from the outer seed handler. Seed the
    # partial model explicitly so AutoMultivariateNormal can build its prototype.
    model_hidden = handlers.block(
        handlers.seed(model_c, rng_seed=0),
        hide_fn=_hide_from_autoguide,
    )
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
    prior_modes: dict[str, str] | None = None,
) -> tuple[dict, jnp.ndarray, Callable, dict, AutoGuideList]:
    prior_modes = dict(prior_modes or {})
    guide = build_guide(
        model_c,
        Mp_mean,
        Mp_std,
        Mp_upper_3sigma,
        Rp_mean,
        Rp_std,
        Rstar_mean,
        Rstar_std,
        prior_modes,
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
    prior_init = {"Mp": Mp_init, "Rp": Rp_mean, "Rstar": Rstar_mean}
    svi_median.update(
        {
            name: value
            for name, value in prior_init.items()
            if prior_modes.get(
                name,
                "upper_limit" if name == "Mp" and Mp_upper_3sigma is not None else "normal",
            )
            != "fixed"
        }
    )
    init_strategy = init_to_value(values=svi_median)

    save_svi_outputs(params, losses, svi_median, output_dir)

    return params, losses, init_strategy, svi_median, guide


@dataclass(frozen=True)
class InferenceResult:
    """Sampler state needed for posterior products; mcmc is None for SVI-only."""

    mcmc: MCMC | None
    rng_key: jax.Array
    svi_params: dict | None
    svi_losses: np.ndarray | None
    svi_guide: object | None


def run_inference(
    model_c: Callable,
    model_inputs: dict,
    model_params: dict,
    *,
    output_dir: str | os.PathLike[str],
    seed: int,
    skip_svi: bool,
    svi_only: bool,
    shared_prior_modes: dict[str, str] | None,
    save_mcmc_diagnostics: bool,
) -> InferenceResult:
    """Run the existing SVI warm start and NUTS sampling sequence."""
    rng_key = random.PRNGKey(seed)

    init_strategy = init_to_median(num_samples=config.INIT_TO_MEDIAN_SAMPLES)
    svi_params: dict | None = None
    svi_guide: object | None = None
    svi_losses: np.ndarray | None = None

    if not skip_svi:
        svi_lr_message = f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}"
        if config.SVI_LR_DECAY_STEPS is not None and config.SVI_LR_DECAY_RATE is not None:
            svi_lr_message += (
                " with exponential decay "
                f"(steps={config.SVI_LR_DECAY_STEPS}, rate={config.SVI_LR_DECAY_RATE})"
            )
        print(svi_lr_message)
        rng_key, rng_key_ = random.split(rng_key)
        svi_params, svi_losses, init_strategy, _, svi_guide = run_svi(
            model_c,
            rng_key_,
            model_inputs=model_inputs,
            Mp_mean=model_params["M_p"],
            Mp_std=model_params["M_p_err"],
            Mp_upper_3sigma=model_params.get("M_p_upper_3sigma"),
            Rp_mean=model_params["R_p"],
            Rp_std=model_params["R_p_err"],
            Rstar_mean=model_params["R_star"],
            Rstar_std=model_params["R_star_err"],
            output_dir=str(output_dir),
            num_steps=config.SVI_NUM_STEPS,
            lr=config.SVI_LEARNING_RATE,
            lr_decay_steps=config.SVI_LR_DECAY_STEPS,
            lr_decay_rate=config.SVI_LR_DECAY_RATE,
            prior_modes=shared_prior_modes,
        )

        if svi_only:
            return InferenceResult(None, rng_key, svi_params, svi_losses, svi_guide)

    print(f"\n  Running HMC-NUTS sampling...")
    print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
    print(f"  Chains: {config.MCMC_NUM_CHAINS}")
    print(f"  Chain method: {config.MCMC_CHAIN_METHOD}")

    print(f"  JAX default backend: {jax.default_backend()}")
    print(f"  JAX local devices: {jax.local_devices()}")

    kernel = NUTS(
        model_c,
        max_tree_depth=config.MCMC_MAX_TREE_DEPTH,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=config.MCMC_NUM_WARMUP,
        num_samples=config.MCMC_NUM_SAMPLES,
        num_chains=config.MCMC_NUM_CHAINS,
        chain_method=config.MCMC_CHAIN_METHOD,
    )

    rng_key, rng_key_ = random.split(rng_key)
    mcmc_run_kwargs = dict(model_inputs)
    if save_mcmc_diagnostics:
        mcmc_run_kwargs["extra_fields"] = DEFAULT_MCMC_EXTRA_FIELDS
    mcmc.run(
        rng_key_,
        **mcmc_run_kwargs,
    )
    return InferenceResult(mcmc, rng_key, svi_params, svi_losses, svi_guide)
