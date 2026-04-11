"""End-to-end TESS transit fitting helpers for retrieval bandpass constraints.

This module packages the workflow that previously lived only in the
`*-tess-mlexo.ipynb` notebooks:

1. fetch or accept raw TESS light curves
2. build a joint multi-sector transit dataset
3. fit the transit with the local ``mlexo`` checkout plus per-sector GP noise
4. summarize the posterior
5. emit a retrieval-ready TESS bandpass constraint

The retrieval itself already knows how to consume the final scalar constraint;
this module is the missing bridge from raw photometry to that scalar input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TESS_INSTRUMENT = "TESS CCD Array"
TESS_FACILITY = "Transiting Exoplanet Survey Satellite"
TESS_CENTRAL_WAVELENGTH_MICRON = 0.8
TESS_BANDWIDTH_MICRON = 0.2
MLEXO_G_RSUN3_PER_MSUN_DAY2 = 2942.2062175044193
RHO_SUN_UNIT = 3.0 / (4.0 * np.pi)
DEFAULT_GP_RHO_GUESS_D = 0.05
DEFAULT_GP_EPS = 1.0e-9


@dataclass(frozen=True)
class TessTransitFitConfig:
    """Configuration for an end-to-end TESS transit fit."""

    target: str
    period_d: float
    t0_btjd: float
    transit_duration_d: float
    radius_ratio_guess: float
    impact_guess: float
    rho_star_solar_guess: float = 0.4
    rho_star_solar_sigma: float = 0.10
    mission: str = "TESS"
    author: str = "SPOC"
    exptime_s: int = 120
    quality_bitmask: str | int = "default"
    flux_column: str = "pdcsap_flux"
    sectors: tuple[int, ...] | None = None
    model_window_d: float = 0.15
    plot_window_d: float = 0.15
    lc_order: int = 20
    flatten_window_length: int = 901
    outlier_sigma: float = 5.0
    gp_rho_guess_d: float = DEFAULT_GP_RHO_GUESS_D
    gp_rho_min_d: float | None = None
    gp_rho_max_d: float = 0.30
    gp_eps: float = DEFAULT_GP_EPS
    emcee_nwalkers_min: int = 48
    emcee_burnin_steps: int = 150
    emcee_production_steps: int = 300
    emcee_thin: int = 5
    emcee_init_scale: float = 1.0e-3
    emcee_use_pool: bool = False
    emcee_n_processes: int = field(
        default_factory=lambda: max(1, min(4, (os.cpu_count() or 1) - 1))
    )
    emcee_start_method: str = "fork"
    mlexo_root: str | Path | None = None
    planet_name: str | None = None
    reference: str | None = None
    note: str | None = None

    @property
    def exposure_time_d(self) -> float:
        return float(self.exptime_s) / 86400.0

    @property
    def resolved_gp_rho_min_d(self) -> float:
        if self.gp_rho_min_d is not None:
            return float(self.gp_rho_min_d)
        return max(5.0 * self.exposure_time_d, 0.01)


@dataclass
class TessTransitFitResult:
    """Artifacts from a TESS transit fit and retrieval handoff."""

    config: TessTransitFitConfig
    dataset: dict[str, Any]
    best_fit: dict[str, Any]
    bandpass_constraint: dict[str, Any]


@dataclass(frozen=True)
class _MlexoRuntime:
    """Lazy-loaded runtime pieces from the local mlexo checkout."""

    mx: Any
    lc_transforms: Any
    limb_dark_light_curve: Any
    Body: Any
    Central: Any
    OrbitalBody: Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_mlexo_root() -> Path:
    return _repo_root().parent / "mlexo"


def _coerce_optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _ensure_parent_on_sys_path(path: Path) -> None:
    parent = str(path.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _load_module_or_raise(module_name: str, *, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Required dependency '{module_name}' is not available. {install_hint}"
        ) from exc


def _load_mlexo_runtime(config: TessTransitFitConfig) -> _MlexoRuntime:
    mlexo_root = _coerce_optional_path(config.mlexo_root) or _default_mlexo_root()
    if not mlexo_root.exists():
        raise FileNotFoundError(
            f"Local mlexo checkout not found: {mlexo_root}. "
            "Pass TessTransitFitConfig(mlexo_root=...) if it lives elsewhere."
        )

    _ensure_parent_on_sys_path(mlexo_root)

    mx = _load_module_or_raise(
        "mlx.core",
        install_hint="Install MLX in the active environment before running the fit.",
    )
    lc_transforms = _load_module_or_raise(
        "mlexo.light_curves.transforms",
        install_hint="Ensure the local mlexo checkout is importable.",
    )
    limb_dark = _load_module_or_raise(
        "mlexo.light_curves.limb_dark",
        install_hint="Ensure the local mlexo checkout is importable.",
    )
    keplerian = _load_module_or_raise(
        "mlexo.orbits.keplerian",
        install_hint="Ensure the local mlexo checkout is importable.",
    )

    return _MlexoRuntime(
        mx=mx,
        lc_transforms=lc_transforms,
        limb_dark_light_curve=limb_dark.light_curve,
        Body=keplerian.Body,
        Central=keplerian.Central,
        OrbitalBody=keplerian.OrbitalBody,
    )


def download_tess_lightcurves(config: TessTransitFitConfig) -> Any:
    """Fetch TESS light curves via lightkurve.

    The download explicitly pins both the cadence quality mask and flux column
    so the retrieval input is reproducible across Lightkurve versions and
    author products.
    """

    lk = _load_module_or_raise(
        "lightkurve",
        install_hint="Install lightkurve before downloading TESS light curves.",
    )

    search_kwargs: dict[str, Any] = {
        "mission": config.mission,
        "author": config.author,
        "exptime": config.exptime_s,
    }
    if config.sectors is not None:
        search_kwargs["sector"] = list(config.sectors)

    search = lk.search_lightcurve(config.target, **search_kwargs)
    if len(search) == 0:
        raise RuntimeError(
            f"No TESS light curves found for target={config.target!r} "
            f"mission={config.mission!r} author={config.author!r}."
        )

    collection = search.download_all(
        quality_bitmask=config.quality_bitmask,
        flux_column=config.flux_column,
    )
    if collection is None or len(collection) == 0:
        raise RuntimeError(f"Failed to download TESS light curves for {config.target!r}.")
    return collection


def build_joint_tess_dataset(
    lc_collection: Iterable[Any],
    *,
    period_d: float,
    t0_btjd: float,
    transit_duration_d: float,
    model_window_d: float,
    flatten_window_length: int = 901,
    outlier_sigma: float = 5.0,
) -> dict[str, Any]:
    """Normalize, flatten, and window TESS light curves for a joint transit fit."""

    sector_labels: list[str] = []
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    yerr_parts: list[np.ndarray] = []
    sector_idx_parts: list[np.ndarray] = []

    for sector_number, raw_lc in enumerate(lc_collection, start=1):
        sector_label = raw_lc.meta.get("MISSION", f"sector_{sector_number}")
        sector_lc = raw_lc.remove_nans().normalize()

        transit_mask = sector_lc.create_transit_mask(
            period=period_d,
            transit_time=t0_btjd,
            duration=transit_duration_d,
        )
        _, oot_outlier_mask = sector_lc[~transit_mask].remove_outliers(
            sigma=outlier_sigma,
            return_mask=True,
        )
        keep_mask = np.ones(len(sector_lc), dtype=bool)
        oot_rows = np.flatnonzero(~transit_mask)
        keep_mask[oot_rows[oot_outlier_mask]] = False
        sector_lc = sector_lc[keep_mask]

        transit_mask = sector_lc.create_transit_mask(
            period=period_d,
            transit_time=t0_btjd,
            duration=transit_duration_d,
        )
        sector_flat = sector_lc.flatten(
            window_length=int(flatten_window_length),
            mask=transit_mask,
        )

        sector_phase = (
            (sector_flat.time.value - t0_btjd + 0.5 * period_d) % period_d
        ) - 0.5 * period_d
        window_mask = np.abs(sector_phase) < model_window_d
        if not np.any(window_mask):
            continue

        flux = np.asarray(sector_flat.flux.value, dtype=np.float64)
        flux_err = np.asarray(sector_flat.flux_err.value, dtype=np.float64)

        finite_flux = flux[np.isfinite(flux)]
        flux_fill = float(np.median(finite_flux)) if finite_flux.size else 1.0
        flux = np.where(np.isfinite(flux), flux, flux_fill)

        finite_err = flux_err[np.isfinite(flux_err) & (flux_err > 0)]
        err_fill = float(np.median(finite_err)) if finite_err.size else 5.0e-4
        flux_err = np.where(
            np.isfinite(flux_err) & (flux_err > 0),
            flux_err,
            err_fill,
        )

        x_parts.append(
            np.ascontiguousarray(sector_flat.time.value[window_mask], dtype=np.float64)
        )
        y_parts.append(
            np.ascontiguousarray(flux[window_mask] - 1.0, dtype=np.float64)
        )
        yerr_parts.append(
            np.ascontiguousarray(flux_err[window_mask], dtype=np.float64)
        )
        sector_idx_parts.append(
            np.full(np.count_nonzero(window_mask), len(sector_labels), dtype=np.int32)
        )
        sector_labels.append(str(sector_label))

    if not x_parts:
        raise RuntimeError("No TESS cadences survived the preprocessing window.")

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    yerr = np.concatenate(yerr_parts)
    sector_idx = np.concatenate(sector_idx_parts)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    yerr = yerr[sort_idx]
    sector_idx = sector_idx[sort_idx]

    n_sectors = len(sector_labels)
    sector_counts = np.bincount(sector_idx, minlength=n_sectors)
    sector_rows = [np.flatnonzero(sector_idx == idx) for idx in range(n_sectors)]

    return {
        "time": x,
        "flux": y,
        "flux_err": yerr,
        "sector_idx": sector_idx,
        "sector_labels": sector_labels,
        "sector_counts": sector_counts,
        "sector_rows": sector_rows,
        "n_sectors": n_sectors,
    }


def _kipping_q_to_u(q1: float, q2: float) -> np.ndarray:
    sqrt_q1 = np.sqrt(np.clip(q1, 0.0, 1.0))
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return np.asarray([u1, u2], dtype=np.float64)


def _rho_star_solar_to_a_over_rstar(
    period: float | np.ndarray,
    rho_star_solar: float | np.ndarray,
) -> np.ndarray:
    period_arr = np.asarray(period, dtype=np.float64)
    rho_star_arr = np.asarray(rho_star_solar, dtype=np.float64)
    rho_star_unit = rho_star_arr * RHO_SUN_UNIT
    return (
        (MLEXO_G_RSUN3_PER_MSUN_DAY2 * period_arr**2 * rho_star_unit) / (3.0 * np.pi)
    ) ** (1.0 / 3.0)


def _circular_transit_duration(
    period: float | np.ndarray,
    a_over_rstar: float | np.ndarray,
    impact_param: float | np.ndarray,
    radius_ratio: float | np.ndarray,
) -> np.ndarray:
    chord = np.sqrt(np.clip((1.0 + radius_ratio) ** 2 - impact_param**2, 0.0, None))
    sin_i = np.sqrt(
        np.clip(1.0 - (impact_param / a_over_rstar) ** 2, 1.0e-12, 1.0)
    )
    arg = np.clip(chord / (a_over_rstar * sin_i), 0.0, 1.0)
    return (period / np.pi) * np.arcsin(arg)


def _matern32_kernel_mx(
    time_mx: Any,
    gp_sigma_mx: Any,
    gp_rho_mx: Any,
    runtime: _MlexoRuntime,
    *,
    dtype: Any,
) -> Any:
    mx = runtime.mx
    dt = mx.abs(time_mx[:, None] - time_mx[None, :])
    sqrt3 = mx.sqrt(mx.array(3.0, dtype=dtype))
    scaled = sqrt3 * dt / gp_rho_mx
    return gp_sigma_mx**2 * (1.0 + scaled) * mx.exp(-scaled)


def _gp_cholesky_solve(cov: Any, rhs: Any, runtime: _MlexoRuntime) -> tuple[Any, Any]:
    mx = runtime.mx
    chol = mx.linalg.cholesky(cov, stream=mx.cpu)
    y = mx.linalg.solve_triangular(chol, rhs, upper=False, stream=mx.cpu)
    alpha = mx.linalg.solve_triangular(mx.transpose(chol), y, upper=True, stream=mx.cpu)
    return chol, alpha


def _sector_gp_marginal_nll(
    time: np.ndarray,
    resid: np.ndarray,
    flux_err: np.ndarray,
    gp_sigma: float,
    gp_rho: float,
    jitter: float,
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
) -> float:
    mx = runtime.mx
    dtype = mx.float32
    time_mx = mx.array(time, dtype=dtype)
    resid_mx = mx.array(resid, dtype=dtype)[:, None]
    flux_err_mx = mx.array(flux_err, dtype=dtype)
    gp_sigma_mx = mx.array(gp_sigma, dtype=dtype)
    gp_rho_mx = mx.array(gp_rho, dtype=dtype)
    jitter_mx = mx.array(jitter, dtype=dtype)

    k_signal = _matern32_kernel_mx(
        time_mx,
        gp_sigma_mx,
        gp_rho_mx,
        runtime,
        dtype=dtype,
    )
    diag = flux_err_mx**2 + jitter_mx**2 + mx.array(config.gp_eps, dtype=dtype)
    cov = mx.array(k_signal + mx.diag(diag), dtype=dtype)

    try:
        chol, alpha = _gp_cholesky_solve(cov, resid_mx, runtime)
    except Exception:
        return float(np.inf)

    quad = mx.sum(resid_mx * alpha)
    logdet = 2.0 * mx.sum(mx.log(mx.diag(chol)))
    n_data = float(time.shape[0])
    nll = 0.5 * (quad + logdet + n_data * np.log(2.0 * np.pi))
    return float(np.asarray(nll))


def _sector_gp_posterior_mean(
    time: np.ndarray,
    resid: np.ndarray,
    flux_err: np.ndarray,
    gp_sigma: float,
    gp_rho: float,
    jitter: float,
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
) -> np.ndarray:
    mx = runtime.mx
    dtype = mx.float32
    time_mx = mx.array(time, dtype=dtype)
    resid_mx = mx.array(resid, dtype=dtype)[:, None]
    flux_err_mx = mx.array(flux_err, dtype=dtype)
    gp_sigma_mx = mx.array(gp_sigma, dtype=dtype)
    gp_rho_mx = mx.array(gp_rho, dtype=dtype)
    jitter_mx = mx.array(jitter, dtype=dtype)

    k_signal = _matern32_kernel_mx(
        time_mx,
        gp_sigma_mx,
        gp_rho_mx,
        runtime,
        dtype=dtype,
    )
    diag = flux_err_mx**2 + jitter_mx**2 + mx.array(config.gp_eps, dtype=dtype)
    cov = mx.array(k_signal + mx.diag(diag), dtype=dtype)
    _, alpha = _gp_cholesky_solve(cov, resid_mx, runtime)
    gp_mean = k_signal @ alpha
    return np.asarray(gp_mean, dtype=np.float64).reshape(-1)


def _unpack_params(
    theta: np.ndarray,
    n_sectors: int,
    runtime: _MlexoRuntime | None,
) -> dict[str, Any]:
    theta = np.asarray(theta, dtype=np.float64)
    period = np.exp(theta[0])
    t0 = theta[1]
    rho_star_solar = np.exp(theta[2])
    radius_ratio = np.exp(theta[3])
    impact_scaled = theta[4]
    q1 = theta[5]
    q2 = theta[6]

    mean_flux = theta[7 : 7 + n_sectors]
    log_jitter = theta[7 + n_sectors : 7 + 2 * n_sectors]
    log_gp_sigma = theta[7 + 2 * n_sectors : 7 + 3 * n_sectors]
    log_gp_rho = theta[7 + 3 * n_sectors : 7 + 4 * n_sectors]

    jitter = np.exp(log_jitter)
    gp_sigma = np.exp(log_gp_sigma)
    gp_rho = np.exp(log_gp_rho)

    impact_param = impact_scaled * (1.0 + radius_ratio)
    a_over_rstar = _rho_star_solar_to_a_over_rstar(period, rho_star_solar)
    cos_i = np.clip(impact_param / a_over_rstar, -1.0, 1.0)
    duration = _circular_transit_duration(period, a_over_rstar, impact_param, radius_ratio)

    return {
        "period": period,
        "t0": t0,
        "rho_star_solar": rho_star_solar,
        "rho_star_unit": rho_star_solar * RHO_SUN_UNIT,
        "a_over_rstar": a_over_rstar,
        "duration": duration,
        "inclination_deg": np.degrees(np.arccos(cos_i)),
        "r": radius_ratio,
        "b": impact_param,
        "q1": q1,
        "q2": q2,
        "u": _kipping_q_to_u(q1, q2),
        "mean_flux": mean_flux,
        "log_jitter": log_jitter,
        "jitter": jitter,
        "log_gp_sigma": log_gp_sigma,
        "gp_sigma": gp_sigma,
        "log_gp_rho": log_gp_rho,
        "gp_rho": gp_rho,
    }


def _mlexo_transit_flux(
    time: np.ndarray,
    *,
    period: float,
    t0: float,
    a_over_rstar: float,
    impact_param: float,
    radius_ratio: float,
    limb_darkening_u: np.ndarray,
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
) -> np.ndarray:
    central = runtime.Central.from_orbital_properties(
        period=period,
        semimajor=a_over_rstar,
        radius=1.0,
    )
    body = runtime.Body(
        period=period,
        time_transit=t0,
        impact_param=impact_param,
        radius=radius_ratio,
    )
    orbit = runtime.OrbitalBody(central, body)
    flux_func = runtime.limb_dark_light_curve(orbit, limb_darkening_u, order=config.lc_order)
    if config.exposure_time_d > 0:
        flux_func = runtime.lc_transforms.integrate(
            flux_func,
            exposure_time=config.exposure_time_d,
            order=1,
            num_samples=7,
        )
    flux = flux_func(runtime.mx.array(time))
    return np.asarray(flux, dtype=np.float64)


def _build_gp_trend(
    dataset: Mapping[str, Any],
    transit_model: np.ndarray,
    pars: Mapping[str, Any],
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
) -> np.ndarray:
    gp_trend = np.zeros_like(dataset["flux"])
    for sector_id, rows in enumerate(dataset["sector_rows"]):
        baseline = transit_model[rows] + pars["mean_flux"][sector_id]
        resid = dataset["flux"][rows] - baseline
        gp_trend[rows] = _sector_gp_posterior_mean(
            time=dataset["time"][rows],
            resid=resid,
            flux_err=dataset["flux_err"][rows],
            gp_sigma=pars["gp_sigma"][sector_id],
            gp_rho=pars["gp_rho"][sector_id],
            jitter=pars["jitter"][sector_id],
            runtime=runtime,
            config=config,
        )
    return gp_trend


def _negative_log_posterior(
    theta: np.ndarray,
    dataset: Mapping[str, Any],
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
) -> float:
    pars = _unpack_params(theta, int(dataset["n_sectors"]), runtime)
    transit_model = _mlexo_transit_flux(
        dataset["time"],
        period=pars["period"],
        t0=pars["t0"],
        a_over_rstar=pars["a_over_rstar"],
        impact_param=pars["b"],
        radius_ratio=pars["r"],
        limb_darkening_u=pars["u"],
        runtime=runtime,
        config=config,
    )

    nll = 0.0
    for sector_id, rows in enumerate(dataset["sector_rows"]):
        baseline = transit_model[rows] + pars["mean_flux"][sector_id]
        resid = dataset["flux"][rows] - baseline
        nll += _sector_gp_marginal_nll(
            time=dataset["time"][rows],
            resid=resid,
            flux_err=dataset["flux_err"][rows],
            gp_sigma=pars["gp_sigma"][sector_id],
            gp_rho=pars["gp_rho"][sector_id],
            jitter=pars["jitter"][sector_id],
            runtime=runtime,
            config=config,
        )

    flux_err_ref = max(float(np.median(dataset["flux_err"])), 1.0e-6)
    prior = 0.0
    prior += 0.5 * ((pars["period"] - config.period_d) / 2.0e-4) ** 2
    prior += 0.5 * ((pars["t0"] - config.t0_btjd) / 3.0e-3) ** 2
    prior += (
        0.5
        * ((pars["rho_star_solar"] - config.rho_star_solar_guess) / config.rho_star_solar_sigma)
        ** 2
    )
    prior += 0.5 * ((pars["r"] - config.radius_ratio_guess) / 0.03) ** 2
    prior += 0.5 * ((pars["b"] - config.impact_guess) / 0.25) ** 2
    prior += 0.5 * np.sum((pars["mean_flux"] / 5.0e-4) ** 2)
    prior += 0.5 * np.sum(((pars["log_jitter"] - np.log(flux_err_ref)) / 2.0) ** 2)
    prior += 0.5 * np.sum(((pars["log_gp_sigma"] - np.log(flux_err_ref)) / 2.0) ** 2)
    prior += 0.5 * np.sum(
        ((pars["log_gp_rho"] - np.log(config.gp_rho_guess_d)) / 1.0) ** 2
    )
    return float(nll + prior)


def _summarize_interval(samples: Sequence[float]) -> dict[str, float]:
    sample_arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    q16, q50, q84 = np.percentile(sample_arr, [16.0, 50.0, 84.0])
    return {
        "lower": float(q16),
        "median": float(q50),
        "upper": float(q84),
        "plus": float(q84 - q50),
        "minus": float(q50 - q16),
    }


def summarize_posterior_samples(theta_samples: np.ndarray, n_sectors: int) -> dict[str, Any] | None:
    """Summarize the transit posterior into percentile intervals."""

    theta_arr = np.asarray(theta_samples, dtype=np.float64)
    if theta_arr.size == 0:
        return None

    keys = [
        "r",
        "b",
        "rho_star_solar",
        "a_over_rstar",
        "inclination_deg",
        "duration",
        "q1",
        "q2",
        "u1",
        "u2",
    ]
    derived: dict[str, list[float]] = {key: [] for key in keys}
    depth_percent: list[float] = []

    for theta in theta_arr:
        pars = _unpack_params(theta, n_sectors, runtime=None)
        derived["r"].append(float(pars["r"]))
        derived["b"].append(float(pars["b"]))
        derived["rho_star_solar"].append(float(pars["rho_star_solar"]))
        derived["a_over_rstar"].append(float(pars["a_over_rstar"]))
        derived["inclination_deg"].append(float(pars["inclination_deg"]))
        derived["duration"].append(float(pars["duration"]))
        derived["q1"].append(float(pars["q1"]))
        derived["q2"].append(float(pars["q2"]))
        derived["u1"].append(float(pars["u"][0]))
        derived["u2"].append(float(pars["u"][1]))
        depth_percent.append(100.0 * float(pars["r"]) ** 2)

    summary = {key: _summarize_interval(values) for key, values in derived.items()}
    summary["transit_depth_percent"] = _summarize_interval(depth_percent)
    return summary


def summarize_posterior_samples_with_runtime(
    theta_samples: np.ndarray,
    n_sectors: int,
    runtime: _MlexoRuntime,
) -> dict[str, Any] | None:
    _ = runtime
    return summarize_posterior_samples(theta_samples, n_sectors)


def _build_bounds(config: TessTransitFitConfig, n_sectors: int) -> np.ndarray:
    bounds: list[tuple[float, float]] = [
        (np.log(config.period_d - 0.01), np.log(config.period_d + 0.01)),
        (config.t0_btjd - 0.05, config.t0_btjd + 0.05),
        (np.log(0.05), np.log(3.0)),
        (np.log(0.01), np.log(0.25)),
        (0.0, 1.0),
        (1.0e-6, 1.0),
        (1.0e-6, 1.0),
    ]
    bounds.extend([(-5.0e-3, 5.0e-3)] * n_sectors)
    bounds.extend([(np.log(1.0e-7), np.log(5.0e-3))] * n_sectors)
    bounds.extend([(np.log(1.0e-7), np.log(5.0e-3))] * n_sectors)
    bounds.extend(
        [(np.log(config.resolved_gp_rho_min_d), np.log(config.gp_rho_max_d))] * n_sectors
    )
    return np.asarray(bounds, dtype=np.float64)


def _in_bounds(theta: np.ndarray, bounds: np.ndarray) -> bool:
    theta_arr = np.asarray(theta, dtype=np.float64)
    return bool(np.all(theta_arr >= bounds[:, 0]) and np.all(theta_arr <= bounds[:, 1]))


def _log_probability(
    theta: np.ndarray,
    dataset: Mapping[str, Any],
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
    bounds: np.ndarray,
) -> float:
    theta_arr = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(theta_arr)) or not _in_bounds(theta_arr, bounds):
        return float("-inf")

    try:
        nlp = _negative_log_posterior(theta_arr, dataset, runtime, config)
    except Exception:
        return float("-inf")
    if not np.isfinite(nlp):
        return float("-inf")
    return -float(nlp)


def _initialize_walkers(
    theta_map: np.ndarray,
    bounds: np.ndarray,
    n_walkers: int,
    *,
    seed: int = 123,
    init_scale: float = 1.0e-3,
) -> np.ndarray:
    theta_map_arr = np.asarray(theta_map, dtype=np.float64)
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    widths = bounds_arr[:, 1] - bounds_arr[:, 0]
    eps = np.maximum(1.0e-8, 1.0e-6 * widths)
    lower = bounds_arr[:, 0] + eps
    upper = bounds_arr[:, 1] - eps

    rng = np.random.default_rng(seed)
    walkers = theta_map_arr + init_scale * widths * rng.standard_normal(
        (n_walkers, theta_map_arr.size)
    )
    walkers = np.clip(walkers, lower, upper)

    for idx in range(n_walkers):
        if np.allclose(walkers[idx], theta_map_arr):
            walkers[idx] = np.clip(
                theta_map_arr + eps * rng.standard_normal(theta_map_arr.size),
                lower,
                upper,
            )
    return walkers


def _run_emcee_sampler(
    initial_walkers: np.ndarray,
    dataset: Mapping[str, Any],
    runtime: _MlexoRuntime,
    config: TessTransitFitConfig,
    bounds: np.ndarray,
) -> tuple[Any, bool, int]:
    emcee = _load_module_or_raise(
        "emcee",
        install_hint="Install emcee before running the transit sampler.",
    )

    ndim = initial_walkers.shape[1]
    n_walkers = initial_walkers.shape[0]
    worker_count = min(int(config.emcee_n_processes), int(n_walkers))

    if config.emcee_use_pool and worker_count > 1:
        try:
            ctx = get_context(config.emcee_start_method)
            with ctx.Pool(processes=worker_count) as pool:
                sampler = emcee.EnsembleSampler(
                    n_walkers,
                    ndim,
                    _log_probability,
                    args=(dataset, runtime, config, bounds),
                    pool=pool,
                )
                state = sampler.run_mcmc(
                    initial_walkers,
                    config.emcee_burnin_steps,
                    progress=True,
                )
                sampler.reset()
                sampler.run_mcmc(
                    state,
                    config.emcee_production_steps,
                    progress=True,
                )
                return sampler, True, worker_count
        except Exception:
            pass

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        _log_probability,
        args=(dataset, runtime, config, bounds),
    )
    state = sampler.run_mcmc(initial_walkers, config.emcee_burnin_steps, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, config.emcee_production_steps, progress=True)
    return sampler, False, 1


def fit_tess_transit_with_mlexo(
    config: TessTransitFitConfig,
    *,
    lc_collection: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the raw-TESS to mlexo transit fit and return dataset plus best-fit state."""

    runtime = _load_mlexo_runtime(config)
    scipy_optimize = _load_module_or_raise(
        "scipy.optimize",
        install_hint="Install scipy before running the transit optimizer.",
    )

    collection = lc_collection if lc_collection is not None else download_tess_lightcurves(config)
    dataset = build_joint_tess_dataset(
        collection,
        period_d=config.period_d,
        t0_btjd=config.t0_btjd,
        transit_duration_d=config.transit_duration_d,
        model_window_d=config.model_window_d,
        flatten_window_length=config.flatten_window_length,
        outlier_sigma=config.outlier_sigma,
    )

    gp_sigma_guess = max(float(np.median(dataset["flux_err"])), 1.0e-6)
    bounds = _build_bounds(config, int(dataset["n_sectors"]))

    theta0 = np.concatenate(
        [
            np.array(
                [
                    np.log(config.period_d),
                    config.t0_btjd,
                    np.log(config.rho_star_solar_guess),
                    np.log(config.radius_ratio_guess),
                    config.impact_guess / (1.0 + config.radius_ratio_guess),
                    0.25,
                    0.25,
                ],
                dtype=np.float64,
            ),
            np.zeros(int(dataset["n_sectors"]), dtype=np.float64),
            np.full(int(dataset["n_sectors"]), np.log(gp_sigma_guess), dtype=np.float64),
            np.full(int(dataset["n_sectors"]), np.log(gp_sigma_guess), dtype=np.float64),
            np.full(int(dataset["n_sectors"]), np.log(config.gp_rho_guess_d), dtype=np.float64),
        ]
    )

    result = scipy_optimize.minimize(
        _negative_log_posterior,
        theta0,
        args=(dataset, runtime, config),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "maxfun": 20000},
    )

    ndim = result.x.size
    n_walkers = max(config.emcee_nwalkers_min, 2 * ndim)
    initial_walkers = _initialize_walkers(
        result.x,
        bounds,
        n_walkers,
        init_scale=config.emcee_init_scale,
    )

    sampler, used_parallel_emcee, emcee_worker_count = _run_emcee_sampler(
        initial_walkers,
        dataset,
        runtime,
        config,
        bounds,
    )
    posterior_theta = sampler.get_chain(flat=True, thin=config.emcee_thin)
    posterior_log_prob = sampler.get_log_prob(flat=True, thin=config.emcee_thin)

    finite_mask = np.isfinite(posterior_log_prob)
    posterior_theta = posterior_theta[finite_mask]
    posterior_log_prob = posterior_log_prob[finite_mask]
    if posterior_theta.size == 0:
        raise RuntimeError("emcee produced no finite posterior samples.")

    best_idx = int(np.argmax(posterior_log_prob))
    theta_best = posterior_theta[best_idx]

    best_fit = _unpack_params(theta_best, int(dataset["n_sectors"]), runtime)
    best_fit["transit_model"] = _mlexo_transit_flux(
        dataset["time"],
        period=float(best_fit["period"]),
        t0=float(best_fit["t0"]),
        a_over_rstar=float(best_fit["a_over_rstar"]),
        impact_param=float(best_fit["b"]),
        radius_ratio=float(best_fit["r"]),
        limb_darkening_u=np.asarray(best_fit["u"], dtype=np.float64),
        runtime=runtime,
        config=config,
    )
    best_fit["gp_trend"] = _build_gp_trend(
        dataset,
        best_fit["transit_model"],
        best_fit,
        runtime,
        config,
    )
    best_fit["mean_model"] = (
        best_fit["transit_model"]
        + best_fit["mean_flux"][dataset["sector_idx"]]
        + best_fit["gp_trend"]
    )
    best_fit["detrended_flux"] = (
        dataset["flux"] - best_fit["mean_flux"][dataset["sector_idx"]] - best_fit["gp_trend"]
    )
    best_fit["residual_flux"] = dataset["flux"] - best_fit["mean_model"]
    best_fit["phase_days"] = (
        (dataset["time"] - best_fit["t0"] + 0.5 * best_fit["period"]) % best_fit["period"]
    ) - 0.5 * best_fit["period"]
    best_fit["phase_grid"] = np.linspace(-config.plot_window_d, config.plot_window_d, 1500)
    best_fit["phase_model_grid"] = _mlexo_transit_flux(
        best_fit["t0"] + best_fit["phase_grid"],
        period=float(best_fit["period"]),
        t0=float(best_fit["t0"]),
        a_over_rstar=float(best_fit["a_over_rstar"]),
        impact_param=float(best_fit["b"]),
        radius_ratio=float(best_fit["r"]),
        limb_darkening_u=np.asarray(best_fit["u"], dtype=np.float64),
        runtime=runtime,
        config=config,
    )
    best_fit["posterior_theta"] = posterior_theta
    best_fit["posterior_log_prob"] = posterior_log_prob
    best_fit["summary_stats"] = summarize_posterior_samples_with_runtime(
        posterior_theta,
        int(dataset["n_sectors"]),
        runtime,
    )
    best_fit["map_result"] = result
    best_fit["emcee_acceptance_fraction"] = float(np.mean(sampler.acceptance_fraction))
    best_fit["emcee_parallel"] = bool(used_parallel_emcee)
    best_fit["emcee_worker_count"] = int(emcee_worker_count)

    return dataset, best_fit


def _value_sigma_from_summary(
    summary: Mapping[str, float],
    *,
    sigma_mode: str = "max",
) -> tuple[float, float]:
    value = float(summary["median"])
    plus = float(summary["plus"])
    minus = float(summary["minus"])
    if sigma_mode == "mean":
        sigma = 0.5 * (plus + minus)
    else:
        sigma = max(plus, minus)
    return value, float(sigma)


def make_tess_bandpass_constraint_from_mlexo(
    *,
    best_fit: Mapping[str, Any] | None = None,
    summary_stats: Mapping[str, Any] | None = None,
    observable: str = "transit_depth",
    name: str = "tess_transit",
    sigma_mode: str = "max",
    photon_weighted: bool | None = None,
) -> dict[str, Any]:
    """Convert a mlexo transit fit summary into a retrieval-ready constraint dict."""

    summary = summary_stats or (
        best_fit.get("summary_stats") if best_fit is not None else None
    )
    if summary is None:
        raise ValueError("summary_stats or best_fit['summary_stats'] is required.")

    if observable == "radius_ratio":
        value, sigma = _value_sigma_from_summary(summary["r"], sigma_mode=sigma_mode)
    elif observable == "transit_depth":
        value_pct, sigma_pct = _value_sigma_from_summary(
            summary["transit_depth_percent"],
            sigma_mode=sigma_mode,
        )
        value = value_pct / 100.0
        sigma = sigma_pct / 100.0
    else:
        raise ValueError(
            f"Unsupported observable {observable!r}. "
            "Use 'radius_ratio' or 'transit_depth'."
        )

    constraint: dict[str, Any] = {
        "name": name,
        "mode": "transmission",
        "observable": observable,
        "value": float(value),
        "sigma": float(sigma),
    }
    if photon_weighted is not None:
        constraint["photon_weighted"] = bool(photon_weighted)
    return constraint


def write_tess_bandpass_tbl(
    path: str | Path,
    *,
    constraint: Mapping[str, Any],
    planet_name: str | None = None,
    reference: str | None = None,
    note: str | None = None,
    central_wavelength_micron: float = TESS_CENTRAL_WAVELENGTH_MICRON,
    bandwidth_micron: float = TESS_BANDWIDTH_MICRON,
) -> Path:
    """Write a NASA-style one-row TESS .tbl that the retrieval CLI can ingest."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    observable = str(constraint["observable"])
    mode = str(constraint["mode"]).lower().strip()
    value = float(constraint["value"])
    sigma = float(constraint["sigma"])

    if mode != "transmission":
        raise ValueError(
            "write_tess_bandpass_tbl currently supports transmission constraints only."
        )
    if observable not in {"radius_ratio", "transit_depth"}:
        raise ValueError(
            "Transmission TESS .tbl export expects 'radius_ratio' or 'transit_depth'."
        )

    transit_depth_fraction = value if observable == "transit_depth" else value**2
    if observable == "transit_depth":
        transit_depth_sigma_fraction = sigma
    else:
        transit_depth_sigma_fraction = 2.0 * max(value, 1.0e-12) * sigma

    transit_depth_percent = transit_depth_fraction * 100.0
    transit_depth_sigma_percent = transit_depth_sigma_fraction * 100.0

    lines = [
        f"\\PL_NAME='{planet_name or 'Unknown planet'}'",
        "\\SPEC_TYPE='Transit'",
        f"\\INSTRUMENT='{TESS_INSTRUMENT}'",
        f"\\FACILITY='{TESS_FACILITY}'",
        f"\\NOTE='{note or 'Generated from mlexo TESS transit fit'}'",
        f"\\REFERENCE='{reference or 'Generated locally'}'",
        "|CENTRALWAVELNG|BANDWIDTH|SPECTRANSDEP|SPECTRANSDEPERR1|SPECTRANSDEPERR2|",
        "|        double|   double|      double|           double|           double|",
        "|       microns|  microns|           %|                 |                 |",
        "|          null|     null|        null|             null|             null|",
        (
            f"{central_wavelength_micron:>15.6f}"
            f"{bandwidth_micron:>10.6f}"
            f"{transit_depth_percent:>13.8f}"
            f"{transit_depth_sigma_percent:>17.8f}"
            f"{-transit_depth_sigma_percent:>17.8f}"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines))
    return output_path


def fit_tess_transit_to_bandpass_constraint(
    config: TessTransitFitConfig,
    *,
    lc_collection: Any | None = None,
    observable: str = "transit_depth",
    constraint_name: str = "tess_transit",
    photon_weighted: bool | None = None,
    tbl_path: str | Path | None = None,
) -> TessTransitFitResult:
    """Run the full raw-TESS-to-retrieval pipeline."""

    dataset, best_fit = fit_tess_transit_with_mlexo(config, lc_collection=lc_collection)
    constraint = make_tess_bandpass_constraint_from_mlexo(
        best_fit=best_fit,
        observable=observable,
        name=constraint_name,
        photon_weighted=photon_weighted,
    )

    if tbl_path is not None:
        write_tess_bandpass_tbl(
            tbl_path,
            constraint=constraint,
            planet_name=config.planet_name or config.target,
            reference=config.reference,
            note=config.note,
        )

    return TessTransitFitResult(
        config=config,
        dataset=dataset,
        best_fit=best_fit,
        bandpass_constraint=constraint,
    )


def serialize_tess_fit_summary(result: TessTransitFitResult) -> dict[str, Any]:
    """Return a JSON-friendly summary of the fit and exported constraint."""

    best_fit = result.best_fit
    summary = {
        "target": result.config.target,
        "n_sectors": int(result.dataset["n_sectors"]),
        "sector_labels": [str(label) for label in result.dataset["sector_labels"]],
        "sector_counts": [int(count) for count in result.dataset["sector_counts"]],
        "constraint": dict(result.bandpass_constraint),
        "fit": {
            "period": float(best_fit["period"]),
            "t0": float(best_fit["t0"]),
            "rho_star_solar": float(best_fit["rho_star_solar"]),
            "a_over_rstar": float(best_fit["a_over_rstar"]),
            "duration": float(best_fit["duration"]),
            "r": float(best_fit["r"]),
            "b": float(best_fit["b"]),
            "emcee_acceptance_fraction": float(best_fit["emcee_acceptance_fraction"]),
            "emcee_parallel": bool(best_fit["emcee_parallel"]),
            "emcee_worker_count": int(best_fit["emcee_worker_count"]),
            "summary_stats": best_fit.get("summary_stats"),
        },
    }
    return json.loads(json.dumps(summary))
