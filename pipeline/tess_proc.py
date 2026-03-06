"""Standalone TESS eclipse inference using jaxoplanet + NumPyro.

This module fits eclipse depth with a physical model that combines thermal
emission and reflected light in the TESS bandpass:

    delta_sec = (Rp/Rs)^2 * [B_band(T_day) / B_band(T_star)] + Ag * (Rp/a)^2

where B_band(T) is the Planck function integrated over the instrument
bandpass response.

It also defines a jaxoplanet TransitOrbit and computes a model mid-transit
depth from the sampled radius ratio.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit
from numpyro.infer import MCMC, NUTS

import config
from dataio.load import load_nasa_archive_spectrum


def _to_float(value: Any, default: float | None = None) -> float | None:
    """Convert scalar-like values (including uncertainties objects) to float."""
    if value is None:
        return default

    if hasattr(value, "nominal_value"):
        value = value.nominal_value

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(out):
        return default
    return out


def _sanitize_planet_name(planet: str) -> str:
    return planet.lower().replace("-", "")


def _trapz_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)


def _download_tess_bandpass(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = urlopen(config.TESS_BANDPASS_URL).read().decode("utf-8")
    path.write_text(text)


def load_tess_bandpass(
    path: str | Path | None = None,
    *,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load TESS response function CSV.

    Returns wavelength in meters and dimensionless response.
    """
    bandpass_path = Path(path) if path is not None else config.TESS_BANDPASS_PATH

    if not bandpass_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Bandpass file not found: {bandpass_path}. "
                "Provide --bandpass-path or allow download."
            )
        _download_tess_bandpass(bandpass_path)

    wavelength_nm: list[float] = []
    response: list[float] = []

    for raw_line in bandpass_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "," not in line:
            continue
        left, right = line.split(",", 1)

        try:
            lam_nm = float(left.strip())
            rsp = float(right.strip())
        except ValueError:
            continue

        wavelength_nm.append(lam_nm)
        response.append(rsp)

    wavelength_nm_arr = np.asarray(wavelength_nm, dtype=float)
    response_arr = np.asarray(response, dtype=float)

    finite_mask = np.isfinite(wavelength_nm_arr) & np.isfinite(response_arr)
    positive_mask = response_arr > 0.0
    mask = finite_mask & positive_mask

    wavelength_m = wavelength_nm_arr[mask] * 1.0e-9
    response_clean = response_arr[mask]

    if wavelength_m.size < 2:
        raise ValueError(
            f"Bandpass file {bandpass_path} does not contain enough valid points."
        )

    order = np.argsort(wavelength_m)
    wavelength_m = wavelength_m[order]
    response_clean = response_clean[order]

    return wavelength_m, response_clean, bandpass_path


def planck_lambda(wavelength_m: jnp.ndarray, temperature_k: jnp.ndarray) -> jnp.ndarray:
    """Spectral radiance B_lambda(T) in SI units (up to sr factor cancellation)."""
    wl = jnp.asarray(wavelength_m)
    temp = jnp.asarray(temperature_k)

    x = (config.H_PLANCK * config.C_LIGHT) / jnp.clip(
        wl * config.K_BOLTZMANN * temp,
        config.NUMERIC_EPS,
        None,
    )
    x = jnp.clip(x, 1.0e-12, 700.0)

    numerator = 2.0 * config.H_PLANCK * config.C_LIGHT**2
    denominator = jnp.clip(wl**5 * jnp.expm1(x), config.NUMERIC_EPS, None)
    return numerator / denominator


def bandpass_integrated_planck(
    temperature_k: jnp.ndarray,
    wavelength_m: jnp.ndarray,
    response: jnp.ndarray,
    *,
    photon_weighted: bool,
) -> jnp.ndarray:
    """Integrate the Planck function over bandpass response."""
    weights = response * wavelength_m if photon_weighted else response
    spectral = planck_lambda(wavelength_m, temperature_k)
    return _trapz_jax(wavelength_m, spectral * weights)


def _eclipse_components(
    *,
    radius_ratio: jnp.ndarray,
    geometric_albedo: jnp.ndarray,
    dayside_temperature: jnp.ndarray,
    semi_major_axis_au: float,
    stellar_radius_rsun: float,
    wavelength_m: jnp.ndarray,
    response: jnp.ndarray,
    star_band_integral: jnp.ndarray,
    photon_weighted: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    planet_band = bandpass_integrated_planck(
        dayside_temperature,
        wavelength_m,
        response,
        photon_weighted=photon_weighted,
    )

    thermal = (radius_ratio**2) * planet_band / jnp.clip(
        star_band_integral,
        config.NUMERIC_EPS,
        None,
    )

    a_over_rstar = (semi_major_axis_au * config.AU_M) / (
        stellar_radius_rsun * config.R_SUN_M
    )
    rp_over_a = radius_ratio / jnp.clip(a_over_rstar, config.NUMERIC_EPS, None)
    reflected = geometric_albedo * rp_over_a**2

    eclipse = thermal + reflected
    return eclipse, thermal, reflected


def _compute_mid_transit_depth(
    *,
    period_day: float,
    duration_day: float,
    impact_param: float,
    radius_ratio: jnp.ndarray,
    u1: float,
    u2: float,
) -> jnp.ndarray:
    """Compute model mid-transit depth using jaxoplanet TransitOrbit."""

    orbit = TransitOrbit(
        period=period_day,
        duration=duration_day,
        time_transit=0.0,
        impact_param=impact_param,
        radius_ratio=radius_ratio,
    )
    light_curve_fn = limb_dark_light_curve(orbit, u1, u2)
    flux_mid = light_curve_fn(jnp.array([0.0]))[0]
    return -flux_mid


def _estimate_radius_ratio_prior_sigma(params: dict[str, Any], rp_rs: float) -> float:
    rp_rs_err = _to_float(params.get("rp_rs_err"))
    if rp_rs_err is not None and rp_rs_err > 0.0:
        return rp_rs_err

    rp = _to_float(params.get("R_p"))
    rp_err = _to_float(params.get("R_p_err"))
    rstar = _to_float(params.get("R_star"))
    rstar_err = _to_float(params.get("R_star_err"))

    if (
        rp is not None
        and rp_err is not None
        and rstar is not None
        and rstar_err is not None
        and rp > 0.0
        and rstar > 0.0
    ):
        frac = math.sqrt((rp_err / rp) ** 2 + (rstar_err / rstar) ** 2)
        return max(frac * rp_rs, config.TESS_RADIUS_RATIO_SIGMA_MIN)

    return max(
        config.TESS_RADIUS_RATIO_FALLBACK_FRAC * rp_rs,
        config.TESS_RADIUS_RATIO_SIGMA_MIN,
    )


def _load_default_eclipse_measurement(tbl_path: str | Path) -> tuple[float, float]:
    """Load eclipse depth from a NASA archive .tbl file.

    Returns depth and uncertainty in fractional units (not ppm).
    """
    wav_angstrom, spectrum, sigma, _meta = load_nasa_archive_spectrum(
        tbl_path,
        mode="emission",
    )

    if spectrum.size == 0:
        raise ValueError(f"No eclipse points found in {tbl_path}")

    if spectrum.size == 1:
        idx = 0
    else:
        target_tess_angstrom = config.TESS_TARGET_WAVELENGTH_ANGSTROM
        idx = int(np.argmin(np.abs(wav_angstrom - target_tess_angstrom)))

    return float(spectrum[idx]), float(sigma[idx])


def make_tess_proc_model(
    *,
    period_day: float,
    duration_day: float,
    impact_param: float,
    stellar_temperature: float,
    semi_major_axis_au: float,
    stellar_radius_rsun: float,
    radius_ratio_prior: float,
    radius_ratio_prior_sigma: float,
    u1: float,
    u2: float,
    eclipse_depth_obs: float,
    eclipse_depth_err: float,
    eclipse_model_sigma: float,
    wavelength_m: np.ndarray,
    response: np.ndarray,
    photon_weighted: bool,
    dayside_temp_bounds: tuple[float, float],
    albedo_bounds: tuple[float, float],
    transit_depth_obs: float | None,
    transit_depth_err: float | None,
):
    wl = jnp.asarray(wavelength_m)
    rsp = jnp.asarray(response)

    star_band_integral = bandpass_integrated_planck(
        jnp.asarray(stellar_temperature),
        wl,
        rsp,
        photon_weighted=photon_weighted,
    )

    temp_min, temp_max = dayside_temp_bounds
    albedo_min, albedo_max = albedo_bounds

    def model() -> None:
        radius_ratio = numpyro.sample(
            "radius_ratio",
            dist.TruncatedNormal(radius_ratio_prior, radius_ratio_prior_sigma, low=0.0),
        )

        transit_depth_geom = numpyro.deterministic("transit_depth_geom", radius_ratio**2)
        transit_depth = numpyro.deterministic(
            "transit_depth",
            _compute_mid_transit_depth(
                period_day=period_day,
                duration_day=duration_day,
                impact_param=impact_param,
                radius_ratio=radius_ratio,
                u1=u1,
                u2=u2,
            ),
        )
        numpyro.deterministic("transit_depth_difference", transit_depth - transit_depth_geom)

        if transit_depth_obs is not None and transit_depth_err is not None:
            numpyro.sample(
                "obs_transit_depth",
                dist.Normal(transit_depth, transit_depth_err),
                obs=transit_depth_obs,
            )

        geometric_albedo = numpyro.sample(
            "geometric_albedo",
            dist.Uniform(albedo_min, albedo_max),
        )
        dayside_temperature = numpyro.sample(
            "dayside_temperature",
            dist.Uniform(temp_min, temp_max),
        )

        eclipse_model, thermal_component, reflected_component = _eclipse_components(
            radius_ratio=radius_ratio,
            geometric_albedo=geometric_albedo,
            dayside_temperature=dayside_temperature,
            semi_major_axis_au=semi_major_axis_au,
            stellar_radius_rsun=stellar_radius_rsun,
            wavelength_m=wl,
            response=rsp,
            star_band_integral=star_band_integral,
            photon_weighted=photon_weighted,
        )
        numpyro.deterministic("eclipse_depth_model", eclipse_model)
        numpyro.deterministic("eclipse_thermal_component", thermal_component)
        numpyro.deterministic("eclipse_reflected_component", reflected_component)

        eclipse_depth = numpyro.sample(
            "eclipse_depth",
            dist.TruncatedNormal(eclipse_model, eclipse_model_sigma, low=0.0),
        )
        numpyro.sample(
            "obs_eclipse_depth",
            dist.Normal(eclipse_depth, eclipse_depth_err),
            obs=eclipse_depth_obs,
        )

    return model


def _create_output_dir(
    *,
    planet: str,
    ephemeris: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    base = (
        config.PROJECT_ROOT
        / "output"
        / _sanitize_planet_name(planet)
        / ephemeris
        / config.TESS_OUTPUT_SUBDIR
    )
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _quantiles(x: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])
    return float(q16), float(q50), float(q84)


def run_tess_proc(
    *,
    planet: str,
    ephemeris: str,
    eclipse_depth_obs: float,
    eclipse_depth_err: float,
    transit_depth_obs: float | None,
    transit_depth_err: float | None,
    bandpass_path: str | Path | None,
    download_bandpass: bool,
    photon_weighted: bool,
    dayside_temp_bounds: tuple[float, float],
    albedo_bounds: tuple[float, float],
    radius_ratio_prior: float | None,
    radius_ratio_prior_sigma: float | None,
    eclipse_model_sigma: float,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    max_tree_depth: int,
    seed: int,
    output_dir: str | Path | None,
) -> tuple[MCMC, dict[str, np.ndarray], Path]:
    if eclipse_depth_obs <= 0.0:
        raise ValueError("eclipse_depth_obs must be > 0")
    if eclipse_depth_err <= 0.0:
        raise ValueError("eclipse_depth_err must be > 0")
    if eclipse_model_sigma <= 0.0:
        raise ValueError("eclipse_model_sigma must be > 0")

    params = config.get_params(planet=planet, ephemeris=ephemeris)

    period_day = _to_float(params.get("period"))
    duration_day = _to_float(params.get("duration"))
    impact_param = _to_float(params.get("b"), default=config.TESS_DEFAULT_IMPACT_PARAM)
    stellar_temperature = _to_float(params.get("T_star"))
    semi_major_axis_au = _to_float(params.get("a"))
    stellar_radius_rsun = _to_float(params.get("R_star"))

    if period_day is None:
        raise ValueError("Planet parameter 'period' is required")
    if duration_day is None:
        raise ValueError("Planet parameter 'duration' is required")
    if stellar_temperature is None:
        raise ValueError("Planet parameter 'T_star' is required")
    if semi_major_axis_au is None:
        raise ValueError("Planet parameter 'a' is required")
    if stellar_radius_rsun is None:
        raise ValueError("Planet parameter 'R_star' is required")
    if impact_param is None:
        impact_param = config.TESS_DEFAULT_IMPACT_PARAM

    rp_rs_default = _to_float(params.get("rp_rs"))
    if rp_rs_default is None:
        rp_rj = _to_float(params.get("R_p"))
        if rp_rj is None:
            raise ValueError("Need either rp_rs or R_p in planet parameters")
        rp_rs_default = (rp_rj * config.RJUP_M) / (stellar_radius_rsun * config.R_SUN_M)

    rp_rs_prior = radius_ratio_prior if radius_ratio_prior is not None else rp_rs_default
    if rp_rs_prior <= 0.0:
        raise ValueError("radius ratio prior must be > 0")

    if radius_ratio_prior_sigma is None:
        rp_rs_prior_sigma = _estimate_radius_ratio_prior_sigma(params, rp_rs_prior)
    else:
        rp_rs_prior_sigma = radius_ratio_prior_sigma
    if rp_rs_prior_sigma <= 0.0:
        raise ValueError("radius ratio prior sigma must be > 0")

    gamma1 = _to_float(params.get("gamma1"), default=config.TESS_DEFAULT_GAMMA1)
    gamma2 = _to_float(params.get("gamma2"), default=config.TESS_DEFAULT_GAMMA2)
    if gamma1 is None:
        gamma1 = config.TESS_DEFAULT_GAMMA1
    if gamma2 is None:
        gamma2 = config.TESS_DEFAULT_GAMMA2

    wavelength_m, response, used_bandpass_path = load_tess_bandpass(
        bandpass_path,
        download_if_missing=download_bandpass,
    )

    model = make_tess_proc_model(
        period_day=period_day,
        duration_day=duration_day,
        impact_param=impact_param,
        stellar_temperature=stellar_temperature,
        semi_major_axis_au=semi_major_axis_au,
        stellar_radius_rsun=stellar_radius_rsun,
        radius_ratio_prior=rp_rs_prior,
        radius_ratio_prior_sigma=rp_rs_prior_sigma,
        u1=gamma1,
        u2=gamma2,
        eclipse_depth_obs=eclipse_depth_obs,
        eclipse_depth_err=eclipse_depth_err,
        eclipse_model_sigma=eclipse_model_sigma,
        wavelength_m=wavelength_m,
        response=response,
        photon_weighted=photon_weighted,
        dayside_temp_bounds=dayside_temp_bounds,
        albedo_bounds=albedo_bounds,
        transit_depth_obs=transit_depth_obs,
        transit_depth_err=transit_depth_err,
    )

    kernel = NUTS(model, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(random.PRNGKey(seed))

    output_path = _create_output_dir(
        planet=planet,
        ephemeris=ephemeris,
        output_dir=output_dir,
    )

    with open(output_path / "mcmc_summary.txt", "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()

    samples = {str(k): np.asarray(v) for k, v in mcmc.get_samples().items()}
    posterior_dir = output_path / "posterior_samples"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    for name, values in samples.items():
        np.save(str(posterior_dir / f"{name}.npy"), values)

    run_meta = {
        "planet": planet,
        "ephemeris": ephemeris,
        "eclipse_depth_obs": eclipse_depth_obs,
        "eclipse_depth_err": eclipse_depth_err,
        "transit_depth_obs": transit_depth_obs,
        "transit_depth_err": transit_depth_err,
        "period_day": period_day,
        "duration_day": duration_day,
        "impact_param": impact_param,
        "stellar_temperature": stellar_temperature,
        "semi_major_axis_au": semi_major_axis_au,
        "stellar_radius_rsun": stellar_radius_rsun,
        "radius_ratio_prior": rp_rs_prior,
        "radius_ratio_prior_sigma": rp_rs_prior_sigma,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "photon_weighted": photon_weighted,
        "dayside_temp_bounds": list(dayside_temp_bounds),
        "albedo_bounds": list(albedo_bounds),
        "eclipse_model_sigma": eclipse_model_sigma,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "max_tree_depth": max_tree_depth,
        "seed": seed,
        "bandpass_path": str(used_bandpass_path),
    }
    (output_path / "run_config.json").write_text(json.dumps(run_meta, indent=2))

    return mcmc, samples, output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone TESS eclipse + transit depth inference",
    )

    parser.add_argument("--planet", type=str, default=config.PLANET, help="Planet name")
    parser.add_argument("--ephemeris", type=str, default=config.EPHEMERIS, help="Ephemeris key")

    parser.add_argument(
        "--eclipse-depth-ppm",
        type=float,
        default=None,
        help="Observed eclipse depth in ppm. If omitted, loads from --eclipse-tbl.",
    )
    parser.add_argument(
        "--eclipse-err-ppm",
        type=float,
        default=None,
        help="Observed eclipse depth uncertainty in ppm.",
    )
    parser.add_argument(
        "--eclipse-tbl",
        type=str,
        default=str(config.TESS_DEFAULT_ECLIPSE_TBL),
        help="NASA archive .tbl file used when eclipse depth inputs are omitted.",
    )

    parser.add_argument(
        "--transit-depth-ppm",
        type=float,
        default=None,
        help="Optional observed transit depth in ppm.",
    )
    parser.add_argument(
        "--transit-err-ppm",
        type=float,
        default=None,
        help="Optional observed transit depth uncertainty in ppm.",
    )

    parser.add_argument(
        "--bandpass-path",
        type=str,
        default=None,
        help=(
            "Path to TESS response CSV. "
            f"Default: {config.TESS_BANDPASS_PATH}"
        ),
    )
    parser.add_argument(
        "--no-bandpass-download",
        action="store_true",
        help="Do not auto-download TESS bandpass if file is missing.",
    )
    parser.add_argument(
        "--photon-weighted",
        action="store_true",
        help="Use photon-weighted bandpass integration instead of energy-weighted.",
    )

    parser.add_argument(
        "--tday-min",
        type=float,
        default=config.TESS_DEFAULT_TDAY_MIN_K,
        help="Lower bound of dayside temperature prior [K].",
    )
    parser.add_argument(
        "--tday-max",
        type=float,
        default=config.TESS_DEFAULT_TDAY_MAX_K,
        help="Upper bound of dayside temperature prior [K].",
    )
    parser.add_argument(
        "--albedo-min",
        type=float,
        default=config.TESS_DEFAULT_ALBEDO_MIN,
        help="Lower bound of geometric albedo prior.",
    )
    parser.add_argument(
        "--albedo-max",
        type=float,
        default=config.TESS_DEFAULT_ALBEDO_MAX,
        help="Upper bound of geometric albedo prior.",
    )
    parser.add_argument(
        "--rp-rs-prior",
        type=float,
        default=None,
        help="Override radius ratio prior mean.",
    )
    parser.add_argument(
        "--rp-rs-prior-sigma",
        type=float,
        default=None,
        help="Override radius ratio prior sigma.",
    )
    parser.add_argument(
        "--eclipse-model-sigma-ppm",
        type=float,
        default=config.TESS_DEFAULT_ECLIPSE_MODEL_SIGMA_PPM,
        help="Width of latent eclipse-depth distribution around Eq(5) model [ppm].",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=config.TESS_DEFAULT_NUM_WARMUP,
        help="NUTS warmup steps",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=config.TESS_DEFAULT_NUM_SAMPLES,
        help="NUTS posterior samples",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=config.TESS_DEFAULT_NUM_CHAINS,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=config.TESS_DEFAULT_MAX_TREE_DEPTH,
        help="NUTS max tree depth",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.TESS_DEFAULT_SEED,
        help="Random seed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. "
            f"Default: output/<planet>/<ephemeris>/{config.TESS_OUTPUT_SUBDIR}/<timestamp>/"
        ),
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if (args.eclipse_depth_ppm is None) != (args.eclipse_err_ppm is None):
        raise ValueError("Provide both --eclipse-depth-ppm and --eclipse-err-ppm, or neither.")
    if (args.transit_depth_ppm is None) != (args.transit_err_ppm is None):
        raise ValueError("Provide both --transit-depth-ppm and --transit-err-ppm, or neither.")

    if args.eclipse_depth_ppm is None:
        eclipse_depth_obs, eclipse_depth_err = _load_default_eclipse_measurement(args.eclipse_tbl)
    else:
        eclipse_depth_obs = args.eclipse_depth_ppm * 1.0e-6
        eclipse_depth_err = args.eclipse_err_ppm * 1.0e-6

    transit_depth_obs = None
    transit_depth_err = None
    if args.transit_depth_ppm is not None:
        transit_depth_obs = args.transit_depth_ppm * 1.0e-6
        transit_depth_err = args.transit_err_ppm * 1.0e-6

    mcmc, samples, output_path = run_tess_proc(
        planet=args.planet,
        ephemeris=args.ephemeris,
        eclipse_depth_obs=eclipse_depth_obs,
        eclipse_depth_err=eclipse_depth_err,
        transit_depth_obs=transit_depth_obs,
        transit_depth_err=transit_depth_err,
        bandpass_path=args.bandpass_path,
        download_bandpass=not args.no_bandpass_download,
        photon_weighted=bool(args.photon_weighted),
        dayside_temp_bounds=(args.tday_min, args.tday_max),
        albedo_bounds=(args.albedo_min, args.albedo_max),
        radius_ratio_prior=args.rp_rs_prior,
        radius_ratio_prior_sigma=args.rp_rs_prior_sigma,
        eclipse_model_sigma=args.eclipse_model_sigma_ppm * 1.0e-6,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        max_tree_depth=args.max_tree_depth,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\nTESS eclipse inference complete")
    print(f"Output directory: {output_path}")
    print(f"Observed eclipse depth: {eclipse_depth_obs * 1.0e6:.2f} +/- {eclipse_depth_err * 1.0e6:.2f} ppm")

    q16, q50, q84 = _quantiles(samples["eclipse_depth"] * 1.0e6)
    print(f"eclipse_depth [ppm]: {q50:.2f} (+{q84 - q50:.2f}, -{q50 - q16:.2f})")

    q16, q50, q84 = _quantiles(samples["transit_depth"] * 1.0e6)
    print(f"transit_depth [ppm]: {q50:.2f} (+{q84 - q50:.2f}, -{q50 - q16:.2f})")

    q16, q50, q84 = _quantiles(samples["dayside_temperature"])
    print(f"dayside_temperature [K]: {q50:.1f} (+{q84 - q50:.1f}, -{q50 - q16:.1f})")

    q16, q50, q84 = _quantiles(samples["geometric_albedo"])
    print(f"geometric_albedo: {q50:.3f} (+{q84 - q50:.3f}, -{q50 - q16:.3f})")

    # Keep this call so console users get standard diagnostics immediately.
    mcmc.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
