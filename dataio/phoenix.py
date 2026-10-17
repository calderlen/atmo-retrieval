"""PHOENIX stellar surface spectra, on-grid interpolation, and persistent caching."""

from __future__ import annotations

import hashlib
import numpy as np
import config
from functools import lru_cache
from pathlib import Path
from typing import Any
from astropy import constants as const, units as u


def _normalize_phoenix_cache_dir(path: str | Path | None) -> Path:
    candidate = config.PHOENIX_CACHE_DIR if path is None else Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


@lru_cache(maxsize=None)
def _read_phoenix_surface_flux_cache(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path_str) as raw:
        return (
            np.asarray(raw["wavelength_angstrom"], dtype=float),
            np.asarray(raw["stellar_surface_flux"], dtype=float),
        )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _derive_stellar_logg_cgs(Mstar_msun: float, Rstar_rsun: float) -> float:
    g_cgs = (const.G * (Mstar_msun * const.M_sun) / (Rstar_rsun * const.R_sun) ** 2).to_value(u.cm / u.s**2)
    return float(np.log10(g_cgs))


def _resolve_chromatic_phoenix_parameters(
    *,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> tuple[float, float, float]:
    temperature = _coerce_optional_float(Tstar)

    resolved_logg = _coerce_optional_float(logg_star)
    if resolved_logg is None:
        Mstar_val = _coerce_optional_float(Mstar)
        Rstar_val = _coerce_optional_float(Rstar)
        resolved_logg = _derive_stellar_logg_cgs(Mstar_val, Rstar_val)

    resolved_metallicity = _coerce_optional_float(metallicity)
    if resolved_metallicity is None:
        resolved_metallicity = 0.0

    return temperature, resolved_logg, resolved_metallicity


def _format_phoenix_cache_float(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _build_phoenix_surface_flux_cache_path(
    *,
    cache_dir: Path,
    temperature: float,
    logg: float,
    metallicity: float,
    target_wavelength_angstrom: np.ndarray,
) -> Path:
    cache_key = hashlib.sha1()
    cache_key.update(b"phoenix-surface-flux-v1\0")
    cache_key.update(
        np.asarray(
            [temperature, logg, metallicity],
            dtype=np.float64,
        ).tobytes()
    )
    cache_key.update(np.asarray(target_wavelength_angstrom, dtype=np.float64).tobytes())
    input_hash = cache_key.hexdigest()[:16]
    filename = (
        "phoenix_"
        f"T{_format_phoenix_cache_float(temperature)}_"
        f"logg{_format_phoenix_cache_float(logg)}_"
        f"feh{_format_phoenix_cache_float(metallicity)}_"
        f"{input_hash}.npz"
    )
    return cache_dir / filename


def _convert_chromatic_surface_flux_to_exojax_units(
    wavelength: u.Quantity,
    surface_flux: u.Quantity,
) -> np.ndarray:
    wavelength_cm = u.Quantity(wavelength).to(u.cm)
    energy_per_photon = (const.h * const.c / wavelength_cm) / u.photon
    surface_flux_lambda = (u.Quantity(surface_flux) * energy_per_photon).to(u.erg / (u.s * u.cm**2 * u.cm))
    surface_flux_wavenumber = surface_flux_lambda * wavelength_cm**2
    return np.asarray(
        surface_flux_wavenumber.to_value(u.erg / (u.s * u.cm)),
        dtype=float,
    )


def _load_chromatic_phoenix_surface_flux_on_grid(
    *,
    nu_grid: np.ndarray,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    phoenix_cache_dir: str | Path | None,
) -> np.ndarray:
    target_wavelength_angstrom = 1.0e8 / np.asarray(nu_grid, dtype=float)
    temperature, resolved_logg, resolved_metallicity = _resolve_chromatic_phoenix_parameters(
        component_name=component_name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    cache_dir = _normalize_phoenix_cache_dir(phoenix_cache_dir)
    cache_path = _build_phoenix_surface_flux_cache_path(
        cache_dir=cache_dir,
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        target_wavelength_angstrom=target_wavelength_angstrom,
    )

    if cache_path.exists():
        _, cached_flux = _read_phoenix_surface_flux_cache(str(cache_path))
        return cached_flux

    from chromatic import get_phoenix_photons

    sort_idx = np.argsort(target_wavelength_angstrom)
    query_wavelength_um = (target_wavelength_angstrom[sort_idx] / 1.0e4) * u.micron
    wavelength_um, surface_flux = get_phoenix_photons(
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        wavelength=query_wavelength_um,
    )
    flux_sorted = _convert_chromatic_surface_flux_to_exojax_units(
        wavelength=wavelength_um,
        surface_flux=surface_flux,
    )
    stellar_surface_flux = np.empty_like(flux_sorted)
    stellar_surface_flux[sort_idx] = flux_sorted

    if np.any(~np.isfinite(stellar_surface_flux)):
        raise ValueError(
            f"chromatic returned non-finite PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )
    if np.any(stellar_surface_flux <= 0):
        raise ValueError(
            f"chromatic returned non-positive PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )

    np.savez_compressed(
        cache_path,
        wavelength_angstrom=target_wavelength_angstrom,
        stellar_surface_flux=stellar_surface_flux,
        temperature=np.asarray(temperature, dtype=float),
        logg=np.asarray(resolved_logg, dtype=float),
        metallicity=np.asarray(resolved_metallicity, dtype=float),
        cache_schema_version=np.asarray(1, dtype=int),
    )
    return stellar_surface_flux


def _load_phoenix_surface_flux_on_grid(
    *,
    phoenix_cache_dir: str | Path | None,
    nu_grid: np.ndarray,
    mode: str,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> np.ndarray | None:
    if str(mode).lower().strip() != "emission":
        return None
    return _load_chromatic_phoenix_surface_flux_on_grid(
        nu_grid=nu_grid,
        component_name=component_name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
        phoenix_cache_dir=phoenix_cache_dir,
    )
