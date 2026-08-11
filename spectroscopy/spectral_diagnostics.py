"""Numerical diagnostics for prepared high-resolution spectra.

All functions are read-only and operate on arrays or loaded bundle mappings.
Plotting and filesystem output live in their respective modules and CLIs.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

import config


C_KMS = 299792.458


def wave_1d(wavelength: np.ndarray) -> np.ndarray:
    """Return a representative one-dimensional wavelength grid."""

    wave = np.asarray(wavelength, dtype=float)
    if wave.ndim == 1:
        return wave
    if wave.ndim == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(wave, axis=0)
    raise ValueError(f"Expected a 1D or 2D wavelength grid, found shape {wave.shape}.")


def robust_limit(
    values: np.ndarray,
    *,
    percentile: float = 99.5,
    floor: float = 1.0e-8,
) -> float:
    """Return a finite symmetric display limit."""

    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    return max(float(np.percentile(finite, percentile)), floor) if finite.size else floor


def phase_values(bundle: Mapping[str, Any]) -> np.ndarray:
    """Return raw phase for transmission and modulo-one phase for emission."""

    phase = np.asarray(bundle["phase"], dtype=float)
    return np.mod(phase, 1.0) if bundle.get("mode") == "emission" else phase


def phase_order(bundle: Mapping[str, Any]) -> np.ndarray:
    """Return a stable finite-first phase ordering."""

    phase = phase_values(bundle)
    return np.argsort(np.where(np.isfinite(phase), phase, np.inf), kind="stable")


def finite_good(
    values: np.ndarray,
    sigma: np.ndarray | None = None,
    *,
    sigma_threshold: float = 0.5,
) -> np.ndarray:
    """Return the standard finite and positive-uncertainty diagnostic mask."""

    mask = np.isfinite(values)
    if sigma is not None:
        uncertainty = np.asarray(sigma, dtype=float)
        mask &= np.isfinite(uncertainty) & (uncertainty > 0)
        mask &= uncertainty < float(sigma_threshold)
    return mask


def typical_spacing(wavelength: np.ndarray) -> float:
    """Return the median positive adjacent wavelength spacing."""

    wave = wave_1d(wavelength)
    delta = np.diff(wave)
    positive = delta[np.isfinite(delta) & (delta > 0)]
    return float(np.median(positive)) if positive.size else float("nan")


def wavelength_gap_mask(
    wavelength: np.ndarray,
    *,
    gap_factor: float = 8.0,
) -> np.ndarray:
    """Flag boundaries whose spacing is much larger than the native spacing."""

    wave = wave_1d(wavelength)
    spacing = typical_spacing(wave)
    if not np.isfinite(spacing) or spacing <= 0:
        return np.zeros(max(0, wave.size - 1), dtype=bool)
    delta = np.diff(wave)
    return ~np.isfinite(delta) | (delta <= 0) | (delta > float(gap_factor) * spacing)


def contiguous_slices(
    wavelength: np.ndarray,
    *,
    gap_factor: float = 8.0,
) -> tuple[slice, ...]:
    """Split a wavelength grid at discontinuities."""

    wave = wave_1d(wavelength)
    boundaries = np.flatnonzero(wavelength_gap_mask(wave, gap_factor=gap_factor)) + 1
    edges = np.concatenate(([0], boundaries, [wave.size]))
    return tuple(slice(int(left), int(right)) for left, right in zip(edges[:-1], edges[1:]))


def gap_aware_bin_indices(
    wavelength: np.ndarray,
    *,
    max_bins: int = 900,
    gap_factor: float = 8.0,
) -> tuple[np.ndarray, ...]:
    """Return contiguous index groups without binning across wavelength gaps."""

    wave = wave_1d(wavelength)
    groups: list[np.ndarray] = []
    for segment in contiguous_slices(wave, gap_factor=gap_factor):
        indices = np.arange(segment.start, segment.stop)
        if not indices.size:
            continue
        n_groups = min(indices.size, max(1, int(np.ceil(max_bins * indices.size / wave.size))))
        groups.extend(group for group in np.array_split(indices, n_groups) if group.size)
    return tuple(groups)


def binned_series(
    wavelength: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None = None,
    *,
    max_bins: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Bin a one-dimensional series without bridging spectral gaps."""

    wave = wave_1d(wavelength)
    value_array = np.asarray(values, dtype=float)
    error_array = None if errors is None else np.asarray(errors, dtype=float)
    out_wave: list[float] = []
    out_value: list[float] = []
    out_error: list[float] = []
    for indices in gap_aware_bin_indices(wave, max_bins=max_bins):
        good = np.isfinite(value_array[indices])
        if error_array is not None:
            good &= np.isfinite(error_array[indices]) & (error_array[indices] > 0)
        if not np.any(good):
            continue
        selected = indices[good]
        out_wave.append(float(np.nanmean(wave[selected])))
        if error_array is None:
            out_value.append(float(np.nanmean(value_array[selected])))
        else:
            weight = 1.0 / np.square(error_array[selected])
            out_value.append(float(np.sum(weight * value_array[selected]) / np.sum(weight)))
            out_error.append(float(1.0 / np.sqrt(np.sum(weight))))
    return (
        np.asarray(out_wave),
        np.asarray(out_value),
        None if error_array is None else np.asarray(out_error),
    )


def observed_spectrum(
    data: np.ndarray,
    sigma: np.ndarray,
    *,
    sigma_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an inverse-variance weighted spectrum across exposures."""

    values = np.asarray(data, dtype=float)
    uncertainty = np.asarray(sigma, dtype=float)
    good = finite_good(values, uncertainty, sigma_threshold=sigma_threshold)
    weight = np.where(good, 1.0 / np.square(uncertainty), 0.0)
    denominator = np.sum(weight, axis=0)
    mean = np.divide(
        np.sum(np.where(good, values, 0.0) * weight, axis=0),
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )
    error = np.divide(
        1.0,
        np.sqrt(denominator),
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )
    return mean, error, np.sum(good, axis=0)


def column_metrics(bundle: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return per-column finite coverage and uncertainty statistics."""

    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    good = finite_good(data, sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "finite_fraction": np.mean(good, axis=0),
            "median_sigma": np.nanmedian(np.where(good, sigma, np.nan), axis=0),
            "median_abs_data": np.nanmedian(np.where(good, np.abs(data), np.nan), axis=0),
        }


def region_metrics(bundle: Mapping[str, Any], *, edge_width_A: float = 30.0) -> list[dict[str, Any]]:
    """Summarize retained edges, configured telluric regions, and clean interior."""

    wave = wave_1d(bundle["wavelength"])
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    good = finite_good(data, sigma)
    arm = str(bundle["arm"])
    regions: list[tuple[str, float, float]] = [
        ("left_edge", float(np.nanmin(wave)), float(np.nanmin(wave) + edge_width_A)),
        ("right_edge", float(np.nanmax(wave) - edge_width_A), float(np.nanmax(wave))),
    ]
    telluric_config = config.TELLURIC_REGIONS.get(arm, {})
    for kind in ("telluric", "deep_mask"):
        for index, (left, right) in enumerate(telluric_config.get(kind, []), start=1):
            regions.append((f"{kind}_{index}", float(left), float(right)))
    rows: list[dict[str, Any]] = []
    for label, left, right in regions:
        columns = (wave >= left) & (wave <= right)
        if not np.any(columns):
            continue
        selected_good = good[:, columns]
        selected_data = np.where(selected_good, data[:, columns], np.nan)
        selected_sigma = np.where(selected_good, sigma[:, columns], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append(
                {
                    "epoch": bundle["epoch"],
                    "arm": arm,
                    "product": bundle["product"],
                    "region": label,
                    "left_A": left,
                    "right_A": right,
                    "n_columns": int(np.count_nonzero(columns)),
                    "finite_fraction": float(np.mean(selected_good)),
                    "median_abs_data": float(np.nanmedian(np.abs(selected_data))),
                    "median_sigma": float(np.nanmedian(selected_sigma)),
                }
            )
    return rows


def _relativistic_doppler_factor(velocity_kms: np.ndarray | float) -> np.ndarray:
    beta = np.asarray(velocity_kms, dtype=float) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def stack_line_window(
    bundle: Mapping[str, Any],
    *,
    rest_vacuum_A: float,
    velocity_kms: np.ndarray | None = None,
    half_width_kms: float = 120.0,
    step_kms: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sample and coadd one vacuum transition from each native exposure row."""

    wavelength = np.asarray(bundle["wavelength"], dtype=float)
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    if wavelength.ndim == 1:
        wavelength = np.broadcast_to(wavelength, data.shape)
    if wavelength.shape != data.shape:
        raise ValueError("Wavelength and data arrays must have matching exposure rows.")
    trail = np.zeros(data.shape[0]) if velocity_kms is None else np.asarray(velocity_kms, dtype=float)
    if trail.shape != (data.shape[0],):
        raise ValueError("velocity_kms must have one value per exposure.")
    velocity_grid = np.arange(-half_width_kms, half_width_kms + 0.5 * step_kms, step_kms)
    rest_wave = float(rest_vacuum_A) * _relativistic_doppler_factor(velocity_grid)
    sampled_data = np.full((data.shape[0], velocity_grid.size), np.nan)
    sampled_sigma = np.full_like(sampled_data, np.nan)
    for row in range(data.shape[0]):
        observed_wave = rest_wave * _relativistic_doppler_factor(trail[row])
        order = np.argsort(wavelength[row])
        source_wave = wavelength[row, order]
        sampled_data[row] = np.interp(observed_wave, source_wave, data[row, order], left=np.nan, right=np.nan)
        sampled_sigma[row] = np.interp(observed_wave, source_wave, sigma[row, order], left=np.nan, right=np.nan)
    coadd, coadd_error, coverage = observed_spectrum(sampled_data, sampled_sigma)
    return {
        "velocity_kms": velocity_grid,
        "data": sampled_data,
        "sigma": sampled_sigma,
        "coadd": coadd,
        "coadd_error": coadd_error,
        "coverage": coverage,
    }


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def build_stacked_spectrum_bundle(
    records: Iterable[Mapping[str, Any]],
    *,
    group_by: str = "epoch_arm",
) -> dict[str, Any]:
    """Build one weighted observed-spectrum row per epoch or epoch/arm."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if group_by == "epoch":
            key = str(record["epoch"])
        elif group_by == "epoch_arm":
            key = f"{record['epoch']} {record['arm']}"
        else:
            raise ValueError("group_by must be 'epoch' or 'epoch_arm'.")
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for label, candidates in grouped.items():
        preferred = min(
            candidates,
            key=lambda record: 0 if record.get("product") == "timeseries" else 1,
        )
        mean, error, coverage = observed_spectrum(preferred["data"], preferred["sigma"])
        rows.append(
            {
                "label": label,
                "epoch": preferred["epoch"],
                "arm": preferred["arm"],
                "product": preferred["product"],
                "wavelength": wave_1d(preferred["wavelength"]),
                "spectrum": mean,
                "uncertainty": error,
                "coverage": coverage,
                "phase": _safe_mean(preferred.get("phase", [])),
            }
        )
    return {"group_by": group_by, "rows": rows}


def contrast_summary(stacked: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare each stacked row with the median of the remaining rows."""

    rows = list(stacked["rows"])
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        peers = [candidate for peer_index, candidate in enumerate(rows) if peer_index != index]
        if not peers:
            continue
        same_grid = [
            peer for peer in peers
            if np.array_equal(peer["wavelength"], row["wavelength"], equal_nan=True)
        ]
        if not same_grid:
            continue
        reference = np.nanmedian(np.stack([peer["spectrum"] for peer in same_grid]), axis=0)
        residual = np.asarray(row["spectrum"]) - reference
        finite = residual[np.isfinite(residual)]
        output.append(
            {
                "label": row["label"],
                "n_reference_rows": len(same_grid),
                "median_residual": float(np.median(finite)) if finite.size else np.nan,
                "rms_residual": float(np.sqrt(np.mean(np.square(finite)))) if finite.size else np.nan,
            }
        )
    return output


def coadd_in_planet_rest_frame(
    wavelength: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    velocity_kms: np.ndarray,
    *,
    rest_wavelength: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Shift exposure rows relativistically and inverse-variance coadd them."""

    values = np.asarray(data, dtype=float)
    uncertainty = np.asarray(sigma, dtype=float)
    source_wave = np.asarray(wavelength, dtype=float)
    if source_wave.ndim == 1:
        source_wave = np.broadcast_to(source_wave, values.shape)
    if source_wave.shape != values.shape or uncertainty.shape != values.shape:
        raise ValueError("wavelength, data, and sigma must have matching shapes.")
    velocity = np.asarray(velocity_kms, dtype=float)
    if velocity.shape != (values.shape[0],):
        raise ValueError("velocity_kms must have one value per exposure.")
    common_wave = wave_1d(source_wave) if rest_wavelength is None else np.asarray(rest_wavelength, dtype=float)
    shifted = np.full((values.shape[0], common_wave.size), np.nan)
    shifted_sigma = np.full_like(shifted, np.nan)
    for row in range(values.shape[0]):
        observed_sample = common_wave * _relativistic_doppler_factor(velocity[row])
        order = np.argsort(source_wave[row])
        shifted[row] = np.interp(observed_sample, source_wave[row, order], values[row, order], left=np.nan, right=np.nan)
        shifted_sigma[row] = np.interp(observed_sample, source_wave[row, order], uncertainty[row, order], left=np.nan, right=np.nan)
    coadd, coadd_error, coverage = observed_spectrum(shifted, shifted_sigma)
    return {
        "wavelength": common_wave,
        "data": shifted,
        "sigma": shifted_sigma,
        "coadd": coadd,
        "coadd_error": coadd_error,
        "coverage": coverage,
    }


def nan_box_smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    """Box-smooth finite values while ignoring missing samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    width = min(int(width), array.size)
    if width <= 1:
        return array.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    finite = np.isfinite(array)
    numerator = np.convolve(np.where(finite, array, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
