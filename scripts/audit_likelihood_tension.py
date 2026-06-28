#!/usr/bin/env python3
"""Audit prepared HRS bundles for likelihood/noise-model tension.

This script is intentionally read-only. It inspects the prepared
``wavelength.npy``, ``data.npy``, ``sigma.npy``, optional pre-SYSREM arrays,
and optional chunked ``U_sysrem.npz`` files, then writes compact CSV summaries
that help separate underestimated errors from covariance, bad chunks, weight
concentration, and SYSREM bookkeeping issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import config_utils


DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def slug_planet(value: str) -> str:
    return str(value).lower().replace("-", "")


def resolve_bundle_dir(args: argparse.Namespace, arm: str) -> Path:
    if args.input_root is not None:
        return args.input_root / slug_planet(args.planet) / args.epoch / arm
    config_utils.set_runtime_config("PLANET", args.planet)
    config_utils.set_runtime_config("RETRIEVAL_MODE", args.mode)
    if args.ephemeris is not None:
        config_utils.set_runtime_config("EPHEMERIS", args.ephemeris)
    return Path(
        config_utils.get_data_dir(
            planet=args.planet,
            epoch=args.epoch,
            arm=arm,
            mode=args.mode,
        )
    )


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "wavelength": np.load(bundle_dir / "wavelength.npy"),
        "data": np.load(bundle_dir / "data.npy"),
        "sigma": np.load(bundle_dir / "sigma.npy"),
        "phase": np.load(bundle_dir / "phase.npy"),
        "pre_sysrem_data": None,
        "pre_sysrem_sigma": None,
        "sysrem": {},
    }
    pre_data = bundle_dir / "pre_sysrem_data.npy"
    pre_sigma = bundle_dir / "pre_sysrem_sigma.npy"
    if pre_data.exists() and pre_sigma.exists():
        payload["pre_sysrem_data"] = np.load(pre_data)
        payload["pre_sysrem_sigma"] = np.load(pre_sigma)

    sysrem_path = bundle_dir / "U_sysrem.npz"
    if sysrem_path.exists():
        with np.load(sysrem_path) as sysrem:
            payload["sysrem"] = {name: np.asarray(sysrem[name]) for name in sysrem.files}
    return payload


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def finite_percentile(values: np.ndarray, percentiles: tuple[float, ...]) -> list[float]:
    vals = finite_values(values)
    if vals.size == 0:
        return [math.nan for _ in percentiles]
    return [float(x) for x in np.nanpercentile(vals, percentiles)]


def robust_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(arr - med), axis=axis)


def ratio_or_nan(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0.0:
        return math.nan
    return float(num / den)


def stage_arrays(bundle: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    stages = [("post_sysrem", np.asarray(bundle["data"]), np.asarray(bundle["sigma"]))]
    if bundle.get("pre_sysrem_data") is not None and bundle.get("pre_sysrem_sigma") is not None:
        stages.append(
            (
                "pre_sysrem",
                np.asarray(bundle["pre_sysrem_data"]),
                np.asarray(bundle["pre_sysrem_sigma"]),
            )
        )
    return stages


def normalized_residual(data: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = np.full_like(data, np.nan, dtype=float)
    mask = np.isfinite(data) & np.isfinite(sigma) & (sigma > 0)
    z[mask] = data[mask] / sigma[mask]
    return z


def top_weight_fraction(sigma: np.ndarray, top_fraction: float) -> float:
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return math.nan
    weights = 1.0 / np.square(sigma[mask])
    weights = np.sort(weights.reshape(-1))[::-1]
    n_top = max(1, int(math.ceil(weights.size * float(top_fraction))))
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    return float(np.sum(weights[:n_top]) / total)


def lag_correlations(data: np.ndarray, lag: int, columns: np.ndarray | None = None) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if columns is not None:
        data = data[:, columns]
    if data.ndim != 2 or data.shape[1] <= lag:
        return np.asarray([], dtype=float)

    values: list[float] = []
    for row in data:
        left = row[:-lag]
        right = row[lag:]
        mask = np.isfinite(left) & np.isfinite(right)
        if int(np.sum(mask)) < 10:
            continue
        a = left[mask] - np.nanmean(left[mask])
        b = right[mask] - np.nanmean(right[mask])
        den = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
        if den > 0.0:
            values.append(float(np.sum(a * b) / den))
    return np.asarray(values, dtype=float)


def approximate_neff_fraction(median_corr_by_lag: dict[int, float]) -> tuple[float, float]:
    positive_sum = 0.0
    for lag in sorted(median_corr_by_lag):
        corr = median_corr_by_lag[lag]
        if np.isfinite(corr) and corr > 0.0:
            positive_sum += float(corr)
    tau = 1.0 + 2.0 * positive_sum
    return tau, 1.0 / tau if tau > 0.0 else math.nan


def chunk_label_for(labels: np.ndarray | None, indices: np.ndarray) -> str:
    if labels is None:
        return ""
    values = np.unique(np.asarray(labels)[indices])
    if values.size == 1:
        return str(int(values[0]))
    return "mixed:" + "|".join(str(int(value)) for value in values)


def summarize_stage(
    *,
    arm: str,
    stage: str,
    data: np.ndarray,
    sigma: np.ndarray,
    wavelength: np.ndarray,
    lags: tuple[int, ...],
) -> dict[str, Any]:
    z = normalized_residual(data, sigma)
    exp_scatter = robust_std(data, axis=1)
    exp_sigma = np.nanmedian(sigma, axis=1)
    wav_scatter = robust_std(data, axis=0)
    wav_sigma = np.nanmedian(sigma, axis=0)
    exp_ratio = exp_scatter / exp_sigma
    wav_ratio = wav_scatter / wav_sigma

    lag_medians: dict[int, float] = {}
    for lag in lags:
        corrs = lag_correlations(data, lag)
        lag_medians[lag] = float(np.nanmedian(corrs)) if corrs.size else math.nan
    tau, neff_fraction = approximate_neff_fraction(lag_medians)

    z_abs = np.abs(z)
    z_p50, z_p90, z_p99 = finite_percentile(z_abs, (50, 90, 99))
    exp_q10, exp_q50, exp_q90 = finite_percentile(exp_ratio, (10, 50, 90))
    wav_q10, wav_q50, wav_q90 = finite_percentile(wav_ratio, (10, 50, 90))
    return {
        "arm": arm,
        "stage": stage,
        "n_exposures": int(data.shape[0]) if data.ndim == 2 else 1,
        "n_wave": int(np.asarray(wavelength).size),
        "finite_fraction": float(np.mean(np.isfinite(data) & np.isfinite(sigma) & (sigma > 0))),
        "z_robust_std": float(robust_std(finite_values(z))),
        "abs_z_p50": z_p50,
        "abs_z_p90": z_p90,
        "abs_z_p99": z_p99,
        "exposure_scatter_sigma_q10": exp_q10,
        "exposure_scatter_sigma_q50": exp_q50,
        "exposure_scatter_sigma_q90": exp_q90,
        "wavelength_scatter_sigma_q10": wav_q10,
        "wavelength_scatter_sigma_q50": wav_q50,
        "wavelength_scatter_sigma_q90": wav_q90,
        "top_1pct_weight_fraction": top_weight_fraction(sigma, 0.01),
        "top_0p1pct_weight_fraction": top_weight_fraction(sigma, 0.001),
        "autocorr_tau_approx": tau,
        "autocorr_neff_fraction_approx": neff_fraction,
    }


def exposure_rows(
    *,
    arm: str,
    stage: str,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    z = normalized_residual(data, sigma)
    for idx in range(data.shape[0]):
        scatter = float(robust_std(data[idx]))
        median_sigma = float(np.nanmedian(sigma[idx]))
        z_abs_p99 = finite_percentile(np.abs(z[idx]), (99,))[0]
        rows.append(
            {
                "arm": arm,
                "stage": stage,
                "exposure": idx,
                "phase": float(phase[idx]) if idx < phase.size else math.nan,
                "data_median": float(np.nanmedian(data[idx])),
                "data_robust_scatter": scatter,
                "median_sigma": median_sigma,
                "scatter_over_sigma": ratio_or_nan(scatter, median_sigma),
                "z_robust_std": float(robust_std(finite_values(z[idx]))),
                "abs_z_p99": z_abs_p99,
            }
        )
    return rows


def wavelength_chunk_rows(
    *,
    arm: str,
    stage: str,
    data: np.ndarray,
    sigma: np.ndarray,
    wavelength: np.ndarray,
    chunk_count: int,
    lags: tuple[int, ...],
    chunk_labels: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices_by_chunk = np.array_split(np.arange(wavelength.size), int(chunk_count))
    for idx, indices in enumerate(indices_by_chunk):
        if indices.size == 0:
            continue
        d = data[:, indices]
        s = sigma[:, indices]
        z = normalized_residual(d, s)
        scatter = float(robust_std(finite_values(d)))
        median_sigma = float(np.nanmedian(s))
        lag1 = lag_correlations(data, 1, columns=indices)
        lag_medians = {}
        for lag in lags:
            corrs = lag_correlations(data, lag, columns=indices)
            lag_medians[lag] = float(np.nanmedian(corrs)) if corrs.size else math.nan
        tau, neff_fraction = approximate_neff_fraction(lag_medians)
        rows.append(
            {
                "arm": arm,
                "stage": stage,
                "chunk": idx,
                "sysrem_label": chunk_label_for(chunk_labels, indices),
                "n_wave": int(indices.size),
                "wavelength_min": float(np.nanmin(wavelength[indices])),
                "wavelength_max": float(np.nanmax(wavelength[indices])),
                "data_robust_scatter": scatter,
                "median_sigma": median_sigma,
                "scatter_over_sigma": ratio_or_nan(scatter, median_sigma),
                "z_robust_std": float(robust_std(finite_values(z))),
                "abs_z_p99": finite_percentile(np.abs(z), (99,))[0],
                "top_1pct_weight_fraction": top_weight_fraction(s, 0.01),
                "lag1_corr_median": float(np.nanmedian(lag1)) if lag1.size else math.nan,
                "autocorr_tau_approx": tau,
                "autocorr_neff_fraction_approx": neff_fraction,
            }
        )
    return rows


def autocorr_rows(
    *,
    arm: str,
    stage: str,
    data: np.ndarray,
    lags: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lag in lags:
        corrs = lag_correlations(data, lag)
        q10, q50, q90 = finite_percentile(corrs, (10, 50, 90))
        rows.append(
            {
                "arm": arm,
                "stage": stage,
                "lag_pixels": lag,
                "n_exposures_used": int(corrs.size),
                "corr_q10": q10,
                "corr_q50": q50,
                "corr_q90": q90,
            }
        )
    return rows


def sysrem_rows(arm: str, wavelength: np.ndarray, sysrem: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if "chunk_labels" not in sysrem:
        return []
    labels = np.asarray(sysrem["chunk_labels"], dtype=int)
    inst_nus = 1.0e8 / np.asarray(wavelength, dtype=float)
    sort_idx = np.argsort(inst_nus) if np.any(np.diff(inst_nus) <= 0) else np.arange(wavelength.size)
    old_to_new = np.empty_like(sort_idx)
    old_to_new[sort_idx] = np.arange(sort_idx.size, dtype=int)

    rows: list[dict[str, Any]] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        old_indices = np.where(labels == label)[0]
        new_indices = np.sort(old_to_new[old_indices])
        gaps = np.diff(new_indices)
        rows.append(
            {
                "arm": arm,
                "sysrem_label": label,
                "n_wave": int(old_indices.size),
                "basis_count": int(np.asarray(sysrem.get("basis_counts", []))[label])
                if "basis_counts" in sysrem and label < np.asarray(sysrem["basis_counts"]).size
                else "",
                "wavelength_min": float(np.nanmin(wavelength[old_indices])),
                "wavelength_max": float(np.nanmax(wavelength[old_indices])),
                "retrieval_sort_reorders_columns": bool(not np.array_equal(sort_idx, np.arange(sort_idx.size))),
                "sorted_position_min": int(np.min(new_indices)),
                "sorted_position_max": int(np.max(new_indices)),
                "contiguous_after_sort": bool(new_indices.size <= 1 or np.all(gaps == 1)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default=config.EPHEMERIS)
    parser.add_argument("--mode", choices=("transmission", "emission"), default="transmission")
    parser.add_argument("--epoch", default="20190504")
    parser.add_argument("--arms", default="red,blue")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional root with layout <planet>/<epoch>/<arm>; defaults to config_utils paths.",
    )
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--lags", default=",".join(str(lag) for lag in DEFAULT_LAGS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Defaults to diagnostics/likelihood_tension_<planet>_<epoch>.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    arms = parse_csv_tuple(args.arms)
    lags = parse_int_tuple(args.lags)
    out_dir = args.output_dir or (
        REPO_ROOT / "diagnostics" / f"likelihood_tension_{slug_planet(args.planet)}_{args.mode}_{args.epoch}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    autocorr: list[dict[str, Any]] = []
    sysrem_summary: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "planet": args.planet,
        "ephemeris": args.ephemeris,
        "mode": args.mode,
        "epoch": args.epoch,
        "arms": list(arms),
        "lags": list(lags),
        "chunks": int(args.chunks),
        "bundles": {},
    }

    for arm in arms:
        bundle_dir = resolve_bundle_dir(args, arm)
        manifest["bundles"][arm] = str(bundle_dir)
        if not bundle_dir.exists():
            print(f"missing bundle for {arm}: {bundle_dir}", file=sys.stderr)
            continue
        bundle = load_bundle(bundle_dir)
        wavelength = np.asarray(bundle["wavelength"], dtype=float)
        labels = None
        if "chunk_labels" in bundle["sysrem"]:
            labels = np.asarray(bundle["sysrem"]["chunk_labels"], dtype=int)
        sysrem_summary.extend(sysrem_rows(arm, wavelength, bundle["sysrem"]))

        for stage, data, sigma in stage_arrays(bundle):
            summary.append(
                summarize_stage(
                    arm=arm,
                    stage=stage,
                    data=np.asarray(data, dtype=float),
                    sigma=np.asarray(sigma, dtype=float),
                    wavelength=wavelength,
                    lags=lags,
                )
            )
            exposures.extend(
                exposure_rows(
                    arm=arm,
                    stage=stage,
                    data=np.asarray(data, dtype=float),
                    sigma=np.asarray(sigma, dtype=float),
                    phase=np.asarray(bundle["phase"], dtype=float),
                )
            )
            chunks.extend(
                wavelength_chunk_rows(
                    arm=arm,
                    stage=stage,
                    data=np.asarray(data, dtype=float),
                    sigma=np.asarray(sigma, dtype=float),
                    wavelength=wavelength,
                    chunk_count=args.chunks,
                    lags=lags,
                    chunk_labels=labels,
                )
            )
            autocorr.extend(
                autocorr_rows(
                    arm=arm,
                    stage=stage,
                    data=np.asarray(data, dtype=float),
                    lags=lags,
                )
            )

    write_csv(out_dir / "bundle_summary.csv", summary)
    write_csv(out_dir / "exposure_stats.csv", exposures)
    write_csv(out_dir / "wavelength_chunk_stats.csv", chunks)
    write_csv(out_dir / "autocorrelation.csv", autocorr)
    write_csv(out_dir / "sysrem_chunks.csv", sysrem_summary)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
