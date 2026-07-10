#!/usr/bin/env python3
"""Deep-dive retrieval-space residual/model comparisons for saved HRS runs.

This diagnostic is intentionally post-hoc. It reads prepared time-series
bundles and saved retrieval runs, rebuilds the median processed model through
the same retrieval diagnostic path, and writes tables/figures that separate:

- native/pre-SYSREM residual arrays
- post-SYSREM retrieval-space residual arrays
- observed-frame processed model arrays
- matched-filter-scaled residuals, observed - alpha * model
- sign controls and coarse Kp-dRV likelihood controls
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


C_KMS = 299792.458
DEFAULT_LABELS = ("tiny_red_fe_only_sigma1p5", "tiny_red_fe_only_sigma3", "tiny_red_fe_only_sigma8")
DEFAULT_CONTROLS = ("observed", "negative", "phase_shuffle", "wavelength_roll")
_PLT = None


def parse_version_triplet(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(value))
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def preflight_model_runtime() -> str | None:
    """Return a user-facing error if model reconstruction cannot be imported."""
    try:
        jax_version = importlib_metadata.version("jax")
    except importlib_metadata.PackageNotFoundError:
        return "JAX is not installed in this Python environment; rerun with --skip-model or use the retrieval/HPC environment."
    try:
        jaxlib_version = importlib_metadata.version("jaxlib")
    except importlib_metadata.PackageNotFoundError:
        return "jaxlib is not installed in this Python environment; rerun with --skip-model or use the retrieval/HPC environment."

    jax_triplet = parse_version_triplet(jax_version)
    jaxlib_triplet = parse_version_triplet(jaxlib_version)
    if jax_triplet is not None and jaxlib_triplet is not None and jaxlib_triplet > jax_triplet:
        return (
            "JAX package mismatch before import: "
            f"jax=={jax_version}, jaxlib=={jaxlib_version}. "
            "jaxlib is newer than jax, which leaves jax partially initialized after a failed import. "
            "Install a compatible jax/jaxlib pair in this environment, or rerun with --skip-model."
        )

    try:
        from pipeline import diagnostics as _diag_module  # noqa: F401
    except Exception as exc:
        return f"model reconstruction import failed: {type(exc).__name__}: {exc}"
    return None


def get_pyplot():
    global _PLT
    if _PLT is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PLT = plt
    return _PLT


def parse_csv_tuple(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def slug_planet(value: str) -> str:
    return str(value).lower().replace("-", "").replace(" ", "")


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def robust_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(arr - med), axis=axis)


def finite_percentile(values: np.ndarray, q: float) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return math.nan
    return float(np.nanpercentile(vals, q))


def finite_median(values: np.ndarray) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return math.nan
    return float(np.nanmedian(vals))


def finite_min(values: np.ndarray) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return math.nan
    return float(np.nanmin(vals))


def finite_max(values: np.ndarray) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return math.nan
    return float(np.nanmax(vals))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {path}")


def parse_run_config_text(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {"run_config_text": path.read_text(encoding="utf-8", errors="replace")}
    current_list_key: str | None = None
    list_headers = {
        "Molecules (HITEMP):": "molecules_hitemp",
        "Molecules (ExoMol):": "molecules_exomol",
        "Atomic species:": "atoms",
    }
    for raw_line in parsed["run_config_text"].splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if set(stripped) in ({"="}, {"-"}):
            continue
        if stripped in list_headers:
            current_list_key = list_headers[stripped]
            parsed.setdefault(current_list_key, [])
            continue
        if current_list_key is not None:
            if stripped.startswith("- "):
                parsed[current_list_key].append(stripped[2:].strip())
                continue
            current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def discover_runs_by_label(run_root: Path, labels: tuple[str, ...]) -> tuple[list[Path], dict[str, list[str]]]:
    selected: list[Path] = []
    duplicates: dict[str, list[str]] = {}
    for label in labels:
        candidates: list[Path] = []
        for config_path in sorted(run_root.glob("**/run_config.log")):
            parsed = parse_run_config_text(config_path)
            if parsed.get("Diagnostic label") == label:
                candidates.append(config_path.parent)
        if not candidates:
            print(f"warning: no run found for diagnostic label {label!r}", file=sys.stderr)
            continue
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        selected.append(candidates[0])
        if len(candidates) > 1:
            duplicates[label] = [str(path) for path in candidates]
    return selected, duplicates


def discover_run_dirs(args: argparse.Namespace) -> tuple[list[Path], dict[str, list[str]]]:
    explicit = [Path(path).expanduser() for path in args.runs]
    if explicit:
        return [path.resolve() for path in explicit], {}
    labels = parse_csv_tuple(args.run_labels) or DEFAULT_LABELS
    return discover_runs_by_label(Path(args.run_root).expanduser(), labels)


def load_npz_numeric(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        out: dict[str, np.ndarray] = {}
        for key in payload.files:
            arr = np.asarray(payload[key])
            if arr.dtype == object:
                continue
            out[key] = arr
        return out


def load_posterior_medians(run_dir: Path) -> dict[str, Any]:
    posterior_path = run_dir / "posterior_sample.npz"
    if not posterior_path.exists():
        return {}
    medians: dict[str, Any] = {}
    for key, arr in load_npz_numeric(posterior_path).items():
        if arr.shape == ():
            medians[key] = arr.item()
            continue
        vals = finite_values(arr)
        if vals.size:
            medians[key] = float(np.nanmedian(vals))
    return medians


def parse_mcmc_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "mcmc_summary.txt"
    if not path.exists():
        return {
            "mcmc_summary_exists": False,
            "divergences": "",
            "max_finite_r_hat": "",
            "n_nan_n_eff": "",
            "n_inf_r_hat": "",
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    divergence_match = re.search(r"Number of divergences:\s*([0-9]+)", text)
    r_hats: list[float] = []
    n_eff_nan = 0
    r_hat_inf = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        if parts[0] in {"mean", "Number"}:
            continue
        n_eff_token = parts[-2]
        r_hat_token = parts[-1]
        if n_eff_token.lower() == "nan":
            n_eff_nan += 1
        if r_hat_token.lower() == "inf":
            r_hat_inf += 1
            continue
        try:
            r_hats.append(float(r_hat_token))
        except ValueError:
            continue
    return {
        "mcmc_summary_exists": True,
        "divergences": int(divergence_match.group(1)) if divergence_match else "",
        "max_finite_r_hat": max(r_hats) if r_hats else "",
        "n_nan_n_eff": n_eff_nan,
        "n_inf_r_hat": r_hat_inf,
    }


def summarize_run_artifact(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_cfg = parse_run_config_text(run_dir / "run_config.log")
    row: dict[str, Any] = {
        "run_label": run_cfg.get("Diagnostic label", run_dir.name),
        "run_dir": str(run_dir),
        "component": "",
        "mode": run_cfg.get("Mode", ""),
        "epoch": run_cfg.get("Epoch", ""),
        "observing_mode": run_cfg.get("Observing mode", ""),
        "sigma_scale": run_cfg.get("Spectroscopic sigma scale", ""),
        "likelihood_kind": "matched_filter",
        "n_exposures": "",
        "n_wave": "",
        "alpha_median": "",
        "alpha_q16": "",
        "alpha_q84": "",
        "alpha_negative_fraction": "",
        "weighted_corr_median": "",
        "weighted_corr_q16": "",
        "weighted_corr_q84": "",
        "observed_model_weighted_corr_all_pixels": "",
        "logL_observed": "",
        "logL_negative_data": "",
        "logL_observed_minus_negative_data": "",
        "logL_negative_model": "",
        "logL_observed_minus_negative_model": "",
        "residual_robust_std": "",
        "observed_robust_std": "",
        "model_robust_std": "",
        "recommendation": "model reconstruction skipped; run without --skip-model in a working retrieval env",
        "posterior_sample_exists": (run_dir / "posterior_sample.npz").exists(),
        "posterior_sample_by_chain_exists": (run_dir / "posterior_sample_by_chain.npz").exists(),
        "diagnostics_summary_exists": (run_dir / "diagnostics" / "mcmc_diagnostics_summary.json").exists(),
    }
    row.update(parse_mcmc_summary(run_dir))
    return row


def bundle_dir_for(mode: str, planet: str, epoch: str, arm: str) -> Path:
    return REPO_ROOT / "input" / "hrs" / mode / slug_planet(planet) / epoch / arm


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sysrem_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "has_U_sysrem": False,
            "sysrem_basis_counts": "",
            "sysrem_chunk_names": "",
            "sysrem_stop_tol": "",
        }
    with np.load(path, allow_pickle=True) as payload:
        basis_counts = np.asarray(payload["basis_counts"]).astype(int).tolist() if "basis_counts" in payload else []
        chunk_names = [str(x) for x in np.asarray(payload["chunk_names"]).tolist()] if "chunk_names" in payload else []
        stop_tol = ""
        if "sysrem_stop_delta_stddev" in payload:
            raw = np.asarray(payload["sysrem_stop_delta_stddev"])
            stop_tol = float(raw.reshape(-1)[0]) if raw.size else ""
    return {
        "has_U_sysrem": True,
        "sysrem_basis_counts": basis_counts,
        "sysrem_chunk_names": chunk_names,
        "sysrem_stop_tol": stop_tol,
    }


def summarize_bundle(mode: str, planet: str, epoch: str, arm: str) -> dict[str, Any]:
    bundle_dir = bundle_dir_for(mode, planet, epoch, arm)
    row: dict[str, Any] = {
        "arm": arm,
        "bundle_dir": str(bundle_dir),
        "bundle_exists": bundle_dir.exists(),
    }
    if not bundle_dir.exists():
        return row

    wavelength = np.load(bundle_dir / "wavelength.npy")
    data = np.load(bundle_dir / "data.npy")
    sigma = np.load(bundle_dir / "sigma.npy")
    phase = np.load(bundle_dir / "phase.npy")
    metadata = load_json(bundle_dir / "timeseries_prep.json")
    pre_data_path = bundle_dir / "pre_sysrem_data.npy"
    pre_sigma_path = bundle_dir / "pre_sysrem_sigma.npy"
    pre_present = pre_data_path.exists() and pre_sigma_path.exists()
    pre_diff_rms = math.nan
    pre_corr = math.nan
    if pre_present:
        pre_data = np.load(pre_data_path)
        if pre_data.shape == data.shape:
            diff = data - pre_data
            pre_diff_rms = float(np.sqrt(np.nanmean(np.square(diff))))
            pre_corr = weighted_correlation(data, pre_data, sigma)

    row.update(
        {
            "data_shape": tuple(int(x) for x in data.shape),
            "sigma_shape": tuple(int(x) for x in sigma.shape),
            "wavelength_shape": tuple(int(x) for x in wavelength.shape),
            "wavelength_min_A": finite_min(wavelength),
            "wavelength_max_A": finite_max(wavelength),
            "phase_min": finite_min(phase),
            "phase_max": finite_max(phase),
            "run_sysrem": metadata.get("run_sysrem"),
            "subtract_median": metadata.get("subtract_median"),
            "doppler_shadow_applied": metadata.get("doppler_shadow_applied"),
            "doppler_shadow_scaling": metadata.get("doppler_shadow_scaling"),
            "pre_sysrem_present": pre_present,
            "post_minus_pre_rms": pre_diff_rms,
            "post_pre_weighted_corr": pre_corr,
            "post_z_robust_std": float(robust_std(finite_values(data / np.clip(sigma, 1.0e-30, None)))),
            "data_robust_std": float(robust_std(finite_values(data))),
            "sigma_median": finite_median(sigma),
        }
    )
    row.update(load_sysrem_summary(bundle_dir / "U_sysrem.npz"))
    return row


def weighted_correlation(data: np.ndarray, model: np.ndarray, sigma: np.ndarray) -> float:
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(data) & np.isfinite(model) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.any(mask):
        return math.nan
    w = 1.0 / np.square(np.clip(sigma[mask], 1.0e-30, None))
    d = data[mask]
    m = model[mask]
    num = float(np.sum(w * d * m))
    den = math.sqrt(float(np.sum(w * d * d) * np.sum(w * m * m)))
    return num / den if den > 0.0 else math.nan


def matched_filter_arrays(
    data: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w = 1.0 / np.square(np.clip(sigma, 1.0e-30, None))
    num = np.sum(w * data * model, axis=1)
    model_power = np.sum(w * np.square(model), axis=1)
    data_power = np.sum(w * np.square(data), axis=1)
    alpha = num / np.clip(model_power, 1.0e-30, None)
    corr = num / np.sqrt(np.clip(data_power * model_power, 1.0e-30, None))
    residual = data - alpha[:, None] * model
    return alpha, corr, residual


def planet_rv_kms(phase: np.ndarray, kp: float, vsys: float, drv: float) -> np.ndarray:
    return float(kp) * np.sin(2.0 * np.pi * np.asarray(phase, dtype=float)) + float(vsys) + float(drv)


def component_drv_value(params: dict[str, Any], component_name: str) -> float:
    for key in (f"{component_name}/dRV", "spectroscopy/dRV", "dRV"):
        if key in params:
            try:
                return float(params[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def control_data(control: str, observed: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    control = str(control).lower().strip()
    if control == "observed":
        return None
    if control == "negative":
        return -np.asarray(observed, dtype=float)
    if control == "phase_shuffle":
        return np.asarray(observed, dtype=float)[rng.permutation(observed.shape[0])]
    if control == "wavelength_roll":
        return np.roll(np.asarray(observed, dtype=float), shift=max(1, observed.shape[1] // 3), axis=1)
    raise ValueError(f"Unknown control: {control}")


def save_processed_panel(
    path: Path,
    *,
    wavelength: np.ndarray,
    phase: np.ndarray,
    observed: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    wavelength_stride: int,
    title: str,
) -> None:
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    wavelength = np.asarray(wavelength, dtype=float)
    order = np.argsort(wavelength)
    stride = max(1, int(wavelength_stride))
    idx = order[::stride]
    wav = wavelength[idx]
    extent = [float(wav[0]), float(wav[-1]), float(np.nanmin(phase)), float(np.nanmax(phase))]
    panels = [
        ("Observed data.npy\npost-SYSREM, shadow-corrected, likelihood space", observed),
        ("Processed median-posterior model\nobserved frame, demeaned + SYSREM-distorted", model),
        ("Observed - alpha*model\nmatched-filter residual space", residual),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for ax, (panel_title, arr) in zip(axes, panels):
        arr_plot = np.asarray(arr, dtype=float)[:, idx]
        limit = finite_percentile(np.abs(arr_plot), 98.0)
        if not math.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        im = ax.imshow(
            arr_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Wavelength [A]")
        fig.colorbar(im, ax=ax, shrink=0.82)
    axes[0].set_ylabel("Orbital phase")
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def rest_frame_stack(
    *,
    wavelength: np.ndarray,
    phase: np.ndarray,
    observed: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    kp: float,
    vsys: float,
    drv: float,
) -> dict[str, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=float)
    order = np.argsort(wavelength)
    wave = wavelength[order]
    phase = np.asarray(phase, dtype=float)
    observed = np.asarray(observed, dtype=float)[:, order]
    model_scaled = (np.asarray(alpha, dtype=float)[:, None] * np.asarray(model, dtype=float))[:, order]
    sigma = np.asarray(sigma, dtype=float)[:, order]
    rv = planet_rv_kms(phase, kp=kp, vsys=vsys, drv=drv)

    obs_rows = []
    model_rows = []
    sigma_rows = []
    for i, rv_i in enumerate(rv):
        rest_wave = wave / (1.0 + rv_i / C_KMS)
        obs_rows.append(np.interp(wave, rest_wave, observed[i], left=np.nan, right=np.nan))
        model_rows.append(np.interp(wave, rest_wave, model_scaled[i], left=np.nan, right=np.nan))
        sigma_rows.append(np.interp(wave, rest_wave, sigma[i], left=np.nan, right=np.nan))
    obs_rest = np.asarray(obs_rows)
    model_rest = np.asarray(model_rows)
    sigma_rest = np.asarray(sigma_rows)
    weights = np.where(np.isfinite(sigma_rest) & (sigma_rest > 0.0), 1.0 / np.square(sigma_rest), 0.0)
    weight_sum = np.sum(weights, axis=0)
    obs_stack = np.sum(weights * np.nan_to_num(obs_rest, nan=0.0), axis=0) / np.clip(weight_sum, 1.0e-30, None)
    model_stack = np.sum(weights * np.nan_to_num(model_rest, nan=0.0), axis=0) / np.clip(weight_sum, 1.0e-30, None)
    residual_stack = obs_stack - model_stack
    valid = weight_sum > 0.0
    obs_stack = np.where(valid, obs_stack, np.nan)
    model_stack = np.where(valid, model_stack, np.nan)
    residual_stack = np.where(valid, residual_stack, np.nan)
    return {
        "wavelength_A": wave,
        "observed_rest_stack": obs_stack,
        "alpha_model_rest_stack": model_stack,
        "residual_rest_stack": residual_stack,
        "weight_sum": weight_sum,
        "rv_kms": rv,
    }


def save_rest_stack_plot(
    path: Path,
    stack: dict[str, np.ndarray],
    *,
    max_points: int,
    title: str,
) -> None:
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(stack["wavelength_A"], dtype=float)
    stride = max(1, int(math.ceil(wave.size / max(1, int(max_points)))))
    idx = np.arange(0, wave.size, stride, dtype=int)
    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax.plot(wave[idx], stack["observed_rest_stack"][idx], color="k", lw=0.9, label="Observed rest-frame stack")
    ax.plot(
        wave[idx],
        stack["alpha_model_rest_stack"][idx],
        color="C0",
        lw=1.0,
        label="alpha-scaled processed model stack",
    )
    ax.set_ylabel("Processed residual flux")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    ax_resid.plot(wave[idx], stack["residual_rest_stack"][idx], color="C3", lw=0.9)
    ax_resid.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax_resid.set_xlabel("Planet-rest wavelength [A]")
    ax_resid.set_ylabel("Obs - alpha*model")
    ax_resid.grid(True, alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def save_surface_plot(path: Path, scan: dict[str, Any]) -> None:
    from pipeline import diagnostics as diag_module

    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    diag_module.plot_kp_drv_surface(scan, ax=ax)
    ax.set_title(f"{scan['component_name']}: {scan.get('control', 'observed')}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def summarize_scan(control: str, scan: dict[str, Any], component_name: str) -> dict[str, Any]:
    surface = np.asarray(scan.get("surface", scan["log_likelihood"]), dtype=float)
    finite = finite_values(surface)
    if finite.size:
        median = float(np.nanmedian(finite))
        p95 = float(np.nanpercentile(finite, 95.0))
    else:
        median = math.nan
        p95 = math.nan
    best = float(scan["best_surface_value"])
    return {
        "component": component_name,
        "control": control,
        "best_kp": scan["best_params"]["Kp"],
        "best_drv": scan["best_params"]["dRV"],
        "best_surface": best,
        "surface_median": median,
        "surface_p95": p95,
        "best_minus_median": best - median if math.isfinite(best) and math.isfinite(median) else math.nan,
        "best_minus_p95": best - p95 if math.isfinite(best) and math.isfinite(p95) else math.nan,
    }


def analyze_run(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    from pipeline import diagnostics as diag_module

    run_dir = run_dir.resolve()
    run_cfg = parse_run_config_text(run_dir / "run_config.log")
    run_label = str(run_cfg.get("Diagnostic label", run_dir.name))
    print(f"\n== {run_label}: {run_dir} ==")

    config_payload = diag_module.build_diag_config_from_run_dir(run_dir)
    if args.apply_sysrem is not None:
        config_payload["apply_sysrem"] = bool(args.apply_sysrem)
    context = diag_module.build_diagnostic_context(**config_payload)
    posterior = load_posterior_medians(run_dir)
    if not posterior:
        raise FileNotFoundError(f"No numeric posterior_sample.npz values found in {run_dir}")

    selected_components = parse_csv_tuple(args.components) or context.spectroscopic_component_names
    run_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    data_paths: list[str] = []

    for component_name in selected_components:
        component = context.spectroscopic_components[component_name]
        model_ts, _atmo_state = diag_module.synthesize_processed_model_timeseries(
            context,
            posterior,
            component_name=component_name,
        )
        observed = np.asarray(component.data, dtype=float)
        sigma = np.asarray(component.sigma, dtype=float)
        model_ts = np.asarray(model_ts, dtype=float)
        alpha, corr, residual = matched_filter_arrays(observed, model_ts, sigma)
        likelihood_kind = str(component.observation_config.likelihood_kind)
        logl_obs = diag_module.spectroscopic_log_likelihood(
            observed,
            model_ts,
            sigma,
            likelihood_kind=likelihood_kind,
        )
        logl_negative = diag_module.spectroscopic_log_likelihood(
            -observed,
            model_ts,
            sigma,
            likelihood_kind=likelihood_kind,
        )
        logl_unscaled_negative_model = diag_module.spectroscopic_log_likelihood(
            observed,
            -model_ts,
            sigma,
            likelihood_kind=likelihood_kind,
        )
        negative_alpha_fraction = float(np.mean(alpha < 0.0))
        if negative_alpha_fraction >= 0.75:
            recommendation = "plain model has opposite sign to data; interpret alpha*model or -model overlay"
        elif negative_alpha_fraction <= 0.25:
            recommendation = "plain model sign broadly matches data, but alpha*model is still the likelihood object"
        else:
            recommendation = "mixed alpha signs; inspect exposure-level and rest-frame panels"

        component_dir = out_dir / run_label / component_name
        component_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = component_dir / "processed_model_comparison_arrays.npz"
        np.savez_compressed(
            arrays_path,
            wavelength_A=np.asarray(component.wav_obs),
            phase=np.asarray(component.phase),
            observed=observed,
            sigma=sigma,
            model=model_ts,
            alpha=alpha,
            weighted_corr=corr,
            matched_filter_residual=residual,
        )
        data_paths.append(str(arrays_path))
        print(f"wrote {arrays_path}")

        panel_path = component_dir / "processed_observed_model_residuals.png"
        save_processed_panel(
            panel_path,
            wavelength=np.asarray(component.wav_obs),
            phase=np.asarray(component.phase),
            observed=observed,
            model=model_ts,
            residual=residual,
            wavelength_stride=args.wavelength_stride,
            title=f"{run_label} / {component_name}: retrieval-space comparison",
        )
        figure_paths.append(str(panel_path))

        drv = component_drv_value(posterior, component_name)
        kp = float(posterior.get("Kp", context.model_params["Kp"]))
        vsys = float(context.model_params["RV_abs"])
        stack = rest_frame_stack(
            wavelength=np.asarray(component.wav_obs),
            phase=np.asarray(component.phase),
            observed=observed,
            model=model_ts,
            sigma=sigma,
            alpha=alpha,
            kp=kp,
            vsys=vsys,
            drv=drv,
        )
        stack_path = component_dir / "rest_frame_stack_arrays.npz"
        np.savez_compressed(stack_path, **stack)
        data_paths.append(str(stack_path))
        print(f"wrote {stack_path}")

        stack_plot_path = component_dir / "rest_frame_observed_alpha_model_stack.png"
        save_rest_stack_plot(
            stack_plot_path,
            stack,
            max_points=args.rest_stack_max_points,
            title=f"{run_label} / {component_name}: planet-rest stack, alpha-scaled model",
        )
        figure_paths.append(str(stack_plot_path))

        for i, (phase_i, alpha_i, corr_i) in enumerate(zip(np.asarray(component.phase), alpha, corr)):
            alpha_rows.append(
                {
                    "run_label": run_label,
                    "run_dir": str(run_dir),
                    "component": component_name,
                    "exposure_index": i,
                    "phase": float(phase_i),
                    "alpha": float(alpha_i),
                    "weighted_corr": float(corr_i),
                }
            )

        row = {
            "run_label": run_label,
            "run_dir": str(run_dir),
            "component": component_name,
            "mode": context.mode,
            "epoch": context.epoch,
            "observing_mode": run_cfg.get("Observing mode", ""),
            "sigma_scale": config_payload.get("sigma_scale", 1.0),
            "likelihood_kind": likelihood_kind,
            "n_exposures": int(observed.shape[0]),
            "n_wave": int(observed.shape[1]),
            "alpha_median": finite_median(alpha),
            "alpha_q16": finite_percentile(alpha, 16.0),
            "alpha_q84": finite_percentile(alpha, 84.0),
            "alpha_negative_fraction": negative_alpha_fraction,
            "weighted_corr_median": finite_median(corr),
            "weighted_corr_q16": finite_percentile(corr, 16.0),
            "weighted_corr_q84": finite_percentile(corr, 84.0),
            "observed_model_weighted_corr_all_pixels": weighted_correlation(observed, model_ts, sigma),
            "logL_observed": float(logl_obs),
            "logL_negative_data": float(logl_negative),
            "logL_observed_minus_negative_data": float(logl_obs - logl_negative),
            "logL_negative_model": float(logl_unscaled_negative_model),
            "logL_observed_minus_negative_model": float(logl_obs - logl_unscaled_negative_model),
            "residual_robust_std": float(robust_std(finite_values(residual))),
            "observed_robust_std": float(robust_std(finite_values(observed))),
            "model_robust_std": float(robust_std(finite_values(model_ts))),
            "recommendation": recommendation,
        }
        row.update(parse_mcmc_summary(run_dir))
        run_rows.append(row)

        if not args.skip_controls:
            base_params = dict(posterior)
            kp_grid, drv_grid = diag_module.default_kp_drv_grids(
                context,
                num_kp=args.num_kp,
                num_drv=args.num_drv,
                drv_bounds=(args.drv_min, args.drv_max),
            )
            rng = np.random.default_rng(args.seed)
            control_dir = component_dir / "kp_drv_controls"
            control_dir.mkdir(parents=True, exist_ok=True)
            for control in parse_csv_tuple(args.controls):
                data_override = control_data(control, observed, rng)
                scan = diag_module.scan_kp_drv_surface(
                    context,
                    component_name=component_name,
                    base_params=base_params,
                    kp_grid=kp_grid,
                    drv_grid=drv_grid,
                    data_override=data_override,
                    include_log_prior=False,
                )
                scan["control"] = control
                npz_path = control_dir / f"{control}_surface.npz"
                np.savez_compressed(
                    npz_path,
                    kp_grid=np.asarray(scan["kp_grid"]),
                    drv_grid=np.asarray(scan["drv_grid"]),
                    log_likelihood=np.asarray(scan["log_likelihood"]),
                    surface=np.asarray(scan["surface"]),
                )
                data_paths.append(str(npz_path))
                print(f"wrote {npz_path}")
                fig_path = control_dir / f"{control}_surface.png"
                save_surface_plot(fig_path, scan)
                figure_paths.append(str(fig_path))
                control_row = summarize_scan(control, scan, component_name)
                control_row.update({"run_label": run_label, "run_dir": str(run_dir)})
                control_rows.append(control_row)

    return run_rows, alpha_rows, control_rows, figure_paths, data_paths


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    run_rows: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    duplicate_runs: dict[str, list[str]],
    figure_paths: list[str],
    data_paths: list[str],
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.6g}"
        if value is None:
            return ""
        return str(value)

    lines: list[str] = []
    lines.append("# Retrieval-Space vs Residual-Spectrum Deep Dive")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Target: {args.planet} {args.mode} epoch={args.epoch}")
    lines.append("")
    lines.append("## High-confidence code-path findings")
    lines.append("")
    lines.append("- The matched-filter likelihood fits a free per-exposure `alpha`, so the likelihood is invariant to flipping both data/model sign in the residual comparison. The sign convention must be diagnosed through `alpha`, not by a plain 1D overlay.")
    lines.append("- Current transmission spectrum plots are ambiguous for time-series runs: the median posterior model is a 2D exposure x wavelength array, but `plot_transmission_spectrum` treats axis 0 as posterior samples and labels exposure variation as model scatter.")
    lines.append("- Transmission prep saves `data.npy` after SYSREM and after the default Doppler-shadow correction in `prepare_retrieval_timeseries.py`; saved `pre_sysrem_*` and `U_sysrem.npz` come from the earlier `get_pepsi_data` return path and are not the same shadow-corrected stage.")
    lines.append("- When model reconstruction is available, the diagnostic figures generated by this script label each plotted array by its retrieval space: post-SYSREM observed residuals, observed-frame processed model, matched-filter residuals, and planet-rest alpha-scaled stacks.")
    lines.append("")

    if duplicate_runs:
        lines.append("## Duplicate Label Resolution")
        lines.append("")
        for label, paths in duplicate_runs.items():
            lines.append(f"- `{label}` matched {len(paths)} runs; selected newest by mtime: `{paths[0]}`.")
        lines.append("")

    lines.append("## Prepared Bundle Summary")
    lines.append("")
    if bundle_rows:
        lines.append("| arm | data shape | run_sysrem | shadow | pre_SYSREM | U_SYSREM | post z robust std | post-pre corr |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in bundle_rows:
            lines.append(
                "| {arm} | {shape} | {run_sysrem} | {shadow} | {pre} | {u} | {z} | {corr} |".format(
                    arm=fmt(row.get("arm")),
                    shape=fmt(row.get("data_shape")),
                    run_sysrem=fmt(row.get("run_sysrem")),
                    shadow=fmt(row.get("doppler_shadow_applied")),
                    pre=fmt(row.get("pre_sysrem_present")),
                    u=fmt(row.get("has_U_sysrem")),
                    z=fmt(row.get("post_z_robust_std")),
                    corr=fmt(row.get("post_pre_weighted_corr")),
                )
            )
    else:
        lines.append("No prepared bundle rows were written.")
    lines.append("")

    lines.append("## Run Sign/Geometry Summary")
    lines.append("")
    if run_rows:
        lines.append("| run | component | sigma | div | alpha med | alpha neg frac | corr med | logL obs-neg | recommendation |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in run_rows:
            lines.append(
                "| {run} | {component} | {sigma} | {div} | {alpha} | {neg} | {corr} | {dlogl} | {rec} |".format(
                    run=fmt(row.get("run_label")),
                    component=fmt(row.get("component")),
                    sigma=fmt(row.get("sigma_scale")),
                    div=fmt(row.get("divergences")),
                    alpha=fmt(row.get("alpha_median")),
                    neg=fmt(row.get("alpha_negative_fraction")),
                    corr=fmt(row.get("weighted_corr_median")),
                    dlogl=fmt(row.get("logL_observed_minus_negative_data")),
                    rec=fmt(row.get("recommendation")),
                )
            )
    else:
        lines.append("No model-space run rows were written.")
    lines.append("")

    lines.append("## Kp-dRV Controls")
    lines.append("")
    if control_rows:
        lines.append("| run | component | control | best Kp | best dRV | best-median |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in control_rows:
            lines.append(
                "| {run} | {component} | {control} | {kp} | {drv} | {score} |".format(
                    run=fmt(row.get("run_label")),
                    component=fmt(row.get("component")),
                    control=fmt(row.get("control")),
                    kp=fmt(row.get("best_kp")),
                    drv=fmt(row.get("best_drv")),
                    score=fmt(row.get("best_minus_median")),
                )
            )
    else:
        lines.append("Controls were skipped or failed before surface outputs were written.")
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    for fig in figure_paths:
        lines.append(f"- Figure: `{fig}`")
    for data_path in data_paths:
        lines.append(f"- Data: `{data_path}`")
    if not figure_paths and not data_paths:
        lines.append("- No model-space output files were produced.")
    lines.append("")

    if errors:
        lines.append("## Errors")
        lines.append("")
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")

    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- A visually mismatched plain model overlay is not enough to diagnose a likelihood failure in matched-filter mode; inspect `alpha*model` and `observed - alpha*model`.")
    lines.append("- If `logL_observed_minus_negative_data` is approximately zero, that confirms the sign degeneracy of the current matched-filter objective.")
    lines.append("- If `alpha` is mostly negative, the physical transmission residual sign is opposite the positive model convention even when the sampler can still optimize the likelihood.")
    lines.append("- If Kp-dRV controls show phase-shuffle or wavelength-roll surfaces comparable to observed, then the issue is real likelihood geometry/data covariance rather than only plotting semantics.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Explicit run directories. If omitted, labels are resolved under --run-root.")
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default="Duck24")
    parser.add_argument("--mode", choices=("transmission", "emission"), default="transmission")
    parser.add_argument("--epoch", default="20190504")
    parser.add_argument("--arms", default="red,blue")
    parser.add_argument("--run-root", type=Path, default=REPO_ROOT / "output" / "kelt20b" / "Duck24" / "transmission")
    parser.add_argument("--run-labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--components", default="", help="Comma-separated component names. Defaults to all components in each run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "diagnostics" / "retrieval_space_deep_dive_kelt20b_20190504",
    )
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only write bundle/run artifact summaries; do not import JAX diagnostics or rebuild models.",
    )
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--num-kp", type=int, default=9)
    parser.add_argument("--num-drv", type=int, default=13)
    parser.add_argument("--drv-min", type=float, default=-30.0)
    parser.add_argument("--drv-max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--wavelength-stride", type=int, default=16)
    parser.add_argument("--rest-stack-max-points", type=int, default=6000)
    parser.add_argument(
        "--apply-sysrem",
        choices=("auto", "true", "false"),
        default="auto",
        help="Override SYSREM use in diagnostic context. Default reads the run config/current defaults.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.apply_sysrem == "auto":
        args.apply_sysrem = None
    else:
        args.apply_sysrem = args.apply_sysrem == "true"

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs, duplicate_runs = discover_run_dirs(args)
    bundle_rows = [
        summarize_bundle(args.mode, args.planet, args.epoch, arm)
        for arm in parse_csv_tuple(args.arms)
    ]
    write_csv(out_dir / "prepared_bundle_summary.csv", bundle_rows)

    run_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    data_paths: list[str] = []
    errors: list[str] = []
    manifest: dict[str, Any] = {
        "planet": args.planet,
        "ephemeris": args.ephemeris,
        "mode": args.mode,
        "epoch": args.epoch,
        "arms": list(parse_csv_tuple(args.arms)),
        "run_dirs": [str(path) for path in run_dirs],
        "duplicate_runs": duplicate_runs,
        "output_dir": str(out_dir),
        "skip_controls": bool(args.skip_controls),
    }

    if args.skip_model:
        for run_dir in run_dirs:
            try:
                run_rows.append(summarize_run_artifact(run_dir))
            except Exception as exc:
                message = f"{run_dir}: {type(exc).__name__}: {exc}"
                errors.append(message)
                print(f"error: {message}", file=sys.stderr)
    else:
        runtime_error = preflight_model_runtime()
        if runtime_error is not None:
            message = f"model runtime unavailable: {runtime_error}"
            errors.append(message)
            print(f"error: {message}", file=sys.stderr)
            for run_dir in run_dirs:
                try:
                    row = summarize_run_artifact(run_dir)
                    row["model_runtime_error"] = runtime_error
                    row["recommendation"] = "model reconstruction blocked by local runtime; fix JAX env or rerun with --skip-model"
                    run_rows.append(row)
                except Exception as exc:
                    run_message = f"{run_dir}: {type(exc).__name__}: {exc}"
                    errors.append(run_message)
                    print(f"error: {run_message}", file=sys.stderr)
        else:
            for run_dir in run_dirs:
                try:
                    rows, alphas, controls, figs, data = analyze_run(
                        run_dir,
                        args=args,
                        out_dir=out_dir,
                    )
                    run_rows.extend(rows)
                    alpha_rows.extend(alphas)
                    control_rows.extend(controls)
                    figure_paths.extend(figs)
                    data_paths.extend(data)
                except Exception as exc:
                    message = f"{run_dir}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    print(f"error: {message}", file=sys.stderr)

    write_csv(out_dir / "run_sign_geometry_summary.csv", run_rows)
    write_csv(out_dir / "alpha_by_exposure.csv", alpha_rows)
    write_csv(out_dir / "kp_drv_control_summary.csv", control_rows)
    manifest["errors"] = errors
    manifest["figures"] = figure_paths
    manifest["data_products"] = data_paths
    write_json(out_dir / "manifest.json", manifest)
    write_report(
        out_dir / "report.md",
        args=args,
        run_rows=run_rows,
        bundle_rows=bundle_rows,
        control_rows=control_rows,
        duplicate_runs=duplicate_runs,
        figure_paths=figure_paths,
        data_paths=data_paths,
        errors=errors,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
