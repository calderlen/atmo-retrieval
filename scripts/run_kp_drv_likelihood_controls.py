#!/usr/bin/env python3
"""Run Kp-dRV likelihood-surface controls for a saved retrieval setup."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


DEFAULT_CONTROLS = ("observed", "phase_shuffle", "sign_shuffle", "negative", "wavelength_roll")


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def load_npz_medians(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    with np.load(path) as payload:
        for name in payload.files:
            arr = np.asarray(payload[name])
            if arr.dtype == object:
                continue
            if arr.shape == ():
                values[name] = arr.item()
                continue
            flat = np.asarray(arr, dtype=float).reshape(-1)
            flat = flat[np.isfinite(flat)]
            if flat.size:
                values[name] = float(np.nanmedian(flat))
    return values


def load_base_params(run_dir: Path, source: str, diag_module: Any) -> dict[str, Any]:
    source = str(source).lower().strip()
    if source == "default":
        return {}
    if source in {"auto", "posterior"}:
        posterior = load_npz_medians(run_dir / "posterior_sample.npz")
        if posterior or source == "posterior":
            return posterior
    if source in {"auto", "init"}:
        init_path = run_dir / "svi_init_values.npz"
        if init_path.exists():
            return diag_module.load_named_init_values(run_dir)
    if source not in {"auto", "posterior", "init", "default"}:
        raise ValueError(f"Unknown base source: {source}")
    return {}


def control_data(
    control: str,
    observed: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray | None:
    control = str(control).lower().strip()
    if control == "observed":
        return None
    if control == "phase_shuffle":
        return observed[rng.permutation(observed.shape[0])]
    if control == "sign_shuffle":
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=observed.shape[0])
        return observed * signs[:, None]
    if control == "negative":
        return -observed
    if control == "wavelength_roll":
        return np.roll(observed, shift=max(1, observed.shape[1] // 3), axis=1)
    raise ValueError(f"Unknown control: {control}")


def save_scan_npz(path: Path, scan: dict[str, Any]) -> None:
    arrays: dict[str, Any] = {}
    for key in ("kp_grid", "drv_grid", "log_likelihood", "surface", "log_prior"):
        value = scan.get(key)
        if value is not None:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def surface_summary(control: str, scan: dict[str, Any]) -> dict[str, Any]:
    surface = np.asarray(scan.get("surface", scan["log_likelihood"]), dtype=float)
    finite = surface[np.isfinite(surface)]
    if finite.size:
        median = float(np.nanmedian(finite))
        p95 = float(np.nanpercentile(finite, 95.0))
        best = float(scan["best_surface_value"])
    else:
        median = np.nan
        p95 = np.nan
        best = np.nan
    best_params = scan["best_params"]
    raw_params = scan["best_log_likelihood_params"]
    return {
        "control": control,
        "component": scan["component_name"],
        "best_kp": best_params["Kp"],
        "best_drv": best_params["dRV"],
        "best_raw_kp": raw_params["Kp"],
        "best_raw_drv": raw_params["dRV"],
        "best_surface": best,
        "surface_median": median,
        "surface_p95": p95,
        "best_minus_median": best - median if np.isfinite(best) and np.isfinite(median) else np.nan,
        "best_minus_p95": best - p95 if np.isfinite(best) and np.isfinite(p95) else np.nan,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_scan(path: Path, scan: dict[str, Any], title: str, diag_module: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    diag_module.plot_kp_drv_surface(scan, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def parse_scale(value: str, run_dir: Path, diag_module: Any) -> float:
    if str(value).lower().strip() != "auto":
        return float(value)
    saved = diag_module.load_saved_run_config(run_dir)
    raw = saved.get("Spectroscopic sigma scale", "1.0")
    return float(str(raw).replace(",", ""))


def parse_stride(value: str, run_dir: Path, key: str, default: int, diag_module: Any) -> int:
    if str(value).lower().strip() != "auto":
        return int(value)
    saved = diag_module.load_saved_run_config(run_dir)
    return int(str(saved.get(key, default)).replace(",", ""))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Saved retrieval run directory with run_config.log.")
    parser.add_argument("--component", default=None, help="Spectroscopic component name; defaults to first component.")
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--base-source", choices=("auto", "posterior", "init", "default"), default="auto")
    parser.add_argument("--include-log-prior", action="store_true")
    parser.add_argument("--num-kp", type=int, default=15)
    parser.add_argument("--num-drv", type=int, default=21)
    parser.add_argument("--drv-min", type=float, default=-30.0)
    parser.add_argument("--drv-max", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--sigma-scale",
        default="auto",
        help="Sigma scale for context; use 'auto' to read run_config.log.",
    )
    parser.add_argument(
        "--spectral-stride",
        default="auto",
        help="Spectral stride for context; use 'auto' to read run_config.log.",
    )
    parser.add_argument(
        "--spectral-offset",
        default="auto",
        help="Spectral offset for context; use 'auto' to read run_config.log.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    from pipeline import diagnostics as diag_module

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.output_dir or (run_dir / "likelihood_controls")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = diag_module.build_diag_config_from_run_dir(run_dir)
    config_payload["sigma_scale"] = parse_scale(args.sigma_scale, run_dir, diag_module)
    config_payload["spectral_stride"] = parse_stride(
        args.spectral_stride,
        run_dir,
        "Spectral stride",
        1,
        diag_module,
    )
    config_payload["spectral_offset"] = parse_stride(
        args.spectral_offset,
        run_dir,
        "Spectral offset",
        0,
        diag_module,
    )
    context = diag_module.build_diagnostic_context(**config_payload)
    component_name = args.component or context.spectroscopic_component_names[0]
    component = context.spectroscopic_components[component_name]
    observed = np.asarray(component.data, dtype=float)

    base_params = load_base_params(run_dir, args.base_source, diag_module)
    kp_grid, drv_grid = diag_module.default_kp_drv_grids(
        context,
        num_kp=args.num_kp,
        num_drv=args.num_drv,
        drv_bounds=(args.drv_min, args.drv_max),
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for control in parse_csv_tuple(args.controls):
        data_override = control_data(control, observed, rng=rng)
        scan = diag_module.scan_kp_drv_surface(
            context,
            component_name=component_name,
            base_params=base_params,
            kp_grid=kp_grid,
            drv_grid=drv_grid,
            data_override=data_override,
            include_log_prior=args.include_log_prior,
        )
        scan["component_name"] = component_name
        save_scan_npz(out_dir / f"{control}_surface.npz", scan)
        plot_scan(
            out_dir / f"{control}_surface.png",
            scan,
            f"{component_name}: {control}",
            diag_module,
        )
        rows.append(surface_summary(control, scan))

    write_summary_csv(out_dir / "surface_summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
