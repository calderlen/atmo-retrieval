#!/usr/bin/env python
"""Measure saved SYSREM model-space attenuation for narrow moving lines.

This is a diagnostic for prepared time-series bundles. It does not rerun prep;
it reads ``wavelength.npy``, ``phase.npy``, ``sigma.npy``, and ``U_sysrem.npz``,
builds a synthetic planet-frame Gaussian line, applies the saved chunked SYSREM
projection to that model line, and reports the matched-filter amplitude that
survives.
"""

from __future__ import annotations

import argparse
import csv
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


C_KMS = 299792.458

DEFAULT_EPOCHS = ("20210501", "20210518", "20230430", "20230615", "20240516")
DEFAULT_ARMS = ("blue", "red")
DEFAULT_LINE_LIST = (
    {"label": "H beta", "species": "H I", "rest_A": 4861.333},
    {"label": "Mg I b1", "species": "Mg I", "rest_A": 5167.321},
    {"label": "Mg I b2", "species": "Mg I", "rest_A": 5172.684},
    {"label": "Mg I b3", "species": "Mg I", "rest_A": 5183.604},
    {"label": "Fe II 5018", "species": "Fe II", "rest_A": 5018.440},
    {"label": "Fe I 5270", "species": "Fe I", "rest_A": 5270.356},
    {"label": "Fe I 5328", "species": "Fe I", "rest_A": 5328.039},
    {"label": "Fe I 6335", "species": "Fe I", "rest_A": 6335.337},
    {"label": "Fe II 6516", "species": "Fe II", "rest_A": 6516.080},
    {"label": "H alpha", "species": "H I", "rest_A": 6562.790},
    {"label": "Ca I 6573", "species": "Ca I", "rest_A": 6572.779},
)


def _planet_slug(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace(" ", "")


def resolve_planet_name(value: str) -> str:
    requested = _planet_slug(value)
    for planet_name in config.PLANETS:
        if _planet_slug(planet_name) == requested:
            return planet_name
    raise KeyError(f"Could not match planet={value!r} to config.PLANETS keys: {list(config.PLANETS)}")


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_line_spec(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) == 1:
        rest_A = float(parts[0])
        return {"label": f"{rest_A:.3f} A", "species": "", "rest_A": rest_A}
    if len(parts) == 2:
        label, rest_A = parts
        return {"label": label, "species": "", "rest_A": float(rest_A)}
    label, species, rest_A = parts[:3]
    return {"label": label, "species": species, "rest_A": float(rest_A)}


def line_display_name(line: dict[str, Any]) -> str:
    label = str(line.get("label", "")).strip()
    species = str(line.get("species", "")).strip()
    if species and species.lower() not in label.lower():
        return f"{label} {species}".strip()
    return label or species or f"{float(line['rest_A']):.3f} A"


def planet_rv_kms(phase: np.ndarray, kp_kms: float, vsys_kms: float, drv_kms: float) -> np.ndarray:
    return kp_kms * np.sin(2.0 * np.pi * np.asarray(phase, dtype=float)) + vsys_kms + drv_kms


def moving_gaussian_line(
    wave: np.ndarray,
    phase: np.ndarray,
    *,
    rest_A: float,
    kp_kms: float,
    vsys_kms: float,
    drv_kms: float,
    fwhm_kms: float,
    half_width_kms: float,
    amplitude: float,
) -> tuple[np.ndarray, float]:
    rv = planet_rv_kms(phase, kp_kms, vsys_kms, drv_kms)
    centers = float(rest_A) * (1.0 + rv / C_KMS)
    velocity = C_KMS * (wave[np.newaxis, :] / centers[:, np.newaxis] - 1.0)
    sigma_v = float(fwhm_kms) / 2.354820045
    model = float(amplitude) * np.exp(-0.5 * np.square(velocity / sigma_v))
    model[np.abs(velocity) > float(half_width_kms)] = 0.0

    margin_A = float(rest_A) * float(half_width_kms) / C_KMS
    covered = (centers >= float(np.nanmin(wave)) + margin_A) & (
        centers <= float(np.nanmax(wave)) - margin_A
    )
    return model, float(np.mean(covered))


def matched_filter_scale(observed: np.ndarray, template: np.ndarray, sigma: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    template = np.asarray(template, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(template) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return math.nan
    weight = np.zeros_like(sigma, dtype=float)
    weight[mask] = 1.0 / np.square(np.clip(sigma[mask], config.F32_FLOOR_RECIP, None))
    denominator = float(np.sum(weight * np.square(template)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return math.nan
    numerator = float(np.sum(weight * observed * template))
    return numerator / denominator


def weighted_power_ratio(observed: np.ndarray, template: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(template) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return math.nan
    weight = np.zeros_like(sigma, dtype=float)
    weight[mask] = 1.0 / np.square(np.clip(sigma[mask], config.F32_FLOOR_RECIP, None))
    denominator = float(np.sum(weight * np.square(template)))
    numerator = float(np.sum(weight * np.square(observed)))
    if denominator <= 0.0 or numerator < 0.0:
        return math.nan
    return math.sqrt(numerator / denominator)


def chunk_indices_from_labels(chunk_labels: np.ndarray) -> tuple[np.ndarray, ...]:
    labels = sorted(int(label) for label in np.unique(chunk_labels) if int(label) >= 0)
    return tuple(np.where(chunk_labels == label)[0].astype(int) for label in labels)


def apply_saved_chunked_sysrem(model: np.ndarray, sysrem: dict[str, np.ndarray]) -> np.ndarray:
    corrected = np.asarray(model, dtype=float).copy()
    chunk_labels = np.asarray(sysrem["chunk_labels"], dtype=int)
    U_full = np.asarray(sysrem["U_sysrem"], dtype=float)
    if U_full.ndim == 2:
        U_full = U_full[:, :, np.newaxis]
    V_chunk_diag = np.asarray(sysrem["V_chunk_diag"], dtype=float)
    if V_chunk_diag.ndim == 1:
        V_chunk_diag = V_chunk_diag[np.newaxis, :]

    for chunk_id, indices in enumerate(chunk_indices_from_labels(chunk_labels)):
        if indices.size == 0:
            continue
        U_chunk = np.asarray(U_full[:, :, chunk_id], dtype=float)
        keep = np.any(np.isfinite(U_chunk), axis=0)
        U_chunk = np.nan_to_num(U_chunk[:, keep], nan=0.0)
        if U_chunk.shape[1] == 0:
            continue

        model_chunk = np.asarray(model[:, indices], dtype=float)
        v_diag = np.asarray(V_chunk_diag[chunk_id], dtype=float)
        weighted_basis = v_diag[:, np.newaxis] * U_chunk
        weighted_model = v_diag[:, np.newaxis] * model_chunk
        gram = weighted_basis.T @ weighted_basis
        rhs = weighted_basis.T @ weighted_model
        try:
            coeffs = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(gram) @ rhs
        corrected[:, indices] = model_chunk - U_chunk @ coeffs

    return corrected


def load_bundle(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    wave = np.load(bundle_dir / "wavelength.npy")
    phase = np.load(bundle_dir / "phase.npy")
    sigma = np.load(bundle_dir / "sigma.npy")
    with np.load(bundle_dir / "U_sysrem.npz") as raw:
        sysrem = {name: np.asarray(raw[name]) for name in raw.files}
    return wave, phase, sigma, sysrem


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), str(row["line"])), []).append(row)

    summary = []
    for (arm, line), values in sorted(grouped.items()):
        survival = np.asarray([row["survival_scale"] for row in values], dtype=float)
        attenuation = np.asarray([row["attenuation_frac"] for row in values], dtype=float)
        summary.append(
            {
                "arm": arm,
                "line": line,
                "n_records": len(values),
                "survival_median": float(np.nanmedian(survival)),
                "survival_min": float(np.nanmin(survival)),
                "survival_max": float(np.nanmax(survival)),
                "attenuation_median": float(np.nanmedian(attenuation)),
            }
        )
    return summary


def format_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def print_table(rows: list[dict[str, Any]], *, title: str) -> None:
    if not rows:
        print(f"{title}: no rows")
        return
    print(f"\n{title}")
    print("-" * len(title))
    header = (
        "epoch arm line              rest_A  counts    stop_tol coverage survival attenuation power"
    )
    print(header)
    for row in rows:
        print(
            f"{row['epoch']:8s} {row['arm']:4s} {row['line'][:17]:17s} "
            f"{row['rest_A']:7.1f} {str(row['basis_counts']):9s} "
            f"{row['saved_stop_tol']:.1e} {format_float(row['coverage_frac']):>8s} "
            f"{format_float(row['survival_scale']):>8s} "
            f"{format_float(row['attenuation_frac']):>11s} "
            f"{format_float(row['power_ratio']):>5s}"
        )


def print_summary(rows: list[dict[str, Any]]) -> None:
    summary = summarize_rows(rows)
    if not summary:
        return
    print("\nBy-line summary")
    print("---------------")
    print("arm  line              n survival_med survival_min survival_max attenuation_med")
    for row in summary:
        print(
            f"{row['arm']:4s} {row['line'][:17]:17s} {row['n_records']:1d} "
            f"{format_float(row['survival_median']):>12s} "
            f"{format_float(row['survival_min']):>12s} "
            f"{format_float(row['survival_max']):>12s} "
            f"{format_float(row['attenuation_median']):>15s}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default=config.EPHEMERIS)
    parser.add_argument("--mode", choices=("emission", "transmission"), default="emission")
    parser.add_argument("--epochs", default=",".join(DEFAULT_EPOCHS))
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--line", action="append", default=None, help="Line as rest_A, label:rest_A, or label:species:rest_A")
    parser.add_argument("--fwhm-kms", type=float, default=6.0)
    parser.add_argument("--half-width-kms", type=float, default=30.0)
    parser.add_argument("--amplitude", type=float, default=1.0e-3)
    parser.add_argument("--kp-kms", type=float, default=None)
    parser.add_argument("--vsys-kms", type=float, default=None)
    parser.add_argument("--drv-kms", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--csv", type=Path, default=None)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    planet_name = resolve_planet_name(args.planet)
    planet_cfg = config_utils.get_params(planet_name, args.ephemeris)
    kp_kms = float(planet_cfg.get("Kp") if args.kp_kms is None else args.kp_kms)
    vsys_kms = float(planet_cfg.get("RV_abs", 0.0) if args.vsys_kms is None else args.vsys_kms)
    lines = [parse_line_spec(value) for value in args.line] if args.line else list(DEFAULT_LINE_LIST)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for epoch in parse_csv_tuple(args.epochs):
        for arm in parse_csv_tuple(args.arms):
            bundle_dir = config_utils.get_data_dir(
                planet=planet_name,
                epoch=epoch,
                arm=arm,
                mode=args.mode,
            )
            try:
                wave, phase, sigma, sysrem = load_bundle(bundle_dir)
            except FileNotFoundError as exc:
                skipped.append(f"{epoch} {arm}: missing {exc.filename or exc}")
                continue

            chunk_names = tuple(str(name) for name in sysrem.get("chunk_names", ()))
            basis_counts = tuple(int(value) for value in np.asarray(sysrem["basis_counts"]).ravel())
            saved_stop_tol = float(np.asarray(sysrem.get("sysrem_stop_delta_stddev", np.nan)))

            for line in lines:
                rest_A = float(line["rest_A"])
                model, coverage = moving_gaussian_line(
                    wave,
                    phase,
                    rest_A=rest_A,
                    kp_kms=kp_kms,
                    vsys_kms=vsys_kms,
                    drv_kms=float(args.drv_kms),
                    fwhm_kms=float(args.fwhm_kms),
                    half_width_kms=float(args.half_width_kms),
                    amplitude=float(args.amplitude),
                )
                if coverage < float(args.min_coverage) or not np.any(model):
                    continue

                filtered = apply_saved_chunked_sysrem(model, sysrem)
                survival = matched_filter_scale(filtered, model, sigma)
                power = weighted_power_ratio(filtered, model, sigma)
                rows.append(
                    {
                        "epoch": epoch,
                        "arm": arm,
                        "line": line_display_name(line),
                        "species": str(line.get("species", "")),
                        "rest_A": rest_A,
                        "basis_counts": basis_counts,
                        "chunk_names": chunk_names,
                        "saved_stop_tol": saved_stop_tol,
                        "coverage_frac": coverage,
                        "survival_scale": survival,
                        "attenuation_frac": 1.0 - survival if np.isfinite(survival) else math.nan,
                        "power_ratio": power,
                        "fwhm_kms": float(args.fwhm_kms),
                        "half_width_kms": float(args.half_width_kms),
                        "kp_kms": kp_kms,
                        "vsys_kms": vsys_kms,
                        "drv_kms": float(args.drv_kms),
                        "bundle_dir": str(bundle_dir),
                    }
                )

    print(
        f"SYSREM line attenuation diagnostic: planet={planet_name}, ephemeris={args.ephemeris}, "
        f"Kp={kp_kms:g} km/s, Vsys={vsys_kms:g} km/s, FWHM={args.fwhm_kms:g} km/s"
    )
    if skipped:
        print("\nSkipped bundles")
        print("---------------")
        for item in skipped:
            print(item)
    print_table(rows, title="Per-bundle attenuation")
    print_summary(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
