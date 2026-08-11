#!/usr/bin/env python3
"""Audit an isolated raw-FITS-to-collapsed HRS regeneration tree."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.lsd_doppler_shadow import (
    FIXED_LSD_SHADOW_METHOD,
    FIXED_LSD_SHADOW_SCHEMA_VERSION,
)


MANIFEST_FILENAME = "regeneration_manifest.json"
AUDIT_FILENAME = "regeneration_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write the complete audit to {AUDIT_FILENAME} in the output root.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def _record_problem(
    rows: list[dict[str, str]], *, dataset: str, product: str, detail: str
) -> None:
    rows.append({"dataset": dataset, "product": product, "detail": detail})


def _audit_timeseries_product(
    path: Path,
    *,
    root: Path,
    dataset: str,
    expected_frame: str,
    expected_trim: tuple[float, float],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any] | None:
    product = path.name
    metadata_path = path / "timeseries_prep.json"
    if not metadata_path.is_file():
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="missing timeseries_prep.json",
        )
        return None
    metadata = _load_json(metadata_path)

    try:
        wave = _array(path / "wavelength.npy")
        data = _array(path / "data.npy")
        sigma = _array(path / "sigma.npy")
        phase = _array(path / "phase.npy")
    except FileNotFoundError as exc:
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=f"missing required array: {exc}",
        )
        return metadata

    if wave.ndim != 1 or not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="wavelength.npy is not a finite, strictly increasing 1D grid",
        )
    if data.ndim != 2 or sigma.shape != data.shape:
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=f"data/sigma shape mismatch: data={data.shape}, sigma={sigma.shape}",
        )
    elif data.shape != (phase.size, wave.size):
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=(
                f"array axes disagree: data={data.shape}, phase={phase.shape}, "
                f"wavelength={wave.shape}"
            ),
        )
    if not np.all(np.isfinite(data)):
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="saved data contains nonfinite values",
        )
    if not np.all(np.isfinite(sigma)) or not np.all(sigma > 0):
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="saved sigma contains nonfinite or nonpositive values",
        )

    contract = metadata.get("wavelength_frame_contract")
    if metadata.get("wavelength_medium") != "vacuum":
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=f"wavelength_medium={metadata.get('wavelength_medium')!r}",
        )
    if metadata.get("wavelength_frame") != expected_frame:
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=(
                f"wavelength_frame={metadata.get('wavelength_frame')!r}; "
                f"expected {expected_frame!r}"
            ),
        )
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="missing or unsupported wavelength_frame_contract",
        )
    else:
        per_exposure = contract.get("per_exposure", [])
        n_source = int(contract.get("n_source_exposures", -1))
        if n_source <= 0 or len(per_exposure) != n_source:
            _record_problem(
                errors,
                dataset=dataset,
                product=product,
                detail=(
                    f"frame-contract exposure count mismatch: n_source={n_source}, "
                    f"rows={len(per_exposure)}"
                ),
            )
        required = {
            "input_wavelength_medium",
            "barycentric_reconstruction_method",
            "pepsi_velocity_terms_removed_kms",
            "barycentric_reconstruction_operations",
        }
        for index, exposure in enumerate(per_exposure):
            missing = required - set(exposure)
            if missing:
                _record_problem(
                    errors,
                    dataset=dataset,
                    product=product,
                    detail=f"frame-contract exposure {index} missing {sorted(missing)}",
                )
                break
            if exposure.get("velocity_frame_parse_error"):
                _record_problem(
                    warnings,
                    dataset=dataset,
                    product=product,
                    detail=(
                        f"frame parse warning at exposure {index}: "
                        f"{exposure['velocity_frame_parse_error']}"
                    ),
                )
        stellar = contract.get("stellar_rest_correction", {})
        if expected_frame == "barycentric" and stellar.get("applied"):
            _record_problem(
                errors,
                dataset=dataset,
                product=product,
                detail="stellar-rest correction was applied to requested barycentric output",
            )

    operator_path = path / "timeseries_operator.npz"
    if not operator_path.is_file():
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail="missing timeseries_operator.npz",
        )
    else:
        with np.load(operator_path, allow_pickle=False) as operator:
            source_phase = np.asarray(operator["source_phase"])
            selected = np.asarray(operator["selected_exposure_indices"])
            if selected.size != data.shape[0]:
                _record_problem(
                    errors,
                    dataset=dataset,
                    product=product,
                    detail=(
                        f"operator selected rows={selected.size}; data rows={data.shape[0]}"
                    ),
                )
            if isinstance(contract, dict) and int(
                contract.get("n_source_exposures", -1)
            ) != source_phase.size:
                _record_problem(
                    errors,
                    dataset=dataset,
                    product=product,
                    detail=(
                        "operator source rows do not match wavelength-frame contract: "
                        f"{source_phase.size} versus "
                        f"{contract.get('n_source_exposures')}"
                    ),
                )

    trim = metadata.get("arm_edge_trim", {})
    actual_trim = (
        float(trim.get("left_trim_A", np.nan)),
        float(trim.get("right_trim_A", np.nan)),
    )
    if not np.allclose(actual_trim, expected_trim, rtol=0.0, atol=1.0e-9):
        _record_problem(
            errors,
            dataset=dataset,
            product=product,
            detail=f"edge trim={actual_trim}; expected={expected_trim}",
        )

    snr_path = path / "snr.npy"
    all_snr_missing = False
    if snr_path.is_file():
        snr = _array(snr_path)
        all_snr_missing = bool(not np.any(np.isfinite(snr)))
        if all_snr_missing:
            _record_problem(
                warnings,
                dataset=dataset,
                product=product,
                detail="all saved SNR header values are nonfinite",
            )

    return {
        "path": _relative(path, root),
        "n_rows": int(data.shape[0]),
        "n_wavelengths": int(wave.size),
        "all_snr_missing": all_snr_missing,
        "metadata": metadata,
    }


def _audit_collapsed_products(
    dataset_root: Path,
    *,
    root: Path,
    dataset: str,
    mode: str,
    source_contract: dict[str, Any] | None,
    expected_frame: str,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    expected = (
        ("full_transit",)
        if mode == "transmission"
        else ("full_emission", "pre_eclipse", "post_eclipse")
    )
    rows: list[dict[str, Any]] = []
    for selection in expected:
        path = dataset_root / "collapsed" / selection
        metadata_path = path / "collapse_metadata.json"
        if not metadata_path.is_file():
            _record_problem(
                errors,
                dataset=dataset,
                product=f"collapsed/{selection}",
                detail="missing collapse_metadata.json",
            )
            continue
        metadata = _load_json(metadata_path)
        status = str(metadata.get("status", "ready"))
        if metadata.get("wavelength_medium") != "vacuum":
            _record_problem(
                errors,
                dataset=dataset,
                product=f"collapsed/{selection}",
                detail=f"wavelength_medium={metadata.get('wavelength_medium')!r}",
            )
        if metadata.get("wavelength_frame") != expected_frame:
            _record_problem(
                errors,
                dataset=dataset,
                product=f"collapsed/{selection}",
                detail=f"wavelength_frame={metadata.get('wavelength_frame')!r}",
            )
        if source_contract is not None and metadata.get(
            "wavelength_frame_contract"
        ) != source_contract:
            _record_problem(
                errors,
                dataset=dataset,
                product=f"collapsed/{selection}",
                detail="collapsed frame contract does not exactly inherit collapse_source",
            )

        row = {
            "path": _relative(path, root),
            "selection": selection,
            "status": status,
            "skip_reason": metadata.get("skip_reason"),
        }
        if status != "skipped":
            suffix = "transmission" if mode == "transmission" else "emission"
            wave = _array(path / f"wavelength_{suffix}.npy")
            spectrum = _array(path / f"spectrum_{suffix}.npy")
            uncertainty = _array(path / f"uncertainty_{suffix}.npy")
            if (
                wave.ndim != 1
                or spectrum.shape != wave.shape
                or uncertainty.shape != wave.shape
                or not np.all(np.isfinite(wave))
                or not np.all(np.diff(wave) > 0)
                or not np.all(np.isfinite(spectrum))
                or not np.all(np.isfinite(uncertainty))
                or not np.all(uncertainty > 0)
            ):
                _record_problem(
                    errors,
                    dataset=dataset,
                    product=f"collapsed/{selection}",
                    detail="invalid collapsed wavelength/spectrum/uncertainty arrays",
                )
            row["n_wavelengths"] = int(wave.size)
        rows.append(row)
    return rows


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_json(root / MANIFEST_FILENAME)
    product_scope = str(manifest.get("product_scope", "full"))
    if product_scope not in {
        "full",  # Legacy manifest spelling.
        "timeseries_and_collapsed",
        "timeseries_only",
    }:
        raise ValueError(f"Unsupported regeneration product scope: {product_scope!r}")
    expected_frame = str(manifest.get("requested_wavelength_frame"))
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    dataset_rows: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    trim_statuses: Counter[str] = Counter()
    shadow_statuses: Counter[str] = Counter()
    collapse_statuses: Counter[str] = Counter()

    for manifest_row in manifest.get("datasets", []):
        if manifest_row.get("status") != "complete":
            continue
        mode = str(manifest_row["mode"])
        dataset_root = Path(manifest_row["dataset_root"])
        dataset = "/".join(
            (
                mode,
                str(manifest_row["planet_slug"]),
                str(manifest_row["epoch"]),
                str(manifest_row["arm"]),
            )
        )
        expected_trim = (
            float(manifest_row["edge_trim_left_A"]),
            float(manifest_row["edge_trim_right_A"]),
        )
        trim_statuses[str(manifest_row.get("edge_trim_status"))] += 1
        products: dict[str, Any] = {}
        product_names = (
            ("timeseries",)
            if product_scope == "timeseries_only"
            else ("timeseries", "collapse_source")
        )
        for product in product_names:
            products[product] = _audit_timeseries_product(
                dataset_root / product,
                root=root,
                dataset=dataset,
                expected_frame=expected_frame,
                expected_trim=expected_trim,
                errors=errors,
                warnings=warnings,
            )
            if products[product] is not None:
                metadata = products[product]["metadata"]
                contract = metadata.get("wavelength_frame_contract", {})
                methods.update(contract.get("barycentric_reconstruction_methods", []))
        if mode == "transmission":
            for product_name in product_names:
                product = products.get(product_name)
                metadata = {} if product is None else product["metadata"]
                shadow = metadata.get("fixed_doppler_shadow")
                current = (
                    isinstance(shadow, dict)
                    and bool(shadow.get("enabled", False))
                    and int(shadow.get("schema_version", -1))
                    == FIXED_LSD_SHADOW_SCHEMA_VERSION
                    and str(shadow.get("method")) == FIXED_LSD_SHADOW_METHOD
                )
                shadow_statuses["enabled_shared_basis_lsd" if current else "missing"] += 1
                if not current:
                    _record_problem(
                        errors,
                        dataset=dataset,
                        product=product_name,
                        detail="required shared-basis LSD Doppler shadow is absent or stale",
                    )
        source_contract = None
        if products.get("collapse_source") is not None:
            source_contract = products["collapse_source"]["metadata"].get(
                "wavelength_frame_contract"
            )
        collapsed = (
            []
            if product_scope == "timeseries_only"
            else _audit_collapsed_products(
                dataset_root,
                root=root,
                dataset=dataset,
                mode=mode,
                source_contract=source_contract,
                expected_frame=expected_frame,
                errors=errors,
            )
        )
        collapse_statuses.update(row["status"] for row in collapsed)
        dataset_rows.append(
            {
                "dataset": dataset,
                "products": {
                    name: None
                    if value is None
                    else {
                        key: item
                        for key, item in value.items()
                        if key != "metadata"
                    }
                    for name, value in products.items()
                },
                "collapsed": collapsed,
            }
        )

    status_counts = Counter(
        str(row.get("status")) for row in manifest.get("datasets", [])
    )
    structural_passed = not errors
    production_ready = (
        structural_passed
        and product_scope in {"full", "timeseries_and_collapsed"}
        and status_counts == {"complete": len(manifest.get("datasets", []))}
        and set(trim_statuses) <= {"accepted_post_sysrem"}
        and set(shadow_statuses) <= {"enabled_shared_basis_lsd"}
    )
    return {
        "schema_version": 1,
        "kind": "isolated_hrs_regeneration_audit",
        "output_root": str(root),
        "product_scope": product_scope,
        "manifest_status": manifest.get("status"),
        "requested_wavelength_frame": expected_frame,
        "wavelength_medium": manifest.get("wavelength_medium"),
        "passed": structural_passed,
        "production_ready": production_ready,
        "dataset_status_counts": dict(sorted(status_counts.items())),
        "edge_trim_status_counts": dict(sorted(trim_statuses.items())),
        "frame_reconstruction_method_counts": dict(sorted(methods.items())),
        "doppler_shadow_status_counts": dict(sorted(shadow_statuses.items())),
        "collapse_status_counts": dict(sorted(collapse_statuses.items())),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "datasets": dataset_rows,
    }


def main() -> int:
    args = parse_args()
    report = audit(args.output_root)
    summary = {
        key: report[key]
        for key in (
            "passed",
            "production_ready",
            "manifest_status",
            "dataset_status_counts",
            "edge_trim_status_counts",
            "frame_reconstruction_method_counts",
            "doppler_shadow_status_counts",
            "collapse_status_counts",
            "n_errors",
            "n_warnings",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.write:
        path = args.output_root.resolve() / AUDIT_FILENAME
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(path)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
