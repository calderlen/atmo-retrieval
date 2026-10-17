#!/usr/bin/env python3
"""Check numerical results in an isolated HRS regeneration tree."""

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
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    data = _array(path / "data.npy")
    sigma = _array(path / "sigma.npy")
    if not np.all(np.isfinite(data)):
        _record_problem(
            errors,
            dataset=dataset,
            product=path.name,
            detail="saved data contains nonfinite values",
        )
    if not np.all(np.isfinite(sigma)) or not np.all(sigma > 0):
        _record_problem(
            errors,
            dataset=dataset,
            product=path.name,
            detail="saved sigma contains nonfinite or nonpositive values",
        )
    return {
        "path": _relative(path, root),
        "n_rows": int(data.shape[0]),
        "n_wavelengths": int(data.shape[1]),
    }


def _audit_collapsed_products(
    dataset_root: Path,
    *,
    root: Path,
    dataset: str,
    mode: str,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selections = (
        ("full_transit",)
        if mode == "transmission"
        else ("full_emission", "pre_eclipse", "post_eclipse")
    )
    rows: list[dict[str, Any]] = []
    for selection in selections:
        path = dataset_root / "collapsed" / selection
        if mode == "transmission":
            status = "ready"
        else:
            metadata = _load_json(path / "collapse_metadata.json")
            status = metadata["status"]
        row = {"path": _relative(path, root), "selection": selection, "status": status}
        if status == "skipped":
            row["skip_reason"] = metadata["skip_reason"]
        else:
            suffix = "transmission" if mode == "transmission" else "emission"
            spectrum = _array(path / f"spectrum_{suffix}.npy")
            uncertainty = _array(path / f"uncertainty_{suffix}.npy")
            if (
                not np.all(np.isfinite(spectrum))
                or not np.all(np.isfinite(uncertainty))
                or not np.all(uncertainty > 0)
            ):
                _record_problem(
                    errors,
                    dataset=dataset,
                    product=f"collapsed/{selection}",
                    detail="collapsed spectrum or uncertainty has invalid numerical values",
                )
            row["n_wavelengths"] = int(spectrum.size)
        rows.append(row)
    return rows


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_json(root / MANIFEST_FILENAME)
    product_scope = manifest["product_scope"]
    product_names = (
        ("timeseries",)
        if product_scope == "timeseries_only"
        else ("timeseries", "collapse_source")
    )
    errors: list[dict[str, str]] = []
    dataset_rows: list[dict[str, Any]] = []
    collapse_statuses: Counter[str] = Counter()

    for manifest_row in manifest["datasets"]:
        if manifest_row["status"] != "complete":
            continue
        mode = manifest_row["mode"]
        dataset_root = Path(manifest_row["dataset_root"])
        dataset = "/".join(
            str(manifest_row[key]) for key in ("mode", "planet_slug", "epoch", "arm")
        )
        products = {
            product: _audit_timeseries_product(
                dataset_root / product,
                root=root,
                dataset=dataset,
                errors=errors,
            )
            for product in product_names
        }
        collapsed = (
            []
            if product_scope == "timeseries_only"
            else _audit_collapsed_products(
                dataset_root,
                root=root,
                dataset=dataset,
                mode=mode,
                errors=errors,
            )
        )
        collapse_statuses.update(row["status"] for row in collapsed)
        dataset_rows.append(
            {"dataset": dataset, "products": products, "collapsed": collapsed}
        )

    status_counts = Counter(row["status"] for row in manifest["datasets"])
    return {
        "schema_version": 1,
        "kind": "isolated_hrs_regeneration_audit",
        "audit_scope": "numerical_results",
        "output_root": str(root),
        "product_scope": product_scope,
        "manifest_status": manifest["status"],
        "passed": not errors,
        "dataset_status_counts": dict(sorted(status_counts.items())),
        "collapse_status_counts": dict(sorted(collapse_statuses.items())),
        "n_errors": len(errors),
        "errors": errors,
        "datasets": dataset_rows,
    }


def main() -> int:
    args = parse_args()
    report = audit(args.output_root)
    summary = {
        key: report[key]
        for key in (
            "audit_scope",
            "passed",
            "manifest_status",
            "dataset_status_counts",
            "collapse_status_counts",
            "n_errors",
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
