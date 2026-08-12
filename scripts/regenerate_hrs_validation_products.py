#!/usr/bin/env python
"""Regenerate the configured HRS product inventory into an isolated tree.

The raw ``input/hrs/<mode>/raw/<planet>/<epoch>`` folders define the inventory.
Every spectrum is rebuilt from the corresponding raw FITS folder. Every dataset
requires its newest adaptive schema-v4 calibration manifest to be accepted
post-SYSREM. No canonical prepared or collapsed arrays are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_utils
from dataio.collapse_emission_timeseries_to_1d import (
    EMISSION_COLLAPSE_SELECTIONS,
    collapse_epoch_arm as collapse_emission_epoch_arm,
)
from dataio.collapse_transmission_timeseries_to_1d import (
    collapse_transmission_epoch_arm,
)
from dataio.edge_trim_manifest import (
    EdgeTrimManifestError,
    load_accepted_edge_trim_manifest,
    normalize_dataset_key,
)
from dataio.hrs_diagnostic_products import discover_raw_calibration_inventory
from dataio.prepare_emission_retrieval_timeseries import (
    prepare_arm as prepare_emission_arm,
)
from dataio.prepare_retrieval_timeseries import (
    prepare_arm as prepare_transmission_arm,
)
from spectroscopy.doppler_shadow import (
    DopplerShadowFitConfig,
    fit_doppler_shadow,
)


CANONICAL_HRS_ROOT = PROJECT_ROOT / "input" / "hrs"
DEFAULT_EDGE_TRIM_ROOT = PROJECT_ROOT / "diagnostics" / "edge_trim_calibration"
MANIFEST_FILENAME = "regeneration_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _write_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_supported_inventory(
    *,
    modes: tuple[str, ...],
    planets: set[str],
) -> list[dict[str, str]]:
    """Discover every configured raw epoch/arm without reading prepared arrays."""

    records: dict[tuple[str, str, str, str], dict[str, str]] = {}
    raw_inventory = discover_raw_calibration_inventory(CANONICAL_HRS_ROOT)
    for (mode, raw_slug), entry in raw_inventory.items():
        if mode not in modes:
            continue
        planet = str(entry["planet_display"])
        planet_slug = normalize_dataset_key(planet)
        if planets and planet_slug not in planets:
            continue
        ephemeris = str(entry["ephemeris"])
        for epoch, arm in entry["datasets"]:
            raw_dir = config_utils.get_raw_hrs_dir(
                planet=planet,
                epoch=epoch,
                mode=mode,
            )
            if not raw_dir.is_dir():
                raise FileNotFoundError(
                    f"Inventory entry {mode}/{planet}/{epoch}/{arm} has no raw directory: {raw_dir}"
                )
            metadata_candidates = (
                CANONICAL_HRS_ROOT / mode / raw_slug / epoch / arm / "timeseries_prep.json",
                CANONICAL_HRS_ROOT / mode / raw_slug / epoch / arm / "timeseries" / "timeseries_prep.json",
            )
            metadata_path = next(
                (path for path in metadata_candidates if path.is_file()),
                None,
            )
            key = (mode, planet_slug, epoch, arm)
            records[key] = {
                "mode": mode,
                "planet": planet,
                "planet_slug": planet_slug,
                "ephemeris": ephemeris,
                "epoch": epoch,
                "arm": arm,
                "raw_dir": str(raw_dir),
                "inventory_metadata": str(metadata_path) if metadata_path else "",
                "inventory_source": "raw_directory_scan",
            }
    return [records[key] for key in sorted(records)]


def _edge_trim_for(
    record: dict[str, str],
    *,
    calibration_root: Path,
) -> tuple[tuple[float, float], str, str]:
    path, _manifest, rows = load_accepted_edge_trim_manifest(
        calibration_root,
        planet=record["planet"],
        mode=record["mode"],
        required_datasets=((record["epoch"], record["arm"]),),
    )

    row = rows[(record["epoch"], normalize_dataset_key(record["arm"]))]
    widths = (float(row["left_trim_A"]), float(row["right_trim_A"]))
    return widths, str(path.resolve()), "accepted_post_sysrem"


def _namespace(
    record: dict[str, str],
    *,
    product_kind: str,
    trim_source: str,
    apply_stellar_rest: bool,
) -> argparse.Namespace:
    phase_bin = "full" if record["mode"] == "transmission" and product_kind == "timeseries" else "all"
    return argparse.Namespace(
        planet=record["planet"],
        ephemeris=record["ephemeris"],
        epoch=record["epoch"],
        arm=record["arm"],
        phase_bin=phase_bin,
        product_kind=product_kind,
        output_dir=None,
        molecfit=True,
        regrid=True,
        subtract_median=True,
        run_sysrem=True,
        edge_trim_manifest=Path(trim_source),
        apply_stellar_rest=bool(apply_stellar_rest),
    )


def _process_prepared_product(
    record: dict[str, str],
    *,
    product_kind: str,
    output_dir: Path,
    trim_source: str,
    apply_stellar_rest: bool,
) -> None:
    args = _namespace(
        record,
        product_kind=product_kind,
        trim_source=trim_source,
        apply_stellar_rest=apply_stellar_rest,
    )
    params = config_utils.get_params(record["planet"], record["ephemeris"])
    if record["mode"] == "transmission":
        prepare_transmission_arm(
            arm=record["arm"],
            args=args,
            planet_cfg=params,
            output_dir=output_dir,
        )
        return
    prepare_emission_arm(
        arm=record["arm"],
        args=args,
        planet_cfg=params,
        reference_epoch=float(params["epoch"]),
        period=float(params["period"]),
        ra=str(params["RA"]),
        dec=str(params["Dec"]),
        output_dir=output_dir,
    )


def _collapse_products(record: dict[str, str], *, dataset_root: Path) -> list[dict[str, Any]]:
    params = config_utils.get_params(record["planet"], record["ephemeris"])
    kp_kms = float(params["Kp"])
    if not math.isfinite(kp_kms):
        raise ValueError(
            f"No finite Kp is configured for {record['planet']} {record['ephemeris']}."
        )
    source_dir = dataset_root / "collapse_source"
    results: list[dict[str, Any]] = []
    if record["mode"] == "transmission":
        output_dir = dataset_root / "collapsed" / "full_transit"
        results.append(
            collapse_transmission_epoch_arm(
                planet=record["planet"],
                ephemeris=record["ephemeris"],
                epoch=record["epoch"],
                arm=record["arm"],
                kp_kms=kp_kms,
                bin_size=1,
                source_dir=source_dir,
                output_dir=output_dir,
            )
        )
        return results

    for selection in EMISSION_COLLAPSE_SELECTIONS:
        output_dir = dataset_root / "collapsed" / selection
        results.append(
            collapse_emission_epoch_arm(
                planet=record["planet"],
                ephemeris=record["ephemeris"],
                epoch=record["epoch"],
                arm=record["arm"],
                selection=selection,
                kp_kms=kp_kms,
                bin_size=1,
                min_exposures=1,
                source_dir=source_dir,
                output_dir=output_dir,
            )
        )
    return results


def _fit_transmission_lsd_shadow(
    record: dict[str, str],
    *,
    dataset_root: Path,
) -> None:
    """Fit once and install exact-grid LSD models in both source products."""

    fit_doppler_shadow(
        DopplerShadowFitConfig(
            planet=record["planet"],
            ephemeris=record["ephemeris"],
            shadow_source="Recommended",
            epoch=record["epoch"],
            arm=record["arm"],
            prepared_root=dataset_root.parent,
            diagnostic_root=dataset_root / "diagnostics" / "doppler_shadow",
        )
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the active HRS inventory from raw FITS into an isolated tree."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["transmission", "emission"],
        action="append",
        dest="modes",
        help="Limit to one or both modes (default: both)",
    )
    parser.add_argument(
        "--planet",
        action="append",
        default=[],
        help="Limit to configured planets; may be repeated",
    )
    parser.add_argument(
        "--edge-trim-calibration-root",
        type=Path,
        default=DEFAULT_EDGE_TRIM_ROOT,
    )
    parser.add_argument(
        "--apply-stellar-rest",
        action="store_true",
        help="Opt into accepted LSD stellar-rest corrections; default output is barycentric",
    )
    parser.add_argument(
        "--include-raw-only-transmission",
        action="store_true",
        help=(
            "Compatibility flag; raw transmission epochs are always included."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write products. Without this flag, print and save only a plan.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue through the inventory and record per-dataset failures",
    )
    parser.add_argument(
        "--timeseries-only",
        action="store_true",
        help=(
            "Generate only emission retrieval time series and frozen operators. "
            "Transmission requires both source grids for the standard LSD shadow."
        ),
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    output_root = args.output_root.resolve()
    if _inside(output_root, CANONICAL_HRS_ROOT):
        raise ValueError(
            f"Validation output must not be inside the canonical HRS tree: {CANONICAL_HRS_ROOT}"
        )
    modes = tuple(args.modes or ("transmission", "emission"))
    if args.timeseries_only and "transmission" in modes:
        raise ValueError(
            "--timeseries-only is not valid for transmission regeneration: the "
            "standard LSD shadow requires both timeseries and collapse_source grids."
        )
    planets = {normalize_dataset_key(value) for value in args.planet}
    inventory = discover_supported_inventory(modes=modes, planets=planets)
    if not inventory:
        raise ValueError("No supported HRS datasets matched the requested inventory filters.")
    if args.execute and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a regeneration with existing files in {output_root}."
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "isolated_hrs_raw_to_collapsed_regeneration",
        "created_utc": _utc_now(),
        "status": "planned" if not args.execute else "running",
        "canonical_arrays_modified": False,
        "canonical_inventory_metadata_read_only": True,
        "output_root": str(output_root),
        "wavelength_medium": "vacuum",
        "requested_wavelength_frame": (
            "stellar_rest" if args.apply_stellar_rest else "barycentric"
        ),
        "product_scope": (
            "timeseries_only" if args.timeseries_only else "timeseries_and_collapsed"
        ),
        "edge_trim_policy": "newest_adaptive_schema_v4_accepted_manifest_only",
        "datasets": [],
    }

    planned: list[
        tuple[dict[str, str], tuple[float, float], str, str]
    ] = []
    for record in inventory:
        try:
            widths, trim_source, trim_status = _edge_trim_for(
                record,
                calibration_root=args.edge_trim_calibration_root,
            )
        except (FileNotFoundError, EdgeTrimManifestError) as exc:
            entry = {
                **record,
                "status": "skipped",
                "skip_reason": f"no_accepted_edge_trim: {exc}",
            }
            manifest["datasets"].append(entry)
            continue
        planned.append((record, widths, trim_source, trim_status))
        manifest["datasets"].append(
            {
                **record,
                "status": "planned",
                "edge_trim_left_A": widths[0],
                "edge_trim_right_A": widths[1],
                "edge_trim_source": trim_source,
                "edge_trim_status": trim_status,
            }
        )

    _write_manifest(output_root, manifest)
    print(
        f"Planned {len(planned)} epoch-arm datasets under {output_root}; "
        f"skipped={len(inventory) - len(planned)}."
    )
    if not args.execute:
        return 0

    manifest_rows = {
        (row["mode"], row["planet_slug"], row["epoch"], row["arm"]): row
        for row in manifest["datasets"]
        if row.get("status") == "planned"
    }
    failures = 0
    for index, (record, widths, trim_source, trim_status) in enumerate(
        planned,
        start=1,
    ):
        key = (
            record["mode"],
            record["planet_slug"],
            record["epoch"],
            record["arm"],
        )
        row = manifest_rows[key]
        dataset_root = output_root.joinpath(*key)
        row["status"] = "running"
        row["started_utc"] = _utc_now()
        row["dataset_root"] = str(dataset_root)
        _write_manifest(output_root, manifest)
        print(
            f"[{index}/{len(planned)}] {record['mode']} {record['planet']} "
            f"{record['epoch']} {record['arm']}"
        )
        try:
            _process_prepared_product(
                record,
                product_kind="timeseries",
                output_dir=dataset_root / "timeseries",
                trim_source=trim_source,
                apply_stellar_rest=bool(args.apply_stellar_rest),
            )
            if args.timeseries_only:
                collapsed = []
            else:
                _process_prepared_product(
                    record,
                    product_kind="collapse-source",
                    output_dir=dataset_root / "collapse_source",
                    trim_source=trim_source,
                    apply_stellar_rest=bool(args.apply_stellar_rest),
                )
                if record["mode"] == "transmission":
                    _fit_transmission_lsd_shadow(
                        record,
                        dataset_root=dataset_root,
                    )
                collapsed = _collapse_products(record, dataset_root=dataset_root)
            row["status"] = "complete"
            row["collapsed_statuses"] = [item.get("status", "ready") for item in collapsed]
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["failure_type"] = type(exc).__name__
            row["failure_reason"] = str(exc)
            row["traceback"] = traceback.format_exc()
            if not args.continue_on_error:
                row["finished_utc"] = _utc_now()
                manifest["status"] = "failed"
                manifest["finished_utc"] = _utc_now()
                _write_manifest(output_root, manifest)
                raise
        row["finished_utc"] = _utc_now()
        _write_manifest(output_root, manifest)

    manifest["status"] = "complete" if failures == 0 else "complete_with_failures"
    manifest["finished_utc"] = _utc_now()
    manifest["n_complete"] = sum(
        row.get("status") == "complete" for row in manifest["datasets"]
    )
    manifest["n_failed"] = failures
    manifest["n_skipped"] = sum(
        row.get("status") == "skipped" for row in manifest["datasets"]
    )
    _write_manifest(output_root, manifest)
    print(
        f"Regeneration {manifest['status']}: complete={manifest['n_complete']}, "
        f"failed={manifest['n_failed']}, skipped={manifest['n_skipped']}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
