"""Exact, manifest-driven MAST product downloads."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


def validate_mast_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an exact MAST selection manifest."""

    if selection.get("schema_version") != 1:
        raise ValueError("MAST selection schema_version must be 1.")
    mission = str(selection.get("mission", "")).strip()
    dataset_ids = [str(value).strip() for value in selection.get("dataset_ids", [])]
    product_keys = [str(value).strip() for value in selection.get("product_keys", [])]
    if not mission or not dataset_ids or not product_keys:
        raise ValueError("MAST selection requires mission, dataset_ids, and product_keys.")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("MAST selection contains duplicate dataset IDs.")
    if len(set(product_keys)) != len(product_keys):
        raise ValueError("MAST selection contains duplicate product keys.")
    dataset_set = set(dataset_ids)
    unknown = sorted(
        key for key in product_keys if key.split("_", 1)[0].upper() not in dataset_set
    )
    if unknown:
        raise ValueError(f"Product keys reference unselected datasets: {unknown[:5]}")
    return {
        "schema_version": 1,
        "mission": mission,
        "dataset_ids": dataset_ids,
        "product_keys": product_keys,
    }


def load_mast_selection(path: str | Path) -> dict[str, Any]:
    """Load and validate one exact MAST selection manifest."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if not isinstance(selection, dict):
        raise ValueError(f"MAST selection must be a JSON object: {manifest_path}")
    return validate_mast_selection(selection)


def _column_values(table: Any, name: str) -> list[str]:
    if name not in getattr(table, "colnames", ()):
        return []
    return [str(value) for value in table[name]]


def download_mast_selection(
    selection: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Download every exact product in a validated selection."""

    normalized = validate_mast_selection(selection)
    try:
        from astroquery.mast import MastMissions
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "astroquery is required for MAST downloads; install it in the active environment."
        ) from exc

    missions = MastMissions(mission=normalized["mission"])
    products = missions.get_product_list(normalized["dataset_ids"])
    available = set(_column_values(products, "product_key"))
    missing = sorted(set(normalized["product_keys"]) - available)
    if missing:
        raise RuntimeError(
            f"MAST returned no match for {len(missing)} requested product keys; "
            f"first missing keys: {missing[:5]}"
        )
    filtered = missions.filter_products(products, product_key=normalized["product_keys"])
    if len(filtered) != len(normalized["product_keys"]):
        raise RuntimeError(
            f"Expected {len(normalized['product_keys'])} filtered products, got {len(filtered)}."
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    result = missions.download_products(filtered, download_dir=str(directory))
    statuses = Counter(_column_values(result, "Status"))
    summary = {
        "schema_version": 1,
        "mission": normalized["mission"],
        "dataset_count": len(normalized["dataset_ids"]),
        "product_count": len(normalized["product_keys"]),
        "download_result_count": len(result),
        "status_counts": dict(sorted(statuses.items())),
        "output_dir": str(directory.resolve()),
    }
    (directory / "download_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
