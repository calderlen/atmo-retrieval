"""Exact, manifest-driven MAST product downloads."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


def _normalize_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mission": str(selection["mission"]).strip(),
        "dataset_ids": [str(value).strip() for value in selection["dataset_ids"]],
        "product_keys": [str(value).strip() for value in selection["product_keys"]],
    }


def load_mast_selection(path: str | Path) -> dict[str, Any]:
    """Read an exact MAST selection manifest."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    return _normalize_selection(selection)


def _column_values(table: Any, name: str) -> list[str]:
    if name not in getattr(table, "colnames", ()):
        return []
    return [str(value) for value in table[name]]


def download_mast_selection(
    selection: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Download every exact product in the selection."""

    normalized = _normalize_selection(selection)
    from astroquery.mast import MastMissions

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
