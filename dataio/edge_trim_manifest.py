"""Load edge-trim calibration manifests and dataset widths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ACCEPTED_STATUS = "accepted_post_sysrem"
MANIFEST_FILENAME = "proposed_edge_trim_manifest.json"


def normalize_dataset_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def load_accepted_edge_trim_manifest(
    calibration_root: Path,
    *,
    planet: str,
    mode: str,
    required_datasets: Iterable[tuple[str, str]] = (),
    manifest_path: Path | None = None,
) -> tuple[Path, dict, dict[tuple[str, str], dict]]:
    """Read the requested manifest, or the newest calibration run."""
    if manifest_path is None:
        group_dir = Path(calibration_root) / normalize_dataset_key(mode) / normalize_dataset_key(planet)
        newest_run = sorted(path for path in group_dir.iterdir() if path.is_dir())[-1]
        selected_path = newest_run / MANIFEST_FILENAME
    else:
        selected_path = Path(manifest_path)
    manifest = json.loads(selected_path.read_text())
    rows = {
        (str(row["epoch"]), normalize_dataset_key(row["arm"])): row
        for row in manifest["datasets"]
    }
    return selected_path, manifest, rows


def load_accepted_edge_trim_widths(
    manifest_path: Path,
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
) -> tuple[Path, tuple[float, float]]:
    """Read one dataset's trim widths from the manifest."""

    key = (str(epoch), normalize_dataset_key(arm))
    selected_path, _manifest, rows = load_accepted_edge_trim_manifest(
        Path(manifest_path).parent,
        planet=planet,
        mode=mode,
        required_datasets=(key,),
        manifest_path=manifest_path,
    )
    row = rows[key]
    return selected_path, (float(row["left_trim_A"]), float(row["right_trim_A"]))
