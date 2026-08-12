"""Fail-closed loading of accepted edge-trim calibration manifests.

Calibration manifests are diagnostic proposals.  This module resolves their
validated boundaries for in-memory diagnostics only; importing it does not
authorize or write canonical prepared arrays.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable


ACCEPTED_STATUS = "accepted_post_sysrem"
MANIFEST_FILENAME = "proposed_edge_trim_manifest.json"
REQUIRED_PRODUCTS = frozenset({"timeseries", "collapse_source"})
DIAGNOSTIC_ARTIFACT_SUFFIXES = frozenset({".csv", ".json", ".md", ".pdf"})
MANIFEST_KIND = "proposed_dataset_specific_edge_trim_calibration"
ADAPTIVE_SCHEMA_VERSION = 4
ADAPTIVE_CANDIDATE_STRATEGY = "coarse_grid_with_transition_refinement"


class EdgeTrimManifestError(ValueError):
    """Raised when no unambiguous, acceptance-grade calibration is available."""


def normalize_dataset_key(value: object) -> str:
    """Normalize a planet, mode, epoch, or arm identifier for comparison."""

    if value is None:
        return ""
    return "".join(character for character in str(value).lower() if character.isalnum())


def _generated_at(manifest: dict, path: Path) -> tuple[datetime, str]:
    raw = manifest.get("generated_utc")
    if not isinstance(raw, str) or not raw.strip():
        raise EdgeTrimManifestError(f"{path}: missing generated_utc.")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EdgeTrimManifestError(
            f"{path}: invalid generated_utc={raw!r}."
        ) from exc
    if parsed.tzinfo is None:
        raise EdgeTrimManifestError(
            f"{path}: generated_utc must include an explicit timezone."
        )
    return parsed, str(path)


def validate_accepted_edge_trim_manifest(
    manifest: dict,
    path: Path,
    *,
    planet: str,
    mode: str,
    required_datasets: Iterable[tuple[str, str]] = (),
) -> dict[tuple[str, str], dict]:
    """Validate an accepted manifest and return rows keyed by ``(epoch, arm)``."""

    path = Path(path)
    if not isinstance(manifest, dict):
        raise EdgeTrimManifestError(f"{path}: manifest root must be an object.")
    if manifest.get("kind") != MANIFEST_KIND:
        raise EdgeTrimManifestError(
            f"{path}: kind must be {MANIFEST_KIND!r}, found {manifest.get('kind')!r}."
        )
    if manifest.get("schema_version") != ADAPTIVE_SCHEMA_VERSION:
        raise EdgeTrimManifestError(
            f"{path}: schema_version must be {ADAPTIVE_SCHEMA_VERSION}, found "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("canonical_generation_authorized") is not False:
        raise EdgeTrimManifestError(
            f"{path}: diagnostic proposal must record "
            "canonical_generation_authorized=false."
        )
    if manifest.get("prepared_arrays_written") is not False:
        raise EdgeTrimManifestError(
            f"{path}: diagnostic proposal must record prepared_arrays_written=false."
        )
    artifact_suffixes = manifest.get("allowed_artifact_suffixes")
    if (
        not isinstance(artifact_suffixes, list)
        or len(artifact_suffixes) != len(DIAGNOSTIC_ARTIFACT_SUFFIXES)
        or set(artifact_suffixes) != DIAGNOSTIC_ARTIFACT_SUFFIXES
    ):
        raise EdgeTrimManifestError(
            f"{path}: allowed_artifact_suffixes must be exactly "
            f"{sorted(DIAGNOSTIC_ARTIFACT_SUFFIXES)}."
        )
    if manifest.get("overall_status") != ACCEPTED_STATUS:
        raise EdgeTrimManifestError(
            f"{path}: overall_status must be {ACCEPTED_STATUS!r}, found "
            f"{manifest.get('overall_status')!r}."
        )
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise EdgeTrimManifestError(f"{path}: settings must be an object.")
    if manifest.get("calibration_wavelength_frame") != "barycentric":
        raise EdgeTrimManifestError(
            f"{path}: calibration_wavelength_frame must be 'barycentric'."
        )
    if manifest.get("calibration_wavelength_medium") != "vacuum":
        raise EdgeTrimManifestError(
            f"{path}: calibration_wavelength_medium must be 'vacuum'."
        )
    if settings.get("run_sysrem_finalists") is not True:
        raise EdgeTrimManifestError(
            f"{path}: accepted manifest must record run_sysrem_finalists=true."
        )
    required_settings = {
        "coarse_min_nm": 0.0,
        "coarse_max_nm": 20.0,
        "coarse_step_nm": 0.1,
        "refinement_step_nm": 0.02,
        "eval_width_nm": 3.0,
        "baseline_exclude_nm": 8.0,
        "accept_ratio": 1.35,
        "protected_line_pad_nm": 0.25,
        "minimum_eval_columns": 20,
        "maximum_finalists": 256,
        "run_sysrem_finalists": True,
        "use_molecfit_for_red": True,
    }
    if set(settings) != set(required_settings):
        raise EdgeTrimManifestError(
            f"{path}: settings fields must be exactly {sorted(required_settings)}."
        )
    for field, expected in required_settings.items():
        if settings.get(field) != expected:
            raise EdgeTrimManifestError(
                f"{path}: settings.{field} must be {expected!r}, found "
                f"{settings.get(field)!r}."
            )
    score_semantics = manifest.get("score_semantics")
    if not isinstance(score_semantics, dict):
        raise EdgeTrimManifestError(f"{path}: score_semantics must be an object.")
    if score_semantics.get("candidate_strategy") != ADAPTIVE_CANDIDATE_STRATEGY:
        raise EdgeTrimManifestError(
            f"{path}: candidate_strategy must be {ADAPTIVE_CANDIDATE_STRATEGY!r}."
        )
    products = manifest.get("products_required")
    if (
        not isinstance(products, list)
        or len(products) != len(REQUIRED_PRODUCTS)
        or set(products) != REQUIRED_PRODUCTS
    ):
        raise EdgeTrimManifestError(
            f"{path}: products_required must be exactly {sorted(REQUIRED_PRODUCTS)}."
        )

    expected_planet = normalize_dataset_key(planet)
    manifest_planet = normalize_dataset_key(manifest.get("planet_slug"))
    if manifest_planet != expected_planet:
        raise EdgeTrimManifestError(
            f"{path}: planet {manifest_planet!r} does not match {expected_planet!r}."
        )
    expected_mode = normalize_dataset_key(mode)
    manifest_mode = normalize_dataset_key(manifest.get("mode"))
    if manifest_mode != expected_mode:
        raise EdgeTrimManifestError(
            f"{path}: mode {manifest_mode!r} does not match {expected_mode!r}."
        )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise EdgeTrimManifestError(f"{path}: datasets must be a non-empty list.")
    rows: dict[tuple[str, str], dict] = {}
    for index, row in enumerate(datasets):
        if not isinstance(row, dict):
            raise EdgeTrimManifestError(f"{path}: datasets[{index}] must be an object.")
        if row.get("status") != ACCEPTED_STATUS:
            raise EdgeTrimManifestError(
                f"{path}: dataset {row.get('epoch')}/{row.get('arm')} has "
                f"status {row.get('status')!r}, not {ACCEPTED_STATUS!r}."
            )
        key = (str(row.get("epoch", "")), normalize_dataset_key(row.get("arm")))
        if not key[0] or not key[1]:
            raise EdgeTrimManifestError(f"{path}: datasets[{index}] lacks epoch or arm.")
        if key in rows:
            raise EdgeTrimManifestError(f"{path}: duplicate dataset row for {key}.")
        for field in ("left_trim_A", "right_trim_A"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise EdgeTrimManifestError(
                    f"{path}: {key} has invalid {field}."
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise EdgeTrimManifestError(
                    f"{path}: {key} has invalid {field}={value!r}."
                )
        if row.get("candidate_strategy") != ADAPTIVE_CANDIDATE_STRATEGY:
            raise EdgeTrimManifestError(
                f"{path}: {key} does not record the adaptive candidate strategy."
            )
        tested = row.get("tested_candidates_nm")
        if not isinstance(tested, dict) or set(tested) != {"left", "right"}:
            raise EdgeTrimManifestError(
                f"{path}: {key} tested_candidates_nm must contain left and right grids."
            )
        for side, width_field in (("left", "left_trim_A"), ("right", "right_trim_A")):
            grid = tested[side]
            if not isinstance(grid, list) or not grid:
                raise EdgeTrimManifestError(
                    f"{path}: {key} has no tested {side} candidate grid."
                )
            try:
                values = [float(value) for value in grid]
            except (TypeError, ValueError) as exc:
                raise EdgeTrimManifestError(
                    f"{path}: {key} has a non-numeric tested {side} grid."
                ) from exc
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise EdgeTrimManifestError(
                    f"{path}: {key} has an invalid tested {side} grid."
                )
            selected_nm = float(row[width_field]) / 10.0
            if not any(math.isclose(selected_nm, value, rel_tol=0.0, abs_tol=1e-10) for value in values):
                raise EdgeTrimManifestError(
                    f"{path}: {key} selected {side} trim {selected_nm} nm was not tested."
                )
        intervals = row.get("adaptive_refinement_intervals")
        if not isinstance(intervals, dict) or set(intervals) != {"left", "right"}:
            raise EdgeTrimManifestError(
                f"{path}: {key} adaptive_refinement_intervals must contain left and right."
            )
        rows[key] = row

    required = {
        (str(epoch), normalize_dataset_key(arm))
        for epoch, arm in required_datasets
    }
    missing = sorted(required - set(rows))
    if missing:
        raise EdgeTrimManifestError(
            f"{path}: accepted manifest is missing required datasets: {missing}."
        )
    _generated_at(manifest, path)
    return rows


def load_accepted_edge_trim_manifest(
    calibration_root: Path,
    *,
    planet: str,
    mode: str,
    required_datasets: Iterable[tuple[str, str]] = (),
    manifest_path: Path | None = None,
) -> tuple[Path, dict, dict[tuple[str, str], dict]]:
    """Load an explicit manifest or validate the newest calibration run.

    Discovery never searches backward for an older accepted result. The newest
    run directory is selected first and then must pass the complete adaptive
    schema and dataset-coverage checks.
    """

    required_datasets = tuple(required_datasets)
    if manifest_path is not None:
        selected_path = Path(manifest_path)
    else:
        root = Path(calibration_root)
        planet_slug = normalize_dataset_key(planet)
        mode_slug = normalize_dataset_key(mode)
        group_dir = root / mode_slug / planet_slug
        run_directories = sorted(path for path in group_dir.glob("*") if path.is_dir())
        if not run_directories:
            raise FileNotFoundError(
                f"No edge-trim calibration runs found for {mode}/{planet} under "
                f"{group_dir}."
            )
        newest_run = run_directories[-1]
        selected_path = newest_run / MANIFEST_FILENAME
        if not selected_path.is_file():
            raise EdgeTrimManifestError(
                f"Newest calibration run {newest_run} has no {MANIFEST_FILENAME}; "
                "older runs are not considered."
            )
    if not selected_path.is_file():
        raise FileNotFoundError(
            Path(manifest_path)
            if manifest_path is not None
            else selected_path
        )
    try:
        selected_manifest = json.loads(selected_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EdgeTrimManifestError(
            f"Could not read newest calibration manifest {selected_path}: {exc}"
        ) from exc
    if selected_manifest.get("overall_status") != ACCEPTED_STATUS:
        raise EdgeTrimManifestError(
            f"{selected_path}: selected manifest is not {ACCEPTED_STATUS}."
        )
    selected_rows = validate_accepted_edge_trim_manifest(
        selected_manifest,
        selected_path,
        planet=planet,
        mode=mode,
        required_datasets=required_datasets,
    )
    return selected_path, selected_manifest, selected_rows


def load_accepted_edge_trim_widths(
    manifest_path: Path,
    *,
    planet: str,
    mode: str,
    epoch: str,
    arm: str,
) -> tuple[Path, tuple[float, float]]:
    """Resolve one dataset's exact widths from an accepted schema-v4 manifest."""

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
