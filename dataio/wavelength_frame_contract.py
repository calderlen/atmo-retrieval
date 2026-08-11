"""Build and validate persisted wavelength-frame provenance for HRS products."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


WAVELENGTH_FRAME_CONTRACT_SCHEMA_VERSION = 1
_PER_EXPOSURE_KEYS = (
    "input_exposure_files",
    "wavelength_frame_source",
    "observer_velocity_removed_kms",
    "stellar_velocity_removed_kms",
    "instrument_velocity_removed_kms",
    "synthetic_molecfit_shift_undone_kms",
    "velocity_history_recipe",
    "velocity_history_entries",
    "velocity_frame_parse_error",
    "wavelength_velocity_correction_mps",
    "wavelength_velocity_components",
)


def _json_value(value: Any) -> Any:
    """Convert NumPy and tuple values into deterministic JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def subset_loader_frame_extras(
    extras: dict[str, Any],
    selection: np.ndarray,
) -> dict[str, Any]:
    """Subset loader provenance arrays when the saved source exposure set changes."""

    selection = np.asarray(selection)
    if selection.ndim != 1:
        raise ValueError("Loader-frame provenance selection must be one-dimensional.")
    if selection.dtype == bool:
        indices = np.flatnonzero(selection)
        expected_length = int(selection.size)
    else:
        indices = selection.astype(int, copy=False)
        expected_length = None

    result = dict(extras)
    for key in _PER_EXPOSURE_KEYS:
        if key not in extras or extras[key] is None:
            continue
        value = extras[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                continue
            length = int(value.shape[0])
            if expected_length is not None and length != expected_length:
                raise ValueError(
                    f"Loader-frame provenance field {key!r} has {length} rows; "
                    f"expected {expected_length}."
                )
            result[key] = value[indices]
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if expected_length is not None and len(value) != expected_length:
                raise ValueError(
                    f"Loader-frame provenance field {key!r} has {len(value)} rows; "
                    f"expected {expected_length}."
                )
            result[key] = [value[int(index)] for index in indices]
    return result


def _per_exposure_values(
    extras: dict[str, Any],
    key: str,
    n_source_exposures: int,
    *,
    default: Any,
) -> list[Any]:
    value = extras.get(key)
    if value is None:
        return [default for _ in range(n_source_exposures)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [_json_value(value.item()) for _ in range(n_source_exposures)]
        values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value for _ in range(n_source_exposures)]
    if len(values) != n_source_exposures:
        raise ValueError(
            f"Loader-frame provenance field {key!r} has {len(values)} rows; "
            f"expected {n_source_exposures}."
        )
    return [_json_value(item) for item in values]


def _reconstruction_operations(
    *,
    frame_source: str,
    observer_removed_kms: float | None,
    stellar_removed_kms: float | None,
    instrument_removed_kms: float | None,
    synthetic_shift_undone_kms: float | None,
) -> list[dict[str, Any]]:
    """Describe the exact medium/frame operations represented by loader code."""

    if frame_source in {"raw_pipeline_history", "paired_raw_product"}:
        return [
            {
                "operation": "convert_air_to_vacuum",
                "velocity_kms": None,
            },
            {
                "operation": "restore_pepsi_stellar_velocity_term",
                "velocity_kms": stellar_removed_kms,
            },
        ]
    if frame_source == "molecfit_header_reconstruction":
        return [
            {
                "operation": "use_molecfit_vacuum_grid",
                "velocity_kms": None,
            },
            {
                "operation": "undo_synthetic_molecfit_shift",
                "velocity_kms": synthetic_shift_undone_kms,
            },
            {
                "operation": "remove_observer_velocity_from_molecfit_grid",
                "velocity_kms": observer_removed_kms,
            },
            {
                "operation": "remove_instrument_velocity_from_molecfit_grid",
                "velocity_kms": instrument_removed_kms,
            },
        ]
    return [
        {
            "operation": "unclassified_loader_frame_path",
            "velocity_kms": None,
        }
    ]


def build_wavelength_frame_contract(
    extras: dict[str, Any],
    *,
    n_source_exposures: int,
    stellar_velocity: dict[str, Any],
) -> dict[str, Any]:
    """Return the complete persisted wavelength medium/reference-frame contract."""

    n_source_exposures = int(n_source_exposures)
    if n_source_exposures <= 0:
        raise ValueError("A wavelength-frame contract requires at least one exposure.")

    medium = str(extras.get("wavelength_medium", "")).strip().lower()
    if medium != "vacuum":
        raise ValueError(
            f"Prepared HRS products require wavelength_medium='vacuum'; found {medium!r}."
        )
    final_frame = str(extras.get("wavelength_frame", "")).strip().lower()
    if final_frame not in {"barycentric", "stellar_rest"}:
        raise ValueError(
            "Prepared HRS products require wavelength_frame='barycentric' or "
            f"'stellar_rest'; found {final_frame!r}."
        )

    sources = _per_exposure_values(
        extras,
        "wavelength_frame_source",
        n_source_exposures,
        default="unknown",
    )
    observer = _per_exposure_values(
        extras,
        "observer_velocity_removed_kms",
        n_source_exposures,
        default=None,
    )
    stellar = _per_exposure_values(
        extras,
        "stellar_velocity_removed_kms",
        n_source_exposures,
        default=None,
    )
    instrument = _per_exposure_values(
        extras,
        "instrument_velocity_removed_kms",
        n_source_exposures,
        default=None,
    )
    synthetic = _per_exposure_values(
        extras,
        "synthetic_molecfit_shift_undone_kms",
        n_source_exposures,
        default=0.0,
    )
    history_recipe = _per_exposure_values(
        extras,
        "velocity_history_recipe",
        n_source_exposures,
        default=[],
    )
    history_entries = _per_exposure_values(
        extras,
        "velocity_history_entries",
        n_source_exposures,
        default=[],
    )
    parse_errors = _per_exposure_values(
        extras,
        "velocity_frame_parse_error",
        n_source_exposures,
        default=None,
    )
    legacy_correction_mps = _per_exposure_values(
        extras,
        "wavelength_velocity_correction_mps",
        n_source_exposures,
        default=0.0,
    )
    legacy_components = _per_exposure_values(
        extras,
        "wavelength_velocity_components",
        n_source_exposures,
        default=[],
    )

    exposure_rows: list[dict[str, Any]] = []
    for index in range(n_source_exposures):
        frame_source = str(sources[index])
        uses_molecfit = frame_source in {
            "paired_raw_product",
            "molecfit_header_reconstruction",
        }
        input_medium = (
            "vacuum"
            if frame_source == "molecfit_header_reconstruction"
            else "air"
        )
        exposure_rows.append(
            {
                "source_exposure_index": index,
                "flux_product": "molecfit" if uses_molecfit else "raw_pepsi",
                "input_wavelength_medium": input_medium,
                "barycentric_reconstruction_method": frame_source,
                "pepsi_velocity_terms_removed_kms": {
                    "observer": observer[index],
                    "stellar": stellar[index],
                    "instrument": instrument[index],
                },
                "barycentric_reconstruction_operations": _reconstruction_operations(
                    frame_source=frame_source,
                    observer_removed_kms=observer[index],
                    stellar_removed_kms=stellar[index],
                    instrument_removed_kms=instrument[index],
                    synthetic_shift_undone_kms=synthetic[index],
                ),
                "synthetic_molecfit_shift_undone_kms": synthetic[index],
                "pepsi_velocity_history_recipe": history_recipe[index],
                "pepsi_velocity_history_entries": history_entries[index],
                "velocity_frame_parse_error": parse_errors[index],
                "legacy_native_velocity_correction_mps": legacy_correction_mps[index],
                "legacy_native_velocity_components": legacy_components[index],
            }
        )

    stellar_rest_velocity_kms = extras.get("stellar_rest_velocity_kms")
    stellar_rest_correction = dict(stellar_velocity)
    stellar_rest_correction["applied_velocity_kms"] = _json_value(
        stellar_rest_velocity_kms
    )
    stellar_rest_correction["output_frame"] = final_frame

    methods = list(
        dict.fromkeys(row["barycentric_reconstruction_method"] for row in exposure_rows)
    )
    return {
        "schema_version": WAVELENGTH_FRAME_CONTRACT_SCHEMA_VERSION,
        "wavelength_medium": medium,
        "wavelength_frame": final_frame,
        "intermediate_frame": "barycentric",
        "barycentric_reconstruction_recipe": extras.get(
            "wavelength_velocity_recipe",
            "unknown",
        ),
        "barycentric_reconstruction_methods": methods,
        "molecfit_reconstruction_methods": [
            method
            for method in methods
            if method in {"paired_raw_product", "molecfit_header_reconstruction"}
        ],
        "stellar_rest_correction": _json_value(stellar_rest_correction),
        "n_source_exposures": n_source_exposures,
        "per_exposure": exposure_rows,
    }
