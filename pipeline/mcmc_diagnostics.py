"""Structured diagnostics for NumPyro MCMC retrieval runs."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MCMC_EXTRA_FIELDS: tuple[str, ...] = (
    "potential_energy",
    "energy",
    "num_steps",
    "accept_prob",
    "diverging",
    "adapt_state.step_size",
)

POSTERIOR_BY_CHAIN_FILENAME = "posterior_sample_by_chain.npz"
DIAGNOSTICS_DIRNAME = "diagnostics"
EXTRA_FIELDS_FILENAME = "mcmc_extra_fields.npz"
SUMMARY_FILENAME = "mcmc_diagnostics_summary.json"
CHAIN_SUMMARY_FILENAME = "mcmc_chain_summary.csv"
PARAMETER_MOVEMENT_FILENAME = "mcmc_parameter_movement.csv"


def sanitize_diagnostic_label(label: str | None) -> str | None:
    """Return a filesystem-friendly diagnostic label, or None for no label."""
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return slug or "diagnostic"


def get_samples_by_chain(mcmc: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    """Fetch posterior samples with chain dimension preserved."""
    warnings: list[str] = []
    try:
        samples = mcmc.get_samples(group_by_chain=True)
    except TypeError:
        warnings.append("mcmc.get_samples(group_by_chain=True) is unavailable; using flattened samples.")
        samples = mcmc.get_samples()
    except Exception as exc:
        warnings.append(f"failed to get chain-grouped samples: {exc}")
        return {}, warnings
    return _arrays_from_mapping(samples), warnings


def get_extra_fields_by_chain(mcmc: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    """Fetch sampler extra fields with chain dimension preserved when possible."""
    warnings: list[str] = []
    try:
        fields = mcmc.get_extra_fields(group_by_chain=True)
    except TypeError:
        warnings.append("mcmc.get_extra_fields(group_by_chain=True) is unavailable; using default shape.")
        try:
            fields = mcmc.get_extra_fields()
        except Exception as exc:
            warnings.append(f"failed to get extra fields: {exc}")
            return {}, warnings
    except Exception as exc:
        warnings.append(f"failed to get extra fields: {exc}")
        return {}, warnings
    return _arrays_from_mapping(fields), warnings


def save_chain_grouped_posterior(output_dir: str | Path, samples_by_chain: Mapping[str, Any]) -> Path | None:
    """Save chain-grouped posterior samples beside the legacy flattened sample file."""
    arrays = _arrays_from_mapping(samples_by_chain)
    if not arrays:
        return None
    path = Path(output_dir) / POSTERIOR_BY_CHAIN_FILENAME
    np.savez_compressed(path, **arrays)
    return path


def write_mcmc_diagnostics(
    output_dir: str | Path,
    *,
    posterior_by_chain: Mapping[str, Any] | None,
    extra_fields: Mapping[str, Any] | None,
    num_chains: int,
    num_samples: int,
    max_tree_depth: int,
    requested_extra_fields: tuple[str, ...] = DEFAULT_MCMC_EXTRA_FIELDS,
    warnings: list[str] | None = None,
    diagnostic_label: str | None = None,
) -> dict[str, Any]:
    """Write structured MCMC diagnostics and return the summary payload."""
    output_path = Path(output_dir)
    diagnostics_dir = output_path / DIAGNOSTICS_DIRNAME
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    warning_messages = list(warnings or [])
    posterior_arrays = _arrays_from_mapping(posterior_by_chain or {})
    extra_arrays = _arrays_from_mapping(extra_fields or {})

    missing_fields = [field for field in requested_extra_fields if field not in extra_arrays]
    if missing_fields:
        warning_messages.append("missing extra fields: " + ", ".join(missing_fields))

    extra_path = diagnostics_dir / EXTRA_FIELDS_FILENAME
    if extra_arrays:
        np.savez_compressed(extra_path, **extra_arrays)
    else:
        np.savez_compressed(extra_path, __empty__=np.asarray([], dtype=float))

    movement_rows = _parameter_movement_rows(posterior_arrays, num_chains=num_chains)
    _write_csv(
        diagnostics_dir / PARAMETER_MOVEMENT_FILENAME,
        [
            "parameter",
            "chain",
            "mean",
            "std",
            "min",
            "max",
            "unique_rounded_1e10",
            "zero_diff_fraction",
        ],
        movement_rows,
    )

    max_num_steps = int(2**int(max_tree_depth) - 1)
    chain_rows = _chain_summary_rows(
        extra_arrays,
        num_chains=num_chains,
        num_samples=num_samples,
        max_num_steps=max_num_steps,
    )
    _write_csv(
        diagnostics_dir / CHAIN_SUMMARY_FILENAME,
        [
            "chain",
            "samples",
            "divergences",
            "divergence_fraction",
            "median_accept_prob",
            "median_num_steps",
            "max_num_steps_seen",
            "fraction_at_max_tree",
            "median_step_size",
            "energy_std",
        ],
        chain_rows,
    )

    summary = _run_summary(
        extra_arrays,
        movement_rows=movement_rows,
        num_chains=num_chains,
        num_samples=num_samples,
        max_tree_depth=max_tree_depth,
        max_num_steps=max_num_steps,
        missing_fields=missing_fields,
        warnings=warning_messages,
    )
    if diagnostic_label:
        summary["diagnostic_label"] = diagnostic_label
    with (diagnostics_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def _arrays_from_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in values.items():
        _flatten_value(str(key), value, arrays)
    return arrays


def _flatten_value(prefix: str, value: Any, arrays: dict[str, np.ndarray]) -> None:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _flatten_value(f"{prefix}.{child_key}", child_value, arrays)
        return
    if hasattr(value, "_asdict"):
        _flatten_value(prefix, value._asdict(), arrays)
        return
    try:
        import jax

        value = jax.device_get(value)
    except Exception:
        pass
    try:
        array = np.asarray(value)
    except Exception:
        return
    if array.dtype == object:
        return
    arrays[prefix] = array


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isfinite(number):
        return number
    return None


def _finite_values(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _nan_stat(array: np.ndarray | None, stat: str) -> float | None:
    if array is None:
        return None
    values = _finite_values(array)
    if values.size == 0:
        return None
    if stat == "median":
        return float(np.median(values))
    if stat == "min":
        return float(np.min(values))
    if stat == "max":
        return float(np.max(values))
    if stat == "std":
        return float(np.std(values))
    raise ValueError(f"Unknown stat: {stat}")


def _field(extra_fields: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in extra_fields:
            return np.asarray(extra_fields[name])
    return None


def _as_chain_sample_array(
    array: np.ndarray | None,
    *,
    num_chains: int,
    num_samples: int | None = None,
) -> np.ndarray | None:
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim == 0:
        return None
    if arr.ndim >= 2 and arr.shape[0] == num_chains:
        return arr
    if num_samples is not None and arr.shape[0] == num_chains * num_samples:
        return arr.reshape((num_chains, num_samples) + arr.shape[1:])
    return None


def _parameter_movement_rows(
    posterior_by_chain: Mapping[str, np.ndarray],
    *,
    num_chains: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, values in sorted(posterior_by_chain.items()):
        arr = np.asarray(values)
        if arr.ndim < 2 or arr.shape[0] != num_chains:
            continue
        for chain in range(num_chains):
            chain_values = np.asarray(arr[chain], dtype=float)
            flat = chain_values.reshape(-1)
            finite = flat[np.isfinite(flat)]
            if finite.size == 0:
                mean = std = min_value = max_value = None
                unique_count = 0
            else:
                mean = float(np.mean(finite))
                std = float(np.std(finite))
                min_value = float(np.min(finite))
                max_value = float(np.max(finite))
                unique_count = int(np.unique(np.round(finite, 10)).size)

            if chain_values.shape[0] <= 1:
                zero_diff_fraction = None
            else:
                diffs = np.diff(chain_values.reshape((chain_values.shape[0], -1)), axis=0).reshape(-1)
                finite_diffs = diffs[np.isfinite(diffs)]
                if finite_diffs.size == 0:
                    zero_diff_fraction = None
                else:
                    zero_diff_fraction = float(np.mean(np.abs(finite_diffs) <= 1.0e-10))

            rows.append(
                {
                    "parameter": name,
                    "chain": chain,
                    "mean": mean,
                    "std": std,
                    "min": min_value,
                    "max": max_value,
                    "unique_rounded_1e10": unique_count,
                    "zero_diff_fraction": zero_diff_fraction,
                }
            )
    return rows


def _chain_summary_rows(
    extra_fields: Mapping[str, np.ndarray],
    *,
    num_chains: int,
    num_samples: int,
    max_num_steps: int,
) -> list[dict[str, Any]]:
    divergences = _as_chain_sample_array(
        _field(extra_fields, "diverging"),
        num_chains=num_chains,
        num_samples=num_samples,
    )
    accept_prob = _as_chain_sample_array(
        _field(extra_fields, "accept_prob"),
        num_chains=num_chains,
        num_samples=num_samples,
    )
    num_steps = _as_chain_sample_array(
        _field(extra_fields, "num_steps"),
        num_chains=num_chains,
        num_samples=num_samples,
    )
    step_size = _as_chain_sample_array(
        _field(extra_fields, "adapt_state.step_size", "step_size"),
        num_chains=num_chains,
        num_samples=num_samples,
    )
    energy = _as_chain_sample_array(
        _field(extra_fields, "energy"),
        num_chains=num_chains,
        num_samples=num_samples,
    )

    rows: list[dict[str, Any]] = []
    for chain in range(num_chains):
        chain_div = None if divergences is None else np.asarray(divergences[chain])
        chain_accept = None if accept_prob is None else np.asarray(accept_prob[chain])
        chain_steps = None if num_steps is None else np.asarray(num_steps[chain])
        chain_step_size = None if step_size is None else np.asarray(step_size[chain])
        chain_energy = None if energy is None else np.asarray(energy[chain])

        divergence_count = int(np.sum(chain_div.astype(bool))) if chain_div is not None else None
        divergence_fraction = (
            float(divergence_count / chain_div.size)
            if chain_div is not None and chain_div.size
            else None
        )
        fraction_at_max_tree = None
        max_num_steps_seen = None
        if chain_steps is not None:
            finite_steps = _finite_values(chain_steps)
            if finite_steps.size:
                max_num_steps_seen = int(np.max(finite_steps))
                fraction_at_max_tree = float(np.mean(finite_steps >= max_num_steps))

        rows.append(
            {
                "chain": chain,
                "samples": int(chain_accept.shape[0]) if chain_accept is not None and chain_accept.ndim else num_samples,
                "divergences": divergence_count,
                "divergence_fraction": divergence_fraction,
                "median_accept_prob": _nan_stat(chain_accept, "median"),
                "median_num_steps": _nan_stat(chain_steps, "median"),
                "max_num_steps_seen": max_num_steps_seen,
                "fraction_at_max_tree": fraction_at_max_tree,
                "median_step_size": _nan_stat(chain_step_size, "median"),
                "energy_std": _nan_stat(chain_energy, "std"),
            }
        )
    return rows


def _run_summary(
    extra_fields: Mapping[str, np.ndarray],
    *,
    movement_rows: list[dict[str, Any]],
    num_chains: int,
    num_samples: int,
    max_tree_depth: int,
    max_num_steps: int,
    missing_fields: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    divergences = _field(extra_fields, "diverging")
    accept_prob = _field(extra_fields, "accept_prob")
    num_steps = _field(extra_fields, "num_steps")
    step_size = _field(extra_fields, "adapt_state.step_size", "step_size")
    energy = _field(extra_fields, "energy")
    potential_energy = _field(extra_fields, "potential_energy")

    divergence_count = int(np.sum(np.asarray(divergences).astype(bool))) if divergences is not None else None
    divergence_fraction = (
        float(divergence_count / np.asarray(divergences).size)
        if divergences is not None and np.asarray(divergences).size
        else None
    )

    fraction_at_max_tree = None
    max_num_steps_seen = None
    if num_steps is not None:
        finite_steps = _finite_values(num_steps)
        if finite_steps.size:
            max_num_steps_seen = int(np.max(finite_steps))
            fraction_at_max_tree = float(np.mean(finite_steps >= max_num_steps))

    frozen_fractions = [
        _finite_float(row.get("zero_diff_fraction"))
        for row in movement_rows
        if _finite_float(row.get("zero_diff_fraction")) is not None
    ]
    frozen_like = [value for value in frozen_fractions if value >= 0.95]
    frozen_parameter_fraction = (
        float(len(frozen_like) / len(frozen_fractions)) if frozen_fractions else None
    )
    chains_frozen = bool(frozen_parameter_fraction is not None and frozen_parameter_fraction >= 0.5)

    final_step_size_by_chain = None
    step_array = _as_chain_sample_array(step_size, num_chains=num_chains, num_samples=num_samples)
    if step_array is not None:
        final_values = []
        for chain in range(num_chains):
            final_value = _finite_values(np.asarray(step_array[chain])[-1:])
            final_values.append(float(final_value[-1]) if final_value.size else None)
        final_step_size_by_chain = final_values

    median_accept = _nan_stat(accept_prob, "median")
    low_accept = bool(median_accept is not None and median_accept < 0.6)
    hit_max_tree = bool(fraction_at_max_tree is not None and fraction_at_max_tree > 0.01)
    divergences_present = bool(divergence_count is not None and divergence_count > 0)
    missing_extra_fields = bool(missing_fields)

    return {
        "num_chains": int(num_chains),
        "num_samples": int(num_samples),
        "max_tree_depth": int(max_tree_depth),
        "max_num_steps": int(max_num_steps),
        "divergence_count": divergence_count,
        "divergence_fraction": divergence_fraction,
        "median_accept_prob": median_accept,
        "min_accept_prob": _nan_stat(accept_prob, "min"),
        "max_accept_prob": _nan_stat(accept_prob, "max"),
        "median_num_steps": _nan_stat(num_steps, "median"),
        "max_num_steps_seen": max_num_steps_seen,
        "fraction_at_max_tree": fraction_at_max_tree,
        "median_step_size": _nan_stat(step_size, "median"),
        "final_step_size_by_chain": final_step_size_by_chain,
        "energy_min": _nan_stat(energy, "min"),
        "energy_max": _nan_stat(energy, "max"),
        "energy_std": _nan_stat(energy, "std"),
        "potential_energy_min": _nan_stat(potential_energy, "min"),
        "potential_energy_max": _nan_stat(potential_energy, "max"),
        "potential_energy_std": _nan_stat(potential_energy, "std"),
        "frozen_parameter_fraction": frozen_parameter_fraction,
        "warning_flags": {
            "hit_max_tree_depth": hit_max_tree,
            "low_accept_prob": low_accept,
            "chains_frozen": chains_frozen,
            "divergences_present": divergences_present,
            "missing_extra_fields": missing_extra_fields,
        },
        "missing_extra_fields": missing_fields,
        "warnings": warnings,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
