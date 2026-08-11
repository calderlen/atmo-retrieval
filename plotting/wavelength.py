"""Gap-preserving helpers for wavelength-coordinate diagnostic plots."""

from __future__ import annotations

from typing import Any

import numpy as np


def wavelength_segment_slices(
    wavelength: np.ndarray,
    *,
    gap_factor: float = 5.0,
) -> tuple[slice, ...]:
    """Return contiguous slices separated by unusually large wavelength steps."""

    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("Wavelength coordinates must be a nonempty 1-D array.")
    if wavelength.size == 1:
        return (slice(0, 1),)
    if not np.all(np.isfinite(wavelength)):
        raise ValueError("Wavelength coordinates must all be finite.")

    steps = np.abs(np.diff(wavelength))
    positive = steps[steps > 0.0]
    if positive.size == 0:
        raise ValueError("Wavelength coordinates must not all be identical.")
    typical_step = float(np.nanmedian(positive))
    gap_after = np.flatnonzero(steps > float(gap_factor) * typical_step)
    indices = np.arange(wavelength.size)
    return tuple(
        slice(int(group[0]), int(group[-1]) + 1)
        for group in np.split(indices, gap_after + 1)
        if group.size
    )


def _coordinate_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    midpoint = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - (midpoint[0] - centers[0])],
            midpoint,
            [centers[-1] + (centers[-1] - midpoint[-1])],
        )
    )


def pcolormesh_wavelength_segments(
    ax: Any,
    wavelength: np.ndarray,
    matrix: np.ndarray,
    *,
    y_edges: np.ndarray | None = None,
    gap_factor: float = 5.0,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Plot a matrix in exact wavelength coordinates without spanning gaps."""

    wavelength = np.asarray(wavelength, dtype=float)
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[1] != wavelength.size:
        raise ValueError("Matrix columns must match the 1-D wavelength array.")
    if y_edges is None:
        y_edges = np.arange(matrix.shape[0] + 1, dtype=float) - 0.5
    else:
        y_edges = np.asarray(y_edges, dtype=float)
    if y_edges.shape != (matrix.shape[0] + 1,):
        raise ValueError("y_edges must have exactly one more value than matrix rows.")

    meshes = []
    for segment in wavelength_segment_slices(wavelength, gap_factor=gap_factor):
        meshes.append(
            ax.pcolormesh(
                _coordinate_edges(wavelength[segment]),
                y_edges,
                matrix[:, segment],
                shading="flat",
                **kwargs,
            )
        )
    return tuple(meshes)


def plot_wavelength_segments(
    ax: Any,
    wavelength: np.ndarray,
    values: np.ndarray,
    *args: Any,
    gap_factor: float = 5.0,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Plot a 1-D wavelength series without drawing lines across gaps."""

    wavelength = np.asarray(wavelength, dtype=float)
    values = np.asarray(values)
    if values.ndim != 1 or values.shape != wavelength.shape:
        raise ValueError("Line values must match the 1-D wavelength array.")

    lines = []
    label = kwargs.pop("label", None)
    for index, segment in enumerate(wavelength_segment_slices(wavelength, gap_factor=gap_factor)):
        segment_kwargs = dict(kwargs)
        if label is not None:
            segment_kwargs["label"] = label if index == 0 else "_nolegend_"
        lines.extend(ax.plot(wavelength[segment], values[segment], *args, **segment_kwargs))
    return tuple(lines)


def fill_between_wavelength_segments(
    ax: Any,
    wavelength: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *args: Any,
    gap_factor: float = 5.0,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Fill a wavelength envelope independently within each contiguous segment."""

    wavelength = np.asarray(wavelength, dtype=float)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if lower.ndim != 1 or lower.shape != wavelength.shape:
        raise ValueError("Lower envelope must match the 1-D wavelength array.")
    if upper.ndim != 1 or upper.shape != wavelength.shape:
        raise ValueError("Upper envelope must match the 1-D wavelength array.")

    collections = []
    label = kwargs.pop("label", None)
    for index, segment in enumerate(wavelength_segment_slices(wavelength, gap_factor=gap_factor)):
        segment_kwargs = dict(kwargs)
        if label is not None:
            segment_kwargs["label"] = label if index == 0 else "_nolegend_"
        collections.append(
            ax.fill_between(
                wavelength[segment],
                lower[segment],
                upper[segment],
                *args,
                **segment_kwargs,
            )
        )
    return tuple(collections)
