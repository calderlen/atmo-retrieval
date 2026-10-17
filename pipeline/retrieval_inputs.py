"""Load and align prepared observations and their frozen preprocessing operators."""

from __future__ import annotations

import json
import jax.numpy as jnp
import numpy as np
import config
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from exojax.utils.grids import wav2nu
from physics.observation_operators import (
    ChunkedSysremInputs,
    CollapsedEmissionInputs,
    CollapsedTransmissionInputs,
    FrozenTimeseriesInputs,
)


def _load_timeseries_metadata(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "timeseries_prep.json").read_text(encoding="utf-8"))


def _load_collapsed_emission_operator(
    data_dir: Path,
    target_wavelength: np.ndarray,
) -> CollapsedEmissionInputs:
    """Load and align the fixed preprocessing operator for a 1D emission product."""
    metadata_path = data_dir / "collapse_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    operator_name = metadata.get(
        "collapse_operator_file",
        "emission_collapse_operator.npz",
    )
    operator_path = data_dir / str(operator_name)

    with np.load(operator_path) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}

    source_wavelength = np.asarray(arrays["source_wavelength"], dtype=float)
    output_wavelength = np.asarray(arrays["output_wavelength"], dtype=float)
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    output_indices: list[int] = []
    for wavelength in target_wavelength:
        insertion = int(np.searchsorted(output_wavelength, wavelength))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < output_wavelength.size
        ]
        best = min(
            candidates,
            key=lambda index: abs(output_wavelength[index] - wavelength),
        )
        output_indices.append(best)

    # ExoJAX requires increasing wavenumber. Reversing the increasing-wavelength
    # source grid supplies that order; the model operator reverses it back before
    # applying the saved wavelength-space interpolation.
    source_inst_nus = wav2nu(source_wavelength[::-1], "AA")
    chunked_sysrem = None
    has_sysrem = bool(np.asarray(arrays.get("has_sysrem", False)).item())
    if has_sysrem:
        sysrem_bundle = _build_sysrem_inputs(
            {
                "U_sysrem": arrays["sysrem_U"],
                "chunk_labels": arrays["sysrem_chunk_labels"],
                "projection_sigma": arrays["sysrem_projection_sigma"],
            }
        )
        chunked_sysrem = _build_model_chunked_sysrem(sysrem_bundle)
    return CollapsedEmissionInputs(
        source_wavelength=jnp.asarray(source_wavelength),
        source_inst_nus=jnp.asarray(source_inst_nus),
        source_phase=jnp.asarray(arrays["source_phase"], dtype=float),
        selected_exposure_indices=jnp.asarray(
            arrays["selected_exposure_indices"],
            dtype=jnp.int32,
        ),
        shift_left_indices=jnp.asarray(
            arrays["shift_left_indices"],
            dtype=jnp.int32,
        ),
        shift_fractions=jnp.asarray(arrays["shift_fractions"], dtype=float),
        coadd_weights=jnp.asarray(arrays["coadd_weights"], dtype=float),
        bin_indices=jnp.asarray(arrays["bin_indices"], dtype=jnp.int32),
        bin_weights=jnp.asarray(arrays["bin_weights"], dtype=float),
        output_wavelength=jnp.asarray(output_wavelength),
        output_indices=jnp.asarray(output_indices, dtype=jnp.int32),
        kp_reference_kms=jnp.asarray(arrays["kp_reference_kms"], dtype=float),
        velocity_offset_reference_kms=jnp.asarray(
            arrays["velocity_offset_reference_kms"],
            dtype=float,
        ),
        chunked_sysrem=chunked_sysrem,
    )


def _load_collapsed_transmission_operator(
    data_dir: Path,
    target_wavelength: np.ndarray,
) -> CollapsedTransmissionInputs:
    """Load the frozen preprocessing operator for a 1D transmission product."""
    metadata_path = data_dir / "collapse_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    operator_path = data_dir / str(
        metadata.get(
            "collapse_operator_file",
            "transmission_collapse_operator.npz",
        )
    )
    with np.load(operator_path) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}

    source_wavelength = np.asarray(arrays["source_wavelength"], dtype=float)
    source_phase = np.asarray(arrays["source_phase"], dtype=float)
    fixed_source_model = np.asarray(arrays["fixed_source_model"], dtype=float)
    output_wavelength = np.asarray(arrays["output_wavelength"], dtype=float)
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    output_indices: list[int] = []
    for wavelength in target_wavelength:
        insertion = int(np.searchsorted(output_wavelength, wavelength))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < output_wavelength.size
        ]
        best = min(
            candidates,
            key=lambda index: abs(output_wavelength[index] - wavelength),
        )
        output_indices.append(best)

    chunked_sysrem = None
    has_sysrem = bool(np.asarray(arrays.get("has_sysrem", False)).item())
    if has_sysrem:
        chunked_sysrem = _build_model_chunked_sysrem(
            _build_sysrem_inputs(
                {
                    "U_sysrem": arrays["sysrem_U"],
                    "chunk_labels": arrays["sysrem_chunk_labels"],
                    "projection_sigma": arrays["sysrem_projection_sigma"],
                }
            )
        )

    return CollapsedTransmissionInputs(
        source_wavelength=jnp.asarray(source_wavelength),
        source_inst_nus=jnp.asarray(
            wav2nu(source_wavelength[::-1], "AA")
        ),
        source_phase=jnp.asarray(source_phase, dtype=float),
        fixed_source_model=jnp.asarray(fixed_source_model, dtype=float),
        active_exposure_mask=jnp.asarray(
            arrays["active_exposure_mask"],
            dtype=float,
        ),
        selected_exposure_indices=jnp.asarray(
            arrays["selected_exposure_indices"],
            dtype=jnp.int32,
        ),
        shift_left_indices=jnp.asarray(
            arrays["shift_left_indices"],
            dtype=jnp.int32,
        ),
        shift_fractions=jnp.asarray(
            arrays["shift_fractions"],
            dtype=float,
        ),
        coadd_weights=jnp.asarray(arrays["coadd_weights"], dtype=float),
        bin_indices=jnp.asarray(arrays["bin_indices"], dtype=jnp.int32),
        bin_weights=jnp.asarray(arrays["bin_weights"], dtype=float),
        output_wavelength=jnp.asarray(output_wavelength),
        output_indices=jnp.asarray(output_indices, dtype=jnp.int32),
        kp_reference_kms=jnp.asarray(
            arrays["kp_reference_kms"],
            dtype=float,
        ),
        velocity_offset_reference_kms=jnp.asarray(
            arrays["velocity_offset_reference_kms"],
            dtype=float,
        ),
        chunked_sysrem=chunked_sysrem,
    )


def load_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    return (
        np.load(data_dir / "wavelength.npy"),
        np.load(data_dir / "data.npy"),
        np.load(data_dir / "sigma.npy"),
        np.load(data_dir / "phase.npy"),
    )


def load_pre_sysrem_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    data_dir = Path(data_dir)
    data_path = data_dir / "pre_sysrem_data.npy"
    sigma_path = data_dir / "pre_sysrem_sigma.npy"

    if not data_path.exists() and not sigma_path.exists():
        return None

    return np.load(data_path), np.load(sigma_path)


@dataclass(frozen=True)
class SysremInputBundle:
    U: np.ndarray | None = None
    chunk_indices: tuple[np.ndarray, ...] | None = None
    U_chunks: tuple[np.ndarray, ...] | None = None
    projection_sigma: np.ndarray | None = None

    @property
    def is_chunked(self) -> bool:
        return self.chunk_indices is not None


@dataclass(frozen=True)
class FrozenTimeseriesOperatorSpec:
    source_wavelength: np.ndarray
    source_phase: np.ndarray
    source_bjd_tdb: np.ndarray
    active_exposure_mask: np.ndarray
    selected_exposure_indices: np.ndarray
    subtract_time_median: bool
    has_sysrem: bool
    fixed_source_model: np.ndarray | None = None


def _load_frozen_timeseries_operator_spec(
    data_dir: str | Path,
) -> FrozenTimeseriesOperatorSpec:
    data_dir = Path(data_dir)
    metadata = _load_timeseries_metadata(data_dir)
    operator_path = data_dir / str(
        metadata.get("timeseries_operator_file", "timeseries_operator.npz")
    )
    with np.load(operator_path) as raw:
        source_wavelength = np.asarray(raw["source_wavelength"], dtype=float)
        source_phase = _normalize_phase(np.asarray(raw["source_phase"], dtype=float))
        source_bjd_tdb = np.asarray(raw["source_bjd_tdb"], dtype=float)
        active_mask = np.asarray(raw["active_exposure_mask"], dtype=float)
        selected_indices = np.asarray(raw["selected_exposure_indices"], dtype=int)
        subtract_time_median = bool(np.asarray(raw["subtract_time_median"]).item())
        has_sysrem = bool(np.asarray(raw["has_sysrem"]).item())

    fixed_source_model = None
    shadow_metadata = metadata.get("fixed_doppler_shadow")
    if shadow_metadata and shadow_metadata.get("enabled", False):
        source_model_path = data_dir / str(
            shadow_metadata.get("source_model_file", "shadow_source_model.npy")
        )
        fixed_source_model = np.asarray(np.load(source_model_path), dtype=float)

    return FrozenTimeseriesOperatorSpec(
        source_wavelength=source_wavelength,
        source_phase=source_phase,
        source_bjd_tdb=source_bjd_tdb,
        active_exposure_mask=active_mask,
        selected_exposure_indices=selected_indices,
        subtract_time_median=subtract_time_median,
        has_sysrem=has_sysrem,
        fixed_source_model=fixed_source_model,
    )


def _remap_frozen_timeseries_wavelengths(
    operator: FrozenTimeseriesOperatorSpec | None,
    indices: np.ndarray | None,
) -> FrozenTimeseriesOperatorSpec | None:
    """Apply the same wavelength selection or sort to a fixed shadow cube."""
    if operator is None or indices is None:
        return operator
    indices = np.asarray(indices, dtype=int)
    source_wavelength = np.asarray(operator.source_wavelength)[indices]
    fixed_source_model = (
        None
        if operator.fixed_source_model is None
        else np.asarray(operator.fixed_source_model)[:, indices]
    )
    return replace(
        operator,
        source_wavelength=source_wavelength,
        fixed_source_model=fixed_source_model,
    )


def _build_model_frozen_timeseries(
    operator: FrozenTimeseriesOperatorSpec | None,
    sysrem: SysremInputBundle | None,
) -> FrozenTimeseriesInputs | None:
    if operator is None:
        return None
    return FrozenTimeseriesInputs(
        source_phase=jnp.asarray(operator.source_phase),
        active_exposure_mask=jnp.asarray(operator.active_exposure_mask),
        selected_exposure_indices=jnp.asarray(
            operator.selected_exposure_indices,
            dtype=jnp.int32,
        ),
        subtract_time_median=operator.subtract_time_median,
        chunked_sysrem=_build_model_chunked_sysrem(sysrem),
        fixed_source_model=(
            None
            if operator.fixed_source_model is None
            else jnp.asarray(operator.fixed_source_model)
        ),
    )


def _chunk_indices_from_labels(chunk_labels: np.ndarray) -> tuple[np.ndarray, ...]:
    chunk_labels = np.asarray(chunk_labels, dtype=int)
    return tuple(np.where(chunk_labels == label)[0] for label in np.unique(chunk_labels))


def _load_sysrem_inputs(data_dir: str | Path) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    u_candidates = [
        data_dir / "U_sysrem.npz",
        data_dir / "U.npy",
        data_dir / "U_sysrem.npy",
    ]
    u_path = next((path for path in u_candidates if path.exists()), None)
    if u_path is None:
        raise FileNotFoundError(f"No SYSREM basis file found in {data_dir}.")
    if u_path.suffix == ".npz":
        with np.load(u_path) as data:
            return {
                name: np.asarray(data[name])
                for name in data.files
                if name != "V"
            }
    return {"U": np.load(u_path)}


def _build_sysrem_inputs(raw: dict[str, np.ndarray]) -> SysremInputBundle:
    U = np.asarray(raw["U_sysrem"] if "U_sysrem" in raw else raw["U"])
    if "chunk_labels" in raw:
        if U.ndim == 2:
            U = U[:, :, np.newaxis]
        chunk_indices = _chunk_indices_from_labels(raw["chunk_labels"])
        U_chunks = []
        for chunk in range(len(chunk_indices)):
            U_chunk = np.asarray(U[:, :, chunk], dtype=float)
            keep = np.any(np.isfinite(U_chunk), axis=0)
            U_chunks.append(U_chunk[:, keep])
        return SysremInputBundle(
            chunk_indices=chunk_indices,
            U_chunks=tuple(U_chunks),
            projection_sigma=np.asarray(raw["projection_sigma"], dtype=float),
        )
    return SysremInputBundle(U=U)


def _build_model_chunked_sysrem(
    sysrem: SysremInputBundle | None,
) -> ChunkedSysremInputs | None:
    if sysrem is None or not sysrem.is_chunked:
        return None

    return ChunkedSysremInputs(
        chunk_indices=tuple(jnp.asarray(indices, dtype=jnp.int32) for indices in sysrem.chunk_indices),
        U_chunks=tuple(jnp.asarray(U_chunk) for U_chunk in sysrem.U_chunks),
        sigma_chunks=tuple(
            jnp.asarray(
                sysrem.projection_sigma[:, np.asarray(indices, dtype=int)]
            )
            for indices in sysrem.chunk_indices
        ),
    )


def _describe_sysrem_inputs(sysrem: SysremInputBundle) -> str:
    if sysrem.is_chunked:
        chunk_sizes = [int(indices.size) for indices in sysrem.chunk_indices]
        basis_counts = [int(U_chunk.shape[1]) for U_chunk in sysrem.U_chunks]
        return (
            f"chunked SYSREM: {len(chunk_sizes)} chunks, "
            f"chunk_sizes={chunk_sizes}, basis_counts={basis_counts}, "
            f"per_pixel_sigma_shape={sysrem.projection_sigma.shape}"
        )

    return f"U shape={sysrem.U.shape}, per-pixel sigma from observation"


def _subset_sysrem_inputs(
    sysrem: SysremInputBundle | None,
    indices: np.ndarray,
) -> SysremInputBundle | None:
    if sysrem is None:
        return None

    indices = np.asarray(indices, dtype=int)
    if sysrem.is_chunked:
        return SysremInputBundle(
            chunk_indices=tuple(np.asarray(chunk_indices, dtype=int) for chunk_indices in sysrem.chunk_indices),
            U_chunks=tuple(np.asarray(U_chunk)[indices] for U_chunk in sysrem.U_chunks),
            projection_sigma=np.asarray(sysrem.projection_sigma)[indices],
        )

    return SysremInputBundle(
        U=np.asarray(sysrem.U)[indices],
    )


def _spectral_subset_indices(
    n_wave: int,
    spectral_stride: int,
    spectral_offset: int,
) -> np.ndarray:
    stride, offset = int(spectral_stride), int(spectral_offset)
    indices = np.arange(offset, int(n_wave), stride, dtype=int)
    if indices.size < 2:
        raise ValueError(
            "Spectral thinning left fewer than two wavelength pixels; "
            f"n_wave={n_wave}, stride={stride}, offset={offset}."
        )
    return indices


def _take_spectral_axis(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr[indices]
    return arr[:, indices]


def _subset_sysrem_wavelengths(
    sysrem: SysremInputBundle | None,
    indices: np.ndarray,
    n_wave: int,
) -> SysremInputBundle | None:
    if sysrem is None or not sysrem.is_chunked:
        return sysrem

    selected = np.asarray(indices, dtype=int)
    old_to_new = np.full((int(n_wave),), -1, dtype=int)
    old_to_new[selected] = np.arange(selected.size, dtype=int)

    new_chunk_indices: list[np.ndarray] = []
    new_u_chunks: list[np.ndarray] = []
    for chunk_indices, U_chunk in zip(
        sysrem.chunk_indices or (),
        sysrem.U_chunks or (),
    ):
        remapped = old_to_new[np.asarray(chunk_indices, dtype=int)]
        remapped = remapped[remapped >= 0]
        if remapped.size == 0:
            continue
        new_chunk_indices.append(np.sort(remapped).astype(int))
        new_u_chunks.append(np.asarray(U_chunk))

    if not new_chunk_indices:
        raise ValueError("Spectral thinning removed all chunked SYSREM wavelength columns.")

    return SysremInputBundle(
        chunk_indices=tuple(new_chunk_indices),
        U_chunks=tuple(new_u_chunks),
        projection_sigma=np.asarray(sysrem.projection_sigma)[:, selected],
    )


def _remap_sysrem_wavelength_sort(
    sysrem: SysremInputBundle | None,
    sort_idx: np.ndarray | None,
) -> SysremInputBundle | None:
    if sysrem is None or not sysrem.is_chunked or sort_idx is None:
        return sysrem

    sort_idx = np.asarray(sort_idx, dtype=int)
    old_to_new = np.empty_like(sort_idx)
    old_to_new[sort_idx] = np.arange(sort_idx.size, dtype=int)
    return SysremInputBundle(
        chunk_indices=tuple(
            np.sort(old_to_new[np.asarray(chunk_indices, dtype=int)]).astype(int)
            for chunk_indices in (sysrem.chunk_indices or ())
        ),
        U_chunks=tuple(np.asarray(U_chunk) for U_chunk in (sysrem.U_chunks or ())),
        projection_sigma=np.asarray(sysrem.projection_sigma)[:, sort_idx],
    )


def _apply_spectral_thinning(
    *,
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    sysrem: SysremInputBundle | None,
    pre_sysrem_data: np.ndarray | None,
    pre_sysrem_sigma: np.ndarray | None,
    spectral_stride: int,
    spectral_offset: int,
    component_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SysremInputBundle | None, np.ndarray | None, np.ndarray | None]:
    stride, offset = int(spectral_stride), int(spectral_offset)
    if stride == 1 and offset == 0:
        return wav_obs, data, sigma, sysrem, pre_sysrem_data, pre_sysrem_sigma

    n_wave = int(np.asarray(wav_obs).size)
    indices = _spectral_subset_indices(n_wave, stride, offset)
    thinned_sysrem = _subset_sysrem_wavelengths(sysrem, indices, n_wave)
    thinned_pre_data = None if pre_sysrem_data is None else _take_spectral_axis(pre_sysrem_data, indices)
    thinned_pre_sigma = None if pre_sysrem_sigma is None else _take_spectral_axis(pre_sysrem_sigma, indices)
    print(
        f"  Applied spectral thinning to {component_name}: "
        f"kept {indices.size}/{n_wave} pixels "
        f"(stride={stride}, offset={offset})"
    )
    return (
        np.asarray(wav_obs)[indices],
        _take_spectral_axis(data, indices),
        _take_spectral_axis(sigma, indices),
        thinned_sysrem,
        thinned_pre_data,
        thinned_pre_sigma,
    )


def _normalize_phase(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase)
    if phase.size == 0:
        return phase

    phase_min = float(np.nanmin(phase))
    phase_max = float(np.nanmax(phase))

    if phase_min < -0.5 or phase_max > 0.5:
        phase = 0.5 - ((0.5 - phase) % 1.0)

    return phase


def _prepare_observed_spectrum_arrays(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)

    inst_nus = wav2nu(wav_obs, "AA")
    if inst_nus.size > 1 and np.any(np.diff(inst_nus) <= 0):
        sort_idx = np.argsort(inst_nus)
        inst_nus = inst_nus[sort_idx]
        wav_obs = wav_obs[sort_idx]
        if data.ndim == 2:
            data = data[:, sort_idx]
            sigma = sigma[:, sort_idx]
        else:
            data = data[sort_idx]
            sigma = sigma[sort_idx]

    return wav_obs, data, sigma, inst_nus


def _scale_spectroscopic_sigma(
    sigma: np.ndarray,
    sigma_scale: float,
) -> np.ndarray:
    if sigma_scale == 1.0:
        return np.asarray(sigma)
    return np.asarray(sigma, dtype=float) * sigma_scale


def _subtract_inverse_variance_weighted_constant(
    values: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Project one wavelength-independent offset out of a 1D spectrum."""
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    weights = 1.0 / np.clip(
        sigma,
        config.F32_FLOOR_RECIPSQ,
        None,
    ) ** 2
    weighted_mean = np.sum(weights * values) / np.clip(
        np.sum(weights),
        config.F32_FLOOR_RECIP,
        None,
    )
    return values - weighted_mean
