"""Replay frozen observational preprocessing on model exposure sequences.

Operator inputs and their JAX implementations live together here."""

from __future__ import annotations

from typing import NamedTuple
import jax.numpy as jnp
import config


class ChunkedSysremInputs(NamedTuple):
    chunk_indices: tuple[jnp.ndarray, ...]
    U_chunks: tuple[jnp.ndarray, ...]
    sigma_chunks: tuple[jnp.ndarray, ...]


class FrozenTimeseriesInputs(NamedTuple):
    """Full-exposure operator for an ordinary phase-selected time series."""

    source_phase: jnp.ndarray
    active_exposure_mask: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    subtract_time_median: bool
    chunked_sysrem: ChunkedSysremInputs | None = None
    fixed_source_model: jnp.ndarray | None = None


class CollapsedEmissionInputs(NamedTuple):
    """Fixed operator used to turn an emission time series into one 1D spectrum."""

    source_wavelength: jnp.ndarray
    source_inst_nus: jnp.ndarray
    source_phase: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    shift_left_indices: jnp.ndarray
    shift_fractions: jnp.ndarray
    coadd_weights: jnp.ndarray
    bin_indices: jnp.ndarray
    bin_weights: jnp.ndarray
    output_wavelength: jnp.ndarray
    output_indices: jnp.ndarray
    kp_reference_kms: jnp.ndarray
    velocity_offset_reference_kms: jnp.ndarray
    chunked_sysrem: ChunkedSysremInputs | None = None


class CollapsedTransmissionInputs(NamedTuple):
    """Fixed operator used to turn a transmission time series into one spectrum."""

    source_wavelength: jnp.ndarray
    source_inst_nus: jnp.ndarray
    source_phase: jnp.ndarray
    fixed_source_model: jnp.ndarray
    active_exposure_mask: jnp.ndarray
    selected_exposure_indices: jnp.ndarray
    shift_left_indices: jnp.ndarray
    shift_fractions: jnp.ndarray
    coadd_weights: jnp.ndarray
    bin_indices: jnp.ndarray
    bin_weights: jnp.ndarray
    output_wavelength: jnp.ndarray
    output_indices: jnp.ndarray
    kp_reference_kms: jnp.ndarray
    velocity_offset_reference_kms: jnp.ndarray
    chunked_sysrem: ChunkedSysremInputs | None = None


def sysrem_model_distortion(
    M: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the SYSREM-induced distortion to the model matrix.

    Parameters
    ----------
    M : jnp.ndarray
        Model matrix with shape (n_exposures, n_wavelengths).
    U : jnp.ndarray
        SYSREM basis matrix with shape (n_exposures, n_basis). A constant
        offset term is represented by a column of ones.
    V : jnp.ndarray
        Diagonal whitening matrix with entries 1 / sigma and shape
        (n_exposures, n_exposures).

    The filtered model is computed from the SYSREM projection

        M_fit = U ((V U)^T (V U))^-1 (V U)^T (V M)
        M_filtered = M - M_fit

    so the effective least-squares weights are V^T V = diag(1 / sigma^2).
    """
    if U.shape[1] == 0:
        return M

    weighted_basis = V @ U
    weighted_model = V @ M
    gram = weighted_basis.T @ weighted_basis
    rhs = weighted_basis.T @ weighted_model

    # The common phase-binned SYSREM case uses a single basis vector.
    # Avoid routing that 1x1 solve through cuSolver, which has been unstable
    # on some GPU setups despite the problem being just a scalar division.
    if U.shape[1] == 1:
        coeffs = rhs / jnp.clip(gram[0, 0], config.F32_FLOOR_RECIP, None)
    else:
        coeffs = jnp.linalg.solve(gram, rhs)

    M_fit = U @ coeffs
    return M - M_fit


def sysrem_model_distortion_per_pixel(
    M: jnp.ndarray,
    U: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a frozen SYSREM basis with exposure-by-wavelength uncertainties.

    Each wavelength column gets its own
    weighted least-squares solve. This preserves the pixel-level uncertainties
    used by preparation instead of reducing a whole wavelength chunk to one
    exposure weight.
    """
    M = jnp.asarray(M)
    U = jnp.asarray(U)
    sigma = jnp.asarray(sigma)
    if U.shape[1] == 0:
        return M

    weights = 1.0 / sigma**2
    if U.shape[1] == 1:
        basis = U[:, 0][:, None]
        denominator = jnp.sum(weights * basis**2, axis=0)
        numerator = jnp.sum(weights * basis * M, axis=0)
        coeffs = numerator / denominator
        return M - basis * coeffs[None, :]

    gram = jnp.einsum("eb,ew,ec->wbc", U, weights, U)
    rhs = jnp.einsum("eb,ew,ew->wb", U, weights, M)
    coeffs = jnp.linalg.solve(gram, rhs[..., None])[..., 0]
    fitted = jnp.einsum("eb,wb->ew", U, coeffs)
    return M - fitted


def sysrem_model_distortion_chunked(
    M: jnp.ndarray,
    chunked_sysrem: ChunkedSysremInputs,
) -> jnp.ndarray:
    corrected = M
    for chunk_indices, U_chunk, sigma_chunk in zip(
        chunked_sysrem.chunk_indices,
        chunked_sysrem.U_chunks,
        chunked_sysrem.sigma_chunks,
    ):
        if chunk_indices.shape[0] == 0 or U_chunk.shape[1] == 0:
            continue
        corrected_chunk = sysrem_model_distortion_per_pixel(
            M[:, chunk_indices],
            U_chunk,
            sigma_chunk,
        )
        corrected = corrected.at[:, chunk_indices].set(corrected_chunk)
    return corrected


def apply_collapsed_emission_operator(
    model_ts: jnp.ndarray,
    operator: CollapsedEmissionInputs,
) -> jnp.ndarray:
    """Apply the data's time-median subtraction and 1D collapse to a model."""
    model_ts = jnp.asarray(model_ts)

    # ExoJAX samples on increasing wavenumber, which is decreasing wavelength.
    # The saved collapse operator uses increasing wavelength, matching NumPy's
    # interpolation convention in the data collapser.
    model_wavelength_order = model_ts[:, ::-1]
    residual_ts = model_wavelength_order - jnp.median(
        model_wavelength_order,
        axis=0,
        keepdims=True,
    )
    if operator.chunked_sysrem is not None:
        residual_ts = sysrem_model_distortion_chunked(
            residual_ts,
            operator.chunked_sysrem,
        )
    return _coadd_model_timeseries(residual_ts, operator)


def apply_collapsed_transmission_operator(
    model_ts: jnp.ndarray,
    operator: CollapsedTransmissionInputs,
) -> jnp.ndarray:
    """Apply the data's frozen SYSREM and 1D transmission collapse."""
    model_ts = jnp.asarray(model_ts)

    fixed_source_model = jnp.asarray(operator.fixed_source_model)
    model_wavelength_order = model_ts[:, ::-1]
    model_wavelength_order = model_wavelength_order + fixed_source_model
    residual_ts = (
        model_wavelength_order
        * operator.active_exposure_mask[:, None]
    )
    if operator.chunked_sysrem is not None:
        residual_ts = sysrem_model_distortion_chunked(
            residual_ts,
            operator.chunked_sysrem,
        )
    return _coadd_model_timeseries(residual_ts, operator)


def _coadd_model_timeseries(
    residual_ts: jnp.ndarray,
    operator: CollapsedEmissionInputs | CollapsedTransmissionInputs,
) -> jnp.ndarray:
    """Replay the saved planet-frame interpolation, weighted coadd, and binning."""
    selected = residual_ts[operator.selected_exposure_indices]
    exposure_indices = jnp.arange(selected.shape[0])[:, None]
    left = selected[exposure_indices, operator.shift_left_indices]
    right = selected[exposure_indices, operator.shift_left_indices + 1]
    shifted = left + operator.shift_fractions * (right - left)
    spectrum_unbinned = jnp.sum(operator.coadd_weights * shifted, axis=0)
    spectrum_binned = jnp.zeros_like(operator.output_wavelength).at[
        operator.bin_indices
    ].add(operator.bin_weights * spectrum_unbinned)
    return spectrum_binned[operator.output_indices][None, :]


def apply_frozen_timeseries_operator(
    model_ts: jnp.ndarray,
    operator: FrozenTimeseriesInputs,
) -> jnp.ndarray:
    """Replay full-exposure preprocessing before selecting likelihood rows."""
    model_ts = jnp.asarray(model_ts)

    if operator.fixed_source_model is not None:
        fixed_source_model = jnp.asarray(operator.fixed_source_model)
        model_ts = model_ts + fixed_source_model

    processed = model_ts * operator.active_exposure_mask[:, None]
    median_subtracted = processed - jnp.median(
        processed,
        axis=0,
        keepdims=True,
    )
    processed = jnp.where(
        jnp.asarray(operator.subtract_time_median),
        median_subtracted,
        processed,
    )
    if operator.chunked_sysrem is not None:
        processed = sysrem_model_distortion_chunked(
            processed,
            operator.chunked_sysrem,
        )
    return processed[operator.selected_exposure_indices]


def apply_model_pipeline_corrections(
    model_ts: jnp.ndarray,
    *,
    subtract_weighted_global_mean: bool = False,
    apply_sysrem: bool,
    sigma: jnp.ndarray | None = None,
    U: jnp.ndarray | None = None,
    chunked_sysrem: ChunkedSysremInputs | None = None,
) -> jnp.ndarray:
    if model_ts.ndim == 1:
        model_ts = model_ts[None, :]

    if subtract_weighted_global_mean:
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            sigma = sigma[None, :]
        weights = 1.0 / sigma**2
        weighted_mean = jnp.sum(weights * model_ts) / jnp.sum(weights)
        model_ts = model_ts - weighted_mean

    if apply_sysrem:
        if chunked_sysrem is not None:
            model_ts = sysrem_model_distortion_chunked(model_ts, chunked_sysrem)
        else:
            model_ts = sysrem_model_distortion_per_pixel(
                model_ts,
                U,
                sigma,
            )

    return model_ts
