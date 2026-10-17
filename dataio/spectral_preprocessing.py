"""Shared spectral residual construction, wavelength masks, and SYSREM preprocessing."""

from __future__ import annotations

import numpy as np
import config
from config import TELLURIC_REGIONS


def subtract_median_spectrum(
    flux: np.ndarray,
    error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the time-median spectrum and propagate its uncertainty.

    flux and error have shape (n_spectra, npix). Return residual flux,
    propagated errors (both the input shape), and median flux (npix,).
    The median removes stellar lines and time-invariant tellurics.
    """
    n_spectra = flux.shape[0]
    npix = flux.shape[1]

    median_flux = np.median(flux, axis=0)

    # Median error estimation from https://mathworld.wolfram.com/StatisticalMedian.html
    # For large n, var(median) ≈ pi/(2n) * var(mean)
    little_n = (n_spectra - 1) / 2
    correction_factor = np.sqrt(4 * little_n / (np.pi * n_spectra))
    median_error = np.sqrt(np.sum(error**2, axis=0)) / n_spectra / correction_factor

    residual_flux = flux - median_flux[np.newaxis, :]

    # Propagate errors: sigma_residual^2 = sigma_flux^2 + sigma_median^2
    residual_error = np.sqrt(error**2 + median_error[np.newaxis, :]**2)

    return residual_flux, residual_error, median_flux


def get_sysrem_chunk_indices(
    wave: np.ndarray,
    arm: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Partition wavelength columns into SYSREM chunks.

    The current scheme is:
    - ``red``: two chunks, non-telluric and telluric
    - ``blue``: one chunk, non-telluric only

    ``arm='full'`` is not supported: run SYSREM independently per arm.
    """

    wave = np.asarray(wave, dtype=float)

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})

    telluric_mask = np.zeros(wave.shape[0], dtype=bool)
    for wmin, wmax in telluric_config["telluric"]:
        telluric_mask |= (wave > wmin) & (wave <= wmax)

    no_tellurics = np.where(~telluric_mask)[0]
    if arm == "red":
        has_tellurics = np.where(telluric_mask)[0]
        return ("non_telluric", "telluric"), (no_tellurics, has_tellurics), telluric_mask

    return ("non_telluric",), (no_tellurics,), telluric_mask


def get_sysrem_deep_mask(wave: np.ndarray, arm: str) -> np.ndarray:
    """Return wavelengths that should be ignored after Molecfit correction."""
    wave = np.asarray(wave, dtype=float)

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})

    deep_mask = np.zeros(wave.shape[0], dtype=bool)
    for wmin, wmax in telluric_config["deep_mask"]:
        deep_mask |= (wave >= wmin) & (wave < wmax)
    return deep_mask


def get_telluric_edge_mask(
    wave: np.ndarray,
    arm: str,
    *,
    edge_width_angstrom: float = config.DEFAULT_TELLURIC_EDGE_MASK_WIDTH_A,
) -> np.ndarray:
    """Return columns close to configured telluric/deep-mask boundaries."""
    wave = np.asarray(wave, dtype=float)

    edge_width_angstrom = float(edge_width_angstrom)
    edge_mask = np.zeros(wave.shape[0], dtype=bool)
    if edge_width_angstrom <= 0.0:
        return edge_mask

    telluric_config = TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})
    boundaries: list[float] = []
    for key in ("telluric", "deep_mask"):
        for wmin, wmax in telluric_config[key]:
            boundaries.extend((float(wmin), float(wmax)))
    for boundary in sorted(set(boundaries)):
        edge_mask |= np.abs(wave - boundary) <= edge_width_angstrom
    return edge_mask


def get_arm_edge_trim_mask(
    wave: np.ndarray,
    arm: str,
    *,
    explicit_widths_A: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return columns selected by exact, explicitly supplied trim widths."""
    wave = np.asarray(wave, dtype=float)

    finite = np.isfinite(wave)
    edge_mask = np.zeros(wave.shape[0], dtype=bool)
    if not np.any(finite) or explicit_widths_A is None:
        return edge_mask

    left_width, right_width = tuple(float(value) for value in explicit_widths_A)
    if left_width <= 0.0 and right_width <= 0.0:
        return edge_mask

    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    if left_width > 0.0:
        edge_mask |= finite & (wave < lo + left_width)
    if right_width > 0.0:
        edge_mask |= finite & (wave > hi - right_width)
    return edge_mask


def arm_edge_trim_metadata(
    wave: np.ndarray,
    arm: str,
    *,
    planet: str | None = None,
    mode: str | None = None,
    epoch: str | None = None,
    explicit_widths_A: tuple[float, float] | None = None,
    source: str | None = None,
) -> dict[str, float | int | str]:
    """Describe the configured arm-edge trim for a wavelength grid."""
    wave = np.asarray(wave, dtype=float)
    finite = np.isfinite(wave)
    widths = (
        None
        if explicit_widths_A is None
        else tuple(float(value) for value in explicit_widths_A)
    )
    left_width, right_width = (0.0, 0.0) if widths is None else widths
    if not np.any(finite):
        return {
            "left_trim_A": left_width,
            "right_trim_A": right_width,
            "planet": planet or "",
            "mode": mode or "",
            "epoch": epoch or "",
            "raw_min_A": float("nan"),
            "raw_max_A": float("nan"),
            "keep_min_A": float("nan"),
            "keep_max_A": float("nan"),
            "n_trimmed_columns": 0,
            "source": source or "none",
        }

    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    trim_mask = get_arm_edge_trim_mask(wave, arm, explicit_widths_A=explicit_widths_A)
    return {
        "left_trim_A": left_width,
        "right_trim_A": right_width,
        "planet": planet or "",
        "mode": mode or "",
        "epoch": epoch or "",
        "raw_min_A": lo,
        "raw_max_A": hi,
        "keep_min_A": lo + left_width,
        "keep_max_A": hi - right_width,
        "n_trimmed_columns": int(np.count_nonzero(trim_mask)),
        "source": source or "none",
    }


def get_sysrem_ignore_mask(
    wave: np.ndarray,
    arm: str,
    *,
    explicit_edge_trim_widths_A: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return data-quality columns to ignore in SYSREM output."""
    return (
        get_sysrem_deep_mask(wave, arm)
        | get_telluric_edge_mask(wave, arm)
        | get_arm_edge_trim_mask(
            wave,
            arm,
            explicit_widths_A=explicit_edge_trim_widths_A,
        )
    )


def get_sysrem_max_systematics(arm: str) -> list[int]:
    if arm == "red":
        return list(config.DEFAULT_SYSREM_MAX_SYSTEMATICS_RED)
    return list(config.DEFAULT_SYSREM_MAX_SYSTEMATICS_OTHER)


def get_sysrem_min_systematics(arm: str) -> list[int]:
    if arm == "red":
        return list(config.DEFAULT_SYSREM_MIN_SYSTEMATICS_RED)
    return list(config.DEFAULT_SYSREM_MIN_SYSTEMATICS_OTHER)


def do_sysrem(
    wave: np.ndarray,
    residual_flux: np.ndarray,
    residual_error: np.ndarray,
    arm: str,
    airmass: np.ndarray,
    niter: int = 10,
    do_molecfit: bool = True,
    stop_delta_stddev: float = config.DEFAULT_SYSREM_STOP_TOL,
    return_diagnostics: bool = False,
    planet_name: str | None = None,
    data_mode: str | None = None,
    observation_epoch: str | None = None,
    edge_trim_widths_A: tuple[float, float] | None = None,
) -> tuple[np.ndarray, ...]:
    """Run SYSREM with separate treatment for telluric and non-telluric regions.

    The first systematic is initialized with airmass (physically motivated for
    telluric residuals), subsequent systematics start with unity.

    Args:
        wave: Common wavelength grid, shape (npix,)
        residual_flux: Median-subtracted flux, shape (n_spectra, npix)
        residual_error: Uncertainties, shape (n_spectra, npix)
        arm: Spectrograph arm ('red' or 'blue')
        airmass: Per-exposure airmass values, shape (n_spectra,)
        niter: Number of iterations per systematic (default: 10)
        do_molecfit: If True, mask deep telluric regions (default: True)
        stop_delta_stddev: Minimum sigma improvement required to accept a component
            after the configured minimum component count has been reached
        return_diagnostics: If True, append per-component SYSREM attempt
            diagnostics to the returned tuple

    Returns:
        corrected_flux: Systematics-corrected flux, shape (n_spectra, npix)
        corrected_error: Propagated uncertainties, shape (n_spectra, npix)
        U_sysrem: Systematic vectors, shape (n_spectra, n_sys_used_max, n_chunks)
            Columns not used in a given chunk are NaN.
        no_tellurics: Indices of non-telluric pixels
    """
    n_spectra = residual_flux.shape[0]

    corrected_flux = residual_flux.copy()
    corrected_error = residual_error.copy()

    _, chunk_indices, _telluric_mask = get_sysrem_chunk_indices(wave, arm)
    ignore_mask = get_arm_edge_trim_mask(
        wave,
        arm,
        explicit_widths_A=edge_trim_widths_A,
    )
    if do_molecfit:
        ignore_mask |= get_sysrem_deep_mask(wave, arm) | get_telluric_edge_mask(wave, arm)
    if np.any(ignore_mask):
        corrected_flux[:, ignore_mask] = 0.0
        corrected_error[:, ignore_mask] = 1.0
        chunk_indices = tuple(
            np.asarray(indices, dtype=int)[~ignore_mask[np.asarray(indices, dtype=int)]]
            for indices in chunk_indices
        )

    no_tellurics = np.asarray(chunk_indices[0], dtype=int)
    chunks = len(chunk_indices)

    max_systematics = get_sysrem_max_systematics(arm)
    min_systematics = get_sysrem_min_systematics(arm)

    max_n_sys = max(max_systematics)
    U_sysrem = np.full((n_spectra, max_n_sys, chunks), np.nan, dtype=float)
    n_systematics_used = [0] * chunks
    sysrem_stddev_before = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_stddev_after = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_delta_stddev = np.full((max_n_sys, chunks), np.nan, dtype=float)
    sysrem_component_attempted = np.zeros((max_n_sys, chunks), dtype=bool)
    sysrem_component_accepted = np.zeros((max_n_sys, chunks), dtype=bool)

    for chunk in range(chunks):
        this_one = chunk_indices[chunk]
        if len(this_one) == 0:
            continue

        npixhere = len(this_one)
        n_sys_chunk = max_systematics[chunk]
        min_sys_chunk = min_systematics[chunk]

        stddev_prev = np.std(corrected_flux[:, this_one])

        for system in range(n_sys_chunk):
            c = np.zeros(npixhere)
            sigma_c = np.zeros(npixhere)
            sigma_a = np.zeros(n_spectra)

            # Initialize: first systematic uses airmass, others use unity
            if system == 0:
                a = np.array(airmass, dtype=float)
            else:
                a = np.ones(n_spectra)

            for iteration in range(niter):
                # Minimize c for each pixel
                for s in range(npixhere):
                    pix_idx = this_one[s]
                    err_squared = corrected_error[:, pix_idx]**2

                    numerator = np.sum(a * corrected_flux[:, pix_idx] / err_squared)
                    denominator = np.sum(a**2 / err_squared)

                    # Error propagation
                    abs_a = np.abs(a)
                    saoa = np.divide(
                        sigma_a,
                        abs_a,
                        out=np.zeros_like(sigma_a),
                        where=abs_a != 0.0,
                    )
                    abs_flux = np.abs(corrected_flux[:, pix_idx])
                    eof = np.divide(
                        corrected_error[:, pix_idx],
                        abs_flux,
                        out=np.zeros_like(corrected_error[:, pix_idx]),
                        where=abs_flux != 0.0,
                    )

                    sigma_1 = np.abs(a * corrected_flux[:, pix_idx] / err_squared) * np.sqrt(saoa**2 + eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))

                    sigma_2 = np.sqrt(2.0) * np.abs(a) * sigma_a / err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))

                    if denominator != 0:
                        c[s] = numerator / denominator
                        if numerator != 0 and sigma_denominator >= 0:
                            sigma_c[s] = np.abs(c[s]) * np.sqrt(
                                (sigma_numerator / np.abs(numerator))**2 +
                                (sigma_denominator / np.abs(denominator))**2
                            )
                    else:
                        c[s] = 0.0
                        sigma_c[s] = 0.0

                # Using c, minimize a for each epoch
                for ep in range(n_spectra):
                    pix_indices = this_one
                    err_squared = corrected_error[ep, pix_indices]**2

                    numerator = np.sum(c * corrected_flux[ep, pix_indices] / err_squared)
                    denominator = np.sum(c**2 / err_squared)

                    # Error propagation
                    abs_c = np.abs(c)
                    scoc = np.divide(
                        sigma_c,
                        abs_c,
                        out=np.zeros_like(sigma_c),
                        where=abs_c != 0.0,
                    )
                    abs_flux = np.abs(corrected_flux[ep, pix_indices])
                    eof = np.divide(
                        corrected_error[ep, pix_indices],
                        abs_flux,
                        out=np.zeros_like(corrected_error[ep, pix_indices]),
                        where=abs_flux != 0.0,
                    )

                    sigma_1 = np.abs(c * corrected_flux[ep, pix_indices] / err_squared) * np.sqrt(scoc**2 + eof**2)
                    sigma_numerator = np.sqrt(np.sum(sigma_1**2))

                    sigma_2 = np.sqrt(2.0) * np.abs(c) * sigma_c / err_squared
                    sigma_denominator = np.sqrt(np.sum(sigma_2**2))

                    if denominator != 0:
                        a[ep] = numerator / denominator
                        if numerator != 0 and sigma_denominator >= 0:
                            sigma_a[ep] = np.abs(a[ep]) * np.sqrt(
                                (sigma_numerator / np.abs(numerator))**2 +
                                (sigma_denominator / np.abs(denominator))**2
                            )
                    else:
                        a[ep] = 0.0
                        sigma_a[ep] = 0.0

            syserr = a[:, np.newaxis] * c[np.newaxis, :]

            # Error on systematic term
            abs_a = np.abs(a)
            abs_c = np.abs(c)
            ratio_a = np.divide(
                sigma_a,
                abs_a,
                out=np.zeros_like(sigma_a),
                where=abs_a != 0.0,
            )[:, np.newaxis]
            ratio_c = np.divide(
                sigma_c,
                abs_c,
                out=np.zeros_like(sigma_c),
                where=abs_c != 0.0,
            )[np.newaxis, :]
            sigma_syserr = np.abs(syserr) * np.sqrt(ratio_a**2 + ratio_c**2)

            # Stop when additional components stop improving scatter enough
            trial_flux = corrected_flux[:, this_one] - syserr
            stddev_next = np.std(trial_flux)
            delta_stddev = stddev_prev - stddev_next
            sysrem_stddev_before[system, chunk] = stddev_prev
            sysrem_stddev_after[system, chunk] = stddev_next
            sysrem_delta_stddev[system, chunk] = delta_stddev
            sysrem_component_attempted[system, chunk] = True
            if (
                n_systematics_used[chunk] >= min_sys_chunk
                and delta_stddev <= stop_delta_stddev
            ):
                print(
                    f"  SYSREM chunk {chunk + 1}/{chunks}, component {system + 1}: "
                    f"delta_stddev={delta_stddev:.3e} <= stop_tol={stop_delta_stddev:.1e}; rejected"
                )
                break

            # Accept and apply this systematic
            U_sysrem[:, system, chunk] = a
            corrected_flux[:, this_one] = trial_flux
            corrected_error[:, this_one] = np.sqrt(corrected_error[:, this_one]**2 + sigma_syserr**2)
            stddev_prev = stddev_next
            n_systematics_used[chunk] += 1
            sysrem_component_accepted[system, chunk] = True
            if n_systematics_used[chunk] <= min_sys_chunk:
                reason = "forced minimum"
            else:
                reason = f"delta_stddev > stop_tol={stop_delta_stddev:.1e}"
            print(
                f"  SYSREM chunk {chunk + 1}/{chunks}, component {system + 1}: "
                f"delta_stddev={delta_stddev:.3e}; accepted ({reason})"
            )

        print(
            f"SYSREM chunk {chunk + 1}/{chunks}: used {n_systematics_used[chunk]} "
            f"of min/max {min_sys_chunk}/{n_sys_chunk} systematics"
        )

    # Trim trailing all-NaN columns if discovery mode stopped early.
    used_max = max(n_systematics_used) if n_systematics_used else 0
    if used_max > 0:
        U_sysrem = U_sysrem[:, :used_max, :]
    else:
        U_sysrem = U_sysrem[:, :0, :]

    if return_diagnostics:
        diagnostics = {
            "sysrem_stddev_before": sysrem_stddev_before,
            "sysrem_stddev_after": sysrem_stddev_after,
            "sysrem_delta_stddev": sysrem_delta_stddev,
            "sysrem_component_attempted": sysrem_component_attempted,
            "sysrem_component_accepted": sysrem_component_accepted,
            "sysrem_min_systematics": np.asarray(min_systematics, dtype=int),
            "sysrem_max_systematics": np.asarray(max_systematics, dtype=int),
            "sysrem_stop_delta_stddev": np.asarray(stop_delta_stddev, dtype=float),
        }
        return corrected_flux, corrected_error, U_sysrem, no_tellurics, diagnostics

    return corrected_flux, corrected_error, U_sysrem, no_tellurics


def build_out_of_transit_residuals(
    flux: np.ndarray,
    error: np.ndarray,
    out_transit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide every exposure by a shared out-of-transit master and subtract one."""
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    out_transit = np.asarray(out_transit, dtype=bool)
    n_out = int(np.count_nonzero(out_transit))

    master_out = np.median(flux[out_transit], axis=0)
    # Large-sample variance of the median is pi/2 times the variance of the
    # mean. This remains a diagonal approximation to the shared-master error.
    master_error = (
        np.sqrt(np.pi / 2.0)
        * np.sqrt(np.sum(error[out_transit] ** 2, axis=0))
        / n_out
    )
    valid_master = (
        np.isfinite(master_out)
        & (master_out != 0.0)
        & np.isfinite(master_error)
        & (master_error >= 0.0)
    )
    residual = np.full_like(flux, np.nan, dtype=float)
    residual_error = np.full_like(error, np.inf, dtype=float)
    residual[:, valid_master] = (
        flux[:, valid_master] / master_out[None, valid_master] - 1.0
    )
    residual_error[:, valid_master] = np.sqrt(
        (error[:, valid_master] / master_out[None, valid_master]) ** 2
        + (
            flux[:, valid_master]
            * master_error[None, valid_master]
            / master_out[None, valid_master] ** 2
        )
        ** 2
    )
    return residual, residual_error
