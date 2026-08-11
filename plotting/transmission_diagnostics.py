"""Shared, notebook-facing plots for prepared transmission time series.

The transition catalog is deliberately a compact set of representative optical
lines, not a retrieval line list.  Original NIST air wavelengths are retained
for provenance and converted once to vacuum Angstroms.  All coverage, sampling,
and velocity calculations use ``rest_vacuum_A``.  Every plot checks the
wavelength samples in the active bundle, so a transition is skipped rather than
assigned to an arm by assumption.

Planet-frame stacks sample each exposure directly from its native wavelength
grid with the same relativistic orbital-Doppler convention used by the
persisted collapse operator.  The displayed output window is therefore never
used as an observed-frame pre-crop.  These helpers only read the arrays passed
by the diagnostics notebooks; they do not alter prepared products or
participate in the retrieval likelihood.
"""

from __future__ import annotations

import math

from plotting.style import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from exojax.database.core_atom.io import air_to_vac

from dataio.orbital_velocity import planet_radial_velocity_kms
from plotting.wavelength import pcolormesh_wavelength_segments, plot_wavelength_segments


C_KMS = 299792.458

# Representative strong/useful lines inside the PEPSI blue and red ranges used
# by this repository.  Some emission epochs use the bluer 4220--4780 A setup,
# while the other prepared blue bundles start near 4750 A.  Sources:
# https://physics.nist.gov/PhysRefData/Handbook/periodictable.htm
# https://physics.nist.gov/PhysRefData/ASD/lines_form.html
_AIR_TRANSMISSION_LINE_CATALOG = (
    # Bluer setup: approximately 4220--4780 A in several emission bundles.
    {"label": "H gamma", "species": "H I", "rest_air_A": 4340.472, "arm": "blue"},
    {"label": "Mg I 4352", "species": "Mg I", "rest_air_A": 4351.906, "arm": "blue"},
    {"label": "Mg I 4571", "species": "Mg I", "rest_air_A": 4571.096, "arm": "blue"},
    {"label": "Mg II 4391", "species": "Mg II", "rest_air_A": 4390.572, "arm": "blue"},
    {"label": "Mg II 4481.1", "species": "Mg II", "rest_air_A": 4481.126, "arm": "blue"},
    {"label": "Mg II 4481.3", "species": "Mg II", "rest_air_A": 4481.325, "arm": "blue"},
    {"label": "Fe I 4384", "species": "Fe I", "rest_air_A": 4383.545, "arm": "blue"},
    {"label": "Fe I 4405", "species": "Fe I", "rest_air_A": 4404.750, "arm": "blue"},
    {"label": "Fe I 4427", "species": "Fe I", "rest_air_A": 4427.310, "arm": "blue"},
    {"label": "Fe I 4495", "species": "Fe I", "rest_air_A": 4494.563, "arm": "blue"},
    {"label": "Fe II 4233", "species": "Fe II", "rest_air_A": 4233.163, "arm": "blue"},
    {"label": "Fe II 4508", "species": "Fe II", "rest_air_A": 4508.281, "arm": "blue"},
    {"label": "Fe II 4515", "species": "Fe II", "rest_air_A": 4515.334, "arm": "blue"},
    {"label": "Fe II 4520", "species": "Fe II", "rest_air_A": 4520.221, "arm": "blue"},
    {"label": "Fe II 4549", "species": "Fe II", "rest_air_A": 4549.467, "arm": "blue"},
    {"label": "Fe II 4584", "species": "Fe II", "rest_air_A": 4583.829, "arm": "blue"},
    {"label": "Fe II 4629", "species": "Fe II", "rest_air_A": 4629.332, "arm": "blue"},
    {"label": "Ti I 4427", "species": "Ti I", "rest_air_A": 4427.098, "arm": "blue"},
    {"label": "Ti I 4449", "species": "Ti I", "rest_air_A": 4449.143, "arm": "blue"},
    {"label": "Ti I 4453", "species": "Ti I", "rest_air_A": 4453.312, "arm": "blue"},
    {"label": "Ti I 4457", "species": "Ti I", "rest_air_A": 4457.426, "arm": "blue"},
    {"label": "Ti I 4513", "species": "Ti I", "rest_air_A": 4512.733, "arm": "blue"},
    {"label": "Ti I 4518", "species": "Ti I", "rest_air_A": 4518.021, "arm": "blue"},
    {"label": "Ti I 4533", "species": "Ti I", "rest_air_A": 4533.239, "arm": "blue"},
    {"label": "Ti II 4300", "species": "Ti II", "rest_air_A": 4300.042, "arm": "blue"},
    {"label": "Ti II 4395", "species": "Ti II", "rest_air_A": 4395.031, "arm": "blue"},
    {"label": "Ti II 4444", "species": "Ti II", "rest_air_A": 4443.801, "arm": "blue"},
    {"label": "Ti II 4468", "species": "Ti II", "rest_air_A": 4468.492, "arm": "blue"},
    {"label": "Ti II 4501", "species": "Ti II", "rest_air_A": 4501.270, "arm": "blue"},
    {"label": "Ti II 4564", "species": "Ti II", "rest_air_A": 4563.758, "arm": "blue"},
    {"label": "Ti II 4572", "species": "Ti II", "rest_air_A": 4571.971, "arm": "blue"},
    {"label": "Cr I 4254", "species": "Cr I", "rest_air_A": 4254.331, "arm": "blue"},
    {"label": "Cr I 4275", "species": "Cr I", "rest_air_A": 4274.806, "arm": "blue"},
    {"label": "Cr I 4290", "species": "Cr I", "rest_air_A": 4289.733, "arm": "blue"},
    {"label": "Cr I 4646", "species": "Cr I", "rest_air_A": 4646.151, "arm": "blue"},
    {"label": "Cr II 4559", "species": "Cr II", "rest_air_A": 4558.644, "arm": "blue"},
    {"label": "Cr II 4588", "species": "Cr II", "rest_air_A": 4588.198, "arm": "blue"},
    {"label": "Cr II 4617", "species": "Cr II", "rest_air_A": 4616.624, "arm": "blue"},
    {"label": "Cr II 4634", "species": "Cr II", "rest_air_A": 4634.073, "arm": "blue"},
    # Standard blue setup: approximately 4750--5430 A.
    {"label": "H beta", "species": "H I", "rest_air_A": 4861.333, "arm": "blue"},
    {"label": "Mg I b1", "species": "Mg I", "rest_air_A": 5167.322, "arm": "blue"},
    {"label": "Mg I b2", "species": "Mg I", "rest_air_A": 5172.684, "arm": "blue"},
    {"label": "Mg I b3", "species": "Mg I", "rest_air_A": 5183.604, "arm": "blue"},
    {"label": "Fe I 4921", "species": "Fe I", "rest_air_A": 4920.503, "arm": "blue"},
    {"label": "Fe I 4958", "species": "Fe I", "rest_air_A": 4957.597, "arm": "blue"},
    {"label": "Fe I 5227", "species": "Fe I", "rest_air_A": 5227.151, "arm": "blue"},
    {"label": "Fe I 5270", "species": "Fe I", "rest_air_A": 5269.538, "arm": "blue"},
    {"label": "Fe I 5270.4", "species": "Fe I", "rest_air_A": 5270.356, "arm": "blue"},
    {"label": "Fe I 5328", "species": "Fe I", "rest_air_A": 5328.039, "arm": "blue"},
    {"label": "Fe I 5341", "species": "Fe I", "rest_air_A": 5341.024, "arm": "blue"},
    {"label": "Fe I 5371", "species": "Fe I", "rest_air_A": 5371.490, "arm": "blue"},
    {"label": "Fe I 5397", "species": "Fe I", "rest_air_A": 5397.128, "arm": "blue"},
    {"label": "Fe II 4924", "species": "Fe II", "rest_air_A": 4923.927, "arm": "blue"},
    {"label": "Fe II 5018", "species": "Fe II", "rest_air_A": 5018.440, "arm": "blue"},
    {"label": "Fe II 5169", "species": "Fe II", "rest_air_A": 5169.033, "arm": "blue"},
    {"label": "Ti I 4982", "species": "Ti I", "rest_air_A": 4981.730, "arm": "blue"},
    {"label": "Ti I 4991", "species": "Ti I", "rest_air_A": 4991.066, "arm": "blue"},
    {"label": "Ti I 5007", "species": "Ti I", "rest_air_A": 5007.206, "arm": "blue"},
    {"label": "Ti I 5014", "species": "Ti I", "rest_air_A": 5014.186, "arm": "blue"},
    {"label": "Ti I 5210", "species": "Ti I", "rest_air_A": 5210.384, "arm": "blue"},
    {"label": "Cr I 5205", "species": "Cr I", "rest_air_A": 5204.505, "arm": "blue"},
    {"label": "Cr I 5206", "species": "Cr I", "rest_air_A": 5206.021, "arm": "blue"},
    {"label": "Cr I 5208", "species": "Cr I", "rest_air_A": 5208.415, "arm": "blue"},
    {"label": "Cr I 5297", "species": "Cr I", "rest_air_A": 5296.690, "arm": "blue"},
    {"label": "Cr I 5346", "species": "Cr I", "rest_air_A": 5345.770, "arm": "blue"},
    {"label": "Cr I 5410", "species": "Cr I", "rest_air_A": 5409.780, "arm": "blue"},
    {"label": "Ca I 4878", "species": "Ca I", "rest_air_A": 4878.132, "arm": "blue"},
    {"label": "Ca I 5189", "species": "Ca I", "rest_air_A": 5188.848, "arm": "blue"},
    {"label": "Ca I 5266", "species": "Ca I", "rest_air_A": 5265.557, "arm": "blue"},
    {"label": "Ca I 5349", "species": "Ca I", "rest_air_A": 5349.472, "arm": "blue"},
    {"label": "Ca II 5020", "species": "Ca II", "rest_air_A": 5019.971, "arm": "blue"},
    # Red arm: approximately 6230--7430 A.
    {"label": "H alpha", "species": "H I", "rest_air_A": 6562.790, "arm": "red"},
    {"label": "Mg II 6347", "species": "Mg II", "rest_air_A": 6346.742, "arm": "red"},
    {"label": "Mg II 6546", "species": "Mg II", "rest_air_A": 6545.973, "arm": "red"},
    {"label": "Fe I 6335", "species": "Fe I", "rest_air_A": 6335.337, "arm": "red"},
    {"label": "Fe II 6248", "species": "Fe II", "rest_air_A": 6247.560, "arm": "red"},
    {"label": "Fe II 6456", "species": "Fe II", "rest_air_A": 6456.380, "arm": "red"},
    {"label": "Fe II 6516", "species": "Fe II", "rest_air_A": 6516.080, "arm": "red"},
    {"label": "Ti I 6258.1", "species": "Ti I", "rest_air_A": 6258.099, "arm": "red"},
    {"label": "Ti I 6258.7", "species": "Ti I", "rest_air_A": 6258.705, "arm": "red"},
    {"label": "Ti I 6261", "species": "Ti I", "rest_air_A": 6261.096, "arm": "red"},
    {"label": "Ti I 7209", "species": "Ti I", "rest_air_A": 7209.434, "arm": "red"},
    {"label": "Cr I 7400", "species": "Cr I", "rest_air_A": 7400.220, "arm": "red"},
    {"label": "Si I 6237", "species": "Si I", "rest_air_A": 6237.320, "arm": "red"},
    {"label": "Si I 6254", "species": "Si I", "rest_air_A": 6254.188, "arm": "red"},
    {"label": "Si I 7004", "species": "Si I", "rest_air_A": 7003.567, "arm": "red"},
    {"label": "Si I 7006", "species": "Si I", "rest_air_A": 7005.883, "arm": "red"},
    {"label": "Si I 7035", "species": "Si I", "rest_air_A": 7034.903, "arm": "red"},
    {"label": "Si I 7166", "species": "Si I", "rest_air_A": 7165.545, "arm": "red"},
    {"label": "Si I 7251", "species": "Si I", "rest_air_A": 7250.625, "arm": "red"},
    {"label": "Si I 7275", "species": "Si I", "rest_air_A": 7275.294, "arm": "red"},
    {"label": "Si II 6347", "species": "Si II", "rest_air_A": 6347.103, "arm": "red"},
    {"label": "Si II 6371", "species": "Si II", "rest_air_A": 6371.359, "arm": "red"},
    {"label": "Ca I 6439", "species": "Ca I", "rest_air_A": 6439.073, "arm": "red"},
    {"label": "Ca I 6463", "species": "Ca I", "rest_air_A": 6462.566, "arm": "red"},
    {"label": "Ca I 6494", "species": "Ca I", "rest_air_A": 6493.780, "arm": "red"},
    {"label": "Ca I 6573", "species": "Ca I", "rest_air_A": 6572.777, "arm": "red"},
    {"label": "Ca I 6718", "species": "Ca I", "rest_air_A": 6717.685, "arm": "red"},
    {"label": "Ca I 7148", "species": "Ca I", "rest_air_A": 7148.147, "arm": "red"},
    {"label": "Ca I 7202", "species": "Ca I", "rest_air_A": 7202.194, "arm": "red"},
    {"label": "Ca I 7326", "species": "Ca I", "rest_air_A": 7326.146, "arm": "red"},
    {"label": "Ca II 6457", "species": "Ca II", "rest_air_A": 6456.874, "arm": "red"},
    {"label": "Li I 6708", "species": "Li I", "rest_air_A": 6707.840, "arm": "red"},
)


def _vacuum_line_catalog(air_catalog):
    """Return line records with explicit air provenance and vacuum coordinates."""

    air_wavelengths = np.asarray(
        [line["rest_air_A"] for line in air_catalog],
        dtype=float,
    )
    vacuum_wavelengths = np.asarray(air_to_vac(air_wavelengths), dtype=float)
    return tuple(
        {
            **line,
            "rest_vacuum_A": float(vacuum_wavelength),
            "rest_wavelength_medium": "vacuum",
            "provenance_wavelength_medium": "air",
        }
        for line, vacuum_wavelength in zip(air_catalog, vacuum_wavelengths)
    )


TRANSMISSION_LINE_CATALOG = _vacuum_line_catalog(_AIR_TRANSMISSION_LINE_CATALOG)

DEFAULT_SPECIES = tuple(dict.fromkeys(line["species"] for line in TRANSMISSION_LINE_CATALOG))


def _finite_good(values, sigma, sigma_threshold):
    return (
        np.isfinite(values)
        & np.isfinite(sigma)
        & (sigma > 0)
        & (sigma < float(sigma_threshold))
    )


def _robust_symmetric_limit(values, percentile=99.2, floor=2e-5):
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    if finite.size == 0:
        return float(floor)
    limit = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(limit) or limit <= 0:
        return float(floor)
    return max(limit, float(floor))


def _velocity_grid(window_kms, bin_kms):
    if window_kms <= 0 or bin_kms <= 0:
        raise ValueError("window_kms and bin_kms must be positive")
    count = int(math.floor(2.0 * window_kms / bin_kms)) + 1
    return np.linspace(-float(window_kms), float(window_kms), count)


def _relativistic_doppler_factor(velocity_kms):
    velocity = np.asarray(velocity_kms, dtype=float)
    beta = velocity / C_KMS
    if np.any(~np.isfinite(beta)) or np.any(np.abs(beta) >= 1.0):
        raise ValueError("Doppler velocities must be finite and subluminal")
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def _wavelength_at_velocity(rest_vacuum_A, velocity_kms):
    return float(rest_vacuum_A) * _relativistic_doppler_factor(velocity_kms)


def _nearest_sample_distance(sorted_wave, sample_wave):
    positions = np.searchsorted(sorted_wave, sample_wave)
    left = np.clip(positions - 1, 0, sorted_wave.size - 1)
    right = np.clip(positions, 0, sorted_wave.size - 1)
    return np.minimum(
        np.abs(sample_wave - sorted_wave[left]),
        np.abs(sample_wave - sorted_wave[right]),
    )


def _sample_line_window(
    bundle,
    line,
    velocity_grid,
    *,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    minimum_native_pixels=4,
):
    """Interpolate an exposure cube around one transition with a gap guard."""

    rest_vacuum_A = float(line["rest_vacuum_A"])
    wave = np.asarray(bundle["wavelength"], dtype=float)
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    if data.shape != sigma.shape or data.ndim != 2 or wave.size != data.shape[1]:
        raise ValueError(
            f"Incompatible bundle shapes: wave={wave.shape}, data={data.shape}, sigma={sigma.shape}"
        )

    sample_wave = _wavelength_at_velocity(rest_vacuum_A, velocity_grid)
    margin_A = rest_vacuum_A * float(max_interp_gap_kms) / C_KMS
    lo = float(np.nanmin(sample_wave) - margin_A)
    hi = float(np.nanmax(sample_wave) + margin_A)
    native = np.flatnonzero(np.isfinite(wave) & (wave >= lo) & (wave <= hi))
    if native.size < int(minimum_native_pixels):
        return None

    wave_native = wave[native]
    order = np.argsort(wave_native)
    wave_native = wave_native[order]
    native = native[order]
    too_far = _nearest_sample_distance(wave_native, sample_wave) > margin_A

    matrix = np.full((data.shape[0], velocity_grid.size), np.nan, dtype=float)
    sigma_matrix = np.full_like(matrix, np.nan)
    for row in range(data.shape[0]):
        row_data = data[row, native]
        row_sigma = sigma[row, native]
        good = _finite_good(row_data, row_sigma, sigma_threshold)
        if np.count_nonzero(good) < 2:
            continue
        row_wave = wave_native[good]
        matrix[row] = np.interp(sample_wave, row_wave, row_data[good], left=np.nan, right=np.nan)
        sigma_matrix[row] = np.interp(sample_wave, row_wave, row_sigma[good], left=np.nan, right=np.nan)
        matrix[row, too_far] = np.nan
        sigma_matrix[row, too_far] = np.nan

    valid = _finite_good(matrix, sigma_matrix, sigma_threshold)
    matrix = np.where(valid, matrix, np.nan)
    sigma_matrix = np.where(valid, sigma_matrix, np.nan)
    if np.count_nonzero(valid) == 0:
        return None
    return {
        "line": dict(line),
        "matrix": matrix,
        "sigma": sigma_matrix,
        "native_pixels": int(native.size),
    }


def _sample_line_planet_frame(
    bundle,
    line,
    velocity_grid,
    trail,
    *,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    minimum_native_pixels=4,
):
    """Sample one transition directly on a requested planet-frame grid.

    The observed wavelength queried for each exposure is the requested output
    wavelength multiplied by that exposure's relativistic orbital Doppler
    factor.  Sampling the original wavelength array directly keeps the output
    window independent of the (often much larger) observed-frame trail.
    """

    rest_vacuum_A = float(line["rest_vacuum_A"])
    wave = np.asarray(bundle["wavelength"], dtype=float)
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    if data.shape != sigma.shape or data.ndim != 2 or wave.size != data.shape[1]:
        raise ValueError(
            f"Incompatible bundle shapes: wave={wave.shape}, data={data.shape}, sigma={sigma.shape}"
        )

    output_wave = _wavelength_at_velocity(rest_vacuum_A, velocity_grid)
    if trail is None:
        trail = np.zeros(data.shape[0], dtype=float)
    else:
        trail = np.asarray(trail, dtype=float)
    if trail.shape != (data.shape[0],):
        raise ValueError(
            f"Planet trail has shape {trail.shape}; expected ({data.shape[0]},)"
        )
    sample_wave = _relativistic_doppler_factor(trail)[:, None] * output_wave[None, :]
    margin_A = rest_vacuum_A * float(max_interp_gap_kms) / C_KMS
    lo = float(np.nanmin(sample_wave) - margin_A)
    hi = float(np.nanmax(sample_wave) + margin_A)
    native = np.flatnonzero(np.isfinite(wave) & (wave >= lo) & (wave <= hi))
    if native.size < int(minimum_native_pixels):
        return None

    wave_native = wave[native]
    order = np.argsort(wave_native)
    wave_native = wave_native[order]
    native = native[order]
    too_far = _nearest_sample_distance(wave_native, sample_wave) > margin_A

    matrix = np.full(sample_wave.shape, np.nan, dtype=float)
    sigma_matrix = np.full_like(matrix, np.nan)
    for row in range(data.shape[0]):
        row_data = data[row, native]
        row_sigma = sigma[row, native]
        good = _finite_good(row_data, row_sigma, sigma_threshold)
        if np.count_nonzero(good) < 2:
            continue
        row_wave = wave_native[good]
        matrix[row] = np.interp(
            sample_wave[row], row_wave, row_data[good], left=np.nan, right=np.nan
        )
        sigma_matrix[row] = np.interp(
            sample_wave[row], row_wave, row_sigma[good], left=np.nan, right=np.nan
        )
        matrix[row, too_far[row]] = np.nan
        sigma_matrix[row, too_far[row]] = np.nan

    valid = _finite_good(matrix, sigma_matrix, sigma_threshold)
    matrix = np.where(valid, matrix, np.nan)
    sigma_matrix = np.where(valid, sigma_matrix, np.nan)
    if np.count_nonzero(valid) == 0:
        return None
    return {
        "line": dict(line),
        "matrix": matrix,
        "sigma": sigma_matrix,
        "native_pixels": int(native.size),
    }


def line_coverage_rows(
    bundle,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    *,
    window_kms=150.0,
    minimum_native_pixels=4,
):
    """Return coverage records suitable for a notebook DataFrame."""

    wave = np.asarray(bundle["wavelength"], dtype=float)
    rows = []
    for line in line_catalog:
        rest_vacuum_A = float(line["rest_vacuum_A"])
        bounds = _wavelength_at_velocity(
            rest_vacuum_A,
            np.array([-float(window_kms), float(window_kms)]),
        )
        n_pixels = int(np.count_nonzero((wave >= bounds[0]) & (wave <= bounds[1])))
        rows.append(
            {
                "epoch": bundle.get("epoch"),
                "arm": bundle.get("arm"),
                "species": line["species"],
                "line": line["label"],
                "rest_air_A": float(line["rest_air_A"]),
                "rest_vacuum_A": rest_vacuum_A,
                "native_pixels": n_pixels,
                "covered": n_pixels >= int(minimum_native_pixels),
            }
        )
    return rows


def _combine_sampled_lines(sampled, sigma_threshold):
    matrices = np.stack([result["matrix"] for result in sampled], axis=0)
    uncertainties = np.stack([result["sigma"] for result in sampled], axis=0)
    valid = _finite_good(matrices, uncertainties, sigma_threshold)
    weights = np.where(valid, 1.0 / np.square(uncertainties), 0.0)
    weight_sum = np.sum(weights, axis=0)
    combined = np.divide(
        np.sum(np.where(valid, matrices * weights, 0.0), axis=0),
        weight_sum,
        out=np.full(matrices.shape[1:], np.nan),
        where=weight_sum > 0,
    )
    combined_sigma = np.divide(
        1.0,
        np.sqrt(weight_sum),
        out=np.full(matrices.shape[1:], np.nan),
        where=weight_sum > 0,
    )
    return combined, combined_sigma


def _species_products(
    bundle,
    *,
    line_catalog,
    species,
    velocity_grid,
    sigma_threshold,
    max_interp_gap_kms,
):
    products = []
    for species_name in species:
        sampled = []
        for line in line_catalog:
            if line["species"] != species_name:
                continue
            result = _sample_line_window(
                bundle,
                line,
                velocity_grid,
                sigma_threshold=sigma_threshold,
                max_interp_gap_kms=max_interp_gap_kms,
            )
            if result is not None:
                sampled.append(result)
        if not sampled:
            continue

        combined, combined_sigma = _combine_sampled_lines(sampled, sigma_threshold)
        products.append(
            {
                "species": species_name,
                "matrix": combined,
                "sigma": combined_sigma,
                "lines": tuple(result["line"] for result in sampled),
            }
        )
    return products


def _planet_frame_species_products(
    bundle,
    *,
    line_catalog,
    species,
    velocity_grid,
    trail,
    sigma_threshold,
    max_interp_gap_kms,
):
    products = []
    for species_name in species:
        sampled = []
        for line in line_catalog:
            if line["species"] != species_name:
                continue
            result = _sample_line_planet_frame(
                bundle,
                line,
                velocity_grid,
                trail,
                sigma_threshold=sigma_threshold,
                max_interp_gap_kms=max_interp_gap_kms,
            )
            if result is not None:
                sampled.append(result)
        if not sampled:
            continue
        combined, combined_sigma = _combine_sampled_lines(sampled, sigma_threshold)
        products.append(
            {
                "species": species_name,
                "matrix": combined,
                "sigma": combined_sigma,
                "lines": tuple(result["line"] for result in sampled),
            }
        )
    return products


def _planet_velocity(
    phase,
    kp_kms,
    vsys_kms,
    *,
    eccentricity=0.0,
    omega_deg=None,
):
    if kp_kms is None or not np.isfinite(float(kp_kms)):
        return None
    vsys = 0.0 if vsys_kms is None or not np.isfinite(float(vsys_kms)) else float(vsys_kms)
    return planet_radial_velocity_kms(
        np.asarray(phase, dtype=float),
        kp_kms=float(kp_kms),
        eccentricity=float(eccentricity),
        omega_deg=omega_deg,
    ) + vsys


def plot_species_2d_atlas(
    bundle,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    kp_kms=None,
    vsys_kms=0.0,
    eccentricity=0.0,
    omega_deg=None,
    phase_rows_top_to_bottom=True,
    percentile=99.2,
    ncols=3,
):
    """Plot a coverage-aware 2D residual atlas, combining lines by species."""

    phase = np.asarray(bundle["phase"], dtype=float)
    order = np.argsort(phase)
    trail = _planet_velocity(
        phase[order],
        kp_kms,
        vsys_kms,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
    )
    atlas_window_kms = float(window_kms)
    if trail is not None and np.any(np.isfinite(trail)):
        trail_margin_kms = max(20.0, 0.2 * float(window_kms))
        required = float(np.nanmax(np.abs(trail))) + trail_margin_kms
        atlas_window_kms = max(
            atlas_window_kms,
            float(math.ceil(required / float(bin_kms)) * float(bin_kms)),
        )
    velocity_grid = _velocity_grid(atlas_window_kms, bin_kms)
    products = _species_products(
        bundle,
        line_catalog=line_catalog,
        species=species,
        velocity_grid=velocity_grid,
        sigma_threshold=sigma_threshold,
        max_interp_gap_kms=max_interp_gap_kms,
    )
    if not products:
        print(f"No catalogued species lines covered by {bundle.get('epoch')} {bundle.get('arm')}.")
        return None

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(products) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7.5, 4.35 * ncols), max(3.8, 3.0 * nrows)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, product in zip(axes.ravel(), products):
        vmax = _robust_symmetric_limit(product["matrix"], percentile=percentile)
        line_count = len(product["lines"])
        line_word = "line" if line_count == 1 else "lines"
        ax.pcolormesh(
            velocity_grid,
            phase[order],
            product["matrix"][order],
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        ax.axvline(0.0, color="0.2", lw=0.65, ls=":")
        if trail is not None:
            ax.plot(trail, phase[order], color="gold", lw=0.9, ls="--")
        if phase_rows_top_to_bottom:
            ax.set_ylim(float(np.nanmax(phase)), float(np.nanmin(phase)))
        line_labels = ", ".join(line["label"] for line in product["lines"])
        ax.set_title(
            f"{product['species']} ({line_count} {line_word}; scale +/-{vmax:.2g})",
            fontsize=9.5,
        )
        ax.text(
            0.02,
            0.02,
            line_labels,
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
            color="0.15",
            bbox={"facecolor": "white", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )
    for ax in axes.ravel()[len(products) :]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Velocity relative to vacuum rest wavelength [km/s]")
    for ax in axes[:, 0]:
        if ax.axison:
            ax.set_ylabel("Orbital phase")
    trail_note = "gold: expected planet trail" if trail is not None else "Kp unavailable: no trail overlay"
    window_note = f"observed-frame window +/-{atlas_window_kms:.0f} km/s"
    fig.suptitle(
        f"Species-combined 2D residual atlas: {bundle.get('epoch')} {bundle.get('arm')}\n"
        f"{trail_note}; {window_note}\n"
        "Each panel uses its own robust residual scale",
        fontsize=13,
    )
    return fig, axes, products


def _coadd_sampled_frame(matrix, sigma, velocity_grid, sigma_threshold):
    valid = _finite_good(matrix, sigma, sigma_threshold)
    weights = np.where(valid, 1.0 / np.square(sigma), 0.0)
    weight_sum = np.sum(weights, axis=0)
    mean = np.divide(
        np.sum(np.where(valid, matrix * weights, 0.0), axis=0),
        weight_sum,
        out=np.full(velocity_grid.shape, np.nan),
        where=weight_sum > 0,
    )
    error = np.divide(
        1.0,
        np.sqrt(weight_sum),
        out=np.full(velocity_grid.shape, np.nan),
        where=weight_sum > 0,
    )
    contributing_exposures = np.sum(valid, axis=0).astype(int)
    contributing_fraction = contributing_exposures / float(matrix.shape[0])
    finite_weight = weight_sum[np.isfinite(weight_sum) & (weight_sum > 0)]
    maximum_weight = float(np.nanmax(finite_weight)) if finite_weight.size else np.nan
    relative_weight = np.divide(
        weight_sum,
        maximum_weight,
        out=np.zeros_like(weight_sum),
        where=np.isfinite(maximum_weight) & (maximum_weight > 0),
    )
    return (
        mean,
        error,
        contributing_exposures,
        contributing_fraction,
        weight_sum,
        relative_weight,
    )


def species_frame_stack_results(
    bundle,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    kp_kms=None,
    vsys_kms=0.0,
    eccentricity=0.0,
    omega_deg=None,
):
    """Return coverage-aware species stacks without creating a figure.

    This public data helper lets emission diagnostics compare stacks across
    epochs and eclipse sides while keeping interpolation and masking shared
    with the atlas.  Unlike the observed-frame atlas, it samples the original
    wavelength grid directly at each exposure's relativistic Doppler query, so
    high-Kp trails cannot be clipped by the requested output window.
    """

    velocity_grid = _velocity_grid(window_kms, bin_kms)
    phase = np.asarray(bundle["phase"], dtype=float)
    trail = _planet_velocity(
        phase,
        kp_kms,
        vsys_kms,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
    )
    products = _planet_frame_species_products(
        bundle,
        line_catalog=line_catalog,
        species=species,
        velocity_grid=velocity_grid,
        trail=trail,
        sigma_threshold=sigma_threshold,
        max_interp_gap_kms=max_interp_gap_kms,
    )
    results = []
    for product in products:
        (
            mean,
            error,
            contributing_exposures,
            contributing_fraction,
            weight_sum,
            relative_weight,
        ) = _coadd_sampled_frame(
            product["matrix"],
            product["sigma"],
            velocity_grid,
            sigma_threshold,
        )
        center_index = int(np.argmin(np.abs(velocity_grid)))
        finite_profile = np.isfinite(mean) & np.isfinite(error)
        minimum_fraction = (
            float(np.nanmin(contributing_fraction[finite_profile]))
            if np.any(finite_profile)
            else 0.0
        )
        center_fraction = float(contributing_fraction[center_index])
        center_relative_weight = float(relative_weight[center_index])
        coverage_warning = bool(
            center_fraction < 0.95
            or center_relative_weight < 0.5
            or minimum_fraction < 0.5
        )
        results.append(
            {
                "species": product["species"],
                "velocity": velocity_grid.copy(),
                "mean": mean,
                "error": error,
                "lines": product["lines"],
                "frame": "planet" if trail is not None else "line_rest",
                "n_exposures": int(phase.size),
                "contributing_exposures": contributing_exposures,
                "contributing_fraction": contributing_fraction,
                "weight_sum": weight_sum,
                "relative_weight": relative_weight,
                "center_coverage_fraction": center_fraction,
                "center_relative_weight": center_relative_weight,
                "minimum_coverage_fraction": minimum_fraction,
                "coverage_warning": coverage_warning,
            }
        )
    return results


def species_velocity_coverage_rows(
    bundle,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    kp_kms=None,
    vsys_kms=0.0,
):
    """Return per-species contribution coverage for direct frame sampling."""

    rows = []
    for result in species_frame_stack_results(
        bundle,
        line_catalog=line_catalog,
        species=species,
        window_kms=window_kms,
        bin_kms=bin_kms,
        sigma_threshold=sigma_threshold,
        max_interp_gap_kms=max_interp_gap_kms,
        kp_kms=kp_kms,
        vsys_kms=vsys_kms,
    ):
        rows.append(
            {
                "epoch": bundle.get("epoch"),
                "arm": bundle.get("arm"),
                "species": result["species"],
                "n_lines": len(result["lines"]),
                "n_exposures": result["n_exposures"],
                "center_coverage_fraction": result["center_coverage_fraction"],
                "center_relative_weight": result["center_relative_weight"],
                "minimum_coverage_fraction": result["minimum_coverage_fraction"],
                "coverage_warning": result["coverage_warning"],
            }
        )
    return rows


def plot_species_frame_stacks(
    bundle,
    *,
    line_catalog=TRANSMISSION_LINE_CATALOG,
    species=DEFAULT_SPECIES,
    window_kms=150.0,
    bin_kms=2.0,
    sigma_threshold=0.5,
    max_interp_gap_kms=5.0,
    kp_kms=None,
    vsys_kms=0.0,
    ncols=3,
):
    """Plot one combined 1D residual stack for every covered species."""

    results = species_frame_stack_results(
        bundle,
        line_catalog=line_catalog,
        species=species,
        window_kms=window_kms,
        bin_kms=bin_kms,
        sigma_threshold=sigma_threshold,
        max_interp_gap_kms=max_interp_gap_kms,
        kp_kms=kp_kms,
        vsys_kms=vsys_kms,
    )
    if not results:
        return None
    velocity_grid = results[0]["velocity"]
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(len(results) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.35 * ncols, 2.75 * nrows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    active_axes = list(axes.ravel()[: len(results)])
    for panel_index, (ax, result) in enumerate(zip(active_axes, results)):
        line_count = len(result["lines"])
        line_word = "line" if line_count == 1 else "lines"
        mean = result["mean"]
        error = result["error"]
        ax.plot(velocity_grid, mean, color="k", lw=0.9)
        ax.fill_between(velocity_grid, mean - error, mean + error, color="tab:blue", alpha=0.2, lw=0)
        ax.axhline(0.0, color="0.4", lw=0.7, ls="--")
        ax.axvline(0.0, color="0.25", lw=0.7, ls=":")
        limit = _robust_symmetric_limit(mean, percentile=98.8)
        ax.set_ylim(-limit, limit)
        warning_note = "; coverage warning" if result["coverage_warning"] else ""
        ax.set_title(
            f"{result['species']} ({line_count} {line_word}{warning_note})",
            fontsize=10,
            color="tab:red" if result["coverage_warning"] else "black",
        )
        ax.set_ylabel("Residual")
        coverage_ax = ax.twinx()
        coverage_ax.plot(
            velocity_grid,
            result["contributing_fraction"],
            color="0.45",
            lw=0.75,
            alpha=0.75,
            label="exposure fraction",
        )
        coverage_ax.plot(
            velocity_grid,
            result["relative_weight"],
            color="tab:orange",
            lw=0.7,
            ls=":",
            alpha=0.8,
            label="relative weight",
        )
        coverage_ax.set_ylim(-0.03, 1.03)
        coverage_ax.tick_params(axis="y", colors="0.45", labelsize=7)
        if panel_index % ncols == ncols - 1 or panel_index == len(results) - 1:
            coverage_ax.set_ylabel("Contribution fraction", color="0.45", fontsize=8)
        else:
            coverage_ax.tick_params(labelright=False)
        ax.text(
            0.02,
            0.96,
            (
                f"coverage @ 0: {result['center_coverage_fraction']:.0%}; "
                f"weight: {result['center_relative_weight']:.0%}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.8,
            color="tab:red" if result["coverage_warning"] else "0.30",
            bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none", "pad": 1.2},
        )
    for ax in axes.ravel()[len(results) :]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Velocity relative to vacuum rest wavelength [km/s]")
    frame = "planet frame" if results[0]["frame"] == "planet" else "line rest frame"
    fig.suptitle(
        f"Species-combined {frame} stacks: {bundle.get('epoch')} {bundle.get('arm')}\n"
        "direct relativistic sampling; gray=exposure coverage, orange=relative weight",
        fontsize=13,
    )
    return fig, axes, results


def _thin_indices(size, max_points):
    if size <= max_points:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, int(max_points)).astype(int))


def _masked_rms(values, valid, axis):
    counts = np.sum(valid, axis=axis)
    sums = np.sum(np.where(valid, np.square(values), 0.0), axis=axis)
    return np.sqrt(
        np.divide(sums, counts, out=np.full(np.shape(sums), np.nan, dtype=float), where=counts > 0)
    )


def plot_residual_quality_summary(
    bundle,
    *,
    sigma_threshold=0.5,
    phase_rows_top_to_bottom=True,
    max_wavelength_points=1400,
):
    """Plot whitened residuals and uncertainty-calibration summaries."""

    wave = np.asarray(bundle["wavelength"], dtype=float)
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    phase = np.asarray(bundle["phase"], dtype=float)
    valid = _finite_good(data, sigma, sigma_threshold)
    order = np.argsort(phase)
    if not phase_rows_top_to_bottom:
        order = order[::-1]
    index = _thin_indices(wave.size, max_wavelength_points)
    whitened = np.divide(
        data[:, index],
        sigma[:, index],
        out=np.full((data.shape[0], index.size), np.nan),
        where=valid[:, index],
    )
    whitened_limit = _robust_symmetric_limit(whitened, percentile=99.2, floor=3.0)

    row_rms = _masked_rms(data, valid, axis=1)
    row_noise = _masked_rms(sigma, valid, axis=1)
    standardized = np.divide(
        data,
        sigma,
        out=np.full_like(data, np.nan),
        where=valid,
    )
    row_chi2 = np.divide(
        np.sum(np.where(valid, np.square(standardized), 0.0), axis=1),
        np.sum(valid, axis=1),
        out=np.full(data.shape[0], np.nan),
        where=np.sum(valid, axis=1) > 0,
    )
    column_rms = _masked_rms(data, valid, axis=0)
    column_noise = _masked_rms(sigma, valid, axis=0)
    column_ratio = np.divide(
        column_rms,
        column_noise,
        out=np.full(wave.shape, np.nan),
        where=np.isfinite(column_noise) & (column_noise > 0),
    )
    valid_fraction = np.mean(valid, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0), constrained_layout=True)
    images = pcolormesh_wavelength_segments(
        axes[0, 0],
        wave[index],
        whitened[order],
        cmap="RdBu_r",
        vmin=-whitened_limit,
        vmax=whitened_limit,
    )
    image = images[0]
    axes[0, 0].set_ylim(len(order) - 0.5, -0.5)
    ticks = _thin_indices(len(order), 10)
    axes[0, 0].set_yticks(ticks)
    axes[0, 0].set_yticklabels([f"{phase[order][i]:+.4f}" for i in ticks])
    axes[0, 0].set_xlabel("Vacuum wavelength [Å]")
    axes[0, 0].set_ylabel("Orbital phase")
    axes[0, 0].set_title("Whitened residual matrix (data / sigma)")
    fig.colorbar(image, ax=axes[0, 0], pad=0.012, fraction=0.045, label="Standardized residual")

    axes[0, 1].plot(phase[order], row_rms[order], "o-", ms=3, lw=0.8, label="residual RMS")
    axes[0, 1].plot(phase[order], row_noise[order], "o-", ms=3, lw=0.8, label="reported sigma RMS")
    axes[0, 1].set_xlabel("Orbital phase")
    axes[0, 1].set_ylabel("Flux scale")
    axes[0, 1].set_title("Exposure-wise residual and noise scale")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(phase[order], row_chi2[order], "o-", ms=3, lw=0.8, color="tab:red")
    axes[1, 0].axhline(1.0, color="0.35", lw=0.8, ls="--")
    axes[1, 0].set_xlabel("Orbital phase")
    axes[1, 0].set_ylabel("mean (data / sigma)^2")
    axes[1, 0].set_title("Exposure-wise uncertainty calibration")

    ratio_axis = axes[1, 1]
    plot_wavelength_segments(
        ratio_axis,
        wave[index],
        column_ratio[index],
        lw=0.75,
        color="tab:purple",
        label="RMS / sigma",
    )
    ratio_axis.axhline(1.0, color="0.35", lw=0.8, ls="--")
    ratio_axis.set_xlabel("Vacuum wavelength [Å]")
    ratio_axis.set_ylabel("Residual RMS / reported sigma", color="tab:purple")
    fraction_axis = ratio_axis.twinx()
    plot_wavelength_segments(
        fraction_axis,
        wave[index],
        valid_fraction[index],
        lw=0.65,
        color="tab:green",
        alpha=0.75,
        label="valid fraction",
    )
    fraction_axis.set_ylabel("Valid exposure fraction", color="tab:green")
    fraction_axis.set_ylim(-0.03, 1.03)
    ratio_axis.set_title("Wavelength-wise noise ratio and coverage")

    fig.suptitle(f"Residual quality summary: {bundle.get('epoch')} {bundle.get('arm')}", fontsize=13)
    return fig, axes


def plot_pre_post_sysrem_comparison(
    bundle,
    *,
    sigma_threshold=0.5,
    max_wavelength_points=1400,
):
    """Compare optional pre-SYSREM arrays with the final prepared residual cube."""

    pre_data = bundle.get("pre_sysrem_data")
    pre_sigma = bundle.get("pre_sysrem_sigma")
    if pre_data is None or pre_sigma is None:
        print(f"No pre-SYSREM arrays for {bundle.get('epoch')} {bundle.get('arm')}; comparison skipped.")
        return None

    wave = np.asarray(bundle["wavelength"], dtype=float)
    post_data = np.asarray(bundle["data"], dtype=float)
    post_sigma = np.asarray(bundle["sigma"], dtype=float)
    pre_data = np.asarray(pre_data, dtype=float)
    pre_sigma = np.asarray(pre_sigma, dtype=float)
    if pre_data.shape != post_data.shape or pre_sigma.shape != post_sigma.shape:
        print(
            f"Pre/post shapes differ for {bundle.get('epoch')} {bundle.get('arm')}: "
            f"pre={pre_data.shape}/{pre_sigma.shape}, post={post_data.shape}/{post_sigma.shape}; comparison skipped."
        )
        return None

    pre_valid = _finite_good(pre_data, pre_sigma, sigma_threshold)
    post_valid = _finite_good(post_data, post_sigma, sigma_threshold)
    shared = pre_valid & post_valid
    removed = np.where(shared, pre_data - post_data, np.nan)
    index = _thin_indices(wave.size, max_wavelength_points)
    phase = np.asarray(bundle["phase"], dtype=float)
    order = np.argsort(phase)
    limit = _robust_symmetric_limit(
        np.concatenate((pre_data[:, index].ravel(), post_data[:, index].ravel(), removed[:, index].ravel())),
        percentile=99.2,
    )
    row_pre = _masked_rms(pre_data, pre_valid, axis=1)
    row_post = _masked_rms(post_data, post_valid, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0), constrained_layout=True)
    image = None
    for ax, matrix, title in (
        (axes[0, 0], pre_data, "Pre-SYSREM residuals"),
        (axes[0, 1], post_data, "Final residuals"),
        (axes[1, 0], removed, "Removed component (pre - final)"),
    ):
        images = pcolormesh_wavelength_segments(
            ax,
            wave[index],
            matrix[order][:, index],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        image = images[0]
        ax.set_ylim(len(order) - 0.5, -0.5)
        ticks = _thin_indices(len(order), 9)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{phase[order][i]:+.4f}" for i in ticks])
        ax.set_xlabel("Vacuum wavelength [Å]")
        ax.set_ylabel("Orbital phase")
        ax.set_title(title)
    axes[1, 1].plot(phase[order], row_pre[order], "o-", ms=3, lw=0.8, label="pre-SYSREM")
    axes[1, 1].plot(phase[order], row_post[order], "o-", ms=3, lw=0.8, label="final")
    axes[1, 1].set_xlabel("Orbital phase")
    axes[1, 1].set_ylabel("Residual RMS")
    axes[1, 1].set_title("Exposure-wise RMS suppression")
    axes[1, 1].legend(loc="best")
    if image is not None:
        fig.colorbar(image, ax=axes[:2, :].ravel().tolist(), pad=0.012, fraction=0.025, label="Residual flux")
    fig.suptitle(f"Pre/post-SYSREM comparison: {bundle.get('epoch')} {bundle.get('arm')}", fontsize=13)
    return fig, axes


__all__ = [
    "DEFAULT_SPECIES",
    "TRANSMISSION_LINE_CATALOG",
    "line_coverage_rows",
    "plot_pre_post_sysrem_comparison",
    "plot_residual_quality_summary",
    "plot_species_2d_atlas",
    "plot_species_frame_stacks",
    "species_velocity_coverage_rows",
    "species_frame_stack_results",
]
