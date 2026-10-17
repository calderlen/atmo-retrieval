"""Build collapsed transmission spectra from prepared full-exposure source cubes."""

from __future__ import annotations

import argparse
import json
import numpy as np
import config
import config_utils
from pathlib import Path
from config import EPHEMERIS
from dataio.orbital_velocity import planet_radial_velocity_kms
from dataio.pepsi import _SPEED_OF_LIGHT_KMS
from dataio.timing import (
    active_transit_mask,
    compute_contact_phases,
    get_bjd_tdb,
    get_ephemeris_epoch_bjd_tdb,
    get_orbital_phase,
)
from dataio.spectral_preprocessing import build_out_of_transit_residuals


TRANSMISSION_COLLAPSE_SCHEMA_VERSION = 4


def calculate_transmission_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    jd: np.ndarray,
    transit_params: dict,
    RA: str,
    Dec: str,
    Kp_kms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a planet-rest-frame differential transmission spectrum.

    The returned spectrum is the inverse-variance coadd of active-transit
    fractional flux residuals (T23 normally, T14 for a grazing transit),

        flux / median(flux_out_of_transit) - 1,

    after shifting each exposure into the planet rest frame using the fixed
    orbital semi-amplitude ``Kp_kms``. Absorption is therefore negative.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    jd = np.asarray(jd, dtype=float)

    wave_grid = wave

    order = np.argsort(wave_grid)
    wave_grid = wave_grid[order]
    flux = flux[:, order]
    error = error[:, order]

    bjd_tdb = get_bjd_tdb(jd, RA, Dec)
    epoch_bjd_tdb = get_ephemeris_epoch_bjd_tdb(
        transit_params['T0'],
        transit_params.get('T0_scale', transit_params.get('epoch_scale')),
        transit_params.get('T0_reference', transit_params.get('epoch_reference')),
    )
    orbital_phase = get_orbital_phase(
        bjd_tdb,
        epoch_bjd_tdb,
        transit_params['period'],
    )

    contacts = compute_contact_phases(transit_params)
    active_transit = active_transit_mask(orbital_phase, transit_params)
    active_interval = (
        "T14_grazing"
        if bool(transit_params.get("grazing_transit", False))
        else "T23"
    )
    out_transit = (orbital_phase < contacts["T1"]) | (orbital_phase > contacts["T4"])

    print(f"Active transit ({active_interval}): {np.sum(active_transit)} exposures")
    print(f"Out-of-transit: {np.sum(out_transit)} exposures")
    print(f"Orbital phase range: {orbital_phase.min():.4f} to {orbital_phase.max():.4f}")

    residual, residual_error = build_out_of_transit_residuals(
        flux,
        error,
        out_transit,
    )

    active_phase = orbital_phase[active_transit]
    planet_velocity = planet_radial_velocity_kms(
        active_phase,
        kp_kms=float(Kp_kms),
        eccentricity=float(transit_params.get("eccentricity", 0.0)),
        omega_deg=transit_params.get("omega"),
    )
    active_residual = residual[active_transit]
    active_error = residual_error[active_transit]

    shifted_residual = np.full_like(active_residual, np.nan)
    shifted_error = np.full_like(active_error, np.inf)
    for i, velocity_kms in enumerate(planet_velocity):
        beta = velocity_kms / _SPEED_OF_LIGHT_KMS
        doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta))
        observed_wavelength = wave_grid * doppler_factor
        shifted_residual[i] = np.interp(
            observed_wavelength,
            wave_grid,
            active_residual[i],
            left=np.nan,
            right=np.nan,
        )
        shifted_error[i] = np.interp(
            observed_wavelength,
            wave_grid,
            active_error[i],
            left=np.inf,
            right=np.inf,
        )

    valid_shifted = (
        np.isfinite(shifted_residual)
        & np.isfinite(shifted_error)
        & (shifted_error > 0.0)
    )
    full_coverage = np.all(valid_shifted, axis=0)
    if not np.any(full_coverage):
        raise ValueError(
            f"No wavelength columns retain coverage from every {active_interval} exposure "
            "after shifting into the planet rest frame."
        )

    wave_grid = wave_grid[full_coverage]
    shifted_residual = shifted_residual[:, full_coverage]
    shifted_error = shifted_error[:, full_coverage]
    weights = 1.0 / shifted_error**2
    weight_sum = np.sum(weights, axis=0)
    spectrum = np.sum(weights * shifted_residual, axis=0) / weight_sum
    spectrum_error = np.sqrt(1.0 / weight_sum)

    return (
        wave_grid,
        spectrum,
        spectrum_error,
        orbital_phase,
        active_transit,
        out_transit,
    )


def bin_spectrum(
    wave: np.ndarray, flux: np.ndarray, error: np.ndarray, bin_size: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance bin a 1D spectrum."""
    npix = len(wave)
    n_bins = npix // bin_size

    wave_binned = np.zeros(n_bins)
    flux_binned = np.zeros(n_bins)
    error_binned = np.zeros(n_bins)

    for i in range(n_bins):
        idx_start = i * bin_size
        idx_end = (i + 1) * bin_size

        wave_chunk = wave[idx_start:idx_end]
        flux_chunk = flux[idx_start:idx_end]
        error_chunk = error[idx_start:idx_end]
        valid = (
            np.isfinite(wave_chunk)
            & np.isfinite(flux_chunk)
            & np.isfinite(error_chunk)
            & (error_chunk > 0.0)
        )
        if not np.any(valid):
            wave_binned[i] = np.nan
            flux_binned[i] = np.nan
            error_binned[i] = np.inf
            continue
        weights = 1.0 / error_chunk[valid] ** 2
        weight_sum = np.sum(weights)
        wave_binned[i] = np.sum(weights * wave_chunk[valid]) / weight_sum
        flux_binned[i] = np.sum(weights * flux_chunk[valid]) / weight_sum
        error_binned[i] = np.sqrt(1.0 / weight_sum)

    return wave_binned, flux_binned, error_binned


def collapsed_transmission_dir(
    *,
    planet: str,
    epoch: str,
    arm: str,
) -> Path:
    """Return the standard directory for a collapsed transmission product."""
    return config_utils.get_collapsed_transmission_dir(
        planet=planet,
        epoch=epoch,
        arm=arm,
    )


def _load_transmission_collapse_source(
    data_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    metadata = json.loads((data_dir / "timeseries_prep.json").read_text(encoding="utf-8"))
    wavelength = np.asarray(np.load(data_dir / "wavelength.npy"), dtype=float)
    data = np.asarray(np.load(data_dir / "data.npy"), dtype=float)
    sigma = np.asarray(np.load(data_dir / "sigma.npy"), dtype=float)
    phase = np.asarray(np.load(data_dir / "phase.npy"), dtype=float)
    return wavelength, data, sigma, phase, metadata


def _load_required_lsd_shadow_source_model(
    data_dir: Path,
    *,
    metadata: dict,
    wavelength: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Read the source cube's fixed LSD shadow model."""
    shadow = metadata["fixed_doppler_shadow"]
    model = np.asarray(np.load(data_dir / shadow["source_model_file"]), dtype=float)
    return model, dict(shadow)


def collapse_transmission_epoch_arm(
    *,
    planet: str,
    ephemeris: str,
    shadow_source: str = "Recommended",
    epoch: str,
    arm: str,
    kp_kms: float,
    bin_size: int = 1,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Build one SYSREM-aware, active-transit 1D spectrum."""
    from dataio.collapse_emission_timeseries_to_1d import (
        COLLAPSE_COVERAGE_POLICY,
        apply_emission_collapse_operator,
        build_emission_collapse_operator,
        load_frozen_sysrem_arrays,
    )

    params = config_utils.resolve_parameter_domains(
        planet=planet,
        timing_source=ephemeris,
        shadow_source=shadow_source,
    )
    if source_dir is None:
        source_dir = config_utils.get_collapse_source_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
            mode="transmission",
        )
    else:
        source_dir = Path(source_dir)
    if output_dir is None:
        output_dir = collapsed_transmission_dir(
            planet=planet,
            epoch=epoch,
            arm=arm,
        )
    else:
        output_dir = Path(output_dir)
    wavelength, data, sigma, phase, source_metadata = (
        _load_transmission_collapse_source(source_dir)
    )
    fixed_source_model, fixed_shadow_metadata = (
        _load_required_lsd_shadow_source_model(
            source_dir,
            metadata=source_metadata,
            wavelength=wavelength,
            phase=phase,
        )
    )
    active_transit = active_transit_mask(phase, params)
    active_interval = (
        "T14_grazing" if bool(params.get("grazing_transit", False)) else "T23"
    )
    selected_indices = np.flatnonzero(active_transit)

    selected_phase = phase[selected_indices]
    selected_sigma = sigma[selected_indices]
    fixed = build_emission_collapse_operator(
        wavelength,
        selected_sigma,
        selected_phase,
        kp_kms=kp_kms,
        eccentricity=float(params.get("eccentricity", 0.0)),
        omega_deg=params.get("omega"),
        bin_size=bin_size,
    )
    wavelength_1d, spectrum_1d, uncertainty_1d = (
        apply_emission_collapse_operator(
            data[selected_indices],
            fixed,
        )
    )
    frozen_sysrem = load_frozen_sysrem_arrays(source_dir) if source_metadata["run_sysrem"] else None

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "wavelength_transmission.npy", wavelength_1d)
    np.save(output_dir / "spectrum_transmission.npy", spectrum_1d)
    np.save(
        output_dir / "uncertainty_transmission.npy",
        uncertainty_1d,
    )
    operator_name = "transmission_collapse_operator.npz"
    np.savez_compressed(
        output_dir / operator_name,
        schema_version=np.asarray(
            TRANSMISSION_COLLAPSE_SCHEMA_VERSION,
            dtype=np.int32,
        ),
        source_wavelength=wavelength,
        source_phase=phase,
        fixed_source_model=fixed_source_model,
        active_exposure_mask=active_transit.astype(float),
        active_transit_interval=np.asarray(active_interval),
        selected_exposure_indices=selected_indices.astype(np.int32),
        shift_left_indices=fixed["shift_left_indices"],
        shift_fractions=fixed["shift_fractions"],
        coadd_weights=fixed["coadd_weights"],
        bin_indices=fixed["bin_indices"],
        bin_weights=fixed["bin_weights"],
        output_wavelength=fixed["output_wavelength"],
        covered_source_indices=fixed["covered_source_indices"],
        n_source_wavelengths=fixed["n_source_wavelengths"],
        n_covered_wavelengths=fixed["n_covered_wavelengths"],
        n_dropped_out_of_bounds=fixed["n_dropped_out_of_bounds"],
        n_dropped_gap_crossing=fixed["n_dropped_gap_crossing"],
        max_native_gap_factor=fixed["max_native_gap_factor"],
        max_native_gap_angstrom=fixed["max_native_gap_angstrom"],
        coverage_policy=np.asarray(COLLAPSE_COVERAGE_POLICY),
        kp_reference_kms=np.asarray(kp_kms, dtype=float),
        planet_velocity_kms=fixed["planet_velocity_kms"],
        eccentricity=np.asarray(params.get("eccentricity", 0.0), dtype=float),
        omega_planet_deg=np.asarray(params.get("omega", np.nan), dtype=float),
        velocity_offset_reference_kms=np.asarray(0.0, dtype=float),
        has_sysrem=np.asarray(frozen_sysrem is not None, dtype=bool),
        **({} if frozen_sysrem is None else frozen_sysrem),
    )
    metadata = {
        "schema_version": TRANSMISSION_COLLAPSE_SCHEMA_VERSION,
        "product_kind": "collapsed_transmission_spectrum",
        "observable_kind": "continuum_removed_negative_transmission_flux",
        "model_preprocessing": (
            "fixed_shared_basis_lsd_shadow_then_"
            f"{active_interval.lower()}_visibility_then_frozen_per_pixel_sysrem_"
            "then_planet_frame_"
            "inverse_variance_coadd_then_subtract_inverse_variance_"
            "weighted_constant"
        ),
        "active_transit_interval": active_interval,
        "planet": planet,
        "ephemeris": ephemeris,
        "shadow_source": shadow_source,
        "parameter_resolution": params.get("parameter_resolution"),
        "epoch": epoch,
        "arm": arm,
        "kp_reference_kms": float(kp_kms),
        "planet_velocity_model": (
            "keplerian_transit_centered"
            if float(params.get("eccentricity", 0.0)) != 0.0
            else "circular_sinusoid"
        ),
        "eccentricity": float(params.get("eccentricity", 0.0)),
        "omega_planet_deg": (
            None
            if params.get("omega") is None
            else float(params.get("omega"))
        ),
        "velocity_offset_reference_kms": 0.0,
        "source_data_dir": str(source_dir),
        "source_run_sysrem": source_metadata.get("run_sysrem"),
        "fixed_doppler_shadow": fixed_shadow_metadata,
        "wavelength_medium": source_metadata.get("wavelength_medium"),
        "wavelength_frame": source_metadata.get("wavelength_frame"),
        "wavelength_frame_contract": source_metadata.get(
            "wavelength_frame_contract"
        ),
        "arm_edge_trim": source_metadata.get("arm_edge_trim"),
        "n_source_exposures": int(phase.size),
        "n_selected_exposures": int(selected_indices.size),
        "selected_exposure_indices": selected_indices.tolist(),
        "bin_size": int(bin_size),
        "collapse_operator_file": operator_name,
        "wavelength_coverage_policy": COLLAPSE_COVERAGE_POLICY,
        "n_source_wavelengths": int(wavelength.size),
        "n_covered_unbinned_wavelengths": int(
            fixed["n_covered_wavelengths"]
        ),
        "n_dropped_wavelengths": int(
            wavelength.size - int(fixed["n_covered_wavelengths"])
        ),
        "n_dropped_out_of_bounds": int(
            fixed["n_dropped_out_of_bounds"]
        ),
        "n_dropped_gap_crossing": int(
            fixed["n_dropped_gap_crossing"]
        ),
        "retained_unbinned_wavelength_fraction": float(
            int(fixed["n_covered_wavelengths"]) / wavelength.size
        ),
        "max_native_gap_factor": float(fixed["max_native_gap_factor"]),
        "max_native_gap_angstrom": float(
            fixed["max_native_gap_angstrom"]
        ),
        "shift_fraction_min": float(np.min(fixed["shift_fractions"])),
        "shift_fraction_max": float(np.max(fixed["shift_fractions"])),
        "n_output_wavelengths": int(wavelength_1d.size),
    }
    (output_dir / "collapse_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse dedicated transmission source cubes into SYSREM-aware "
            "active-transit 1D spectra."
        )
    )
    parser.add_argument("--planet", required=True)
    parser.add_argument("--ephemeris", default=EPHEMERIS)
    parser.add_argument("--shadow-source", default="Recommended")
    parser.add_argument("--epoch", nargs="+", required=True)
    parser.add_argument(
        "--arm",
        choices=["red", "blue", "full"],
        default=config.DEFAULT_DATA_ARM,
    )
    parser.add_argument("--kp-kms", type=float, default=None)
    parser.add_argument("--bin-size", type=int, default=1)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    params = config_utils.resolve_parameter_domains(
        planet=args.planet,
        timing_source=args.ephemeris,
        shadow_source=args.shadow_source,
    )
    kp_kms = params.get("Kp") if args.kp_kms is None else args.kp_kms
    arms = config.FULL_ARM_MEMBERS if args.arm == "full" else (args.arm,)
    for epoch in args.epoch:
        for arm in arms:
            result = collapse_transmission_epoch_arm(
                planet=args.planet,
                ephemeris=args.ephemeris,
                shadow_source=args.shadow_source,
                epoch=str(epoch),
                arm=arm,
                kp_kms=float(kp_kms),
                bin_size=int(args.bin_size),
            )
            print(
                f"READY {epoch}/{arm}/{result['active_transit_interval']}: "
                f"{result['n_selected_exposures']} exposures, "
                f"{result['n_output_wavelengths']} wavelengths"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
