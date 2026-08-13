import hashlib
import json
import os
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from astropy import constants as const
from astropy import units as u
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_median

from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.utils.grids import wav2nu
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import RJ, Rs

import config
import config_utils
from dataio.load import (
    load_nasa_archive_spectrum,
    load_observed_spectrum,
    parse_nasa_archive_tbl,
)
from dataio.collapse_transmission_timeseries_to_1d import get_sysrem_deep_mask
from dataio.lsd_doppler_shadow import (
    FIXED_LSD_SHADOW_METHOD,
    FIXED_LSD_SHADOW_SCHEMA_VERSION,
)
from physics.chemistry import (
    ConstantVMR,
    FastChemHybridChemistry,
    FastChemMetallicityEquilibriumChemistry,
    FreeVMR,
)
from physics.grid_setup import setup_wavenumber_grid, setup_spectral_operators
from opacities import (
    load_atomic_opacities,
    load_molecular_opacities,
    premodit_cache_signature,
    setup_cia_opacities,
)
from physics.model import (
    BandpassObservationInputs,
    ChunkedSysremInputs,
    CollapsedEmissionInputs,
    CollapsedTransmissionInputs,
    FrozenTimeseriesInputs,
    SpectroscopicObservationInputs,
    apply_collapsed_emission_operator,
    apply_collapsed_transmission_operator,
    apply_frozen_timeseries_operator,
    compute_model_timeseries,
    compute_atmospheric_state_from_posterior,
    reconstruct_temperature_profile,
    _bandpass_weighted_mean,
    _compute_native_observable_spectrum,
    _transform_bandpass_observable,
    apply_model_pipeline_corrections,
    build_atmosphere_region_config,
    build_bandpass_observation_config,
    build_shared_system_config,
    build_spectroscopic_observation_config,
    create_joint_retrieval_model,
)
from pipeline.inference import run_svi
from pipeline.mcmc_diagnostics import (
    DEFAULT_MCMC_EXTRA_FIELDS,
    get_extra_fields_by_chain,
    get_samples_by_chain,
    sanitize_diagnostic_label,
    save_chain_grouped_posterior,
    write_mcmc_diagnostics,
)
from dataio.bandpass import load_tess_bandpass
from plotting.plot import (
    plot_svi_loss,
    plot_transmission_spectrum,
    plot_emission_spectrum,
    plot_temperature_profile,
    plot_contribution_function,
    plot_contribution_per_species,
    plot_contribution_combined,
    save_retrieval_corner_plots,
)
from plotting.publication import (
    PUBLICATION_MODEL_DRAW_COUNT,
    PUBLICATION_TEMPERATURE_DRAW_COUNT,
    PublicationBundle,
    apply_planet_frame_operator,
    deterministic_draw_indices,
    plot_abundance_constraints,
    plot_bandpass_posterior_predictive,
    plot_kp_vsys_posterior,
    plot_likelihood_space_triptych,
    plot_mcmc_chain_traces,
    plot_planet_frame_posterior_predictive,
    plot_temperature_pressure_posterior,
    prepare_planet_frame_operator,
)
from plotting.transmission_diagnostics import (
    plot_pre_post_sysrem_comparison,
    plot_residual_quality_summary,
)


DEFAULT_ROTATION_VSINI_MAX_KMS = 100.0
STELLAR_ROTATION_VSINI_MARGIN = 1.10
COLLAPSED_EMISSION_OPERATOR_SCHEMA_VERSION = 4
COLLAPSED_TRANSMISSION_OPERATOR_SCHEMA_VERSION = 4


def _rotation_operator_vsini_max(
    stellar_vsini: float | None,
) -> float:
    """Return rotation-kernel support for the planet and star."""
    try:
        stellar_vsini_value = float(stellar_vsini)
    except (TypeError, ValueError):
        return DEFAULT_ROTATION_VSINI_MAX_KMS
    if not np.isfinite(stellar_vsini_value) or stellar_vsini_value <= 0.0:
        return DEFAULT_ROTATION_VSINI_MAX_KMS
    return max(
        DEFAULT_ROTATION_VSINI_MAX_KMS,
        STELLAR_ROTATION_VSINI_MARGIN * stellar_vsini_value,
    )


def _load_timeseries_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "timeseries_prep.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc


def _validate_collapsed_operator_arrays(
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    operator_path: Path,
    expected_schema_version: int,
    product_label: str,
) -> None:
    """Fail fast on obsolete or internally inconsistent collapse operators."""
    metadata_version = int(metadata.get("schema_version", -1))
    operator_version = int(
        np.asarray(arrays.get("schema_version", -1)).item()
    )
    if (
        metadata_version != expected_schema_version
        or operator_version != expected_schema_version
    ):
        raise ValueError(
            f"{product_label} collapse operator {operator_path} uses metadata/operator "
            f"schema {metadata_version}/{operator_version}; expected "
            f"{expected_schema_version}. Regenerate the collapsed product to "
            "remove unsafe planet-frame edge extrapolation."
        )

    source_wavelength = np.asarray(arrays["source_wavelength"], dtype=float)
    selected_indices = np.asarray(arrays["selected_exposure_indices"])
    left_indices = np.asarray(arrays["shift_left_indices"])
    fractions = np.asarray(arrays["shift_fractions"], dtype=float)
    coadd_weights = np.asarray(arrays["coadd_weights"], dtype=float)
    bin_indices = np.asarray(arrays["bin_indices"])
    bin_weights = np.asarray(arrays["bin_weights"], dtype=float)
    output_wavelength = np.asarray(arrays["output_wavelength"], dtype=float)
    covered_source_indices = np.asarray(arrays["covered_source_indices"])
    maximum_gap = float(np.asarray(arrays["max_native_gap_angstrom"]).item())

    if left_indices.ndim != 2 or left_indices.shape[0] != selected_indices.size:
        raise ValueError(
            f"{operator_path} shift operator shape {left_indices.shape} does not "
            f"match {selected_indices.size} selected exposures."
        )
    if fractions.shape != left_indices.shape or coadd_weights.shape != left_indices.shape:
        raise ValueError(
            f"{operator_path} shift indices, fractions, and coadd weights must "
            "have identical shapes."
        )
    if np.any(left_indices < 0) or np.any(
        left_indices + 1 >= source_wavelength.size
    ):
        raise ValueError(f"{operator_path} contains out-of-range source indices.")
    if np.any(~np.isfinite(fractions)) or np.any(
        (fractions < 0.0) | (fractions > 1.0)
    ):
        raise ValueError(
            f"{operator_path} contains non-finite or extrapolating shift fractions."
        )
    if np.any(~np.isfinite(coadd_weights)) or np.any(coadd_weights < 0.0):
        raise ValueError(f"{operator_path} contains invalid coadd weights.")
    if not np.allclose(
        np.sum(coadd_weights, axis=0),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(f"{operator_path} coadd weights do not sum to one.")
    if covered_source_indices.shape != (left_indices.shape[1],):
        raise ValueError(
            f"{operator_path} covered_source_indices does not match the shifted "
            "wavelength count."
        )
    if np.any(covered_source_indices < 0) or np.any(
        covered_source_indices >= source_wavelength.size
    ):
        raise ValueError(
            f"{operator_path} contains out-of-range covered source indices."
        )
    if np.any(np.diff(covered_source_indices) <= 0):
        raise ValueError(
            f"{operator_path} covered source indices must be strictly increasing."
        )
    bracket_width = (
        source_wavelength[left_indices + 1]
        - source_wavelength[left_indices]
    )
    if not np.isfinite(maximum_gap) or maximum_gap <= 0.0 or np.any(
        bracket_width > maximum_gap * (1.0 + 1.0e-12)
    ):
        raise ValueError(
            f"{operator_path} contains an interpolation bracket across a "
            "native or masked wavelength gap."
        )
    if bin_indices.shape != (left_indices.shape[1],) or bin_weights.shape != bin_indices.shape:
        raise ValueError(
            f"{operator_path} must assign every shifted wavelength to one bin."
        )
    if np.any(bin_indices < 0) or np.any(bin_indices >= output_wavelength.size):
        raise ValueError(f"{operator_path} contains out-of-range bin indices.")
    if np.any(~np.isfinite(bin_weights)) or np.any(bin_weights <= 0.0):
        raise ValueError(f"{operator_path} contains invalid bin weights.")
    bin_weight_sums = np.bincount(
        bin_indices.astype(np.int64),
        weights=bin_weights,
        minlength=output_wavelength.size,
    )
    if not np.allclose(
        bin_weight_sums,
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(f"{operator_path} bin weights do not sum to one.")


def _load_collapsed_emission_operator(
    data_dir: Path,
    target_wavelength: np.ndarray,
) -> CollapsedEmissionInputs:
    """Load and align the fixed preprocessing operator for a 1D emission product."""
    metadata_path = data_dir / "collapse_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Collapsed emission spectrum {data_dir} is missing collapse_metadata.json."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {metadata_path}: {exc}") from exc
    operator_name = metadata.get(
        "collapse_operator_file",
        "emission_collapse_operator.npz",
    )
    operator_path = data_dir / str(operator_name)
    if not operator_path.exists():
        raise FileNotFoundError(
            f"Collapsed emission spectrum {data_dir} is missing "
            f"{operator_path.name}. Regenerate the collapsed product so the "
            "forward model can receive the same time-domain preprocessing."
        )

    required = {
        "schema_version",
        "source_wavelength",
        "source_phase",
        "selected_exposure_indices",
        "shift_left_indices",
        "shift_fractions",
        "coadd_weights",
        "bin_indices",
        "bin_weights",
        "output_wavelength",
        "covered_source_indices",
        "max_native_gap_angstrom",
        "kp_reference_kms",
        "velocity_offset_reference_kms",
    }
    with np.load(operator_path) as raw:
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(
                f"{operator_path} is missing required arrays: {', '.join(missing)}."
            )
        arrays = {name: np.asarray(raw[name]) for name in raw.files}

    _validate_collapsed_operator_arrays(
        arrays=arrays,
        metadata=metadata,
        operator_path=operator_path,
        expected_schema_version=COLLAPSED_EMISSION_OPERATOR_SCHEMA_VERSION,
        product_label="Emission",
    )

    source_wavelength = np.asarray(arrays["source_wavelength"], dtype=float)
    output_wavelength = np.asarray(arrays["output_wavelength"], dtype=float)
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    if source_wavelength.ndim != 1 or np.any(np.diff(source_wavelength) <= 0.0):
        raise ValueError(
            f"{operator_path} source_wavelength must be strictly increasing."
        )
    if output_wavelength.ndim != 1 or np.any(np.diff(output_wavelength) <= 0.0):
        raise ValueError(
            f"{operator_path} output_wavelength must be strictly increasing."
        )
    output_indices: list[int] = []
    for wavelength in target_wavelength:
        insertion = int(np.searchsorted(output_wavelength, wavelength))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < output_wavelength.size
        ]
        if not candidates:
            raise ValueError(
                f"Wavelength {wavelength:.12g} is absent from {operator_path}."
            )
        best = min(
            candidates,
            key=lambda index: abs(output_wavelength[index] - wavelength),
        )
        if not np.isclose(
            output_wavelength[best],
            wavelength,
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError(
                f"Wavelength {wavelength:.12g} is not represented by "
                f"{operator_path}; closest value is {output_wavelength[best]:.12g}."
            )
        output_indices.append(best)

    # ExoJAX requires increasing wavenumber. Reversing the increasing-wavelength
    # source grid supplies that order; the model operator reverses it back before
    # applying the saved wavelength-space interpolation.
    source_inst_nus = wav2nu(source_wavelength[::-1], "AA")
    chunked_sysrem = None
    has_sysrem = bool(np.asarray(arrays.get("has_sysrem", False)).item())
    if has_sysrem:
        sysrem_required = {
            "sysrem_U",
            "sysrem_chunk_labels",
            "sysrem_projection_sigma",
        }
        missing_sysrem = sorted(sysrem_required.difference(arrays))
        if missing_sysrem:
            raise ValueError(
                f"{operator_path} declares has_sysrem=true but is missing: "
                f"{', '.join(missing_sysrem)}."
            )
        sysrem_bundle = _validate_sysrem_inputs(
            {
                "U_sysrem": arrays["sysrem_U"],
                "chunk_labels": arrays["sysrem_chunk_labels"],
                "projection_sigma": arrays["sysrem_projection_sigma"],
            },
            n_exp=np.asarray(arrays["source_phase"]).size,
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
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Collapsed transmission spectrum {data_dir} is missing "
            "collapse_metadata.json."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {metadata_path}: {exc}") from exc
    operator_path = data_dir / str(
        metadata.get(
            "collapse_operator_file",
            "transmission_collapse_operator.npz",
        )
    )
    required = {
        "schema_version",
        "source_wavelength",
        "source_phase",
        "fixed_source_model",
        "active_exposure_mask",
        "selected_exposure_indices",
        "shift_left_indices",
        "shift_fractions",
        "coadd_weights",
        "bin_indices",
        "bin_weights",
        "output_wavelength",
        "covered_source_indices",
        "max_native_gap_angstrom",
        "kp_reference_kms",
        "velocity_offset_reference_kms",
    }
    if not operator_path.exists():
        raise FileNotFoundError(
            f"Collapsed transmission spectrum {data_dir} is missing "
            f"{operator_path.name}."
        )
    with np.load(operator_path) as raw:
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(
                f"{operator_path} is missing required arrays: "
                f"{', '.join(missing)}."
            )
        arrays = {name: np.asarray(raw[name]) for name in raw.files}

    _validate_collapsed_operator_arrays(
        arrays=arrays,
        metadata=metadata,
        operator_path=operator_path,
        expected_schema_version=COLLAPSED_TRANSMISSION_OPERATOR_SCHEMA_VERSION,
        product_label="Transmission",
    )

    source_wavelength = np.asarray(arrays["source_wavelength"], dtype=float)
    source_phase = np.asarray(arrays["source_phase"], dtype=float)
    fixed_source_model = np.asarray(arrays["fixed_source_model"], dtype=float)
    output_wavelength = np.asarray(arrays["output_wavelength"], dtype=float)
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    if source_wavelength.ndim != 1 or np.any(np.diff(source_wavelength) <= 0.0):
        raise ValueError(
            f"{operator_path} source_wavelength must be strictly increasing."
        )
    expected_shadow_shape = (source_phase.size, source_wavelength.size)
    if (
        fixed_source_model.shape != expected_shadow_shape
        or np.any(~np.isfinite(fixed_source_model))
    ):
        raise ValueError(
            f"{operator_path} fixed_source_model must be finite with shape "
            f"{expected_shadow_shape}; got {fixed_source_model.shape}."
        )
    shadow_metadata = metadata.get("fixed_doppler_shadow")
    if (
        not isinstance(shadow_metadata, dict)
        or not bool(shadow_metadata.get("enabled", False))
        or int(shadow_metadata.get("schema_version", -1))
        != FIXED_LSD_SHADOW_SCHEMA_VERSION
        or str(shadow_metadata.get("method")) != FIXED_LSD_SHADOW_METHOD
    ):
        raise ValueError(
            f"{metadata_path} does not declare the required shared-basis LSD "
            "Doppler-shadow contract."
        )
    if output_wavelength.ndim != 1 or np.any(np.diff(output_wavelength) <= 0.0):
        raise ValueError(
            f"{operator_path} output_wavelength must be strictly increasing."
        )
    output_indices: list[int] = []
    for wavelength in target_wavelength:
        insertion = int(np.searchsorted(output_wavelength, wavelength))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < output_wavelength.size
        ]
        if not candidates:
            raise ValueError(
                f"Wavelength {wavelength:.12g} is absent from {operator_path}."
            )
        best = min(
            candidates,
            key=lambda index: abs(output_wavelength[index] - wavelength),
        )
        if not np.isclose(
            output_wavelength[best],
            wavelength,
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError(
                f"Wavelength {wavelength:.12g} is not represented by "
                f"{operator_path}; closest value is "
                f"{output_wavelength[best]:.12g}."
            )
        output_indices.append(best)

    chunked_sysrem = None
    has_sysrem = bool(np.asarray(arrays.get("has_sysrem", False)).item())
    if has_sysrem:
        sysrem_required = {
            "sysrem_U",
            "sysrem_chunk_labels",
            "sysrem_projection_sigma",
        }
        missing_sysrem = sorted(sysrem_required.difference(arrays))
        if missing_sysrem:
            raise ValueError(
                f"{operator_path} declares has_sysrem=true but is missing: "
                f"{', '.join(missing_sysrem)}."
            )
        chunked_sysrem = _build_model_chunked_sysrem(
            _validate_sysrem_inputs(
                {
                    "U_sysrem": arrays["sysrem_U"],
                    "chunk_labels": arrays["sysrem_chunk_labels"],
                    "projection_sigma": arrays["sysrem_projection_sigma"],
                },
                n_exp=np.asarray(arrays["source_phase"]).size,
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


def _validate_no_deep_telluric_columns(
    data_dir: Path,
    wavelength: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> None:
    arm = metadata.get("arm")
    if not arm:
        arm = data_dir.name if data_dir.name in {"blue", "red"} else None
    candidate_arms = (str(arm),) if arm in {"blue", "red"} else ("red", "blue")

    wave = np.asarray(wavelength, dtype=float)
    for candidate_arm in candidate_arms:
        deep_mask = get_sysrem_deep_mask(wave, candidate_arm)
        n_deep = int(np.count_nonzero(deep_mask))
        if not n_deep:
            continue

        deep_wave = wave[deep_mask]
        raise ValueError(
            f"{data_dir} still contains {n_deep} configured {candidate_arm}-arm "
            f"deep-telluric wavelength columns "
            f"({float(np.nanmin(deep_wave)) / 10.0:.3f}-"
            f"{float(np.nanmax(deep_wave)) / 10.0:.3f} nm). "
            "This usually means the prepared time-series bundle is stale relative to the current "
            "deep-mask configuration. Regenerate it with the relevant "
            "`python -m dataio.prepare_*_retrieval_timeseries ... --run-sysrem` command."
        )


def load_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)

    expected_paths = {
        "wavelength": data_dir / "wavelength.npy",
        "data": data_dir / "data.npy",
        "sigma": data_dir / "sigma.npy",
        "phase": data_dir / "phase.npy",
    }
    missing = [path.name for path in expected_paths.values() if not path.exists()]
    if missing:
        transmission_spectrum_paths = [
            data_dir / "wavelength_transmission.npy",
            data_dir / "spectrum_transmission.npy",
            data_dir / "uncertainty_transmission.npy",
        ]
        emission_spectrum_paths = [
            data_dir / "wavelength_emission.npy",
            data_dir / "spectrum_emission.npy",
            data_dir / "uncertainty_emission.npy",
        ]
        found_transmission_spectrum = all(path.exists() for path in transmission_spectrum_paths)
        found_emission_spectrum = all(path.exists() for path in emission_spectrum_paths)

        if found_transmission_spectrum or found_emission_spectrum:
            found_names = transmission_spectrum_paths if found_transmission_spectrum else emission_spectrum_paths
            found_label = "transmission" if found_transmission_spectrum else "emission"
            found_text = ", ".join(path.name for path in found_names)
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                f"{data_dir} does not contain a retrieval-ready time-series bundle; missing {missing_text}. "
                f"Found {found_label} single-spectrum products instead ({found_text}). "
                "Use --data-format spectrum for the collapsed 1D products, or run "
                "`python -m dataio.prepare_retrieval_timeseries ... --run-sysrem` "
                "to generate wavelength.npy/data.npy/sigma.npy/phase.npy for a time-series retrieval."
            )

        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"{data_dir} is missing the time-series files required for data_format='timeseries': "
            f"{missing_text}."
        )

    wavelength = np.load(expected_paths["wavelength"])
    data = np.load(expected_paths["data"])
    sigma = np.load(expected_paths["sigma"])
    phase = np.load(expected_paths["phase"])
    metadata = _load_timeseries_metadata(data_dir)
    _validate_no_deep_telluric_columns(data_dir, wavelength, metadata=metadata)

    return wavelength, data, sigma, phase


def load_pre_sysrem_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    data_dir = Path(data_dir)
    data_path = data_dir / "pre_sysrem_data.npy"
    sigma_path = data_dir / "pre_sysrem_sigma.npy"

    if not data_path.exists() and not sigma_path.exists():
        return None
    if not data_path.exists() or not sigma_path.exists():
        print(
            "  Warning: incomplete pre-SYSREM diagnostic bundle in "
            f"{data_dir}; expected both pre_sysrem_data.npy and pre_sysrem_sigma.npy."
        )
        return None

    return np.load(data_path), np.load(sigma_path)


def _normalize_phoenix_cache_dir(path: str | Path | None) -> Path:
    candidate = config.PHOENIX_CACHE_DIR if path is None else Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


@lru_cache(maxsize=None)
def _read_phoenix_surface_flux_cache(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    try:
        with np.load(path) as raw:
            wavelength_angstrom = np.asarray(raw["wavelength_angstrom"], dtype=float)
            stellar_surface_flux = np.asarray(raw["stellar_surface_flux"], dtype=float)
    except Exception as exc:
        raise ValueError(f"Failed to read cached PHOENIX spectrum from {path}: {exc}") from exc

    if wavelength_angstrom.ndim != 1 or stellar_surface_flux.ndim != 1:
        raise ValueError(
            f"Cached PHOENIX spectrum {path} must contain 1D wavelength_angstrom and "
            "stellar_surface_flux arrays."
        )
    if wavelength_angstrom.shape != stellar_surface_flux.shape:
        raise ValueError(
            f"Cached PHOENIX spectrum {path} has mismatched shapes "
            f"{wavelength_angstrom.shape} and {stellar_surface_flux.shape}."
        )
    return wavelength_angstrom, stellar_surface_flux


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _derive_stellar_logg_cgs(Mstar_msun: float, Rstar_rsun: float) -> float:
    g_cgs = (const.G * (Mstar_msun * const.M_sun) / (Rstar_rsun * const.R_sun) ** 2).to_value(u.cm / u.s**2)
    return float(np.log10(g_cgs))


def _resolve_chromatic_phoenix_parameters(
    *,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> tuple[float, float, float]:
    temperature = _coerce_optional_float(Tstar)
    if temperature is None:
        raise ValueError(
            f"Emission component '{component_name}' requires a finite Tstar for chromatic "
            "PHOENIX retrieval."
        )

    resolved_logg = _coerce_optional_float(logg_star)
    if resolved_logg is None:
        Mstar_val = _coerce_optional_float(Mstar)
        Rstar_val = _coerce_optional_float(Rstar)
        if Mstar_val is None or Rstar_val is None:
            raise ValueError(
                f"Emission component '{component_name}' requires either a finite logg_star "
                "or both finite M_star and R_star for chromatic PHOENIX retrieval."
            )
        resolved_logg = _derive_stellar_logg_cgs(Mstar_val, Rstar_val)

    resolved_metallicity = _coerce_optional_float(metallicity)
    if resolved_metallicity is None:
        resolved_metallicity = 0.0

    return temperature, resolved_logg, resolved_metallicity


def _format_phoenix_cache_float(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _build_phoenix_surface_flux_cache_path(
    *,
    cache_dir: Path,
    temperature: float,
    logg: float,
    metallicity: float,
    target_wavelength_angstrom: np.ndarray,
) -> Path:
    cache_key = hashlib.sha1()
    cache_key.update(b"phoenix-surface-flux-v1\0")
    cache_key.update(
        np.asarray(
            [temperature, logg, metallicity],
            dtype=np.float64,
        ).tobytes()
    )
    cache_key.update(
        np.asarray(target_wavelength_angstrom, dtype=np.float64).tobytes()
    )
    input_hash = cache_key.hexdigest()[:16]
    filename = (
        "phoenix_"
        f"T{_format_phoenix_cache_float(temperature)}_"
        f"logg{_format_phoenix_cache_float(logg)}_"
        f"feh{_format_phoenix_cache_float(metallicity)}_"
        f"{input_hash}.npz"
    )
    return cache_dir / filename


def _convert_chromatic_surface_flux_to_exojax_units(
    wavelength: u.Quantity,
    surface_flux: u.Quantity,
) -> np.ndarray:
    wavelength_cm = u.Quantity(wavelength).to(u.cm)
    energy_per_photon = (const.h * const.c / wavelength_cm) / u.photon
    surface_flux_lambda = (u.Quantity(surface_flux) * energy_per_photon).to(u.erg / (u.s * u.cm**2 * u.cm))
    surface_flux_wavenumber = surface_flux_lambda * wavelength_cm**2
    return np.asarray(
        surface_flux_wavenumber.to_value(u.erg / (u.s * u.cm)),
        dtype=float,
    )


def _load_chromatic_phoenix_surface_flux_on_grid(
    *,
    nu_grid: np.ndarray,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    phoenix_cache_dir: str | Path | None,
) -> np.ndarray:
    target_wavelength_angstrom = 1.0e8 / np.asarray(nu_grid, dtype=float)
    temperature, resolved_logg, resolved_metallicity = _resolve_chromatic_phoenix_parameters(
        component_name=component_name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    cache_dir = _normalize_phoenix_cache_dir(phoenix_cache_dir)
    cache_path = _build_phoenix_surface_flux_cache_path(
        cache_dir=cache_dir,
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        target_wavelength_angstrom=target_wavelength_angstrom,
    )

    if cache_path.exists():
        cached_wavelength_angstrom, cached_flux = (
            _read_phoenix_surface_flux_cache(str(cache_path))
        )
        if cached_wavelength_angstrom.shape == target_wavelength_angstrom.shape and np.array_equal(
            cached_wavelength_angstrom,
            target_wavelength_angstrom,
        ):
            return cached_flux

    try:
        from chromatic import get_phoenix_photons
    except ImportError as exc:
        raise ImportError(
            "chromatic-lightcurves is required to auto-fetch and cache PHOENIX spectra. "
            "Install chromatic-lightcurves before running an emission retrieval."
        ) from exc

    sort_idx = np.argsort(target_wavelength_angstrom)
    query_wavelength_um = (target_wavelength_angstrom[sort_idx] / 1.0e4) * u.micron
    wavelength_um, surface_flux = get_phoenix_photons(
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        wavelength=query_wavelength_um,
    )
    flux_sorted = _convert_chromatic_surface_flux_to_exojax_units(
        wavelength=wavelength_um,
        surface_flux=surface_flux,
    )
    stellar_surface_flux = np.empty_like(flux_sorted)
    stellar_surface_flux[sort_idx] = flux_sorted

    if np.any(~np.isfinite(stellar_surface_flux)):
        raise ValueError(
            f"chromatic returned non-finite PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )
    if np.any(stellar_surface_flux <= 0):
        raise ValueError(
            f"chromatic returned non-positive PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )

    np.savez_compressed(
        cache_path,
        wavelength_angstrom=target_wavelength_angstrom,
        stellar_surface_flux=stellar_surface_flux,
        temperature=np.asarray(temperature, dtype=float),
        logg=np.asarray(resolved_logg, dtype=float),
        metallicity=np.asarray(resolved_metallicity, dtype=float),
        cache_schema_version=np.asarray(1, dtype=int),
    )
    return stellar_surface_flux


def _load_phoenix_surface_flux_on_grid(
    *,
    phoenix_cache_dir: str | Path | None,
    nu_grid: np.ndarray,
    mode: str,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> np.ndarray | None:
    if _normalize_retrieval_mode(mode) != "emission":
        return None
    return _load_chromatic_phoenix_surface_flux_on_grid(
        nu_grid=nu_grid,
        component_name=component_name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
        phoenix_cache_dir=phoenix_cache_dir,
    )


@dataclass(frozen=True)
class SysremInputBundle:
    U: np.ndarray | None = None
    V: np.ndarray | None = None
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
    target_wavelength: np.ndarray,
    selected_phase: np.ndarray,
    *,
    require_lsd_shadow: bool = False,
) -> FrozenTimeseriesOperatorSpec:
    data_dir = Path(data_dir)
    metadata = _load_timeseries_metadata(data_dir)
    operator_name = metadata.get(
        "timeseries_operator_file",
        "timeseries_operator.npz",
    )
    operator_path = data_dir / str(operator_name)
    if not operator_path.exists():
        raise FileNotFoundError(
            f"Prepared time series {data_dir} is missing {operator_path.name}. "
            "Regenerate it so the forward model can replay preprocessing on "
            "the complete source exposure sequence before row selection."
        )

    required = {
        "schema_version",
        "source_wavelength",
        "source_phase",
        "source_bjd_tdb",
        "active_exposure_mask",
        "selected_exposure_indices",
        "subtract_time_median",
        "has_sysrem",
    }
    with np.load(operator_path) as raw:
        missing = sorted(required.difference(raw.files))
        if missing:
            raise ValueError(
                f"{operator_path} is missing required arrays: "
                f"{', '.join(missing)}. Regenerate the prepared time series."
            )
        schema_version = int(np.asarray(raw["schema_version"]).item())
        source_wavelength = np.asarray(raw["source_wavelength"], dtype=float)
        source_phase = _normalize_phase(
            np.asarray(raw["source_phase"], dtype=float)
        )
        source_bjd_tdb = np.asarray(raw["source_bjd_tdb"], dtype=float)
        active_mask = np.asarray(
            raw["active_exposure_mask"],
            dtype=float,
        )
        selected_indices = np.asarray(
            raw["selected_exposure_indices"],
            dtype=int,
        )
        subtract_time_median = bool(
            np.asarray(raw["subtract_time_median"]).item()
        )
        has_sysrem = bool(np.asarray(raw["has_sysrem"]).item())

    if schema_version != 1:
        raise ValueError(
            f"Unsupported frozen time-series operator schema {schema_version} "
            f"in {operator_path}; expected schema 1."
        )
    target_wavelength = np.asarray(target_wavelength, dtype=float)
    if source_wavelength.ndim != 1 or target_wavelength.ndim != 1:
        raise ValueError(
            f"{operator_path} source and prepared wavelength grids must be 1D."
        )
    if np.any(~np.isfinite(source_wavelength)) or np.any(
        np.diff(source_wavelength) <= 0.0
    ):
        raise ValueError(
            f"{operator_path} source_wavelength must be finite and strictly increasing."
        )
    if source_wavelength.shape != target_wavelength.shape or not np.allclose(
        source_wavelength,
        target_wavelength,
        rtol=1.0e-12,
        atol=1.0e-8,
    ):
        raise ValueError(
            f"{operator_path} source_wavelength does not match wavelength.npy. "
            "Regenerate the prepared bundle atomically."
        )
    if source_phase.ndim != 1 or active_mask.shape != source_phase.shape:
        raise ValueError(
            f"{operator_path} source_phase and active_exposure_mask must be "
            "matching 1D arrays."
        )
    if np.any(~np.isfinite(source_phase)):
        raise ValueError(f"{operator_path} source_phase must be finite.")
    if source_bjd_tdb.shape != source_phase.shape or np.any(
        ~np.isfinite(source_bjd_tdb)
    ):
        raise ValueError(
            f"{operator_path} source_bjd_tdb must be finite and match source_phase."
        )
    if selected_indices.ndim != 1:
        raise ValueError(
            f"{operator_path} selected_exposure_indices must be 1D."
        )
    selected_phase = _normalize_phase(np.asarray(selected_phase, dtype=float))
    if selected_indices.size != selected_phase.size:
        raise ValueError(
            f"{operator_path} selects {selected_indices.size} rows but phase.npy "
            f"contains {selected_phase.size}."
        )
    if np.any(selected_indices < 0) or np.any(
        selected_indices >= source_phase.size
    ):
        raise ValueError(
            f"{operator_path} contains out-of-range selected exposure indices."
        )
    if np.unique(selected_indices).size != selected_indices.size or np.any(
        np.diff(selected_indices) <= 0
    ):
        raise ValueError(
            f"{operator_path} selected exposure indices must be unique and "
            "strictly increasing."
        )
    if not np.allclose(
        source_phase[selected_indices],
        selected_phase,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            f"{operator_path} selected phases do not match phase.npy."
        )
    if np.any(~np.isfinite(active_mask)) or np.any(active_mask < 0.0) or np.any(
        active_mask > 1.0
    ):
        raise ValueError(
            f"{operator_path} active_exposure_mask must be finite and within [0, 1]."
        )

    selected_bjd_path = data_dir / "bjd_tdb.npy"
    if not selected_bjd_path.exists():
        raise FileNotFoundError(
            f"Prepared time series {data_dir} is missing bjd_tdb.npy. "
            "Regenerate it with the canonical timing contract."
        )
    selected_bjd_tdb = np.asarray(np.load(selected_bjd_path), dtype=float)
    if selected_bjd_tdb.shape != selected_phase.shape or not np.allclose(
        source_bjd_tdb[selected_indices],
        selected_bjd_tdb,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            f"{operator_path} selected source times do not match bjd_tdb.npy."
        )

    metadata_run_sysrem = metadata.get("run_sysrem")
    if metadata_run_sysrem is not None and bool(metadata_run_sysrem) != has_sysrem:
        raise ValueError(
            f"{operator_path} has_sysrem={has_sysrem} disagrees with "
            f"timeseries_prep.json run_sysrem={metadata_run_sysrem}."
        )
    metadata_subtract_median = metadata.get("subtract_median")
    if (
        metadata_subtract_median is not None
        and bool(metadata_subtract_median) != subtract_time_median
    ):
        raise ValueError(
            f"{operator_path} subtract_time_median={subtract_time_median} "
            "disagrees with timeseries_prep.json."
        )
    sysrem_path = data_dir / "U_sysrem.npz"
    if has_sysrem != sysrem_path.exists():
        raise ValueError(
            f"{operator_path} has_sysrem={has_sysrem}, but "
            f"U_sysrem.npz existence is {sysrem_path.exists()}."
        )

    metadata_n_source = metadata.get("n_source_exposures")
    if metadata_n_source is not None and int(metadata_n_source) != source_phase.size:
        raise ValueError(
            f"{operator_path} contains {source_phase.size} source rows, but "
            f"timeseries_prep.json records n_source_exposures={metadata_n_source}."
        )
    metadata_n_selected = metadata.get("n_exposures")
    if metadata_n_selected is not None and int(metadata_n_selected) != selected_indices.size:
        raise ValueError(
            f"{operator_path} selects {selected_indices.size} rows, but "
            f"timeseries_prep.json records n_exposures={metadata_n_selected}."
        )
    metadata_selected_indices = metadata.get("selected_exposure_indices")
    if metadata_selected_indices is not None and not np.array_equal(
        np.asarray(metadata_selected_indices, dtype=int),
        selected_indices,
    ):
        raise ValueError(
            f"{operator_path} selected_exposure_indices disagree with "
            "timeseries_prep.json. Regenerate the prepared bundle atomically."
        )

    fixed_source_model = None
    shadow_metadata = metadata.get("fixed_doppler_shadow")
    shadow_enabled = isinstance(shadow_metadata, dict) and bool(
        shadow_metadata.get("enabled", False)
    )
    if require_lsd_shadow and not shadow_enabled:
        raise ValueError(
            f"{data_dir} is a transmission time series without the required "
            "shared-basis LSD Doppler-shadow model. Regenerate both prepared "
            "source products and run python -m spectroscopy.doppler_shadow."
        )
    if shadow_enabled:
        shadow_schema = int(shadow_metadata.get("schema_version", -1))
        if shadow_schema != FIXED_LSD_SHADOW_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported fixed Doppler-shadow schema {shadow_schema} in "
                f"{data_dir / 'timeseries_prep.json'}; expected "
                f"{FIXED_LSD_SHADOW_SCHEMA_VERSION}."
            )
        if str(shadow_metadata.get("method")) != FIXED_LSD_SHADOW_METHOD:
            raise ValueError(
                f"{data_dir} does not use the required "
                f"{FIXED_LSD_SHADOW_METHOD} shadow method."
            )
        source_model_name = str(
            shadow_metadata.get("source_model_file", "shadow_source_model.npy")
        )
        source_model_path = (data_dir / source_model_name).resolve()
        if source_model_path.parent != data_dir.resolve():
            raise ValueError("Fixed Doppler-shadow source model must live in its data directory.")
        if not source_model_path.is_file():
            raise FileNotFoundError(
                f"{data_dir} declares a fixed Doppler shadow but is missing "
                f"{source_model_path.name}."
            )
        expected_sha256 = str(shadow_metadata.get("source_model_sha256", ""))
        actual_sha256 = hashlib.sha256(source_model_path.read_bytes()).hexdigest()
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise ValueError(
                f"Fixed Doppler-shadow source-model hash mismatch for {source_model_path}. "
                "Re-run python -m spectroscopy.doppler_shadow."
            )
        fixed_source_model = np.asarray(np.load(source_model_path), dtype=float)
        expected_shape = (source_phase.size, source_wavelength.size)
        if fixed_source_model.shape != expected_shape:
            raise ValueError(
                f"{source_model_path} has shape {fixed_source_model.shape}; expected "
                f"source exposure x wavelength shape {expected_shape}."
            )
        if np.any(~np.isfinite(fixed_source_model)):
            raise ValueError(f"{source_model_path} contains non-finite values.")

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
    if operator.has_sysrem != (sysrem is not None):
        raise ValueError(
            "Frozen time-series operator and retrieval SYSREM settings disagree: "
            f"operator has_sysrem={operator.has_sysrem}, "
            f"retrieval supplied SYSREM={sysrem is not None}."
        )
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
    if chunk_labels.ndim != 1:
        raise ValueError(f"chunk_labels must be 1D, got shape {chunk_labels.shape}.")
    if chunk_labels.size == 0:
        raise ValueError("chunk_labels is empty.")
    if np.any(chunk_labels < 0):
        raise ValueError("chunk_labels must be non-negative for all wavelength columns.")

    labels = sorted(int(label) for label in np.unique(chunk_labels))
    expected = list(range(len(labels)))
    if labels != expected:
        raise ValueError(f"chunk_labels must be contiguous and start at 0; got labels {labels}.")

    return tuple(np.where(chunk_labels == label)[0].astype(int) for label in labels)


def _load_sysrem_inputs(data_dir: str | Path) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)

    u_candidates = [
        data_dir / "U_sysrem.npz",
        data_dir / "U.npy",
        data_dir / "U_sysrem.npy",
    ]
    v_candidates = [
        data_dir / "V.npy",
        data_dir / "V_diag.npy",
        data_dir / "inv_sigma.npy",
        data_dir / "invsigma.npy",
    ]

    u_path = next((p for p in u_candidates if p.exists()), None)
    v_path = next((p for p in v_candidates if p.exists()), None)

    if u_path is None:
        raise FileNotFoundError(f"No SYSREM basis file found in {data_dir}.")

    if u_path.suffix == ".npz":
        with np.load(u_path) as u_data:
            raw = {}
            for name in u_data.files:
                raw[name] = np.asarray(u_data[name])
        if "chunk_labels" in raw:
            return raw
        if v_path is not None:
            raw["V"] = np.load(v_path)
        return raw

    raw = {"U": np.load(u_path)}
    if v_path is not None:
        raw["V"] = np.load(v_path)
    return raw


def _validate_sysrem_inputs(
    raw: dict[str, np.ndarray],
    n_exp: int,
) -> SysremInputBundle:
    if "chunk_labels" in raw:
        U = np.asarray(raw["U_sysrem"] if "U_sysrem" in raw else raw["U"])
        if U.ndim == 2:
            U = U[:, :, np.newaxis]
        if U.ndim != 3:
            raise ValueError(f"Chunked SYSREM U must have shape (n_exp, n_basis, n_chunks); got {U.shape}.")
        if U.shape[0] != n_exp:
            raise ValueError(f"U exposure axis mismatch: U.shape[0]={U.shape[0]} but n_exp={n_exp}.")

        chunk_indices = _chunk_indices_from_labels(raw["chunk_labels"])
        n_chunks = len(chunk_indices)
        if U.shape[2] != n_chunks:
            raise ValueError(f"Chunk count mismatch: U has {U.shape[2]} chunks but chunk_labels encodes {n_chunks}.")

        projection_sigma = raw.get("projection_sigma")
        if projection_sigma is None:
            if raw.get("V_chunk_diag") is not None:
                raise ValueError(
                    "This SYSREM bundle uses the retired V_chunk_diag chunk "
                    "approximation. Regenerate the prepared time series to save "
                    "the required per-pixel projection_sigma matrix."
                )
            raise ValueError(
                "Chunked SYSREM bundle is missing projection_sigma. Regenerate "
                "the prepared time series with the current preparation code."
            )
        projection_sigma = np.asarray(projection_sigma, dtype=float)
        expected_sigma_shape = (
            n_exp,
            np.asarray(raw["chunk_labels"]).size,
        )
        if projection_sigma.shape != expected_sigma_shape:
            raise ValueError(
                "projection_sigma shape mismatch: got "
                f"{projection_sigma.shape}, expected {expected_sigma_shape}."
            )
        if np.any(~np.isfinite(projection_sigma)) or np.any(
            projection_sigma <= 0.0
        ):
            raise ValueError(
                "projection_sigma must contain only finite positive values."
            )

        U_chunks: list[np.ndarray] = []
        for chunk in range(n_chunks):
            U_chunk = np.asarray(U[:, :, chunk], dtype=float)
            keep = np.any(np.isfinite(U_chunk), axis=0)
            U_chunk = U_chunk[:, keep]
            if U_chunk.ndim != 2 or U_chunk.shape[0] != n_exp:
                raise ValueError(f"Chunk {chunk} has invalid U shape {U_chunk.shape} for n_exp={n_exp}.")
            if np.any(~np.isfinite(U_chunk)):
                raise ValueError(
                    f"Chunk {chunk} has non-finite values in active SYSREM basis columns."
                )

            U_chunks.append(U_chunk)

        return SysremInputBundle(
            chunk_indices=tuple(chunk_indices),
            U_chunks=tuple(U_chunks),
            projection_sigma=projection_sigma,
        )

    U = np.asarray(raw["U_sysrem"] if "U_sysrem" in raw else raw["U"])
    V_raw = raw.get("V")
    V = None if V_raw is None else np.asarray(V_raw)

    if U.ndim == 3:
        raise ValueError(
            "3D SYSREM inputs now require an explicit chunked bundle with chunk_labels and "
            "projection_sigma in U_sysrem.npz. Legacy 3D U arrays without chunk metadata are unsupported."
        )

    if U.shape[0] != n_exp:
        raise ValueError(f"U exposure axis mismatch: U.shape[0]={U.shape[0]} but n_exp={n_exp}.")
    if V is None:
        return SysremInputBundle(U=U)
    if V.ndim == 1:
        if V.size != n_exp:
            raise ValueError(f"V exposure axis mismatch: V.size={V.size} but n_exp={n_exp}.")
        V = np.diag(V)
    elif V.ndim == 2:
        expected_shape = (n_exp, n_exp)
        if V.shape != expected_shape:
            raise ValueError(f"V shape mismatch: V.shape={V.shape} but expected {expected_shape}.")

    return SysremInputBundle(U=U, V=V)


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

    v_description = "unused" if sysrem.V is None else str(sysrem.V.shape)
    return (
        f"U shape={sysrem.U.shape}, per-pixel sigma from observation, "
        f"legacy V shape={v_description}"
    )


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
        V=(
            None
            if sysrem.V is None
            else np.asarray(sysrem.V)[np.ix_(indices, indices)]
        ),
    )


def _validate_spectral_subset(
    spectral_stride: int,
    spectral_offset: int,
) -> tuple[int, int]:
    try:
        stride = int(spectral_stride)
        offset = int(spectral_offset)
    except Exception as exc:
        raise ValueError("spectral_stride and spectral_offset must be integers.") from exc
    if stride < 1:
        raise ValueError("spectral_stride must be >= 1.")
    if offset < 0:
        raise ValueError("spectral_offset must be >= 0.")
    if offset >= stride:
        raise ValueError("spectral_offset must be smaller than spectral_stride.")
    return stride, offset


def _spectral_subset_indices(
    n_wave: int,
    spectral_stride: int,
    spectral_offset: int,
) -> np.ndarray:
    stride, offset = _validate_spectral_subset(spectral_stride, spectral_offset)
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
    stride, offset = _validate_spectral_subset(spectral_stride, spectral_offset)
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


def _build_composition_solver(
    chemistry_model: str,
    fastchem_parameter_file: str | None,
):
    model = chemistry_model.lower().strip()
    if model == "constant":
        return ConstantVMR()

    if model == "free":
        return FreeVMR()

    if model == "fastchem_hybrid_grid":
        parameter_file = fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
        if parameter_file is None:
            raise ValueError(
                "chemistry_model='fastchem_hybrid_grid' requires a FastChem "
                "parameters.dat path. Pass --fastchem-parameter-file or set "
                "FASTCHEM_PARAMETER_FILE in config."
            )
        parameter_file = Path(parameter_file).expanduser()
        if not parameter_file.exists():
            raise FileNotFoundError(
                "chemistry_model='fastchem_hybrid_grid' requires an existing "
                f"FastChem parameter file, but '{parameter_file}' was not found."
            )

        solver = FastChemHybridChemistry(
            fastchem_parameter_file=str(parameter_file),
            continuum_species=tuple(config.FASTCHEM_HYBRID_CONTINUUM_SPECIES),
            metallicity_range=tuple(config.FASTCHEM_HYBRID_METALLICITY_RANGE),
            co_ratio_range=tuple(config.FASTCHEM_HYBRID_CO_RATIO_RANGE),
            n_metallicity=int(config.FASTCHEM_HYBRID_N_METALLICITY),
            n_co_ratio=int(config.FASTCHEM_HYBRID_N_CO_RATIO),
            log_vmr_min=float(config.LOG_VMR_MIN),
            log_vmr_max=float(config.LOG_VMR_MAX),
            h2_he_ratio=float(config.H2_HE_RATIO),
            n_temp=int(config.FASTCHEM_N_TEMP),
            n_pressure=int(config.FASTCHEM_N_PRESSURE),
            t_min=float(config.FASTCHEM_T_MIN),
            t_max=float(config.FASTCHEM_T_MAX),
            cache_dir=config.FASTCHEM_CACHE_DIR,
        )
        if not solver.requires_hybrid_parameters():
            raise ValueError(
                "chemistry_model='fastchem_hybrid_grid' requires hidden continuum "
                "drivers including at least 'H'/'H I' and 'e-' in "
                "FASTCHEM_HYBRID_CONTINUUM_SPECIES."
            )
        return solver

    if model == "fastchem_equilibrium_metallicity_grid":
        parameter_file = fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
        if parameter_file is None:
            raise ValueError(
                "chemistry_model='fastchem_equilibrium_metallicity_grid' requires "
                "a FastChem parameters.dat path. Pass fastchem_parameter_file or "
                "set FASTCHEM_PARAMETER_FILE in config."
            )
        parameter_file = Path(parameter_file).expanduser()
        if not parameter_file.exists():
            raise FileNotFoundError(
                "chemistry_model='fastchem_equilibrium_metallicity_grid' requires "
                f"an existing FastChem parameter file, but '{parameter_file}' was not found."
            )
        return FastChemMetallicityEquilibriumChemistry(
            fastchem_parameter_file=str(parameter_file),
            continuum_species=tuple(config.FASTCHEM_HYBRID_CONTINUUM_SPECIES),
            metallicity_range=tuple(config.FASTCHEM_HYBRID_METALLICITY_RANGE),
            co_ratio_range=tuple(config.FASTCHEM_HYBRID_CO_RATIO_RANGE),
            n_metallicity=int(config.FASTCHEM_HYBRID_N_METALLICITY),
            n_co_ratio=int(config.FASTCHEM_HYBRID_N_CO_RATIO),
            h2_he_ratio=float(config.H2_HE_RATIO),
            n_temp=int(config.FASTCHEM_N_TEMP),
            n_pressure=int(config.FASTCHEM_N_PRESSURE),
            t_min=float(config.FASTCHEM_T_MIN),
            t_max=float(config.FASTCHEM_T_MAX),
            cache_dir=config.FASTCHEM_CACHE_DIR,
        )

    raise ValueError(
        f"Unknown chemistry_model: {chemistry_model}. "
        "Choose from {'constant', 'free', 'fastchem_hybrid_grid', "
        "'fastchem_equilibrium_metallicity_grid'}."
    )


def _preflight_spectrum_checks(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    inst_nus: np.ndarray,
) -> None:
    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)
    phase = np.asarray(phase)
    inst_nus = np.asarray(inst_nus)

    if sigma.shape != data.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match data shape {data.shape}")

    if data.ndim == 1:
        if data.size != wav_obs.size:
            raise ValueError(f"data length {data.size} != wavelength length {wav_obs.size}")
        expected_exposures = 1
    else:
        if data.shape[1] != wav_obs.size:
            raise ValueError(f"data spectral axis {data.shape[1]} != wavelength length {wav_obs.size}")
        expected_exposures = data.shape[0]

    if phase.size != expected_exposures:
        raise ValueError(f"phase length {phase.size} != number of exposures {expected_exposures}")


def _preflight_grid_checks(inst_nus: np.ndarray, nu_grid: np.ndarray) -> None:
    inst_nus = np.asarray(inst_nus)
    nu_grid = np.asarray(nu_grid)

    inst_min = np.nanmin(inst_nus)
    inst_max = np.nanmax(inst_nus)
    nu_min = np.nanmin(nu_grid)
    nu_max = np.nanmax(nu_grid)

    if inst_min < nu_min or inst_max > nu_max:
        raise ValueError(
            "instrument wavenumber grid is outside model grid: "
            f"inst_nus={inst_min:.4g}..{inst_max:.4g}, "
            f"nu_grid={nu_min:.4g}..{nu_max:.4g}"
        )


def _sample_svi_posterior(
    guide: object | None,
    params: dict | None,
    rng_key: jax.Array,
    num_samples: int,
) -> dict[str, np.ndarray] | None:
    if guide is None or params is None or num_samples <= 0:
        return None

    try:
        svi_draws = guide.sample_posterior(
            rng_key,
            params,
            sample_shape=(num_samples,),
        )
    except Exception as exc:
        print(f"  Warning: failed to sample SVI posterior for corner plots: {exc}")
        return None

    svi_draws_np = {}
    for name, values in svi_draws.items():
        svi_draws_np[name] = np.asarray(jax.device_get(values))
    return svi_draws_np


def _summarize_observed_spectrum(
    data: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data_arr = np.asarray(data)
    sigma_arr = np.asarray(sigma)

    if data_arr.ndim == 1:
        return data_arr, sigma_arr

    valid = (
        np.isfinite(data_arr)
        & np.isfinite(sigma_arr)
        & (sigma_arr > 0.0)
    )
    weights = np.where(valid, 1.0 / np.square(sigma_arr), 0.0)
    weight_sum = np.sum(weights, axis=0)
    obs_mean = np.divide(
        np.sum(np.where(valid, weights * data_arr, 0.0), axis=0),
        weight_sum,
        out=np.full(data_arr.shape[1], np.nan),
        where=weight_sum > 0.0,
    )
    obs_err = np.divide(
        1.0,
        np.sqrt(weight_sum),
        out=np.full(data_arr.shape[1], np.nan),
        where=weight_sum > 0.0,
    )
    return obs_mean, obs_err


def _validate_mcmc_device_layout(
    *,
    num_chains: int,
    chain_method: str,
    require_gpu_per_chain: bool,
) -> None:
    chain_method = str(chain_method).strip().lower()
    local_devices = list(jax.local_devices())
    gpu_devices = []
    for device in local_devices:
        if device.platform == "gpu":
            gpu_devices.append(device)

    print(f"  JAX default backend: {jax.default_backend()}")
    print(f"  JAX local devices: {local_devices}")

    if not require_gpu_per_chain:
        return

    if chain_method != "parallel":
        raise RuntimeError("MCMC_REQUIRE_GPU_PER_CHAIN requires MCMC_CHAIN_METHOD='parallel'.")

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "This run requires GPU-backed parallel chains, but JAX default backend "
            f"is {jax.default_backend()!r}."
        )

    if len(gpu_devices) < num_chains:
        raise RuntimeError(
            f"This run requires at least {num_chains} visible GPU devices for "
            f"{num_chains} MCMC chains, but JAX sees {len(gpu_devices)} GPU device(s). "
            "Request more GPUs or reduce --mcmc-chains."
        )


def _posterior_site_value(
    params: dict[str, float],
    site_name: str,
    *,
    sample_prefix: str | None = None,
    default: float | None = None,
):
    if sample_prefix is not None:
        scoped_name = f"{sample_prefix}/{site_name}"
        if scoped_name in params:
            return params[scoped_name]
    return params.get(site_name, default)


def _synthesize_timeseries_from_atmospheric_state(
    *,
    atmo_state: dict,
    model_params: dict,
    region_config: object,
    component: "SpectroscopicComponentBundle",
) -> np.ndarray:
    params = atmo_state["params"]
    observation_config = component.observation_config

    def _resolved_velocity_offset(base_offset: float, *, use_shared: bool) -> float:
        mode = region_config.velocity_offset_mode
        if mode == "shared":
            return float(params.get("v_sys", base_offset)) if use_shared else float(base_offset)
        if mode == "region":
            region_offset = _posterior_site_value(
                params,
                "delta_v",
                sample_prefix=region_config.sample_prefix,
                default=0.0,
            )
            return float(base_offset) + float(region_offset)
        if mode in {"species", "none"}:
            return float(base_offset)
        raise ValueError(f"Unsupported velocity offset mode: {mode!r}")

    # Existing posterior artifacts store this sampled site as ``Rp``.  Its
    # explicit runtime meaning is the adopted transmission reference radius
    # (and, for emission, the same planet radius used for projected area).
    R_ref_rj = float(params.get("Rp", model_params["R_p"]))
    Mp_mj = float(params.get("Mp", model_params["M_p"]))
    Rstar_rs = float(params.get("Rstar", model_params["R_star"]))

    radius_btm_cm = R_ref_rj * RJ
    Rstar_cm = Rstar_rs * Rs
    gravity_btm = gravity_surface(R_ref_rj, Mp_mj)

    dtau = jnp.asarray(atmo_state["dtau"])
    Tarr = jnp.asarray(atmo_state["Tarr"])
    mmw_profile = jnp.asarray(atmo_state["mmw"])
    collapsed_emission = component.observation_inputs.collapsed_emission
    collapsed_transmission = (
        component.observation_inputs.collapsed_transmission
    )
    frozen_timeseries = component.observation_inputs.frozen_timeseries
    collapsed_operator = (
        collapsed_emission
        if collapsed_emission is not None
        else collapsed_transmission
    )
    phase = np.asarray(component.phase)
    if collapsed_operator is not None:
        phase = np.asarray(collapsed_operator.source_phase)
        Kp_kms = float(collapsed_operator.kp_reference_kms)
        v_sys_kms = _resolved_velocity_offset(
            float(collapsed_operator.velocity_offset_reference_kms),
            use_shared=False,
        )
        model_inst_nus = collapsed_operator.source_inst_nus
    elif frozen_timeseries is not None:
        phase = np.asarray(frozen_timeseries.source_phase)
        Kp_kms = float(params.get("Kp", model_params["Kp"]))
        v_sys_kms = _resolved_velocity_offset(0.0, use_shared=True)
        model_inst_nus = jnp.asarray(component.inst_nus)
    elif observation_config.radial_velocity_mode == "none":
        phase = np.zeros_like(phase)
        Kp_kms = 0.0
        v_sys_kms = _resolved_velocity_offset(0.0, use_shared=False)
        model_inst_nus = jnp.asarray(component.inst_nus)
    else:
        Kp_kms = float(params.get("Kp", model_params["Kp"]))
        v_sys_kms = _resolved_velocity_offset(0.0, use_shared=True)
        model_inst_nus = jnp.asarray(component.inst_nus)

    model_ts = compute_model_timeseries(
        mode=observation_config.mode,
        art=region_config.art,
        dtau=dtau,
        Tarr=Tarr,
        mmw_profile=mmw_profile,
        radius_btm=radius_btm_cm,
        Rstar=Rstar_cm,
        gravity_btm=gravity_btm,
        phase=jnp.asarray(phase),
        Kp=Kp_kms,
        v_sys=v_sys_kms,
        sop_rot=component.sop_rot,
        sop_inst=component.sop_inst,
        inst_nus=model_inst_nus,
        nu_grid=jnp.asarray(component.nu_grid),
        beta_inst=observation_config.beta_inst,
        period_day=float(model_params["period"]),
        Tstar=observation_config.Tstar,
        stellar_surface_flux=observation_config.stellar_surface_flux,
    )
    if collapsed_emission is not None:
        model_ts = apply_collapsed_emission_operator(
            model_ts,
            collapsed_emission,
        )
    elif collapsed_transmission is not None:
        model_ts = apply_collapsed_transmission_operator(
            model_ts,
            collapsed_transmission,
        )
    elif frozen_timeseries is not None:
        model_ts = apply_frozen_timeseries_operator(
            model_ts,
            frozen_timeseries,
        )
    model_ts = apply_model_pipeline_corrections(
        model_ts,
        subtract_weighted_global_mean=observation_config.subtract_weighted_global_mean,
        apply_sysrem=(
            observation_config.apply_sysrem and frozen_timeseries is None
        ),
        sigma=jnp.asarray(component.sigma),
        U=None if component.sysrem is None or component.sysrem.U is None else jnp.asarray(component.sysrem.U),
        V=None if component.sysrem is None or component.sysrem.V is None else jnp.asarray(component.sysrem.V),
        chunked_sysrem=(
            None
            if frozen_timeseries is not None
            else _build_model_chunked_sysrem(component.sysrem)
        ),
    )

    return np.asarray(jax.device_get(model_ts))


def _compute_model_timeseries_for_plot(
    *,
    posterior_samples: dict[str, np.ndarray],
    model_params: dict,
    region_config: object,
    component: "SpectroscopicComponentBundle",
    region_sample_prefix: str | None,
    atmo_state: dict | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    try:
        if atmo_state is None:
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_samples,
                region_config=region_config,
                opa_mols=component.opa_mols,
                opa_atoms=component.opa_atoms,
                opa_cias=component.opa_cias,
                nu_grid=component.nu_grid,
                use_median=True,
                sample_prefix=region_sample_prefix,
            )

        model_ts = _synthesize_timeseries_from_atmospheric_state(
            atmo_state=atmo_state,
            model_params=model_params,
            region_config=region_config,
            component=component,
        )
        return model_ts, atmo_state
    except Exception as exc:
        print(f"  Warning: failed to build diagnostic spectrum plot data: {exc}")
        return None, atmo_state


def _component_output_filename(
    base_filename: str,
    component_name: str,
    *,
    num_components: int,
) -> str:
    if num_components <= 1:
        return base_filename
    stem, ext = os.path.splitext(base_filename)
    return f"{stem}_{component_name}{ext}"


@dataclass(frozen=True)
class SpectroscopicComponentBundle:
    name: str
    wav_obs: np.ndarray
    grid_source_wavelength_range: tuple[float, float]
    data: np.ndarray
    sigma: np.ndarray
    pre_sysrem_data: np.ndarray | None
    pre_sysrem_sigma: np.ndarray | None
    phase: np.ndarray
    sysrem: SysremInputBundle | None
    inst_nus: np.ndarray
    nu_grid: np.ndarray
    sop_rot: object
    sop_inst: object
    instrument_resolution: float
    opa_cias: dict
    opa_mols: dict
    opa_atoms: dict
    observation_config: object
    observation_inputs: SpectroscopicObservationInputs


@dataclass(frozen=True)
class BandpassConstraintBundle:
    name: str
    observation_config: object
    observation_inputs: BandpassObservationInputs


def _append_resolved_spectral_grid_config(
    output_dir: str | Path,
    components: Sequence[SpectroscopicComponentBundle],
) -> None:
    """Append the data-resolved model grids after observations are loaded."""
    log_path = Path(output_dir) / "run_config.log"
    with log_path.open("a") as handle:
        handle.write("\nRESOLVED SPECTRAL GRIDS\n")
        handle.write("-" * 70 + "\n")
        for component in components:
            observed_min, observed_max = component.grid_source_wavelength_range
            model_wavelength = 1.0e8 / np.asarray(component.nu_grid, dtype=float)
            model_min = float(np.min(model_wavelength))
            model_max = float(np.max(model_wavelength))
            signature = premodit_cache_signature(
                component.nu_grid,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                config.PREMODIT_CUTWING,
            )
            handle.write(f"Component: {component.name}\n")
            handle.write(
                "  Observed wavelength range before thinning: "
                f"{observed_min:.6f} - {observed_max:.6f} Angstroms\n"
            )
            handle.write(
                f"  Model wavelength range: {model_min:.6f} - "
                f"{model_max:.6f} Angstroms\n"
            )
            handle.write(f"  Model spectral points: {component.nu_grid.size}\n")
            handle.write("  Grid source: complete prepared wavelength array before thinning\n")
            handle.write(f"  Opacity grid signature: {signature}\n")


def _posterior_sample_count(samples: dict[str, np.ndarray]) -> int:
    sizes = []
    for values in samples.values():
        array = np.asarray(values)
        if array.ndim > 0:
            sizes.append(int(array.shape[0]))
    return min(sizes) if sizes else 0


def _posterior_draw_subset(
    samples: dict[str, np.ndarray],
    index: int,
) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {}
    for name, values in samples.items():
        array = np.asarray(values)
        subset[name] = array if array.ndim == 0 else array[index : index + 1]
    return subset


def _posterior_parameter_median(
    samples: dict[str, np.ndarray],
    basename: str,
    default: float,
) -> float:
    keys = [name for name in samples if name == basename]
    if not keys:
        keys = [name for name in samples if name.rsplit("/", 1)[-1] == basename]
    if not keys:
        return float(default)
    values = np.asarray(samples[sorted(keys)[0]], dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float(default)


def _temperature_profile_draws_for_publication(
    *,
    posterior_samples: dict[str, np.ndarray],
    art: object,
    pt_profile: str,
    sample_prefix: str | None,
    Tint_fixed: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = _posterior_sample_count(posterior_samples)
    indices = deterministic_draw_indices(
        sample_count,
        PUBLICATION_TEMPERATURE_DRAW_COUNT,
    )
    profiles: list[np.ndarray] = []
    successful_indices: list[int] = []
    for index in indices:
        sample_params: dict[str, np.ndarray] = {}
        for key, values in posterior_samples.items():
            array = np.asarray(values)
            sample_params[key] = array if array.ndim == 0 else array[index]
        try:
            profile = reconstruct_temperature_profile(
                sample_params,
                art,
                pt_profile=pt_profile,
                Tint_fixed=Tint_fixed,
                sample_prefix=sample_prefix,
            )
        except Exception:
            continue
        profile = np.asarray(profile, dtype=float)
        if profile.ndim != 1 or np.any(~np.isfinite(profile)):
            continue
        profiles.append(profile)
        successful_indices.append(int(index))
    if not profiles:
        raise ValueError("No finite posterior temperature profiles could be reconstructed.")
    return np.asarray(profiles), np.asarray(successful_indices, dtype=int)


def _component_publication_model_draws(
    *,
    posterior_samples: dict[str, np.ndarray],
    model_params: dict,
    region_config: object,
    component: SpectroscopicComponentBundle,
    region_sample_prefix: str | None,
    prepared_operator: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if region_config.velocity_offset_mode == "species":
        raise ValueError(
            "Publication model reconstruction for species-specific velocity offsets "
            "is not yet lossless; refusing to plot an approximate surrogate."
        )
    sample_count = _posterior_sample_count(posterior_samples)
    indices = deterministic_draw_indices(sample_count, PUBLICATION_MODEL_DRAW_COUNT)
    spectra: list[np.ndarray] = []
    successful_indices: list[int] = []
    warnings: list[str] = []
    for index in indices:
        subset = _posterior_draw_subset(posterior_samples, int(index))
        model_ts, _ = _compute_model_timeseries_for_plot(
            posterior_samples=subset,
            model_params=model_params,
            region_config=region_config,
            component=component,
            region_sample_prefix=region_sample_prefix,
        )
        if model_ts is None:
            warnings.append(f"posterior draw {int(index)} model reconstruction failed")
            continue
        model_array = np.asarray(model_ts, dtype=float)
        try:
            if prepared_operator is None:
                spectrum = model_array.reshape(-1, model_array.shape[-1])[0]
            else:
                _, spectrum, _ = apply_planet_frame_operator(
                    model_array,
                    prepared_operator,
                )
        except Exception as exc:
            warnings.append(f"posterior draw {int(index)} collapse failed: {exc}")
            continue
        if np.any(~np.isfinite(spectrum)):
            warnings.append(f"posterior draw {int(index)} produced non-finite spectrum")
            continue
        spectra.append(np.asarray(spectrum, dtype=float))
        successful_indices.append(int(index))
    if not spectra:
        raise ValueError("No posterior model draws could be reconstructed for this component.")
    return (
        np.asarray(spectra),
        np.asarray(successful_indices, dtype=int),
        warnings,
    )


def _bandpass_publication_model_draws(
    *,
    posterior_samples: dict[str, np.ndarray],
    model_params: dict,
    region_config: object,
    component: BandpassConstraintBundle,
    region_sample_prefix: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Reconstruct scalar likelihood predictions for deterministic HMC draws."""
    sample_count = _posterior_sample_count(posterior_samples)
    indices = deterministic_draw_indices(sample_count, PUBLICATION_MODEL_DRAW_COUNT)
    model_draws: list[float] = []
    thermal_draws: list[float] = []
    reflected_draws: list[float] = []
    successful_indices: list[int] = []
    warnings: list[str] = []
    observation_config = component.observation_config
    site_prefix = "".join(
        character if character.isalnum() else "_"
        for character in str(observation_config.sample_prefix or component.name)
    ).strip("_")
    albedo_site = f"{site_prefix}_geometric_albedo"
    effective_site = f"{site_prefix}_effective_model"

    for index in indices:
        subset = _posterior_draw_subset(posterior_samples, int(index))
        try:
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=subset,
                region_config=region_config,
                opa_mols=observation_config.opa_mols,
                opa_atoms=observation_config.opa_atoms,
                opa_cias=observation_config.opa_cias,
                nu_grid=observation_config.nu_grid,
                use_median=True,
                sample_prefix=region_sample_prefix,
            )
            params = atmo_state["params"]
            radius_rj = float(params.get("Rp", model_params["R_p"]))
            mass_mj = float(params.get("Mp", model_params["M_p"]))
            stellar_radius_rs = float(params.get("Rstar", model_params["R_star"]))
            radius_cm = radius_rj * RJ
            stellar_radius_cm = stellar_radius_rs * Rs
            gravity_btm = gravity_surface(radius_rj, mass_mj)
            spectrum = _compute_native_observable_spectrum(
                mode=observation_config.mode,
                art=region_config.art,
                dtau=jnp.asarray(atmo_state["dtau"]),
                Tarr=jnp.asarray(atmo_state["Tarr"]),
                mmw_profile=jnp.asarray(atmo_state["mmw"]),
                radius_btm=radius_cm,
                Rstar=stellar_radius_cm,
                gravity_btm=gravity_btm,
                nu_grid=observation_config.nu_grid,
                Tstar=observation_config.Tstar,
                stellar_surface_flux=observation_config.stellar_surface_flux,
            )
            observable_spectrum = _transform_bandpass_observable(
                spectrum,
                observation_config.observable,
            )
            thermal = _bandpass_weighted_mean(
                observable_spectrum,
                observation_config.nu_grid,
                observation_config.wavelength_m,
                observation_config.response,
                photon_weighted=observation_config.photon_weighted,
            )
            thermal_value = float(np.asarray(jax.device_get(thermal)))
            reflected_value = 0.0
            if observation_config.mode == "emission" and observation_config.include_reflection:
                if albedo_site in subset:
                    albedo = float(np.asarray(subset[albedo_site]).reshape(-1)[0])
                else:
                    bounds = observation_config.geometric_albedo_bounds
                    if bounds is None or bounds[0] != bounds[1]:
                        raise KeyError(f"Missing sampled geometric albedo site {albedo_site!r}.")
                    albedo = float(bounds[0])
                semi_major_axis_m = float(observation_config.semi_major_axis_au) * config.AU_M
                radius_m = radius_cm * 1.0e-2
                reflected_value = albedo * (radius_m / semi_major_axis_m) ** 2
            physical_value = thermal_value + reflected_value
            if effective_site in subset:
                predictive_value = float(np.asarray(subset[effective_site]).reshape(-1)[0])
            else:
                predictive_value = physical_value
            values = (predictive_value, thermal_value, reflected_value)
            if not np.all(np.isfinite(values)):
                raise ValueError("Bandpass reconstruction produced non-finite values.")
        except Exception as exc:
            warnings.append(f"posterior draw {int(index)} reconstruction failed: {exc}")
            continue
        model_draws.append(predictive_value)
        thermal_draws.append(thermal_value)
        reflected_draws.append(reflected_value)
        successful_indices.append(int(index))

    if not model_draws:
        raise ValueError("No posterior bandpass predictions could be reconstructed.")
    components = {
        "thermal_component": np.asarray(thermal_draws),
        "reflected_component": np.asarray(reflected_draws),
    }
    return (
        np.asarray(model_draws),
        np.asarray(successful_indices, dtype=int),
        components,
        warnings,
    )


def _generate_publication_bundle(
    *,
    output_dir: str | Path,
    mode: str,
    epochs: Sequence[str],
    pt_profile: str,
    chemistry_model: str,
    posterior_samples: dict[str, np.ndarray],
    posterior_by_chain: dict[str, np.ndarray],
    svi_samples: dict[str, np.ndarray] | None,
    svi_losses: np.ndarray | None,
    model_params: dict,
    atmosphere_region_lookup: dict[str, object],
    spectroscopic_components: Sequence[SpectroscopicComponentBundle],
    bandpass_components: Sequence[BandpassConstraintBundle],
    compute_contribution: bool,
) -> dict[str, Any]:
    """Create deterministic paper/supplement/QC figures after HMC completes."""
    bundle = PublicationBundle(
        run_dir=Path(output_dir),
        metadata={
            "target": config.PLANET,
            "ephemeris": config.EPHEMERIS,
            "retrieval_mode": mode,
            "epochs": list(epochs),
            "pt_profile": pt_profile,
            "chemistry_model": chemistry_model,
            "inference": "HMC/NUTS",
            "svi_role": "initialization diagnostic only",
            "posterior_model_draw_target": PUBLICATION_MODEL_DRAW_COUNT,
            "temperature_profile_draw_target": PUBLICATION_TEMPERATURE_DRAW_COUNT,
            "spectroscopic_components": [item.name for item in spectroscopic_components],
            "bandpass_components": [item.name for item in bandpass_components],
        },
    )

    # Temperature-pressure posterior: one core paper figure per atmosphere.
    multiple_regions = len(atmosphere_region_lookup) > 1
    for region_name, region_config in atmosphere_region_lookup.items():
        region_id = "".join(
            character if character.isalnum() else "_" for character in region_name
        ).strip("_").lower()
        figure_id = (
            f"temperature_pressure_profile_{region_id}"
            if multiple_regions
            else "temperature_pressure_profile"
        )
        try:
            temperature_draws, temperature_indices = _temperature_profile_draws_for_publication(
                posterior_samples=posterior_samples,
                art=region_config.art,
                pt_profile=region_config.pt_profile,
                sample_prefix=region_config.sample_prefix,
                Tint_fixed=region_config.Tint_fixed,
            )
            figure, plotted = plot_temperature_pressure_posterior(
                pressure_bar=np.asarray(region_config.art.pressure),
                temperature_draws_K=temperature_draws,
                profile_label=f"{region_name}; {region_config.pt_profile}",
            )
            plotted["posterior_draw_indices"] = temperature_indices
            bundle.save_figure(
                figure,
                figure_id=figure_id,
                tier="paper",
                required=True,
                plotted_data=plotted,
                metadata={
                    "atmosphere_region": region_name,
                    "posterior_draw_count": int(temperature_draws.shape[0]),
                },
            )
        except Exception as exc:
            bundle.record_failure(
                figure_id=figure_id,
                tier="paper",
                required=True,
                error=exc,
                metadata={"atmosphere_region": region_name},
            )

    velocity_plot = plot_kp_vsys_posterior(posterior_samples)
    if velocity_plot is not None:
        figure, plotted = velocity_plot
        bundle.save_figure(
            figure,
            figure_id="kp_vsys_posterior",
            tier="paper",
            required=True,
            plotted_data=plotted,
            metadata={"velocity_frame": "stellar-rest wavelength frame"},
        )

    abundance_plot = plot_abundance_constraints(posterior_samples)
    if abundance_plot is not None:
        figure, plotted = abundance_plot
        bundle.save_figure(
            figure,
            figure_id="abundance_constraints",
            tier="paper",
            required=True,
            plotted_data=plotted,
            metadata={"intervals_percent": [2.5, 16.0, 50.0, 84.0, 97.5]},
        )

    # Chain-aware convergence visualization.
    chain_cpu = {
        name: np.asarray(jax.device_get(values))
        for name, values in posterior_by_chain.items()
    }
    chain_plot = plot_mcmc_chain_traces(chain_cpu)
    if chain_plot is not None:
        figure, plotted = chain_plot
        bundle.save_figure(
            figure,
            figure_id="mcmc_chain_traces",
            tier="qc",
            required=True,
            plotted_data=plotted,
        )
    else:
        bundle.record_failure(
            figure_id="mcmc_chain_traces",
            tier="qc",
            required=True,
            error="No scalar chain-grouped posterior parameters were available.",
        )

    if svi_losses is not None:
        svi_path = bundle.figure_path("svi_loss", "qc")
        try:
            plot_svi_loss(np.asarray(svi_losses), str(svi_path))
            bundle.register_existing(
                figure_id="svi_loss",
                tier="qc",
                path=svi_path,
                required=False,
                metadata={"role": "optimization diagnostic; not a scientific posterior"},
            )
        except Exception as exc:
            bundle.record_failure(
                figure_id="svi_loss",
                tier="qc",
                required=False,
                error=exc,
            )

    # HMC-only corner plot belongs in the supplement, never in the paper core.
    supplement_dir = bundle.figure_root / "supplement"
    try:
        save_retrieval_corner_plots(
            output_dir=str(supplement_dir),
            hmc_samples=posterior_samples,
            svi_samples=None,
        )
        bundle.register_existing(
            figure_id="corner_plot_hmc",
            tier="supplement",
            path=supplement_dir / "corner_plot_hmc.pdf",
            required=True,
        )
    except Exception as exc:
        bundle.record_failure(
            figure_id="corner_plot_hmc",
            tier="supplement",
            required=True,
            error=exc,
        )
    if svi_samples is not None:
        qc_dir = bundle.figure_root / "qc"
        try:
            save_retrieval_corner_plots(
                output_dir=str(qc_dir),
                hmc_samples=None,
                svi_samples=svi_samples,
            )
            bundle.register_existing(
                figure_id="corner_plot_svi",
                tier="qc",
                path=qc_dir / "corner_plot_svi.pdf",
                required=False,
                metadata={"role": "initialization comparison only"},
            )
        except Exception as exc:
            bundle.record_failure(
                figure_id="corner_plot_svi",
                tier="qc",
                required=False,
                error=exc,
            )

    shared_median_kp = _posterior_parameter_median(
        posterior_samples,
        "Kp",
        float(model_params["Kp"]),
    )
    shared_median_vsys = _posterior_parameter_median(posterior_samples, "v_sys", 0.0)
    epoch_label = ", ".join(epochs) if epochs else "unspecified epoch"

    for component in spectroscopic_components:
        region_config = atmosphere_region_lookup[component.observation_config.region_name]
        region_sample_prefix = region_config.sample_prefix
        component_id = "".join(
            character if character.isalnum() else "_" for character in component.name
        ).strip("_").lower()
        title_base = (
            f"{config.PLANET} {component.observation_config.mode}; "
            f"{epoch_label}; {component.name}"
        )
        median_model, median_atmo_state = _compute_model_timeseries_for_plot(
            posterior_samples=posterior_samples,
            model_params=model_params,
            region_config=region_config,
            component=component,
            region_sample_prefix=region_sample_prefix,
        )
        if median_model is None:
            error = "Posterior-median likelihood-space model reconstruction failed."
            bundle.record_failure(
                figure_id=f"likelihood_space_{component_id}",
                tier="paper",
                required=np.asarray(component.data).ndim == 2,
                error=error,
            )
            bundle.record_failure(
                figure_id=f"planet_frame_spectrum_{component_id}",
                tier="paper",
                required=True,
                error=error,
            )
            continue
        median_model = np.asarray(median_model, dtype=float)
        component_data = np.asarray(component.data, dtype=float)
        component_sigma = np.asarray(component.sigma, dtype=float)
        component_phase = np.asarray(component.phase, dtype=float)
        component_wave = np.asarray(component.wav_obs, dtype=float)
        median_kp = (
            shared_median_kp
            if component.observation_config.radial_velocity_mode == "orbital"
            else 0.0
        )
        if region_config.velocity_offset_mode == "shared":
            median_vsys = shared_median_vsys
        elif region_config.velocity_offset_mode == "region":
            median_vsys = float(
                _posterior_site_value(
                    median_atmo_state["params"],
                    "delta_v",
                    sample_prefix=region_config.sample_prefix,
                    default=0.0,
                )
            )
        else:
            median_vsys = 0.0

        if component_data.ndim == 2 and component_data.shape[0] > 1:
            try:
                figure, plotted = plot_likelihood_space_triptych(
                    wavelength_A=component_wave,
                    phase=component_phase,
                    data=component_data,
                    model=median_model,
                    sigma=component_sigma,
                    title=title_base,
                )
                bundle.save_figure(
                    figure,
                    figure_id=f"likelihood_space_{component_id}",
                    tier="paper",
                    required=True,
                    plotted_data=plotted,
                    metadata={
                        "comparison_space": "processed likelihood space",
                        "posterior_summary": "median atmospheric state",
                    },
                )
            except Exception as exc:
                bundle.record_failure(
                    figure_id=f"likelihood_space_{component_id}",
                    tier="paper",
                    required=True,
                    error=exc,
                )

            diagnostic_bundle = {
                "epoch": epoch_label,
                "arm": component.name,
                "wavelength": component_wave,
                "data": component_data,
                "sigma": component_sigma,
                "phase": component_phase,
                "pre_sysrem_data": component.pre_sysrem_data,
                "pre_sysrem_sigma": component.pre_sysrem_sigma,
            }
            try:
                figure, _ = plot_residual_quality_summary(diagnostic_bundle)
                bundle.save_figure(
                    figure,
                    figure_id=f"residual_quality_{component_id}",
                    tier="qc",
                    required=True,
                    plotted_data={
                        "wavelength_A": component_wave,
                        "phase": component_phase,
                        "data": component_data,
                        "sigma": component_sigma,
                    },
                )
            except Exception as exc:
                bundle.record_failure(
                    figure_id=f"residual_quality_{component_id}",
                    tier="qc",
                    required=True,
                    error=exc,
                )
            if component.pre_sysrem_data is not None and component.pre_sysrem_sigma is not None:
                try:
                    result = plot_pre_post_sysrem_comparison(diagnostic_bundle)
                    if result is not None:
                        figure, _ = result
                        bundle.save_figure(
                            figure,
                            figure_id=f"pre_post_sysrem_{component_id}",
                            tier="qc",
                            required=True,
                            plotted_data={
                                "wavelength_A": component_wave,
                                "phase": component_phase,
                                "pre_sysrem_data": component.pre_sysrem_data,
                                "pre_sysrem_sigma": component.pre_sysrem_sigma,
                                "post_sysrem_data": component_data,
                                "post_sysrem_sigma": component_sigma,
                            },
                        )
                except Exception as exc:
                    bundle.record_failure(
                        figure_id=f"pre_post_sysrem_{component_id}",
                        tier="qc",
                        required=True,
                        error=exc,
                    )

        prepared_operator: dict[str, Any] | None = None
        operator_metadata: dict[str, Any] = {
            "frame": "native likelihood wavelength grid",
        }
        try:
            if component_data.ndim == 2 and component_data.shape[0] > 1:
                prepared_operator = prepare_planet_frame_operator(
                    wavelength_A=component_wave,
                    sigma=component_sigma,
                    phase=component_phase,
                    kp_kms=median_kp,
                    v_sys_kms=median_vsys,
                )
                planet_wave, planet_data, planet_error = apply_planet_frame_operator(
                    component_data,
                    prepared_operator,
                )
                operator = prepared_operator["operator"]
                operator_metadata = {
                    "frame": "planet rest frame",
                    "orbital_velocity_model": prepared_operator["orbital_velocity_model"],
                    "kp_kms": median_kp,
                    "v_sys_kms": median_vsys,
                    "n_source_wavelengths": int(np.asarray(operator["n_source_wavelengths"]).item()),
                    "n_covered_wavelengths": int(np.asarray(operator["n_covered_wavelengths"]).item()),
                    "n_dropped_out_of_bounds": int(np.asarray(operator["n_dropped_out_of_bounds"]).item()),
                    "n_dropped_gap_crossing": int(np.asarray(operator["n_dropped_gap_crossing"]).item()),
                }
            else:
                order = np.argsort(component_wave)
                planet_wave = component_wave[order]
                planet_data = component_data.reshape(-1, component_data.shape[-1])[0][order]
                planet_error = component_sigma.reshape(-1, component_sigma.shape[-1])[0][order]

            model_draws, model_indices, model_warnings = _component_publication_model_draws(
                posterior_samples=posterior_samples,
                model_params=model_params,
                region_config=region_config,
                component=component,
                region_sample_prefix=region_sample_prefix,
                prepared_operator=prepared_operator,
            )
            figure, plotted = plot_planet_frame_posterior_predictive(
                wavelength_A=planet_wave,
                observed=planet_data,
                observed_error=planet_error,
                model_draws=model_draws,
                title=title_base,
            )
            plotted["posterior_draw_indices"] = model_indices
            if prepared_operator is not None:
                plotted["planet_velocity_kms"] = prepared_operator["operator"]["planet_velocity_kms"]
                plotted["covered_source_indices"] = prepared_operator["operator"]["covered_source_indices"]
            bundle.save_figure(
                figure,
                figure_id=f"planet_frame_spectrum_{component_id}",
                tier="paper",
                required=True,
                plotted_data=plotted,
                metadata={
                    **operator_metadata,
                    "posterior_draw_count": int(model_draws.shape[0]),
                    "comparison_space": "processed likelihood space",
                },
                warnings=model_warnings,
            )
        except Exception as exc:
            bundle.record_failure(
                figure_id=f"planet_frame_spectrum_{component_id}",
                tier="paper",
                required=True,
                error=exc,
                metadata=operator_metadata,
            )

        if compute_contribution:
            atmo_state = median_atmo_state
            if atmo_state is None:
                bundle.record_failure(
                    figure_id=f"contribution_total_{component_id}",
                    tier="paper",
                    required=True,
                    error="Atmospheric state unavailable for contribution function.",
                )
            else:
                total_path = bundle.figure_path(f"contribution_total_{component_id}", "paper")
                try:
                    plot_contribution_function(
                        nu_grid=np.asarray(component.nu_grid),
                        dtau=np.asarray(atmo_state["dtau"]),
                        Tarr=np.asarray(atmo_state["Tarr"]),
                        pressure=np.asarray(atmo_state["pressure"]),
                        dParr=np.asarray(atmo_state["dParr"]),
                        mode=component.observation_config.mode,
                        save_path=str(total_path),
                        wavelength_unit="AA",
                        title=f"{config.PLANET} contribution function [{component.name}]",
                    )
                    bundle.register_existing(
                        figure_id=f"contribution_total_{component_id}",
                        tier="paper",
                        path=total_path,
                        required=True,
                        metadata={"atmospheric_state": "posterior median"},
                    )
                except Exception as exc:
                    bundle.record_failure(
                        figure_id=f"contribution_total_{component_id}",
                        tier="paper",
                        required=True,
                        error=exc,
                    )
                if atmo_state.get("dtau_per_species"):
                    species_path = bundle.figure_path(
                        f"contribution_per_species_{component_id}",
                        "supplement",
                    )
                    try:
                        plot_contribution_per_species(
                            nu_grid=np.asarray(component.nu_grid),
                            dtau_per_species={
                                key: np.asarray(value)
                                for key, value in atmo_state["dtau_per_species"].items()
                            },
                            Tarr=np.asarray(atmo_state["Tarr"]),
                            pressure=np.asarray(atmo_state["pressure"]),
                            dParr=np.asarray(atmo_state["dParr"]),
                            mode=component.observation_config.mode,
                            save_path=str(species_path),
                            wavelength_unit="AA",
                        )
                        bundle.register_existing(
                            figure_id=f"contribution_per_species_{component_id}",
                            tier="supplement",
                            path=species_path,
                            required=False,
                            metadata={"atmospheric_state": "posterior median"},
                        )
                    except Exception as exc:
                        bundle.record_failure(
                            figure_id=f"contribution_per_species_{component_id}",
                            tier="supplement",
                            required=False,
                            error=exc,
                        )

    for component in bandpass_components:
        component_id = "".join(
            character if character.isalnum() else "_" for character in component.name
        ).strip("_").lower()
        region_config = atmosphere_region_lookup[component.observation_config.region_name]
        try:
            model_draws, draw_indices, components, warnings = (
                _bandpass_publication_model_draws(
                    posterior_samples=posterior_samples,
                    model_params=model_params,
                    region_config=region_config,
                    component=component,
                    region_sample_prefix=region_config.sample_prefix,
                )
            )
            figure, plotted = plot_bandpass_posterior_predictive(
                observed=float(np.asarray(component.observation_inputs.value)),
                observed_error=float(np.asarray(component.observation_inputs.sigma)),
                model_draws=model_draws,
                component_name=component.name,
                observable=component.observation_config.observable,
            )
            plotted.update(components)
            plotted["posterior_draw_indices"] = draw_indices
            bundle.save_figure(
                figure,
                figure_id=f"bandpass_fit_{component_id}",
                tier="paper",
                required=True,
                plotted_data=plotted,
                metadata={
                    "atmosphere_region": component.observation_config.region_name,
                    "posterior_draw_count": int(model_draws.size),
                    "observable": component.observation_config.observable,
                    "includes_reflection": component.observation_config.include_reflection,
                },
                warnings=warnings,
            )
        except Exception as exc:
            bundle.record_failure(
                figure_id=f"bandpass_fit_{component_id}",
                tier="paper",
                required=True,
                error=exc,
                metadata={"atmosphere_region": component.observation_config.region_name},
            )

    return bundle.finalize(
        extra={
            "interpretation": {
                "paper_models": "HMC posterior predictive only",
                "svi": "optimization and initialization diagnostic only",
                "likelihood_triptychs": "processed data/model/residual in identical likelihood space",
            }
        }
    )


def _build_spectroscopic_observation_inputs(
    *,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries: FrozenTimeseriesInputs | None = None,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
) -> SpectroscopicObservationInputs:
    return SpectroscopicObservationInputs(
        data=jnp.asarray(data),
        sigma=jnp.asarray(sigma),
        phase=jnp.asarray(phase),
        U=(
            None
            if frozen_timeseries is not None
            or sysrem is None
            or sysrem.U is None
            else jnp.asarray(sysrem.U)
        ),
        V=(
            None
            if frozen_timeseries is not None
            or sysrem is None
            or sysrem.V is None
            else jnp.asarray(sysrem.V)
        ),
        chunked_sysrem=(
            None
            if frozen_timeseries is not None
            else _build_model_chunked_sysrem(sysrem)
        ),
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )


def _coerce_model_params(params: dict) -> dict[str, float | None]:
    def _configured_error(parameter_name: str, default: float) -> float:
        error_name = f"{parameter_name}_err"
        error = params.get(error_name)
        if error is None and hasattr(params[parameter_name], "std_dev"):
            error = params[parameter_name].std_dev
        if error is None:
            return float(default)

        error = float(error)
        if not np.isfinite(error) or error <= 0.0:
            return float(default)
        return error

    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Mp_upper_3sigma = params.get("M_p_upper_3sigma")
    if Kp_low is not None and Kp_low != Kp_low:
        Kp_low = None
    if Kp_high is not None and Kp_high != Kp_high:
        Kp_high = None
    if Mp_upper_3sigma is not None and Mp_upper_3sigma != Mp_upper_3sigma:
        Mp_upper_3sigma = None

    return {
        "Kp": params.get("Kp", config.DEFAULT_KP),
        "Kp_err": params.get("Kp_err", config.DEFAULT_KP_ERR),
        "Kp_low": Kp_low,
        "Kp_high": Kp_high,
        # Retained for provenance/reporting only; the stellar-rest model does
        # not add this absolute stellar velocity to the planet Doppler shift.
        "RV_abs": params.get("RV_abs", config.DEFAULT_RV_ABS),
        "RV_abs_err": params.get("RV_abs_err", config.DEFAULT_RV_ABS_ERR),
        "R_p": params["R_p"].nominal_value if hasattr(params["R_p"], "nominal_value") else params["R_p"],
        "R_p_err": _configured_error("R_p", config.DEFAULT_RP_ERR),
        "M_p": params["M_p"].nominal_value if hasattr(params["M_p"], "nominal_value") else params["M_p"],
        "M_p_err": _configured_error("M_p", config.DEFAULT_MP_ERR),
        "M_p_upper_3sigma": Mp_upper_3sigma,
        "M_star": (
            params["M_star"].nominal_value
            if ("M_star" in params and hasattr(params["M_star"], "nominal_value"))
            else params.get("M_star")
        ),
        "R_star": params["R_star"].nominal_value if hasattr(params["R_star"], "nominal_value") else params["R_star"],
        "R_star_err": _configured_error("R_star", config.DEFAULT_RSTAR_ERR),
        "T_star": params.get("T_star", config.DEFAULT_TSTAR),
        "logg_star": params.get("logg_star"),
        "Fe_H": params.get("Fe_H"),
        "v_sini_star": params.get("v_sini_star"),
        "gamma1": params.get("gamma1"),
        "gamma2": params.get("gamma2"),
        "T_eq": params.get("T_eq"),
        "Tirr_mean": params.get("Tirr_mean", params.get("T_eq")),
        "Tirr_std": params.get("Tirr_std"),
        "a": (
            params["a"].nominal_value
            if ("a" in params and hasattr(params["a"], "nominal_value"))
            else params.get("a")
        ),
        "period": params["period"].nominal_value if hasattr(params["period"], "nominal_value") else params["period"],
    }


def _normalize_retrieval_mode(mode: str) -> str:
    normalized = str(mode).lower().strip()
    if normalized not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    return normalized


def _default_region_name_for_mode(mode: str) -> str:
    if mode == "transmission":
        return "terminator"
    if mode == "emission":
        return "dayside"


def _build_art_for_mode(
    mode: str,
    *,
    reference_pressure_bar: float | None = None,
) -> object:
    mode = _normalize_retrieval_mode(mode)
    pressure_top, pressure_btm = config_utils.get_pressure_bounds_for_mode(mode)
    if mode == "transmission":
        if reference_pressure_bar is None:
            reference_pressure_bar = config_utils.get_transmission_reference_pressure_bar()
        if not np.isclose(
            float(reference_pressure_bar),
            pressure_btm,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "Transmission reference pressure must equal the RT lower "
                f"boundary: P_ref={float(reference_pressure_bar):g} bar, "
                f"P_btm={pressure_btm:g} bar"
            )
        art = ArtTransPure(
            pressure_top=pressure_top,
            pressure_btm=pressure_btm,
            nlayer=config.NLAYER,
        )
    else:
        art = ArtEmisPure(
            pressure_top=pressure_top,
            pressure_btm=pressure_btm,
            nlayer=config.NLAYER,
        )
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    return art


def _normalize_region_name(region_name: str | None, mode: str) -> str:
    if region_name is None:
        return _default_region_name_for_mode(mode)
    normalized = str(region_name).strip()
    return normalized


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


def _resolve_model_wavelength_range(
    wav_obs_before_thinning: np.ndarray,
) -> tuple[float, float, float, float]:
    """Resolve observed and padded model bounds from the complete data grid."""
    wavelengths = np.asarray(wav_obs_before_thinning, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("Model-grid wavelengths must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("Model-grid wavelengths must be finite and positive.")
    observed_min = float(np.min(wavelengths))
    observed_max = float(np.max(wavelengths))
    if observed_max <= observed_min:
        raise ValueError("Model-grid wavelengths must span a positive interval.")
    return (
        observed_min,
        observed_max,
        observed_min - float(config.WAV_MIN_OFFSET),
        observed_max + float(config.WAV_MAX_OFFSET),
    )


def _build_component_grid_and_ops(
    wav_obs: np.ndarray,
    instrument_resolution: float,
    *,
    wav_obs_before_thinning: np.ndarray | None = None,
    stellar_vsini: float | None = None,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    inst_nus = wav2nu(wav_obs, "AA")
    grid_source = (
        wav_obs
        if wav_obs_before_thinning is None
        else wav_obs_before_thinning
    )
    _, _, model_min, model_max = _resolve_model_wavelength_range(grid_source)
    nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
        model_min,
        model_max,
        config.N_SPECTRAL_POINTS,
        unit="AA",
    )
    _preflight_grid_checks(inst_nus, nu_grid)
    rotation_vsini_max = _rotation_operator_vsini_max(stellar_vsini)
    sop_rot, sop_inst, _ = setup_spectral_operators(
        nu_grid,
        instrument_resolution,
        vsini_max=rotation_vsini_max,
    )
    if rotation_vsini_max > DEFAULT_ROTATION_VSINI_MAX_KMS:
        print(
            "  Stellar rotation operator support: "
            f"vsini_max={rotation_vsini_max:g} km/s"
        )
    return inst_nus, nu_grid, sop_rot, sop_inst


def _build_in_memory_timeseries_component_spec(
    *,
    name: str,
    mode: str,
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries_operator: FrozenTimeseriesOperatorSpec | None,
    instrument_resolution: float,
    apply_sysrem: bool,
    radial_velocity_mode: str,
    data_format: str = "timeseries",
    subtract_weighted_global_mean: bool = False,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
    region_name: str | None = None,
    pre_sysrem_data: np.ndarray | None = None,
    pre_sysrem_sigma: np.ndarray | None = None,
    wav_obs_before_thinning: np.ndarray | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "data_format": data_format,
        "wav_obs": np.asarray(wav_obs),
        "wav_obs_before_thinning": (
            np.asarray(wav_obs)
            if wav_obs_before_thinning is None
            else np.asarray(wav_obs_before_thinning)
        ),
        "data": np.asarray(data),
        "sigma": np.asarray(sigma),
        "phase": np.asarray(phase),
        "sysrem": sysrem,
        "frozen_timeseries_operator": frozen_timeseries_operator,
        "pre_sysrem_data": None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        "pre_sysrem_sigma": None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        "instrument_resolution": float(instrument_resolution),
        "apply_sysrem": bool(apply_sysrem),
        "radial_velocity_mode": radial_velocity_mode,
        "subtract_weighted_global_mean": bool(subtract_weighted_global_mean),
        "collapsed_emission": collapsed_emission,
        "collapsed_transmission": collapsed_transmission,
    }
    if region_name is not None:
        spec["region_name"] = region_name
    return spec


def _load_opacity_bundle(
    nu_grid: np.ndarray,
    *,
    atomic_species: dict[str, dict] | None = None,
    molpath_hitemp: dict[str, str | Path] | None = None,
    molpath_exomol: dict[str, str | Path] | None = None,
) -> tuple[dict, dict, dict]:
    if atomic_species is None:
        atomic_species = config.ATOMIC_SPECIES
    if molpath_hitemp is None:
        molpath_hitemp = config.MOLPATH_HITEMP
    if molpath_exomol is None:
        molpath_exomol = config.MOLPATH_EXOMOL
    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    opa_mols, _ = load_molecular_opacities(
        molpath_hitemp,
        molpath_exomol,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    opa_atoms, _ = load_atomic_opacities(
        atomic_species,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    return opa_cias, opa_mols, opa_atoms


def _build_primary_spectroscopic_component(
    *,
    name: str,
    mode: str,
    wav_obs: np.ndarray,
    grid_source_wavelength_range: tuple[float, float],
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    frozen_timeseries: FrozenTimeseriesInputs | None,
    instrument_resolution: float,
    nu_grid: np.ndarray,
    inst_nus: np.ndarray,
    sop_rot: object,
    sop_inst: object,
    opa_cias: dict,
    opa_mols: dict,
    opa_atoms: dict,
    region_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    stellar_vsini: float | None,
    stellar_limb_darkening_u1: float | None,
    stellar_limb_darkening_u2: float | None,
    phoenix_cache_dir: str | Path | None,
    apply_sysrem: bool,
    radial_velocity_mode: str,
    subtract_weighted_global_mean: bool,
    collapsed_emission: CollapsedEmissionInputs | None = None,
    collapsed_transmission: CollapsedTransmissionInputs | None = None,
    sample_prefix: str | None = None,
    pre_sysrem_data: np.ndarray | None = None,
    pre_sysrem_sigma: np.ndarray | None = None,
) -> SpectroscopicComponentBundle:
    mode = _normalize_retrieval_mode(mode)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        stellar_vsini=stellar_vsini,
        stellar_limb_darkening_u1=stellar_limb_darkening_u1,
        stellar_limb_darkening_u2=stellar_limb_darkening_u2,
        radial_velocity_mode=radial_velocity_mode,
        subtract_weighted_global_mean=subtract_weighted_global_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=sample_prefix,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        grid_source_wavelength_range=grid_source_wavelength_range,
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        pre_sysrem_data=None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        pre_sysrem_sigma=None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _sanitize_name_for_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _infer_tbl_mode(metadata: dict[str, str]) -> str | None:
    spec_type = metadata.get("SPEC_TYPE", "").strip().lower()
    if "eclipse" in spec_type:
        return "emission"
    if "transit" in spec_type or "transmission" in spec_type:
        return "transmission"
    return None


def _convert_unit_to_micron(values: np.ndarray, unit: str) -> np.ndarray:
    unit_norm = unit.lower().strip()
    if "angstrom" in unit_norm or unit_norm == "aa":
        return values / 10000.0
    if unit_norm in {"nm", "nanometer", "nanometers"}:
        return values / 1000.0
    return values


def _combine_scalar_measurements(values: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
    if np.any(mask):
        weights = 1.0 / np.square(sigma[mask])
        value = float(np.sum(values[mask] * weights) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        return value, uncertainty

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite scalar measurements available to combine.")

    value = float(np.mean(finite_values))
    if finite_values.size == 1:
        return value, float("nan")
    return value, float(np.std(finite_values, ddof=1) / np.sqrt(finite_values.size))


def _make_tophat_bandpass(center_micron: float, bandwidth_micron: float, n_samples: int = 64) -> tuple[np.ndarray, np.ndarray]:
    width = float(bandwidth_micron)
    if not np.isfinite(width) or width <= 0:
        width = max(center_micron * 0.05, 0.02)

    half = width / 2.0
    start = max(center_micron - half, 1.0e-6)
    stop = center_micron + half
    wavelength_m = np.linspace(start, stop, int(n_samples), dtype=float) * 1.0e-6
    response = np.ones_like(wavelength_m)
    return wavelength_m, response


def _resolve_lowres_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "lrs" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def _resolve_bandpass_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "phot" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def make_joint_spectrum_component_from_tbl(tbl_path: str | Path) -> dict[str, Any]:
    tbl_path = _resolve_lowres_tbl_path(tbl_path)
    metadata, _columns, _data_by_col, _units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)
    if mode is None:
        raise ValueError(
            f"Could not infer low-res mode from SPEC_TYPE in {tbl_path}. "
            "Use a NASA .tbl with SPEC_TYPE matching eclipse/transmission."
        )

    return {
        "name": f"lrs_{_sanitize_name_for_id(tbl_path.stem)}",
        "mode": mode,
        "tbl_path": str(tbl_path),
        "data_format": "spectrum",
    }


def make_bandpass_constraints_from_tbl(tbl_path: str | Path) -> list[dict[str, Any]]:
    tbl_path = _resolve_bandpass_tbl_path(tbl_path)
    metadata, _columns, data_by_col, units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)
    if mode is None:
        raise ValueError(
            f"Could not infer low-res mode from SPEC_TYPE in {tbl_path}. "
            "Use a NASA .tbl with SPEC_TYPE matching eclipse/transmission."
        )

    wav_angstrom, spectrum, sigma, _meta = load_nasa_archive_spectrum(tbl_path, mode=mode)
    unique_wav = np.unique(np.round(wav_angstrom, 6))
    if unique_wav.size >= 5:
        raise ValueError(
            f"{tbl_path} has {unique_wav.size} unique wavelength bins. "
            "Pass it via --joint-spectrum-tbl instead of --bandpass-tbl."
        )

    bandwidth_values = np.full_like(wav_angstrom, np.nan, dtype=float)
    if "BANDWIDTH" in data_by_col:
        raw_bandwidth_values = []
        for value in data_by_col["BANDWIDTH"]:
            raw_bandwidth_values.append(np.nan if value is None else float(value))
        raw_bandwidth = np.asarray(
            raw_bandwidth_values,
            dtype=float,
        )
        if raw_bandwidth.size == wav_angstrom.size:
            bandwidth_values = _convert_unit_to_micron(
                raw_bandwidth,
                units_by_col.get("BANDWIDTH", ""),
            )

    grouped_indices: dict[float, list[int]] = {}
    for idx, wav in enumerate(np.round(wav_angstrom, 6)):
        grouped_indices.setdefault(float(wav), []).append(idx)

    instrument = (metadata.get("INSTRUMENT") or metadata.get("INSTRUMENT_NAME") or "").strip()
    facility = (metadata.get("FACILITY") or metadata.get("FACILITY_NAME") or "").strip()
    instrument_text = f"{instrument} {facility}".lower()
    base_name = f"lrs_{_sanitize_name_for_id(tbl_path.stem)}"

    constraints: list[dict[str, Any]] = []
    for group_idx, indices in enumerate(grouped_indices.values(), start=1):
        value, value_sigma = _combine_scalar_measurements(
            spectrum[indices],
            sigma[indices],
        )
        if not np.isfinite(value_sigma) or value_sigma <= 0:
            continue

        name = base_name if len(grouped_indices) == 1 else f"{base_name}_{group_idx}"
        observable = "eclipse_depth" if mode == "emission" else "transit_depth"
        constraint: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "observable": observable,
            "value": value,
            "sigma": value_sigma,
        }

        if "tess" not in instrument_text:
            center_micron = float(np.mean(wav_angstrom[indices]) / 10000.0)
            bandwidth = bandwidth_values[indices]
            finite_bandwidth = bandwidth[np.isfinite(bandwidth) & (bandwidth > 0)]
            bandwidth_micron = (
                float(np.mean(finite_bandwidth))
                if finite_bandwidth.size > 0
                else float("nan")
            )
            wavelength_m, response = _make_tophat_bandpass(center_micron, bandwidth_micron)
            constraint["wavelength_m"] = wavelength_m
            constraint["response"] = response

        constraints.append(constraint)

    return constraints


def _build_atmosphere_regions(
    *,
    model_params: dict[str, float | None],
    primary_mode: str,
    primary_region_name: str,
    primary_art: object,
    observation_configs: list[object],
    default_pt_profile: str,
    default_chemistry_model: str,
    default_fastchem_parameter_file: str | None,
    atmosphere_regions: list[dict[str, Any]] | None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    explicit_specs: dict[str, dict[str, Any]] = {}
    if atmosphere_regions:
        for raw_spec in atmosphere_regions:
            spec = dict(raw_spec)
            name = str(spec.get("name", "")).strip()
            if "mode" in spec and spec["mode"] is not None:
                spec["mode"] = _normalize_retrieval_mode(spec["mode"])
            explicit_specs[name] = spec

    component_modes: dict[str, str] = {}
    region_mol_names: dict[str, set[str]] = {}
    region_atom_names: dict[str, set[str]] = {}
    for observation_config in observation_configs:
        region_name = str(observation_config.region_name)
        region_mode = _normalize_retrieval_mode(observation_config.mode)
        if region_name in component_modes and component_modes[region_name] != region_mode:
            raise ValueError(
                f"Atmosphere region '{region_name}' is referenced by mixed modes "
                f"({component_modes[region_name]} and {region_mode}). Split them into "
                "separate region_name values."
            )
        component_modes[region_name] = region_mode
        region_mol_names.setdefault(region_name, set()).update(observation_config.opa_mols.keys())
        region_atom_names.setdefault(region_name, set()).update(observation_config.opa_atoms.keys())

    region_configs = []
    region_lookup: dict[str, object] = {}
    for region_name in component_modes:
        region_mode = component_modes[region_name]
        spec = explicit_specs.get(region_name, {})
        if spec.get("mode") is not None and spec["mode"] != region_mode:
            raise ValueError(
                f"Atmosphere region '{region_name}' is configured as mode='{spec['mode']}' "
                f"but observation components require mode='{region_mode}'."
            )

        chemistry_name = str(spec.get("chemistry_model", default_chemistry_model))
        chemistry_param_file = spec.get(
            "fastchem_parameter_file",
            default_fastchem_parameter_file,
        )
        composition_solver = _build_composition_solver(
            chemistry_model=chemistry_name,
            fastchem_parameter_file=chemistry_param_file,
        )
        kappa_bounds = spec.get("kappa_ir_cgs_bounds")
        gamma_bounds = spec.get("gamma_bounds")

        art = (
            primary_art
            if region_name == primary_region_name and region_mode == primary_mode
            else _build_art_for_mode(region_mode)
        )
        region_pt_profile = config_utils.resolve_pt_profile_for_region(
            region_mode,
            primary_pt_profile=default_pt_profile,
            is_primary=(
                region_name == primary_region_name and region_mode == primary_mode
            ),
            pt_profile=spec.get("pt_profile"),
        )
        region_config = build_atmosphere_region_config(
            mode=region_mode,
            art=art,
            mol_names=tuple(sorted(region_mol_names[region_name])),
            atom_names=tuple(sorted(region_atom_names[region_name])),
            pt_profile=region_pt_profile,
            T_low=spec.get("T_low"),
            T_high=spec.get("T_high"),
            Tirr_mean=spec.get("Tirr_mean", model_params.get("Tirr_mean")),
            Tirr_std=spec.get("Tirr_std", model_params.get("Tirr_std")),
            Tint_fixed=spec.get("Tint_fixed"),
            kappa_ir_cgs_bounds=None if kappa_bounds is None else tuple(kappa_bounds),
            gamma_bounds=None if gamma_bounds is None else tuple(gamma_bounds),
            composition_solver=composition_solver,
            name=region_name,
            sample_prefix=spec.get("sample_prefix"),
            velocity_offset_mode=str(
                spec.get("velocity_offset_mode", "shared")
            ),
            velocity_offset_species=tuple(
                spec.get("velocity_offset_species", ())
            ),
            velocity_offset_bounds_kms=tuple(
                spec.get("velocity_offset_bounds_kms", (-20.0, 20.0))
            ),
        )
        region_configs.append(region_config)
        region_lookup[region_name] = region_config

    return tuple(region_configs), region_lookup


def _load_joint_spectroscopic_component(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_stellar_vsini: float | None,
    default_stellar_limb_darkening_u1: float | None,
    default_stellar_limb_darkening_u2: float | None,
    default_phoenix_cache_dir: str | Path | None,
    default_sigma_scale: float = 1.0,
    default_spectral_stride: int = 1,
    default_spectral_offset: int = 0,
) -> SpectroscopicComponentBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_component"))
    data_format = str(spec.get("data_format", "spectrum")).lower().strip()
    instrument_resolution = float(spec.get("instrument_resolution", config_utils.get_resolution()))
    radial_velocity_mode = str(spec.get("radial_velocity_mode", "orbital" if data_format == "timeseries" else "none"))
    apply_sysrem_explicit = "apply_sysrem" in spec
    apply_sysrem = bool(spec.get("apply_sysrem", data_format == "timeseries" and config.APPLY_SYSREM_DEFAULT))
    subtract_weighted_global_mean = bool(
        spec.get(
            "subtract_weighted_global_mean",
            component_mode == "transmission"
            and data_format == "spectrum"
            and "tbl_path" not in spec,
        )
    )
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    stellar_vsini = spec.get("v_sini_star", default_stellar_vsini)
    stellar_limb_darkening_u1 = spec.get(
        "gamma1",
        spec.get(
            "stellar_limb_darkening_u1",
            default_stellar_limb_darkening_u1,
        ),
    )
    stellar_limb_darkening_u2 = spec.get(
        "gamma2",
        spec.get(
            "stellar_limb_darkening_u2",
            default_stellar_limb_darkening_u2,
        ),
    )
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)
    sigma_scale = _validate_sigma_scale(spec.get("sigma_scale", default_sigma_scale))
    spectral_stride, spectral_offset = _validate_spectral_subset(
        spec.get("spectral_stride", default_spectral_stride),
        spec.get("spectral_offset", default_spectral_offset),
    )
    pre_sysrem_data = None
    pre_sysrem_sigma = None
    collapsed_emission = spec.get("collapsed_emission")
    collapsed_transmission = spec.get("collapsed_transmission")
    frozen_timeseries_operator = spec.get("frozen_timeseries_operator")
    if frozen_timeseries_operator is not None and not isinstance(
        frozen_timeseries_operator,
        FrozenTimeseriesOperatorSpec,
    ):
        raise TypeError(
            f"Unsupported frozen time-series operator for component '{name}': "
            f"{type(frozen_timeseries_operator)!r}"
        )
    component_data_dir: Path | None = None

    if "tbl_path" in spec:
        wav_obs, spectrum, uncertainty, _meta = load_nasa_archive_spectrum(
            spec["tbl_path"],
            mode=component_mode,
        )
        # The spectroscopy forward model represents transmission as the
        # negative perturbation of normalized stellar flux, whereas archive
        # tables conventionally store a positive absolute transit depth.
        # Convert only at this internal spectroscopic boundary; scalar
        # bandpass constraints continue to use positive transit depths.
        if component_mode == "transmission":
            spectrum = -np.asarray(spectrum)
        data = spectrum[np.newaxis, :]
        sigma = uncertainty[np.newaxis, :]
        phase = np.zeros((1,), dtype=float)
        sysrem = None
    elif all(key in spec for key in ("wav_obs", "data", "sigma")):
        wav_obs = np.asarray(spec["wav_obs"])
        data = np.asarray(spec["data"])
        sigma = np.asarray(spec["sigma"])
        phase = np.asarray(spec.get("phase", np.zeros((1 if data.ndim == 1 else data.shape[0],), dtype=float)))
        if spec.get("pre_sysrem_data") is not None and spec.get("pre_sysrem_sigma") is not None:
            pre_sysrem_data = np.asarray(spec["pre_sysrem_data"])
            pre_sysrem_sigma = np.asarray(spec["pre_sysrem_sigma"])
        if spec.get("sysrem") is not None:
            sysrem_spec = spec["sysrem"]
            if isinstance(sysrem_spec, SysremInputBundle):
                sysrem = sysrem_spec
            elif isinstance(sysrem_spec, dict):
                n_sysrem_exp = (
                    frozen_timeseries_operator.source_phase.size
                    if frozen_timeseries_operator is not None
                    else (1 if data.ndim == 1 else data.shape[0])
                )
                sysrem = _validate_sysrem_inputs(
                    sysrem_spec,
                    n_exp=n_sysrem_exp,
                )
            else:
                raise TypeError(f"Unsupported sysrem spec type for component '{name}': {type(sysrem_spec)!r}")
        elif spec.get("U") is not None or spec.get("V") is not None:
            if spec.get("U") is None:
                raise ValueError(
                    f"Joint spectroscopic component '{name}' cannot provide V "
                    "without U. Per-pixel projection weights come from sigma."
                )
            raw_sysrem = {"U": np.asarray(spec["U"])}
            if spec.get("V") is not None:
                raw_sysrem["V"] = np.asarray(spec["V"])
            sysrem = _validate_sysrem_inputs(
                raw_sysrem,
                n_exp=(1 if data.ndim == 1 else data.shape[0]),
            )
        else:
            sysrem = None
    elif "data_dir" in spec:
        data_dir = Path(spec["data_dir"])
        component_data_dir = data_dir
        if data_format == "timeseries":
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            frozen_timeseries_operator = (
                _load_frozen_timeseries_operator_spec(
                    data_dir,
                    wav_obs,
                    phase,
                    require_lsd_shadow=component_mode == "transmission",
                )
            )
            if (
                apply_sysrem_explicit
                and apply_sysrem != frozen_timeseries_operator.has_sysrem
            ):
                raise ValueError(
                    f"Joint spectroscopic component '{name}' requests "
                    f"apply_sysrem={apply_sysrem}, but its frozen operator "
                    f"records has_sysrem={frozen_timeseries_operator.has_sysrem}. "
                    "Use a bundle prepared with the requested preprocessing."
                )
            apply_sysrem = frozen_timeseries_operator.has_sysrem
            pre_sysrem_bundle = load_pre_sysrem_timeseries_data(data_dir)
            if pre_sysrem_bundle is not None:
                pre_sysrem_data, pre_sysrem_sigma = pre_sysrem_bundle
            phase = _normalize_phase(phase)
            if apply_sysrem:
                sysrem = _validate_sysrem_inputs(
                    _load_sysrem_inputs(data_dir),
                    n_exp=frozen_timeseries_operator.source_phase.size,
                )
            else:
                sysrem = None
        elif data_format == "spectrum":
            suffix = "transmission" if component_mode == "transmission" else "emission"
            wav_obs, spectrum, uncertainty, _ = load_observed_spectrum(
                str(data_dir / f"wavelength_{suffix}.npy"),
                str(data_dir / f"spectrum_{suffix}.npy"),
                str(data_dir / f"uncertainty_{suffix}.npy"),
            )
            data = spectrum[np.newaxis, :]
            sigma = uncertainty[np.newaxis, :]
            phase = np.zeros((1,), dtype=float)
            sysrem = None
        else:
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            sysrem = None

    if apply_sysrem and sysrem is None:
        raise ValueError(
            f"Joint spectroscopic component '{name}' requested SYSREM but no valid "
            "SYSREM inputs were provided."
        )

    wav_obs_before_thinning = np.asarray(
        spec.get("wav_obs_before_thinning", wav_obs),
        dtype=float,
    ).copy()
    observed_min, observed_max, _, _ = _resolve_model_wavelength_range(
        wav_obs_before_thinning
    )

    frozen_subset_indices = (
        None
        if frozen_timeseries_operator is None
        or (spectral_stride == 1 and spectral_offset == 0)
        else _spectral_subset_indices(
            np.asarray(wav_obs).size,
            spectral_stride,
            spectral_offset,
        )
    )
    wav_obs, data, sigma, sysrem, pre_sysrem_data, pre_sysrem_sigma = _apply_spectral_thinning(
        wav_obs=wav_obs,
        data=data,
        sigma=sigma,
        sysrem=sysrem,
        pre_sysrem_data=pre_sysrem_data,
        pre_sysrem_sigma=pre_sysrem_sigma,
        spectral_stride=spectral_stride,
        spectral_offset=spectral_offset,
        component_name=name,
    )
    frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
        frozen_timeseries_operator,
        frozen_subset_indices,
    )

    inst_nus_before_prepare = wav2nu(np.asarray(wav_obs), "AA")
    sort_idx_for_prepare = None
    if inst_nus_before_prepare.size > 1 and np.any(np.diff(inst_nus_before_prepare) <= 0):
        sort_idx_for_prepare = np.argsort(inst_nus_before_prepare)

    wav_obs, data, sigma, inst_nus = _prepare_observed_spectrum_arrays(wav_obs, data, sigma)
    sysrem = _remap_sysrem_wavelength_sort(sysrem, sort_idx_for_prepare)
    frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
        frozen_timeseries_operator,
        sort_idx_for_prepare,
    )
    frozen_timeseries = _build_model_frozen_timeseries(
        frozen_timeseries_operator,
        sysrem,
    )
    if (
        collapsed_emission is None
        and component_data_dir is not None
        and component_mode == "emission"
        and data_format == "spectrum"
        and (component_data_dir / "collapse_metadata.json").exists()
    ):
        collapsed_emission = _load_collapsed_emission_operator(
            component_data_dir,
            wav_obs,
        )
        if "subtract_weighted_global_mean" not in spec:
            subtract_weighted_global_mean = True
    if (
        collapsed_transmission is None
        and component_data_dir is not None
        and component_mode == "transmission"
        and data_format == "spectrum"
        and (component_data_dir / "collapse_metadata.json").exists()
    ):
        collapsed_transmission = _load_collapsed_transmission_operator(
            component_data_dir,
            wav_obs,
        )
    sigma = _scale_spectroscopic_sigma(sigma, sigma_scale)
    if subtract_weighted_global_mean:
        data = _subtract_inverse_variance_weighted_constant(data, sigma)
    if pre_sysrem_data is not None and pre_sysrem_sigma is not None:
        pre_sysrem_data = np.asarray(pre_sysrem_data)
        pre_sysrem_sigma = np.asarray(pre_sysrem_sigma)
        if pre_sysrem_data.shape != data.shape or pre_sysrem_sigma.shape != sigma.shape:
            raise ValueError(
                f"Pre-SYSREM diagnostic arrays for component '{name}' must match "
                f"data/sigma shape {data.shape}; got {pre_sysrem_data.shape} and {pre_sysrem_sigma.shape}."
            )
        if sort_idx_for_prepare is not None:
            if pre_sysrem_data.ndim == 2:
                pre_sysrem_data = pre_sysrem_data[:, sort_idx_for_prepare]
                pre_sysrem_sigma = pre_sysrem_sigma[:, sort_idx_for_prepare]
            else:
                pre_sysrem_data = pre_sysrem_data[sort_idx_for_prepare]
                pre_sysrem_sigma = pre_sysrem_sigma[sort_idx_for_prepare]
        pre_sysrem_sigma = _scale_spectroscopic_sigma(pre_sysrem_sigma, sigma_scale)
    if phase.ndim == 0:
        phase = np.asarray([float(phase)])
    if data_format == "timeseries":
        phase = _normalize_phase(phase)
    elif phase.size == 0:
        phase = np.zeros((1,), dtype=float)
    _preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

    inst_nus_component, nu_grid, sop_rot, sop_inst = _build_component_grid_and_ops(
        wav_obs,
        instrument_resolution,
        wav_obs_before_thinning=wav_obs_before_thinning,
        stellar_vsini=(stellar_vsini if component_mode == "emission" else None),
    )
    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(
        nu_grid,
        atomic_species=spec.get("atomic_species"),
        molpath_hitemp=spec.get("molpath_hitemp"),
        molpath_exomol=spec.get("molpath_exomol"),
    )
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )

    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus_component,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        stellar_vsini=stellar_vsini,
        stellar_limb_darkening_u1=stellar_limb_darkening_u1,
        stellar_limb_darkening_u2=stellar_limb_darkening_u2,
        radial_velocity_mode=radial_velocity_mode,
        subtract_weighted_global_mean=subtract_weighted_global_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=name,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
        frozen_timeseries=frozen_timeseries,
        collapsed_emission=collapsed_emission,
        collapsed_transmission=collapsed_transmission,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        grid_source_wavelength_range=(observed_min, observed_max),
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        pre_sysrem_data=None if pre_sysrem_data is None else np.asarray(pre_sysrem_data),
        pre_sysrem_sigma=None if pre_sysrem_sigma is None else np.asarray(pre_sysrem_sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus_component),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _load_bandpass_constraint(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_semi_major_axis_au: float | None,
    default_phoenix_cache_dir: str | Path | None,
) -> BandpassConstraintBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_bandpass"))
    observable = str(spec["observable"])
    value = float(spec["value"])
    sigma = float(spec["sigma"])
    photon_weighted = bool(spec.get("photon_weighted", False))
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)
    include_reflection = bool(spec.get("include_reflection", False))
    semi_major_axis_au = spec.get("semi_major_axis_au", default_semi_major_axis_au)
    if semi_major_axis_au is not None:
        semi_major_axis_au = float(semi_major_axis_au)
    geometric_albedo_bounds_raw = spec.get("geometric_albedo_bounds")
    geometric_albedo_bounds = (
        None
        if geometric_albedo_bounds_raw is None
        else tuple(float(x) for x in geometric_albedo_bounds_raw)
    )
    model_sigma_raw = spec.get("model_sigma")
    model_sigma = None if model_sigma_raw is None else float(model_sigma_raw)
    model_sigma_bounds_raw = spec.get("model_sigma_bounds")
    model_sigma_bounds = (
        None
        if model_sigma_bounds_raw is None
        else tuple(float(x) for x in model_sigma_bounds_raw)
    )

    if "wavelength_m" in spec and "response" in spec:
        wavelength_m = np.asarray(spec["wavelength_m"], dtype=float)
        response = np.asarray(spec["response"], dtype=float)
    else:
        bandpass_path = spec.get("bandpass_path")
        wavelength_m, response, _used_path = load_tess_bandpass(
            bandpass_path,
            download_if_missing=bool(spec.get("download_bandpass", True)),
        )

    wavelength_angstrom = np.asarray(wavelength_m, dtype=float) * 1.0e10
    if "nu_grid" in spec:
        nu_grid = np.asarray(spec["nu_grid"], dtype=float)
    else:
        nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
            float(np.min(wavelength_angstrom)) - config.WAV_MIN_OFFSET,
            float(np.max(wavelength_angstrom)) + config.WAV_MAX_OFFSET,
            config.N_SPECTRAL_POINTS,
            unit="AA",
        )

    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(nu_grid)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_bandpass_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        wavelength_m=wavelength_m,
        response=response,
        observable=observable,
        photon_weighted=photon_weighted,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        include_reflection=include_reflection,
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo_bounds=geometric_albedo_bounds,
        model_sigma=model_sigma,
        model_sigma_bounds=model_sigma_bounds,
        sample_prefix=name,
    )
    observation_inputs = BandpassObservationInputs(
        value=jnp.asarray(value),
        sigma=jnp.asarray(sigma),
    )
    return BandpassConstraintBundle(
        name=name,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _normalize_epoch_values(epoch: str | Sequence[str] | None) -> tuple[str, ...]:
    if epoch is None:
        return ()
    if isinstance(epoch, str):
        values = [epoch]
    else:
        values = []
        for item in epoch:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
    return tuple(values)


class _StepTimer:
    def __init__(self, label: str):
        self.label = label
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.start
        status = "failed after" if exc_type is not None else "completed in"
        print(f"  {self.label} {status} {elapsed:.2f}s")
        return False


def _validate_sigma_scale(sigma_scale: float) -> float:
    try:
        scale = float(sigma_scale)
    except Exception as exc:
        raise ValueError("sigma_scale must be a finite positive number.") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("sigma_scale must be a finite positive number.")
    return scale


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
    if values.shape != sigma.shape:
        raise ValueError(
            f"Cannot subtract weighted constant: values shape {values.shape} "
            f"does not match sigma shape {sigma.shape}."
        )
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


def run_retrieval(
    mode: str = "transmission",
    epoch: str | Sequence[str] | None = None,
    data_format: str = config.DEFAULT_DATA_FORMAT,
    skip_svi: bool = False,
    svi_only: bool = False,
    pt_profile: str | None = None,
    chemistry_model: str | None = None,
    fastchem_parameter_file: str | None = None,
    compute_contribution: bool = True,
    seed: int = 42,
    wav_obs: np.ndarray | None = None,
    data: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    U: np.ndarray | None = None,
    V: np.ndarray | None = None,
    sysrem_inputs: SysremInputBundle | None = None,
    joint_spectra: list[dict[str, Any]] | None = None,
    bandpass_constraints: list[dict[str, Any]] | None = None,
    atmosphere_regions: list[dict[str, Any]] | None = None,
    phoenix_cache_dir: str | Path | None = None,
    save_mcmc_diagnostics: bool = True,
    sigma_scale: float = 1.0,
    spectral_stride: int = 1,
    spectral_offset: int = 0,
    diagnostic_label: str | None = None,
    apply_sysrem_override: bool | None = None,
    emission_selection: str | None = None,
    shared_prior_modes: dict[str, str] | None = None,
    primary_atomic_species: dict[str, dict] | None = None,
    primary_molpath_hitemp: dict[str, str | Path] | None = None,
    primary_molpath_exomol: dict[str, str | Path] | None = None,
    retrieval_intent: dict[str, Any] | None = None,
    model_param_overrides: dict[str, float] | None = None,
) -> None:
    retrieval_start = perf_counter()
    mode = _normalize_retrieval_mode(mode)
    reference_pressure_bar = (
        config_utils.get_transmission_reference_pressure_bar()
        if mode == "transmission"
        else None
    )
    if emission_selection is not None:
        emission_selection = str(emission_selection).strip().lower().replace("-", "_")
        if emission_selection in {"full", "full_transit"}:
            emission_selection = "full_emission"
        if emission_selection not in {
            "full_emission",
            "pre_eclipse",
            "post_eclipse",
        }:
            raise ValueError(
                f"Unsupported collapsed emission selection: {emission_selection!r}."
            )
        if mode != "emission" or data_format != "spectrum":
            raise ValueError(
                "emission_selection requires mode='emission' and data_format='spectrum'."
            )
    sigma_scale = _validate_sigma_scale(sigma_scale)
    spectral_stride, spectral_offset = _validate_spectral_subset(spectral_stride, spectral_offset)
    epochs = _normalize_epoch_values(epoch)
    primary_epoch = epochs[0] if epochs else None
    if pt_profile is None:
        raise ValueError("pt_profile must be passed explicitly.")
    if chemistry_model is None:
        raise ValueError("chemistry_model must be passed explicitly.")

    # Create timestamped output directory
    base_dir = config.DIR_SAVE or config_utils.get_output_dir()
    sanitized_label = sanitize_diagnostic_label(diagnostic_label)
    if sanitized_label is not None:
        base_dir = Path(base_dir) / sanitized_label
    output_dir = config_utils.create_timestamped_dir(base_dir)
    print(f"\nOutput directory: {output_dir}")

    # Save run configuration
    config_utils.save_run_config(
        output_dir=output_dir,
        mode=mode,
        pt_profile=pt_profile,
        skip_svi=skip_svi,
        svi_only=svi_only,
        seed=seed,
        chemistry_model=chemistry_model,
        epoch=epochs or None,
        phoenix_cache_dir=None if phoenix_cache_dir is None else str(phoenix_cache_dir),
        save_mcmc_diagnostics=save_mcmc_diagnostics,
        sigma_scale=sigma_scale,
        spectral_stride=spectral_stride,
        spectral_offset=spectral_offset,
        diagnostic_label=sanitized_label,
        apply_sysrem_override=apply_sysrem_override,
        emission_selection=emission_selection,
        reference_pressure_bar=reference_pressure_bar,
        retrieval_intent=retrieval_intent,
        model_param_overrides=model_param_overrides,
    )

    # Get planet parameters
    params = dict(config_utils.get_params())
    if model_param_overrides:
        params.update(model_param_overrides)
    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")

    apply_sysrem_default = config.APPLY_SYSREM_DEFAULT if apply_sysrem_override is None else apply_sysrem_override
    apply_sysrem = bool(apply_sysrem_default and data_format == "timeseries")
    primary_subtract_weighted_global_mean = bool(
        data_format == "spectrum"
        and (
            mode == "transmission"
            or (mode == "emission" and emission_selection is not None)
        )
    )

    primary_sysrem: SysremInputBundle | None = sysrem_inputs
    primary_frozen_timeseries_operator: FrozenTimeseriesOperatorSpec | None = None
    primary_pre_sysrem_data: np.ndarray | None = None
    primary_pre_sysrem_sigma: np.ndarray | None = None
    primary_collapsed_emission: CollapsedEmissionInputs | None = None
    primary_collapsed_transmission: CollapsedTransmissionInputs | None = None
    primary_wav_obs_before_thinning: np.ndarray | None = None
    primary_grid_source_wavelength_range: tuple[float, float] | None = None

    full_arm_mode = (
        config.OBSERVING_MODE == "full"
        and all(val is None for val in (wav_obs, data, sigma, phase))
    )
    if config.OBSERVING_MODE == "full" and not full_arm_mode:
        raise ValueError(
            "--wavelength-range full requires loading red and blue from their "
            "on-disk component directories; direct in-memory primary arrays are "
            "not supported in full-arm mode."
        )

    primary_component_name = "spectroscopy_red" if full_arm_mode else "spectroscopy"

    if full_arm_mode:
        arm_dirs = (
            config_utils.get_full_arm_timeseries_dirs(
                epoch=primary_epoch,
                mode=mode,
            )
            if data_format == "timeseries"
            else config_utils.get_full_arm_data_dirs(
                epoch=primary_epoch,
                mode=mode,
            )
        )
        if mode == "emission" and emission_selection is not None:
            arm_dirs = {
                arm: config_utils.get_collapsed_emission_dir(
                    epoch=primary_epoch,
                    arm=arm,
                    selection=emission_selection,
                )
                for arm in config.FULL_ARM_MEMBERS
            }
        elif mode == "transmission" and data_format == "spectrum":
            arm_dirs = {
                arm: config_utils.get_collapsed_transmission_dir(
                    epoch=primary_epoch,
                    arm=arm,
                )
                for arm in config.FULL_ARM_MEMBERS
            }
        blue_component_spec = {
            "name": "spectroscopy_blue",
            "mode": mode,
            "data_format": data_format,
            "data_dir": str(arm_dirs["blue"]),
            "radial_velocity_mode": "orbital" if data_format == "timeseries" else "none",
            "subtract_weighted_global_mean": bool(
                data_format == "spectrum"
                and (
                    mode == "transmission"
                    or (
                        mode == "emission"
                        and emission_selection is not None
                    )
                )
            ),
            "instrument_resolution": config_utils.get_resolution(),
            "atomic_species": primary_atomic_species,
            "molpath_hitemp": primary_molpath_hitemp,
            "molpath_exomol": primary_molpath_exomol,
        }
        if apply_sysrem_override is not None and data_format == "timeseries":
            blue_component_spec["apply_sysrem"] = bool(apply_sysrem_override)
        joint_spectra = list(joint_spectra or []) + [blue_component_spec]

    print("\n[1/7] Loading time-series data...")
    with _StepTimer("Step 1/7"):
        if epochs:
            if len(epochs) == 1:
                print(f"  Using epoch: {primary_epoch}")
            else:
                print(f"  Using epochs: {', '.join(epochs)}")
        if any(val is not None for val in (wav_obs, data, sigma, phase)):
            if mode == "emission" and emission_selection is not None:
                raise ValueError(
                    "Collapsed 1D emission retrievals must be loaded from their "
                    "product directory so emission_collapse_operator.npz is "
                    "available; direct wav_obs/data/sigma arrays are insufficient."
                )
            phase = _normalize_phase(phase)
            print(f"  Using provided data: {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
            if apply_sysrem:
                if primary_sysrem is None:
                    if U is None:
                        raise ValueError(
                            "apply_sysrem=True requires either sysrem_inputs or U "
                            "when providing wav_obs/data/sigma/phase directly; "
                            "per-pixel weights come from sigma."
                        )
                    raw_sysrem = {"U": U}
                    if V is not None:
                        raw_sysrem["V"] = V
                    primary_sysrem = _validate_sysrem_inputs(
                        raw_sysrem,
                        n_exp=data.shape[0],
                    )
                print(f"  Using provided SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
        else:
            if full_arm_mode:
                resolved_data_dir = arm_dirs["red"]
                print(
                    "  OBSERVING_MODE='full': loading red as primary, blue as "
                    "a second spectroscopic component."
                )
                data_paths = (
                    config_utils.get_transmission_paths(
                        epoch=primary_epoch,
                        arm="red",
                        collapsed=data_format == "spectrum",
                    )
                    if mode == "transmission"
                    else config_utils.get_emission_paths(
                        epoch=primary_epoch,
                        arm="red",
                        selection=emission_selection,
                    )
                )
            else:
                if mode == "emission" and emission_selection is not None:
                    resolved_data_dir = config_utils.get_collapsed_emission_dir(
                        epoch=primary_epoch,
                        arm=config.OBSERVING_MODE,
                        selection=emission_selection,
                    )
                else:
                    if mode == "transmission" and data_format == "spectrum":
                        resolved_data_dir = (
                            config_utils.get_collapsed_transmission_dir(
                                epoch=primary_epoch,
                                arm=config.OBSERVING_MODE,
                            )
                        )
                    else:
                        resolved_data_dir = (
                            config_utils.get_timeseries_data_dir(
                                epoch=primary_epoch,
                                mode=mode,
                            )
                            if data_format == "timeseries"
                            else config_utils.get_data_dir(
                                epoch=primary_epoch,
                                mode=mode,
                            )
                        )
                data_paths = (
                    config_utils.get_transmission_paths(
                        epoch=primary_epoch,
                        collapsed=data_format == "spectrum",
                    ) if mode == "transmission"
                    else config_utils.get_emission_paths(
                        epoch=primary_epoch,
                        selection=emission_selection,
                    )
                )

            if data_format == "timeseries":
                wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
                primary_frozen_timeseries_operator = (
                    _load_frozen_timeseries_operator_spec(
                        resolved_data_dir,
                        wav_obs,
                        phase,
                        require_lsd_shadow=mode == "transmission",
                    )
                )
                if (
                    apply_sysrem_override is not None
                    and bool(apply_sysrem_override)
                    != primary_frozen_timeseries_operator.has_sysrem
                ):
                    raise ValueError(
                        f"The prepared bundle records has_sysrem="
                        f"{primary_frozen_timeseries_operator.has_sysrem}, but "
                        f"the run requests apply_sysrem={bool(apply_sysrem_override)}. "
                        "Use a bundle prepared with the requested preprocessing."
                    )
                apply_sysrem = primary_frozen_timeseries_operator.has_sysrem
                pre_sysrem_bundle = load_pre_sysrem_timeseries_data(resolved_data_dir)
                if pre_sysrem_bundle is not None:
                    primary_pre_sysrem_data, primary_pre_sysrem_sigma = pre_sysrem_bundle
                phase = _normalize_phase(phase)
                print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
                print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
                if apply_sysrem:
                    primary_sysrem = _validate_sysrem_inputs(
                        _load_sysrem_inputs(resolved_data_dir),
                        n_exp=(
                            primary_frozen_timeseries_operator.source_phase.size
                        ),
                    )
                    print(f"  Loaded SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
            elif data_format == "spectrum":
                wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
                    str(data_paths["wavelength"]),
                    str(data_paths["spectrum"]),
                    str(data_paths["uncertainty"]),
                )
                data = spectrum[np.newaxis, :]
                sigma = uncertainty[np.newaxis, :]
                phase = np.array([0.0])
                print(f"  Loaded single spectrum with {len(wav_obs)} points")
                if apply_sysrem:
                    raise ValueError(
                        "apply_sysrem=True with data_format='spectrum' requires SYSREM "
                        "auxiliaries tied to time-series exposures, which are unavailable "
                        "for single-spectrum input. Use data_format='timeseries' or set "
                        "APPLY_SYSREM_DEFAULT=False."
                    )
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. "
                    "Choose from {'timeseries', 'spectrum'}."
                )

        primary_wav_obs_before_thinning = np.asarray(wav_obs, dtype=float).copy()
        (
            primary_observed_min,
            primary_observed_max,
            _,
            _,
        ) = _resolve_model_wavelength_range(primary_wav_obs_before_thinning)
        primary_grid_source_wavelength_range = (
            primary_observed_min,
            primary_observed_max,
        )

        if primary_pre_sysrem_data is not None and primary_pre_sysrem_sigma is not None:
            primary_pre_sysrem_data = np.asarray(primary_pre_sysrem_data)
            primary_pre_sysrem_sigma = np.asarray(primary_pre_sysrem_sigma)
            if (
                primary_pre_sysrem_data.shape != np.asarray(data).shape
                or primary_pre_sysrem_sigma.shape != np.asarray(sigma).shape
            ):
                raise ValueError(
                    "Pre-SYSREM diagnostic arrays must match the primary "
                    f"data/sigma shape {np.asarray(data).shape}; got "
                    f"{primary_pre_sysrem_data.shape} and {primary_pre_sysrem_sigma.shape}."
                )

        primary_frozen_subset_indices = (
            None
            if primary_frozen_timeseries_operator is None
            or (spectral_stride == 1 and spectral_offset == 0)
            else _spectral_subset_indices(
                np.asarray(wav_obs).size,
                spectral_stride,
                spectral_offset,
            )
        )
        wav_obs, data, sigma, primary_sysrem, primary_pre_sysrem_data, primary_pre_sysrem_sigma = _apply_spectral_thinning(
            wav_obs=wav_obs,
            data=data,
            sigma=sigma,
            sysrem=primary_sysrem,
            pre_sysrem_data=primary_pre_sysrem_data,
            pre_sysrem_sigma=primary_pre_sysrem_sigma,
            spectral_stride=spectral_stride,
            spectral_offset=spectral_offset,
            component_name=primary_component_name,
        )
        primary_frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
            primary_frozen_timeseries_operator,
            primary_frozen_subset_indices,
        )
        print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} Angstroms")

        # Convert to wavenumber
        inst_nus = wav2nu(wav_obs, "AA")

        # Ensure wavenumber grid and data are in ascending order
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
            if primary_pre_sysrem_data is not None and primary_pre_sysrem_sigma is not None:
                if primary_pre_sysrem_data.ndim == 2:
                    primary_pre_sysrem_data = primary_pre_sysrem_data[:, sort_idx]
                    primary_pre_sysrem_sigma = primary_pre_sysrem_sigma[:, sort_idx]
                else:
                    primary_pre_sysrem_data = primary_pre_sysrem_data[sort_idx]
                    primary_pre_sysrem_sigma = primary_pre_sysrem_sigma[sort_idx]
            primary_sysrem = _remap_sysrem_wavelength_sort(primary_sysrem, sort_idx)
            primary_frozen_timeseries_operator = _remap_frozen_timeseries_wavelengths(
                primary_frozen_timeseries_operator,
                sort_idx,
            )

        primary_frozen_timeseries = _build_model_frozen_timeseries(
            primary_frozen_timeseries_operator,
            primary_sysrem,
        )

        if mode == "emission" and emission_selection is not None:
            primary_collapsed_emission = _load_collapsed_emission_operator(
                resolved_data_dir,
                wav_obs,
            )
        elif mode == "transmission" and data_format == "spectrum":
            primary_collapsed_transmission = (
                _load_collapsed_transmission_operator(
                    resolved_data_dir,
                    wav_obs,
                )
            )

        sigma = _scale_spectroscopic_sigma(sigma, sigma_scale)
        if primary_subtract_weighted_global_mean:
            data = _subtract_inverse_variance_weighted_constant(data, sigma)
            print("  Subtracted the inverse-variance-weighted constant from the 1D spectrum")
        if primary_pre_sysrem_sigma is not None:
            primary_pre_sysrem_sigma = _scale_spectroscopic_sigma(primary_pre_sysrem_sigma, sigma_scale)
        if sigma_scale != 1.0:
            print(f"  Applied spectroscopic sigma scale: x{sigma_scale:g}")

        _preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

    # Setup instrumental resolution
    print("\n[2/7] Setting up instrumental resolution...")
    with _StepTimer("Step 2/7"):
        Rinst = config_utils.get_resolution()
        print(f"  Instrument resolving power: R = {Rinst:.0f}")

    nu_grid = None
    sop_rot = None
    sop_inst = None
    opa_cias = None
    opa_mols = None
    opa_atoms = None

    # Setup wavenumber grid
    print("\n[3/7] Building wavenumber grid...")
    with _StepTimer("Step 3/7"):
        if full_arm_mode:
            print(
                "  Full-arm mode: skipping global HRS grid; "
                "red and blue components build arm-specific grids in step 6."
            )
        else:
            if primary_wav_obs_before_thinning is None:
                raise RuntimeError("Primary model-grid wavelengths were not captured.")
            observed_min, observed_max, model_min, model_max = (
                _resolve_model_wavelength_range(primary_wav_obs_before_thinning)
            )
            nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
                model_min,
                model_max,
                config.N_SPECTRAL_POINTS,
                unit="AA",
            )
            _preflight_grid_checks(inst_nus, nu_grid)
            print(
                "  Complete observed range before thinning: "
                f"{observed_min:.3f} - {observed_max:.3f} Angstroms"
            )
            print(
                f"  Padded model range: {model_min:.3f} - "
                f"{model_max:.3f} Angstroms"
            )

            rotation_vsini_max = _rotation_operator_vsini_max(
                params.get("v_sini_star") if mode == "emission" else None,
            )
            sop_rot, sop_inst, _ = setup_spectral_operators(
                nu_grid,
                Rinst,
                vsini_max=rotation_vsini_max,
            )
            print("  Spectral operators initialized")
            if rotation_vsini_max > DEFAULT_ROTATION_VSINI_MAX_KMS:
                print(
                    "  Stellar rotation operator support: "
                    f"vsini_max={rotation_vsini_max:g} km/s"
                )

    # Setup primary atmospheric RT geometry
    print("\n[4/7] Initializing primary atmospheric RT...")
    with _StepTimer("Step 4/7"):
        primary_art = _build_art_for_mode(
            mode,
            reference_pressure_bar=reference_pressure_bar,
        )
        pressure_top, pressure_btm = config_utils.get_pressure_bounds_for_mode(mode)
        print(f"  {config.NLAYER} atmospheric layers")
        print(f"  Pressure range: {pressure_top:.1e} - {pressure_btm:.1e} bar")
        print(f"  Temperature range: {config.T_LOW:.0f} - {config.T_HIGH:.0f} K")

    # Load opacities
    print("\n[5/7] Loading opacities...")
    with _StepTimer("Step 5/7"):
        if full_arm_mode:
            print(
                "  Full-arm mode: skipping global opacity preload; "
                "each arm loads opacities on its own grid in step 6."
            )
        else:
            opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
            n_cia = sum(1 for cia in opa_cias.values() if not getattr(cia, "_is_dummy", False))
            if n_cia == 0:
                print("  Loaded 0 CIA sources (no overlap with nu_grid)")
            else:
                print(f"  Loaded {n_cia} CIA sources")

            resolved_primary_hitemp = (
                config.MOLPATH_HITEMP
                if primary_molpath_hitemp is None
                else primary_molpath_hitemp
            )
            resolved_primary_exomol = (
                config.MOLPATH_EXOMOL
                if primary_molpath_exomol is None
                else primary_molpath_exomol
            )
            resolved_primary_atoms = (
                config.ATOMIC_SPECIES
                if primary_atomic_species is None
                else primary_atomic_species
            )
            opa_mols, _molmass_arr = load_molecular_opacities(
                resolved_primary_hitemp,
                resolved_primary_exomol,
                nu_grid,
                config.OPA_LOAD,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                cutwing=config.PREMODIT_CUTWING,
            )
            print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

            # Load atomic opacities (optional, uses Kurucz with auto-download)
            opa_atoms, _atommass_arr = load_atomic_opacities(
                resolved_primary_atoms,
                nu_grid,
                config.OPA_LOAD,
                config.DIFFMODE,
                config.T_LOW,
                config.T_HIGH,
                cutwing=config.PREMODIT_CUTWING,
            )
            if opa_atoms:
                print(f"  Loaded {len(opa_atoms)} atomic species: {list(opa_atoms.keys())}")

    print(f"\n[6/7] Building retrieval model (primary={mode}, default P-T={pt_profile})...")
    print(f"  Chemistry model: {chemistry_model}")
    with _StepTimer("Step 6/7"):
        model_params = _coerce_model_params(params)
        # Persist the resolved radius-pressure convention alongside the other
        # runtime model parameters.  Catalog ``P0`` values remain metadata;
        # transmission uses the explicit adopted reference pressure below.
        model_params["reference_pressure_bar"] = reference_pressure_bar
        primary_region_name = _default_region_name_for_mode(mode)
        if mode == "emission":
            print("  PHOENIX stellar spectrum: auto-fetch/cache")
            print(
                "  PHOENIX surface-flux cache: "
                f"{_normalize_phoenix_cache_dir(phoenix_cache_dir)}"
            )
            if model_params["v_sini_star"] is None or not np.isfinite(
                float(model_params["v_sini_star"])
            ):
                print("  Stellar rotation: skipped (no finite v_sini_star)")
            else:
                print(
                    "  Stellar rotation: "
                    f"v sin(i)={float(model_params['v_sini_star']):g} km/s"
                )
            print(f"  Stellar instrumental profile: Gaussian convolution at R={Rinst:g}")
            print("  Stellar denominator velocity: 0 km/s (stellar-rest frame)")

        primary_is_timeseries = (
            np.asarray(phase).size > 1
            or bool(apply_sysrem)
            or (np.asarray(data).ndim == 2 and np.asarray(data).shape[0] > 1)
        )
        primary_radial_velocity_mode = "orbital" if primary_is_timeseries else "none"
        primary_sample_prefix = (
            primary_component_name
            if (joint_spectra or bandpass_constraints)
            else None
        )
        if (
            primary_wav_obs_before_thinning is None
            or primary_grid_source_wavelength_range is None
        ):
            raise RuntimeError("Primary model-grid wavelength range was not captured.")

        if full_arm_mode:
            primary_component_spec = _build_in_memory_timeseries_component_spec(
                name=primary_component_name,
                mode=mode,
                wav_obs=wav_obs,
                data=data,
                sigma=sigma,
                phase=phase,
                sysrem=primary_sysrem,
                frozen_timeseries_operator=primary_frozen_timeseries_operator,
                instrument_resolution=Rinst,
                apply_sysrem=apply_sysrem,
                radial_velocity_mode=primary_radial_velocity_mode,
                data_format=data_format,
                subtract_weighted_global_mean=primary_subtract_weighted_global_mean,
                collapsed_emission=primary_collapsed_emission,
                collapsed_transmission=primary_collapsed_transmission,
                region_name=primary_region_name,
                pre_sysrem_data=primary_pre_sysrem_data,
                pre_sysrem_sigma=primary_pre_sysrem_sigma,
                wav_obs_before_thinning=primary_wav_obs_before_thinning,
            )
            primary_component_spec.update(
                {
                    "atomic_species": primary_atomic_species,
                    "molpath_hitemp": primary_molpath_hitemp,
                    "molpath_exomol": primary_molpath_exomol,
                }
            )
            primary_component = _load_joint_spectroscopic_component(
                primary_component_spec,
                default_mode=mode,
                default_tstar=model_params["T_star"],
                default_logg_star=model_params["logg_star"],
                default_metallicity=model_params["Fe_H"],
                default_mstar=model_params["M_star"],
                default_rstar=model_params["R_star"],
                default_stellar_vsini=model_params["v_sini_star"],
                default_stellar_limb_darkening_u1=model_params["gamma1"],
                default_stellar_limb_darkening_u2=model_params["gamma2"],
                default_phoenix_cache_dir=phoenix_cache_dir,
                default_sigma_scale=1.0,
                default_spectral_stride=1,
                default_spectral_offset=0,
            )
        else:
            primary_component = _build_primary_spectroscopic_component(
                name=primary_component_name,
                mode=mode,
                wav_obs=wav_obs,
                grid_source_wavelength_range=primary_grid_source_wavelength_range,
                data=data,
                sigma=sigma,
                phase=phase,
                sysrem=primary_sysrem,
                frozen_timeseries=primary_frozen_timeseries,
                instrument_resolution=Rinst,
                nu_grid=nu_grid,
                inst_nus=inst_nus,
                sop_rot=sop_rot,
                sop_inst=sop_inst,
                opa_cias=opa_cias,
                opa_mols=opa_mols,
                opa_atoms=opa_atoms,
                region_name=primary_region_name,
                Tstar=model_params["T_star"],
                logg_star=model_params["logg_star"],
                metallicity=model_params["Fe_H"],
                Mstar=model_params["M_star"],
                Rstar=model_params["R_star"],
                stellar_vsini=model_params["v_sini_star"],
                stellar_limb_darkening_u1=model_params["gamma1"],
                stellar_limb_darkening_u2=model_params["gamma2"],
                phoenix_cache_dir=phoenix_cache_dir,
                apply_sysrem=apply_sysrem,
                radial_velocity_mode=primary_radial_velocity_mode,
                subtract_weighted_global_mean=primary_subtract_weighted_global_mean,
                collapsed_emission=primary_collapsed_emission,
                collapsed_transmission=primary_collapsed_transmission,
                sample_prefix=primary_sample_prefix,
                pre_sysrem_data=primary_pre_sysrem_data,
                pre_sysrem_sigma=primary_pre_sysrem_sigma,
            )
        observation_configs: list[object] = [primary_component.observation_config]
        observations_payload: dict[str, object] = {primary_component.name: primary_component.observation_inputs}

        auxiliary_components: list[SpectroscopicComponentBundle] = []
        if joint_spectra:
            for spec in joint_spectra:
                component = _load_joint_spectroscopic_component(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_stellar_vsini=model_params["v_sini_star"],
                    default_stellar_limb_darkening_u1=model_params["gamma1"],
                    default_stellar_limb_darkening_u2=model_params["gamma2"],
                    default_phoenix_cache_dir=phoenix_cache_dir,
                    default_sigma_scale=sigma_scale,
                    default_spectral_stride=spectral_stride,
                    default_spectral_offset=spectral_offset,
                )
                if component.name in observations_payload:
                    raise ValueError(f"Duplicate joint component name: {component.name}")
                auxiliary_components.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        scalar_constraints: list[BandpassConstraintBundle] = []
        if bandpass_constraints:
            for spec in bandpass_constraints:
                component = _load_bandpass_constraint(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_semi_major_axis_au=model_params["a"],
                    default_phoenix_cache_dir=phoenix_cache_dir,
                )
                if component.name in observations_payload:
                    raise ValueError(f"Duplicate joint component name: {component.name}")
                scalar_constraints.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        if full_arm_mode:
            orbital_component_names = tuple(
                [primary_component.name]
                + [
                    component.name
                    for component in auxiliary_components
                    if component.observation_config.radial_velocity_mode != "none"
                ]
            )
            print(
                f"  Full-arm mode: sharing one global v_sys across "
                f"{len(orbital_component_names)} spectroscopic components: "
                f"{', '.join(orbital_component_names)}."
            )

        shared_system = build_shared_system_config(
            params=model_params,
            reference_pressure_bar=reference_pressure_bar,
            prior_modes=shared_prior_modes,
        )

        atmosphere_region_configs, atmosphere_region_lookup = _build_atmosphere_regions(
            model_params=model_params,
            primary_mode=mode,
            primary_region_name=primary_region_name,
            primary_art=primary_art,
            observation_configs=observation_configs,
            default_pt_profile=pt_profile,
            default_chemistry_model=chemistry_model,
            default_fastchem_parameter_file=fastchem_parameter_file,
            atmosphere_regions=atmosphere_regions,
        )
        joint_model = create_joint_retrieval_model(
            shared_system=shared_system,
            atmosphere_regions=atmosphere_region_configs,
            observations=tuple(observation_configs),
        )
        model_c = joint_model
        model_inputs = {"observations": observations_payload}
        shared_region_config = atmosphere_region_lookup[primary_region_name]
        shared_pt_profile = shared_region_config.pt_profile
        shared_region_sample_prefix = shared_region_config.sample_prefix
        spectroscopic_components: list[SpectroscopicComponentBundle] = [primary_component]
        spectroscopic_components.extend(auxiliary_components)
        spectroscopic_component_count = len(spectroscopic_components)
        component_names = [component.name for component in spectroscopic_components]
        component_names.extend(component.name for component in scalar_constraints)
        print(
            f"  Joint model created with {len(component_names)} component(s): "
            f"{', '.join(component_names)}"
        )
        print(
            "  Atmosphere regions: "
            + ", ".join(
                f"{region_config.name} "
                f"[{next(cfg.mode for cfg in observation_configs if cfg.region_name == region_config.name)}]"
                for region_config in atmosphere_region_configs
            )
        )

    _append_resolved_spectral_grid_config(output_dir, spectroscopic_components)

    # Run inference
    print("\n[7/7] Running Bayesian inference...")
    rng_key = random.PRNGKey(seed)

    init_strategy = init_to_median(num_samples=config.INIT_TO_MEDIAN_SAMPLES)
    svi_params: dict | None = None
    svi_guide: object | None = None
    svi_losses: np.ndarray | None = None

    with _StepTimer("Step 7/7"):
        if not skip_svi:
            svi_lr_message = f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}"
            if config.SVI_LR_DECAY_STEPS is not None and config.SVI_LR_DECAY_RATE is not None:
                svi_lr_message += (
                    " with exponential decay "
                    f"(steps={config.SVI_LR_DECAY_STEPS}, rate={config.SVI_LR_DECAY_RATE})"
                )
            print(svi_lr_message)
            rng_key, rng_key_ = random.split(rng_key)
            svi_params, svi_losses, init_strategy, _, svi_guide = run_svi(
                model_c,
                rng_key_,
                model_inputs=model_inputs,
                Mp_mean=model_params["M_p"],
                Mp_std=model_params["M_p_err"],
                Mp_upper_3sigma=model_params.get("M_p_upper_3sigma"),
                Rp_mean=model_params["R_p"],
                Rp_std=model_params["R_p_err"],
                Rstar_mean=model_params["R_star"],
                Rstar_std=model_params["R_star_err"],
                output_dir=str(output_dir),
                num_steps=config.SVI_NUM_STEPS,
                lr=config.SVI_LEARNING_RATE,
                lr_decay_steps=config.SVI_LR_DECAY_STEPS,
                lr_decay_rate=config.SVI_LR_DECAY_RATE,
                prior_modes=shared_prior_modes,
            )

            if svi_only:
                print(
                    "  SVI-only outputs are approximate diagnostics; "
                    "use MCMC/NUTS after SVI warm-up for production posterior inference."
                )
                print("  Generating corner plots from SVI posterior...")
                rng_key, rng_key_plot = random.split(rng_key)
                svi_samples_for_plots = _sample_svi_posterior(
                    guide=svi_guide,
                    params=svi_params,
                    rng_key=rng_key_plot,
                    num_samples=max(100, int(config.MCMC_NUM_SAMPLES)),
                )
                save_retrieval_corner_plots(
                    output_dir=str(output_dir),
                    svi_samples=svi_samples_for_plots,
                )

                if svi_losses is not None:
                    plot_svi_loss(
                        np.asarray(jax.device_get(svi_losses)),
                        os.path.join(output_dir, "svi_loss.pdf"),
                    )

                if svi_samples_for_plots is not None:
                    try:
                        plot_temperature_profile(
                            posterior_samples=svi_samples_for_plots,
                            art=shared_region_config.art,
                            save_path=os.path.join(output_dir, "temperature_profile.pdf"),
                            pt_profile=shared_pt_profile,
                            sample_prefix=shared_region_sample_prefix,
                            Tint_fixed=shared_region_config.Tint_fixed,
                        )
                    except Exception as exc:
                        print(
                            "  Skipping temperature profile plot for SVI samples: "
                            f"{exc}"
                        )
                    for component in spectroscopic_components:
                        component_region = atmosphere_region_lookup[
                            component.observation_config.region_name
                        ]
                        component_obs_mean, component_obs_err = _summarize_observed_spectrum(
                            component.data,
                            component.sigma,
                        )
                        component_pre_obs_mean = None
                        component_pre_obs_err = None
                        if component.pre_sysrem_data is not None and component.pre_sysrem_sigma is not None:
                            component_pre_obs_mean, component_pre_obs_err = _summarize_observed_spectrum(
                                component.pre_sysrem_data,
                                component.pre_sysrem_sigma,
                            )
                        component_wav_obs_nm = np.asarray(component.wav_obs) / 10.0
                        svi_model_ts, _ = _compute_model_timeseries_for_plot(
                            posterior_samples=svi_samples_for_plots,
                            model_params=model_params,
                            region_config=component_region,
                            component=component,
                            region_sample_prefix=component_region.sample_prefix,
                        )

                        if svi_model_ts is not None:
                            svi_line = np.mean(np.asarray(svi_model_ts), axis=0)
                            if component.observation_config.mode == "transmission":
                                plot_transmission_spectrum(
                                    wavelength_nm=component_wav_obs_nm,
                                    rp_obs=component_obs_mean,
                                    rp_err=component_obs_err,
                                    rp_hmc=np.atleast_2d(svi_line),
                                    rp_svi=None,
                                    rp_pre_sysrem=component_pre_obs_mean,
                                    rp_pre_sysrem_err=component_pre_obs_err,
                                    save_path=os.path.join(
                                        output_dir,
                                        _component_output_filename(
                                            "transmission_spectrum.pdf",
                                            component.name,
                                            num_components=spectroscopic_component_count,
                                        ),
                                    ),
                                )
                            else:
                                plot_emission_spectrum(
                                    wavelength_nm=component_wav_obs_nm,
                                    fp_obs=component_obs_mean,
                                    fp_err=component_obs_err,
                                    fp_hmc=np.atleast_2d(svi_line),
                                    fp_svi=svi_line,
                                    save_path=os.path.join(
                                        output_dir,
                                        _component_output_filename(
                                            "emission_spectrum.pdf",
                                            component.name,
                                            num_components=spectroscopic_component_count,
                                        ),
                                    ),
                                )
                print(
                    "  SVI complete (svi_only=True); skipping MCMC. "
                    "Treat posterior products as approximate diagnostics."
                )
                svi_bundle = PublicationBundle(
                    run_dir=Path(output_dir),
                    metadata={
                        "target": config.PLANET,
                        "ephemeris": config.EPHEMERIS,
                        "retrieval_mode": mode,
                        "epochs": list(epochs),
                        "inference": "SVI-only",
                        "publication_eligible": False,
                    },
                )
                svi_bundle.record_failure(
                    figure_id="hmc_posterior_required",
                    tier="paper",
                    required=True,
                    error=(
                        "SVI-only runs are approximate diagnostics and do not produce "
                        "a publication-complete posterior figure bundle."
                    ),
                )
                svi_bundle.finalize()
                return

        print(f"\n  Running HMC-NUTS sampling...")
        print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
        print(f"  Chains: {config.MCMC_NUM_CHAINS}")
        print(f"  Chain method: {config.MCMC_CHAIN_METHOD}")

        _validate_mcmc_device_layout(
            num_chains=config.MCMC_NUM_CHAINS,
            chain_method=config.MCMC_CHAIN_METHOD,
            require_gpu_per_chain=config.MCMC_REQUIRE_GPU_PER_CHAIN,
        )

        kernel = NUTS(
            model_c,
            max_tree_depth=config.MCMC_MAX_TREE_DEPTH,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=config.MCMC_NUM_WARMUP,
            num_samples=config.MCMC_NUM_SAMPLES,
            num_chains=config.MCMC_NUM_CHAINS,
            chain_method=config.MCMC_CHAIN_METHOD,
        )

        rng_key, rng_key_ = random.split(rng_key)
        mcmc_run_kwargs = dict(model_inputs)
        if save_mcmc_diagnostics:
            mcmc_run_kwargs["extra_fields"] = DEFAULT_MCMC_EXTRA_FIELDS
        mcmc.run(
            rng_key_,
            **mcmc_run_kwargs,
        )
    
    mcmc.print_summary()
    
    # Save results
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()
    
    posterior_sample_by_chain, diagnostics_warnings = get_samples_by_chain(mcmc)
    save_chain_grouped_posterior(output_dir, posterior_sample_by_chain)

    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)

    if save_mcmc_diagnostics:
        mcmc_extra_fields, extra_warnings = get_extra_fields_by_chain(mcmc)
        diagnostics_warnings.extend(extra_warnings)
        diagnostics_summary = write_mcmc_diagnostics(
            output_dir,
            posterior_by_chain=posterior_sample_by_chain,
            extra_fields=mcmc_extra_fields,
            num_chains=int(config.MCMC_NUM_CHAINS),
            num_samples=int(config.MCMC_NUM_SAMPLES),
            max_tree_depth=int(config.MCMC_MAX_TREE_DEPTH),
            warnings=diagnostics_warnings,
            diagnostic_label=sanitized_label,
        )
        flags = diagnostics_summary.get("warning_flags", {})
        flagged = ", ".join(name for name, value in flags.items() if value)
        print(
            "  MCMC diagnostics saved to "
            f"{os.path.join(output_dir, 'diagnostics')}"
            + (f" (flags: {flagged})" if flagged else "")
        )

    posterior_np: dict[str, np.ndarray] | None = None
    svi_samples_for_plots: dict[str, np.ndarray] | None = None

    posterior_np = {}
    for name, values in posterior_sample.items():
        posterior_np[name] = np.asarray(jax.device_get(values))

    if svi_params is not None and svi_guide is not None:
        n_hmc_samples = max(100, int(config.MCMC_NUM_SAMPLES))
        if posterior_np:
            first_site = next(iter(posterior_np))
            n_hmc_samples = int(np.asarray(posterior_np[first_site]).shape[0])

        rng_key, rng_key_plot = random.split(rng_key)
        svi_samples_for_plots = _sample_svi_posterior(
            guide=svi_guide,
            params=svi_params,
            rng_key=rng_key_plot,
            num_samples=n_hmc_samples,
        )

    if svi_losses is not None:
        try:
            plot_svi_loss(
                np.asarray(jax.device_get(svi_losses)),
                os.path.join(output_dir, "svi_loss.pdf"),
            )
        except Exception as exc:
            print(f"  Warning: failed to generate SVI loss plot; continuing. ({exc})")

    try:
        plot_temperature_profile(
            posterior_samples=posterior_np,
            art=shared_region_config.art,
            save_path=os.path.join(output_dir, "temperature_profile.pdf"),
            pt_profile=shared_pt_profile,
            sample_prefix=shared_region_sample_prefix,
            Tint_fixed=shared_region_config.Tint_fixed,
        )
    except Exception as exc:
        print(
            "  Skipping temperature profile plot for HMC samples: "
            f"{exc}"
        )

    component_atmo_states: dict[str, dict] = {}
    print("\n  Computing atmospheric state from posterior...")

    for component in spectroscopic_components:
        try:
            component_region = atmosphere_region_lookup[
                component.observation_config.region_name
            ]
            component_atmo_states[component.name] = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_np,
                region_config=component_region,
                opa_mols=component.opa_mols,
                opa_atoms=component.opa_atoms,
                opa_cias=component.opa_cias,
                nu_grid=component.nu_grid,
                use_median=True,
                sample_prefix=component_region.sample_prefix,
            )
        except Exception as exc:
            if compute_contribution:
                raise
            print(
                "  Warning: unable to compute atmospheric state for "
                f"{component.name}; skipping that component's diagnostics. ({exc})"
            )

    if component_atmo_states:
        print("  Plotting fitted spectrum diagnostics...")
        for component in spectroscopic_components:
            try:
                component_region = atmosphere_region_lookup[
                    component.observation_config.region_name
                ]
                atmo_state = component_atmo_states.get(component.name)
                if atmo_state is None:
                    continue

                component_wav_obs_nm = np.asarray(component.wav_obs) / 10.0
                component_obs_mean, component_obs_err = _summarize_observed_spectrum(
                    component.data,
                    component.sigma,
                )
                component_pre_obs_mean = None
                component_pre_obs_err = None
                if component.pre_sysrem_data is not None and component.pre_sysrem_sigma is not None:
                    component_pre_obs_mean, component_pre_obs_err = _summarize_observed_spectrum(
                        component.pre_sysrem_data,
                        component.pre_sysrem_sigma,
                    )

                hmc_model_ts, atmo_state = _compute_model_timeseries_for_plot(
                    posterior_samples=posterior_np,
                    model_params=model_params,
                    region_config=component_region,
                    component=component,
                    region_sample_prefix=component_region.sample_prefix,
                    atmo_state=atmo_state,
                )
                component_atmo_states[component.name] = atmo_state

                svi_model_ts = None
                if svi_samples_for_plots is not None:
                        svi_model_ts, _ = _compute_model_timeseries_for_plot(
                        posterior_samples=svi_samples_for_plots,
                        model_params=model_params,
                            region_config=component_region,
                            component=component,
                            region_sample_prefix=component_region.sample_prefix,
                    )

                if hmc_model_ts is not None or svi_model_ts is not None:
                    hmc_plot = hmc_model_ts
                    if hmc_plot is None and svi_model_ts is not None:
                        hmc_plot = np.atleast_2d(np.mean(np.asarray(svi_model_ts), axis=0))

                    if hmc_plot is not None:
                        hmc_line = np.mean(np.asarray(hmc_plot), axis=0)
                        svi_line = hmc_line
                        if svi_model_ts is not None:
                            svi_line = np.mean(np.asarray(svi_model_ts), axis=0)

                        if component.observation_config.mode == "transmission":
                            plot_transmission_spectrum(
                                wavelength_nm=component_wav_obs_nm,
                                rp_obs=component_obs_mean,
                                rp_err=component_obs_err,
                                rp_hmc=np.atleast_2d(hmc_line),
                                rp_svi=None if svi_model_ts is None else np.asarray(svi_line),
                                rp_pre_sysrem=component_pre_obs_mean,
                                rp_pre_sysrem_err=component_pre_obs_err,
                                save_path=os.path.join(
                                    output_dir,
                                    _component_output_filename(
                                        "transmission_spectrum.pdf",
                                        component.name,
                                        num_components=spectroscopic_component_count,
                                    ),
                                ),
                            )
                        else:
                            plot_emission_spectrum(
                                wavelength_nm=component_wav_obs_nm,
                                fp_obs=component_obs_mean,
                                fp_err=component_obs_err,
                                fp_hmc=np.atleast_2d(hmc_line),
                                fp_svi=np.asarray(svi_line),
                                save_path=os.path.join(
                                    output_dir,
                                    _component_output_filename(
                                        "emission_spectrum.pdf",
                                        component.name,
                                        num_components=spectroscopic_component_count,
                                    ),
                                ),
                            )
            except Exception as exc:
                print(
                    "  Warning: failed to generate fitted spectrum diagnostic for "
                    f"{component.name}; continuing. ({exc})"
                )

    if compute_contribution and component_atmo_states:
        for component in spectroscopic_components:
            atmo_state = component_atmo_states.get(component.name)
            if atmo_state is None:
                continue

            np.savez(
                os.path.join(
                    output_dir,
                    _component_output_filename(
                        "atmospheric_state.npz",
                        component.name,
                        num_components=spectroscopic_component_count,
                    ),
                ),
                dtau=np.array(atmo_state['dtau']),
                Tarr=np.array(atmo_state['Tarr']),
                pressure=np.array(atmo_state['pressure']),
                dParr=np.array(atmo_state['dParr']),
                mmw=np.array(atmo_state['mmw']),
                vmrH2=np.array(atmo_state['vmrH2']),
                vmrHe=np.array(atmo_state['vmrHe']),
            )

        print("  Plotting contribution function(s)...")

        for component in spectroscopic_components:
            try:
                atmo_state = component_atmo_states.get(component.name)
                if atmo_state is None:
                    continue

                contribution_title = (
                    f"{config.PLANET} Contribution Function "
                    f"({component.observation_config.mode})"
                )
                if spectroscopic_component_count > 1:
                    contribution_title += f" [{component.name}]"

                plot_contribution_function(
                    nu_grid=np.array(component.nu_grid),
                    dtau=np.array(atmo_state['dtau']),
                    Tarr=np.array(atmo_state['Tarr']),
                    pressure=np.array(atmo_state['pressure']),
                    dParr=np.array(atmo_state['dParr']),
                    mode=component.observation_config.mode,
                    save_path=os.path.join(
                        output_dir,
                        _component_output_filename(
                            "contribution_function.pdf",
                            component.name,
                            num_components=spectroscopic_component_count,
                        ),
                    ),
                    wavelength_unit="AA",
                    title=contribution_title,
                )

                if atmo_state['dtau_per_species']:
                    dtau_per_species_np = {}
                    for k, v in atmo_state['dtau_per_species'].items():
                        dtau_per_species_np[k] = np.array(v)

                    plot_contribution_per_species(
                        nu_grid=np.array(component.nu_grid),
                        dtau_per_species=dtau_per_species_np,
                        Tarr=np.array(atmo_state['Tarr']),
                        pressure=np.array(atmo_state['pressure']),
                        dParr=np.array(atmo_state['dParr']),
                        mode=component.observation_config.mode,
                        save_path=os.path.join(
                            output_dir,
                            _component_output_filename(
                                "contribution_per_species.pdf",
                                component.name,
                                num_components=spectroscopic_component_count,
                            ),
                        ),
                        wavelength_unit="AA",
                    )

                    plot_contribution_combined(
                        nu_grid=np.array(component.nu_grid),
                        dtau=np.array(atmo_state['dtau']),
                        dtau_per_species=dtau_per_species_np,
                        Tarr=np.array(atmo_state['Tarr']),
                        pressure=np.array(atmo_state['pressure']),
                        dParr=np.array(atmo_state['dParr']),
                        mode=component.observation_config.mode,
                        save_path=os.path.join(
                            output_dir,
                            _component_output_filename(
                                "contribution_combined.pdf",
                                component.name,
                                num_components=spectroscopic_component_count,
                            ),
                        ),
                        wavelength_unit="AA",
                    )
            except Exception as exc:
                print(
                    "  Warning: failed to generate contribution plot for "
                    f"{component.name}; continuing. ({exc})"
                )

        print(f"  Contribution function plots saved to {output_dir}/")

    print("\n  Generating corner plots...")
    try:
        save_retrieval_corner_plots(
            output_dir=str(output_dir),
            hmc_samples=posterior_np,
            svi_samples=svi_samples_for_plots,
        )
    except Exception as exc:
        print(f"  Warning: failed to generate corner plots; continuing. ({exc})")

    print("\n  Generating versioned publication figure bundle...")
    try:
        publication_manifest = _generate_publication_bundle(
            output_dir=output_dir,
            mode=mode,
            epochs=epochs,
            pt_profile=shared_pt_profile,
            chemistry_model=chemistry_model,
            posterior_samples=posterior_np,
            posterior_by_chain=posterior_sample_by_chain,
            svi_samples=svi_samples_for_plots,
            svi_losses=(
                None
                if svi_losses is None
                else np.asarray(jax.device_get(svi_losses))
            ),
            model_params=model_params,
            atmosphere_region_lookup=atmosphere_region_lookup,
            spectroscopic_components=spectroscopic_components,
            bandpass_components=scalar_constraints,
            compute_contribution=compute_contribution,
        )
        publication_status = publication_manifest["publication_bundle_complete"]
        print(
            "  Publication bundle status: "
            f"{'complete' if publication_status else 'incomplete'}; "
            f"manifest={Path(output_dir) / 'figures_manifest.json'}"
        )
        if not publication_status:
            print(
                "  Required figure failures: "
                + ", ".join(publication_manifest.get("required_failures", []))
            )
    except Exception as exc:
        emergency_bundle = PublicationBundle(
            run_dir=Path(output_dir),
            metadata={
                "target": config.PLANET,
                "ephemeris": config.EPHEMERIS,
                "retrieval_mode": mode,
                "epochs": list(epochs),
                "inference": "HMC/NUTS",
            },
        )
        emergency_bundle.record_failure(
            figure_id="publication_bundle_generation",
            tier="paper",
            required=True,
            error=exc,
        )
        emergency_bundle.finalize()
        print(
            "  Warning: publication bundle generation failed; inference results "
            f"remain saved and the incomplete manifest records the failure. ({exc})"
        )
    
    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"Total runtime: {perf_counter() - retrieval_start:.2f}s")
    print("="*70)


if __name__ == "__main__":
    raise RuntimeError(
        "Direct __main__ execution no longer provides a chemistry default. "
        "Import run_retrieval(...) and pass chemistry_model explicitly."
    )
