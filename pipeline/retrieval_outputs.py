"""Evaluate posterior predictions and write retrieval diagnostics and figure bundles."""

from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import numpy as np
import config
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from jax import random
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import RJ, Rs
from physics.observation_operators import (
    apply_collapsed_emission_operator,
    apply_collapsed_transmission_operator,
    apply_frozen_timeseries_operator,
    apply_model_pipeline_corrections,
)
from physics.model import (
    compute_model_timeseries,
    _bandpass_weighted_mean,
    _compute_native_observable_spectrum,
    _transform_bandpass_observable,
)
from physics.posterior import (
    compute_atmospheric_state_from_posterior,
    reconstruct_temperature_profile,
)
from pipeline.mcmc_diagnostics import (
    get_extra_fields_by_chain,
    get_samples_by_chain,
    save_chain_grouped_posterior,
    write_mcmc_diagnostics,
)
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
from pipeline.retrieval_setup import (
    BandpassConstraintBundle,
    SpectroscopicComponentBundle,
    PreparedRetrieval,
)
from pipeline.retrieval_inputs import _build_model_chunked_sysrem
from pipeline.inference import InferenceResult


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
        return float(base_offset)

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
                    albedo = float(observation_config.geometric_albedo_bounds[0])
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


def write_retrieval_outputs(
    prepared: PreparedRetrieval,
    result: InferenceResult,
    *,
    output_dir: str | Path,
    mode: str,
    epochs: Sequence[str],
    chemistry_model: str | None,
    compute_contribution: bool,
    save_mcmc_diagnostics: bool,
    sanitized_label: str | None,
) -> None:
    """Save HMC posterior products and render their existing diagnostics."""
    model_params = prepared.model_params
    atmosphere_region_lookup = prepared.atmosphere_region_lookup
    shared_region_config = prepared.shared_region_config
    spectroscopic_components = prepared.spectroscopic_components
    scalar_constraints = prepared.scalar_constraints
    shared_pt_profile = shared_region_config.pt_profile
    shared_region_sample_prefix = shared_region_config.sample_prefix
    spectroscopic_component_count = len(spectroscopic_components)
    mcmc = result.mcmc
    rng_key = result.rng_key
    svi_params = result.svi_params
    svi_losses = result.svi_losses
    svi_guide = result.svi_guide
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


def write_svi_only_outputs(
    prepared: PreparedRetrieval,
    result: InferenceResult,
    *,
    output_dir: str | Path,
    mode: str,
    epochs: Sequence[str],
) -> None:
    """Write the existing approximate SVI diagnostics without HMC products."""
    model_params = prepared.model_params
    atmosphere_region_lookup = prepared.atmosphere_region_lookup
    shared_region_config = prepared.shared_region_config
    spectroscopic_components = prepared.spectroscopic_components
    scalar_constraints = prepared.scalar_constraints
    shared_pt_profile = shared_region_config.pt_profile
    shared_region_sample_prefix = shared_region_config.sample_prefix
    spectroscopic_component_count = len(spectroscopic_components)
    mcmc = result.mcmc
    rng_key = result.rng_key
    svi_params = result.svi_params
    svi_losses = result.svi_losses
    svi_guide = result.svi_guide
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
