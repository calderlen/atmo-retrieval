"""Retrieval orchestration: prepare observations and model, infer, and write results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import config
import config_utils
from pipeline.mcmc_diagnostics import sanitize_diagnostic_label
from pipeline.retrieval_inputs import SysremInputBundle
from pipeline.retrieval_setup import (
    _StepTimer,
    _normalize_retrieval_mode,
    prepare_retrieval,
)
from pipeline.inference import run_inference
from pipeline.retrieval_outputs import write_retrieval_outputs, write_svi_only_outputs


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
    sigma_scale = float(sigma_scale)
    spectral_stride, spectral_offset = int(spectral_stride), int(spectral_offset)
    epochs = _normalize_epoch_values(epoch)
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

    prepared = prepare_retrieval(
        mode=mode,
        epochs=epochs,
        data_format=data_format,
        pt_profile=pt_profile,
        chemistry_model=chemistry_model,
        fastchem_parameter_file=fastchem_parameter_file,
        wav_obs=wav_obs,
        data=data,
        sigma=sigma,
        phase=phase,
        U=U,
        sysrem_inputs=sysrem_inputs,
        joint_spectra=joint_spectra,
        bandpass_constraints=bandpass_constraints,
        atmosphere_regions=atmosphere_regions,
        phoenix_cache_dir=phoenix_cache_dir,
        sigma_scale=sigma_scale,
        spectral_stride=spectral_stride,
        spectral_offset=spectral_offset,
        apply_sysrem_override=apply_sysrem_override,
        emission_selection=emission_selection,
        shared_prior_modes=shared_prior_modes,
        primary_atomic_species=primary_atomic_species,
        primary_molpath_hitemp=primary_molpath_hitemp,
        primary_molpath_exomol=primary_molpath_exomol,
        model_param_overrides=model_param_overrides,
        output_dir=output_dir,
        reference_pressure_bar=reference_pressure_bar,
    )

    print("\n[7/7] Running Bayesian inference...")
    with _StepTimer("Step 7/7"):
        result = run_inference(
            prepared.model_c, prepared.model_inputs, prepared.model_params,
            output_dir=output_dir, seed=seed, skip_svi=skip_svi, svi_only=svi_only,
            shared_prior_modes=shared_prior_modes,
            save_mcmc_diagnostics=save_mcmc_diagnostics,
        )
        if result.mcmc is None:
            write_svi_only_outputs(
                prepared, result, output_dir=output_dir, mode=mode, epochs=epochs,
            )
            return

    write_retrieval_outputs(
        prepared, result, output_dir=output_dir, mode=mode, epochs=epochs,
        chemistry_model=chemistry_model, compute_contribution=compute_contribution,
        save_mcmc_diagnostics=save_mcmc_diagnostics, sanitized_label=sanitized_label,
    )

    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"Total runtime: {perf_counter() - retrieval_start:.2f}s")
    print("="*70)
