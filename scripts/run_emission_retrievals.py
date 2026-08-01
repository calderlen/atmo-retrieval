#!/usr/bin/env python3
"""Run the emission HRS retrieval specified in IMG_5664.JPG."""

from __future__ import annotations

import argparse

from retrieval_plan_common import (
    RetrievalCase,
    add_common_arguments,
    atmosphere_region_spec,
    atomic_species_dict,
    auxiliary_timeseries_specs,
    configure_runtime,
    intent_manifest,
    model_param_overrides,
    preflight_timeseries,
    print_plan,
    require_ready_timeseries,
    shared_prior_modes,
    validate_epochs,
)


EMISSION_CASE = RetrievalCase(
    name="free_fe_na_ca",
    description=(
        "Free constant Fe I, Na I, and Ca I VMRs; Guillot Tirr/kappa_IR/gamma; "
        "shared Kp and one dayside Delta-v."
    ),
    atoms=("Fe I", "Na I", "Ca I"),
    chemistry_model="constant",
    pt_profile="guillot",
    velocity_offset_mode="region",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Describe the retrieval case and exit.")
    parser.add_argument("--epoch", action="append", default=None)
    add_common_arguments(parser)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.list:
        print(f"{EMISSION_CASE.name:20s} {EMISSION_CASE.description}")
        return 0

    epochs = validate_epochs(args.epoch)
    configure_runtime(args, mode="emission", case_name=EMISSION_CASE.name)
    preflight = preflight_timeseries(mode="emission", epochs=epochs, args=args)
    manifest = intent_manifest(
        runner="emission",
        case=EMISSION_CASE,
        args=args,
        extra={
            "epochs": epochs,
            "temperature_prior": "target Tirr_mean/Tirr_std when finite",
            "guillot_parameters": ("Tirr", "kappa_ir_cgs", "gamma"),
            "fixed_ephemeris_parameters": ("period", "transit_epoch (image T0)"),
            "rotation_period": "tidally locked to the fixed orbital period",
        },
    )
    print_plan(manifest, preflight)
    if args.dry_run:
        return 0
    require_ready_timeseries(preflight, args)

    joint_spectra = auxiliary_timeseries_specs(
        mode="emission",
        epochs=epochs[1:],
        args=args,
        region_name="dayside",
        atoms=EMISSION_CASE.atoms,
    )
    region = atmosphere_region_spec(
        EMISSION_CASE,
        name="dayside",
        mode="emission",
        args=args,
    )

    from pipeline.retrieval import run_retrieval

    run_retrieval(
        mode="emission",
        epoch=epochs[0],
        data_format="timeseries",
        skip_svi=args.skip_svi,
        svi_only=args.svi_only,
        pt_profile=EMISSION_CASE.pt_profile,
        chemistry_model=EMISSION_CASE.chemistry_model,
        seed=args.seed,
        joint_spectra=joint_spectra or None,
        atmosphere_regions=[region],
        sigma_scale=args.sigma_scale,
        spectral_stride=args.spectral_stride,
        spectral_offset=args.spectral_offset,
        apply_sysrem_override=False if args.no_sysrem else None,
        shared_prior_modes=shared_prior_modes(args),
        primary_atomic_species=atomic_species_dict(EMISSION_CASE.atoms),
        primary_molpath_hitemp={},
        primary_molpath_exomol={},
        retrieval_intent=manifest,
        model_param_overrides=model_param_overrides(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
