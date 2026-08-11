#!/usr/bin/env python3
"""Run the joint transmission + emission HRS retrieval from IMG_5664.JPG."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.retrieval_plan import (
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


TERMINATOR_CASE = RetrievalCase(
    name="terminator_free_fe",
    description="Isothermal terminator with free Fe I/Fe II VMRs and separate Delta-v values.",
    atoms=("Fe I", "Fe II"),
    chemistry_model="constant",
    pt_profile="isothermal",
    velocity_offset_mode="species",
    velocity_offset_species=("Fe I", "Fe II"),
)

DAYSIDE_CASE = RetrievalCase(
    name="dayside_free_fe_na_ca",
    description=(
        "Guillot dayside with free Fe I/Fe II/Na I/Ca I VMRs and one dayside Delta-v."
    ),
    atoms=("Fe I", "Fe II", "Na I", "Ca I"),
    chemistry_model="constant",
    pt_profile="guillot",
    velocity_offset_mode="region",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Describe the joint model and exit.")
    parser.add_argument("--transmission-epoch", action="append", default=None)
    parser.add_argument("--emission-epoch", action="append", default=None)
    add_common_arguments(parser)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.list:
        print("joint_terminator_dayside")
        print(f"  terminator: {TERMINATOR_CASE.description}")
        print(f"  dayside:   {DAYSIDE_CASE.description}")
        print(
            "  shared:    Kp, Mp-derived gravity, Rp, Rstar, and fixed orbital ephemeris."
        )
        print("  likelihood: logL = logL_transmission + logL_emission across all components.")
        return 0

    transmission_epochs = validate_epochs(
        args.transmission_epoch,
        "--transmission-epoch",
    )
    emission_epochs = validate_epochs(args.emission_epoch, "--emission-epoch")
    case_name = "joint_terminator_dayside"
    configure_runtime(
        args,
        mode="transmission",
        case_name=case_name,
        output_category="joint",
    )

    transmission_preflight = preflight_timeseries(
        mode="transmission",
        epochs=transmission_epochs,
        args=args,
    )
    emission_preflight = preflight_timeseries(
        mode="emission",
        epochs=emission_epochs,
        args=args,
    )
    preflight = transmission_preflight + emission_preflight
    manifest = intent_manifest(
        runner="joint",
        case=case_name,
        args=args,
        extra={
            "transmission_epochs": transmission_epochs,
            "emission_epochs": emission_epochs,
            "terminator": asdict(TERMINATOR_CASE),
            "dayside": asdict(DAYSIDE_CASE),
            "shared_parameters": (
                "Kp",
                "Mp",
                "Rp",
                "Rstar",
                "period",
                "transit_epoch",
            ),
            "fixed_ephemeris_parameters": (
                "period",
                "transit_epoch (image T0)",
            ),
            "likelihood": "sum of every transmission and emission component logL",
        },
    )
    print_plan(manifest, preflight)
    if args.dry_run:
        return 0
    require_ready_timeseries(preflight, args)

    joint_spectra = auxiliary_timeseries_specs(
        mode="transmission",
        epochs=transmission_epochs[1:],
        args=args,
        region_name="terminator",
        atoms=TERMINATOR_CASE.atoms,
    )
    joint_spectra.extend(
        auxiliary_timeseries_specs(
            mode="emission",
            epochs=emission_epochs,
            args=args,
            region_name="dayside",
            atoms=DAYSIDE_CASE.atoms,
        )
    )
    atmosphere_regions = [
        atmosphere_region_spec(
            TERMINATOR_CASE,
            name="terminator",
            mode="transmission",
            args=args,
        ),
        atmosphere_region_spec(
            DAYSIDE_CASE,
            name="dayside",
            mode="emission",
            args=args,
        ),
    ]

    from pipeline.retrieval import run_retrieval

    run_retrieval(
        mode="transmission",
        epoch=transmission_epochs[0],
        data_format="timeseries",
        skip_svi=args.skip_svi,
        svi_only=args.svi_only,
        pt_profile=TERMINATOR_CASE.pt_profile,
        chemistry_model=TERMINATOR_CASE.chemistry_model,
        seed=args.seed,
        joint_spectra=joint_spectra,
        atmosphere_regions=atmosphere_regions,
        sigma_scale=args.sigma_scale,
        spectral_stride=args.spectral_stride,
        spectral_offset=args.spectral_offset,
        apply_sysrem_override=False if args.no_sysrem else None,
        shared_prior_modes=shared_prior_modes(args),
        primary_atomic_species=atomic_species_dict(TERMINATOR_CASE.atoms),
        primary_molpath_hitemp={},
        primary_molpath_exomol={},
        retrieval_intent=manifest,
        model_param_overrides=model_param_overrides(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
