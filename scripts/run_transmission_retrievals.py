#!/usr/bin/env python3
"""Run the transmission HRS retrieval ladder specified in IMG_5664.JPG."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_plan_common import (
    REPO_ROOT,
    RetrievalCase,
    add_common_arguments,
    all_atomic_species,
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


def _cases() -> dict[str, RetrievalCase]:
    common = {
        "chemistry_model": "constant",
        "pt_profile": "isothermal",
        "velocity_offset_mode": "species",
        "velocity_offset_species": ("Fe I", "Fe II"),
    }
    return {
        "free_fe": RetrievalCase(
            name="free_fe",
            description="Free constant Fe I + Fe II VMRs with separate Delta-v values.",
            atoms=("Fe I", "Fe II"),
            **common,
        ),
        "free_fe_na": RetrievalCase(
            name="free_fe_na",
            description="Fe I + Fe II + Na I species-ladder step.",
            atoms=("Fe I", "Fe II", "Na I"),
            **common,
        ),
        "free_fe_cr": RetrievalCase(
            name="free_fe_cr",
            description="Fe I + Fe II + Cr I interpretation of the Cr I/Ba II step.",
            atoms=("Fe I", "Fe II", "Cr I"),
            **common,
        ),
        "free_fe_ba": RetrievalCase(
            name="free_fe_ba",
            description="Fe I + Fe II + Ba II interpretation of the Cr I/Ba II step.",
            atoms=("Fe I", "Fe II", "Ba II"),
            **common,
        ),
        "free_fe_cr_ba": RetrievalCase(
            name="free_fe_cr_ba",
            description="Fe I + Fe II + Cr I + Ba II combined ladder step.",
            atoms=("Fe I", "Fe II", "Cr I", "Ba II"),
            **common,
        ),
        "free_all_atoms": RetrievalCase(
            name="free_all_atoms",
            description="Every configured atomic opacity; molecules remain disabled.",
            atoms=all_atomic_species(),
            **common,
        ),
        "equilibrium_fe": RetrievalCase(
            name="equilibrium_fe",
            description=(
                "FastChem Fe I/Fe II equilibrium abundances from free [M/H] "
                "with fixed solar C/O; no free logVMR sites."
            ),
            atoms=("Fe I", "Fe II"),
            chemistry_model="fastchem_equilibrium_metallicity_grid",
            pt_profile="isothermal",
            velocity_offset_mode="species",
            velocity_offset_species=("Fe I", "Fe II"),
        ),
    }


def create_parser() -> argparse.ArgumentParser:
    cases = _cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List retrieval cases and exit.")
    parser.add_argument("--case", choices=tuple(cases), default="free_fe")
    parser.add_argument("--epoch", action="append", default=None)
    parser.add_argument(
        "--fastchem-parameter-file",
        type=Path,
        default=REPO_ROOT / "input" / "fastchem" / "parameters.dat",
    )
    add_common_arguments(parser)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    cases = _cases()
    if args.list:
        for case in cases.values():
            print(f"{case.name:20s} {case.description}")
        return 0

    epochs = validate_epochs(args.epoch)
    case = cases[args.case]
    configure_runtime(args, mode="transmission", case_name=case.name)
    preflight = preflight_timeseries(mode="transmission", epochs=epochs, args=args)
    fastchem_file = (
        args.fastchem_parameter_file
        if case.chemistry_model == "fastchem_equilibrium_metallicity_grid"
        else None
    )
    manifest_extra: dict[str, object] = {
        "epochs": epochs,
        "fixed_ephemeris_parameters": ("period", "transit_epoch (image T0)"),
        "rotation_period": "tidally locked to the fixed orbital period",
        "isothermal_temperature_parameter": "terminator/T0",
    }
    if fastchem_file is not None:
        manifest_extra.update(
            {
                "fastchem_parameter_file": str(fastchem_file),
                "fastchem_parameter_file_exists": fastchem_file.exists(),
                "fixed_C_O": "solar",
                "free_chemistry_parameters": ("log_metallicity",),
            }
        )
    manifest = intent_manifest(
        runner="transmission",
        case=case,
        args=args,
        extra=manifest_extra,
    )
    print_plan(manifest, preflight)
    if args.dry_run:
        return 0
    require_ready_timeseries(preflight, args)

    joint_spectra = auxiliary_timeseries_specs(
        mode="transmission",
        epochs=epochs[1:],
        args=args,
        region_name="terminator",
        atoms=case.atoms,
    )
    region = atmosphere_region_spec(
        case,
        name="terminator",
        mode="transmission",
        args=args,
        fastchem_parameter_file=fastchem_file,
    )

    from pipeline.retrieval import run_retrieval

    run_retrieval(
        mode="transmission",
        epoch=epochs[0],
        data_format="timeseries",
        skip_svi=args.skip_svi,
        svi_only=args.svi_only,
        pt_profile=case.pt_profile,
        chemistry_model=case.chemistry_model,
        fastchem_parameter_file=None if fastchem_file is None else str(fastchem_file),
        seed=args.seed,
        joint_spectra=joint_spectra or None,
        atmosphere_regions=[region],
        sigma_scale=args.sigma_scale,
        spectral_stride=args.spectral_stride,
        spectral_offset=args.spectral_offset,
        apply_sysrem_override=False if args.no_sysrem else None,
        shared_prior_modes=shared_prior_modes(args),
        primary_atomic_species=atomic_species_dict(case.atoms),
        primary_molpath_hitemp={},
        primary_molpath_exomol={},
        retrieval_intent=manifest,
        model_param_overrides=model_param_overrides(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
