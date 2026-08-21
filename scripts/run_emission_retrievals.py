#!/usr/bin/env python3
"""Run named or explicitly overridden emission HRS retrieval cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import config_utils
from dataio.collapse_emission_timeseries_to_1d import EMISSION_COLLAPSE_SELECTIONS

from pipeline.retrieval_plan import (
    REPO_ROOT,
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


EMISSION_CHEMISTRY_MODELS = ("constant", "fastchem_hybrid_grid")
PAPER_TEMPERATURE_RANGE_K = (1500.0, 7500.0)
PAPER_FASTCHEM_N_TEMP_MIN = 75
PAPER_LOG_KAPPA_IR_BOUNDS = (-4.0, 0.0)
PAPER_LOG_GAMMA_BOUNDS = (0.0, 2.0)


def _configure_paper_emission_atmosphere() -> None:
    """Apply the KELT-20b paper's P-T prior and its required support."""
    temperature_low, temperature_high = PAPER_TEMPERATURE_RANGE_K
    config_utils.set_runtime_config("T_LOW", temperature_low)
    config_utils.set_runtime_config("T_HIGH", temperature_high)
    config_utils.set_runtime_config("FASTCHEM_T_MIN", temperature_low)
    config_utils.set_runtime_config("FASTCHEM_T_MAX", temperature_high)
    config_utils.set_runtime_config(
        "FASTCHEM_N_TEMP",
        max(int(config.FASTCHEM_N_TEMP), PAPER_FASTCHEM_N_TEMP_MIN),
    )
    config_utils.set_runtime_config(
        "LOG_KAPPA_IR_BOUNDS",
        PAPER_LOG_KAPPA_IR_BOUNDS,
    )
    config_utils.set_runtime_config(
        "LOG_GAMMA_BOUNDS",
        PAPER_LOG_GAMMA_BOUNDS,
    )


def _cases() -> dict[str, RetrievalCase]:
    common = {
        "pt_profile": "guillot",
        "velocity_offset_mode": "region",
    }
    return {
        "free_fe_na_ca": RetrievalCase(
            name="free_fe_na_ca",
            description=(
                "Free constant Fe I, Na I, and Ca I VMRs; Guillot "
                "Tirr/kappa_IR/gamma; shared Kp and one dayside Delta-v."
            ),
            atoms=("Fe I", "Na I", "Ca I"),
            chemistry_model="constant",
            **common,
        ),
        "fe_only_hybrid": RetrievalCase(
            name="fe_only_hybrid",
            description=(
                "Free constant Fe I VMR with FastChem-grid H/e-/H- continuum; "
                "Guillot Tirr/kappa_IR/gamma; shared Kp and one dayside Delta-v."
            ),
            atoms=("Fe I",),
            chemistry_model="fastchem_hybrid_grid",
            **common,
        ),
    }


def _parse_atoms(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("--atoms requires at least one species name.")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("--atoms contains duplicate species names.")
    try:
        atomic_species_dict(names)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return names


def _case_slug(atoms: tuple[str, ...], chemistry_model: str) -> str:
    atom_slug = "_".join(name.lower().replace(" ", "_") for name in atoms)
    chemistry_slug = {
        "constant": "constant",
        "fastchem_hybrid_grid": "hybrid",
    }[chemistry_model]
    return f"custom_{atom_slug}_{chemistry_slug}"


def _resolve_case(
    args: argparse.Namespace,
    cases: dict[str, RetrievalCase],
) -> RetrievalCase:
    base = cases[args.case]
    atoms = base.atoms if args.atoms is None else args.atoms
    chemistry_model = (
        base.chemistry_model
        if args.chemistry_model is None
        else args.chemistry_model
    )
    if atoms == base.atoms and chemistry_model == base.chemistry_model:
        return base
    return RetrievalCase(
        name=_case_slug(atoms, chemistry_model),
        description=(
            f"Custom emission override of {base.name}: atoms={atoms}, "
            f"chemistry_model={chemistry_model}."
        ),
        atoms=atoms,
        chemistry_model=chemistry_model,
        pt_profile=base.pt_profile,
        velocity_offset_mode=base.velocity_offset_mode,
        velocity_offset_species=base.velocity_offset_species,
    )


def create_parser() -> argparse.ArgumentParser:
    cases = _cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List retrieval cases and exit.")
    parser.add_argument("--case", choices=tuple(cases), default="free_fe_na_ca")
    parser.add_argument(
        "--atoms",
        type=_parse_atoms,
        default=None,
        metavar='"Fe I,Ni I,Ca I"',
        help="Comma-separated atomic species override for the selected case.",
    )
    parser.add_argument(
        "--chemistry-model",
        choices=EMISSION_CHEMISTRY_MODELS,
        default=None,
        help="Chemistry override for the selected case.",
    )
    parser.add_argument(
        "--fastchem-parameter-file",
        type=Path,
        default=REPO_ROOT / "input" / "fastchem" / "parameters.dat",
        help="FastChem parameters.dat used by fastchem_hybrid_grid cases.",
    )
    parser.add_argument(
        "--data-format",
        choices=("timeseries", "spectrum"),
        default="timeseries",
        help="Load a prepared exposure time series or a collapsed 1D spectrum.",
    )
    parser.add_argument(
        "--emission-selection",
        choices=EMISSION_COLLAPSE_SELECTIONS,
        default=None,
        help=(
            "Collapsed emission selection; required with --data-format spectrum."
        ),
    )
    parser.add_argument("--epoch", action="append", default=None)
    add_common_arguments(parser)
    return parser


def _preflight_collapsed_emission(
    *,
    epochs: tuple[str, ...],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    arms = tuple(config.FULL_ARM_MEMBERS) if args.arm == "full" else (args.arm,)
    for epoch in epochs:
        for arm in arms:
            data_dir = config_utils.get_collapsed_emission_dir(
                planet=args.planet,
                epoch=epoch,
                arm=arm,
                selection=args.emission_selection,
            )
            required = (
                data_dir / "wavelength_emission.npy",
                data_dir / "spectrum_emission.npy",
                data_dir / "uncertainty_emission.npy",
                data_dir / "emission_collapse_operator.npz",
                data_dir / "collapse_metadata.json",
            )
            missing = [path.name for path in required if not path.is_file()]
            status = None
            if not missing:
                metadata = json.loads(
                    (data_dir / "collapse_metadata.json").read_text(encoding="utf-8")
                )
                status = str(metadata.get("status", "")).strip().lower()
            results.append(
                {
                    "data_dir": str(data_dir),
                    "missing": missing,
                    "status": status,
                }
            )
    return results


def _require_ready_collapsed_emission(
    preflight: list[dict[str, object]],
) -> None:
    failures: list[str] = []
    for row in preflight:
        if row["missing"]:
            failures.append(f"{row['data_dir']}: missing {', '.join(row['missing'])}")
        elif row["status"] != "ready":
            failures.append(
                f"{row['data_dir']}: collapse status is {row['status']!r}, expected 'ready'"
            )
    if failures:
        raise FileNotFoundError(
            "Collapsed emission preflight failed:\n  - " + "\n  - ".join(failures)
        )


def _auxiliary_emission_specs(
    *,
    epochs: tuple[str, ...],
    args: argparse.Namespace,
    case: RetrievalCase,
) -> list[dict[str, object]]:
    if args.data_format == "timeseries":
        return auxiliary_timeseries_specs(
            mode="emission",
            epochs=epochs,
            args=args,
            region_name="dayside",
            atoms=case.atoms,
        )

    atom_config = atomic_species_dict(case.atoms)
    arms = tuple(config.FULL_ARM_MEMBERS) if args.arm == "full" else (args.arm,)
    return [
        {
            "name": f"emission_{arm}_{epoch}",
            "mode": "emission",
            "region_name": "dayside",
            "data_format": "spectrum",
            "data_dir": str(
                config_utils.get_collapsed_emission_dir(
                    planet=args.planet,
                    epoch=epoch,
                    arm=arm,
                    selection=args.emission_selection,
                )
            ),
            "radial_velocity_mode": "none",
            "subtract_weighted_global_mean": True,
            "atomic_species": atom_config,
            "molpath_hitemp": {},
            "molpath_exomol": {},
        }
        for epoch in epochs
        for arm in arms
    ]


def main() -> int:
    args = create_parser().parse_args()
    cases = _cases()
    if args.list:
        for listed_case in cases.values():
            print(f"{listed_case.name:20s} {listed_case.description}")
        return 0

    epochs = validate_epochs(args.epoch)
    if args.data_format == "spectrum" and args.emission_selection is None:
        raise ValueError(
            "--data-format spectrum requires --emission-selection "
            f"({', '.join(EMISSION_COLLAPSE_SELECTIONS)})."
        )
    if args.data_format == "timeseries" and args.emission_selection is not None:
        raise ValueError(
            "--emission-selection is only valid with --data-format spectrum."
        )
    if args.data_format == "spectrum" and args.no_sysrem:
        raise ValueError(
            "--no-sysrem is not valid for collapsed spectra; their saved collapse "
            "operator already records the preprocessing applied to the source cube."
        )
    case = _resolve_case(args, cases)
    configure_runtime(args, mode="emission", case_name=case.name)
    _configure_paper_emission_atmosphere()
    preflight = (
        preflight_timeseries(mode="emission", epochs=epochs, args=args)
        if args.data_format == "timeseries"
        else _preflight_collapsed_emission(epochs=epochs, args=args)
    )
    fastchem_file = (
        args.fastchem_parameter_file
        if case.chemistry_model == "fastchem_hybrid_grid"
        else None
    )
    pressure_top, pressure_bottom = config_utils.get_pressure_bounds_for_mode("emission")
    manifest_extra: dict[str, object] = {
        "epochs": epochs,
        "pressure_range_bar": {"top": pressure_top, "bottom": pressure_bottom},
        "temperature_support_K": {
            "low": config.T_LOW,
            "high": config.T_HIGH,
        },
        "fastchem_temperature_grid_points": config.FASTCHEM_N_TEMP,
        "log_kappa_ir_bounds": config.LOG_KAPPA_IR_BOUNDS,
        "log_gamma_bounds": config.LOG_GAMMA_BOUNDS,
        "selected_case": args.case,
        "data_format": args.data_format,
        "emission_selection": args.emission_selection,
        "case_overrides": {
            "atoms": None if args.atoms is None else args.atoms,
            "chemistry_model": args.chemistry_model,
        },
        "temperature_prior": "target Tirr_mean/Tirr_std when finite",
        "guillot_parameters": ("Tirr", "kappa_ir_cgs", "gamma"),
        "fixed_ephemeris_parameters": ("period", "transit_epoch (image T0)"),
        "rotation_period": "tidally locked to the fixed orbital period",
    }
    if fastchem_file is not None:
        manifest_extra.update(
            {
                "fastchem_parameter_file": str(fastchem_file),
                "fastchem_parameter_file_exists": fastchem_file.exists(),
                "free_hybrid_parameters": ("log_metallicity", "C_O_ratio"),
                "fastchem_continuum_species": ("H-", "e-", "H"),
            }
        )
    manifest = intent_manifest(
        runner="emission",
        case=case,
        args=args,
        extra=manifest_extra,
    )
    print_plan(manifest, preflight)
    if args.dry_run:
        return 0
    if fastchem_file is not None and not fastchem_file.is_file():
        raise FileNotFoundError(
            f"FastChem parameter file does not exist: {fastchem_file}"
        )
    if args.data_format == "timeseries":
        require_ready_timeseries(preflight, args)
    else:
        _require_ready_collapsed_emission(preflight)

    joint_spectra = _auxiliary_emission_specs(
        epochs=epochs[1:],
        args=args,
        case=case,
    )
    region = atmosphere_region_spec(
        case,
        name="dayside",
        mode="emission",
        args=args,
        fastchem_parameter_file=fastchem_file,
    )

    from pipeline.retrieval import run_retrieval

    run_retrieval(
        mode="emission",
        epoch=epochs[0],
        data_format=args.data_format,
        emission_selection=args.emission_selection,
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
        apply_sysrem_override=(
            False if args.data_format == "timeseries" and args.no_sysrem else None
        ),
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
