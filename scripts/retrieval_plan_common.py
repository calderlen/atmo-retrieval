"""Shared utilities for the image-specified retrieval runner scripts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import config_utils


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    description: str
    atoms: tuple[str, ...]
    chemistry_model: str
    pt_profile: str
    velocity_offset_mode: str
    velocity_offset_species: tuple[str, ...] = ()


def atomic_species_dict(names: Iterable[str]) -> dict[str, dict]:
    names = tuple(names)
    missing = sorted(set(names).difference(config.ATOMIC_SPECIES))
    if missing:
        raise ValueError("Unsupported atomic species: " + ", ".join(missing))
    return {name: dict(config.ATOMIC_SPECIES[name]) for name in names}


def all_atomic_species() -> tuple[str, ...]:
    return tuple(config.ATOMIC_SPECIES)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--ephemeris", default="Duck24")
    parser.add_argument(
        "--arm",
        choices=("red", "blue", "full"),
        default="red",
        help="PEPSI arm selection; full loads red and blue as separate components.",
    )
    parser.add_argument(
        "--profile",
        choices=config_utils.list_runtime_profiles(),
        default=config_utils.get_runtime_profile_name(),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-svi", action="store_true")
    parser.add_argument("--svi-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--svi-steps", type=int, default=None)
    parser.add_argument("--mcmc-warmup", type=int, default=None)
    parser.add_argument("--mcmc-samples", type=int, default=None)
    parser.add_argument("--mcmc-chains", type=int, default=None)
    parser.add_argument("--nlayer", type=int, default=None)
    parser.add_argument("--n-spectral-points", type=int, default=None)
    parser.add_argument("--spectral-stride", type=int, default=1)
    parser.add_argument("--spectral-offset", type=int, default=0)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument(
        "--radius-policy",
        choices=("fixed", "tight"),
        default="fixed",
        help="Fix Rp and Rstar, or sample their catalog normal priors.",
    )
    parser.add_argument(
        "--mass-policy",
        choices=("normal", "fixed", "upper-limit"),
        default="normal",
        help="Prior form for Mp. The image-specified default is an informative normal prior.",
    )
    parser.add_argument("--mass-mean", type=float, default=None)
    parser.add_argument("--mass-sigma", type=float, default=None)
    parser.add_argument(
        "--velocity-bound",
        type=float,
        default=20.0,
        help="Symmetric absolute bound in km/s for region/species Delta-v parameters.",
    )
    parser.add_argument(
        "--no-sysrem",
        action="store_true",
        help="Require a prepared bundle whose saved operator has SYSREM disabled.",
    )
    parser.add_argument("--dry-run", action="store_true")


def configure_runtime(
    args: argparse.Namespace,
    *,
    mode: str,
    case_name: str,
    output_category: str | None = None,
) -> None:
    config_utils.apply_runtime_profile(args.profile)
    config_utils.set_runtime_config("PLANET", args.planet)
    config_utils.set_runtime_config("EPHEMERIS", args.ephemeris)
    config_utils.set_runtime_config("RETRIEVAL_MODE", mode)
    config_utils.set_runtime_config("OBSERVING_MODE", args.arm)
    config_utils.get_params()

    if args.nlayer is not None:
        if args.nlayer < 1:
            raise ValueError("--nlayer must be >= 1.")
        config_utils.set_runtime_config("NLAYER", args.nlayer)
    if args.n_spectral_points is not None:
        if args.n_spectral_points < 1:
            raise ValueError("--n-spectral-points must be >= 1.")
        config_utils.set_runtime_config("N_SPECTRAL_POINTS", args.n_spectral_points)

    if args.quick:
        config_utils.set_runtime_config("SVI_NUM_STEPS", config.QUICK_SVI_STEPS)
        config_utils.set_runtime_config("MCMC_NUM_WARMUP", config.QUICK_MCMC_WARMUP)
        config_utils.set_runtime_config("MCMC_NUM_SAMPLES", config.QUICK_MCMC_SAMPLES)
        config_utils.set_runtime_config("MCMC_NUM_CHAINS", config.QUICK_MCMC_CHAINS)
    for argument, config_name in (
        (args.svi_steps, "SVI_NUM_STEPS"),
        (args.mcmc_warmup, "MCMC_NUM_WARMUP"),
        (args.mcmc_samples, "MCMC_NUM_SAMPLES"),
        (args.mcmc_chains, "MCMC_NUM_CHAINS"),
    ):
        if argument is not None:
            if argument < 1:
                raise ValueError(f"{config_name} must be >= 1.")
            config_utils.set_runtime_config(config_name, argument)

    output_base = args.output
    if output_base is None:
        planet_slug = args.planet.lower().replace("-", "").replace(" ", "")
        output_base = (
            REPO_ROOT
            / "output"
            / "intended_retrievals"
            / planet_slug
            / (output_category or mode)
            / case_name
        )
    output_base.mkdir(parents=True, exist_ok=True)
    config_utils.set_runtime_config("DIR_SAVE", output_base)


def shared_prior_modes(args: argparse.Namespace) -> dict[str, str]:
    radius_mode = "fixed" if args.radius_policy == "fixed" else "normal"
    mass_mode = "upper_limit" if args.mass_policy == "upper-limit" else args.mass_policy
    return {"Mp": mass_mode, "Rp": radius_mode, "Rstar": radius_mode}


def model_param_overrides(args: argparse.Namespace) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if args.mass_mean is not None:
        if not np.isfinite(args.mass_mean) or args.mass_mean <= 0.0:
            raise ValueError("--mass-mean must be finite and positive.")
        overrides["M_p"] = float(args.mass_mean)
    if args.mass_sigma is not None:
        if not np.isfinite(args.mass_sigma) or args.mass_sigma <= 0.0:
            raise ValueError("--mass-sigma must be finite and positive.")
        overrides["M_p_err"] = float(args.mass_sigma)
    return overrides


def velocity_bounds(args: argparse.Namespace) -> tuple[float, float]:
    bound = float(args.velocity_bound)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("--velocity-bound must be finite and positive.")
    return (-bound, bound)


def arms_for(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(config.FULL_ARM_MEMBERS) if args.arm == "full" else (args.arm,)


def timeseries_dirs(
    *,
    mode: str,
    epochs: Iterable[str],
    args: argparse.Namespace,
) -> list[Path]:
    return [
        config_utils.get_timeseries_data_dir(
            planet=args.planet,
            epoch=epoch,
            arm=arm,
            mode=mode,
        )
        for epoch in epochs
        for arm in arms_for(args)
    ]


def preflight_timeseries(
    *,
    mode: str,
    epochs: Iterable[str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for data_dir in timeseries_dirs(mode=mode, epochs=epochs, args=args):
        operator_path = data_dir / "timeseries_operator.npz"
        required = [
            data_dir / "wavelength.npy",
            data_dir / "data.npy",
            data_dir / "sigma.npy",
            data_dir / "phase.npy",
            operator_path,
        ]
        missing = [path.name for path in required if not path.exists()]
        has_sysrem = None
        if operator_path.exists():
            with np.load(operator_path) as operator:
                if "has_sysrem" in operator.files:
                    has_sysrem = bool(np.asarray(operator["has_sysrem"]).item())
        results.append(
            {
                "data_dir": str(data_dir),
                "missing": missing,
                "has_sysrem": has_sysrem,
            }
        )
    return results


def require_ready_timeseries(preflight: list[dict[str, object]], args: argparse.Namespace) -> None:
    failures = []
    for row in preflight:
        if row["missing"]:
            failures.append(f"{row['data_dir']}: missing {', '.join(row['missing'])}")
        elif args.no_sysrem and row["has_sysrem"] is not False:
            failures.append(
                f"{row['data_dir']}: --no-sysrem requested but saved has_sysrem="
                f"{row['has_sysrem']}"
            )
    if failures:
        raise FileNotFoundError(
            "Prepared time-series preflight failed:\n  - " + "\n  - ".join(failures)
        )


def auxiliary_timeseries_specs(
    *,
    mode: str,
    epochs: Iterable[str],
    args: argparse.Namespace,
    region_name: str,
    atoms: tuple[str, ...],
) -> list[dict[str, object]]:
    atom_config = atomic_species_dict(atoms)
    specs: list[dict[str, object]] = []
    for epoch in epochs:
        for arm in arms_for(args):
            specs.append(
                {
                    "name": f"{mode}_{arm}_{epoch}",
                    "mode": mode,
                    "region_name": region_name,
                    "data_format": "timeseries",
                    "data_dir": str(
                        config_utils.get_timeseries_data_dir(
                            planet=args.planet,
                            epoch=epoch,
                            arm=arm,
                            mode=mode,
                        )
                    ),
                    "radial_velocity_mode": "orbital",
                    "atomic_species": atom_config,
                    "molpath_hitemp": {},
                    "molpath_exomol": {},
                }
            )
    return specs


def atmosphere_region_spec(
    case: RetrievalCase,
    *,
    name: str,
    mode: str,
    args: argparse.Namespace,
    fastchem_parameter_file: Path | None = None,
) -> dict[str, object]:
    spec: dict[str, object] = {
        "name": name,
        "mode": mode,
        "sample_prefix": name,
        "pt_profile": case.pt_profile,
        "chemistry_model": case.chemistry_model,
        "velocity_offset_mode": case.velocity_offset_mode,
        "velocity_offset_species": case.velocity_offset_species,
        "velocity_offset_bounds_kms": velocity_bounds(args),
    }
    if fastchem_parameter_file is not None:
        spec["fastchem_parameter_file"] = str(fastchem_parameter_file)
    return spec


def intent_manifest(
    *,
    runner: str,
    case: RetrievalCase | str,
    args: argparse.Namespace,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "runner": runner,
        "case": asdict(case) if isinstance(case, RetrievalCase) else case,
        "planet": args.planet,
        "ephemeris": args.ephemeris,
        "arm": args.arm,
        "shared_prior_modes": shared_prior_modes(args),
        "model_param_overrides": model_param_overrides(args),
        "velocity_bounds_kms": velocity_bounds(args),
        "data_format": "timeseries",
    }
    if extra:
        manifest.update(extra)
    return manifest


def print_plan(manifest: dict[str, object], preflight: list[dict[str, object]]) -> None:
    payload = dict(manifest)
    payload["prepared_data_preflight"] = preflight
    print(json.dumps(payload, indent=2, sort_keys=True))


def validate_epochs(epochs: list[str] | None, flag: str = "--epoch") -> tuple[str, ...]:
    if not epochs:
        raise ValueError(f"At least one {flag} value is required.")
    normalized = tuple(str(epoch).strip() for epoch in epochs if str(epoch).strip())
    if not normalized:
        raise ValueError(f"At least one {flag} value is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate {flag} values are not allowed.")
    return normalized
