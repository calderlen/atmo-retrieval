#!/usr/bin/env python3
"""Run the KELT-20b diagnostic retrieval matrix without Slurm.

This is the local/interactive companion to ``slurm_kelt20b_diagnostic_matrix.sh``.
It runs the same case definitions directly with a chosen Python interpreter, so
it is convenient on a GUI/login machine inside tmux or screen.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MatrixCase:
    name: str
    args: tuple[str, ...]
    jax_enable_x64: bool = False


TINY_RED_FE_ONLY_ARGS: tuple[str, ...] = (
    "--wavelength-range",
    "red",
    "--atoms",
    "Fe I",
    "--no-molecules",
    "--nlayer",
    "10",
    "--n-spectral-points",
    "30000",
)


CASES: tuple[MatrixCase, ...] = (
    MatrixCase("tiny_red_fe_only", TINY_RED_FE_ONLY_ARGS),
    MatrixCase(
        "tiny_red_fe_only_no_svi",
        (*TINY_RED_FE_ONLY_ARGS, "--skip-svi"),
    ),
    MatrixCase("tiny_red_fe_only_sigma1p5", (*TINY_RED_FE_ONLY_ARGS, "--sigma-scale", "1.5")),
    MatrixCase("tiny_red_fe_only_sigma2", (*TINY_RED_FE_ONLY_ARGS, "--sigma-scale", "2")),
    MatrixCase("tiny_red_fe_only_sigma3", (*TINY_RED_FE_ONLY_ARGS, "--sigma-scale", "3")),
    MatrixCase(
        "tiny_red_fe_only_sigma5",
        (*TINY_RED_FE_ONLY_ARGS, "--sigma-scale", "5"),
    ),
    MatrixCase("tiny_red_fe_only_sigma8", (*TINY_RED_FE_ONLY_ARGS, "--sigma-scale", "8")),
    MatrixCase(
        "tiny_red_fe_only_x64",
        TINY_RED_FE_ONLY_ARGS,
        jax_enable_x64=True,
    ),
    MatrixCase("tiny_red_fe_only_no_sysrem", (*TINY_RED_FE_ONLY_ARGS, "--no-sysrem")),
    MatrixCase("tiny_red_fe_only_stride2", (*TINY_RED_FE_ONLY_ARGS, "--spectral-stride", "2")),
    MatrixCase("tiny_red_fe_only_stride4", (*TINY_RED_FE_ONLY_ARGS, "--spectral-stride", "4")),
    MatrixCase("tiny_red_fe_only_stride8", (*TINY_RED_FE_ONLY_ARGS, "--spectral-stride", "8")),
    MatrixCase(
        "red_all_atoms",
        (
            "--wavelength-range",
            "red",
            "--atoms",
            "Fe I,Ni I,Cr I,Na I",
            "--no-molecules",
            "--nlayer",
            "20",
            "--n-spectral-points",
            "50000",
        ),
    ),
    MatrixCase(
        "full_arm_all_atoms",
        (
            "--wavelength-range",
            "full",
            "--atoms",
            "Fe I,Ni I,Cr I,Na I",
            "--no-molecules",
            "--nlayer",
            "20",
            "--n-spectral-points",
            "70000",
        ),
    ),
    MatrixCase(
        "full_arm_no_sysrem",
        (
            "--wavelength-range",
            "full",
            "--atoms",
            "Fe I,Ni I,Cr I,Na I",
            "--no-molecules",
            "--nlayer",
            "20",
            "--n-spectral-points",
            "70000",
            "--no-sysrem",
        ),
    ),
)


def case_lookup() -> dict[str, MatrixCase]:
    lookup: dict[str, MatrixCase] = {}
    for idx, case in enumerate(CASES):
        lookup[str(idx)] = case
        lookup[case.name] = case
    return lookup


def resolve_cases(selectors: list[str], run_all: bool) -> list[MatrixCase]:
    if run_all:
        return list(CASES)
    if not selectors:
        selectors = ["0"]

    lookup = case_lookup()
    resolved: list[MatrixCase] = []
    seen: set[str] = set()
    for selector in selectors:
        try:
            case = lookup[selector]
        except KeyError as exc:
            valid = ", ".join(f"{idx}:{case.name}" for idx, case in enumerate(CASES))
            raise SystemExit(f"Unknown case {selector!r}. Valid cases: {valid}") from exc
        if case.name not in seen:
            resolved.append(case)
            seen.add(case.name)
    return resolved


def common_args(args: argparse.Namespace, case: MatrixCase) -> list[str]:
    return [
        "-m",
        "atmo_retrieval",
        "--profile",
        args.profile,
        "--planet",
        args.planet,
        "--mode",
        "transmission",
        "--epoch",
        args.epoch,
        "--bandpass-tbl",
        args.bandpass_tbl,
        "--data-format",
        "timeseries",
        "--chemistry-model",
        "fastchem_hybrid_grid",
        "--pt-profile",
        "guillot",
        "--fastchem-parameter-file",
        args.fastchem_parameter_file,
        "--load-opacities",
        "--resolution-mode",
        "hr",
        "--mcmc-warmup",
        str(args.mcmc_warmup),
        "--mcmc-samples",
        str(args.mcmc_samples),
        "--mcmc-chains",
        str(args.mcmc_chains),
        "--mcmc-chain-method",
        args.mcmc_chain_method,
        "--svi-steps",
        str(args.svi_steps),
        "--svi-learning-rate",
        str(args.svi_learning_rate),
        "--svi-lr-decay-steps",
        str(args.svi_lr_decay_steps),
        "--svi-lr-decay-rate",
        str(args.svi_lr_decay_rate),
        "--save-mcmc-diagnostics",
        "--diagnostic-label",
        case.name,
    ]


def run_case(case: MatrixCase, args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(args.threads))
    env.setdefault("MKL_NUM_THREADS", str(args.threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    env.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_ENABLE_X64"] = "1" if case.jax_enable_x64 else "0"

    command = [
        args.python_bin,
        *common_args(args, case),
        *case.args,
    ]

    print(f"\n=== {case.name} ===", flush=True)
    print(f"cwd: {args.repo_root}", flush=True)
    print(f"JAX_ENABLE_X64={env['JAX_ENABLE_X64']}", flush=True)
    print("command:", shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=args.repo_root, env=env, check=False)
    return int(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KELT-20b diagnostic retrieval matrix cases without Slurm."
    )
    parser.add_argument("cases", nargs="*", help="Case ids or names. Defaults to 0.")
    parser.add_argument("--all", action="store_true", help="Run all cases sequentially.")
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root to run from.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--threads", type=int, default=8, help="Thread env var value for BLAS/OpenMP libraries.")
    parser.add_argument("--profile", default="hpc", help="Runtime profile passed to atmo_retrieval.")
    parser.add_argument("--planet", default="KELT-20b", help="Planet passed to atmo_retrieval.")
    parser.add_argument("--epoch", default="20190504", help="Epoch passed to atmo_retrieval.")
    parser.add_argument(
        "--bandpass-tbl",
        default="input/phot/transmission/kelt20b/kelt20b_tess_bandpass.tbl",
        help="TESS bandpass table path.",
    )
    parser.add_argument(
        "--fastchem-parameter-file",
        default="input/fastchem/parameters.dat",
        help="FastChem parameter file path.",
    )
    parser.add_argument("--mcmc-warmup", type=int, default=200)
    parser.add_argument("--mcmc-samples", type=int, default=200)
    parser.add_argument("--mcmc-chains", type=int, default=2)
    parser.add_argument("--mcmc-chain-method", default="sequential")
    parser.add_argument("--svi-steps", type=int, default=2000)
    parser.add_argument("--svi-learning-rate", type=float, default=0.001)
    parser.add_argument("--svi-lr-decay-steps", type=int, default=2000)
    parser.add_argument("--svi-lr-decay-rate", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        for idx, case in enumerate(CASES):
            x64 = " x64" if case.jax_enable_x64 else ""
            print(f"{idx}: {case.name}{x64}")
        return 0

    selected_cases = resolve_cases(args.cases, args.all)
    for case in selected_cases:
        returncode = run_case(case, args)
        if returncode != 0:
            print(f"\nCase {case.name} failed with exit code {returncode}.", file=sys.stderr)
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
