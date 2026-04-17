"""Programmatic memory profiling utilities for retrieval setup.

This module is intentionally not wired into the CLI. Import and call
`run_memory_profile`/`run_memory_sweep` directly when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

import os

import config
from opacities import load_atomic_opacities, load_molecular_opacities, setup_cia_opacities
from exojax.rt import ArtEmisPure, ArtTransPure
from physics.grid_setup import setup_wavenumber_grid

try:
    import jax
except Exception:  # pragma: no cover
    jax = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    from cuda import cudart
except Exception:  # pragma: no cover
    cudart = None

try:
    from exojax.utils.memuse import device_memory_use
except Exception:  # pragma: no cover
    device_memory_use = None


_GIB = 1024**3


@dataclass
class ProfileResult:
    peak_gpu_used_bytes: float | None = None
    peak_gpu_label: str | None = None
    est_total_bytes: float | None = None
    est_art_bytes: float | None = None
    est_opa_bytes: float | None = None


def _fmt_bytes(value: float | int) -> str:
    return f"{float(value) / _GIB:.2f} GiB"


def _ram_rss_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _cuda_mem_info() -> tuple[int, int] | None:
    if cudart is None:
        return None
    free_bytes, total_bytes = cudart.cudaMemGetInfo()
    return int(free_bytes), int(total_bytes)


def _print_snapshot(label: str, tracker: dict[str, object]) -> None:
    print(f"\n{label}")
    rss = _ram_rss_bytes()
    if rss is None:
        print("  RAM: unavailable (install psutil for process RSS)")
    else:
        print(f"  RAM RSS: {_fmt_bytes(rss)}")

    gpu = _cuda_mem_info()
    if gpu is None:
        print("  GPU: unavailable (install cuda-python / run on CUDA host)")
        return

    free_b, total_b = gpu
    used_b = max(total_b - free_b, 0)
    print(
        "  GPU: "
        f"used {_fmt_bytes(used_b)} / total {_fmt_bytes(total_b)} "
        f"(free {_fmt_bytes(free_b)})"
    )
    max_used = float(cast(float, tracker.get("max_used", 0.0)))
    if used_b > max_used:
        tracker["max_used"] = float(used_b)
        tracker["label"] = label


def _estimate_device_mem(
    opa_items: Iterable[tuple[str, object]],
    art: object,
    nfree: int,
) -> tuple[float | None, float | None, float | None]:
    if device_memory_use is None:
        print("\nExoJAX device_memory_use unavailable; skipping estimate.")
        return None, None, None

    opa_items = list(opa_items)
    if not opa_items:
        return None, None, None

    first_name, first_opa = opa_items[0]
    _ = first_name

    with_art = float(device_memory_use(first_opa, art=art, nfree=nfree, print_summary=False))
    opa_only_first = float(device_memory_use(first_opa, art=None, nfree=None, print_summary=False))
    art_mem = max(with_art - opa_only_first, 0.0)

    opa_only_sum = opa_only_first
    for _, opa in opa_items[1:]:
        opa_only_sum += float(device_memory_use(opa, art=None, nfree=None, print_summary=False))

    total = art_mem + opa_only_sum
    return art_mem, opa_only_sum, total


def run_memory_profile(
    mode: str,
    nfree: int = 10,
    load_only: bool = False,
    skip_opacities: bool = False,
    nlayer: int | None = None,
    n_spectral_points: int | None = None,
    wav_min_override: float | None = None,
    wav_max_override: float | None = None,
) -> ProfileResult:
    """Profile approximate RAM/GPU usage during retrieval setup steps."""
    print("\nMEMORY PROFILE")
    print("-" * 70)
    print(f"Mode: {mode}")
    print(f"PREMODIT_CUTWING: {config.PREMODIT_CUTWING}")
    if jax is not None:
        print(f"JAX backend: {jax.default_backend()}")
        print(f"JAX devices: {jax.devices()}")

    tracker: dict[str, object] = {"max_used": 0.0}
    result = ProfileResult()
    _print_snapshot("Initial", tracker)

    print("\n[1/4] Building wavenumber grid...")
    wav_min, wav_max = config.get_wavelength_range()
    if wav_min_override is not None and wav_max_override is not None:
        wav_min, wav_max = wav_min_override, wav_max_override
    npoints = n_spectral_points if n_spectral_points is not None else config.N_SPECTRAL_POINTS
    nu_grid, _wav_grid, res_high = setup_wavenumber_grid(
        wav_min - config.WAV_MIN_OFFSET,
        wav_max + config.WAV_MAX_OFFSET,
        npoints,
        unit="AA",
    )
    print(f"  Grid size: {len(nu_grid):,} points (R~{res_high:.0f})")
    _print_snapshot("After wavenumber grid", tracker)

    print("\n[2/4] Initializing atmospheric RT...")
    nl = nlayer if nlayer is not None else config.NLAYER
    if mode == "transmission":
        art = ArtTransPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=nl,
        )
    elif mode == "emission":
        art = ArtEmisPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=nl,
        )
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    _print_snapshot("After RT setup", tracker)

    print("\n[3/4] Loading opacities...")
    cia_paths = {}
    for k, v in config.CIA_PATHS.items():
        cia_paths[k] = str(v)
    _ = setup_cia_opacities(cia_paths, nu_grid)
    opa_mols: dict[str, object] = {}
    opa_atoms: dict[str, object] = {}
    if skip_opacities:
        print("  Skipping opacities")
    else:
        opa_load = True if load_only else config.OPA_LOAD
        molpath_hitemp = {}
        for k, v in config.MOLPATH_HITEMP.items():
            molpath_hitemp[k] = str(v)
        molpath_exomol = {}
        for k, v in config.MOLPATH_EXOMOL.items():
            molpath_exomol[k] = str(v)
        opa_mols, _ = load_molecular_opacities(
            molpath_hitemp,
            molpath_exomol,
            nu_grid,
            opa_load,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
            load_only=load_only,
        )
        opa_atoms, _ = load_atomic_opacities(
            config.ATOMIC_SPECIES,
            nu_grid,
            opa_load,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
            load_only=load_only,
        )
        print(f"  Loaded {len(opa_mols)} molecular, {len(opa_atoms)} atomic")
    _print_snapshot("After opacities", tracker)

    print("\n[4/4] Estimating device memory...")
    if not skip_opacities:
        opa_items = []
        for k, v in opa_mols.items():
            opa_items.append((f"mol:{k}", v))
        for k, v in opa_atoms.items():
            opa_items.append((f"atom:{k}", v))
        art_mem, opa_mem, total_mem = _estimate_device_mem(opa_items, art=art, nfree=nfree)
        result.est_art_bytes = art_mem
        result.est_opa_bytes = opa_mem
        result.est_total_bytes = total_mem
        if total_mem is not None:
            print(f"  Estimated total device memory: {_fmt_bytes(total_mem)}")
    else:
        print("  Skipped (no opacities loaded)")

    peak = float(cast(float, tracker.get("max_used", 0.0)))
    if peak > 0.0:
        result.peak_gpu_used_bytes = float(peak)
        result.peak_gpu_label = str(tracker.get("label", "n/a"))
        print(f"\nPeak GPU used: {_fmt_bytes(peak)} (at {result.peak_gpu_label})")
    print("Done.")
    return result


def run_memory_sweep(
    mode: str,
    nfree_values: list[int],
    nlayer_values: list[int],
    nspec_values: list[int],
) -> None:
    """Run a lightweight memory sweep over selected dimensions."""
    base_wav_min, base_wav_max = config.get_wavelength_range()
    print("\nMEMORY SWEEP")
    print("-" * 70)
    print(f"Mode: {mode}")
    print(f"Base range: {base_wav_min:.2f}-{base_wav_max:.2f} Angstrom")

    for nfree in nfree_values:
        print(f"\n--- nfree = {nfree} ---")
        run_memory_profile(mode=mode, nfree=nfree)

    for nlayer in nlayer_values:
        print(f"\n--- nlayer = {nlayer} ---")
        run_memory_profile(mode=mode, nlayer=nlayer)

    for nspec in nspec_values:
        print(f"\n--- N_SPECTRAL_POINTS = {nspec} ---")
        run_memory_profile(mode=mode, n_spectral_points=nspec)
