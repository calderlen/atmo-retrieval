"""Memory profiling helpers for atmo-retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import builtins
import os
import re
import subprocess
import sys
from typing import Iterable, TextIO


_GIB = 1024 ** 3
_RED = "\033[31m"
_RESET = "\033[0m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOG_HANDLE: TextIO | None = None
_LOG_PATH: str | None = None
_MACOS_GPU_INFO: str | None = None
_MACOS_GPU_INFO_PRINTED = False


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def set_profile_log(path: str | None, mode: str = "a") -> None:
    """Enable logging profiler output to a file (ANSI stripped)."""
    global _LOG_HANDLE, _LOG_PATH
    if path is None:
        return
    if _LOG_HANDLE is not None and _LOG_PATH == path:
        return
    if _LOG_HANDLE is not None:
        try:
            _LOG_HANDLE.close()
        except Exception:
            pass
        _LOG_HANDLE = None
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    _LOG_HANDLE = open(path, mode, encoding="utf-8")
    _LOG_PATH = path


def _profile_print(*args, **kwargs) -> None:
    if args:
        colored = tuple(_red(str(arg)) for arg in args)
        builtins.print(*colored, **kwargs)
    else:
        builtins.print(*args, **kwargs)

    if _LOG_HANDLE is None:
        return
    if kwargs.get("file") is not None:
        return
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(arg) for arg in args) + end
    text = _ANSI_RE.sub("", text)
    try:
        _LOG_HANDLE.write(text)
        if kwargs.get("flush"):
            _LOG_HANDLE.flush()
    except Exception:
        pass


# Use red output for profiler logging in this module.
print = _profile_print


def _format_bytes(value: float | int) -> str:
    return f"{value / _GIB:.2f} GiB"


@dataclass
class RamInfo:
    rss_bytes: int | None = None
    total_bytes: int | None = None
    available_bytes: int | None = None


@dataclass
class GpuInfo:
    free_bytes: int
    total_bytes: int

    @property
    def used_bytes(self) -> int:
        return max(self.total_bytes - self.free_bytes, 0)


@dataclass
class ProfileResult:
    peak_gpu_used_bytes: float | None = None
    peak_gpu_label: str | None = None
    est_total_bytes: float | None = None
    est_art_bytes: float | None = None
    est_opa_bytes: float | None = None


def _get_ram_info() -> RamInfo:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss
        vm = psutil.virtual_memory()
        return RamInfo(rss_bytes=rss, total_bytes=vm.total, available_bytes=vm.available)
    except Exception:
        pass

    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = int(rss)
        else:
            rss_bytes = int(rss * 1024)
        return RamInfo(rss_bytes=rss_bytes)
    except Exception:
        return RamInfo()


def _get_gpu_info() -> GpuInfo | None:
    try:
        from cuda import cudart

        free_bytes, total_bytes = cudart.cudaMemGetInfo()
        return GpuInfo(int(free_bytes), int(total_bytes))
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return GpuInfo(int(mem.free), int(mem.total))
    except Exception:
        return None


def _macos_gpu_static_info() -> str | None:
    global _MACOS_GPU_INFO
    if _MACOS_GPU_INFO is not None:
        return _MACOS_GPU_INFO
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        _MACOS_GPU_INFO = None
        return None

    chipset = None
    cores = None
    memory = None
    metal = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in ("Chipset Model", "Model", "GPU") and chipset is None:
            chipset = val
        elif key in ("Total Number of Cores", "Cores") and cores is None:
            cores = val
        elif key in ("VRAM (Total)", "Memory", "VRAM") and memory is None:
            memory = val
        elif key == "Metal Support" and metal is None:
            metal = val

    parts: list[str] = []
    if chipset:
        parts.append(chipset)
    if cores:
        parts.append(f"cores {cores}")
    if memory:
        parts.append(f"mem {memory}")
    if metal:
        metal_label = metal if metal.lower().startswith("metal") else f"Metal {metal}"
        parts.append(metal_label)

    _MACOS_GPU_INFO = " | ".join(parts) if parts else None
    return _MACOS_GPU_INFO


def _maybe_print_powermetrics_sample() -> None:
    flag = os.environ.get("ATMO_POWERMETRICS", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    if sys.platform != "darwin":
        return
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("  GPU (powermetrics): set ATMO_POWERMETRICS=1 and run with sudo to sample")
        return
    try:
        result = subprocess.run(
            ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "1000"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception as exc:
        print(f"  GPU (powermetrics): failed ({exc})")
        return

    freq = None
    active = None
    idle = None
    power = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("GPU HW active frequency:"):
            freq = line.split(":", 1)[1].strip()
        elif line.startswith("GPU HW active residency:"):
            active = line.split(":", 1)[1].strip()
        elif line.startswith("GPU idle residency:"):
            idle = line.split(":", 1)[1].strip()
        elif line.startswith("GPU Power:"):
            power = line.split(":", 1)[1].strip()

    parts: list[str] = []
    if freq:
        parts.append(f"freq {freq}")
    if active:
        parts.append(f"active {active.split('(')[0].strip()}")
    if idle:
        parts.append(f"idle {idle.split('(')[0].strip()}")
    if power:
        parts.append(f"power {power}")
    if parts:
        print(f"  GPU (powermetrics): {' | '.join(parts)}")
    else:
        print("  GPU (powermetrics): sample captured but could not parse output")


def _maybe_print_macos_gpu_info() -> None:
    global _MACOS_GPU_INFO_PRINTED
    if _MACOS_GPU_INFO_PRINTED:
        return
    _MACOS_GPU_INFO_PRINTED = True
    info = _macos_gpu_static_info()
    if info:
        print(f"  GPU (macOS): {info}")
    else:
        print("  GPU (macOS): system_profiler did not return display info")
    _maybe_print_powermetrics_sample()


def _print_memory_snapshot(label: str, tracker: dict[str, object] | None = None) -> None:
    ram = _get_ram_info()
    gpu = _get_gpu_info()

    print(f"\n{label}")
    if ram.rss_bytes is not None:
        line = f"  RAM (process RSS): {_format_bytes(ram.rss_bytes)}"
        if ram.total_bytes is not None and ram.available_bytes is not None:
            line += f" | system free: {_format_bytes(ram.available_bytes)} / total: {_format_bytes(ram.total_bytes)}"
        print(line)
    else:
        print("  RAM: unavailable (install psutil for more detail)")

    if gpu is not None:
        print(
            "  GPU: "
            f"used {_format_bytes(gpu.used_bytes)} / total {_format_bytes(gpu.total_bytes)} "
            f"(free {_format_bytes(gpu.free_bytes)})"
        )
        if tracker is not None:
            max_used = float(tracker.get("max_used", 0.0))
            if gpu.used_bytes > max_used:
                tracker["max_used"] = float(gpu.used_bytes)
                tracker["label"] = label
    else:
        if sys.platform == "darwin":
            print("  GPU: unavailable (macOS does not expose CUDA/NVML memory stats)")
            _maybe_print_macos_gpu_info()
        else:
            print("  GPU: unavailable (install cuda-python or pynvml for GPU stats)")


def _estimate_device_memory(
    opa_items: Iterable[tuple[str, object]],
    art: object,
    nfree: int | None,
    gpu_total_bytes: int | None = None,
    warn_threshold: float = 1.0,
    hard_fail: bool = False,
    return_stats: bool = False,
) -> dict[str, float | None] | None:
    try:
        from exojax.utils.memuse import device_memory_use
    except Exception as exc:
        print(f"\nCould not import exojax.utils.memuse.device_memory_use ({exc}).")
        return None

    opa_items = list(opa_items)
    if not opa_items:
        print("\nNo opacity objects found; skipping device memory estimate.")
        return None

    print("\nEstimated device memory (ExoJAX device_memory_use)")
    print("  Note: ExoJAX returns a float; we format it as bytes (GiB) for readability.")

    def safe_use(opa: object, *, art_obj: object | None, nfree_val: int | None) -> float | None:
        try:
            return float(
                device_memory_use(opa, art=art_obj, nfree=nfree_val, print_summary=False)
            )
        except Exception as exc:
            print(f"  Warning: device_memory_use failed ({exc})")
            return None

    first_name, first_opa = opa_items[0]
    base_with_art = safe_use(first_opa, art_obj=art, nfree_val=nfree)
    base_opa_only = safe_use(first_opa, art_obj=None, nfree_val=None)

    opa_only = []
    if base_opa_only is not None:
        opa_only.append((first_name, base_opa_only))
        for name, opa in opa_items[1:]:
            mem = safe_use(opa, art_obj=None, nfree_val=None)
            if mem is not None:
                opa_only.append((name, mem))

        if base_with_art is not None:
            art_mem = max(base_with_art - base_opa_only, 0.0)
            print(f"  art + optimizer (~nfree={nfree}): {_format_bytes(art_mem)}")
            print("  opa-only by species:")
            for name, mem in sorted(opa_only, key=lambda x: x[1], reverse=True):
                print(f"    {name}: {_format_bytes(mem)}")
            opa_sum = sum(mem for _, mem in opa_only)
            total = art_mem + opa_sum
            print(f"  total (art + sum opa-only): {_format_bytes(total)}")
            if gpu_total_bytes:
                ratio = total / float(gpu_total_bytes)
                if ratio > warn_threshold:
                    msg = (
                        f"  WARNING: estimated device memory {ratio:.2f}x GPU total"
                        f" ({_format_bytes(total)} > {_format_bytes(gpu_total_bytes)})"
                    )
                    print(msg)
                    if hard_fail:
                        raise RuntimeError(msg)
            if return_stats:
                return {
                    "art_bytes": float(art_mem),
                    "opa_bytes": float(opa_sum),
                    "total_bytes": float(total),
                }
            return None

    # Fallback: print per-species estimates with art included (do not sum).
    per_species = []
    for name, opa in opa_items:
        mem = safe_use(opa, art_obj=art, nfree_val=nfree)
        if mem is not None:
            per_species.append((name, mem))

    if per_species:
        print("  per-species (art included; do not sum):")
        for name, mem in sorted(per_species, key=lambda x: x[1], reverse=True):
            print(f"    {name}: {_format_bytes(mem)}")
        if gpu_total_bytes:
            max_mem = max(mem for _, mem in per_species)
            ratio = max_mem / float(gpu_total_bytes)
            if ratio > warn_threshold:
                msg = (
                    f"  WARNING: per-species estimate {ratio:.2f}x GPU total"
                    f" ({_format_bytes(max_mem)} > {_format_bytes(gpu_total_bytes)})"
                )
                print(msg)
                if hard_fail:
                    raise RuntimeError(msg)
        if return_stats:
            max_mem = max(mem for _, mem in per_species)
            return {
                "art_bytes": None,
                "opa_bytes": None,
                "total_bytes": float(max_mem),
            }

    return None


def run_memory_profile(
    mode: str,
    nfree: int = 10,
    load_only: bool = False,
    skip_opacities: bool = False,
    hard_fail: bool = True,
    return_stats: bool = False,
    nlayer: int | None = None,
    n_spectral_points: int | None = None,
    wav_min_override: float | None = None,
    wav_max_override: float | None = None,
    log_path: str | None = None,
) -> ProfileResult | None:
    """Profile approximate GPU/RAM usage for the configured retrieval."""
    import config
    from physics.grid_setup import setup_wavenumber_grid
    from databases.opacity import (
        setup_cia_opacities,
        load_molecular_opacities,
        load_atomic_opacities,
    )
    from exojax.rt import ArtTransPure, ArtEmisPure
    import jax

    if log_path is not None:
        set_profile_log(log_path, mode="a")
    print("\nMEMORY PROFILE (approximate)")
    print("-" * 70)
    print(f"Mode: {mode}")
    print(f"NDIV (stitching): {config.NDIV}")
    print(f"PREMODIT_CUTWING: {config.PREMODIT_CUTWING}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    gpu_tracker: dict[str, object] = {"max_used": 0.0, "label": "n/a"}
    gpu_total_bytes: int | None = None
    result = ProfileResult()
    _print_memory_snapshot("Initial memory snapshot", tracker=gpu_tracker)
    gpu_info = _get_gpu_info()
    if gpu_info is not None:
        gpu_total_bytes = gpu_info.total_bytes

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
    print(f"  Grid size: {len(nu_grid):,} points (R≈{res_high:.0f})")
    _print_memory_snapshot("After wavenumber grid", tracker=gpu_tracker)

    print("\n[2/4] Initializing atmospheric RT...")
    if mode == "transmission":
        art = ArtTransPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=nlayer if nlayer is not None else config.NLAYER,
        )
    else:
        art = ArtEmisPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=nlayer if nlayer is not None else config.NLAYER,
        )
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    _print_memory_snapshot("After RT setup", tracker=gpu_tracker)

    print("\n[3/4] Loading opacities...")
    _ = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    opa_mols: dict[str, object] = {}
    opa_atoms: dict[str, object] = {}
    def _after_species(label: str) -> None:
        _print_memory_snapshot(f"After {label}", tracker=gpu_tracker)

    if skip_opacities:
        print("  Skipping opacity load (profile-only).")
    else:
        opa_load = True if load_only else config.OPA_LOAD
        opa_mols, _ = load_molecular_opacities(
            config.MOLPATH_HITEMP,
            config.MOLPATH_EXOMOL,
            nu_grid,
            opa_load,
            config.NDIV,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
            load_only=load_only,
            on_species_loaded=_after_species,
        )
        opa_atoms, _ = load_atomic_opacities(
            config.ATOMIC_SPECIES,
            nu_grid,
            opa_load,
            config.NDIV,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
            load_only=load_only,
            on_species_loaded=_after_species,
        )
        print(f"  Loaded {len(opa_mols)} molecular species, {len(opa_atoms)} atomic species")
    _print_memory_snapshot("After opacities", tracker=gpu_tracker)

    print("\n[4/4] Estimating device memory usage...")
    opa_items = []
    if skip_opacities:
        print("  Skipped (no opacities loaded).")
    else:
        for name, opa in opa_mols.items():
            opa_items.append((f"mol:{name}", opa))
        for name, opa in opa_atoms.items():
            opa_items.append((f"atom:{name}", opa))
        est = _estimate_device_memory(
            opa_items,
            art=art,
            nfree=nfree,
            gpu_total_bytes=gpu_total_bytes,
            warn_threshold=1.0,
            hard_fail=hard_fail,
            return_stats=return_stats,
        )
        if return_stats and est is not None:
            result.est_total_bytes = est.get("total_bytes")
            result.est_art_bytes = est.get("art_bytes")
            result.est_opa_bytes = est.get("opa_bytes")

    max_used = float(gpu_tracker.get("max_used", 0.0))
    if max_used > 0.0:
        label = gpu_tracker.get("label", "n/a")
        print(f"\nPeak GPU used: {_format_bytes(max_used)} (at {label})")
        result.peak_gpu_used_bytes = max_used
        result.peak_gpu_label = str(label)
    print("\nDone.")
    if return_stats:
        return result
    return None


def run_memory_sweep(
    mode: str,
    nfree_values: list[int],
    nlayer_values: list[int],
    nspec_values: list[int],
    wrange_scales: list[float],
    wrange_mode: str = "fixed_res",
    load_only: bool = False,
    skip_opacities: bool = False,
    hard_fail: bool = False,
    log_path: str | None = None,
) -> None:
    """Sweep memory usage by varying one parameter at a time."""
    import config

    base_wav_min, base_wav_max = config.get_wavelength_range()
    base_nlayer = config.NLAYER
    base_nspec = config.N_SPECTRAL_POINTS
    base_half = (base_wav_max - base_wav_min) / 2.0
    base_center = (base_wav_max + base_wav_min) / 2.0

    def _adjust_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            return value
        adjusted = int(round(value / multiple)) * multiple
        return max(adjusted, multiple)

    def _summary(param: str, value: str, result: ProfileResult | None) -> None:
        if result is None:
            print(f"Summary: {param}={value} (no stats)")
            return
        peak = (
            _format_bytes(result.peak_gpu_used_bytes)
            if result.peak_gpu_used_bytes is not None
            else "n/a"
        )
        est = (
            _format_bytes(result.est_total_bytes)
            if result.est_total_bytes is not None
            else "n/a"
        )
        print(f"Summary: {param}={value} | peak GPU: {peak} | est total: {est}")

    if log_path is not None:
        set_profile_log(log_path, mode="a")
    print("\nMEMORY SWEEP")
    print("-" * 70)
    print(f"Mode: {mode}")
    print(f"Base range: {base_wav_min:.2f}-{base_wav_max:.2f} (Angstrom)")
    print(f"Base N_SPECTRAL_POINTS: {base_nspec}")
    print(f"Base NLAYER: {base_nlayer}")
    print(f"NDIV: {config.NDIV}\n")

    # nfree sweep
    if nfree_values:
        print("Sweep: nfree")
        for nfree in nfree_values:
            print(f"\n--- nfree = {nfree} ---")
            res = run_memory_profile(
                mode=mode,
                nfree=nfree,
                load_only=load_only,
                skip_opacities=skip_opacities,
                hard_fail=hard_fail,
                return_stats=True,
                nlayer=base_nlayer,
                n_spectral_points=base_nspec,
                wav_min_override=base_wav_min,
                wav_max_override=base_wav_max,
            )
            _summary("nfree", str(nfree), res)

    # nlayer sweep
    if nlayer_values:
        print("\nSweep: nlayer")
        for nlayer in nlayer_values:
            print(f"\n--- nlayer = {nlayer} ---")
            res = run_memory_profile(
                mode=mode,
                nfree=nfree_values[0] if nfree_values else 10,
                load_only=load_only,
                skip_opacities=skip_opacities,
                hard_fail=hard_fail,
                return_stats=True,
                nlayer=nlayer,
                n_spectral_points=base_nspec,
                wav_min_override=base_wav_min,
                wav_max_override=base_wav_max,
            )
            _summary("nlayer", str(nlayer), res)

    # wavelength range sweep
    if wrange_scales:
        print("\nSweep: wavelength range")
        for scale in wrange_scales:
            half = base_half * float(scale)
            wav_min = base_center - half
            wav_max = base_center + half
            nspec = base_nspec
            if wrange_mode == "fixed_res":
                nspec = int(round(base_nspec * float(scale)))
                if nspec <= 0:
                    nspec = base_nspec
            nspec = _adjust_to_multiple(nspec, config.NDIV)
            print(f"\n--- range scale = {scale} (N={nspec}) ---")
            res = run_memory_profile(
                mode=mode,
                nfree=nfree_values[0] if nfree_values else 10,
                load_only=load_only,
                skip_opacities=skip_opacities,
                hard_fail=hard_fail,
                return_stats=True,
                nlayer=base_nlayer,
                n_spectral_points=nspec,
                wav_min_override=wav_min,
                wav_max_override=wav_max,
            )
            _summary("wrange_scale", str(scale), res)

    # n_spectral_points sweep
    if nspec_values:
        print("\nSweep: n_spectral_points")
        for nspec in nspec_values:
            if nspec % config.NDIV != 0:
                adj = _adjust_to_multiple(nspec, config.NDIV)
                print(f"\n--- n_spectral_points = {nspec} (adjusted to {adj}) ---")
                nspec = adj
            else:
                print(f"\n--- n_spectral_points = {nspec} ---")
            res = run_memory_profile(
                mode=mode,
                nfree=nfree_values[0] if nfree_values else 10,
                load_only=load_only,
                skip_opacities=skip_opacities,
                hard_fail=hard_fail,
                return_stats=True,
                nlayer=base_nlayer,
                n_spectral_points=nspec,
                wav_min_override=base_wav_min,
                wav_max_override=base_wav_max,
            )
            _summary("n_spectral_points", str(nspec), res)

    print("\nSweep complete.")
