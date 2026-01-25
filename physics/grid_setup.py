"""Wavenumber grid construction and spectral operators."""

import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.postproc.specop import SopRotation, SopInstProfile


def setup_wavenumber_grid(
    wav_min: float,
    wav_max: float,
    N: int,
    unit: str = "nm",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build high-resolution wavenumber grid for forward modeling."""
    nu_grid, wav_grid, res_high = wavenumber_grid(
        wav_min, wav_max, N=N, unit=unit, xsmode="premodit"
    )
    print(f"Wavenumber grid: R≈{res_high:.0f}")
    return nu_grid, wav_grid, res_high


def setup_spectral_operators(
    nu_grid: np.ndarray,
    Rinst: float,
    vsini_max: float = 100.0,
    vrmax: float = 300.0,
) -> tuple[SopRotation, SopInstProfile, float]:
    """Construct spectral operators for rotation and instrumental profile."""
    beta_inst = resolution_to_gaussian_std(Rinst)
    sop_rot = SopRotation(nu_grid, vsini_max=vsini_max)
    sop_inst = SopInstProfile(nu_grid, vrmax=vrmax)

    return sop_rot, sop_inst, beta_inst
