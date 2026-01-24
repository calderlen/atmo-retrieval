"""
Grid and Spectral Operator Setup
=================================

Wavenumber grid construction and spectral operators.
"""

from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.postproc.specop import SopRotation, SopInstProfile


def setup_wavenumber_grid(wav_min, wav_max, N, unit="nm"):
    """
    Build high-resolution wavenumber grid for forward modeling.

    Parameters
    ----------
    wav_min : float
        Minimum wavelength
    wav_max : float
        Maximum wavelength
    N : int
        Number of spectral points
    unit : str
        Wavelength unit (default: "nm")

    Returns
    -------
    nu_grid : np.ndarray
        Wavenumber grid [cm^-1]
    wav_grid : np.ndarray
        Wavelength grid
    res_high : float
        High-resolution grid resolving power
    """
    nu_grid, wav_grid, res_high = wavenumber_grid(
        wav_min, wav_max, N=N, unit=unit, xsmode="premodit"
    )
    print(f"Wavenumber grid: R≈{res_high:.0f}")
    return nu_grid, wav_grid, res_high


def setup_spectral_operators(nu_grid, Rinst, vsini_max=100.0, vrmax=300.0):
    """
    Construct spectral operators for rotation and instrumental profile.

    Parameters
    ----------
    nu_grid : np.ndarray
        Wavenumber grid [cm^-1]
    Rinst : float
        Instrumental resolving power
    vsini_max : float
        Maximum rotation velocity [km/s]
    vrmax : float
        Maximum radial velocity for IP [km/s]

    Returns
    -------
    sop_rot : SopRotation
        Rotation operator
    sop_inst : SopInstProfile
        Instrumental profile operator
    beta_inst : float
        Gaussian width parameter for instrument
    """
    beta_inst = resolution_to_gaussian_std(Rinst)
    sop_rot = SopRotation(nu_grid, vsini_max=vsini_max)
    sop_inst = SopInstProfile(nu_grid, vrmax=vrmax)

    return sop_rot, sop_inst, beta_inst
