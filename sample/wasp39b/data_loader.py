"""
Data Loading Module
===================

Load observed transmission spectrum and instrumental resolution data.
"""

import numpy as np
from astropy.io import fits
from exojax.utils.grids import wav2nu


def load_observed_spectrum(wav_path, rp_mean_path, rp_std_path):
    """
    Load JWST NIRSpec G395H transmission spectrum.

    Parameters
    ----------
    wav_path : str
        Path to wavelength array (nm)
    rp_mean_path : str
        Path to mean R_p/R_s spectrum
    rp_std_path : str
        Path to 1-sigma uncertainty

    Returns
    -------
    wav_obs : np.ndarray
        Observed wavelength grid [nm]
    rp_mean : np.ndarray
        Mean R_p/R_s spectrum
    rp_std : np.ndarray
        1-sigma uncertainty
    inst_nus : np.ndarray
        Wavenumber grid [cm^-1]
    """
    wav_obs = np.load(wav_path)
    rp_mean = np.load(rp_mean_path)
    rp_std = np.load(rp_std_path)

    # Convert wavelength to wavenumber
    inst_nus = wav2nu(wav_obs, "nm")

    return wav_obs, rp_mean, rp_std, inst_nus


def load_resolution_curve(fits_path):
    """
    Load NIRSpec/G395H resolution curve from FITS table.

    Parameters
    ----------
    fits_path : str
        Path to FITS file containing resolution curve

    Returns
    -------
    res_curve : np.ndarray
        Resolution curve data [wavelength_um, ..., R]
    """
    with fits.open(fits_path) as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


class ResolutionInterpolator:
    """Interpolate NIRSpec/G395H resolving power."""

    def __init__(self, res_curve_path):
        """
        Parameters
        ----------
        res_curve_path : str
            Path to resolution curve FITS file
        """
        self.res_curve = load_resolution_curve(res_curve_path)

    def __call__(self, wavelength_nm):
        """
        Return resolving power R at given wavelength.

        Parameters
        ----------
        wavelength_nm : float or np.ndarray
            Wavelength in nm

        Returns
        -------
        R : float or np.ndarray
            Resolving power
        """
        return np.interp(
            wavelength_nm / 1000.0,
            self.res_curve[:, 0],
            self.res_curve[:, 2]
        )
