"""
Data Loading Module
===================

Load observed spectra and instrumental resolution data.
Supports both JWST (WASP-39b) and ground-based PEPSI (KELT-20b) formats.
"""

import numpy as np
from astropy.io import fits
from exojax.utils.grids import wav2nu


def load_observed_spectrum(wav_path, spectrum_path, uncertainty_path, format="npy"):
    """
    Load observed spectrum (transmission or emission).

    Parameters
    ----------
    wav_path : str
        Path to wavelength array
    spectrum_path : str
        Path to spectrum (Rp/Rs for transmission, Fp/Fs for emission)
    uncertainty_path : str
        Path to 1-sigma uncertainty
    format : str
        Data format: "npy", "fits", "ascii"

    Returns
    -------
    wav_obs : np.ndarray
        Observed wavelength grid [nm]
    spectrum : np.ndarray
        Observed spectrum
    uncertainty : np.ndarray
        1-sigma uncertainty
    inst_nus : np.ndarray
        Wavenumber grid [cm^-1]
    """
    if format == "npy":
        wav_obs = np.load(wav_path)
        spectrum = np.load(spectrum_path)
        uncertainty = np.load(uncertainty_path)
    elif format == "fits":
        with fits.open(wav_path) as hdul:
            wav_obs = hdul[0].data
        with fits.open(spectrum_path) as hdul:
            spectrum = hdul[0].data
        with fits.open(uncertainty_path) as hdul:
            uncertainty = hdul[0].data
    elif format == "ascii":
        wav_obs = np.loadtxt(wav_path)
        spectrum = np.loadtxt(spectrum_path)
        uncertainty = np.loadtxt(uncertainty_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Convert wavelength to wavenumber
    inst_nus = wav2nu(wav_obs, "nm")

    return wav_obs, spectrum, uncertainty, inst_nus


def load_pepsi_spectrum(data_dict, barycentric_correction=True):
    """
    Load PEPSI spectrum with optional barycentric correction.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys: "wavelength", "spectrum", "uncertainty"
    barycentric_correction : bool
        Whether to apply barycentric velocity correction

    Returns
    -------
    wav_obs : np.ndarray
        Wavelength [nm]
    spectrum : np.ndarray
        Spectrum
    uncertainty : np.ndarray
        Uncertainty
    inst_nus : np.ndarray
        Wavenumber [cm^-1]
    """
    wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
        data_dict["wavelength"],
        data_dict["spectrum"],
        data_dict["uncertainty"],
    )

    if barycentric_correction:
        # Placeholder - actual correction depends on observation metadata
        # Typically corrections are already applied in reduction pipeline
        print("Note: Barycentric correction should be applied during data reduction")

    return wav_obs, spectrum, uncertainty, inst_nus


def load_transmission_emission_pair(trans_dict, emis_dict):
    """
    Load both transmission and emission spectra for joint retrieval.

    Parameters
    ----------
    trans_dict : dict
        Transmission data paths
    emis_dict : dict
        Emission data paths

    Returns
    -------
    trans_data : tuple
        (wav_trans, spectrum_trans, uncertainty_trans, nus_trans)
    emis_data : tuple
        (wav_emis, spectrum_emis, uncertainty_emis, nus_emis)
    """
    trans_data = load_observed_spectrum(
        trans_dict["wavelength"],
        trans_dict["spectrum"],
        trans_dict["uncertainty"],
    )

    emis_data = load_observed_spectrum(
        emis_dict["wavelength"],
        emis_dict["spectrum"],
        emis_dict["uncertainty"],
    )

    return trans_data, emis_data


def load_resolution_curve(fits_path):
    """
    Load instrumental resolution curve from FITS table.

    Parameters
    ----------
    fits_path : str
        Path to FITS file containing resolution curve

    Returns
    -------
    res_curve : np.ndarray
        Resolution curve data [wavelength, R]
    """
    with fits.open(fits_path) as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


class ResolutionInterpolator:
    """Interpolate instrumental resolving power vs wavelength."""

    def __init__(self, res_curve_path=None, constant_R=None):
        """
        Parameters
        ----------
        res_curve_path : str, optional
            Path to resolution curve FITS file
        constant_R : float, optional
            Constant resolving power (for PEPSI)
        """
        if constant_R is not None:
            self.constant_R = constant_R
            self.res_curve = None
        elif res_curve_path is not None:
            self.res_curve = load_resolution_curve(res_curve_path)
            self.constant_R = None
        else:
            raise ValueError("Must provide either res_curve_path or constant_R")

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
        if self.constant_R is not None:
            # PEPSI has essentially constant resolution
            if isinstance(wavelength_nm, np.ndarray):
                return np.full_like(wavelength_nm, self.constant_R)
            else:
                return self.constant_R
        else:
            # Variable resolution (e.g., JWST)
            return np.interp(
                wavelength_nm / 1000.0,
                self.res_curve[:, 0],
                self.res_curve[:, 2]
            )


def mask_telluric_regions(wav_obs, spectrum, uncertainty, telluric_mask_file=None):
    """
    Mask heavily contaminated telluric regions.

    Parameters
    ----------
    wav_obs : np.ndarray
        Wavelength array [nm]
    spectrum : np.ndarray
        Spectrum
    uncertainty : np.ndarray
        Uncertainty
    telluric_mask_file : str, optional
        Path to telluric mask file

    Returns
    -------
    wav_masked : np.ndarray
        Masked wavelength
    spectrum_masked : np.ndarray
        Masked spectrum
    uncertainty_masked : np.ndarray
        Masked uncertainty
    mask : np.ndarray
        Boolean mask (True = keep)
    """
    if telluric_mask_file is None:
        # Default: mask strong telluric bands
        mask = np.ones(len(wav_obs), dtype=bool)

        # Mask O2 A-band (759-771 nm)
        mask &= (wav_obs < 759) | (wav_obs > 771)

        # Mask strong water bands (around 720 nm, 820 nm, etc.)
        # Add more as needed based on your wavelength range

    else:
        # Load custom mask
        mask = np.load(telluric_mask_file)

    return wav_obs[mask], spectrum[mask], uncertainty[mask], mask
