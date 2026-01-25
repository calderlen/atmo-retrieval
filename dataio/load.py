"""Load observed spectra and instrumental resolution data."""

import numpy as np
from astropy.io import fits
from exojax.utils.grids import wav2nu


def load_observed_spectrum(
    wav_path: str,
    spectrum_path: str,
    uncertainty_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    wav_obs = np.load(wav_path)
    spectrum = np.load(spectrum_path)
    uncertainty = np.load(uncertainty_path)

    # Wavelength in Angstroms, convert to wavenumber
    inst_nus = wav2nu(wav_obs, "AA")
    return wav_obs, spectrum, uncertainty, inst_nus


def load_resolution_curve(fits_path: str) -> np.ndarray:
    """Load instrumental resolution curve from FITS table."""
    with fits.open(fits_path) as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


class ResolutionInterpolator:
    """Interpolate instrumental resolving power vs wavelength."""

    def __init__(
        self,
        res_curve_path: str | None = None,
        constant_R: float | None = None,
    ) -> None:
        if constant_R is not None:
            self.constant_R = constant_R
            self.res_curve = None
        elif res_curve_path is not None:
            self.res_curve = load_resolution_curve(res_curve_path)
            self.constant_R = None
        else:
            raise ValueError("Must provide either res_curve_path or constant_R")

    def __call__(self, wavelength_nm: float | np.ndarray) -> float | np.ndarray:
        """Return resolving power R at given wavelength."""
        if self.constant_R is not None:
            if isinstance(wavelength_nm, np.ndarray):
                return np.full_like(wavelength_nm, self.constant_R)
            else:
                return self.constant_R
        else:
            return np.interp(
                wavelength_nm / 1000.0,
                self.res_curve[:, 0],
                self.res_curve[:, 2]
            )


def mask_telluric_regions(
    wav_obs: np.ndarray,
    spectrum: np.ndarray,
    uncertainty: np.ndarray,
    telluric_mask_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask heavily contaminated telluric regions."""
    if telluric_mask_file is None:
        # Default: mask strong telluric bands
        mask = np.ones(len(wav_obs), dtype=bool)

        # Mask O2 A-band (759-771 nm)
        mask &= (wav_obs < 759) | (wav_obs > 771)
    else:
        mask = np.load(telluric_mask_file)

    return wav_obs[mask], spectrum[mask], uncertainty[mask], mask
