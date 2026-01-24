"""Load observed spectra and instrumental resolution data."""

import numpy as np
from astropy.io import fits
from exojax.utils.grids import wav2nu


def load_observed_spectrum(
    wav_path: str,
    spectrum_path: str,
    uncertainty_path: str,
    format: str = "npy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load observed spectrum (transmission or emission)."""
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

    inst_nus = wav2nu(wav_obs, "nm")
    return wav_obs, spectrum, uncertainty, inst_nus


def load_pepsi_spectrum(
    data_dict: dict[str, str],
    barycentric_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load PEPSI spectrum with optional barycentric correction."""
    wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
        data_dict["wavelength"],
        data_dict["spectrum"],
        data_dict["uncertainty"],
    )

    if barycentric_correction:
        print("Note: Barycentric correction should be applied during data reduction")

    return wav_obs, spectrum, uncertainty, inst_nus


def load_transmission_emission_pair(
    trans_dict: dict[str, str],
    emis_dict: dict[str, str],
) -> tuple[tuple, tuple]:
    """Load both transmission and emission spectra for joint retrieval."""
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
