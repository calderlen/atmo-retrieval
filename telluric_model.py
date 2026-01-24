"""
Telluric Absorption Model
==========================

Model telluric absorption for ground-based observations.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from exojax.spec.rtransfer import trans
from exojax.database.api import MdbHitemp


class TelluricModel:
    """
    Model Earth's atmospheric absorption.

    This uses molecular line databases (primarily H2O, O2) to model
    telluric absorption that affects ground-based spectra.
    """

    def __init__(self, nu_grid, species_paths, airmass=1.0):
        """
        Initialize telluric model.

        Parameters
        ----------
        nu_grid : np.ndarray
            Wavenumber grid [cm^-1]
        species_paths : dict
            Dictionary of telluric species paths (e.g., {"H2O": path, "O2": path})
        airmass : float
            Observing airmass
        """
        self.nu_grid = nu_grid
        self.airmass = airmass
        self.species_paths = species_paths
        self.opa_tellurics = {}

        # Standard Earth atmosphere parameters
        self.P_surf = 1.0  # bar (sea level)
        self.T_atm = 280.0  # K (typical)

    def load_telluric_opacities(self, ndiv=6, diffmode=0):
        """
        Load telluric molecular opacities.

        Parameters
        ----------
        ndiv : int
            preMODIT divisions
        diffmode : int
            Diffmode parameter
        """
        from opacity_setup import load_or_build_opacity

        for species, path in self.species_paths.items():
            print(f"Loading telluric {species}...")
            mdb_factory = lambda p: MdbHitemp(p, self.nu_grid, gpu_transfer=False, isotope=1)
            opa, molmass = load_or_build_opacity(
                f"telluric_{species}",
                path,
                mdb_factory,
                opa_load=True,
                nu_grid=self.nu_grid,
                ndiv=ndiv,
                diffmode=diffmode,
                Tlow=250.0,
                Thigh=320.0,
            )
            self.opa_tellurics[species] = opa

    def compute_transmission(self, pwv=5.0, scale_factor=1.0):
        """
        Compute telluric transmission spectrum.

        Parameters
        ----------
        pwv : float
            Precipitable water vapor [mm]
        scale_factor : float
            Scaling factor for telluric strength

        Returns
        -------
        transmission : jnp.ndarray
            Telluric transmission (0-1)
        """
        # Simplified model: use column density scaling
        # More sophisticated: use atmospheric layers

        # Convert PWV to H2O column density (molecules/cm^2)
        # PWV [mm] -> N_H2O [molecules/cm^2]
        # Rough conversion: 1 mm PWV ~ 3.3e22 molecules/cm^2
        N_H2O = pwv * 3.3e22

        # Compute optical depth for each species
        tau_total = jnp.zeros(len(self.nu_grid))

        if "H2O" in self.opa_tellurics:
            # Get cross-section at atmospheric temperature
            xsmatrix_h2o = self.opa_tellurics["H2O"].xsmatrix(
                jnp.array([self.T_atm]), jnp.array([self.P_surf])
            )
            # Column density * cross-section
            tau_h2o = N_H2O * xsmatrix_h2o[0, :] * self.airmass * scale_factor
            tau_total += tau_h2o

        # Add other species (O2, etc.) similarly
        # O2 has relatively constant abundance

        # Transmission
        transmission = jnp.exp(-tau_total)

        return transmission

    def apply_telluric_correction(self, spectrum, pwv=5.0, scale_factor=1.0):
        """
        Apply telluric absorption to model spectrum.

        Parameters
        ----------
        spectrum : jnp.ndarray
            Model spectrum
        pwv : float
            Precipitable water vapor [mm]
        scale_factor : float
            Scaling factor

        Returns
        -------
        corrected_spectrum : jnp.ndarray
            Spectrum with telluric absorption applied
        """
        transmission = self.compute_transmission(pwv, scale_factor)
        return spectrum * transmission


def create_telluric_forward_model(telluric_model, base_model):
    """
    Wrap a base forward model with telluric correction.

    Parameters
    ----------
    telluric_model : TelluricModel
        Telluric absorption model
    base_model : callable
        Base NumPyro model (transmission or emission)

    Returns
    -------
    wrapped_model : callable
        NumPyro model with telluric parameters
    """

    def wrapped_model(obs_mean, obs_std):
        """Forward model with telluric correction."""

        # Sample telluric parameters
        pwv = numpyro.sample("pwv", dist.Uniform(0.5, 20.0))  # [mm]
        telluric_scale = numpyro.sample("telluric_scale", dist.Uniform(0.8, 1.2))

        # Call base model (but don't observe yet)
        # Need to modify base model to return spectrum instead of observing

        # This is a simplified version - actual implementation depends on
        # how base model is structured

        # Placeholder: assume we can get the model spectrum
        # model_spectrum = base_model_spectrum(...)

        # Apply telluric correction
        # corrected_spectrum = telluric_model.apply_telluric_correction(
        #     model_spectrum, pwv, telluric_scale
        # )

        # Observe
        # numpyro.sample("obs", dist.Normal(corrected_spectrum, obs_std), obs=obs_mean)

        pass  # Placeholder

    return wrapped_model


def simple_telluric_correction(spectrum, wav_nm, pwv=5.0, airmass=1.0):
    """
    Simple empirical telluric correction.

    Uses pre-computed telluric models or fits.

    Parameters
    ----------
    spectrum : np.ndarray
        Observed spectrum
    wav_nm : np.ndarray
        Wavelength [nm]
    pwv : float
        Precipitable water vapor [mm]
    airmass : float
        Airmass

    Returns
    -------
    corrected_spectrum : np.ndarray
        Corrected spectrum
    telluric_model : np.ndarray
        Telluric transmission
    """
    # Placeholder for empirical correction
    # In practice, you might use:
    # - Pre-computed telluric models (e.g., from TAPAS)
    # - Telluric standard star observations
    # - Molecfit or TelFit corrections

    # For now, return uncorrected
    telluric_model = jnp.ones_like(spectrum)

    return spectrum / telluric_model, telluric_model
