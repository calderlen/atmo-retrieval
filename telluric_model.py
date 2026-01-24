"""Model telluric absorption for ground-based observations."""

from typing import Callable
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from exojax.database.api import MdbHitemp


class TelluricModel:
    """Model Earth's atmospheric absorption."""

    def __init__(
        self,
        nu_grid: np.ndarray,
        species_paths: dict[str, str],
        airmass: float = 1.0,
    ) -> None:
        self.nu_grid = nu_grid
        self.airmass = airmass
        self.species_paths = species_paths
        self.opa_tellurics = {}

        # Standard Earth atmosphere parameters
        self.P_surf = 1.0  # bar (sea level)
        self.T_atm = 280.0  # K (typical)

    def load_telluric_opacities(self, ndiv: int = 6, diffmode: int = 0) -> None:
        """Load telluric molecular opacities."""
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

    def compute_transmission(self, pwv: float = 5.0, scale_factor: float = 1.0) -> jnp.ndarray:
        """Compute telluric transmission spectrum."""
        # Convert PWV to H2O column density (molecules/cm^2)
        N_H2O = pwv * 3.3e22

        tau_total = jnp.zeros(len(self.nu_grid))

        if "H2O" in self.opa_tellurics:
            xsmatrix_h2o = self.opa_tellurics["H2O"].xsmatrix(
                jnp.array([self.T_atm]), jnp.array([self.P_surf])
            )
            tau_h2o = N_H2O * xsmatrix_h2o[0, :] * self.airmass * scale_factor
            tau_total += tau_h2o

        transmission = jnp.exp(-tau_total)
        return transmission

    def apply_telluric_correction(
        self,
        spectrum: jnp.ndarray,
        pwv: float = 5.0,
        scale_factor: float = 1.0,
    ) -> jnp.ndarray:
        """Apply telluric absorption to model spectrum."""
        transmission = self.compute_transmission(pwv, scale_factor)
        return spectrum * transmission


def create_telluric_forward_model(
    telluric_model: TelluricModel,
    base_model: Callable,
) -> Callable:
    """Wrap a base forward model with telluric correction."""

    def wrapped_model(obs_mean: jnp.ndarray, obs_std: jnp.ndarray) -> None:
        """Forward model with telluric correction."""
        pwv = numpyro.sample("pwv", dist.Uniform(0.5, 20.0))  # [mm]
        telluric_scale = numpyro.sample("telluric_scale", dist.Uniform(0.8, 1.2))
        # Placeholder - actual implementation depends on base model structure
        pass

    return wrapped_model


def simple_telluric_correction(
    spectrum: np.ndarray,
    wav_nm: np.ndarray,
    pwv: float = 5.0,
    airmass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple empirical telluric correction (placeholder)."""
    telluric_model = jnp.ones_like(spectrum)
    return spectrum / telluric_model, telluric_model
