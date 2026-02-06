"""Telluric line fitting and correction.

Fits a simple telluric absorption model (single T, P, column density) to
observed spectra using ADAM optimization. Based on ExoJAX tutorial:
https://secondearths.sakura.ne.jp/exojax/tutorials/Fitting_Telluric_Lines.html

This is a lightweight alternative to Molecfit for quick telluric correction.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

import config
from exojax.database.api import MdbHitran
from exojax.opacity.premodit import OpaPremodit
from exojax.utils.grids import wavenumber_grid, wav2nu
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.postproc.specop import SopInstProfile


class TelluricFitter:
    """Fit and remove telluric absorption from observed spectra.
    
    Uses a single-layer atmospheric model with H2O absorption.
    Fits temperature, pressure, column density, amplitude, and
    instrumental resolution using ADAM optimization.
    """

    def __init__(
        self,
        wav_obs_aa: np.ndarray,
        species: str = config.TELLURIC_SPECIES_DEFAULT,
        N_grid: int = config.TELLURIC_N_GRID,
        T_range: tuple[float, float] = config.TELLURIC_T_RANGE,
    ) -> None:
        """Initialize telluric fitter.
        
        Args:
            wav_obs_aa: Observed wavelength grid in Angstroms (can be ascending or descending)
            species: Telluric species to fit (default: H2O)
            N_grid: Number of points in wavenumber grid
            T_range: Temperature range for opacity calculation [K]
        """
        self.wav_obs_aa = np.asarray(wav_obs_aa)
        self.species = species
        
        # Ensure wavenumber grid is ascending (ExoJAX requirement)
        wav_min, wav_max = self.wav_obs_aa.min(), self.wav_obs_aa.max()
        
        margin = config.TELLURIC_MARGIN_CM1  # cm-1 margin for edge effects
        nus_start = wav2nu(wav_max, unit="AA") - margin
        nus_end = wav2nu(wav_min, unit="AA") + margin
        
        # Create wavenumber grid
        self.nus, self.wav, self.res = wavenumber_grid(
            nus_start, nus_end, N_grid, xsmode="lpf", unit="cm-1"
        )
        
        # Convert observed wavelengths to wavenumbers (ascending order)
        self.nus_obs = wav2nu(self.wav_obs_aa, unit="AA")
        
        # Load molecular database (HITRAN for Earth atmosphere)
        print(f"Loading HITRAN {species} for telluric fitting...")
        mdb = MdbHitran(species, nurange=[nus_start, nus_end], isotope=1)
        
        # Build opacity calculator
        snap = mdb.to_snapshot()
        self.opa = OpaPremodit.from_snapshot(
            snap, 
            nu_grid=self.nus, 
            allow_32bit=True, 
            auto_trange=list(T_range),
        )
        
        # Instrumental profile operator
        self.sop_inst = SopInstProfile(self.nus, vrmax=config.TELLURIC_VRMAX)
        
        # Fitted parameters (set after fit())
        self.params = None
        self.T = None
        self.P = None
        self.column_density = None
        self.amplitude = None
        self.beta_inst = None
        
    def _model(self, params: jnp.ndarray, initial_guess: jnp.ndarray) -> jnp.ndarray:
        """Compute telluric transmission model.
        
        Args:
            params: Normalized parameters [T, P, N, a, beta] / initial_guess
            initial_guess: Initial parameter values
            
        Returns:
            Model transmission at observed wavenumbers
        """
        T, P, nl, a, beta = params * initial_guess
        xsv = self.opa.xsvector(T, P)
        trans = a * jnp.exp(-nl * xsv)
        trans_inst = self.sop_inst.ipgauss(trans, beta)
        mu = self.sop_inst.sampling(trans_inst, 0.0, self.nus_obs)
        return mu
    
    def fit(
        self,
        spectrum: np.ndarray,
        T_init: float = 240.0,
        P_init: float = 0.5,
        column_density_init: float = 2.0e22,
        amplitude_init: float = 1.0,
        R_inst: float = 100000.0,
        n_iterations: int = 300,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> dict:
        """Fit telluric model to observed spectrum.
        
        Args:
            spectrum: Observed spectrum (same shape as wav_obs_aa)
            T_init: Initial temperature guess [K]
            P_init: Initial pressure guess [bar]
            column_density_init: Initial column density [molecules/cm^2]
            amplitude_init: Initial amplitude/continuum level
            R_inst: Instrumental resolving power
            n_iterations: Number of ADAM iterations
            learning_rate: ADAM learning rate
            verbose: Show progress bar
            
        Returns:
            Dictionary with fitted parameters and final loss
        """
        spectrum = jnp.asarray(spectrum)
        beta_inst_init = resolution_to_gaussian_std(R_inst)
        
        initial_guess = jnp.array([
            T_init, P_init, column_density_init, amplitude_init, beta_inst_init
        ])
        params = jnp.ones(5)
        
        def objective(params):
            residual = spectrum - self._model(params, initial_guess)
            return jnp.sum(residual**2)
        
        # ADAM optimization
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        value_and_grad = jax.value_and_grad(objective)
        
        @jax.jit
        def step(params, opt_state):
            loss, grads = value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        iterator = tqdm(range(n_iterations), disable=not verbose, desc="Fitting tellurics")
        losses = []
        for _ in iterator:
            params, opt_state, loss = step(params, opt_state)
            losses.append(float(loss))
            if verbose:
                iterator.set_postfix(loss=f"{loss:.2e}")
        
        # Store fitted parameters
        final_params = np.array(params) * np.array(initial_guess)
        self.T = final_params[0]
        self.P = final_params[1]
        self.column_density = final_params[2]
        self.amplitude = final_params[3]
        self.beta_inst = final_params[4]
        self.params = params
        self._initial_guess = initial_guess
        
        return {
            "T": self.T,
            "P": self.P,
            "column_density": self.column_density,
            "amplitude": self.amplitude,
            "beta_inst": self.beta_inst,
            "R_inst_effective": 1.0 / (self.beta_inst * 2.3548),
            "final_loss": losses[-1],
            "losses": np.array(losses),
        }
    
    def get_telluric_model(self) -> np.ndarray:
        """Get fitted telluric transmission model at observed wavelengths.
        
        Returns:
            Telluric transmission spectrum
        """
        if self.params is None:
            raise RuntimeError("Must call fit() first")
        return np.array(self._model(self.params, self._initial_guess))
    
    def correct_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Divide out telluric absorption from spectrum.
        
        Args:
            spectrum: Observed spectrum to correct
            
        Returns:
            Telluric-corrected spectrum
        """
        telluric = self.get_telluric_model()
        # Avoid division by zero in deep absorption
        telluric_safe = np.clip(telluric, 0.01, None)
        return spectrum / telluric_safe * self.amplitude


def fit_and_correct_tellurics(
    wav_aa: np.ndarray,
    spectrum: np.ndarray,
    R_inst: float = 100000.0,
    n_iterations: int = 300,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convenience function to fit and correct tellurics in one call.
    
    Args:
        wav_aa: Wavelength in Angstroms
        spectrum: Observed spectrum
        R_inst: Instrumental resolving power
        n_iterations: Number of ADAM iterations
        verbose: Show progress
        
    Returns:
        Tuple of (corrected_spectrum, telluric_model, fit_results)
    """
    fitter = TelluricFitter(wav_aa)
    results = fitter.fit(
        spectrum, 
        R_inst=R_inst, 
        n_iterations=n_iterations,
        verbose=verbose,
    )
    
    telluric_model = fitter.get_telluric_model()
    corrected = fitter.correct_spectrum(spectrum)
    
    return corrected, telluric_model, results
