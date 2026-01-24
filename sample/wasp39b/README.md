# WASP-39b Transmission Spectrum Retrieval Modules

Modular implementation of the WASP-39b transmission spectrum retrieval using ExoJAX and NumPyro.

## Structure

The original monolithic script has been separated into the following modules:

### Core Modules

- **config.py** - Configuration, constants, and system parameters
  - Planet/star parameters
  - Database paths
  - Atmospheric RT parameters
  - Inference settings

- **data_loader.py** - Data loading utilities
  - Load observed transmission spectrum
  - Load instrumental resolution curves
  - Wavelength/wavenumber conversions

- **grid_setup.py** - Spectral grid and operators
  - Wavenumber grid construction
  - Spectral rotation operators
  - Instrumental profile operators

- **opacity_setup.py** - Opacity management
  - Load/build CIA opacities
  - Load/build molecular line opacities (HITEMP/ExoMol)
  - preMODIT opacity caching

- **forward_model.py** - Probabilistic forward model
  - NumPyro transmission spectrum model
  - Atmospheric structure (isothermal)
  - Cloud parametrization
  - Radiative transfer

- **inference.py** - Bayesian inference
  - Stochastic Variational Inference (SVI)
  - HMC-NUTS sampling
  - Predictive sampling
  - Result persistence

- **plotting.py** - Visualization
  - SVI loss curves
  - Spectrum overlays
  - Corner plots

- **main.py** - Orchestration script
  - Runs complete retrieval pipeline
  - Coordinates all modules

## Usage

### Run the complete retrieval:

```python
python main.py
```

### Use individual modules:

```python
from wasp39b_modules import config, data_loader, forward_model

# Load data
wav_obs, rp_mean, rp_std, inst_nus = data_loader.load_observed_spectrum(
    config.WAV_OBS_PATH,
    config.RP_MEAN_PATH,
    config.RP_STD_PATH,
)

# Build model
model = forward_model.create_transmission_model(...)
```

## Configuration

Edit `config.py` to customize:
- Database paths (HITEMP, ExoMol, CIA)
- Atmospheric parameters (layers, pressure range, temperature range)
- Inference settings (SVI steps, MCMC samples)
- Output directory

## Dependencies

- JAX
- NumPyro
- ExoJAX
- NumPy
- Matplotlib
- Corner
- Astropy

## Reference

See Section 7.2 of https://arxiv.org/abs/2410.06900 for methodology details.

Shotaro Tada, December 2025
