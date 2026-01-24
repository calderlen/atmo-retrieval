This codebase performs Bayesian atmospheric retrieval on transmission and/or emission spectra of ultra-hot Jupiters, specifically designed for KELT-20b observations with PEPSI/LBT.

## Quick Start

```bash
# Run transmission retrieval
python __main__.py --mode transmission

# Quick test (100 samples)
python __main__.py --quick

# See all options
python __main__.py --help
```

## Modules

```
├── config.py              # Configuration and system parameters
├── data_loader.py         # Data loading (PEPSI, JWST formats)
├── grid_setup.py          # Wavenumber grid and spectral operators
├── opacity_setup.py       # CIA, molecular, atomic opacities
├── thermal_structure.py   # Temperature-pressure profiles
├── transmission_model.py  # Transmission forward model
├── emission_model.py      # Emission forward model
├── telluric_model.py      # Telluric absorption (ground-based)
├── inference.py           # SVI and HMC-NUTS
├── plotting.py            # Visualization
├── retrieval.py           # Retrieval pipeline functions
└── __main__.py            # CLI entry point
```

## Dependencies

```bash
jax
numpyro
exojax
numpy
matplotlib
corner
astropy
arviz (optional, for diagnostics)
```
