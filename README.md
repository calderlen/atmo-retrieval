# KELT-20b Ultra-Hot Jupiter Atmospheric Retrieval

Modular atmospheric retrieval pipeline for ultra-hot Jupiters using ExoJAX and NumPyro.

## Overview

This codebase performs Bayesian atmospheric retrieval on transmission and/or emission spectra of ultra-hot Jupiters, specifically designed for KELT-20b observations with PEPSI/LBT.

## Features

- **Transmission & Emission Retrieval**: Support for both transmission and emission spectroscopy
- **Flexible Temperature Profiles**: Isothermal, gradient, Madhusudhan-Seager, or free temperature nodes
- **Atomic & Molecular Opacities**: TiO, VO, FeH, alkali metals (Na, K), and more
- **Telluric Correction**: Ground-based telluric absorption modeling
- **High Resolution**: Optimized for R~120,000 PEPSI spectra
- **Bayesian Inference**: SVI warm-start + HMC-NUTS sampling

## Module Structure

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
└── main.py                # Main orchestration script
```

## Quick Start

### 1. Configure Parameters

Edit `config.py`:

```python
# Select observing mode
OBSERVING_MODE = "red"  # "blue", "green", "red", or "full"

# Set retrieval mode
RETRIEVAL_MODE = "transmission"  # or "emission", "combined"

# Set database paths
DB_HITEMP = "/path/to/.db_HITEMP/"
DB_EXOMOL = "/path/to/.db_ExoMol/"

# Configure molecular species
MOLPATH_EXOMOL = {
    "TiO": f"{DB_EXOMOL}TiO/48Ti-16O/Toto/",
    "VO": f"{DB_EXOMOL}VO/51V-16O/VOMYT/",
    # Add more as needed
}
```

### 2. Prepare Data

Organize your PEPSI data:

```
data/kelt20b_pepsi/
├── wavelength_transmission.npy
├── spectrum_transmission.npy
└── uncertainty_transmission.npy
```

Or modify `data_loader.py` for your data format.

### 3. Run Retrieval

```bash
python main.py
```

## Retrieval Modes

### Transmission Spectrum

```python
# In config.py
RETRIEVAL_MODE = "transmission"
```

Retrieves:
- Atmospheric composition (molecular/atomic abundances)
- Temperature structure
- Cloud properties
- Planet radius
- Radial velocity

### Emission Spectrum

```python
# In config.py
RETRIEVAL_MODE = "emission"
```

Retrieves:
- Thermal structure (T-P profile)
- Atmospheric composition
- Thermal inversion diagnostics
- Dayside brightness temperature

### Combined (Transmission + Emission)

```python
# In config.py
RETRIEVAL_MODE = "combined"
COMBINE_TRANSMISSION_EMISSION = True
```

Joint retrieval for maximum constraints.

## Temperature Profiles

Edit in `transmission_model.py` or `emission_model.py`:

```python
# Isothermal (1 parameter)
temperature_profile="isothermal"

# Gradient (2 parameters: T_top, T_btm)
temperature_profile="gradient"

# Madhusudhan-Seager (4 parameters: allows inversion)
temperature_profile="madhu_seager"

# Free temperature (5+ nodes)
temperature_profile="free"
```

## Molecular/Atomic Species

### Ultra-Hot Jupiter Species

**Molecules**:
- TiO, VO (optical absorbers, thermal inversion)
- FeH, CrH, CaH (metal hydrides)
- H2O, CO, OH (always present)
- AlO (emerging at high T)

**Atoms**:
- Na I (589 nm doublet)
- K I (770 nm doublet)
- Ca II (H&K lines, IR triplet)
- Fe I, Ti I, V I (forest of lines)

Configure in `config.py`:

```python
MOLPATH_EXOMOL = {
    "TiO": "path/to/TiO",
    "VO": "path/to/VO",
    # ...
}

ATOMIC_SPECIES = {
    "Na": {"element": "Na", "ionization": 0},
    "K": {"element": "K", "ionization": 0},
    # ...
}
```

## Telluric Correction

For ground-based observations, enable telluric modeling:

```python
# In config.py
ENABLE_TELLURICS = True
```

The forward model will include precipitable water vapor (PWV) and telluric scaling as free parameters.

## Output

Results saved to `output_kelt20b/`:

```
├── mcmc_summary.txt           # Parameter estimates
├── posterior_sample.npz       # Posterior samples
├── rp_pred.npy                # Predictive spectrum
├── svi_params.npz             # SVI results
├── transmission_spectrum.png  # Spectrum plot
├── temperature_profile.png    # T-P profile
├── corner_plot_hmc.png        # Parameter correlations
└── corner_plot_overlay.png    # SVI vs HMC
```

## Performance Tips

1. **Start with fewer species**: Test with H2O + CO first, then add TiO/VO
2. **Use saved opacities**: Set `OPA_LOAD = True` after first run
3. **Reduce spectral points**: Lower `N_SPECTRAL_POINTS` for testing
4. **SVI warm-start**: Helps HMC convergence significantly
5. **Multiple chains**: Use 4+ chains to diagnose convergence

## Dependencies

```bash
# Core
jax
numpyro
exojax

# Data & viz
numpy
matplotlib
corner
astropy
arviz (optional, for diagnostics)
```

## Examples

### Example 1: Isothermal Transmission

```python
# config.py
RETRIEVAL_MODE = "transmission"
MOLPATH_EXOMOL = {"TiO": "...", "VO": "..."}
SVI_NUM_STEPS = 1000
MCMC_NUM_SAMPLES = 2000
```

```bash
python main.py
```

### Example 2: Emission with Thermal Inversion

```python
# config.py
RETRIEVAL_MODE = "emission"

# In emission_model.py
temperature_profile="madhu_seager"  # Allows inversion
```

## Troubleshooting

**Q: Opacity loading is slow**
A: First run builds opacities. Set `OPA_SAVE = True`, then `OPA_LOAD = True` for subsequent runs.

**Q: HMC divergences**
A: Increase `MCMC_NUM_WARMUP`, reduce `MCMC_MAX_TREE_DEPTH`, or check prior ranges.

**Q: Telluric features not fitting well**
A: Consider masking heavily contaminated regions or using pre-corrected data.

**Q: Memory issues**
A: Reduce `N_SPECTRAL_POINTS` or `NLAYER`.

## References

- **ExoJAX**: Kawahara et al. (2022), ApJS, 258, 31
- **KELT-20b**: Lund et al. (2017), AJ, 154, 194
- **NumPyro**: Phan et al. (2019), arXiv:1912.11554
- **PEPSI**: Strassmeier et al. (2015), AN, 336, 324

## Citation

If you use this code, please cite:
- ExoJAX: Kawahara et al. (2022)
- NumPyro: Phan et al. (2019)
- Your KELT-20b observations paper

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
