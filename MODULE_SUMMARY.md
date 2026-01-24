# Module Summary for KELT-20b Retrieval

## Overview

This document summarizes the modular structure created for KELT-20b (ultra-hot Jupiter) atmospheric retrieval with PEPSI/LBT observations.

## Module Status

### ✅ Reusable Modules (from WASP-39b)

These modules work for any retrieval without modification:

1. **grid_setup.py** - Wavenumber grid and spectral operators
2. **opacity_setup.py** - Generic opacity loading (CIA, molecules, atoms)
3. **inference.py** - SVI and HMC-NUTS inference

### ⚠️ Adapted Modules

These modules were adapted from WASP-39b to support KELT-20b:

4. **config.py** - MAJOR CHANGES
   - KELT-20b system parameters
   - PEPSI wavelength range (optical: 383-907 nm)
   - PEPSI resolution (R~120,000-250,000)
   - Temperature range: 1500-4500K
   - Different molecules: TiO, VO, FeH, etc.
   - Atomic species: Na, K, Ca, Fe, Ti, V
   - Telluric parameters

5. **data_loader.py** - MINOR CHANGES
   - Added PEPSI data loading
   - Constant resolution support
   - Telluric masking capability
   - Barycentric correction notes

6. **plotting.py** - EXTENDED
   - Separated transmission/emission plots
   - Temperature-pressure profile plots
   - Optical wavelength units

### 🆕 New Modules for Ultra-Hot Jupiters

7. **thermal_structure.py** - NEW
   - Isothermal profiles
   - Gradient profiles
   - Guillot (2010) profiles
   - Madhusudhan-Seager (2009) profiles (allows thermal inversion)
   - Free temperature retrieval
   - NumPyro sampling wrappers

8. **transmission_model.py** - NEW
   - Transmission forward model with flexible T-P profiles
   - Support for atomic + molecular opacities
   - Cloud modeling
   - Rotation and instrumental broadening

9. **emission_model.py** - NEW
   - Emission forward model (dayside/nightside)
   - Thermal emission calculation
   - Planet-to-star flux ratio (Fp/Fs)
   - Thermal inversion diagnostics

10. **telluric_model.py** - NEW
    - Earth atmospheric absorption modeling
    - H2O, O2 telluric lines
    - PWV (precipitable water vapor) parameter
    - Airmass scaling
    - Integration with forward models

11. **main.py** - NEW
    - Orchestration for transmission/emission/combined retrieval
    - Mode selection
    - End-to-end pipeline

## Key Differences: WASP-39b vs KELT-20b

| Aspect | WASP-39b (NIRSpec) | KELT-20b (PEPSI) |
|--------|-------------------|------------------|
| **Target** | Hot Saturn | Ultra-Hot Jupiter |
| **Temperature** | 500-2000 K | 1500-4500 K |
| **Wavelength** | 2.8-5.1 μm (IR) | 383-907 nm (optical) |
| **Resolution** | R~2700 | R~120,000-250,000 |
| **Instrument** | JWST (space) | LBT (ground) |
| **Molecules** | H2O, CO, CO2, SO2 | TiO, VO, FeH, H2O, CO |
| **Atoms** | None | Na, K, Ca, Fe, Ti, V |
| **T-P Profile** | Isothermal | Gradient/Inversion |
| **Tellurics** | None | Yes (H2O, O2) |
| **Retrieval** | Transmission only | Transmission + Emission |

## Directory Structure

```
uhj-atmo-retrieval/
├── config.py                  # KELT-20b configuration
├── data_loader.py             # Data loading utilities
├── grid_setup.py              # Spectral grid setup
├── opacity_setup.py           # Opacity management
├── thermal_structure.py       # T-P profiles (NEW)
├── transmission_model.py      # Transmission model (NEW)
├── emission_model.py          # Emission model (NEW)
├── telluric_model.py          # Telluric correction (NEW)
├── inference.py               # Bayesian inference
├── plotting.py                # Visualization
├── main.py                    # Main script
├── README.md                  # Documentation
├── MODULE_SUMMARY.md          # This file
│
├── sample-other/              # Example scripts
│   ├── WASP39b_transmission_JWST-NIRSpec.py
│   └── wasp39b_modules/       # Modularized WASP-39b
│
├── data/                      # Data directory
│   └── kelt20b_pepsi/         # KELT-20b observations
│
├── output_kelt20b/            # Results output
│
└── environment.*.yml          # Conda environments
```

## Molecular/Atomic Species for Ultra-Hot Jupiters

### Molecules (ExoMol)
- **TiO** (Titanium Oxide) - Strong optical absorber, thermal inversion
- **VO** (Vanadium Oxide) - Strong optical absorber, thermal inversion
- **FeH** (Iron Hydride) - Wing bands in optical
- **CrH** (Chromium Hydride) - Optical features
- **CaH** (Calcium Hydride) - Optical features
- **AlO** (Aluminum Oxide) - Emerging at very high T
- **H2O** (Water) - Always present
- **CO** (Carbon Monoxide) - Always present
- **OH** (Hydroxyl) - High temperature chemistry

### Atoms (Kurucz/VALD)
- **Na I** - 589 nm doublet (D-lines)
- **K I** - 770 nm doublet
- **Ca II** - H&K lines (393, 397 nm), IR triplet
- **Fe I** - Forest of lines throughout optical
- **Ti I** - Multiple lines
- **V I** - Multiple lines
- **Mg I** - Triplet at 518 nm
- **Li I** - 670 nm doublet

## Temperature Profile Options

### 1. Isothermal (simplest)
- 1 parameter: T0
- Use for initial testing or low S/N data

### 2. Gradient
- 2 parameters: T_top, T_btm
- Linear in log(P) space
- Can represent basic vertical structure

### 3. Madhusudhan-Seager (recommended for UHJs)
- 4 parameters: T_deep, T_high, P_trans, delta_P
- Allows thermal inversions
- Smooth transition

### 4. Free Temperature
- 5+ parameters: T at different pressure levels
- Maximum flexibility
- Risk of overfitting

### 5. Guillot (physical)
- 4 parameters: T_irr, T_int, kappa_ir, gamma
- Physically motivated
- Based on radiative equilibrium

## Telluric Correction

For ground-based PEPSI observations, telluric absorption must be addressed:

### Option 1: Pre-corrected Data (recommended)
- Use TelFit, Molecfit, or TAPAS to correct data beforehand
- Set `ENABLE_TELLURICS = False`

### Option 2: Include in Forward Model
- Model telluric absorption within retrieval
- Set `ENABLE_TELLURICS = True`
- Adds PWV and scaling parameters
- More computationally expensive

### Option 3: Mask Contaminated Regions
- Remove heavily affected wavelengths
- Use `mask_telluric_regions()` in data_loader.py

## Workflow

1. **Configure** - Edit `config.py` for your setup
2. **Prepare Data** - Place PEPSI data in `data/kelt20b_pepsi/`
3. **Test Opacities** - Run with few species first
4. **Run SVI** - Get initial parameter estimates
5. **Run MCMC** - Full Bayesian inference
6. **Diagnostics** - Check convergence, plots
7. **Iterate** - Adjust priors, add/remove species

## Next Steps

1. **Implement Atomic Opacities**
   - Complete `load_atomic_opacities()` in opacity_setup.py
   - Interface with Kurucz/VALD line lists

2. **Emission Model**
   - Complete emission retrieval workflow in main.py
   - Test with synthetic data

3. **Combined Retrieval**
   - Joint transmission + emission
   - Shared atmospheric parameters

4. **Telluric Refinement**
   - Test telluric forward modeling
   - Compare with pre-corrected data

5. **Validation**
   - Test with known benchmarks
   - Compare with published KELT-20b results

## References for Implementation

- **Temperature Profiles**: Madhusudhan & Seager (2009), Guillot (2010)
- **ExoJAX**: Kawahara et al. (2022), https://exojax.readthedocs.io
- **NumPyro**: https://num.pyro.ai
- **PEPSI**: Strassmeier et al. (2015)
- **KELT-20b**: Lund et al. (2017), Casasayas-Barris et al. (2019)

## Notes

- The Pylance warnings about ExoJAX imports are expected - they'll resolve once ExoJAX is installed
- Atomic line loading is currently a placeholder - needs VALD/Kurucz interface
- Telluric model is simplified - may need refinement for your data
- Start with isothermal + few molecules for initial tests
