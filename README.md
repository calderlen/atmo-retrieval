This codebase performs Bayesian atmospheric retrieval on transmission and/or emission spectra of ultra-hot Jupiters, specifically designed for KELT-20b observations with PEPSI/LBT.

## Architecture

```mermaid
flowchart TB
    subgraph CLI["Entry Point"]
        main["__main__.py<br/><i>CLI parser</i>"]
    end

    subgraph Config["Configuration (config/)"]
        planets["planets.py<br/><i>System parameters</i>"]
        instrument["instrument.py<br/><i>PEPSI settings</i>"]
        model_cfg["model.py<br/><i>RT parameters</i>"]
        paths["paths.py<br/><i>Database paths</i>"]
        inf_cfg["inference.py<br/><i>Sampling params</i>"]
    end

    subgraph DataPrep["Data Preparation"]
        load["load.py<br/><i>Load spectra</i>"]
        preprocess["preprocess.py<br/><i>PEPSI reduction</i>"]
        grid["grid_setup.py<br/><i>Wavenumber grid</i>"]
    end

    subgraph Opacity["Opacity Setup"]
        opa_setup["opacity_setup.py"]
        cia["CIA<br/><i>H₂-H₂, H₂-He</i>"]
        hitemp["HITEMP<br/><i>H₂O, CO, OH</i>"]
        exomol["ExoMol<br/><i>TiO, VO, FeH...</i>"]
    end

    subgraph Forward["Forward Model"]
        model["model.py<br/><i>NumPyro model</i>"]
        pt["pt.py<br/><i>T-P profiles</i>"]
        telluric["telluric_model.py<br/><i>Ground-based correction</i>"]
    end

    subgraph Inference["Bayesian Inference"]
        svi["SVI<br/><i>Variational warm-up</i>"]
        mcmc["HMC-NUTS<br/><i>Posterior sampling</i>"]
        pred["Predictive<br/><i>Model spectra</i>"]
    end

    subgraph Output["Results"]
        posterior["Posterior Samples<br/><i>.npz files</i>"]
        plots["plot.py<br/><i>Diagnostics</i>"]
    end

    %% Main flow
    main --> retrieval["retrieval.py<br/><i>Pipeline orchestrator</i>"]
    Config --> retrieval

    retrieval --> load
    preprocess --> load
    retrieval --> grid
    retrieval --> opa_setup

    opa_setup --> cia
    opa_setup --> hitemp
    opa_setup --> exomol

    load --> model
    grid --> model
    opa_setup --> model
    pt --> model
    telluric -.-> model

    model --> svi
    svi -->|"init strategy"| mcmc
    mcmc --> pred

    mcmc --> posterior
    pred --> plots
    posterior --> plots

    %% Styling
    classDef entry fill:#e1f5fe,stroke:#01579b
    classDef config fill:#fff3e0,stroke:#e65100
    classDef data fill:#e8f5e9,stroke:#2e7d32
    classDef opacity fill:#fce4ec,stroke:#c2185b
    classDef forward fill:#f3e5f5,stroke:#7b1fa2
    classDef inference fill:#e8eaf6,stroke:#3f51b5
    classDef output fill:#efebe9,stroke:#5d4037

    class main,retrieval entry
    class planets,instrument,model_cfg,paths,inf_cfg config
    class load,preprocess,grid data
    class opa_setup,cia,hitemp,exomol opacity
    class model,pt,telluric forward
    class svi,mcmc,pred inference
    class posterior,plots output
```

### Data Flow Summary

1. **Configuration** → Load planet parameters, instrument settings, and sampling hyperparameters
2. **Data Preparation** → Load observed spectrum, build high-resolution wavenumber grid
3. **Opacity Setup** → Load/compute CIA and molecular cross-sections (cached as `.zarr`)
4. **Forward Model** → NumPyro probabilistic model: T-P profile → opacities → radiative transfer → spectrum
5. **Inference** → SVI warm-up finds good initialization, then HMC-NUTS samples the posterior
6. **Output** → Posterior samples, predictive spectra, and diagnostic plots

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
├── config/                # Configuration package
│   ├── planets.py         #   Planet/system parameters from literature
│   ├── instrument.py      #   Spectrograph settings (PEPSI/LBT)
│   ├── model.py           #   RT and spectral grid parameters
│   ├── paths.py           #   Database paths and output directories
│   └── inference.py       #   SVI and MCMC sampling parameters
├── load.py                # Data loading (PEPSI, JWST formats)
├── preprocess.py          # PEPSI data preprocessing
├── grid_setup.py          # Wavenumber grid and spectral operators
├── opacity_setup.py       # CIA, molecular, atomic opacities
├── model.py               # NumPyro model for HRCCS retrieval
├── transmission_model.py  # Transmission forward model
├── emission_model.py      # Emission forward model
├── telluric_model.py      # Telluric absorption (ground-based)
├── inference.py           # SVI and HMC-NUTS
├── plot.py                # Visualization
├── retrieval.py           # Retrieval pipeline functions
└── __main__.py            # CLI entry point
```

## Configuration

Configuration is split into logical modules under `config/`:

- **planets.py**: Planet parameters from published literature with ephemeris source tracking
  - `PLANET` and `EPHEMERIS` select the active target
  - `get_params()` returns parameters for the current planet/ephemeris
  - Supports multiple ephemeris sources per planet (e.g., "Duck24", "Talens18")

- **instrument.py**: Spectrograph and observatory settings
  - `RESOLUTION`, `OBSERVING_MODE`, wavelength ranges
  - FITS header key mappings and file patterns

- **model.py**: Radiative transfer parameters
  - Pressure/temperature ranges, atmospheric layers
  - Spectral grid resolution, cloud parameters

- **paths.py**: Database and output paths
  - HITEMP, ExoMol, Kurucz database locations
  - Input/output directory structure

- **inference.py**: Sampling parameters
  - SVI steps and learning rate
  - MCMC warmup, samples, chains

All settings are re-exported from `config/__init__.py` for convenience:
```python
from config import PLANET, get_params, RESOLUTION, NLAYER
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
