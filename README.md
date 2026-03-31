# atmo-retrieval

bayesian atmospheric retrieval for high-resolution exoplanet spectra

## install

```bash
pip install jax numpyro exojax astropy matplotlib corner
```

## cli

```bash
python -m atmo_retrieval --planet KELT-20b --mode transmission --epoch 20250601
```

Low-resolution inputs are passed explicitly:
- use `--joint-spectrum-tbl path/to/file.tbl` for multi-bin low-res spectra
- use `--bandpass-tbl path/to/file.tbl` for single-band / sparse broadband constraints
- paths can be full paths or relative to `input/lrs`, e.g. `kelt20b/file.tbl`

## joint retrieval bandpass constraints

```bash
# bandpass constraints are added through pipeline.retrieval
# using build_bandpass_observation_config / bandpass_constraints
```

## code structure

```mermaid
flowchart TD
    subgraph EP[Entry points]
        A[atmo_retrieval.py<br/>CLI]
        M[pipeline/retrieval_binned.py<br/>phase-binned wrapper]
    end

    subgraph DS[Data sources]
        T1[input/hrs time-series products]
        T2[input/hrs collapsed 1D spectrum products]
        T3[input/lrs explicit low-res .tbl files]
        T4[direct Python arrays and component dicts]
    end

    subgraph UT[Optional utilities]
        U1[dataio/collapse_transmission_timeseries_to_1d.py<br/>derive 1D transmission spectrum]
        U2[dataio/collapse_emission_timeseries_to_1d.py<br/>derive 1D emission spectrum]
        U3[dataio/import_nasa_archive.py<br/>NASA archive import helper]
        X[pipeline.retrieval helpers<br/>convert .tbl to joint_spectra or bandpass_constraints]
    end

    subgraph RT[Runtime retrieval path]
        B[pipeline/retrieval.py<br/>orchestration]
        C[config runtime settings]
        D[dataio/load.py<br/>time-series and 1D spectrum loaders]
        D2[dataio/bandpass.py<br/>bandpass response loader]
        E[physics/grid_setup.py<br/>spectral grid and operators]
        F[databases/opacity.py<br/>CIA molecular and atomic opacity loaders]
        G[databases/atomic.py<br/>Kurucz and VALD helpers]
        H[physics/model.py<br/>joint forward model]
        I[physics/pt.py<br/>P-T profiles]
        J[physics/chemistry.py<br/>composition]
        K[pipeline/inference.py<br/>SVI and NUTS]
        L[plotting/plot.py<br/>figures and summaries]
    end

    A --> B
    A --> M
    M --> D
    M --> B

    B --> C
    B --> D
    B --> D2
    B --> E
    B --> F
    F --> G
    B --> H
    H --> I
    H --> J
    B --> K
    B --> L

    T1 -. used directly when data_format is timeseries .-> D
    T1 -. timeseries only .-> M
    T2 -. optional input when data_format is spectrum .-> D
    T3 -. passed via joint-spectrum-tbl or bandpass-tbl .-> X
    X --> B
    T4 -. optional programmatic entry .-> B

    T1 --> U1
    T1 --> U2
    U1 -. writes derived 1D npy products .-> T2
    U2 -. writes derived 1D npy products .-> T2
    U3 -. writes imported archive files .-> T3
```

- Solid arrows: normal code dependencies or execution flow.
- Dashed arrows: optional inputs, helper conversions, or derived products.
- The collapse-to-1D scripts are utilities for the `spectrum` path only; they are not required for the principal `timeseries` path.

modules:

```text
.
├── atmo_retrieval.py
├── config
│   ├── __init__.py
│   ├── chemistry_config.py
│   ├── data_config.py
│   ├── inference_config.py
│   ├── instrument_config.py
│   ├── model_config.py
│   ├── paths_config.py
│   ├── planets_config.py
│   ├── photometry_config.py
│   └── tellurics_config.py
├── databases
│   ├── atomic.py
│   └── opacity.py
├── dataio
│   ├── bandpass.py
│   ├── import_nasa_archive.py
│   ├── load.py
│   ├── collapse_emission_timeseries_to_1d.py
│   ├── collapse_transmission_timeseries_to_1d.py
│   └── tellurics.py
├── environment.yml
├── physics
│   ├── chemistry_draft.py
│   ├── chemistry.py
│   ├── grid_setup.py
│   ├── model.py
│   └── pt.py
├── pipeline
│   ├── inference.py
│   ├── memory_profile.py
│   ├── retrieval_binned.py
│   └── retrieval.py
├── plotting
│   ├── __init__.py
│   ├── aliasing.py
│   └── plot.py
└── tests
```

## expected input directory structure

```
input/hrs/{planet}/{epoch}/{arm}/
  wavelength_transmission.npy
  spectrum_transmission.npy
  uncertainty_transmission.npy

input/hrs/{epoch}_{planet}/
  ... raw PEPSI files ...

input/lrs/{planet}/
  *.tbl
  *.dat
  *.csv
```

## outputs

```
output/{planet}/{ephemeris}/{mode}/{timestamp}/
  run_config.log
  mcmc_summary.txt
  posterior_sample.npz
  atmospheric_state.npz
  contribution_function.pdf
```
