# atmo-retrieval

Bayesian atmospheric retrieval for high-resolution exoplanet spectra.

## Install

```bash
pip install jax numpyro exojax astropy matplotlib corner
```

## Quick Start (CLI)

```bash
# Transmission retrieval
python -m atmo_retrieval --planet KELT-20b --mode transmission --epoch 20250601

# Emission retrieval with explicit P-T profile
python -m atmo_retrieval --planet WASP-76b --mode emission --epoch 20240315 --pt-profile guillot

# Quick smoke run
python -m atmo_retrieval --planet KELT-20b --mode transmission --epoch 20250601 --quick
```

## Quick Start (Python)

```python
from pipeline.retrieval import run_retrieval

run_retrieval(
    mode="transmission",
    epoch="20250601",
    pt_profile="guillot",
    phase_mode="shared",
    skip_svi=False,
    svi_only=False,
    seed=42,
)
```

## Module Flow

```mermaid
flowchart TD
    A[atmo_retrieval.py<br/>CLI entrypoint] --> B[pipeline/retrieval.py<br/>orchestration]

    B --> C[config/*<br/>runtime settings]
    B --> D[dataio/load.py<br/>observed data loading]
    B --> E[physics/grid_setup.py<br/>nu grid + operators]
    B --> F[databases/opacity.py<br/>CIA/molecular/atomic opacities]
    F --> G[databases/atomic.py<br/>Kurucz/VALD helpers]
    B --> H[physics/model.py<br/>forward model]
    H --> I[physics/pt.py<br/>P-T profiles]
    H --> J[physics/chemistry.py<br/>composition]
    B --> K[pipeline/inference.py<br/>SVI + NUTS]
    B --> L[plotting/plot.py<br/>figures]

    A --> M[pipeline/retrieval_binned.py<br/>phase-binned wrapper]
    M --> B

    N[dataio/make_transmission.py<br/>preprocessing utility] -. writes .-> O[input/spectra/.../*.npy]
    O -. read by .-> B
```

## Data Layout

Expected input directory structure:

```
input/spectra/{planet}/{epoch}/{arm}/
  wavelength.npy
  data.npy
  sigma.npy
  phase.npy
```

## Outputs

```
output/{planet}/{ephemeris}/{mode}/{timestamp}/
  run_config.log
  mcmc_summary.txt
  posterior_sample.npz
  atmospheric_state.npz
  contribution_function.pdf
```

## Notes

- Default P-T profile: `guillot`.
- Database directories live under `input/.db_*` (override via env vars in `config/paths_config.py`).
- Run from repo root.
