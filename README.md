# atmo-retrieval

Bayesian atmospheric retrieval for high-resolution exoplanet spectra.

## Install

```bash
pip install jax numpyro exojax astropy matplotlib corner
```

## Quick Start (CLI)

```bash
# Transmission retrieval
python __main__.py --planet KELT-20b --mode transmission --epoch 20250601

# Emission retrieval with explicit P-T profile
python __main__.py --planet WASP-76b --mode emission --epoch 20240315 --pt-profile guillot

# Quick smoke run
python __main__.py --planet KELT-20b --mode transmission --epoch 20250601 --quick
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
