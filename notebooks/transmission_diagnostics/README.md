# Transmission diagnostics notebooks

- `current_runs/` preserves the four executed notebooks and their existing outputs exactly as they were.
- `collapse_source/` contains clean, output-free copies for rerunning against each arm's regenerated `collapse_source/` time-series cube.

The collapse-source notebooks also look for the full-transit 1D products under:

`input/hrs/transmission/<planet>/<epoch>/<arm>/collapsed/full_transit/`

MASCARA-1b currently has no prepared collapse-source cube, so its updated notebook will report that no bundles are available until that preparation blocker is resolved.
