# Joint Retrieval Model Draft

## Core Idea

Build one joint retrieval model that:

- samples the shared planetary and atmospheric parameters once
- constructs one or more atmospheric states from those shared parameters
- passes each atmospheric state to one or more observation components
- sums the log-likelihoods from all enabled components

This keeps the physics shared while letting each dataset own its own forward model details.


## Key Decisions

### 1. Separate `art` objects: yes

Use separate `art` objects per observation component, or at minimum per observable type.

Reason:

- transmission and emission use different radiative transfer classes in exojax
- components can live on different pressure grids or spectral grids
- components can require different radiative transfer assumptions
- photometry is not a new opacity source, it is a different observation operator applied to a spectrum

Expected mapping:

- high-resolution transmission: `ArtTransPure`
- low-resolution transmission: `ArtTransPure`
- high-resolution emission: `ArtEmisPure` or another emission-capable ART class
- low-resolution emission: `ArtEmisPure` or another emission-capable ART class
- reflected-light or mixed reflected-plus-emitted datasets: `ArtReflectPure` or `ArtReflectEmis` if needed later

Photometry should usually reuse an emission or reflection spectrum and then apply `SopPhoto`. It should not be inserted into `_compute_opacity_terms`.


### 2. Shared state vs component state

Keep these separate.

Shared parameters:

- `Mp`
- `Rp`
- `Rstar`
- PT profile parameters
- composition parameters
- any global chemistry configuration

Usually shared for high-resolution spectroscopy:

- `Kp`
- `Vsys`

Not always global:

- `dRV`
- normalization terms
- jitter terms
- offsets
- SYSREM-related nuisance parameters

Those often belong to specific high-resolution datasets, not to the whole joint model.


### 3. Pipeline choices stay out of the model

Phase-bin selection, full-transit versus subset selection, and dataset loading should stay in the pipeline layer.

The model should receive already-selected data and metadata. It should not decide which exposures to keep.


### 4. Opacity helpers stay standalone

Keep:

- `compute_opacity`
- `compute_opacity_per_species`
- `_compute_cia_opacity_terms`
- `_compute_xs_opacity_terms`
- `_compute_opacity_terms`
- `_sum_opacity_terms`

These are physics utilities and should remain reusable outside any class wrapper.


## Recommended Architecture

### A. Shared parameter/config layer

```python
@dataclass(frozen=True)
class SharedPlanetConfig:
    period_day: float
    Mp_prior: object
    Rp_prior: object
    Rstar_prior: object
    Kp_prior: object | None = None
    Vsys_prior: object | None = None
```

```python
@dataclass(frozen=True)
class AtmosphereRegionConfig:
    name: str
    mode: Literal["transmission", "emission"]
    pt_profile: str
    composition_solver: CompositionSolver
```

```python
@dataclass
class AtmosphereState:
    Tarr: jnp.ndarray
    g_ref: jnp.ndarray
    g_profile: jnp.ndarray
    mmw_profile: jnp.ndarray
    mmr_mols: dict[str, jnp.ndarray]
    mmr_atoms: dict[str, jnp.ndarray]
    vmrH2_profile: jnp.ndarray
    vmrHe_profile: jnp.ndarray
```

Notes:

- `g_ref` is derived from `Mp` and `Rp`; it should not be an independent sampled attribute
- `Tarr` should belong to an atmospheric region or state, not to the top-level model object
- if transmission and emission should have different atmospheres, define two regions such as `terminator` and `dayside`


### B. Observation component layer

Each dataset is one component.

Each component owns:

- data and uncertainties
- spectral grid or bandpass grid
- opacities
- `art` object
- instrumental operators
- phase or time arrays if needed
- nuisance settings such as SYSREM inputs or normalization rules
- a reference to the atmospheric region it uses

Suggested component types:

- `HighResTransmissionComponent`
- `HighResEmissionComponent`
- `LowResTransmissionSpectrumComponent`
- `LowResEmissionSpectrumComponent`
- `PhotometryBandComponent`

Common interface:

```python
class ObservationComponent(Protocol):
    name: str
    region_name: str

    def log_likelihood(
        self,
        shared_params: dict[str, jnp.ndarray],
        atmosphere_state: AtmosphereState,
    ) -> jnp.ndarray:
        ...
```


### C. Joint model layer

```python
class JointRetrievalModel:
    def __init__(
        self,
        shared_planet_config: SharedPlanetConfig,
        atmosphere_regions: dict[str, AtmosphereRegionConfig],
        components: dict[str, ObservationComponent],
    ):
        self.shared_planet_config = shared_planet_config
        self.atmosphere_regions = atmosphere_regions
        self.components = components

    def __call__(self) -> None:
        shared = sample_shared_parameters(self.shared_planet_config)

        states = {
            name: build_atmosphere_state(shared, region_cfg)
            for name, region_cfg in self.atmosphere_regions.items()
        }

        total_logl = 0.0
        for component in self.components.values():
            total_logl = total_logl + component.log_likelihood(
                shared_params=shared,
                atmosphere_state=states[component.region_name],
            )

        numpyro.factor("logL_total", total_logl)
```


## What Each Component Actually Does

### High-resolution transmission component

- uses `ArtTransPure`
- computes `dtau` from the shared opacity helpers
- converts `dtau` into a model time series
- applies rotational and instrumental broadening
- shifts by `Kp`, `Vsys`, and the component's `dRV`
- optionally applies SYSREM filtering and per-exposure mean subtraction
- evaluates the high-resolution matched-filter likelihood


### High-resolution emission component

- same general pattern as high-resolution transmission
- uses an emission-capable ART class
- uses the emission observable path rather than the transmission path
- may share `Kp` and `Vsys` with other high-resolution components


### Low-resolution transmission or emission spectrum component

- uses its own `art`, grid, opacities, and instrument operator
- usually does not need high-resolution RV shifting machinery
- likelihood will likely be a direct Gaussian likelihood in spectral space


### Photometry component

- should not introduce separate "photometry opacity terms"
- should start from a model spectrum on the component's photometric grid
- uses `SopPhoto` to integrate through the bandpass
- compares the predicted flux, eclipse depth, transit depth, or magnitude to the observed photometric quantity

This is an observation operator on top of a spectrum, not a new absorber term.


## Treatment Of `tess_proc`

Do not force `tess_proc` directly into the same abstraction unless you are sure you want a fully joint light-curve-plus-spectrum inference.

Two reasonable options:

### Option A: keep `tess_proc` separate

- run `tess_proc` as a preprocessing or standalone inference step
- feed its output into the retrieval as a photometric constraint
- simplest and least disruptive

### Option B: build a true photometry component

- create a `PhotometryBandComponent`
- use `SopPhoto` and a bandpass-aware forward model
- include its likelihood directly in the joint retrieval

If the immediate goal is spectral retrieval with one or two bandpass constraints, Option A is cleaner. If the goal is a general multi-observable framework, Option B is worth building.


## Relationship To Current `physics/model.py`

Keep the existing staged logic and reuse it.

Existing pieces that already map well:

- opacity helpers
- temperature-profile sampling
- composition solver interface
- `compute_model_timeseries`
- pipeline corrections
- high-resolution likelihood
- `RetrievalModelConfig`

Likely refactor direction:

- keep the shared physics helpers in `physics/model.py`
- move dataset-specific forward-model logic into component classes or component configs
- let the joint model orchestrate shared sampling plus component likelihood calls


## What Not To Do

- do not make one global `art` for all datasets
- do not make one global `nu_grid` for all datasets
- do not store `Tarr` as a single top-level class attribute if transmission and emission may differ
- do not treat photometry as an opacity contribution
- do not make `create_model` responsible for data filtering, phase-bin selection, or CLI branching
- do not hard-code exactly five slots named `hrts`, `hres`, `lrts`, `lres`, `phot` if you may need multiple datasets per type


## Suggested First Implementation Target

Start with this minimal scope:

1. one shared atmosphere state
2. one `HighResTransmissionComponent`
3. one `LowResEmissionSpectrumComponent` or one `PhotometryBandComponent`
4. one joint model that sums their log-likelihoods

Once that works, generalize from "one component of each kind" to "arbitrary component list".


## Open Questions

- Should transmission and emission share one atmosphere, or should there be separate `terminator` and `dayside` states?
- Should `dRV` be global across all high-resolution datasets, or dataset-specific?
- Should photometry enter as a direct joint component or as an external constraint from `tess_proc`?
- Will low-resolution datasets share chemistry/PT with the high-resolution datasets, or should the framework allow partial coupling?
