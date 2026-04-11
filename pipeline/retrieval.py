import hashlib
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from astropy import constants as const
from astropy import units as u
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_median

from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.utils.grids import wav2nu
from exojax.utils.astrofunc import gravity_jupiter as gravity_surface
from exojax.utils.constants import RJ, Rs

import config
from dataio.load import (
    load_nasa_archive_spectrum,
    load_observed_spectrum,
    parse_nasa_archive_tbl,
)
from physics.chemistry import ConstantVMR, FastChemHybridChemistry, FreeVMR
from physics.grid_setup import setup_wavenumber_grid, setup_spectral_operators
from databases.opacity import setup_cia_opacities, load_molecular_opacities, load_atomic_opacities
from physics.model import (
    BandpassObservationInputs,
    ChunkedSysremInputs,
    SpectroscopicObservationInputs,
    PhaseMode,
    compute_model_timeseries,
    compute_atmospheric_state_from_posterior,
    apply_model_pipeline_corrections,
    build_atmosphere_region_config,
    build_bandpass_observation_config,
    build_shared_system_config,
    build_spectroscopic_observation_config,
    create_joint_retrieval_model,
)
from pipeline.inference import run_svi
from dataio.bandpass import load_tess_bandpass
from plotting.plot import (
    plot_svi_loss,
    plot_transmission_spectrum,
    plot_emission_spectrum,
    plot_temperature_profile,
    plot_contribution_function,
    plot_contribution_per_species,
    plot_contribution_combined,
    save_retrieval_corner_plots,
)


def load_timeseries_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)

    wavelength = np.load(data_dir / "wavelength.npy")
    data = np.load(data_dir / "data.npy")
    sigma = np.load(data_dir / "sigma.npy")
    phase = np.load(data_dir / "phase.npy")
    
    return wavelength, data, sigma, phase


def _normalize_phoenix_spectrum_path(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate.resolve())


def _normalize_phoenix_cache_dir(path: str | Path | None) -> Path:
    candidate = config.PHOENIX_CACHE_DIR if path is None else Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


@lru_cache(maxsize=None)
def _read_phoenix_spectrum_ascii(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"PHOENIX spectrum file not found: {path}")

    try:
        raw = np.loadtxt(path, dtype=float, comments="#")
    except Exception as exc:
        raise ValueError(f"Failed to read PHOENIX spectrum from {path}: {exc}") from exc

    if raw.ndim == 1:
        if raw.size == 0:
            raise ValueError(f"PHOENIX spectrum file is empty: {path}")
        if raw.size != 2:
            raise ValueError(
                f"PHOENIX spectrum file {path} must have exactly two columns "
                "(wavelength_A, stellar_surface_flux)."
            )
        raw = raw[np.newaxis, :]

    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(
            f"PHOENIX spectrum file {path} must have exactly two columns "
            f"(wavelength_A, stellar_surface_flux); got shape {raw.shape}."
        )

    wavelength_angstrom = np.asarray(raw[:, 0], dtype=float)
    stellar_surface_flux = np.asarray(raw[:, 1], dtype=float)
    if np.any(~np.isfinite(wavelength_angstrom)) or np.any(~np.isfinite(stellar_surface_flux)):
        raise ValueError(f"PHOENIX spectrum file {path} contains non-finite values.")

    order = np.argsort(wavelength_angstrom)
    wavelength_angstrom = wavelength_angstrom[order]
    stellar_surface_flux = stellar_surface_flux[order]
    if np.any(np.diff(wavelength_angstrom) <= 0):
        raise ValueError(
            f"PHOENIX spectrum file {path} must be strictly increasing in wavelength."
        )
    if np.any(stellar_surface_flux <= 0):
        raise ValueError(
            f"PHOENIX spectrum file {path} must have strictly positive stellar surface flux values."
        )

    return wavelength_angstrom, stellar_surface_flux


@lru_cache(maxsize=None)
def _read_processed_phoenix_cache(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_str)
    try:
        with np.load(path) as raw:
            wavelength_angstrom = np.asarray(raw["wavelength_angstrom"], dtype=float)
            stellar_surface_flux = np.asarray(raw["stellar_surface_flux"], dtype=float)
    except Exception as exc:
        raise ValueError(f"Failed to read cached PHOENIX spectrum from {path}: {exc}") from exc

    if wavelength_angstrom.ndim != 1 or stellar_surface_flux.ndim != 1:
        raise ValueError(
            f"Cached PHOENIX spectrum {path} must contain 1D wavelength_angstrom and "
            "stellar_surface_flux arrays."
        )
    if wavelength_angstrom.shape != stellar_surface_flux.shape:
        raise ValueError(
            f"Cached PHOENIX spectrum {path} has mismatched shapes "
            f"{wavelength_angstrom.shape} and {stellar_surface_flux.shape}."
        )
    return wavelength_angstrom, stellar_surface_flux


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _derive_stellar_logg_cgs(Mstar_msun: float, Rstar_rsun: float) -> float:
    g_cgs = (
        const.G * (Mstar_msun * const.M_sun) / (Rstar_rsun * const.R_sun) ** 2
    ).to_value(u.cm / u.s**2)
    return float(np.log10(g_cgs))


def _resolve_chromatic_phoenix_parameters(
    *,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> tuple[float, float, float]:
    temperature = _coerce_optional_float(Tstar)
    if temperature is None:
        raise ValueError(
            f"Emission component '{component_name}' requires a finite Tstar for chromatic "
            "PHOENIX retrieval."
        )

    resolved_logg = _coerce_optional_float(logg_star)
    if resolved_logg is None:
        Mstar_val = _coerce_optional_float(Mstar)
        Rstar_val = _coerce_optional_float(Rstar)
        if Mstar_val is None or Rstar_val is None:
            raise ValueError(
                f"Emission component '{component_name}' requires either a finite logg_star "
                "or both finite M_star and R_star for chromatic PHOENIX retrieval."
            )
        resolved_logg = _derive_stellar_logg_cgs(Mstar_val, Rstar_val)

    resolved_metallicity = _coerce_optional_float(metallicity)
    if resolved_metallicity is None:
        resolved_metallicity = 0.0

    return temperature, resolved_logg, resolved_metallicity


def _format_phoenix_cache_float(value: float) -> str:
    return f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _build_processed_phoenix_cache_path(
    *,
    cache_dir: Path,
    temperature: float,
    logg: float,
    metallicity: float,
    target_wavelength_angstrom: np.ndarray,
) -> Path:
    grid_hash = hashlib.sha1(
        np.asarray(target_wavelength_angstrom, dtype=np.float64).tobytes()
    ).hexdigest()[:16]
    filename = (
        "phoenix_"
        f"T{_format_phoenix_cache_float(temperature)}_"
        f"logg{_format_phoenix_cache_float(logg)}_"
        f"feh{_format_phoenix_cache_float(metallicity)}_"
        f"{grid_hash}.npz"
    )
    return cache_dir / filename


def _convert_chromatic_surface_flux_to_exojax_units(
    wavelength: u.Quantity,
    surface_flux: u.Quantity,
) -> np.ndarray:
    wavelength_cm = u.Quantity(wavelength).to(u.cm)
    energy_per_photon = (const.h * const.c / wavelength_cm) / u.photon
    surface_flux_lambda = (u.Quantity(surface_flux) * energy_per_photon).to(
        u.erg / (u.s * u.cm**2 * u.cm)
    )
    surface_flux_wavenumber = surface_flux_lambda * wavelength_cm**2
    return np.asarray(
        surface_flux_wavenumber.to_value(u.erg / (u.s * u.cm)),
        dtype=float,
    )


def _load_chromatic_phoenix_surface_flux_on_grid(
    *,
    nu_grid: np.ndarray,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    phoenix_cache_dir: str | Path | None,
) -> np.ndarray:
    target_wavelength_angstrom = 1.0e8 / np.asarray(nu_grid, dtype=float)
    temperature, resolved_logg, resolved_metallicity = _resolve_chromatic_phoenix_parameters(
        component_name=component_name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    cache_dir = _normalize_phoenix_cache_dir(phoenix_cache_dir)
    cache_path = _build_processed_phoenix_cache_path(
        cache_dir=cache_dir,
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        target_wavelength_angstrom=target_wavelength_angstrom,
    )

    if cache_path.exists():
        cached_wavelength_angstrom, cached_flux = _read_processed_phoenix_cache(str(cache_path))
        if cached_wavelength_angstrom.shape == target_wavelength_angstrom.shape and np.array_equal(
            cached_wavelength_angstrom,
            target_wavelength_angstrom,
        ):
            return cached_flux

    try:
        from chromatic import get_phoenix_photons
    except ImportError as exc:
        raise ImportError(
            "chromatic-lightcurves is required to auto-fetch PHOENIX spectra when "
            "phoenix_spectrum_path is not provided. Install chromatic-lightcurves or "
            "provide a local two-column PHOENIX spectrum file."
        ) from exc

    sort_idx = np.argsort(target_wavelength_angstrom)
    query_wavelength_um = (target_wavelength_angstrom[sort_idx] / 1.0e4) * u.micron
    wavelength_um, surface_flux = get_phoenix_photons(
        temperature=temperature,
        logg=resolved_logg,
        metallicity=resolved_metallicity,
        wavelength=query_wavelength_um,
    )
    flux_sorted = _convert_chromatic_surface_flux_to_exojax_units(
        wavelength=wavelength_um,
        surface_flux=surface_flux,
    )
    stellar_surface_flux = np.empty_like(flux_sorted)
    stellar_surface_flux[sort_idx] = flux_sorted

    if np.any(~np.isfinite(stellar_surface_flux)):
        raise ValueError(
            f"chromatic returned non-finite PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )
    if np.any(stellar_surface_flux <= 0):
        raise ValueError(
            f"chromatic returned non-positive PHOENIX surface flux values for emission "
            f"component '{component_name}'."
        )

    np.savez_compressed(
        cache_path,
        wavelength_angstrom=target_wavelength_angstrom,
        stellar_surface_flux=stellar_surface_flux,
        temperature=np.asarray(temperature, dtype=float),
        logg=np.asarray(resolved_logg, dtype=float),
        metallicity=np.asarray(resolved_metallicity, dtype=float),
    )
    return stellar_surface_flux


def _load_phoenix_surface_flux_on_grid(
    *,
    phoenix_spectrum_path: str | Path | None,
    phoenix_cache_dir: str | Path | None,
    nu_grid: np.ndarray,
    mode: str,
    component_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
) -> np.ndarray | None:
    if _normalize_retrieval_mode(mode) != "emission":
        return None

    if phoenix_spectrum_path is None:
        return _load_chromatic_phoenix_surface_flux_on_grid(
            nu_grid=nu_grid,
            component_name=component_name,
            Tstar=Tstar,
            logg_star=logg_star,
            metallicity=metallicity,
            Mstar=Mstar,
            Rstar=Rstar,
            phoenix_cache_dir=phoenix_cache_dir,
        )

    normalized_path = _normalize_phoenix_spectrum_path(phoenix_spectrum_path)
    wavelength_angstrom, stellar_surface_flux = _read_phoenix_spectrum_ascii(normalized_path)

    target_wavelength_angstrom = 1.0e8 / np.asarray(nu_grid, dtype=float)
    target_min = float(np.min(target_wavelength_angstrom))
    target_max = float(np.max(target_wavelength_angstrom))
    source_min = float(np.min(wavelength_angstrom))
    source_max = float(np.max(wavelength_angstrom))
    if target_min < source_min or target_max > source_max:
        raise ValueError(
            f"PHOENIX spectrum {normalized_path} does not cover the wavelength range "
            f"required by emission component '{component_name}' "
            f"({target_min:.3f}-{target_max:.3f} A requested, "
            f"{source_min:.3f}-{source_max:.3f} A available)."
        )

    interpolated_flux = np.interp(
        target_wavelength_angstrom,
        wavelength_angstrom,
        stellar_surface_flux,
    )
    if np.any(~np.isfinite(interpolated_flux)):
        raise ValueError(
            f"Interpolated PHOENIX stellar surface flux for component '{component_name}' "
            f"contains non-finite values."
        )

    return np.asarray(interpolated_flux, dtype=float)


@dataclass(frozen=True)
class SysremInputBundle:
    U: np.ndarray | None = None
    V: np.ndarray | None = None
    chunk_indices: tuple[np.ndarray, ...] | None = None
    U_chunks: tuple[np.ndarray, ...] | None = None
    V_chunks: tuple[np.ndarray, ...] | None = None

    @property
    def is_chunked(self) -> bool:
        return self.chunk_indices is not None


def _chunk_indices_from_labels(chunk_labels: np.ndarray) -> tuple[np.ndarray, ...]:
    chunk_labels = np.asarray(chunk_labels, dtype=int)
    if chunk_labels.ndim != 1:
        raise ValueError(f"chunk_labels must be 1D, got shape {chunk_labels.shape}.")
    if chunk_labels.size == 0:
        raise ValueError("chunk_labels is empty.")
    if np.any(chunk_labels < 0):
        raise ValueError("chunk_labels must be non-negative for all wavelength columns.")

    labels = sorted(int(label) for label in np.unique(chunk_labels))
    expected = list(range(len(labels)))
    if labels != expected:
        raise ValueError(
            f"chunk_labels must be contiguous and start at 0; got labels {labels}."
        )

    return tuple(np.where(chunk_labels == label)[0].astype(int) for label in labels)


def _load_sysrem_inputs(data_dir: str | Path) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)

    u_candidates = [
        data_dir / "U_sysrem.npz",
        data_dir / "U.npy",
        data_dir / "U_sysrem.npy",
    ]
    v_candidates = [
        data_dir / "V.npy",
        data_dir / "V_diag.npy",
        data_dir / "inv_sigma.npy",
        data_dir / "invsigma.npy",
    ]

    u_path = next((p for p in u_candidates if p.exists()), None)
    v_path = next((p for p in v_candidates if p.exists()), None)

    if u_path is None:
        raise FileNotFoundError(f"No SYSREM basis file found in {data_dir}.")

    if u_path.suffix == ".npz":
        with np.load(u_path) as u_data:
            raw = {name: np.asarray(u_data[name]) for name in u_data.files}
        if "chunk_labels" in raw:
            return raw
        if v_path is None:
            raise FileNotFoundError(
                f"Legacy SYSREM bundle {u_path.name} does not include chunk metadata and no "
                f"V file was found in {data_dir}."
            )
        raw["V"] = np.load(v_path)
        return raw

    if v_path is None:
        raise FileNotFoundError(
            f"No SYSREM weighting file found alongside {u_path.name} in {data_dir}."
        )
    return {
        "U": np.load(u_path),
        "V": np.load(v_path),
    }


def _validate_sysrem_inputs(
    raw: dict[str, np.ndarray],
    n_exp: int,
) -> SysremInputBundle:
    if "chunk_labels" in raw:
        U = np.asarray(raw["U_sysrem"] if "U_sysrem" in raw else raw["U"])
        if U.ndim == 2:
            U = U[:, :, np.newaxis]
        if U.ndim != 3:
            raise ValueError(
                f"Chunked SYSREM U must have shape (n_exp, n_basis, n_chunks); got {U.shape}."
            )
        if U.shape[0] != n_exp:
            raise ValueError(
                f"U exposure axis mismatch: U.shape[0]={U.shape[0]} but n_exp={n_exp}."
            )

        chunk_indices = _chunk_indices_from_labels(raw["chunk_labels"])
        n_chunks = len(chunk_indices)
        if U.shape[2] != n_chunks:
            raise ValueError(
                f"Chunk count mismatch: U has {U.shape[2]} chunks but chunk_labels encodes {n_chunks}."
            )

        V_chunk_diag = raw.get("V_chunk_diag")
        if V_chunk_diag is None:
            raise ValueError("Chunked SYSREM bundle is missing V_chunk_diag.")
        V_chunk_diag = np.asarray(V_chunk_diag, dtype=float)
        if V_chunk_diag.ndim == 1:
            V_chunk_diag = V_chunk_diag[np.newaxis, :]
        expected_chunk_shape = (n_chunks, n_exp)
        if V_chunk_diag.shape != expected_chunk_shape:
            raise ValueError(
                f"V_chunk_diag shape mismatch: got {V_chunk_diag.shape}, expected {expected_chunk_shape}."
            )

        U_chunks: list[np.ndarray] = []
        V_chunks: list[np.ndarray] = []
        for chunk in range(n_chunks):
            U_chunk = np.asarray(U[:, :, chunk], dtype=float)
            keep = np.any(np.isfinite(U_chunk), axis=0)
            U_chunk = U_chunk[:, keep]
            if U_chunk.ndim != 2 or U_chunk.shape[0] != n_exp:
                raise ValueError(
                    f"Chunk {chunk} has invalid U shape {U_chunk.shape} for n_exp={n_exp}."
                )

            V_diag = np.asarray(V_chunk_diag[chunk], dtype=float)
            if np.any(~np.isfinite(V_diag)) or np.any(V_diag <= 0):
                raise ValueError(
                    f"Chunk {chunk} has invalid V_chunk_diag values; all entries must be finite and > 0."
                )

            U_chunks.append(U_chunk)
            V_chunks.append(np.diag(V_diag))

        return SysremInputBundle(
            chunk_indices=tuple(chunk_indices),
            U_chunks=tuple(U_chunks),
            V_chunks=tuple(V_chunks),
        )

    U = np.asarray(raw["U_sysrem"] if "U_sysrem" in raw else raw["U"])
    V = np.asarray(raw["V"])

    if U.ndim == 3:
        raise ValueError(
            "3D SYSREM inputs now require an explicit chunked bundle with chunk_labels and "
            "V_chunk_diag in U_sysrem.npz. Legacy 3D U arrays without chunk metadata are unsupported."
        )

    if U.shape[0] != n_exp:
        raise ValueError(
            f"U exposure axis mismatch: U.shape[0]={U.shape[0]} but n_exp={n_exp}."
        )
    if V.ndim == 1:
        if V.size != n_exp:
            raise ValueError(
                f"V exposure axis mismatch: V.size={V.size} but n_exp={n_exp}."
            )
        V = np.diag(V)
    elif V.ndim == 2:
        expected_shape = (n_exp, n_exp)
        if V.shape != expected_shape:
            raise ValueError(
                f"V shape mismatch: V.shape={V.shape} but expected {expected_shape}."
            )

    return SysremInputBundle(U=U, V=V)


def _build_model_chunked_sysrem(
    sysrem: SysremInputBundle | None,
) -> ChunkedSysremInputs | None:
    if sysrem is None or not sysrem.is_chunked:
        return None

    return ChunkedSysremInputs(
        chunk_indices=tuple(jnp.asarray(indices, dtype=jnp.int32) for indices in sysrem.chunk_indices),
        U_chunks=tuple(jnp.asarray(U_chunk) for U_chunk in sysrem.U_chunks),
        V_chunks=tuple(jnp.asarray(V_chunk) for V_chunk in sysrem.V_chunks),
    )


def _describe_sysrem_inputs(sysrem: SysremInputBundle) -> str:
    if sysrem.is_chunked:
        chunk_sizes = [int(indices.size) for indices in sysrem.chunk_indices]
        basis_counts = [int(U_chunk.shape[1]) for U_chunk in sysrem.U_chunks]
        return (
            f"chunked SYSREM: {len(chunk_sizes)} chunks, "
            f"chunk_sizes={chunk_sizes}, basis_counts={basis_counts}"
        )

    return f"U shape={sysrem.U.shape}, V shape={sysrem.V.shape}"


def _subset_sysrem_inputs(
    sysrem: SysremInputBundle | None,
    indices: np.ndarray,
) -> SysremInputBundle | None:
    if sysrem is None:
        return None

    indices = np.asarray(indices, dtype=int)
    if sysrem.is_chunked:
        return SysremInputBundle(
            chunk_indices=tuple(np.asarray(chunk_indices, dtype=int) for chunk_indices in sysrem.chunk_indices),
            U_chunks=tuple(np.asarray(U_chunk)[indices] for U_chunk in sysrem.U_chunks),
            V_chunks=tuple(
                np.asarray(V_chunk)[np.ix_(indices, indices)] for V_chunk in sysrem.V_chunks
            ),
        )

    return SysremInputBundle(
        U=np.asarray(sysrem.U)[indices],
        V=np.asarray(sysrem.V)[np.ix_(indices, indices)],
    )


def _normalize_phase(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase)
    if phase.size == 0:
        return phase

    phase_min = float(np.nanmin(phase))
    phase_max = float(np.nanmax(phase))
    median = float(np.nanmedian(phase))

    if 0.0 <= phase_min and phase_max <= 1.0 and abs(median - 0.5) < 0.2:
        print("  Phase appears centered on 0.5; shifting to mid-transit at 0.0")
        phase = phase - 0.5

    if phase_min < -0.5 or phase_max > 0.5:
        phase = (phase + 0.5) % 1.0 - 0.5

    return phase


def _build_composition_solver(
    chemistry_model: str,
    fastchem_parameter_file: str | None,
):
    model = chemistry_model.lower().strip()
    if model == "constant":
        return ConstantVMR()

    if model == "free":
        return FreeVMR()

    if model == "fastchem_hybrid_grid":
        parameter_file = fastchem_parameter_file or config.FASTCHEM_PARAMETER_FILE
        if parameter_file is None:
            raise ValueError(
                "chemistry_model='fastchem_hybrid_grid' requires a FastChem "
                "parameters.dat path. Pass --fastchem-parameter-file or set "
                "FASTCHEM_PARAMETER_FILE in config."
            )

        return FastChemHybridChemistry(
            fastchem_parameter_file=parameter_file,
            continuum_species=tuple(config.FASTCHEM_HYBRID_CONTINUUM_SPECIES),
            metallicity_range=tuple(config.FASTCHEM_HYBRID_METALLICITY_RANGE),
            co_ratio_range=tuple(config.FASTCHEM_HYBRID_CO_RATIO_RANGE),
            n_metallicity=int(config.FASTCHEM_HYBRID_N_METALLICITY),
            n_co_ratio=int(config.FASTCHEM_HYBRID_N_CO_RATIO),
            log_vmr_min=float(config.LOG_VMR_MIN),
            log_vmr_max=float(config.LOG_VMR_MAX),
            h2_he_ratio=float(config.H2_HE_RATIO),
            n_temp=int(config.FASTCHEM_N_TEMP),
            n_pressure=int(config.FASTCHEM_N_PRESSURE),
            t_min=float(config.FASTCHEM_T_MIN),
            t_max=float(config.FASTCHEM_T_MAX),
            cache_dir=config.FASTCHEM_CACHE_DIR,
        )

    raise ValueError(
        f"Unknown chemistry_model: {chemistry_model}. "
        "Choose from {'constant', 'free', 'fastchem_hybrid_grid'}."
    )


def _preflight_spectrum_checks(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    inst_nus: np.ndarray,
) -> None:
    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)
    phase = np.asarray(phase)
    inst_nus = np.asarray(inst_nus)

    if sigma.shape != data.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match data shape {data.shape}")

    if data.ndim == 1:
        if data.size != wav_obs.size:
            raise ValueError(f"data length {data.size} != wavelength length {wav_obs.size}")
        expected_exposures = 1
    else:
        if data.shape[1] != wav_obs.size:
            raise ValueError(
                f"data spectral axis {data.shape[1]} != wavelength length {wav_obs.size}"
            )
        expected_exposures = data.shape[0]

    if phase.size != expected_exposures:
        raise ValueError(
            f"phase length {phase.size} != number of exposures {expected_exposures}"
        )


def _preflight_grid_checks(inst_nus: np.ndarray, nu_grid: np.ndarray) -> None:
    inst_nus = np.asarray(inst_nus)
    nu_grid = np.asarray(nu_grid)

    inst_min = np.nanmin(inst_nus)
    inst_max = np.nanmax(inst_nus)
    nu_min = np.nanmin(nu_grid)
    nu_max = np.nanmax(nu_grid)

    if inst_min < nu_min or inst_max > nu_max:
        raise ValueError(
            "instrument wavenumber grid is outside model grid: "
            f"inst_nus={inst_min:.4g}..{inst_max:.4g}, "
            f"nu_grid={nu_min:.4g}..{nu_max:.4g}"
        )


def _sample_svi_posterior(
    guide: object | None,
    params: dict | None,
    rng_key: jax.Array,
    num_samples: int,
) -> dict[str, np.ndarray] | None:
    if guide is None or params is None or num_samples <= 0:
        return None

    try:
        svi_draws = guide.sample_posterior(
            rng_key,
            params,
            sample_shape=(num_samples,),
        )
    except Exception as exc:
        print(f"  Warning: failed to sample SVI posterior for corner plots: {exc}")
        return None

    return {
        name: np.asarray(jax.device_get(values))
        for name, values in svi_draws.items()
    }


def _summarize_observed_spectrum(
    data: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data_arr = np.asarray(data)
    sigma_arr = np.asarray(sigma)

    if data_arr.ndim == 1:
        return data_arr, sigma_arr

    obs_mean = np.mean(data_arr, axis=0)
    obs_err = np.sqrt(np.mean(np.square(sigma_arr), axis=0))
    return obs_mean, obs_err


def _validate_mcmc_device_layout(
    *,
    num_chains: int,
    chain_method: str,
    require_gpu_per_chain: bool,
) -> None:
    chain_method = str(chain_method).strip().lower()
    local_devices = list(jax.local_devices())
    gpu_devices = [device for device in local_devices if device.platform == "gpu"]

    print(f"  JAX default backend: {jax.default_backend()}")
    print(f"  JAX local devices: {local_devices}")

    if not require_gpu_per_chain:
        return

    if chain_method != "parallel":
        raise RuntimeError(
            "MCMC_REQUIRE_GPU_PER_CHAIN requires MCMC_CHAIN_METHOD='parallel'."
        )

    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "This run requires GPU-backed parallel chains, but JAX default backend "
            f"is {jax.default_backend()!r}."
        )

    if len(gpu_devices) < num_chains:
        raise RuntimeError(
            f"This run requires at least {num_chains} visible GPU devices for "
            f"{num_chains} MCMC chains, but JAX sees {len(gpu_devices)} GPU device(s). "
            "Request more GPUs or reduce --mcmc-chains."
        )


def _phase_dependent_drv(params: dict[str, float], phase: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(params.get("dRV", 0.0))


def _posterior_site_value(
    params: dict[str, float],
    site_name: str,
    *,
    sample_prefix: str | None = None,
    default: float | None = None,
):
    if sample_prefix is not None:
        scoped_name = f"{sample_prefix}/{site_name}"
        if scoped_name in params:
            return params[scoped_name]
    return params.get(site_name, default)


def _synthesize_timeseries_from_atmospheric_state(
    *,
    atmo_state: dict,
    model_params: dict,
    region_config: object,
    component: "SpectroscopicComponentBundle",
    component_sample_prefix: str | None,
) -> np.ndarray:
    params = atmo_state["params"]
    observation_config = component.observation_config

    Rp_rj = float(params.get("Rp", config.DEFAULT_POSTERIOR_RP))
    Mp_mj = float(params.get("Mp", config.DEFAULT_POSTERIOR_MP))
    Rstar_rs = float(params.get("Rstar", model_params["R_star"]))

    Rp_cm = Rp_rj * RJ
    Rstar_cm = Rstar_rs * Rs
    g_ref = gravity_surface(Rp_rj, Mp_mj)

    dtau = jnp.asarray(atmo_state["dtau"])
    Tarr = jnp.asarray(atmo_state["Tarr"])
    mmw_profile = jnp.asarray(atmo_state["mmw"])
    phase = np.asarray(component.phase)
    if observation_config.radial_velocity_mode == "none":
        phase = np.zeros_like(phase)
        Kp_kms = 0.0
        Vsys_kms = 0.0
        dRV = jnp.zeros_like(jnp.asarray(phase))
    else:
        Kp_kms = float(params.get("Kp", model_params["Kp"]))
        Vsys_kms = float(params.get("Vsys", model_params["RV_abs"]))
        dRV_0 = _posterior_site_value(
            params,
            "dRV_0",
            sample_prefix=component_sample_prefix,
        )
        dRV_slope = _posterior_site_value(
            params,
            "dRV_slope",
            sample_prefix=component_sample_prefix,
        )
        if dRV_0 is not None and dRV_slope is not None:
            dRV = jnp.asarray(dRV_0 + dRV_slope * phase)
        else:
            dRV = _phase_dependent_drv(
                {
                    "dRV": _posterior_site_value(
                        params,
                        "dRV",
                        sample_prefix=component_sample_prefix,
                        default=0.0,
                    )
                },
                phase,
            )
    model_ts = compute_model_timeseries(
        mode=observation_config.mode,
        art=region_config.art,
        dtau=dtau,
        Tarr=Tarr,
        mmw_profile=mmw_profile,
        Rp=Rp_cm,
        Rstar=Rstar_cm,
        g_ref=g_ref,
        phase=jnp.asarray(phase),
        Kp=Kp_kms,
        Vsys=Vsys_kms,
        dRV=dRV,
        sop_rot=component.sop_rot,
        sop_inst=component.sop_inst,
        inst_nus=jnp.asarray(component.inst_nus),
        nu_grid=jnp.asarray(component.nu_grid),
        beta_inst=observation_config.beta_inst,
        period_day=float(model_params["period"]),
        Tstar=observation_config.Tstar,
    )
    model_ts = apply_model_pipeline_corrections(
        model_ts,
        subtract_per_exposure_mean=observation_config.subtract_per_exposure_mean,
        apply_sysrem=observation_config.apply_sysrem,
        U=None if component.sysrem is None or component.sysrem.U is None else jnp.asarray(component.sysrem.U),
        V=None if component.sysrem is None or component.sysrem.V is None else jnp.asarray(component.sysrem.V),
        chunked_sysrem=_build_model_chunked_sysrem(component.sysrem),
    )

    return np.asarray(jax.device_get(model_ts))


def _compute_model_timeseries_for_plot(
    *,
    posterior_samples: dict[str, np.ndarray],
    model_params: dict,
    region_config: object,
    component: "SpectroscopicComponentBundle",
    region_sample_prefix: str | None,
    component_sample_prefix: str | None,
    atmo_state: dict | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    try:
        if atmo_state is None:
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_samples,
                region_config=region_config,
                opa_mols=component.opa_mols,
                opa_atoms=component.opa_atoms,
                opa_cias=component.opa_cias,
                nu_grid=component.nu_grid,
                use_median=True,
                sample_prefix=region_sample_prefix,
            )

        model_ts = _synthesize_timeseries_from_atmospheric_state(
            atmo_state=atmo_state,
            model_params=model_params,
            region_config=region_config,
            component=component,
            component_sample_prefix=component_sample_prefix,
        )
        return model_ts, atmo_state
    except Exception as exc:
        print(f"  Warning: failed to build diagnostic spectrum plot data: {exc}")
        return None, atmo_state


@dataclass(frozen=True)
class SpectroscopicComponentBundle:
    name: str
    wav_obs: np.ndarray
    data: np.ndarray
    sigma: np.ndarray
    phase: np.ndarray
    sysrem: SysremInputBundle | None
    inst_nus: np.ndarray
    nu_grid: np.ndarray
    sop_rot: object
    sop_inst: object
    instrument_resolution: float
    opa_cias: dict
    opa_mols: dict
    opa_atoms: dict
    observation_config: object
    observation_inputs: SpectroscopicObservationInputs


@dataclass(frozen=True)
class BandpassConstraintBundle:
    name: str
    observation_config: object
    observation_inputs: BandpassObservationInputs


def _build_spectroscopic_observation_inputs(
    *,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
) -> SpectroscopicObservationInputs:
    return SpectroscopicObservationInputs(
        data=jnp.asarray(data),
        sigma=jnp.asarray(sigma),
        phase=jnp.asarray(phase),
        U=None if sysrem is None or sysrem.U is None else jnp.asarray(sysrem.U),
        V=None if sysrem is None or sysrem.V is None else jnp.asarray(sysrem.V),
        chunked_sysrem=_build_model_chunked_sysrem(sysrem),
    )


def _coerce_model_params(params: dict) -> dict[str, float | None]:
    Kp_low = params.get("Kp_low")
    Kp_high = params.get("Kp_high")
    Mp_upper_3sigma = params.get("M_p_upper_3sigma")
    if Kp_low is not None and Kp_low != Kp_low:
        Kp_low = None
    if Kp_high is not None and Kp_high != Kp_high:
        Kp_high = None
    if Mp_upper_3sigma is not None and Mp_upper_3sigma != Mp_upper_3sigma:
        Mp_upper_3sigma = None

    return {
        "Kp": params.get("Kp", config.DEFAULT_KP),
        "Kp_err": params.get("Kp_err", config.DEFAULT_KP_ERR),
        "Kp_low": Kp_low,
        "Kp_high": Kp_high,
        "RV_abs": params.get("RV_abs", config.DEFAULT_RV_ABS),
        "RV_abs_err": params.get("RV_abs_err", config.DEFAULT_RV_ABS_ERR),
        "R_p": params["R_p"].nominal_value if hasattr(params["R_p"], "nominal_value") else params["R_p"],
        "R_p_err": params["R_p"].std_dev if hasattr(params["R_p"], "std_dev") else config.DEFAULT_RP_ERR,
        "M_p": params["M_p"].nominal_value if hasattr(params["M_p"], "nominal_value") else params["M_p"],
        "M_p_err": params["M_p"].std_dev if hasattr(params["M_p"], "std_dev") else config.DEFAULT_MP_ERR,
        "M_p_upper_3sigma": Mp_upper_3sigma,
        "M_star": (
            params["M_star"].nominal_value
            if ("M_star" in params and hasattr(params["M_star"], "nominal_value"))
            else params.get("M_star")
        ),
        "R_star": params["R_star"].nominal_value if hasattr(params["R_star"], "nominal_value") else params["R_star"],
        "R_star_err": params["R_star"].std_dev if hasattr(params["R_star"], "std_dev") else config.DEFAULT_RSTAR_ERR,
        "T_star": params.get("T_star", config.DEFAULT_TSTAR),
        "logg_star": params.get("logg_star"),
        "Fe_H": params.get("Fe_H"),
        "T_eq": params.get("T_eq"),
        "Tirr_mean": params.get("Tirr_mean", params.get("T_eq")),
        "Tirr_std": params.get("Tirr_std"),
        "a": (
            params["a"].nominal_value
            if ("a" in params and hasattr(params["a"], "nominal_value"))
            else params.get("a")
        ),
        "period": params["period"].nominal_value if hasattr(params["period"], "nominal_value") else params["period"],
    }


def _normalize_retrieval_mode(mode: str) -> str:
    normalized = str(mode).lower().strip()
    if normalized not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    return normalized


def _default_region_name_for_mode(mode: str) -> str:
    if mode == "transmission":
        return "terminator"
    if mode == "emission":
        return "dayside"


def _build_art_for_mode(mode: str) -> object:
    mode = _normalize_retrieval_mode(mode)
    if mode == "transmission":
        art = ArtTransPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=config.NLAYER,
        )
    else:
        art = ArtEmisPure(
            pressure_top=config.PRESSURE_TOP,
            pressure_btm=config.PRESSURE_BTM,
            nlayer=config.NLAYER,
        )
    art.change_temperature_range(config.T_LOW, config.T_HIGH)
    return art


def _normalize_region_name(region_name: str | None, mode: str) -> str:
    if region_name is None:
        return _default_region_name_for_mode(mode)
    normalized = str(region_name).strip()
    return normalized


def _prepare_observed_spectrum_arrays(
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wav_obs = np.asarray(wav_obs)
    data = np.asarray(data)
    sigma = np.asarray(sigma)

    inst_nus = wav2nu(wav_obs, "AA")
    if inst_nus.size > 1 and np.any(np.diff(inst_nus) <= 0):
        sort_idx = np.argsort(inst_nus)
        inst_nus = inst_nus[sort_idx]
        wav_obs = wav_obs[sort_idx]
        if data.ndim == 2:
            data = data[:, sort_idx]
            sigma = sigma[:, sort_idx]
        else:
            data = data[sort_idx]
            sigma = sigma[sort_idx]

    return wav_obs, data, sigma, inst_nus


def _build_component_grid_and_ops(
    wav_obs: np.ndarray,
    instrument_resolution: float,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    inst_nus = wav2nu(wav_obs, "AA")
    nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
        float(np.min(wav_obs)) - config.WAV_MIN_OFFSET,
        float(np.max(wav_obs)) + config.WAV_MAX_OFFSET,
        config.N_SPECTRAL_POINTS,
        unit="AA",
    )
    _preflight_grid_checks(inst_nus, nu_grid)
    sop_rot, sop_inst, _ = setup_spectral_operators(nu_grid, instrument_resolution)
    return inst_nus, nu_grid, sop_rot, sop_inst


def _load_opacity_bundle(
    nu_grid: np.ndarray,
) -> tuple[dict, dict, dict]:
    opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
    opa_mols, _ = load_molecular_opacities(
        config.MOLPATH_HITEMP,
        config.MOLPATH_EXOMOL,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    opa_atoms, _ = load_atomic_opacities(
        config.ATOMIC_SPECIES,
        nu_grid,
        config.OPA_LOAD,
        config.DIFFMODE,
        config.T_LOW,
        config.T_HIGH,
        cutwing=config.PREMODIT_CUTWING,
    )
    return opa_cias, opa_mols, opa_atoms


def _build_primary_spectroscopic_component(
    *,
    name: str,
    mode: str,
    wav_obs: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    phase: np.ndarray,
    sysrem: SysremInputBundle | None,
    instrument_resolution: float,
    nu_grid: np.ndarray,
    inst_nus: np.ndarray,
    sop_rot: object,
    sop_inst: object,
    opa_cias: dict,
    opa_mols: dict,
    opa_atoms: dict,
    region_name: str,
    Tstar: float | None,
    logg_star: float | None,
    metallicity: float | None,
    Mstar: float | None,
    Rstar: float | None,
    phoenix_spectrum_path: str | Path | None,
    phoenix_cache_dir: str | Path | None,
    phase_mode: PhaseMode,
    apply_sysrem: bool,
    radial_velocity_mode: str,
    likelihood_kind: str,
    subtract_per_exposure_mean: bool,
    sample_prefix: str | None = None,
) -> SpectroscopicComponentBundle:
    mode = _normalize_retrieval_mode(mode)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_spectrum_path=phoenix_spectrum_path,
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        radial_velocity_mode=radial_velocity_mode,
        phase_mode=phase_mode,
        likelihood_kind=likelihood_kind,
        subtract_per_exposure_mean=subtract_per_exposure_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=sample_prefix,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _sanitize_name_for_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _infer_tbl_mode(metadata: dict[str, str]) -> str | None:
    spec_type = metadata.get("SPEC_TYPE", "").strip().lower()
    if "eclipse" in spec_type:
        return "emission"
    if "transit" in spec_type or "transmission" in spec_type:
        return "transmission"
    return None


def _convert_unit_to_micron(values: np.ndarray, unit: str) -> np.ndarray:
    unit_norm = unit.lower().strip()
    if "angstrom" in unit_norm or unit_norm == "aa":
        return values / 10000.0
    if unit_norm in {"nm", "nanometer", "nanometers"}:
        return values / 1000.0
    return values


def _combine_scalar_measurements(values: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
    if np.any(mask):
        weights = 1.0 / np.square(sigma[mask])
        value = float(np.sum(values[mask] * weights) / np.sum(weights))
        uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        return value, uncertainty

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite scalar measurements available to combine.")

    value = float(np.mean(finite_values))
    if finite_values.size == 1:
        return value, float("nan")
    return value, float(np.std(finite_values, ddof=1) / np.sqrt(finite_values.size))


def _make_tophat_bandpass(center_micron: float, bandwidth_micron: float, n_samples: int = 64) -> tuple[np.ndarray, np.ndarray]:
    width = float(bandwidth_micron)
    if not np.isfinite(width) or width <= 0:
        width = max(center_micron * 0.05, 0.02)

    half = width / 2.0
    start = max(center_micron - half, 1.0e-6)
    stop = center_micron + half
    wavelength_m = np.linspace(start, stop, int(n_samples), dtype=float) * 1.0e-6
    response = np.ones_like(wavelength_m)
    return wavelength_m, response


def _resolve_lowres_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "lrs" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def _resolve_bandpass_tbl_path(tbl_path: str | Path) -> Path:
    candidate = Path(tbl_path)
    if candidate.exists() or candidate.is_absolute():
        return candidate

    prefixed = config.INPUT_DIR / "phot" / candidate
    if prefixed.exists():
        return prefixed

    return candidate


def make_joint_spectrum_component_from_tbl(tbl_path: str | Path) -> dict[str, Any]:
    tbl_path = _resolve_lowres_tbl_path(tbl_path)
    metadata, _columns, _data_by_col, _units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)
    if mode is None:
        raise ValueError(
            f"Could not infer low-res mode from SPEC_TYPE in {tbl_path}. "
            "Use a NASA .tbl with SPEC_TYPE matching eclipse/transmission."
        )

    return {
        "name": f"lrs_{_sanitize_name_for_id(tbl_path.stem)}",
        "mode": mode,
        "tbl_path": str(tbl_path),
        "data_format": "spectrum",
    }


def make_bandpass_constraints_from_tbl(tbl_path: str | Path) -> list[dict[str, Any]]:
    tbl_path = _resolve_bandpass_tbl_path(tbl_path)
    metadata, _columns, data_by_col, units_by_col = parse_nasa_archive_tbl(tbl_path)
    mode = _infer_tbl_mode(metadata)
    if mode is None:
        raise ValueError(
            f"Could not infer low-res mode from SPEC_TYPE in {tbl_path}. "
            "Use a NASA .tbl with SPEC_TYPE matching eclipse/transmission."
        )

    wav_angstrom, spectrum, sigma, _meta = load_nasa_archive_spectrum(tbl_path, mode=mode)
    unique_wav = np.unique(np.round(wav_angstrom, 6))
    if unique_wav.size >= 5:
        raise ValueError(
            f"{tbl_path} has {unique_wav.size} unique wavelength bins. "
            "Pass it via --joint-spectrum-tbl instead of --bandpass-tbl."
        )

    bandwidth_values = np.full_like(wav_angstrom, np.nan, dtype=float)
    if "BANDWIDTH" in data_by_col:
        raw_bandwidth = np.asarray(
            [
                np.nan if value is None else float(value)
                for value in data_by_col["BANDWIDTH"]
            ],
            dtype=float,
        )
        if raw_bandwidth.size == wav_angstrom.size:
            bandwidth_values = _convert_unit_to_micron(
                raw_bandwidth,
                units_by_col.get("BANDWIDTH", ""),
            )

    grouped_indices: dict[float, list[int]] = {}
    for idx, wav in enumerate(np.round(wav_angstrom, 6)):
        grouped_indices.setdefault(float(wav), []).append(idx)

    instrument = (metadata.get("INSTRUMENT") or metadata.get("INSTRUMENT_NAME") or "").strip()
    facility = (metadata.get("FACILITY") or metadata.get("FACILITY_NAME") or "").strip()
    instrument_text = f"{instrument} {facility}".lower()
    base_name = f"lrs_{_sanitize_name_for_id(tbl_path.stem)}"

    constraints: list[dict[str, Any]] = []
    for group_idx, indices in enumerate(grouped_indices.values(), start=1):
        value, value_sigma = _combine_scalar_measurements(
            spectrum[indices],
            sigma[indices],
        )
        if not np.isfinite(value_sigma) or value_sigma <= 0:
            continue

        name = base_name if len(grouped_indices) == 1 else f"{base_name}_{group_idx}"
        observable = "eclipse_depth" if mode == "emission" else "transit_depth"
        constraint: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "observable": observable,
            "value": value,
            "sigma": value_sigma,
        }

        if "tess" not in instrument_text:
            center_micron = float(np.mean(wav_angstrom[indices]) / 10000.0)
            bandwidth = bandwidth_values[indices]
            finite_bandwidth = bandwidth[np.isfinite(bandwidth) & (bandwidth > 0)]
            bandwidth_micron = (
                float(np.mean(finite_bandwidth))
                if finite_bandwidth.size > 0
                else float("nan")
            )
            wavelength_m, response = _make_tophat_bandpass(center_micron, bandwidth_micron)
            constraint["wavelength_m"] = wavelength_m
            constraint["response"] = response

        constraints.append(constraint)

    return constraints


def _build_atmosphere_regions(
    *,
    model_params: dict[str, float | None],
    primary_mode: str,
    primary_region_name: str,
    primary_art: object,
    observation_configs: list[object],
    default_pt_profile: str,
    default_chemistry_model: str,
    default_fastchem_parameter_file: str | None,
    atmosphere_regions: list[dict[str, Any]] | None,
) -> tuple[tuple[object, ...], dict[str, object]]:
    explicit_specs: dict[str, dict[str, Any]] = {}
    if atmosphere_regions:
        for raw_spec in atmosphere_regions:
            spec = dict(raw_spec)
            name = str(spec.get("name", "")).strip()
            if "mode" in spec and spec["mode"] is not None:
                spec["mode"] = _normalize_retrieval_mode(spec["mode"])
            explicit_specs[name] = spec

    component_modes: dict[str, str] = {}
    region_mol_names: dict[str, set[str]] = {}
    region_atom_names: dict[str, set[str]] = {}
    for observation_config in observation_configs:
        region_name = str(observation_config.region_name)
        region_mode = _normalize_retrieval_mode(observation_config.mode)
        if region_name in component_modes and component_modes[region_name] != region_mode:
            raise ValueError(
                f"Atmosphere region '{region_name}' is referenced by mixed modes "
                f"({component_modes[region_name]} and {region_mode}). Split them into "
                "separate region_name values."
            )
        component_modes[region_name] = region_mode
        region_mol_names.setdefault(region_name, set()).update(observation_config.opa_mols.keys())
        region_atom_names.setdefault(region_name, set()).update(observation_config.opa_atoms.keys())

    region_configs = []
    region_lookup: dict[str, object] = {}
    for region_name in component_modes:
        region_mode = component_modes[region_name]
        spec = explicit_specs.get(region_name, {})
        if spec.get("mode") is not None and spec["mode"] != region_mode:
            raise ValueError(
                f"Atmosphere region '{region_name}' is configured as mode='{spec['mode']}' "
                f"but observation components require mode='{region_mode}'."
            )

        chemistry_name = str(spec.get("chemistry_model", default_chemistry_model))
        chemistry_param_file = spec.get(
            "fastchem_parameter_file",
            default_fastchem_parameter_file,
        )
        composition_solver = _build_composition_solver(
            chemistry_model=chemistry_name,
            fastchem_parameter_file=chemistry_param_file,
        )
        kappa_bounds = spec.get("kappa_ir_cgs_bounds")
        gamma_bounds = spec.get("gamma_bounds")

        art = (
            primary_art
            if region_name == primary_region_name and region_mode == primary_mode
            else _build_art_for_mode(region_mode)
        )
        region_config = build_atmosphere_region_config(
            mode=region_mode,
            art=art,
            mol_names=tuple(sorted(region_mol_names[region_name])),
            atom_names=tuple(sorted(region_atom_names[region_name])),
            pt_profile=str(spec.get("pt_profile", default_pt_profile)),
            T_low=spec.get("T_low"),
            T_high=spec.get("T_high"),
            Tirr_mean=spec.get("Tirr_mean", model_params.get("Tirr_mean")),
            Tirr_std=spec.get("Tirr_std", model_params.get("Tirr_std")),
            Tint_fixed=spec.get("Tint_fixed"),
            kappa_ir_cgs_bounds=None if kappa_bounds is None else tuple(kappa_bounds),
            gamma_bounds=None if gamma_bounds is None else tuple(gamma_bounds),
            composition_solver=composition_solver,
            name=region_name,
            sample_prefix=spec.get("sample_prefix"),
        )
        region_configs.append(region_config)
        region_lookup[region_name] = region_config

    return tuple(region_configs), region_lookup


def _load_joint_spectroscopic_component(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_phoenix_spectrum_path: str | Path | None,
    default_phoenix_cache_dir: str | Path | None,
) -> SpectroscopicComponentBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_component"))
    data_format = str(spec.get("data_format", "spectrum")).lower().strip()
    instrument_resolution = float(spec.get("instrument_resolution", config.get_resolution()))
    radial_velocity_mode = str(spec.get("radial_velocity_mode", "orbital" if data_format == "timeseries" else "none"))
    likelihood_kind = str(spec.get("likelihood_kind", "matched_filter" if data_format == "timeseries" else "gaussian"))
    phase_mode = spec.get("phase_mode", config.DEFAULT_PHASE_MODE if radial_velocity_mode == "orbital" else None)
    apply_sysrem = bool(spec.get("apply_sysrem", data_format == "timeseries" and config.APPLY_SYSREM_DEFAULT))
    subtract_per_exposure_mean = bool(
        spec.get(
            "subtract_per_exposure_mean",
            data_format == "timeseries" and config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT,
        )
    )
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    phoenix_spectrum_path = spec.get("phoenix_spectrum_path", default_phoenix_spectrum_path)
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)

    if "tbl_path" in spec:
        wav_obs, spectrum, uncertainty, _meta = load_nasa_archive_spectrum(
            spec["tbl_path"],
            mode=component_mode,
        )
        data = spectrum[np.newaxis, :]
        sigma = uncertainty[np.newaxis, :]
        phase = np.zeros((1,), dtype=float)
        sysrem = None
    elif all(key in spec for key in ("wav_obs", "data", "sigma")):
        wav_obs = np.asarray(spec["wav_obs"])
        data = np.asarray(spec["data"])
        sigma = np.asarray(spec["sigma"])
        phase = np.asarray(spec.get("phase", np.zeros((1 if data.ndim == 1 else data.shape[0],), dtype=float)))
        if spec.get("sysrem") is not None:
            sysrem_spec = spec["sysrem"]
            if isinstance(sysrem_spec, SysremInputBundle):
                sysrem = sysrem_spec
            elif isinstance(sysrem_spec, dict):
                sysrem = _validate_sysrem_inputs(sysrem_spec, n_exp=(1 if data.ndim == 1 else data.shape[0]))
            else:
                raise TypeError(
                    f"Unsupported sysrem spec type for component '{name}': {type(sysrem_spec)!r}"
                )
        elif spec.get("U") is not None or spec.get("V") is not None:
            if spec.get("U") is None or spec.get("V") is None:
                raise ValueError(
                    f"Joint spectroscopic component '{name}' must provide both U and V together."
                )
            sysrem = _validate_sysrem_inputs(
                {"U": np.asarray(spec["U"]), "V": np.asarray(spec["V"])},
                n_exp=(1 if data.ndim == 1 else data.shape[0]),
            )
        else:
            sysrem = None
    elif "data_dir" in spec:
        data_dir = Path(spec["data_dir"])
        if data_format == "timeseries":
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            phase = _normalize_phase(phase)
            if apply_sysrem:
                sysrem = _validate_sysrem_inputs(
                    _load_sysrem_inputs(data_dir),
                    n_exp=data.shape[0],
                )
            else:
                sysrem = None
        elif data_format == "spectrum":
            suffix = "transmission" if component_mode == "transmission" else "emission"
            wav_obs, spectrum, uncertainty, _ = load_observed_spectrum(
                str(data_dir / f"wavelength_{suffix}.npy"),
                str(data_dir / f"spectrum_{suffix}.npy"),
                str(data_dir / f"uncertainty_{suffix}.npy"),
            )
            data = spectrum[np.newaxis, :]
            sigma = uncertainty[np.newaxis, :]
            phase = np.zeros((1,), dtype=float)
            sysrem = None
        else:
            wav_obs, data, sigma, phase = load_timeseries_data(data_dir)
            sysrem = None

    if apply_sysrem and sysrem is None:
        raise ValueError(
            f"Joint spectroscopic component '{name}' requested SYSREM but no valid "
            "SYSREM inputs were provided."
        )

    wav_obs, data, sigma, inst_nus = _prepare_observed_spectrum_arrays(wav_obs, data, sigma)
    if phase.ndim == 0:
        phase = np.asarray([float(phase)])
    if data_format == "timeseries":
        phase = _normalize_phase(phase)
    elif phase.size == 0:
        phase = np.zeros((1,), dtype=float)
    _preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

    inst_nus_component, nu_grid, sop_rot, sop_inst = _build_component_grid_and_ops(
        wav_obs,
        instrument_resolution,
    )
    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(nu_grid)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_spectrum_path=phoenix_spectrum_path,
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )

    observation_config = build_spectroscopic_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=instrument_resolution,
        inst_nus=inst_nus_component,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        radial_velocity_mode=radial_velocity_mode,
        phase_mode=phase_mode,
        likelihood_kind=likelihood_kind,
        subtract_per_exposure_mean=subtract_per_exposure_mean,
        apply_sysrem=apply_sysrem,
        sample_prefix=name,
    )
    observation_inputs = _build_spectroscopic_observation_inputs(
        data=data,
        sigma=sigma,
        phase=phase,
        sysrem=sysrem,
    )
    return SpectroscopicComponentBundle(
        name=name,
        wav_obs=np.asarray(wav_obs),
        data=np.asarray(data),
        sigma=np.asarray(sigma),
        phase=np.asarray(phase),
        sysrem=sysrem,
        inst_nus=np.asarray(inst_nus_component),
        nu_grid=np.asarray(nu_grid),
        sop_rot=sop_rot,
        sop_inst=sop_inst,
        instrument_resolution=float(instrument_resolution),
        opa_cias=opa_cias,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


def _load_bandpass_constraint(
    spec: dict[str, Any],
    *,
    default_mode: str,
    default_tstar: float | None,
    default_logg_star: float | None,
    default_metallicity: float | None,
    default_mstar: float | None,
    default_rstar: float | None,
    default_semi_major_axis_au: float | None,
    default_phoenix_spectrum_path: str | Path | None,
    default_phoenix_cache_dir: str | Path | None,
) -> BandpassConstraintBundle:
    component_mode = _normalize_retrieval_mode(spec.get("mode", default_mode))
    region_name = _normalize_region_name(spec.get("region_name"), component_mode)
    name = str(spec.get("name", f"{component_mode}_bandpass"))
    observable = str(spec["observable"])
    value = float(spec["value"])
    sigma = float(spec["sigma"])
    photon_weighted = bool(spec.get("photon_weighted", False))
    Tstar = spec.get("Tstar", default_tstar)
    logg_star = spec.get("logg_star", spec.get("phoenix_logg", default_logg_star))
    metallicity = spec.get("Fe_H", spec.get("phoenix_metallicity", default_metallicity))
    Mstar = spec.get("M_star", default_mstar)
    Rstar = spec.get("R_star", default_rstar)
    phoenix_spectrum_path = spec.get("phoenix_spectrum_path", default_phoenix_spectrum_path)
    phoenix_cache_dir = spec.get("phoenix_cache_dir", default_phoenix_cache_dir)
    include_reflection = bool(spec.get("include_reflection", False))
    semi_major_axis_au = spec.get("semi_major_axis_au", default_semi_major_axis_au)
    if semi_major_axis_au is not None:
        semi_major_axis_au = float(semi_major_axis_au)
    geometric_albedo_bounds_raw = spec.get("geometric_albedo_bounds")
    geometric_albedo_bounds = (
        None
        if geometric_albedo_bounds_raw is None
        else tuple(float(x) for x in geometric_albedo_bounds_raw)
    )
    model_sigma_raw = spec.get("model_sigma")
    model_sigma = None if model_sigma_raw is None else float(model_sigma_raw)
    model_sigma_bounds_raw = spec.get("model_sigma_bounds")
    model_sigma_bounds = (
        None
        if model_sigma_bounds_raw is None
        else tuple(float(x) for x in model_sigma_bounds_raw)
    )

    if "wavelength_m" in spec and "response" in spec:
        wavelength_m = np.asarray(spec["wavelength_m"], dtype=float)
        response = np.asarray(spec["response"], dtype=float)
    else:
        bandpass_path = spec.get("bandpass_path")
        wavelength_m, response, _used_path = load_tess_bandpass(
            bandpass_path,
            download_if_missing=bool(spec.get("download_bandpass", True)),
        )

    wavelength_angstrom = np.asarray(wavelength_m, dtype=float) * 1.0e10
    if "nu_grid" in spec:
        nu_grid = np.asarray(spec["nu_grid"], dtype=float)
    else:
        nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
            float(np.min(wavelength_angstrom)) - config.WAV_MIN_OFFSET,
            float(np.max(wavelength_angstrom)) + config.WAV_MAX_OFFSET,
            config.N_SPECTRAL_POINTS,
            unit="AA",
        )

    opa_cias, opa_mols, opa_atoms = _load_opacity_bundle(nu_grid)
    stellar_surface_flux = _load_phoenix_surface_flux_on_grid(
        phoenix_spectrum_path=phoenix_spectrum_path,
        phoenix_cache_dir=phoenix_cache_dir,
        nu_grid=nu_grid,
        mode=component_mode,
        component_name=name,
        Tstar=Tstar,
        logg_star=logg_star,
        metallicity=metallicity,
        Mstar=Mstar,
        Rstar=Rstar,
    )
    observation_config = build_bandpass_observation_config(
        name=name,
        region_name=region_name,
        mode=component_mode,
        opa_mols=opa_mols,
        opa_atoms=opa_atoms,
        opa_cias=opa_cias,
        nu_grid=nu_grid,
        wavelength_m=wavelength_m,
        response=response,
        observable=observable,
        photon_weighted=photon_weighted,
        Tstar=Tstar,
        stellar_surface_flux=stellar_surface_flux,
        include_reflection=include_reflection,
        semi_major_axis_au=semi_major_axis_au,
        geometric_albedo_bounds=geometric_albedo_bounds,
        model_sigma=model_sigma,
        model_sigma_bounds=model_sigma_bounds,
        sample_prefix=name,
    )
    observation_inputs = BandpassObservationInputs(
        value=jnp.asarray(value),
        sigma=jnp.asarray(sigma),
    )
    return BandpassConstraintBundle(
        name=name,
        observation_config=observation_config,
        observation_inputs=observation_inputs,
    )


class _StepTimer:
    def __init__(self, label: str):
        self.label = label
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = perf_counter() - self.start
        status = "failed after" if exc_type is not None else "completed in"
        print(f"  {self.label} {status} {elapsed:.2f}s")
        return False


def run_retrieval(
    mode: str = "transmission",
    epoch: str | None = None,
    data_format: str = config.DEFAULT_DATA_FORMAT,
    skip_svi: bool = False,
    svi_only: bool = False,
    no_plots: bool = False,
    pt_profile: str | None = None,
    phase_mode: PhaseMode = "global",
    chemistry_model: str | None = None,
    fastchem_parameter_file: str | None = None,
    compute_contribution: bool = True,
    seed: int = 42,
    wav_obs: np.ndarray | None = None,
    data: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    U: np.ndarray | None = None,
    V: np.ndarray | None = None,
    sysrem_inputs: SysremInputBundle | None = None,
    joint_spectra: list[dict[str, Any]] | None = None,
    bandpass_constraints: list[dict[str, Any]] | None = None,
    atmosphere_regions: list[dict[str, Any]] | None = None,
    phoenix_spectrum_path: str | Path | None = None,
    phoenix_cache_dir: str | Path | None = None,
) -> None:
    retrieval_start = perf_counter()
    mode = _normalize_retrieval_mode(mode)
    if pt_profile is None:
        raise ValueError("pt_profile must be passed explicitly.")
    if chemistry_model is None:
        raise ValueError("chemistry_model must be passed explicitly.")

    # Create timestamped output directory
    base_dir = config.DIR_SAVE or config.get_output_dir()
    output_dir = config.create_timestamped_dir(base_dir)
    print(f"\nOutput directory: {output_dir}")

    # Save run configuration
    config.save_run_config(
        output_dir=output_dir,
        mode=mode,
        pt_profile=pt_profile,
        skip_svi=skip_svi,
        svi_only=svi_only,
        seed=seed,
        chemistry_model=chemistry_model,
        epoch=epoch,
        phoenix_spectrum_path=None if phoenix_spectrum_path is None else str(phoenix_spectrum_path),
        phoenix_cache_dir=None if phoenix_cache_dir is None else str(phoenix_cache_dir),
    )

    # Get planet parameters
    params = config.get_params()
    print(f"\nTarget: {config.PLANET} ({config.EPHEMERIS})")

    apply_sysrem = bool(config.APPLY_SYSREM_DEFAULT and data_format == "timeseries")

    primary_sysrem: SysremInputBundle | None = sysrem_inputs
    
    print("\n[1/7] Loading time-series data...")
    with _StepTimer("Step 1/7"):
        if epoch:
            print(f"  Using epoch: {epoch}")
        if any(val is not None for val in (wav_obs, data, sigma, phase)):
            phase = _normalize_phase(phase)
            print(f"  Using provided data: {data.shape[0]} exposures x {data.shape[1]} wavelengths")
            print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
            if apply_sysrem:
                if primary_sysrem is None:
                    if U is None or V is None:
                        raise ValueError(
                            "apply_sysrem=True requires either sysrem_inputs or both U and V "
                            "when providing wav_obs/data/sigma/phase directly."
                        )
                    primary_sysrem = _validate_sysrem_inputs(
                        {"U": U, "V": V},
                        n_exp=data.shape[0],
                    )
                print(f"  Using provided SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
        else:
            resolved_data_dir = config.get_data_dir(epoch=epoch)
            data_paths = (
                config.get_transmission_paths(epoch=epoch) if mode == "transmission"
                else config.get_emission_paths(epoch=epoch)
            )

            if data_format == "timeseries":
                wav_obs, data, sigma, phase = load_timeseries_data(resolved_data_dir)
                phase = _normalize_phase(phase)
                print(f"  Loaded {data.shape[0]} exposures x {data.shape[1]} wavelengths")
                print(f"  Phase range: {phase.min():.3f} - {phase.max():.3f}")
                if apply_sysrem:
                    primary_sysrem = _validate_sysrem_inputs(
                        _load_sysrem_inputs(resolved_data_dir),
                        n_exp=data.shape[0],
                    )
                    print(f"  Loaded SYSREM auxiliaries: {_describe_sysrem_inputs(primary_sysrem)}")
            elif data_format == "spectrum":
                wav_obs, spectrum, uncertainty, inst_nus = load_observed_spectrum(
                    str(data_paths["wavelength"]),
                    str(data_paths["spectrum"]),
                    str(data_paths["uncertainty"]),
                )
                data = spectrum[np.newaxis, :]
                sigma = uncertainty[np.newaxis, :]
                phase = np.array([0.0])
                print(f"  Loaded single spectrum with {len(wav_obs)} points")
                if apply_sysrem:
                    raise ValueError(
                        "apply_sysrem=True with data_format='spectrum' requires SYSREM "
                        "auxiliaries tied to time-series exposures, which are unavailable "
                        "for single-spectrum input. Use data_format='timeseries' or set "
                        "APPLY_SYSREM_DEFAULT=False."
                    )
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. "
                    "Choose from {'timeseries', 'spectrum'}."
                )

        print(f"  Wavelength range: {wav_obs.min():.1f} - {wav_obs.max():.1f} Angstroms")

        # Convert to wavenumber
        inst_nus = wav2nu(wav_obs, "AA")
        # Ensure wavenumber grid and data are in ascending order
        if inst_nus.size > 1 and np.any(np.diff(inst_nus) <= 0):
            sort_idx = np.argsort(inst_nus)
            inst_nus = inst_nus[sort_idx]
            wav_obs = wav_obs[sort_idx]
            if data.ndim == 2:
                data = data[:, sort_idx]
                sigma = sigma[:, sort_idx]
            else:
                data = data[sort_idx]
                sigma = sigma[sort_idx]

        _preflight_spectrum_checks(wav_obs, data, sigma, phase, inst_nus)

    # Setup instrumental resolution
    print("\n[2/7] Setting up instrumental resolution...")
    with _StepTimer("Step 2/7"):
        Rinst = config.get_resolution()
        print(f"  Instrument resolving power: R = {Rinst:.0f}")

    # Setup wavenumber grid
    print("\n[3/7] Building wavenumber grid...")
    with _StepTimer("Step 3/7"):
        wav_min, wav_max = config.get_wavelength_range()
        nu_grid, _wav_grid, _res_high = setup_wavenumber_grid(
            wav_min - config.WAV_MIN_OFFSET,
            wav_max + config.WAV_MAX_OFFSET,
            config.N_SPECTRAL_POINTS,
            unit="AA",
        )
        _preflight_grid_checks(inst_nus, nu_grid)

        sop_rot, sop_inst, _ = setup_spectral_operators(nu_grid, Rinst)
        print("  Spectral operators initialized")

    # Setup primary atmospheric RT geometry
    print("\n[4/7] Initializing primary atmospheric RT...")
    with _StepTimer("Step 4/7"):
        primary_art = _build_art_for_mode(mode)
        print(f"  {config.NLAYER} atmospheric layers")
        print(f"  Pressure range: {config.PRESSURE_TOP:.1e} - {config.PRESSURE_BTM:.1e} bar")
        print(f"  Temperature range: {config.T_LOW:.0f} - {config.T_HIGH:.0f} K")

    # Load opacities
    print("\n[5/7] Loading opacities...")
    with _StepTimer("Step 5/7"):
        opa_cias = setup_cia_opacities(config.CIA_PATHS, nu_grid)
        n_cia = sum(1 for cia in opa_cias.values() if not getattr(cia, "_is_dummy", False))
        if n_cia == 0:
            print("  Loaded 0 CIA sources (no overlap with nu_grid)")
        else:
            print(f"  Loaded {n_cia} CIA sources")

        opa_mols, _molmass_arr = load_molecular_opacities(
            config.MOLPATH_HITEMP,
            config.MOLPATH_EXOMOL,
            nu_grid,
            config.OPA_LOAD,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
        )
        print(f"  Loaded {len(opa_mols)} molecular species: {list(opa_mols.keys())}")

        # Load atomic opacities (optional, uses Kurucz with auto-download)
        opa_atoms, _atommass_arr = load_atomic_opacities(
            config.ATOMIC_SPECIES,
            nu_grid,
            config.OPA_LOAD,
            config.DIFFMODE,
            config.T_LOW,
            config.T_HIGH,
            cutwing=config.PREMODIT_CUTWING,
        )
        if opa_atoms:
            print(f"  Loaded {len(opa_atoms)} atomic species: {list(opa_atoms.keys())}")

    print(f"\n[6/7] Building retrieval model (primary={mode}, default P-T={pt_profile})...")
    print(f"  Chemistry model: {chemistry_model}")
    with _StepTimer("Step 6/7"):
        model_params = _coerce_model_params(params)
        primary_region_name = _default_region_name_for_mode(mode)
        if mode == "emission":
            if phoenix_spectrum_path is None:
                print("  PHOENIX stellar spectrum: chromatic auto-fetch")
                print(f"  Processed PHOENIX cache: {_normalize_phoenix_cache_dir(phoenix_cache_dir)}")
            else:
                print(f"  PHOENIX stellar spectrum: {phoenix_spectrum_path}")

        shared_system = build_shared_system_config(params=model_params)

        primary_is_timeseries = (
            np.asarray(phase).size > 1
            or bool(apply_sysrem)
            or (np.asarray(data).ndim == 2 and np.asarray(data).shape[0] > 1)
        )
        primary_radial_velocity_mode = "orbital" if primary_is_timeseries else "none"
        primary_likelihood_kind = "matched_filter" if primary_is_timeseries else "gaussian"
        primary_subtract_mean = config.SUBTRACT_PER_EXPOSURE_MEAN_DEFAULT if primary_is_timeseries else False
        primary_sample_prefix = (
            "spectroscopy"
            if (joint_spectra or bandpass_constraints)
            else None
        )

        primary_component = _build_primary_spectroscopic_component(
            name="spectroscopy",
            mode=mode,
            wav_obs=wav_obs,
            data=data,
            sigma=sigma,
            phase=phase,
            sysrem=primary_sysrem,
            instrument_resolution=Rinst,
            nu_grid=nu_grid,
            inst_nus=inst_nus,
            sop_rot=sop_rot,
            sop_inst=sop_inst,
            opa_cias=opa_cias,
            opa_mols=opa_mols,
            opa_atoms=opa_atoms,
            region_name=primary_region_name,
            Tstar=model_params["T_star"],
            logg_star=model_params["logg_star"],
            metallicity=model_params["Fe_H"],
            Mstar=model_params["M_star"],
            Rstar=model_params["R_star"],
            phoenix_spectrum_path=phoenix_spectrum_path,
            phoenix_cache_dir=phoenix_cache_dir,
            phase_mode=phase_mode,
            apply_sysrem=apply_sysrem,
            radial_velocity_mode=primary_radial_velocity_mode,
            likelihood_kind=primary_likelihood_kind,
            subtract_per_exposure_mean=primary_subtract_mean,
            sample_prefix=primary_sample_prefix,
        )
        observation_configs: list[object] = [primary_component.observation_config]
        observations_payload: dict[str, object] = {
            primary_component.name: primary_component.observation_inputs
        }

        auxiliary_components: list[SpectroscopicComponentBundle] = []
        if joint_spectra:
            for spec in joint_spectra:
                component = _load_joint_spectroscopic_component(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_phoenix_spectrum_path=phoenix_spectrum_path,
                    default_phoenix_cache_dir=phoenix_cache_dir,
                )
                if component.name in observations_payload:
                    raise ValueError(f"Duplicate joint component name: {component.name}")
                auxiliary_components.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        scalar_constraints: list[BandpassConstraintBundle] = []
        if bandpass_constraints:
            for spec in bandpass_constraints:
                component = _load_bandpass_constraint(
                    spec,
                    default_mode=mode,
                    default_tstar=model_params["T_star"],
                    default_logg_star=model_params["logg_star"],
                    default_metallicity=model_params["Fe_H"],
                    default_mstar=model_params["M_star"],
                    default_rstar=model_params["R_star"],
                    default_semi_major_axis_au=model_params["a"],
                    default_phoenix_spectrum_path=phoenix_spectrum_path,
                    default_phoenix_cache_dir=phoenix_cache_dir,
                )
                if component.name in observations_payload:
                    raise ValueError(f"Duplicate joint component name: {component.name}")
                scalar_constraints.append(component)
                observation_configs.append(component.observation_config)
                observations_payload[component.name] = component.observation_inputs

        atmosphere_region_configs, atmosphere_region_lookup = _build_atmosphere_regions(
            model_params=model_params,
            primary_mode=mode,
            primary_region_name=primary_region_name,
            primary_art=primary_art,
            observation_configs=observation_configs,
            default_pt_profile=pt_profile,
            default_chemistry_model=chemistry_model,
            default_fastchem_parameter_file=fastchem_parameter_file,
            atmosphere_regions=atmosphere_regions,
        )
        joint_model = create_joint_retrieval_model(
            shared_system=shared_system,
            atmosphere_regions=atmosphere_region_configs,
            observations=tuple(observation_configs),
        )
        model_c = joint_model
        model_inputs = {"observations": observations_payload}
        primary_region_config = atmosphere_region_lookup[primary_region_name]
        primary_pt_profile = primary_region_config.pt_profile
        primary_component_config = primary_component.observation_config
        primary_region_sample_prefix = primary_region_config.sample_prefix
        primary_component_sample_prefix = primary_component_config.sample_prefix

        component_names = [primary_component.name]
        component_names.extend(component.name for component in auxiliary_components)
        component_names.extend(component.name for component in scalar_constraints)
        print(
            f"  Joint model created with {len(component_names)} component(s): "
            f"{', '.join(component_names)}"
        )
        print(
            "  Atmosphere regions: "
            + ", ".join(
                f"{region_config.name} "
                f"[{next(cfg.mode for cfg in observation_configs if cfg.region_name == region_config.name)}]"
                for region_config in atmosphere_region_configs
            )
        )

    # Run inference
    print("\n[7/7] Running Bayesian inference...")
    rng_key = random.PRNGKey(seed)

    init_strategy = init_to_median(num_samples=config.INIT_TO_MEDIAN_SAMPLES)
    svi_params: dict | None = None
    svi_guide: object | None = None
    svi_losses: np.ndarray | None = None

    with _StepTimer("Step 7/7"):
        if not skip_svi:
            svi_lr_message = f"  SVI warm-up: {config.SVI_NUM_STEPS} steps, LR={config.SVI_LEARNING_RATE}"
            if config.SVI_LR_DECAY_STEPS is not None and config.SVI_LR_DECAY_RATE is not None:
                svi_lr_message += (
                    " with exponential decay "
                    f"(steps={config.SVI_LR_DECAY_STEPS}, rate={config.SVI_LR_DECAY_RATE})"
                )
            print(svi_lr_message)
            rng_key, rng_key_ = random.split(rng_key)
            svi_params, svi_losses, init_strategy, _, svi_guide = run_svi(
                model_c,
                rng_key_,
                model_inputs=model_inputs,
                Mp_mean=model_params["M_p"],
                Mp_std=model_params["M_p_err"],
                Mp_upper_3sigma=model_params.get("M_p_upper_3sigma"),
                Rp_mean=model_params["R_p"],
                Rp_std=model_params["R_p_err"],
                Rstar_mean=model_params["R_star"],
                Rstar_std=model_params["R_star_err"],
                output_dir=str(output_dir),
                num_steps=config.SVI_NUM_STEPS,
                lr=config.SVI_LEARNING_RATE,
                lr_decay_steps=config.SVI_LR_DECAY_STEPS,
                lr_decay_rate=config.SVI_LR_DECAY_RATE,
            )

            if svi_only:
                if not no_plots:
                    print("  Generating corner plots from SVI posterior...")
                    rng_key, rng_key_plot = random.split(rng_key)
                    svi_samples_for_plots = _sample_svi_posterior(
                        guide=svi_guide,
                        params=svi_params,
                        rng_key=rng_key_plot,
                        num_samples=max(100, int(config.MCMC_NUM_SAMPLES)),
                    )
                    save_retrieval_corner_plots(
                        output_dir=str(output_dir),
                        svi_samples=svi_samples_for_plots,
                    )

                    if svi_losses is not None:
                        plot_svi_loss(
                            np.asarray(jax.device_get(svi_losses)),
                            os.path.join(output_dir, "svi_loss.png"),
                        )

                    if svi_samples_for_plots is not None:
                        try:
                            plot_temperature_profile(
                                posterior_samples=svi_samples_for_plots,
                                art=primary_region_config.art,
                                save_path=os.path.join(output_dir, "temperature_profile.png"),
                                pt_profile=primary_pt_profile,
                                sample_prefix=primary_region_sample_prefix,
                                Tint_fixed=primary_region_config.Tint_fixed,
                            )
                        except Exception as exc:
                            print(
                                "  Skipping temperature profile plot for SVI samples: "
                                f"{exc}"
                            )

                        obs_mean, obs_err = _summarize_observed_spectrum(data, sigma)
                        wav_obs_nm = np.asarray(wav_obs) / 10.0

                        svi_model_ts, _ = _compute_model_timeseries_for_plot(
                            posterior_samples=svi_samples_for_plots,
                            model_params=model_params,
                            region_config=primary_region_config,
                            component=primary_component,
                            region_sample_prefix=primary_region_sample_prefix,
                            component_sample_prefix=primary_component_sample_prefix,
                        )

                        if svi_model_ts is not None:
                            svi_line = np.mean(np.asarray(svi_model_ts), axis=0)
                            if mode == "transmission":
                                plot_transmission_spectrum(
                                    wavelength_nm=wav_obs_nm,
                                    rp_obs=obs_mean,
                                    rp_err=obs_err,
                                    rp_hmc=np.atleast_2d(svi_line),
                                    rp_svi=svi_line,
                                    save_path=os.path.join(output_dir, "transmission_spectrum.png"),
                                )
                            else:
                                plot_emission_spectrum(
                                    wavelength_nm=wav_obs_nm,
                                    fp_obs=obs_mean,
                                    fp_err=obs_err,
                                    fp_hmc=np.atleast_2d(svi_line),
                                    fp_svi=svi_line,
                                    save_path=os.path.join(output_dir, "emission_spectrum.png"),
                                )
                print("  SVI complete (svi_only=True); skipping MCMC.")
                return

        print(f"\n  Running HMC-NUTS sampling...")
        print(f"  Warmup: {config.MCMC_NUM_WARMUP}, Samples: {config.MCMC_NUM_SAMPLES}")
        print(f"  Chains: {config.MCMC_NUM_CHAINS}")
        print(f"  Chain method: {config.MCMC_CHAIN_METHOD}")

        _validate_mcmc_device_layout(
            num_chains=config.MCMC_NUM_CHAINS,
            chain_method=config.MCMC_CHAIN_METHOD,
            require_gpu_per_chain=config.MCMC_REQUIRE_GPU_PER_CHAIN,
        )

        kernel = NUTS(
            model_c,
            max_tree_depth=config.MCMC_MAX_TREE_DEPTH,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=config.MCMC_NUM_WARMUP,
            num_samples=config.MCMC_NUM_SAMPLES,
            num_chains=config.MCMC_NUM_CHAINS,
            chain_method=config.MCMC_CHAIN_METHOD,
        )

        rng_key, rng_key_ = random.split(rng_key)
        mcmc.run(
            rng_key_,
            **model_inputs,
        )
    
    mcmc.print_summary()
    
    # Save results
    with open(os.path.join(output_dir, "mcmc_summary.txt"), "w") as f:
        with redirect_stdout(f):
            mcmc.print_summary()
    
    posterior_sample = mcmc.get_samples()
    jnp.savez(os.path.join(output_dir, "posterior_sample"), **posterior_sample)

    posterior_np: dict[str, np.ndarray] | None = None
    svi_samples_for_plots: dict[str, np.ndarray] | None = None

    if not no_plots:
        print("\n  Generating corner plots...")
        posterior_np = {
            name: np.asarray(jax.device_get(values))
            for name, values in posterior_sample.items()
        }

        if svi_params is not None and svi_guide is not None:
            n_hmc_samples = max(100, int(config.MCMC_NUM_SAMPLES))
            if posterior_np:
                first_site = next(iter(posterior_np))
                n_hmc_samples = int(np.asarray(posterior_np[first_site]).shape[0])

            rng_key, rng_key_plot = random.split(rng_key)
            svi_samples_for_plots = _sample_svi_posterior(
                guide=svi_guide,
                params=svi_params,
                rng_key=rng_key_plot,
                num_samples=n_hmc_samples,
            )

        save_retrieval_corner_plots(
            output_dir=str(output_dir),
            hmc_samples=posterior_np,
            svi_samples=svi_samples_for_plots,
        )

        if svi_losses is not None:
            plot_svi_loss(
                np.asarray(jax.device_get(svi_losses)),
                os.path.join(output_dir, "svi_loss.png"),
            )

        try:
            plot_temperature_profile(
                posterior_samples=posterior_np,
                art=primary_region_config.art,
                save_path=os.path.join(output_dir, "temperature_profile.png"),
                pt_profile=primary_pt_profile,
                sample_prefix=primary_region_sample_prefix,
                Tint_fixed=primary_region_config.Tint_fixed,
            )
        except Exception as exc:
            print(
                "  Skipping temperature profile plot for HMC samples: "
                f"{exc}"
            )

    atmo_state = None
    if compute_contribution or not no_plots:
        print("\n  Computing atmospheric state from posterior...")

        if posterior_np is None:
            posterior_np = {
                name: np.asarray(jax.device_get(values))
                for name, values in posterior_sample.items()
            }

        try:
            atmo_state = compute_atmospheric_state_from_posterior(
                posterior_samples=posterior_np,
                region_config=primary_region_config,
                opa_mols=primary_component.opa_mols,
                opa_atoms=primary_component.opa_atoms,
                opa_cias=primary_component.opa_cias,
                nu_grid=primary_component.nu_grid,
                use_median=True,
                sample_prefix=primary_region_sample_prefix,
            )
        except Exception as exc:
            if compute_contribution:
                raise
            print(
                "  Warning: unable to compute atmospheric state; "
                f"skipping spectrum diagnostics. ({exc})"
            )

    if not no_plots and atmo_state is not None:
        print("  Plotting fitted spectrum diagnostics...")
        wav_obs_nm = np.asarray(wav_obs) / 10.0
        obs_mean, obs_err = _summarize_observed_spectrum(data, sigma)

        hmc_model_ts, atmo_state = _compute_model_timeseries_for_plot(
            posterior_samples=posterior_np,
            model_params=model_params,
            region_config=primary_region_config,
            component=primary_component,
            region_sample_prefix=primary_region_sample_prefix,
            component_sample_prefix=primary_component_sample_prefix,
            atmo_state=atmo_state,
        )

        svi_model_ts = None
        if svi_samples_for_plots is not None:
            svi_model_ts, _ = _compute_model_timeseries_for_plot(
                posterior_samples=svi_samples_for_plots,
                model_params=model_params,
                region_config=primary_region_config,
                component=primary_component,
                region_sample_prefix=primary_region_sample_prefix,
                component_sample_prefix=primary_component_sample_prefix,
            )

        if hmc_model_ts is not None or svi_model_ts is not None:
            hmc_plot = hmc_model_ts
            if hmc_plot is None and svi_model_ts is not None:
                hmc_plot = np.atleast_2d(np.mean(np.asarray(svi_model_ts), axis=0))

            if hmc_plot is not None:
                svi_line = np.mean(np.asarray(hmc_plot), axis=0)
                if svi_model_ts is not None:
                    svi_line = np.mean(np.asarray(svi_model_ts), axis=0)

                if mode == "transmission":
                    plot_transmission_spectrum(
                        wavelength_nm=wav_obs_nm,
                        rp_obs=obs_mean,
                        rp_err=obs_err,
                        rp_hmc=np.asarray(hmc_plot),
                        rp_svi=np.asarray(svi_line),
                        save_path=os.path.join(output_dir, "transmission_spectrum.png"),
                    )
                else:
                    plot_emission_spectrum(
                        wavelength_nm=wav_obs_nm,
                        fp_obs=obs_mean,
                        fp_err=obs_err,
                        fp_hmc=np.asarray(hmc_plot),
                        fp_svi=np.asarray(svi_line),
                        save_path=os.path.join(output_dir, "emission_spectrum.png"),
                    )

    if compute_contribution and atmo_state is not None:
        # Save atmospheric state
        np.savez(
            os.path.join(output_dir, "atmospheric_state.npz"),
            dtau=np.array(atmo_state['dtau']),
            Tarr=np.array(atmo_state['Tarr']),
            pressure=np.array(atmo_state['pressure']),
            dParr=np.array(atmo_state['dParr']),
            mmw=np.array(atmo_state['mmw']),
            vmrH2=np.array(atmo_state['vmrH2']),
            vmrHe=np.array(atmo_state['vmrHe']),
        )

        # Plot contribution function
        if not no_plots:
            print("  Plotting contribution function...")

            # Total contribution function
            plot_contribution_function(
                nu_grid=np.array(primary_component.nu_grid),
                dtau=np.array(atmo_state['dtau']),
                Tarr=np.array(atmo_state['Tarr']),
                pressure=np.array(atmo_state['pressure']),
                dParr=np.array(atmo_state['dParr']),
                save_path=os.path.join(output_dir, "contribution_function.pdf"),
                wavelength_unit="AA",
                title=f"{config.PLANET} Contribution Function ({mode})",
            )

            # Per-species contribution functions (if available)
            if atmo_state['dtau_per_species']:
                dtau_per_species_np = {
                    k: np.array(v) for k, v in atmo_state['dtau_per_species'].items()
                }

                plot_contribution_per_species(
                    nu_grid=np.array(primary_component.nu_grid),
                    dtau_per_species=dtau_per_species_np,
                    Tarr=np.array(atmo_state['Tarr']),
                    pressure=np.array(atmo_state['pressure']),
                    dParr=np.array(atmo_state['dParr']),
                    save_path=os.path.join(output_dir, "contribution_per_species.pdf"),
                    wavelength_unit="AA",
                )

                # Combined plot
                plot_contribution_combined(
                    nu_grid=np.array(primary_component.nu_grid),
                    dtau=np.array(atmo_state['dtau']),
                    dtau_per_species=dtau_per_species_np,
                    Tarr=np.array(atmo_state['Tarr']),
                    pressure=np.array(atmo_state['pressure']),
                    dParr=np.array(atmo_state['dParr']),
                    save_path=os.path.join(output_dir, "contribution_combined.pdf"),
                    wavelength_unit="AA",
                )

            print(f"  Contribution function plots saved to {output_dir}/")
    
    print("\n" + "="*70)
    print("RETRIEVAL COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"Total runtime: {perf_counter() - retrieval_start:.2f}s")
    print("="*70)


if __name__ == "__main__":
    raise RuntimeError(
        "Direct __main__ execution no longer provides a chemistry default. "
        "Import run_retrieval(...) and pass chemistry_model explicitly."
    )
