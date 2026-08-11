"""Read-only, raw-exposure calibration of dataset-specific arm-edge trims.

This module is intentionally a diagnostics helper, not a preparation entry
point.  It reads the same raw PEPSI products as the canonical preparation
commands, reconstructs the ``timeseries`` and ``collapse_source`` branches in
memory, tests explicit edge-trim candidates, and writes only figures plus
CSV/JSON/Markdown diagnostics.  It never writes prepared ``.npy`` products.

The selection rule is threshold-first and fail-closed on hard validity checks:

* a 0.1 nm grid spans 0--20 nm on each side;
* every adjacent coarse interval whose joint acceptance state or rejection
  reason changes is resampled at 0.02 nm;
* every selected width is one of those explicitly tested coarse or refined
  candidates;
* candidates where both product branches pass the configured edge/interior
  threshold are preferred;
* if one side has no threshold-passing candidate, its hard-valid candidate
  with the lowest worst product-branch ratio is used as a pre-SYSREM fallback;
* retained telluric edges are compared with retained telluric baselines;
* no protected diagnostic line may be removed; and
* when finalist SYSREM is enabled, the same checks must pass after rerunning
  SYSREM on the zero-trim and candidate grids.

Up to 256 valid left/right pairs are tested with fresh SYSREM, ordered by least
total wavelength removed.  A fallback candidate must still pass the normal
post-SYSREM threshold.  If no tested pair passes, the manifest records a failure
and no boundary is selected.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from plotting.style import configure_matplotlib, save_figure_pdf

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
from exojax.database.core_atom.io import air_to_vac

import config
import config_utils
from config_utils import get_data_patterns, get_params, resolve_transit_midpoint
from dataio.collapse_transmission_timeseries_to_1d import (
    active_transit_mask,
    build_out_of_transit_residuals,
    compute_contact_phases,
    do_sysrem,
    get_bjd_tdb,
    get_ephemeris_epoch_bjd_tdb,
    get_orbital_phase,
    get_sysrem_deep_mask,
    get_telluric_edge_mask,
    subtract_median_spectrum,
)
from dataio.exposure_selection import (
    ScienceExposureSelection,
    select_science_exposures,
)
from dataio.hrs_preparation import resolve_stellar_velocity_correction
from dataio.prepare_retrieval_timeseries import transmission_phase_selection_mask


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "diagnostics" / "edge_trim_calibration"
ARMS = ("blue", "red")
PRODUCTS = ("timeseries", "collapse_source")
SIDES = ("left", "right")
ALLOWED_ARTIFACT_SUFFIXES = {".pdf", ".csv", ".json", ".md"}

# These are the explicit line-preservation sentinels used by the processing
# diagnostics.  They protect the broad, repeatedly inspected features without
# treating every illustrative atlas transition as an untouchable boundary.
_AIR_PROTECTED_EDGE_LINES = (
    {"label": "H beta", "arm": "blue", "rest_air_A": 4861.333},
    {"label": "Mg I b1", "arm": "blue", "rest_air_A": 5167.322},
    {"label": "Mg I b2", "arm": "blue", "rest_air_A": 5172.684},
    {"label": "Mg I b3", "arm": "blue", "rest_air_A": 5183.604},
    {"label": "H alpha", "arm": "red", "rest_air_A": 6562.790},
    {"label": "Li I 6708", "arm": "red", "rest_air_A": 6707.840},
)
PROTECTED_EDGE_LINES = tuple(
    {
        **line,
        "rest_vacuum_A": float(
            air_to_vac(np.asarray([line["rest_air_A"]]))[0]
        ),
        "rest_wavelength_medium": "vacuum",
        "provenance_wavelength_medium": "air",
    }
    for line in _AIR_PROTECTED_EDGE_LINES
)


@dataclass(frozen=True)
class CalibrationSettings:
    """Numerical controls recorded verbatim in every calibration manifest."""

    coarse_min_nm: float = 0.0
    coarse_max_nm: float = 20.0
    coarse_step_nm: float = 0.1
    refinement_step_nm: float = 0.02
    eval_width_nm: float = 3.0
    baseline_exclude_nm: float = 8.0
    accept_ratio: float = 1.35
    protected_line_pad_nm: float = 0.25
    minimum_eval_columns: int = 20
    maximum_finalists: int = 256
    run_sysrem_finalists: bool = True
    use_molecfit_for_red: bool = True
    apply_stellar_velocity_correction: bool = True

    def validated(self) -> "CalibrationSettings":
        grid_controls = (
            self.coarse_min_nm,
            self.coarse_max_nm,
            self.coarse_step_nm,
            self.refinement_step_nm,
        )
        if any(not math.isfinite(float(value)) for value in grid_controls):
            raise ValueError("Every candidate-grid control must be finite.")
        if float(self.coarse_min_nm) != 0.0:
            raise ValueError("coarse_min_nm must be the explicit zero-trim control.")
        if float(self.coarse_max_nm) <= float(self.coarse_min_nm):
            raise ValueError("coarse_max_nm must be greater than coarse_min_nm.")
        if float(self.coarse_step_nm) <= 0.0:
            raise ValueError("coarse_step_nm must be positive.")
        if not 0.0 < float(self.refinement_step_nm) < float(self.coarse_step_nm):
            raise ValueError(
                "refinement_step_nm must be positive and smaller than coarse_step_nm."
            )
        span_steps = (self.coarse_max_nm - self.coarse_min_nm) / self.coarse_step_nm
        refinement_steps = self.coarse_step_nm / self.refinement_step_nm
        if not math.isclose(span_steps, round(span_steps), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("The coarse range must be an integer number of coarse steps.")
        if not math.isclose(
            refinement_steps,
            round(refinement_steps),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "Each coarse interval must be an integer number of refinement steps."
            )
        if self.eval_width_nm <= 0.0 or self.baseline_exclude_nm <= 0.0:
            raise ValueError("Evaluation and baseline widths must be positive.")
        if self.accept_ratio <= 0.0:
            raise ValueError("accept_ratio must be positive.")
        if self.minimum_eval_columns < 1 or self.maximum_finalists < 1:
            raise ValueError("minimum_eval_columns and maximum_finalists must be positive.")
        return self


def _rounded_nm(value: float) -> float:
    """Return a stable wavelength-grid value for equality and serialization."""

    return round(float(value), 10)


def _regular_grid(start_nm: float, stop_nm: float, step_nm: float) -> tuple[float, ...]:
    n_steps = int(round((float(stop_nm) - float(start_nm)) / float(step_nm)))
    return tuple(
        _rounded_nm(float(start_nm) + index * float(step_nm))
        for index in range(n_steps + 1)
    )


def coarse_candidate_grid(settings: CalibrationSettings) -> tuple[float, ...]:
    """Return the validated 0.1 nm calibration grid, including both endpoints."""

    settings.validated()
    return _regular_grid(
        settings.coarse_min_nm,
        settings.coarse_max_nm,
        settings.coarse_step_nm,
    )


def _normalized_planet(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_planet_name(value: str) -> str:
    """Resolve a directory slug or display name to a configured planet key."""

    requested = _normalized_planet(value)
    matches = [name for name in config.PLANETS if _normalized_planet(name) == requested]
    if len(matches) != 1:
        raise KeyError(
            f"Could not uniquely resolve planet {value!r}; configured matches={matches}."
        )
    return matches[0]


def _slug(value: str) -> str:
    return str(value).strip().lower().replace("-", "").replace(" ", "")


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    _assert_diagnostic_artifact(path)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _assert_diagnostic_artifact(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _assert_diagnostic_artifact(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        raise ValueError(
            f"Calibration may write only diagnostic artifacts; refused {path}."
        )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    _assert_diagnostic_artifact(path)
    save_figure_pdf(fig, path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def _select_loader_exposures(
    *,
    raw_dir: Path,
    epoch: str,
    planet: str,
    data_mode: str,
    arm: str,
    do_molecfit: bool,
) -> ScienceExposureSelection:
    patterns = get_data_patterns(
        epoch,
        planet,
        mode=arm,
        do_molecfit=do_molecfit,
        data_dir=str(raw_dir),
    )
    return select_science_exposures(
        patterns,
        planet_name=planet,
        data_mode=data_mode,
        observation_epoch=epoch,
        arm=arm,
        do_molecfit=do_molecfit,
    )


def _raw_inventory(files: Sequence[Path]) -> dict[str, Any]:
    rows = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": rows,
        "n_files": len(rows),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "hash_scope": "path_size_mtime_inventory_not_file_contents",
    }


def _stellar_velocity(
    *,
    planet: str,
    epoch: str,
    arm: str,
    mode: str,
    enabled: bool,
) -> tuple[float | None, dict[str, Any]]:
    if not enabled:
        return None, {"applied": False, "reason": "disabled_in_notebook"}
    return resolve_stellar_velocity_correction(
        mode=mode,
        planet=planet,
        epoch=epoch,
        arm=arm,
    )


def load_raw_epoch_arm(
    *,
    planet: str,
    ephemeris: str,
    mode: str,
    epoch: str,
    arm: str,
    settings: CalibrationSettings,
) -> dict[str, Any]:
    """Load one raw epoch/arm on a common grid without median, SYSREM, or trim."""

    mode = str(mode).lower()
    if mode not in {"transmission", "emission"}:
        raise ValueError(f"Unsupported mode {mode!r}.")
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm {arm!r}.")
    display_planet = resolve_planet_name(planet)
    raw_dir = config_utils.get_raw_hrs_dir(
        planet=display_planet,
        epoch=epoch,
        mode=mode,
    )
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw exposure directory does not exist: {raw_dir}")

    from dataio.collapse_transmission_timeseries_to_1d import get_pepsi_data

    prefer_molecfit = bool(settings.use_molecfit_for_red and arm == "red")
    velocity_kms, velocity_metadata = _stellar_velocity(
        planet=display_planet,
        epoch=epoch,
        arm=arm,
        mode=mode,
        enabled=settings.apply_stellar_velocity_correction,
    )
    result = get_pepsi_data(
        arm=arm,
        observation_epoch=epoch,
        planet_name=display_planet,
        do_molecfit=prefer_molecfit,
        data_dir=raw_dir,
        regrid=True,
        subtract_median=False,
        run_sysrem=False,
        stellar_rest_velocity_kms=velocity_kms,
        data_mode=mode,
    )
    used_molecfit = prefer_molecfit
    if result is None:
        raise FileNotFoundError(
            f"The PEPSI loader found no "
            f"{'Molecfit' if prefer_molecfit else 'raw'} {arm} products in {raw_dir}."
        )
    arrays, extras = result
    wave, flux, error, jd, snr, exptime, airmass, n_spectra, npix = arrays
    wave = np.asarray(wave, dtype=float)
    wave_1d = wave[0] if wave.ndim == 2 else wave
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    planet_cfg = get_params(display_planet, ephemeris)
    bjd_tdb = get_bjd_tdb(
        np.asarray(jd, dtype=float),
        str(planet_cfg["RA"]),
        str(planet_cfg["Dec"]),
        header_bjd_tdb=extras.get("header_bjd_tdb"),
    )
    reference_epoch = get_ephemeris_epoch_bjd_tdb(
        float(planet_cfg["epoch"]),
        planet_cfg["epoch_scale"],
        planet_cfg["epoch_reference"],
    )
    t0 = resolve_transit_midpoint(
        np.asarray(bjd_tdb),
        planet_cfg,
        reference_epoch_bjd_tdb=reference_epoch,
        observation_epoch=epoch,
    )
    phase = np.asarray(
        get_orbital_phase(np.asarray(bjd_tdb), t0, float(planet_cfg["period"])),
        dtype=float,
    )
    exposure_selection = extras.get("science_exposure_selection")
    if not isinstance(exposure_selection, ScienceExposureSelection):
        raise RuntimeError(
            "PEPSI loader did not return the authoritative science-exposure "
            "selection record."
        )
    files = list(exposure_selection.usable_files)
    matched_pattern = exposure_selection.matched_pattern
    if len(files) != int(n_spectra):
        raise RuntimeError(
            f"Loader provenance mismatch for {epoch} {arm}: "
            f"{len(files)} resolved files versus {n_spectra} loaded exposures."
        )
    selected_inventory = _raw_inventory(files)
    companion_files: list[Path] = []
    companion_pattern = None
    companion_selection = None
    if used_molecfit:
        companion_selection = _select_loader_exposures(
            raw_dir=raw_dir,
            epoch=epoch,
            planet=display_planet,
            data_mode=mode,
            arm=arm,
            do_molecfit=False,
        )
        companion_files = list(companion_selection.usable_files)
        companion_pattern = companion_selection.matched_pattern
    combined_files = list(dict.fromkeys([*files, *companion_files]))
    provenance_inventory = {
        "selected_loader_product": "molecfit" if used_molecfit else "raw",
        "selected_loader_pattern": matched_pattern,
        "selected_loader_exposure_selection": exposure_selection.metadata(),
        "selected_loader_inventory": selected_inventory,
        "raw_companion_pattern": companion_pattern,
        "raw_companion_exposure_selection": (
            None if companion_selection is None else companion_selection.metadata()
        ),
        "raw_companion_inventory": (
            _raw_inventory(companion_files) if companion_files else None
        ),
        "combined_input_inventory": _raw_inventory(combined_files),
    }
    return {
        "planet": display_planet,
        "planet_slug": _slug(display_planet),
        "ephemeris": ephemeris,
        "mode": mode,
        "epoch": str(epoch),
        "arm": arm,
        "wave": np.asarray(wave_1d, dtype=float),
        "flux": flux,
        "error": error,
        "phase": phase,
        "bjd_tdb": np.asarray(bjd_tdb, dtype=float),
        "time_provenance": dict(extras.get("input_time_provenance", {})),
        "airmass": np.asarray(airmass, dtype=float),
        "snr": np.asarray(snr, dtype=float),
        "exptime": np.asarray(exptime, dtype=float),
        "planet_cfg": planet_cfg,
        "raw_dir": raw_dir,
        "used_molecfit": used_molecfit,
        "matched_pattern": matched_pattern,
        "raw_inventory": provenance_inventory,
        "stellar_velocity": velocity_metadata,
        "n_spectra": int(n_spectra),
        "n_pixels": int(npix),
    }


def build_pre_sysrem_branches(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reconstruct both canonical branch semantics without applying edge masks."""

    wave = np.asarray(raw["wave"], dtype=float)
    flux = np.asarray(raw["flux"], dtype=float)
    error = np.asarray(raw["error"], dtype=float)
    phase = np.asarray(raw["phase"], dtype=float)
    airmass = np.asarray(raw["airmass"], dtype=float)
    planet_cfg = raw["planet_cfg"]
    mode = str(raw["mode"])

    median_data, median_sigma, _ = subtract_median_spectrum(flux, error)
    if mode == "emission":
        common = {
            "source_data": median_data,
            "source_sigma": median_sigma,
            "source_phase": phase,
            "source_airmass": airmass,
            "selection": np.ones(phase.size, dtype=bool),
            "pre_display_data": median_data,
            "pre_display_sigma": median_sigma,
            "display_phase": phase,
            "pipeline_identity": "emission_common",
            "shadow_status": {
                "applied": False,
                "skip_reason": "emission_pipeline",
                "scaling": None,
            },
        }
        return {
            "timeseries": {**common, "product": "timeseries"},
            "collapse_source": {**common, "product": "collapse_source"},
        }

    selection = transmission_phase_selection_mask(
        phase,
        phase_bin="full",
        planet_params=planet_cfg,
    )
    shadow_status = {
        "applied": False,
        "skip_reason": "whole_arm_subtraction_disabled",
        "scaling": None,
    }
    contacts = compute_contact_phases(planet_cfg)
    out_transit = (phase < contacts["T1"]) | (phase > contacts["T4"])
    active_transit = active_transit_mask(phase, planet_cfg)
    active_interval = (
        "T14_grazing"
        if bool(planet_cfg.get("grazing_transit", False))
        else "T23"
    )
    source_rows = out_transit | active_transit
    if not np.any(out_transit):
        raise ValueError("collapse_source reconstruction has no out-of-transit exposures.")
    if not np.any(active_transit):
        raise ValueError(
            f"collapse_source reconstruction has no {active_interval} exposures."
        )
    collapse_data, collapse_sigma = build_out_of_transit_residuals(
        flux[source_rows],
        error[source_rows],
        out_transit[source_rows],
    )
    return {
        "timeseries": {
            "product": "timeseries",
            "source_data": median_data,
            "source_sigma": median_sigma,
            "source_phase": phase,
            "source_airmass": airmass,
            "selection": np.asarray(selection, dtype=bool),
            "pre_display_data": median_data[selection],
            "pre_display_sigma": median_sigma[selection],
            "display_phase": phase[selection],
            "pipeline_identity": "transmission_timeseries",
            "active_transit_interval": (
                "T14_grazing"
                if bool(planet_cfg.get("grazing_transit", False))
                else "T14"
            ),
            "shadow_status": shadow_status,
        },
        "collapse_source": {
            "product": "collapse_source",
            "source_data": collapse_data,
            "source_sigma": collapse_sigma,
            "source_phase": phase[source_rows],
            "source_airmass": airmass[source_rows],
            "selection": np.ones(np.count_nonzero(source_rows), dtype=bool),
            "pre_display_data": collapse_data,
            "pre_display_sigma": collapse_sigma,
            "display_phase": phase[source_rows],
            "pipeline_identity": "transmission_collapse_source",
            "active_transit_interval": active_interval,
            "shadow_status": {
                "applied": False,
                "skip_reason": "disabled_for_collapsed_1d_source",
                "scaling": None,
            },
        },
    }


def _non_edge_export_mask(wave: np.ndarray, arm: str) -> np.ndarray:
    """Return fixed non-edge masks used by canonical export."""

    return get_sysrem_deep_mask(wave, arm) | get_telluric_edge_mask(wave, arm)


def _candidate_keep_mask(
    wave: np.ndarray,
    *,
    left_trim_nm: float,
    right_trim_nm: float,
) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    finite = np.isfinite(wave)
    if not np.any(finite):
        return np.zeros(wave.size, dtype=bool)
    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    return (
        finite
        & (wave >= lo + float(left_trim_nm) * 10.0)
        & (wave <= hi - float(right_trim_nm) * 10.0)
    )


def _finalize_branch(
    raw: Mapping[str, Any],
    branch: Mapping[str, Any],
    *,
    left_trim_nm: float,
    right_trim_nm: float,
    run_sysrem: bool,
) -> dict[str, Any]:
    """Replay one branch in memory for one exact pair of edge candidates."""

    full_wave = np.asarray(raw["wave"], dtype=float)
    keep = _candidate_keep_mask(
        full_wave,
        left_trim_nm=left_trim_nm,
        right_trim_nm=right_trim_nm,
    )
    if np.count_nonzero(keep) < 2:
        raise ValueError("Candidate trim removes the entire wavelength grid.")
    wave = full_wave[keep]
    source_data = np.asarray(branch["source_data"], dtype=float)[:, keep]
    source_sigma = np.asarray(branch["source_sigma"], dtype=float)[:, keep]
    if run_sysrem:
        result = do_sysrem(
            wave,
            source_data,
            source_sigma,
            str(raw["arm"]),
            np.asarray(branch["source_airmass"], dtype=float),
            do_molecfit=bool(raw["used_molecfit"] and raw["arm"] == "red"),
            stop_delta_stddev=config.DEFAULT_SYSREM_STOP_TOL,
            return_diagnostics=True,
            planet_name=str(raw["planet"]),
            data_mode=str(raw["mode"]),
            observation_epoch=str(raw["epoch"]),
        )
        processed_data, processed_sigma, _u, _no_tellurics, diagnostics = result
    else:
        processed_data = source_data
        processed_sigma = source_sigma
        diagnostics = {}

    selection = np.asarray(branch["selection"], dtype=bool)
    shadow_status = branch["shadow_status"]
    display_data = np.asarray(processed_data, dtype=float)[selection]
    display_sigma = np.asarray(processed_sigma, dtype=float)[selection]

    fixed_mask = _non_edge_export_mask(wave, str(raw["arm"]))
    export_keep = ~fixed_mask
    if not np.any(export_keep):
        raise ValueError("No columns remain after fixed non-edge export masks.")
    return {
        "product": branch["product"],
        "wave": wave[export_keep],
        "data": display_data[:, export_keep],
        "sigma": display_sigma[:, export_keep],
        # Keep the complete post-trim grid for diagnostic plotting.  Science
        # scoring/export continues to use the compact arrays above, but plots
        # must retain fixed-mask columns as NaN so wavelength gaps are neither
        # compressed by imshow nor bridged by line plots.
        "plot_wave": wave,
        "plot_export_keep": export_keep,
        "phase": np.asarray(branch["display_phase"], dtype=float),
        "left_trim_nm": float(left_trim_nm),
        "right_trim_nm": float(right_trim_nm),
        "run_sysrem": bool(run_sysrem),
        "shadow_status": shadow_status,
        "active_transit_interval": branch.get("active_transit_interval"),
        "n_fixed_mask_columns": int(np.count_nonzero(fixed_mask)),
        "sysrem_diagnostics": diagnostics,
    }


def _full_plot_arrays(result: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore fixed-mask columns as NaN on the complete candidate grid."""

    compact_wave = np.asarray(result["wave"], dtype=float)
    compact_data = np.asarray(result["data"], dtype=float)
    compact_sigma = np.asarray(result["sigma"], dtype=float)
    plot_wave_value = result.get("plot_wave")
    plot_keep_value = result.get("plot_export_keep")
    if plot_wave_value is None or plot_keep_value is None:
        return compact_wave, compact_data, compact_sigma

    plot_wave = np.asarray(plot_wave_value, dtype=float)
    plot_keep = np.asarray(plot_keep_value, dtype=bool)
    if plot_wave.ndim != 1 or plot_keep.shape != plot_wave.shape:
        raise ValueError("Plot wavelength grid and export mask must be matching 1-D arrays.")
    if compact_data.ndim != 2 or compact_sigma.shape != compact_data.shape:
        raise ValueError("Compact plotting data and sigma must be matching 2-D arrays.")
    if compact_data.shape[1] != compact_wave.size:
        raise ValueError("Compact wavelength and matrix column counts do not match.")
    if int(np.count_nonzero(plot_keep)) != compact_wave.size:
        raise ValueError("Plot export mask does not reproduce the compact wavelength grid.")
    if not np.array_equal(plot_wave[plot_keep], compact_wave):
        raise ValueError("Plot export mask and compact wavelength values disagree.")

    plot_data = np.full((compact_data.shape[0], plot_wave.size), np.nan, dtype=float)
    plot_sigma = np.full((compact_sigma.shape[0], plot_wave.size), np.nan, dtype=float)
    plot_data[:, plot_keep] = compact_data
    plot_sigma[:, plot_keep] = compact_sigma
    return plot_wave, plot_data, plot_sigma


def _usable_pixel_mask(data: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the per-pixel validity mask shared by scoring and display."""

    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.isfinite(data) & np.isfinite(sigma) & (sigma > 0.0) & (sigma < 0.5)


def _profile_arrays(data: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    usable = _usable_pixel_mask(data, sigma)
    shown_data = np.where(usable, data, np.nan)
    shown_sigma = np.where(usable, sigma, np.nan)
    count = np.sum(np.isfinite(shown_data), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(shown_data, axis=0)
        sigma_center = np.nanmedian(shown_sigma, axis=0)
        scatter = np.nanstd(shown_data, axis=0)
    scatter[count < 2] = np.nan
    return {
        "profile": profile,
        "sigma": sigma_center,
        "scatter": scatter,
        "count": count,
    }


def _metric(values: np.ndarray, mask: np.ndarray, *, kind: str) -> float:
    selected = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return float("nan")
    if kind == "median":
        return float(np.nanmedian(selected))
    if kind == "p95_abs":
        return float(np.nanpercentile(np.abs(selected), 95.0))
    raise ValueError(f"Unknown metric kind {kind!r}.")


def _telluric_exclusion(wave: np.ndarray, arm: str, pad_nm: float) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    mask = np.zeros(wave.size, dtype=bool)
    pad_a = float(pad_nm) * 10.0
    table = config.TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})
    for key in ("telluric", "deep_mask"):
        for lo, hi in table.get(key, []):
            mask |= (wave >= float(lo) - pad_a) & (wave <= float(hi) + pad_a)
    return mask


def _baseline_mask(
    wave: np.ndarray,
    arm: str,
    settings: CalibrationSettings,
) -> np.ndarray:
    finite = np.isfinite(wave)
    if not np.any(finite):
        return finite
    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    pad_a = float(settings.baseline_exclude_nm) * 10.0
    mask = finite & (wave >= lo + pad_a) & (wave <= hi - pad_a)
    mask &= ~_telluric_exclusion(wave, arm, settings.baseline_exclude_nm)
    return mask


def _matched_baseline_mask(
    wave: np.ndarray,
    *,
    arm: str,
    side: str,
    evaluation: np.ndarray,
    settings: CalibrationSettings,
) -> tuple[np.ndarray, float, str, str]:
    """Match an effective edge to a comparable retained interior baseline."""

    wave = np.asarray(wave, dtype=float)
    evaluation = np.asarray(evaluation, dtype=bool)
    finite = np.isfinite(wave)
    if not np.any(finite):
        return finite, float("nan"), "non_telluric", "non_telluric"

    if side == "left":
        effective_edge_a = float(np.nanmin(wave[finite]))
    elif side == "right":
        effective_edge_a = float(np.nanmax(wave[finite]))
    else:
        raise ValueError(f"Unknown side {side!r}.")

    table = config.TELLURIC_REGIONS.get(arm, {"telluric": [], "deep_mask": []})
    telluric_intervals = [
        (float(lo), float(hi)) for lo, hi in table.get("telluric", [])
    ]
    matching_interval = next(
        (
            (lo, hi)
            for lo, hi in telluric_intervals
            if lo <= effective_edge_a <= hi
        ),
        None,
    )
    if matching_interval is None:
        baseline = _baseline_mask(wave, arm, settings) & ~evaluation
        return baseline, effective_edge_a / 10.0, "non_telluric", "non_telluric"

    lo, hi = matching_interval
    same_interval = finite & (wave >= lo) & (wave <= hi) & ~evaluation
    if np.count_nonzero(same_interval) >= settings.minimum_eval_columns:
        return same_interval, effective_edge_a / 10.0, "telluric", "same_interval"

    same_chunk = np.zeros(wave.size, dtype=bool)
    for interval_lo, interval_hi in telluric_intervals:
        same_chunk |= finite & (wave >= interval_lo) & (wave <= interval_hi)
    same_chunk &= ~evaluation
    return same_chunk, effective_edge_a / 10.0, "telluric", "same_chunk"


def _eval_mask(
    wave: np.ndarray,
    *,
    side: str,
    trim_nm: float,
    width_nm: float,
) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(wave)
    if not np.any(finite):
        return finite, float("nan"), float("nan")
    lo = float(np.nanmin(wave[finite]))
    hi = float(np.nanmax(wave[finite]))
    trim_a = float(trim_nm) * 10.0
    width_a = float(width_nm) * 10.0
    if side == "left":
        cut = lo + trim_a
        eval_lo, eval_hi = cut, min(cut + width_a, hi)
    elif side == "right":
        cut = hi - trim_a
        eval_lo, eval_hi = max(cut - width_a, lo), cut
    else:
        raise ValueError(f"Unknown side {side!r}.")
    return finite & (wave >= eval_lo) & (wave <= eval_hi), eval_lo, eval_hi


def _lost_protected_lines(
    wave: np.ndarray,
    *,
    arm: str,
    side: str,
    trim_nm: float,
    pad_nm: float,
) -> list[str]:
    finite = np.asarray(wave, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size or trim_nm <= 0.0:
        return []
    lo_nm = float(np.nanmin(finite) / 10.0)
    hi_nm = float(np.nanmax(finite) / 10.0)
    if side == "left":
        lost_lo, lost_hi = lo_nm, min(lo_nm + trim_nm, hi_nm)
    else:
        lost_lo, lost_hi = max(hi_nm - trim_nm, lo_nm), hi_nm
    hits = []
    for line in PROTECTED_EDGE_LINES:
        if str(line.get("arm")) != arm:
            continue
        rest_nm = float(line["rest_vacuum_A"]) / 10.0
        if lost_lo - pad_nm <= rest_nm <= lost_hi + pad_nm:
            hits.append(f"{line['label']} ({rest_nm:.3f} nm)")
    return list(dict.fromkeys(hits))


def score_candidate_side(
    *,
    wave: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    arm: str,
    side: str,
    trim_nm: float,
    settings: CalibrationSettings,
) -> dict[str, Any]:
    original_wave = np.asarray(wave, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if original_wave.ndim != 1:
        raise ValueError("Candidate scoring requires a one-dimensional wavelength grid.")
    if (
        data.ndim != 2
        or sigma.shape != data.shape
        or data.shape[1] != original_wave.size
    ):
        raise ValueError(
            "Candidate data/sigma must have matching (exposure, wavelength) shapes."
        )

    if side == "left":
        candidate_keep = _candidate_keep_mask(
            original_wave,
            left_trim_nm=trim_nm,
            right_trim_nm=0.0,
        )
    elif side == "right":
        candidate_keep = _candidate_keep_mask(
            original_wave,
            left_trim_nm=0.0,
            right_trim_nm=trim_nm,
        )
    else:
        raise ValueError(f"Unknown side {side!r}.")

    candidate_wave = original_wave[candidate_keep]
    fixed_keep = ~_non_edge_export_mask(candidate_wave, arm)
    retained_wave = candidate_wave[fixed_keep]
    retained_data = data[:, candidate_keep][:, fixed_keep]
    retained_sigma = sigma[:, candidate_keep][:, fixed_keep]

    arrays = _profile_arrays(retained_data, retained_sigma)
    evaluation, eval_lo, eval_hi = _eval_mask(
        retained_wave,
        side=side,
        trim_nm=0.0,
        width_nm=settings.eval_width_nm,
    )
    baseline, effective_edge_nm, edge_class, baseline_scope = _matched_baseline_mask(
        retained_wave,
        arm=arm,
        side=side,
        evaluation=evaluation,
        settings=settings,
    )
    baseline_values = (
        _metric(arrays["profile"], baseline, kind="p95_abs"),
        _metric(arrays["sigma"], baseline, kind="median"),
        _metric(arrays["scatter"], baseline, kind="p95_abs"),
    )
    edge_values = (
        _metric(arrays["profile"], evaluation, kind="p95_abs"),
        _metric(arrays["sigma"], evaluation, kind="median"),
        _metric(arrays["scatter"], evaluation, kind="p95_abs"),
    )
    ratios = [
        float(edge / interior)
        if math.isfinite(edge) and math.isfinite(interior) and interior > 0.0
        else float("nan")
        for edge, interior in zip(edge_values, baseline_values)
    ]
    finite_ratios = [value for value in ratios if math.isfinite(value)]
    quality_ratio = max(finite_ratios) if finite_ratios else float("nan")
    lost_lines = _lost_protected_lines(
        original_wave,
        arm=arm,
        side=side,
        trim_nm=trim_nm,
        pad_nm=settings.protected_line_pad_nm,
    )
    n_eval = int(np.count_nonzero(evaluation))
    n_baseline = int(np.count_nonzero(baseline))
    accepted = bool(
        math.isfinite(quality_ratio)
        and quality_ratio <= settings.accept_ratio
        and n_eval >= settings.minimum_eval_columns
        and n_baseline >= settings.minimum_eval_columns
        and not lost_lines
    )
    reasons = []
    if not math.isfinite(quality_ratio):
        reasons.append("nonfinite_quality_ratio")
    elif quality_ratio > settings.accept_ratio:
        reasons.append("quality_ratio_above_threshold")
    if n_eval < settings.minimum_eval_columns:
        reasons.append("too_few_evaluation_columns")
    if n_baseline < settings.minimum_eval_columns:
        reasons.append("too_few_matched_baseline_columns")
    if lost_lines:
        reasons.append("protected_line_removed")
    return {
        "side": side,
        "trim_nm": float(trim_nm),
        "effective_edge_nm": effective_edge_nm,
        "edge_class": edge_class,
        "baseline_scope": baseline_scope,
        "eval_lo_nm": float(eval_lo / 10.0) if math.isfinite(eval_lo) else None,
        "eval_hi_nm": float(eval_hi / 10.0) if math.isfinite(eval_hi) else None,
        "n_eval_columns": n_eval,
        "n_baseline_columns": n_baseline,
        "profile_ratio": ratios[0],
        "sigma_ratio": ratios[1],
        "scatter_ratio": ratios[2],
        "quality_ratio": quality_ratio,
        "accepted": accepted,
        "rejection_reasons": ";".join(reasons),
        "protected_lines_lost": "; ".join(lost_lines),
    }


def _score_pre_sysrem_grid(
    raw: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    settings: CalibrationSettings,
    *,
    candidates_by_side: Mapping[str, Sequence[float]],
    grid_stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for product, branch in branches.items():
        for side in SIDES:
            for trim_nm in candidates_by_side[side]:
                row = score_candidate_side(
                    wave=np.asarray(raw["wave"], dtype=float),
                    data=np.asarray(branch["pre_display_data"], dtype=float),
                    sigma=np.asarray(branch["pre_display_sigma"], dtype=float),
                    arm=str(raw["arm"]),
                    side=side,
                    trim_nm=float(trim_nm),
                    settings=settings,
                )
                rows.append(
                    {
                        "planet": raw["planet"],
                        "mode": raw["mode"],
                        "epoch": raw["epoch"],
                        "arm": raw["arm"],
                        "product": product,
                        "stage": "pre_sysrem",
                        "candidate_grid_stage": grid_stage,
                        **row,
                    }
                )
    return rows


def _joint_side_state(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
    trim_nm: float,
) -> dict[str, Any]:
    rows = [
        row
        for row in score_rows
        if row["stage"] == "pre_sysrem"
        and row["side"] == side
        and _rounded_nm(row["trim_nm"]) == _rounded_nm(trim_nm)
    ]
    by_product = {str(row["product"]): row for row in rows}
    if set(by_product) != set(PRODUCTS):
        raise ValueError(
            f"Expected one pre-SYSREM score for every product at {side}={trim_nm} nm; "
            f"found {sorted(by_product)}."
        )
    products = {}
    for product in PRODUCTS:
        row = by_product[product]
        products[product] = {
            "accepted": bool(row["accepted"]),
            "rejection_reasons": [
                reason
                for reason in str(row.get("rejection_reasons", "")).split(";")
                if reason
            ],
        }
    return {
        "joint_accepted": all(entry["accepted"] for entry in products.values()),
        "products": products,
    }


def _joint_side_state_key(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(state["joint_accepted"]),
        tuple(
            (
                product,
                bool(state["products"][product]["accepted"]),
                tuple(state["products"][product]["rejection_reasons"]),
            )
            for product in PRODUCTS
        ),
    )


def adaptive_refinement_intervals(
    coarse_score_rows: Sequence[Mapping[str, Any]],
    settings: CalibrationSettings,
) -> dict[str, list[dict[str, Any]]]:
    """Find coarse intervals where joint acceptance or rejection reasons change."""

    coarse = coarse_candidate_grid(settings)
    intervals: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for side in SIDES:
        for lower_nm, upper_nm in zip(coarse[:-1], coarse[1:]):
            lower_state = _joint_side_state(
                coarse_score_rows,
                side=side,
                trim_nm=lower_nm,
            )
            upper_state = _joint_side_state(
                coarse_score_rows,
                side=side,
                trim_nm=upper_nm,
            )
            if _joint_side_state_key(lower_state) == _joint_side_state_key(upper_state):
                continue
            intervals[side].append(
                {
                    "lower_nm": lower_nm,
                    "upper_nm": upper_nm,
                    "lower_state": lower_state,
                    "upper_state": upper_state,
                }
            )
    return intervals


def adaptive_candidate_plan(
    coarse_score_rows: Sequence[Mapping[str, Any]],
    settings: CalibrationSettings,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[float, ...]]]:
    """Return transition intervals and the exact per-side coarse/refined grids."""

    coarse = coarse_candidate_grid(settings)
    intervals = adaptive_refinement_intervals(coarse_score_rows, settings)
    tested: dict[str, tuple[float, ...]] = {}
    for side in SIDES:
        refined = {
            value
            for interval in intervals[side]
            for value in _regular_grid(
                interval["lower_nm"],
                interval["upper_nm"],
                settings.refinement_step_nm,
            )[1:-1]
        }
        tested[side] = tuple(sorted({*coarse, *refined}))
    return intervals, tested


def _score_pre_sysrem_candidates(
    raw: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    settings: CalibrationSettings,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, tuple[float, ...]],
]:
    coarse = coarse_candidate_grid(settings)
    coarse_by_side = {side: coarse for side in SIDES}
    coarse_rows = _score_pre_sysrem_grid(
        raw,
        branches,
        settings,
        candidates_by_side=coarse_by_side,
        grid_stage="coarse_0p1_nm",
    )
    intervals, tested = adaptive_candidate_plan(coarse_rows, settings)
    refined_by_side = {
        side: tuple(value for value in tested[side] if value not in set(coarse))
        for side in SIDES
    }
    refined_rows = _score_pre_sysrem_grid(
        raw,
        branches,
        settings,
        candidates_by_side=refined_by_side,
        grid_stage="adaptive_0p02_nm",
    )
    rows = sorted(
        [*coarse_rows, *refined_rows],
        key=lambda row: (
            PRODUCTS.index(str(row["product"])),
            SIDES.index(str(row["side"])),
            float(row["trim_nm"]),
        ),
    )
    return rows, intervals, tested


def _joint_candidates_for_side(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> list[float]:
    accepted = []
    candidates = sorted(
        {
            _rounded_nm(row["trim_nm"])
            for row in score_rows
            if row["stage"] == "pre_sysrem" and row["side"] == side
        }
    )
    for candidate in candidates:
        rows = [
            row
            for row in score_rows
            if row["stage"] == "pre_sysrem"
            and row["side"] == side
            and _rounded_nm(row["trim_nm"]) == candidate
        ]
        products = {str(row["product"]) for row in rows}
        if products == set(PRODUCTS) and all(bool(row["accepted"]) for row in rows):
            accepted.append(candidate)
    return accepted


def _hard_valid_score_row(row: Mapping[str, Any]) -> bool:
    """Return whether a score fails only the soft quality-ratio threshold."""

    try:
        quality_ratio = float(row["quality_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    reasons = {
        reason
        for reason in str(row.get("rejection_reasons", "")).split(";")
        if reason
    }
    return math.isfinite(quality_ratio) and reasons <= {
        "quality_ratio_above_threshold"
    }


def _best_available_candidate_for_side(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> dict[str, float] | None:
    """Choose the hard-valid trim minimizing the worst product-branch ratio."""

    options = []
    candidates = sorted(
        {
            _rounded_nm(row["trim_nm"])
            for row in score_rows
            if row["stage"] == "pre_sysrem" and row["side"] == side
        }
    )
    for candidate in candidates:
        rows = [
            row
            for row in score_rows
            if row["stage"] == "pre_sysrem"
            and row["side"] == side
            and _rounded_nm(row["trim_nm"]) == candidate
        ]
        products = {str(row["product"]) for row in rows}
        if products != set(PRODUCTS) or not all(
            _hard_valid_score_row(row) for row in rows
        ):
            continue
        options.append(
            {
                "trim_nm": candidate,
                "worst_quality_ratio": max(
                    float(row["quality_ratio"]) for row in rows
                ),
            }
        )
    if not options:
        return None
    return min(
        options,
        key=lambda option: (
            option["worst_quality_ratio"],
            option["trim_nm"],
        ),
    )


def _candidate_pair_plan(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates_by_side: dict[str, list[float]] = {}
    threshold_candidates: dict[str, list[float]] = {}
    fallbacks: dict[str, dict[str, float]] = {}
    for side in SIDES:
        passing = _joint_candidates_for_side(score_rows, side=side)
        threshold_candidates[side] = passing
        if passing:
            candidates_by_side[side] = passing
            continue
        fallback = _best_available_candidate_for_side(score_rows, side=side)
        if fallback is None:
            candidates_by_side[side] = []
            continue
        fallbacks[side] = fallback
        candidates_by_side[side] = [fallback["trim_nm"]]

    left = candidates_by_side["left"]
    right = candidates_by_side["right"]
    pairs = (
        sorted(
            itertools.product(left, right),
            key=lambda pair: (
                _rounded_nm(sum(pair)),
                _rounded_nm(max(pair)),
                pair[0],
                pair[1],
            ),
        )
        if left and right
        else []
    )
    n_threshold_pairs = (
        len(threshold_candidates["left"]) * len(threshold_candidates["right"])
    )
    return {
        "pairs": pairs,
        "fallbacks": fallbacks,
        "n_threshold_pairs": n_threshold_pairs,
    }


def _candidate_pairs(
    score_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    return list(_candidate_pair_plan(score_rows)["pairs"])


def _post_sysrem_score_rows(
    *,
    raw: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, Any]],
    left_trim_nm: float,
    right_trim_nm: float,
    settings: CalibrationSettings,
) -> list[dict[str, Any]]:
    rows = []
    for product, output in outputs.items():
        for side, original_trim in (
            ("left", left_trim_nm),
            ("right", right_trim_nm),
        ):
            # The output wavelength starts at the proposed boundary, so zero is
            # the exact evaluation offset on that finalist grid.
            row = score_candidate_side(
                wave=np.asarray(output["wave"], dtype=float),
                data=np.asarray(output["data"], dtype=float),
                sigma=np.asarray(output["sigma"], dtype=float),
                arm=str(raw["arm"]),
                side=side,
                trim_nm=0.0,
                settings=settings,
            )
            rows.append(
                {
                    "planet": raw["planet"],
                    "mode": raw["mode"],
                    "epoch": raw["epoch"],
                    "arm": raw["arm"],
                    "product": product,
                    "stage": "post_sysrem_finalist",
                    "tested_original_trim_nm": float(original_trim),
                    **row,
                }
            )
    return rows


def _run_pair(
    raw: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    *,
    left_trim_nm: float,
    right_trim_nm: float,
    run_sysrem: bool,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    outputs = {}
    for product, branch in branches.items():
        identity = str(branch["pipeline_identity"])
        if identity not in cache:
            cache[identity] = _finalize_branch(
                raw,
                branch,
                left_trim_nm=left_trim_nm,
                right_trim_nm=right_trim_nm,
                run_sysrem=run_sysrem,
            )
        outputs[product] = {**cache[identity], "product": product}
    return outputs


def _robust_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    finite = []
    for array in arrays:
        values = np.abs(np.asarray(array, dtype=float))
        finite.extend(values[np.isfinite(values)].tolist())
    if not finite:
        return 1.0
    value = float(np.nanpercentile(np.asarray(finite), percentile))
    return value if math.isfinite(value) and value > 0.0 else 1.0


def _plot_before_after(
    *,
    raw: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    status: str,
) -> plt.Figure:
    before_wave, before_data, before_sigma = _full_plot_arrays(before)
    after_wave, after_data, after_sigma = _full_plot_arrays(after)
    before_shown = np.where(
        _usable_pixel_mask(before_data, before_sigma), before_data, np.nan
    )
    after_shown = np.where(
        _usable_pixel_mask(after_data, after_sigma), after_data, np.nan
    )
    before_profiles = _profile_arrays(before_data, before_sigma)
    after_profiles = _profile_arrays(after_data, after_sigma)
    limit = _robust_limit(before_shown, after_shown, percentile=99.3)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.5), constrained_layout=True)
    for ax, wave, matrix, label in (
        (axes[0], before_wave, before_shown, "zero trim"),
        (axes[1], after_wave, after_shown, "selected candidate"),
    ):
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[float(wave[0]), float(wave[-1]), 0, matrix.shape[0]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_ylabel("Exposure")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, pad=0.01, label="Residual flux")

    axes[2].plot(before_wave, before_profiles["profile"], lw=0.75, label="zero trim")
    axes[2].plot(after_wave, after_profiles["profile"], lw=0.75, label="candidate")
    axes[2].set_ylabel("Mean residual")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.2)

    axes[3].plot(before_wave, before_profiles["scatter"], lw=0.7, label="row scatter: zero")
    axes[3].plot(after_wave, after_profiles["scatter"], lw=0.7, label="row scatter: candidate")
    axes[3].plot(before_wave, before_profiles["sigma"], lw=0.7, alpha=0.8, label="uncertainty: zero")
    axes[3].plot(after_wave, after_profiles["sigma"], lw=0.7, alpha=0.8, label="uncertainty: candidate")
    axes[3].set_ylabel("Scatter / uncertainty")
    axes[3].set_xlabel("Vacuum wavelength [Angstrom]")
    axes[3].legend(loc="upper right", fontsize=7, ncol=2)
    axes[3].grid(alpha=0.2)
    fig.suptitle(
        f"{raw['planet']} {raw['mode']} {raw['epoch']} {raw['arm']} "
        f"{after['product']} | {status}\n"
        f"candidate = ({after['left_trim_nm']:g}, {after['right_trim_nm']:g}) nm",
        fontsize=12,
    )
    return fig


def _plot_score_curves(
    raw: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    settings: CalibrationSettings,
    selected_pair: tuple[float, float] | None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, side in zip(axes, SIDES):
        for product, color in zip(PRODUCTS, ("tab:blue", "tab:orange")):
            product_rows = sorted(
                [row for row in rows if row["side"] == side and row["product"] == product],
                key=lambda row: float(row["trim_nm"]),
            )
            ax.plot(
                [row["trim_nm"] for row in product_rows],
                [row["quality_ratio"] for row in product_rows],
                ".-",
                color=color,
                label=product,
            )
            refined_rows = [
                row
                for row in product_rows
                if row.get("candidate_grid_stage") == "adaptive_0p02_nm"
            ]
            if refined_rows:
                ax.scatter(
                    [row["trim_nm"] for row in refined_rows],
                    [row["quality_ratio"] for row in refined_rows],
                    marker="x",
                    s=24,
                    linewidths=0.9,
                    color=color,
                    label=f"{product} adaptive 0.02 nm",
                    zorder=4,
                )
        ax.axhline(settings.accept_ratio, color="0.25", ls="--", lw=1.0, label="accept ratio")
        if selected_pair is not None:
            selected = selected_pair[0 if side == "left" else 1]
            ax.axvline(selected, color="tab:green", ls=":", lw=1.3, label="selected exact candidate")
        else:
            selected = 0.0
        provenance_rows = [
            row
            for row in rows
            if row["side"] == side
            and _rounded_nm(row["trim_nm"]) == _rounded_nm(selected)
        ]
        provenance = list(
            dict.fromkeys(
                (
                    row.get("effective_edge_nm"),
                    str(row.get("edge_class")),
                    str(row.get("baseline_scope")),
                )
                for row in provenance_rows
            )
        )
        if provenance:
            effective_edge_nm, edge_class, baseline_scope = provenance[0]
            heading = "selected" if selected_pair is not None else "0 nm reference"
            edge_text = (
                "unknown"
                if effective_edge_nm is None
                else f"{float(effective_edge_nm):.3f} nm"
            )
            ax.text(
                0.02,
                0.03,
                f"{heading}: {selected:g} nm\n"
                f"effective edge: {edge_text}\n"
                f"{edge_class} / {baseline_scope}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.5,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82},
            )
        ax.set_title(f"{side} edge")
        ax.set_xlabel("Tested trim [nm]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Worst edge/matched-baseline metric ratio")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{raw['planet']} {raw['mode']} {raw['epoch']} {raw['arm']} "
        "pre-SYSREM 0.1 nm grid with adaptive 0.02 nm transition refinement"
    )
    return fig


def _calibrate_epoch_arm(
    *,
    planet: str,
    ephemeris: str,
    mode: str,
    epoch: str,
    arm: str,
    settings: CalibrationSettings,
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    raw = load_raw_epoch_arm(
        planet=planet,
        ephemeris=ephemeris,
        mode=mode,
        epoch=epoch,
        arm=arm,
        settings=settings,
    )
    branches = build_pre_sysrem_branches(raw)
    pre_rows, refinement_intervals, tested_candidates = _score_pre_sysrem_candidates(
        raw,
        branches,
        settings,
    )
    candidate_plan = _candidate_pair_plan(pre_rows)
    pairs = list(candidate_plan["pairs"])
    fallbacks = candidate_plan["fallbacks"]
    figure_paths: list[Path] = []
    score_path = run_dir / f"{epoch}_{arm}_candidate_scores.pdf"

    raw_wave = np.asarray(raw["wave"], dtype=float)
    raw_min = float(np.nanmin(raw_wave))
    raw_max = float(np.nanmax(raw_wave))
    summary: dict[str, Any] = {
        "planet": raw["planet"],
        "planet_slug": raw["planet_slug"],
        "mode": mode,
        "ephemeris": ephemeris,
        "epoch": epoch,
        "arm": arm,
        "raw_min_A": raw_min,
        "raw_max_A": raw_max,
        "n_raw_exposures": raw["n_spectra"],
        "n_raw_columns": raw_wave.size,
        "used_molecfit": raw["used_molecfit"],
        "raw_inventory": raw["raw_inventory"],
        "time_provenance": raw["time_provenance"],
        "active_transit_interval": (
            (
                "T14_grazing"
                if bool(raw["planet_cfg"].get("grazing_transit", False))
                else "T23"
            )
            if mode == "transmission"
            else None
        ),
        "stellar_velocity": raw["stellar_velocity"],
        "candidate_strategy": "coarse_grid_with_transition_refinement",
        "coarse_candidates_nm": list(coarse_candidate_grid(settings)),
        "adaptive_refinement_intervals": refinement_intervals,
        "tested_candidates_nm": {
            side: list(tested_candidates[side]) for side in SIDES
        },
        "candidate_pair_order": "least_total_wavelength_removed",
        "n_joint_pre_sysrem_pairs": candidate_plan["n_threshold_pairs"],
        "n_candidate_pairs_including_fallback": len(pairs),
        "fallback_used": bool(fallbacks),
        "fallback_sides": sorted(fallbacks),
        "fallback_pre_sysrem_quality_ratio": {
            side: fallback["worst_quality_ratio"]
            for side, fallback in fallbacks.items()
        },
        "status": "failed_pre_sysrem",
        "left_trim_nm": None,
        "right_trim_nm": None,
        "keep_min_A": None,
        "keep_max_A": None,
        "protected_lines_lost": [],
        "product_results": {},
    }

    if not pairs:
        _save_figure(_plot_score_curves(raw, pre_rows, settings, None), score_path)
        figure_paths.append(score_path)
        summary["failure_reason"] = (
            "No candidate satisfies the hard validity requirements on both sides "
            "before SYSREM."
        )
        return summary, pre_rows, figure_paths

    zero_outputs = _run_pair(
        raw,
        branches,
        left_trim_nm=0.0,
        right_trim_nm=0.0,
        run_sysrem=settings.run_sysrem_finalists,
    )
    selected_pair: tuple[float, float] | None = None
    selected_outputs: dict[str, dict[str, Any]] | None = None
    diagnostic_pair: tuple[float, float] | None = None
    diagnostic_outputs: dict[str, dict[str, Any]] | None = None
    all_rows = list(pre_rows)
    finalist_failures = []
    finalists_tested = 0
    for pair in pairs[: settings.maximum_finalists]:
        finalists_tested += 1
        if pair == (0.0, 0.0):
            finalist_outputs = zero_outputs
        else:
            finalist_outputs = _run_pair(
                raw,
                branches,
                left_trim_nm=pair[0],
                right_trim_nm=pair[1],
                run_sysrem=settings.run_sysrem_finalists,
            )
        if diagnostic_pair is None:
            diagnostic_pair = pair
            diagnostic_outputs = finalist_outputs
        if settings.run_sysrem_finalists:
            final_rows = _post_sysrem_score_rows(
                raw=raw,
                outputs=finalist_outputs,
                left_trim_nm=pair[0],
                right_trim_nm=pair[1],
                settings=settings,
            )
            all_rows.extend(final_rows)
            passed = len(final_rows) == len(PRODUCTS) * len(SIDES) and all(
                bool(row["accepted"]) for row in final_rows
            )
            if not passed:
                finalist_failures.append(
                    {
                        "left_trim_nm": pair[0],
                        "right_trim_nm": pair[1],
                        "failed_rows": [
                            {
                                "product": row["product"],
                                "side": row["side"],
                                "quality_ratio": row["quality_ratio"],
                                "rejection_reasons": row["rejection_reasons"],
                            }
                            for row in final_rows
                            if not row["accepted"]
                        ],
                    }
                )
                continue
        selected_pair = pair
        selected_outputs = finalist_outputs
        break

    _save_figure(
        _plot_score_curves(raw, pre_rows, settings, selected_pair),
        score_path,
    )
    figure_paths.append(score_path)

    if selected_pair is None or selected_outputs is None:
        summary["status"] = "failed_post_sysrem"
        summary["failure_reason"] = (
            "Pre-SYSREM candidates existed, but no tested finalist passed the fresh SYSREM check."
        )
        summary["finalist_failures"] = finalist_failures
        summary["finalists_tested"] = finalists_tested
        if diagnostic_pair is not None and diagnostic_outputs is not None:
            summary["diagnostic_only_pair"] = {
                "left_trim_nm": diagnostic_pair[0],
                "right_trim_nm": diagnostic_pair[1],
                "accepted": False,
            }
            for product in PRODUCTS:
                figure_path = run_dir / f"{epoch}_{arm}_{product}_before_after_FAILED.pdf"
                _save_figure(
                    _plot_before_after(
                        raw=raw,
                        before=zero_outputs[product],
                        after=diagnostic_outputs[product],
                        status="failed post-SYSREM; diagnostic finalist only",
                    ),
                    figure_path,
                )
                figure_paths.append(figure_path)
        return summary, all_rows, figure_paths

    left, right = selected_pair
    summary.update(
        {
            "status": (
                "accepted_post_sysrem"
                if settings.run_sysrem_finalists
                else "provisional_pre_sysrem_only"
            ),
            "left_trim_nm": left,
            "right_trim_nm": right,
            "left_trim_A": left * 10.0,
            "right_trim_A": right * 10.0,
            "keep_min_A": raw_min + left * 10.0,
            "keep_max_A": raw_max - right * 10.0,
            "finalists_tested": finalists_tested,
            "finalist_failures": finalist_failures,
        }
    )
    for product in PRODUCTS:
        before = zero_outputs[product]
        after = selected_outputs[product]
        summary["product_results"][product] = {
            "n_exposures": int(np.asarray(after["data"]).shape[0]),
            "active_transit_interval": after.get("active_transit_interval"),
            "n_columns_zero_trim": int(np.asarray(before["wave"]).size),
            "n_columns_selected_trim": int(np.asarray(after["wave"]).size),
            "shadow_status": after["shadow_status"],
            "fixed_non_edge_mask_columns_zero_trim": before["n_fixed_mask_columns"],
            "fixed_non_edge_mask_columns_selected_trim": after["n_fixed_mask_columns"],
        }
        figure_path = run_dir / f"{epoch}_{arm}_{product}_before_after.pdf"
        _save_figure(
            _plot_before_after(
                raw=raw,
                before=before,
                after=after,
                status=summary["status"],
            ),
            figure_path,
        )
        figure_paths.append(figure_path)
    return summary, all_rows, figure_paths


def _summary_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "planet": entry.get("planet"),
        "mode": entry.get("mode"),
        "ephemeris": entry.get("ephemeris"),
        "epoch": entry.get("epoch"),
        "arm": entry.get("arm"),
        "status": entry.get("status"),
        "left_trim_nm": entry.get("left_trim_nm"),
        "right_trim_nm": entry.get("right_trim_nm"),
        "raw_min_A": entry.get("raw_min_A"),
        "raw_max_A": entry.get("raw_max_A"),
        "keep_min_A": entry.get("keep_min_A"),
        "keep_max_A": entry.get("keep_max_A"),
        "n_raw_exposures": entry.get("n_raw_exposures"),
        "n_raw_columns": entry.get("n_raw_columns"),
        "n_joint_pre_sysrem_pairs": entry.get("n_joint_pre_sysrem_pairs"),
        "failure_reason": entry.get("failure_reason", ""),
    }


def _report_text(manifest: Mapping[str, Any]) -> str:
    rows = [_summary_row(entry) for entry in manifest["datasets"]]
    lines = [
        "# Dataset-specific spectral edge-trim calibration",
        "",
        f"Generated: {manifest['generated_utc']}",
        f"Target: {manifest['planet']} {manifest['mode']} ({manifest['ephemeris']})",
        f"Overall status: **{manifest['overall_status']}**",
        "",
        "This was a read-only raw-exposure dry run. No prepared `.npy` arrays were written.",
        (
            "Candidate scores use a 0.1 nm coarse grid and 0.02 nm refinement wherever "
            "adjacent acceptance states or rejection reasons change. The proposed trim "
            "and fixed non-edge masks are applied before each surviving edge is compared "
            "with a matched telluric or non-telluric baseline."
        ),
        "",
        "| Epoch | Arm | Status | Left nm | Right nm | Keep min A | Keep max A |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def shown(value: Any) -> str:
            return "" if value is None else f"{float(value):.4f}"

        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["epoch"]),
                    str(row["arm"]),
                    str(row["status"]),
                    shown(row["left_trim_nm"]),
                    shown(row["right_trim_nm"]),
                    shown(row["keep_min_A"]),
                    shown(row["keep_max_A"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Promotion guard",
            "",
            (
                "Every epoch/arm passed post-SYSREM validation. The values remain proposals "
                "until explicitly promoted into canonical preparation."
                if manifest["overall_status"] == "accepted_post_sysrem"
                else "At least one epoch/arm did not pass. Do not generate canonical prepared data from this manifest."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def resolve_calibration_datasets(
    *,
    epochs: Iterable[str] | None = None,
    datasets: Iterable[tuple[str, str]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Resolve explicit epoch/arm pairs, retaining the legacy dual-arm API."""

    if datasets is not None and epochs is not None:
        raise ValueError("Pass datasets or epochs, not both.")
    if datasets is None:
        if epochs is None:
            raise ValueError("At least one calibration dataset is required.")
        resolved = tuple((str(epoch), arm) for epoch in epochs for arm in ARMS)
    else:
        resolved = tuple((str(epoch), str(arm).lower()) for epoch, arm in datasets)
    if not resolved:
        raise ValueError("At least one calibration dataset is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate calibration datasets: {resolved!r}.")
    for epoch, arm in resolved:
        if not epoch:
            raise ValueError("Calibration dataset epochs must be non-empty.")
        if arm not in ARMS:
            raise ValueError(f"Unsupported calibration arm {arm!r}.")
    return resolved


def run_edge_trim_calibration(
    *,
    planet: str,
    ephemeris: str,
    mode: str,
    epochs: Iterable[str] | None = None,
    datasets: Iterable[tuple[str, str]] | None = None,
    settings: CalibrationSettings | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Calibrate requested epoch/arm pairs and write diagnostic artifacts."""

    settings = (settings or CalibrationSettings()).validated()
    display_planet = resolve_planet_name(planet)
    resolved_datasets = resolve_calibration_datasets(
        epochs=epochs,
        datasets=datasets,
    )
    resolved_epochs = tuple(dict.fromkeys(epoch for epoch, _arm in resolved_datasets))
    resolved_arms = tuple(dict.fromkeys(arm for _epoch, arm in resolved_datasets))
    generated = datetime.now(timezone.utc)
    run_stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / str(mode) / _slug(display_planet) / run_stamp
    suffix = 1
    while run_dir.exists():
        run_dir = run_dir.with_name(f"{run_stamp}_{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset_entries = []
    score_rows: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    for epoch, arm in resolved_datasets:
        print(f"\n=== {display_planet} {mode} {epoch} {arm} ===")
        try:
            entry, rows, figures = _calibrate_epoch_arm(
                planet=display_planet,
                ephemeris=ephemeris,
                mode=mode,
                epoch=epoch,
                arm=arm,
                settings=settings,
                run_dir=run_dir,
            )
        except Exception as exc:
            entry = {
                "planet": display_planet,
                "planet_slug": _slug(display_planet),
                "mode": mode,
                "ephemeris": ephemeris,
                "epoch": epoch,
                "arm": arm,
                "status": "error",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "left_trim_nm": None,
                "right_trim_nm": None,
            }
            rows = []
            figures = []
            print(entry["failure_reason"])
        dataset_entries.append(entry)
        score_rows.extend(rows)
        figure_paths.extend(figures)

    accepted_status = (
        "accepted_post_sysrem"
        if settings.run_sysrem_finalists
        else "provisional_pre_sysrem_only"
    )
    all_accepted = bool(dataset_entries) and all(
        entry.get("status") == accepted_status for entry in dataset_entries
    )
    overall_status = accepted_status if all_accepted else "failed_or_incomplete"
    manifest = {
        "schema_version": 3,
        "kind": "proposed_dataset_specific_edge_trim_calibration",
        "generated_utc": generated.isoformat(),
        "planet": display_planet,
        "planet_slug": _slug(display_planet),
        "mode": mode,
        "ephemeris": ephemeris,
        "epochs": list(resolved_epochs),
        "arms": list(resolved_arms),
        "products_required": list(PRODUCTS),
        "overall_status": overall_status,
        "canonical_generation_authorized": False,
        "prepared_arrays_written": False,
        "allowed_artifact_suffixes": sorted(ALLOWED_ARTIFACT_SUFFIXES),
        "settings": asdict(settings),
        "score_semantics": {
            "candidate_strategy": "coarse_grid_with_transition_refinement",
            "coarse_grid_nm": {
                "minimum": settings.coarse_min_nm,
                "maximum": settings.coarse_max_nm,
                "step": settings.coarse_step_nm,
            },
            "adaptive_refinement": {
                "step_nm": settings.refinement_step_nm,
                "trigger": "adjacent_joint_acceptance_or_rejection_reason_change",
            },
            "threshold_fallback": {
                "trigger": "no_joint_candidate_meets_accept_ratio_on_a_side",
                "selection": "minimum_worst_quality_ratio_across_products",
                "hard_validity_requirements_remain_mandatory": True,
                "post_sysrem_threshold_remains_mandatory": True,
            },
            "pair_order": "least_total_wavelength_removed",
            "maximum_fresh_sysrem_pairs": settings.maximum_finalists,
            "candidate_grid": "proposed_trim_then_fixed_non_edge_masks",
            "evaluation": "outermost_retained_window",
            "evaluation_width_nm": settings.eval_width_nm,
            "telluric_baseline_order": ["same_interval", "same_chunk"],
            "non_telluric_baseline": "clean_retained_interior",
            "insufficient_matched_baseline": "reject",
            "protected_lines_checked_against": "original_grid_and_requested_trim",
            "candidate_score_provenance_fields": [
                "effective_edge_nm",
                "edge_class",
                "baseline_scope",
            ],
        },
        "calibration_code": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": _source_hash(),
            "git": _git_revision(),
        },
        "datasets": dataset_entries,
        "candidate_score_csv": "candidate_scores.csv",
        "recommendation_csv": "proposed_boundaries.csv",
        "figures": [path.name for path in figure_paths],
        "promotion_rule": (
            "Only exact coarse or adaptively refined candidates with "
            "accepted_post_sysrem status for every epoch/arm may be considered for "
            "explicit manual promotion."
        ),
    }
    manifest_path = run_dir / "proposed_edge_trim_manifest.json"
    score_path = run_dir / "candidate_scores.csv"
    boundary_path = run_dir / "proposed_boundaries.csv"
    report_path = run_dir / "report.md"
    _write_json(manifest_path, manifest)
    _write_csv(score_path, score_rows)
    _write_csv(boundary_path, [_summary_row(entry) for entry in dataset_entries])
    _assert_diagnostic_artifact(report_path)
    report_path.write_text(_report_text(manifest))
    print(f"\nDiagnostics written to: {run_dir}")
    print(f"Overall status: {overall_status}")
    print("Prepared arrays written: False")
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "candidate_scores_path": score_path,
        "boundaries_path": boundary_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
        "manifest": manifest,
        "summary_rows": [_summary_row(entry) for entry in dataset_entries],
    }


def print_summary(result: Mapping[str, Any]) -> None:
    """Print a compact calibration result table without requiring pandas."""

    rows = result["summary_rows"]
    headers = ("epoch", "arm", "status", "left_trim_nm", "right_trim_nm", "keep_min_A", "keep_max_A")
    widths = {key: len(key) for key in headers}
    for row in rows:
        for key in headers:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))
    print("  ".join(key.ljust(widths[key]) for key in headers))
    print("  ".join("-" * widths[key] for key in headers))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers))
    print(f"\nRun directory: {result['run_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Overall status: {result['manifest']['overall_status']}")
