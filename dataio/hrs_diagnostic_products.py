"""Read-only discovery and loading of prepared HRS diagnostic products."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import config
import config_utils
from dataio.edge_trim_manifest import load_accepted_edge_trim_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "input" / "hrs"
PRODUCTS = ("timeseries", "collapse_source")
ARMS = ("blue", "red")
REQUIRED_ARRAYS = ("wavelength.npy", "data.npy", "sigma.npy", "phase.npy")
OPTIONAL_ARRAYS = (
    "jd",
    "airmass",
    "snr",
    "exptime",
    "pre_sysrem_data",
    "pre_sysrem_sigma",
    "projection_sigma",
)


def planet_slug(planet: str) -> str:
    """Return the repository directory key for a configured planet name."""

    return "".join(character for character in str(planet).lower() if character.isalnum())


def bundle_is_complete(path: str | Path) -> bool:
    """Return whether a directory contains the minimum prepared-bundle contract."""

    directory = Path(path)
    return (directory / "timeseries_prep.json").is_file() and all(
        (directory / name).is_file() for name in REQUIRED_ARRAYS
    )


def discover_prepared_inventory(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Discover complete target/mode bundles from preparation metadata.

    The result keeps the inventory shape used by the calibration and diagnostic
    runners: keys are ``(mode, planet_slug)`` and entries contain ``datasets``,
    ``epochs``, ``planet_display``, and ``ephemeris``.
    """

    root = Path(input_root)
    inventory: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "epochs": defaultdict(dict),
            "planet_display": None,
            "ephemerides": set(),
        }
    )
    for metadata_path in sorted(root.glob("*/*/*/*/timeseries/timeseries_prep.json")):
        rel = metadata_path.relative_to(root)
        mode, slug, epoch, arm, product, _ = rel.parts
        if mode not in {"transmission", "emission"} or arm not in ARMS:
            continue
        if product != "timeseries":
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        display_planet = str(metadata.get("planet") or slug)
        active_arms = config_utils.get_hrs_observation_arms(
            display_planet,
            epoch,
            mode=mode,
        )
        if arm not in active_arms:
            continue
        paths = {
            name: root / mode / slug / epoch / arm / name
            for name in PRODUCTS
        }
        if not all(bundle_is_complete(path) for path in paths.values()):
            continue
        key = (mode, slug)
        inventory[key]["planet_display"] = display_planet
        inventory[key]["ephemerides"].add(metadata.get("ephemeris"))
        inventory[key]["epochs"][epoch][arm] = paths

    complete: dict[tuple[str, str], dict[str, Any]] = {}
    for key, entry in sorted(inventory.items()):
        complete_epochs: dict[str, dict[str, dict[str, Path]]] = {}
        datasets: list[tuple[str, str]] = []
        for epoch, arms in sorted(entry["epochs"].items()):
            active_arms = config_utils.get_hrs_observation_arms(
                entry["planet_display"],
                epoch,
                mode=key[0],
            )
            if set(arms) != set(active_arms):
                continue
            complete_epochs[epoch] = dict(arms)
            datasets.extend((epoch, arm) for arm in active_arms)
        if not complete_epochs:
            continue
        ephemerides = entry["ephemerides"] - {None}
        if len(ephemerides) != 1:
            raise ValueError(
                f"{key}: expected one saved ephemeris, found {sorted(ephemerides)}."
            )
        complete[key] = {
            "epochs": complete_epochs,
            "datasets": tuple(datasets),
            "planet_display": entry["planet_display"],
            "ephemeris": next(iter(ephemerides)),
        }
    return complete


def _configured_planet_from_slug(slug: str) -> str:
    requested = planet_slug(slug)
    matches = [name for name in config.PLANETS if planet_slug(name) == requested]
    if len(matches) != 1:
        raise KeyError(
            f"Could not uniquely resolve raw planet directory {slug!r}; "
            f"configured matches={matches}."
        )
    return matches[0]


def discover_raw_calibration_inventory(
    input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Discover populated raw target/mode datasets for edge-trim calibration."""

    root = Path(input_root)
    prepared = discover_prepared_inventory(root)
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in ("emission", "transmission"):
        raw_root = root / mode / "raw"
        if not raw_root.is_dir():
            continue
        for planet_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            epochs = tuple(
                epoch_dir.name
                for epoch_dir in sorted(path for path in planet_dir.iterdir() if path.is_dir())
                if any(path.is_file() for path in epoch_dir.rglob("*"))
            )
            if not epochs:
                continue
            key = (mode, planet_dir.name)
            prepared_entry = prepared.get(key)
            if prepared_entry is None:
                display_planet = _configured_planet_from_slug(planet_dir.name)
                if "Recommended" not in config.PLANETS[display_planet]:
                    raise KeyError(
                        f"{key}: raw-only target has no Recommended ephemeris profile."
                    )
                ephemeris = "Recommended"
            else:
                display_planet = prepared_entry["planet_display"]
                ephemeris = prepared_entry["ephemeris"]
            datasets = tuple(
                (epoch, arm)
                for epoch in epochs
                for arm in config_utils.get_hrs_observation_arms(
                    display_planet,
                    epoch,
                    mode=mode,
                )
            )
            inventory[key] = {
                "epochs": epochs,
                "datasets": datasets,
                "planet_display": display_planet,
                "ephemeris": ephemeris,
            }
    return inventory


def resolve_product_specs(
    *,
    planet: str,
    mode: str,
    product: str,
    epochs: Iterable[str] | None = None,
    arms: Iterable[str] | None = None,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> list[dict[str, Any]]:
    """Resolve existing prepared product directories for one target and mode."""

    if mode not in {"transmission", "emission"}:
        raise ValueError("mode must be 'transmission' or 'emission'.")
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {PRODUCTS}.")
    root = Path(input_root)
    slug = planet_slug(planet)
    target_root = root / mode / slug
    selected_epochs = tuple(str(epoch) for epoch in epochs) if epochs else tuple(
        path.name for path in sorted(target_root.iterdir()) if path.is_dir()
    ) if target_root.is_dir() else ()
    requested_arms = tuple(str(arm) for arm in arms) if arms else None
    specs: list[dict[str, Any]] = []
    for epoch in selected_epochs:
        epoch_arms = requested_arms or tuple(
            config_utils.get_hrs_observation_arms(planet, epoch, mode=mode)
        )
        for arm in epoch_arms:
            path = target_root / epoch / arm / product
            if not path.is_dir():
                continue
            specs.append(
                {
                    "planet": planet,
                    "planet_slug": slug,
                    "mode": mode,
                    "epoch": epoch,
                    "arm": arm,
                    "product": product,
                    "path": path,
                    "label": f"{epoch} {arm} {product}",
                }
            )
    return specs


def _load_optional_array(path: Path) -> np.ndarray | None:
    return np.load(path, allow_pickle=True) if path.is_file() else None


def _collapsed_selection(mode: str) -> str:
    return "full_emission" if mode == "emission" else "full_transit"


def load_product_bundle(spec: dict[str, Any]) -> dict[str, Any]:
    """Load one prepared bundle without changing any saved product."""

    path = Path(spec["path"])
    missing = [name for name in REQUIRED_ARRAYS if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing required arrays {missing}.")
    metadata_path = path / "timeseries_prep.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    bundle: dict[str, Any] = {
        **spec,
        "path": path,
        "wavelength": np.asarray(np.load(path / "wavelength.npy"), dtype=float),
        "data": np.asarray(np.load(path / "data.npy"), dtype=float),
        "sigma": np.asarray(np.load(path / "sigma.npy"), dtype=float),
        "phase": np.asarray(np.load(path / "phase.npy"), dtype=float),
        "metadata": metadata,
        "meta": metadata,
    }
    for name in OPTIONAL_ARRAYS:
        value = _load_optional_array(path / f"{name}.npy")
        if value is not None:
            bundle[name] = np.asarray(value)

    mode = str(spec["mode"])
    collapsed_root = path.parent / "collapsed" / _collapsed_selection(mode)
    collapsed_names = {
        "wavelength": collapsed_root / f"wavelength_{mode}.npy",
        "spectrum": collapsed_root / f"spectrum_{mode}.npy",
        "uncertainty": collapsed_root / f"uncertainty_{mode}.npy",
    }
    bundle["collapsed"] = (
        {
            name: np.asarray(np.load(array_path), dtype=float)
            for name, array_path in collapsed_names.items()
        }
        if all(array_path.is_file() for array_path in collapsed_names.values())
        else None
    )
    sysrem_path = path / "U_sysrem.npz"
    bundle["sysrem"] = (
        {name: np.asarray(value) for name, value in np.load(sysrem_path).items()}
        if sysrem_path.is_file()
        else None
    )
    return bundle


def load_product_records(specs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load every product spec, raising on incomplete selected products."""

    return [load_product_bundle(spec) for spec in specs]


def accepted_edge_trim_rows(
    *,
    planet: str,
    mode: str,
    datasets: Iterable[tuple[str, str]],
    manifest_path: str | Path,
) -> tuple[Path, dict[tuple[str, str], dict[str, Any]]]:
    """Load an explicit accepted edge-trim manifest for selected datasets."""

    selected_path, _manifest, rows = load_accepted_edge_trim_manifest(
        Path(manifest_path).parent,
        planet=planet,
        mode=mode,
        required_datasets=tuple(datasets),
        manifest_path=Path(manifest_path),
    )
    return selected_path, rows


def verify_bundle_edge_trim(
    bundle: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    absolute_tolerance_A: float = 1.0e-8,
) -> dict[str, Any]:
    """Verify saved preparation metadata against one accepted manifest row."""

    metadata = bundle.get("metadata", {})
    saved = metadata.get("arm_edge_trim") or {}
    expected_left = float(manifest_row["left_trim_A"])
    expected_right = float(manifest_row["right_trim_A"])
    try:
        saved_left = float(saved["left_trim_A"])
        saved_right = float(saved["right_trim_A"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{bundle['path']}: missing saved arm_edge_trim provenance."
        ) from exc
    if not np.isclose(saved_left, expected_left, rtol=0.0, atol=absolute_tolerance_A):
        raise ValueError(
            f"{bundle['path']}: saved left trim {saved_left} A does not match "
            f"manifest {expected_left} A."
        )
    if not np.isclose(saved_right, expected_right, rtol=0.0, atol=absolute_tolerance_A):
        raise ValueError(
            f"{bundle['path']}: saved right trim {saved_right} A does not match "
            f"manifest {expected_right} A."
        )
    return {
        "status": "verified_saved_product",
        "left_trim_A": saved_left,
        "right_trim_A": saved_right,
        "n_trimmed_columns": int(saved.get("n_trimmed_columns", 0)),
    }


def bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON/CSV-friendly structural summary of one bundle."""

    wavelength = np.asarray(bundle["wavelength"], dtype=float)
    data = np.asarray(bundle["data"], dtype=float)
    sigma = np.asarray(bundle["sigma"], dtype=float)
    return {
        "planet": bundle["planet"],
        "mode": bundle["mode"],
        "epoch": bundle["epoch"],
        "arm": bundle["arm"],
        "product": bundle["product"],
        "path": str(bundle["path"]),
        "n_exposures": int(data.shape[0]),
        "n_wavelength": int(data.shape[1]),
        "wavelength_min_A": float(np.nanmin(wavelength)),
        "wavelength_max_A": float(np.nanmax(wavelength)),
        "finite_data_fraction": float(np.count_nonzero(np.isfinite(data)) / data.size),
        "finite_positive_sigma_fraction": float(
            np.count_nonzero(np.isfinite(sigma) & (sigma > 0)) / sigma.size
        ),
        "ephemeris": bundle.get("metadata", {}).get("ephemeris"),
        "has_pre_sysrem": "pre_sysrem_data" in bundle,
        "has_sysrem": bundle.get("sysrem") is not None,
        "has_collapsed": bundle.get("collapsed") is not None,
    }
