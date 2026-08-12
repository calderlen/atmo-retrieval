"""Discover, download, and classify low-resolution exoplanet spectra in MAST.

The archive acquisition step and the scientific reduction step are deliberately
separate.  Calibrated flux spectra and time-series products are downloaded and
recorded, but only tables that explicitly contain a transit/eclipse observable
and its uncertainty are normalized into the repository's NASA ``.tbl`` format.

Examples
--------
Preview public HST/JWST products without downloading them::

    python -m dataio.mast_spectra \
        --target KELT-20 --mode emission --query-only

Download the conservative default set of directly ingestible products::

    python -m dataio.mast_spectra \
        --target KELT-20 --mode emission --download

Download calibrated inputs for an instrument-specific light-curve reduction::

    python -m dataio.mast_spectra \
        --target KELT-20 --planet KELT-20b --mode transmission \
        --product-profile reduction --download

Use explicit column/unit overrides for a reduced depth-spectrum FITS table::

    python -m dataio.mast_spectra \
        --target KELT-20 --mode emission --download \
        --value-column ECLIPSE_DEPTH --uncertainty-column ECLIPSE_DEPTH_ERR \
        --wavelength-unit micron --value-unit ppm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack


DEFAULT_COLLECTIONS = ("HST", "JWST")
DIRECT_PRODUCT_SUBGROUPS = frozenset({"X1D", "X1DINTS", "X1DSUM", "SX1"})
REDUCTION_PRODUCT_SUBGROUPS = frozenset({"CALINTS", "RATEINTS", "IMA", "FLT"})
DEFAULT_PRODUCT_SUBGROUPS = DIRECT_PRODUCT_SUBGROUPS
PRODUCT_PROFILES = ("direct", "reduction", "all")
DEFAULT_ALIAS_REGISTRY = (
    Path(__file__).resolve().parents[1] / "reference" / "mast_target_aliases.json"
)
TABLE_EXTENSIONS = frozenset({".ecsv", ".csv", ".tab", ".tbl"})
DOWNLOAD_EXTENSIONS = frozenset({".fits", ".fit", ".fits.gz", *TABLE_EXTENSIONS})

_RAW_SUBGROUPS = frozenset({"RAW", "UNCAL"})
_TIMESERIES_SUBGROUPS = frozenset({"CALINTS", "RATEINTS", "IMA"})
_EXTRACTED_SUBGROUPS = frozenset({"X1D", "X1DINTS", "X1DSUM", "SX1"})
_SPECTRAL_IMAGE_SUBGROUPS = frozenset({"S2D", "I2D", "FLT", "FLV", "SX2"})

_WAVELENGTH_ALIASES = (
    "CENTRALWAVELNG",
    "WAVELENGTH",
    "WAVELENGTHUM",
    "WAVE",
    "WAVEUM",
    "LAMBDA",
    "LAM",
)
_BANDWIDTH_ALIASES = ("BANDWIDTH", "BINWIDTH", "WAVELENGTHWIDTH", "DELTAWAVE")
_FLUX_ALIASES = ("FLUX", "FLUXDENSITY", "SURFBRIGHT", "SCI")
_TRANSMISSION_VALUE_KINDS = {
    "SPECTRANSDEP": "depth",
    "SPECTRANDEP": "depth",
    "SPECTRANSDEPTH": "depth",
    "PLTRANDEP": "depth",
    "TRANSITDEPTH": "depth",
    "RPRS": "radius_ratio",
    "RADIUSRATIO": "radius_ratio",
}
_EMISSION_VALUE_KINDS = {
    "ESPECLIPDEP": "depth",
    "SPECLIPDEP": "depth",
    "SPECDEP": "depth",
    "ECLIPSEDEPTH": "depth",
    "PLANETSTARFLUXRATIO": "flux_ratio",
    "FLUXRATIO": "flux_ratio",
    "FPFS": "flux_ratio",
}


@dataclass(frozen=True)
class MastQueryConfig:
    """Archive-level query parameters."""

    target: str | None
    mode: str
    planet: str | None = None
    radius_deg: float = 0.001
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS
    proposal_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    archive_target_names: tuple[str, ...] = ()


@dataclass
class ProductRecord:
    """Serializable observation/product metadata plus local processing state."""

    data_uri: str
    filename: str
    obsid: str | None
    obs_id: str | None
    target_name: str | None
    collection: str | None
    instrument: str | None
    filter_name: str | None
    proposal_id: str | None
    product_type: str | None
    subgroup: str | None
    description: str | None
    calibration_level: int | None
    data_rights: str | None
    size_bytes: int | None
    archive_classification: str
    observation_query_matches: tuple[str, ...] = ()
    selected: bool = False
    selection_reason: str = "not evaluated"
    local_path: str | None = None
    download_status: str = "not requested"
    download_message: str | None = None
    sha256: str | None = None
    content_classification: str | None = None
    content_details: dict[str, Any] | None = None
    normalized_path: str | None = None
    normalization_status: str = "not requested"
    recommended_cli_flag: str | None = None


@dataclass(frozen=True)
class ColumnOverrides:
    wavelength_column: str | None = None
    value_column: str | None = None
    uncertainty_column: str | None = None
    uncertainty_low_column: str | None = None
    uncertainty_high_column: str | None = None
    bandwidth_column: str | None = None
    wavelength_unit: str | None = None
    value_unit: str | None = None
    value_kind: str | None = None


@dataclass(frozen=True)
class ExtractedDepthSpectrum:
    wavelength_angstrom: np.ndarray
    depth_fraction: np.ndarray
    uncertainty_fraction: np.ndarray
    bandwidth_angstrom: np.ndarray
    source_location: str
    wavelength_column: str
    value_column: str
    uncertainty_columns: tuple[str, ...]
    wavelength_unit: str
    value_unit: str
    value_kind: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _clean_component(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or fallback


def target_slug(target: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", target.lower()) or "target"


def load_alias_registry(path: str | Path = DEFAULT_ALIAS_REGISTRY) -> dict[str, tuple[str, ...]]:
    """Load curated exact MAST target labels keyed by normalized planet name."""

    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(payload.get("targets"), dict):
        raise ValueError(f"Invalid MAST target alias registry: {path}")

    registry: dict[str, tuple[str, ...]] = {}
    for name, entry in payload["targets"].items():
        aliases = entry.get("archive_target_names", []) if isinstance(entry, dict) else []
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise ValueError(f"Invalid archive_target_names entry for {name!r} in {path}")
        registry[target_slug(name)] = tuple(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
    return registry


def resolve_archive_target_names(
    *,
    target: str | None,
    planet: str | None,
    explicit_names: Iterable[str] = (),
    registry_path: str | Path = DEFAULT_ALIAS_REGISTRY,
    use_registry: bool = True,
) -> tuple[str, ...]:
    """Combine local aliases and repeatable CLI aliases without changing their spelling."""

    aliases: list[str] = []
    if use_registry:
        registry = load_alias_registry(registry_path)
        for candidate in (planet, target):
            if candidate:
                slug = target_slug(candidate)
                aliases.extend(registry.get(slug, ()))
                if not slug.endswith("b"):
                    aliases.extend(registry.get(f"{slug}b", ()))
    aliases.extend(str(name).strip() for name in explicit_names if str(name).strip())
    return tuple(dict.fromkeys(aliases))


def _plain_value(value: Any) -> Any:
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _row_value(row: Any, names: Sequence[str], default: Any = None) -> Any:
    colnames = getattr(row, "colnames", None)
    if colnames is None and hasattr(row, "table"):
        colnames = getattr(row.table, "colnames", None)
    if colnames is None and isinstance(row, Mapping):
        colnames = row.keys()
    available = {str(name).lower(): name for name in (colnames or [])}
    for requested in names:
        actual = available.get(requested.lower())
        if actual is not None:
            return _plain_value(row[actual])
    return default


def _optional_int(value: Any) -> int | None:
    value = _plain_value(value)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filename_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".fits.gz"):
        return ".fits.gz"
    return Path(lower).suffix


def _subgroup_from_filename(filename: str) -> str:
    stem = filename.lower()
    for suffix in (".fits.gz", ".fits", ".fit", ".ecsv", ".csv", ".tbl", ".tab"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    token = re.split(r"[_-]", stem)[-1]
    return token.upper()


def classify_archive_product(row: Any) -> str:
    """Classify a MAST product from archive metadata without opening the file."""

    product_type = str(_row_value(row, ("productType", "product_type"), "") or "").upper()
    filename = str(_row_value(row, ("productFilename", "filename"), "") or "")
    subgroup = str(
        _row_value(row, ("productSubGroupDescription", "subgroup"), "") or ""
    ).upper()
    subgroup = subgroup or _subgroup_from_filename(filename)
    extension = _filename_extension(filename)

    if subgroup in _TIMESERIES_SUBGROUPS:
        return "calibrated_timeseries"
    if product_type and product_type != "SCIENCE":
        return "auxiliary"
    if subgroup in _RAW_SUBGROUPS:
        return "raw_exposure"
    if subgroup in _EXTRACTED_SUBGROUPS:
        return "extracted_flux_spectrum"
    if subgroup in _SPECTRAL_IMAGE_SUBGROUPS:
        return "spectral_image"
    if extension in TABLE_EXTENSIONS:
        description = str(_row_value(row, ("description",), "") or "")
        semantic_text = f"{subgroup} {filename} {description}".lower()
        spectral_terms = ("spec", "transit", "eclipse", "depth", "x1d", "spectrum")
        if any(term in semantic_text for term in spectral_terms):
            return "science_table_candidate"
        return "unclassified_science_table"
    if extension in {".fits", ".fit", ".fits.gz"}:
        return "unclassified_science_fits"
    return "unsupported_product"


def _observation_lookup(observations: Table) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in observations:
        obsid = _row_value(row, ("obsid", "obsID"))
        if obsid is None:
            continue
        lookup[str(obsid)] = {
            "obs_id": _row_value(row, ("obs_id",)),
            "target_name": _row_value(row, ("target_name", "target_name_hlsp")),
            "collection": _row_value(row, ("obs_collection",)),
            "instrument": _row_value(row, ("instrument_name",)),
            "filter_name": _row_value(row, ("filters", "filter_name", "filter")),
            "proposal_id": _row_value(row, ("proposal_id",)),
            "calibration_level": _optional_int(_row_value(row, ("calib_level",))),
            "data_rights": _row_value(row, ("dataRights", "data_rights")),
            "query_matches": tuple(
                item
                for item in str(_row_value(row, ("query_matches",), "") or "").split("|")
                if item
            ),
        }
    return lookup


def build_product_records(observations: Table, products: Table) -> list[ProductRecord]:
    """Join observation metadata onto product rows and deduplicate by data URI."""

    lookup = _observation_lookup(observations)
    records: list[ProductRecord] = []
    seen_uris: set[str] = set()
    for row in products:
        uri = str(_row_value(row, ("dataURI", "data_uri"), "") or "")
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        obsid_raw = _row_value(row, ("obsID", "obsid", "parent_obsid"))
        obsid = None if obsid_raw is None else str(obsid_raw)
        parent_obsid_raw = _row_value(row, ("parent_obsid",))
        parent_obsid = None if parent_obsid_raw is None else str(parent_obsid_raw)
        obs_meta = lookup.get(obsid or "", lookup.get(parent_obsid or "", {}))
        filename = str(
            _row_value(row, ("productFilename", "filename"), Path(uri).name) or Path(uri).name
        )
        subgroup = _row_value(row, ("productSubGroupDescription", "subgroup"))
        if not subgroup:
            subgroup = _subgroup_from_filename(filename)
        record = ProductRecord(
            data_uri=uri,
            filename=filename,
            obsid=obsid,
            obs_id=_row_value(row, ("obs_id",), obs_meta.get("obs_id")),
            target_name=_row_value(row, ("target_name",), obs_meta.get("target_name")),
            collection=_row_value(row, ("obs_collection",), obs_meta.get("collection")),
            instrument=_row_value(row, ("instrument_name",), obs_meta.get("instrument")),
            filter_name=_row_value(
                row,
                ("filters", "filter_name", "filter"),
                obs_meta.get("filter_name"),
            ),
            proposal_id=str(
                _row_value(row, ("proposal_id",), obs_meta.get("proposal_id")) or ""
            )
            or None,
            product_type=_row_value(row, ("productType", "product_type")),
            subgroup=str(subgroup).upper() if subgroup else None,
            description=_row_value(row, ("description",)),
            calibration_level=_optional_int(
                _row_value(row, ("calib_level",), obs_meta.get("calibration_level"))
            ),
            data_rights=_row_value(
                row, ("dataRights", "data_rights"), obs_meta.get("data_rights")
            ),
            size_bytes=_optional_int(_row_value(row, ("size", "size_bytes"))),
            archive_classification=classify_archive_product(row),
            observation_query_matches=tuple(obs_meta.get("query_matches", ())),
        )
        records.append(record)
    return records


def build_uri_records(data_uris: Iterable[str]) -> list[ProductRecord]:
    """Build explicitly selected records from exact MAST data URIs."""

    records: list[ProductRecord] = []
    seen: set[str] = set()
    for raw_uri in data_uris:
        uri = str(raw_uri).strip()
        if not uri or uri.startswith("#"):
            continue
        if not uri.lower().startswith("mast:"):
            raise ValueError(
                f"Exact product {uri!r} is not a MAST data URI. Expected a value beginning with 'mast:'."
            )
        if uri in seen:
            continue
        seen.add(uri)
        filename = uri.rsplit("/", 1)[-1]
        collection_match = re.match(r"mast:([^/]+)", uri, flags=re.IGNORECASE)
        collection = collection_match.group(1) if collection_match else None
        row = {
            "dataURI": uri,
            "productFilename": filename,
            "productType": "SCIENCE",
            "productSubGroupDescription": _subgroup_from_filename(filename),
        }
        records.append(
            ProductRecord(
                data_uri=uri,
                filename=filename,
                obsid=None,
                obs_id=None,
                target_name=None,
                collection=collection,
                instrument=None,
                filter_name=None,
                proposal_id=None,
                product_type="SCIENCE",
                subgroup=_subgroup_from_filename(filename),
                description="explicit MAST data URI",
                calibration_level=None,
                data_rights=None,
                size_bytes=None,
                archive_classification=classify_archive_product(row),
                selected=True,
                selection_reason="explicit data URI",
            )
        )
    return records


def select_products(
    records: list[ProductRecord],
    *,
    product_profile: str = "direct",
    product_subgroups: Iterable[str] | None = None,
    include_raw: bool = False,
    include_2d: bool = False,
    include_proprietary: bool = False,
    max_products: int | None = 200,
    max_file_bytes: int | None = 2 * 1024**3,
    max_total_bytes: int | None = 20 * 1024**3,
) -> list[ProductRecord]:
    """Select a bounded download set using an explicit scientific product profile."""

    product_profile = str(product_profile).lower()
    if product_profile not in PRODUCT_PROFILES:
        raise ValueError(
            f"Unknown product profile {product_profile!r}; choose from {PRODUCT_PROFILES}"
        )

    explicit_subgroups = product_subgroups is not None
    requested_subgroups = {
        _canonical_name(item) for item in (product_subgroups or ())
    }
    if not explicit_subgroups:
        if product_profile == "direct":
            requested_subgroups = set(DIRECT_PRODUCT_SUBGROUPS)
        elif product_profile == "reduction":
            requested_subgroups = set(REDUCTION_PRODUCT_SUBGROUPS)
    priority = {
        "science_table_candidate": 0,
        "extracted_flux_spectrum": 1,
        "calibrated_timeseries": 2,
        "spectral_image": 3,
        "unclassified_science_fits": 4,
        "unclassified_science_table": 4,
        "raw_exposure": 5,
        "auxiliary": 6,
        "unsupported_product": 7,
    }
    ordered = sorted(
        records,
        key=lambda record: (
            priority.get(record.archive_classification, 99),
            -(record.calibration_level or -1),
            record.data_uri,
        ),
    )

    selected_count = 0
    selected_bytes = 0
    for record in ordered:
        record.selected = False
        rights = str(record.data_rights or "PUBLIC").upper()
        if rights != "PUBLIC" and not include_proprietary:
            record.selection_reason = f"excluded data rights: {rights}"
            continue
        if _filename_extension(record.filename) not in DOWNLOAD_EXTENSIONS:
            record.selection_reason = "excluded unsupported file extension"
            continue
        if record.archive_classification == "auxiliary":
            record.selection_reason = "excluded non-science product"
            continue
        if record.archive_classification == "raw_exposure" and not include_raw:
            record.selection_reason = "excluded raw product; use --include-raw"
            continue

        subgroup = _canonical_name(record.subgroup or "")
        if explicit_subgroups:
            profile_candidate = subgroup in requested_subgroups
            exclusion_reason = "excluded by explicit product subgroup filter"
        elif product_profile == "direct":
            profile_candidate = (
                subgroup in requested_subgroups
                or record.archive_classification == "science_table_candidate"
            )
            exclusion_reason = "excluded by direct product profile"
        elif product_profile == "reduction":
            profile_candidate = subgroup in requested_subgroups
            exclusion_reason = "excluded by reduction product profile"
        else:
            profile_candidate = record.archive_classification not in {
                "auxiliary",
                "raw_exposure",
                "unsupported_product",
            }
            exclusion_reason = "excluded by all-science product profile"

        if record.archive_classification == "spectral_image" and not (
            include_2d or profile_candidate
        ):
            record.selection_reason = "excluded 2D product; use --include-2d"
            continue
        if include_2d and record.archive_classification == "spectral_image":
            profile_candidate = True
        if include_raw and record.archive_classification == "raw_exposure":
            profile_candidate = True
        if not profile_candidate:
            record.selection_reason = exclusion_reason
            continue
        if max_file_bytes is not None and record.size_bytes is not None:
            if record.size_bytes > max_file_bytes:
                record.selection_reason = "excluded by per-file size limit"
                continue
        if max_products is not None and selected_count >= max_products:
            record.selection_reason = "excluded by product-count limit"
            continue
        if max_total_bytes is not None and record.size_bytes is not None:
            if selected_bytes + record.size_bytes > max_total_bytes:
                record.selection_reason = "excluded by total-size limit"
                continue

        record.selected = True
        if explicit_subgroups:
            record.selection_reason = "selected by explicit product subgroup filter"
        else:
            record.selection_reason = f"selected by {product_profile} product profile"
        selected_count += 1
        selected_bytes += record.size_bytes or 0

    return records


def _merge_observation_queries(results: Sequence[tuple[str, Table]]) -> Table:
    """Union query results by MAST ``obsid`` and retain every matching query."""

    nonempty: list[Table] = []
    for query_match, table in results:
        if len(table) == 0:
            continue
        annotated = table.copy()
        annotated["_query_match"] = [query_match] * len(annotated)
        nonempty.append(annotated)
    if not nonempty:
        return results[0][1].copy() if results else Table()

    stacked = vstack(nonempty, join_type="outer", metadata_conflicts="silent")
    first_indices: OrderedDict[str, int] = OrderedDict()
    matches: defaultdict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(stacked):
        obsid = _row_value(row, ("obsid", "obsID"))
        if obsid is None:
            obsid = _row_value(row, ("obs_id",), f"row-{index}")
        key = str(obsid)
        first_indices.setdefault(key, index)
        query_match = str(_row_value(row, ("_query_match",), "") or "")
        if query_match and query_match not in matches[key]:
            matches[key].append(query_match)

    merged = stacked[list(first_indices.values())]
    keys = list(first_indices)
    merged.remove_column("_query_match")
    merged["query_matches"] = ["|".join(matches[key]) for key in keys]
    return merged


def query_mast(config: MastQueryConfig, observations_client: Any | None = None) -> tuple[Table, Table]:
    """Query object/cone and exact target labels, then union observations by ``obsid``."""

    if observations_client is None:
        try:
            from astroquery.mast import Observations as observations_client
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MAST fetching requires astroquery. Install the repository environment "
                "or run `pip install astroquery`."
            ) from exc

    base_criteria: dict[str, Any] = {
        "obs_collection": list(config.collections),
        "intentType": "science",
    }
    if config.proposal_ids:
        base_criteria["proposal_id"] = list(config.proposal_ids)
    if config.instruments:
        base_criteria["instrument_name"] = list(config.instruments)

    query_results: list[tuple[str, Table]] = []
    if config.observation_ids:
        criteria = dict(base_criteria)
        criteria["obs_id"] = list(config.observation_ids)
        query_results.append(
            ("observation_id", observations_client.query_criteria(**criteria))
        )
    else:
        if config.target:
            criteria = dict(base_criteria)
            criteria["objectname"] = config.target
            criteria["radius"] = f"{config.radius_deg} deg"
            query_results.append(
                (
                    f"object_cone:{config.target}",
                    observations_client.query_criteria(**criteria),
                )
            )
        for archive_target_name in config.archive_target_names:
            criteria = dict(base_criteria)
            criteria["target_name"] = [archive_target_name]
            query_results.append(
                (
                    f"target_name:{archive_target_name}",
                    observations_client.query_criteria(**criteria),
                )
            )
        if not query_results:
            raise ValueError(
                "MAST discovery requires --target, --archive-target-name, or --obs-id; "
                "proposal-wide discovery is intentionally unsupported."
            )

    observations = _merge_observation_queries(query_results)
    if len(observations) == 0:
        return observations, Table()
    get_unique = getattr(observations_client, "get_unique_product_list", None)
    if get_unique is not None:
        products = get_unique(observations)
    else:
        products = observations_client.get_product_list(observations)
    return observations, products


def _column_lookup(names: Sequence[str]) -> dict[str, str]:
    return {_canonical_name(name): str(name) for name in names}


def _find_column(
    names: Sequence[str],
    aliases: Iterable[str],
    explicit: str | None = None,
) -> str | None:
    lookup = _column_lookup(names)
    if explicit is not None:
        matched = lookup.get(_canonical_name(explicit))
        if matched is None:
            raise ValueError(f"Requested column {explicit!r} not found; available columns: {list(names)}")
        return matched
    for alias in aliases:
        matched = lookup.get(_canonical_name(alias))
        if matched is not None:
            return matched
    return None


def _column_array(data: Any, column: str) -> np.ndarray:
    values = np.asarray(data[column])
    if values.dtype.kind == "O":
        parts = [np.asarray(value).ravel() for value in values]
        values = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    return np.asarray(values, dtype=float).ravel()


def _normalize_unit_text(unit: str | None) -> str:
    return re.sub(r"[\s_-]+", "", str(unit or "").strip().lower())


def _wavelength_to_angstrom(values: np.ndarray, unit: str | None) -> np.ndarray:
    normalized = _normalize_unit_text(unit)
    factors = {
        "angstrom": 1.0,
        "angstroms": 1.0,
        "aa": 1.0,
        "a": 1.0,
        "nm": 10.0,
        "nanometer": 10.0,
        "nanometers": 10.0,
        "um": 1.0e4,
        "micron": 1.0e4,
        "microns": 1.0e4,
        "micrometer": 1.0e4,
        "micrometers": 1.0e4,
        "m": 1.0e10,
        "meter": 1.0e10,
        "meters": 1.0e10,
    }
    if normalized not in factors:
        raise ValueError(
            f"Wavelength unit {unit!r} is missing or unsupported; pass --wavelength-unit."
        )
    return np.asarray(values, dtype=float) * factors[normalized]


def _depth_scale(unit: str | None) -> float:
    normalized = _normalize_unit_text(unit)
    if normalized in {"1", "fraction", "dimensionless", "unitless"}:
        return 1.0
    if normalized in {"%", "percent", "percentage"}:
        return 1.0e-2
    if normalized in {"ppm", "partspermillion"}:
        return 1.0e-6
    if normalized in {"ppt", "partsperthousand", "permille"}:
        return 1.0e-3
    raise ValueError(f"Depth unit {unit!r} is missing or unsupported; pass --value-unit.")


def _units_for_fits_column(hdu: fits.BinTableHDU, column: str) -> str:
    try:
        return str(hdu.columns[column].unit or "")
    except (KeyError, TypeError):
        return ""


def _units_for_table_column(table: Table, column: str) -> str:
    unit = getattr(table[column], "unit", None)
    return str(unit or "")


def _find_value_column(
    names: Sequence[str], mode: str, explicit: str | None, explicit_kind: str | None
) -> tuple[str | None, str | None]:
    if explicit is not None:
        column = _find_column(names, (), explicit=explicit)
        if explicit_kind is None:
            canonical = _canonical_name(column)
            kinds = _TRANSMISSION_VALUE_KINDS if mode == "transmission" else _EMISSION_VALUE_KINDS
            explicit_kind = kinds.get(canonical)
        if explicit_kind is None:
            raise ValueError(
                "An explicit --value-column with an unfamiliar name also requires "
                "--value-kind depth, radius_ratio, or flux_ratio."
            )
        return column, explicit_kind

    kinds = _TRANSMISSION_VALUE_KINDS if mode == "transmission" else _EMISSION_VALUE_KINDS
    lookup = _column_lookup(names)
    for alias, kind in kinds.items():
        if alias in lookup:
            return lookup[alias], kind
    return None, None


def _find_uncertainty_columns(
    names: Sequence[str],
    value_column: str,
    overrides: ColumnOverrides,
) -> tuple[str, ...]:
    if overrides.uncertainty_column:
        matched = _find_column(names, (), explicit=overrides.uncertainty_column)
        return (str(matched),)
    if overrides.uncertainty_low_column or overrides.uncertainty_high_column:
        requested = [overrides.uncertainty_low_column, overrides.uncertainty_high_column]
        return tuple(
            str(_find_column(names, (), explicit=name)) for name in requested if name is not None
        )

    canonical = _canonical_name(value_column)
    lookup = _column_lookup(names)
    pairs = (
        (f"{canonical}ERR1", f"{canonical}ERR2"),
        (f"{canonical}ERRORLOW", f"{canonical}ERRORHIGH"),
        (f"{canonical}ERRLOW", f"{canonical}ERRHIGH"),
    )
    for low, high in pairs:
        present = tuple(lookup[name] for name in (low, high) if name in lookup)
        if present:
            return present
    for suffix in ("ERR", "ERROR", "UNC", "UNCERTAINTY", "SIGMA"):
        candidate = lookup.get(f"{canonical}{suffix}")
        if candidate is not None:
            return (candidate,)
    for generic in ("ERROR", "ERR", "UNCERTAINTY", "SIGMA"):
        candidate = lookup.get(generic)
        if candidate is not None:
            return (candidate,)
    return ()


def _extract_from_tabular(
    *,
    data: Any,
    names: Sequence[str],
    unit_getter: Any,
    source_location: str,
    mode: str,
    overrides: ColumnOverrides,
) -> ExtractedDepthSpectrum | None:
    wavelength_column = _find_column(names, _WAVELENGTH_ALIASES, overrides.wavelength_column)
    value_column, value_kind = _find_value_column(
        names, mode, overrides.value_column, overrides.value_kind
    )
    if wavelength_column is None or value_column is None or value_kind is None:
        return None

    uncertainty_columns = _find_uncertainty_columns(names, value_column, overrides)
    if not uncertainty_columns:
        return None

    wavelength_unit = overrides.wavelength_unit or unit_getter(wavelength_column)
    value_unit = overrides.value_unit or unit_getter(value_column)
    wavelengths = _wavelength_to_angstrom(_column_array(data, wavelength_column), wavelength_unit)
    values = _column_array(data, value_column)
    uncertainty_arrays = []
    for uncertainty_column in uncertainty_columns:
        uncertainty_unit = overrides.value_unit or unit_getter(uncertainty_column) or value_unit
        uncertainty_arrays.append(
            np.abs(_column_array(data, uncertainty_column)) * _depth_scale(uncertainty_unit)
        )
    uncertainty = (
        uncertainty_arrays[0]
        if len(uncertainty_arrays) == 1
        else np.nanmax(np.vstack(uncertainty_arrays), axis=0)
    )
    if wavelengths.size != values.size or uncertainty.size != values.size:
        raise ValueError(
            f"Column sizes do not match in {source_location}: wavelength={wavelengths.size}, "
            f"value={values.size}, uncertainty={uncertainty.size}."
        )

    scale = _depth_scale(value_unit)
    if value_kind == "radius_ratio":
        ratio = values * scale
        ratio_sigma = uncertainty
        depth = ratio**2
        depth_sigma = 2.0 * np.abs(ratio) * ratio_sigma
    else:
        depth = values * scale
        depth_sigma = uncertainty

    bandwidth_column = _find_column(names, _BANDWIDTH_ALIASES, overrides.bandwidth_column)
    if bandwidth_column is None:
        bandwidth = np.full_like(wavelengths, np.nan)
    else:
        bandwidth_unit = unit_getter(bandwidth_column) or overrides.wavelength_unit or wavelength_unit
        bandwidth = _wavelength_to_angstrom(_column_array(data, bandwidth_column), bandwidth_unit)
        if bandwidth.size != wavelengths.size:
            raise ValueError(
                f"Bandwidth size does not match wavelength size in {source_location}."
            )

    mask = (
        np.isfinite(wavelengths)
        & np.isfinite(depth)
        & np.isfinite(depth_sigma)
        & (wavelengths > 0.0)
        & (depth_sigma > 0.0)
    )
    if not np.any(mask):
        raise ValueError(f"No finite depth measurements with positive uncertainty in {source_location}.")
    order = np.argsort(wavelengths[mask])
    return ExtractedDepthSpectrum(
        wavelength_angstrom=wavelengths[mask][order],
        depth_fraction=depth[mask][order],
        uncertainty_fraction=depth_sigma[mask][order],
        bandwidth_angstrom=bandwidth[mask][order],
        source_location=source_location,
        wavelength_column=wavelength_column,
        value_column=value_column,
        uncertainty_columns=tuple(uncertainty_columns),
        wavelength_unit=str(wavelength_unit),
        value_unit=str(value_unit),
        value_kind=value_kind,
    )


def extract_direct_depth_spectrum(
    path: str | Path,
    *,
    mode: str,
    overrides: ColumnOverrides | None = None,
) -> ExtractedDepthSpectrum | None:
    """Extract an already-reduced depth spectrum, never a calibrated flux spectrum."""

    path = Path(path)
    mode = str(mode).lower().strip()
    if mode not in {"transmission", "emission"}:
        raise ValueError("mode must be 'transmission' or 'emission'")
    overrides = overrides or ColumnOverrides()

    if _filename_extension(path.name) in {".fits", ".fit", ".fits.gz"}:
        extraction_errors: list[str] = []
        with fits.open(path, memmap=False) as hdul:
            for index, hdu in enumerate(hdul):
                if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)) or hdu.data is None:
                    continue
                names = list(hdu.columns.names or [])
                try:
                    extracted = _extract_from_tabular(
                        data=hdu.data,
                        names=names,
                        unit_getter=lambda name, current_hdu=hdu: _units_for_fits_column(
                            current_hdu, name
                        ),
                        source_location=f"{path.name}[{index}:{hdu.name}]",
                        mode=mode,
                        overrides=overrides,
                    )
                except ValueError as exc:
                    extraction_errors.append(f"{hdu.name}: {exc}")
                    continue
                if extracted is not None:
                    return extracted
        if extraction_errors and (
            overrides.wavelength_column is not None or overrides.value_column is not None
        ):
            raise ValueError("; ".join(extraction_errors))
        return None

    table = Table.read(path)
    return _extract_from_tabular(
        data=table,
        names=table.colnames,
        unit_getter=lambda name: _units_for_table_column(table, name),
        source_location=path.name,
        mode=mode,
        overrides=overrides,
    )


def inspect_downloaded_product(
    path: str | Path,
    *,
    mode: str,
    overrides: ColumnOverrides | None = None,
) -> tuple[str, dict[str, Any], ExtractedDepthSpectrum | None]:
    """Inspect content and return a conservative scientific classification."""

    path = Path(path)
    details: dict[str, Any] = {"file_extension": _filename_extension(path.name)}
    try:
        extracted = extract_direct_depth_spectrum(path, mode=mode, overrides=overrides)
    except (OSError, ValueError, TypeError) as exc:
        details["normalization_error"] = str(exc)
        extracted = None
    if extracted is not None:
        details.update(
            {
                "n_bins": int(extracted.wavelength_angstrom.size),
                "wavelength_column": extracted.wavelength_column,
                "value_column": extracted.value_column,
                "uncertainty_columns": list(extracted.uncertainty_columns),
                "value_kind": extracted.value_kind,
            }
        )
        classification = (
            "direct_joint_constraint"
            if extracted.wavelength_angstrom.size >= 5
            else "direct_bandpass_constraint"
        )
        return classification, details, extracted

    extension = _filename_extension(path.name)
    if extension in {".fits", ".fit", ".fits.gz"}:
        try:
            with fits.open(path, memmap=False) as hdul:
                table_columns: list[str] = []
                for hdu in hdul:
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        table_columns.extend(hdu.columns.names or [])
                details["table_columns"] = sorted(set(table_columns))
                canonical = {_canonical_name(name) for name in table_columns}
                has_wavelength = any(alias in canonical for alias in _WAVELENGTH_ALIASES)
                has_flux = any(alias in canonical for alias in _FLUX_ALIASES)
                if has_wavelength and has_flux:
                    return "calibrated_flux_spectrum", details, None
        except OSError as exc:
            details["inspection_error"] = str(exc)
            return "unreadable_product", details, None

    return "reduction_required", details, None


def write_joint_spectrum_tbl(
    spectrum: ExtractedDepthSpectrum,
    output_path: str | Path,
    *,
    target: str,
    mode: str,
    instrument: str | None,
    data_uri: str,
    overwrite: bool = False,
) -> Path:
    """Write a normalized depth spectrum in the repository's NASA/IPAC format."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    value_column = "SPECTRANSDEP" if mode == "transmission" else "ESPECLIPDEP"
    table = Table()
    table["CENTRALWAVELNG"] = spectrum.wavelength_angstrom / 1.0e4
    table["CENTRALWAVELNG"].unit = "micron"
    table["BANDWIDTH"] = spectrum.bandwidth_angstrom / 1.0e4
    table["BANDWIDTH"].unit = "micron"
    table[value_column] = spectrum.depth_fraction * 100.0
    table[value_column].unit = "%"
    table[f"{value_column}ERR1"] = spectrum.uncertainty_fraction * 100.0
    table[f"{value_column}ERR1"].unit = "%"
    table[f"{value_column}ERR2"] = -spectrum.uncertainty_fraction * 100.0
    table[f"{value_column}ERR2"].unit = "%"
    table.meta["keywords"] = OrderedDict(
        (
            ("PL_NAME", {"value": target}),
            ("SPEC_TYPE", {"value": "Transmission" if mode == "transmission" else "Eclipse"}),
            ("INSTRUMENT", {"value": instrument or "unknown"}),
            ("FACILITY", {"value": "MAST"}),
            ("REFERENCE", {"value": "MAST archive product"}),
            ("MAST_URI", {"value": data_uri}),
            ("SOURCE_FILE", {"value": spectrum.source_location}),
            ("VALUE_COLUMN", {"value": spectrum.value_column}),
        )
    )
    table.write(output_path, format="ascii.ipac", overwrite=overwrite)
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_product_path(root: Path, record: ProductRecord) -> Path:
    collection = _clean_component(record.collection, "collection")
    observation = _clean_component(record.obs_id or record.obsid, "observation")
    filename = _clean_component(record.filename, "product")
    return root / "products" / collection / observation / filename


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _summary_text(value: Any, fallback: str = "unknown") -> str:
    value = _plain_value(value)
    text = str(value).strip() if value is not None else ""
    return text or fallback


def build_completeness_summary(
    observations: Table,
    records: Sequence[ProductRecord],
) -> dict[str, Any]:
    """Build auditable observation/product counts for a fetch manifest."""

    observation_groups: Counter[tuple[str, str, str, str, str]] = Counter()
    query_match_counts: Counter[str] = Counter()
    for row in observations:
        observation_groups[
            (
                _summary_text(_row_value(row, ("proposal_id",))),
                _summary_text(_row_value(row, ("target_name", "target_name_hlsp"))),
                _summary_text(_row_value(row, ("obs_collection",))),
                _summary_text(_row_value(row, ("instrument_name",))),
                _summary_text(_row_value(row, ("filters", "filter_name", "filter"))),
            )
        ] += 1
        matches = str(_row_value(row, ("query_matches",), "") or "")
        query_match_counts.update(item for item in matches.split("|") if item)

    product_groups: defaultdict[
        tuple[str, str, str, str, str, str], dict[str, int]
    ] = defaultdict(lambda: {"total": 0, "selected": 0, "size_bytes": 0})
    classification_counts: Counter[str] = Counter()
    selected_classification_counts: Counter[str] = Counter()
    for record in records:
        key = (
            _summary_text(record.proposal_id),
            _summary_text(record.target_name),
            _summary_text(record.collection),
            _summary_text(record.instrument),
            _summary_text(record.filter_name),
            record.archive_classification,
        )
        product_groups[key]["total"] += 1
        product_groups[key]["selected"] += int(record.selected)
        product_groups[key]["size_bytes"] += record.size_bytes or 0
        classification_counts[record.archive_classification] += 1
        if record.selected:
            selected_classification_counts[record.archive_classification] += 1

    return {
        "observation_groups": [
            {
                "proposal_id": key[0],
                "target_name": key[1],
                "collection": key[2],
                "instrument": key[3],
                "filter": key[4],
                "count": count,
            }
            for key, count in sorted(observation_groups.items())
        ],
        "observations_by_query_match": dict(sorted(query_match_counts.items())),
        "product_groups": [
            {
                "proposal_id": key[0],
                "target_name": key[1],
                "collection": key[2],
                "instrument": key[3],
                "filter": key[4],
                "classification": key[5],
                **counts,
            }
            for key, counts in sorted(product_groups.items())
        ],
        "products_by_classification": {
            classification: {
                "total": count,
                "selected": selected_classification_counts[classification],
            }
            for classification, count in sorted(classification_counts.items())
        },
    }


def write_manifest(
    path: str | Path,
    *,
    query: MastQueryConfig,
    records: Sequence[ProductRecord],
    observations_count: int,
    products_count: int,
    observations: Table | None = None,
    selection: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "query": asdict(query),
        "selection": dict(selection or {}),
        "observations_count": int(observations_count),
        "products_count": int(products_count),
        "selected_count": sum(record.selected for record in records),
        "selected_bytes": sum(
            (record.size_bytes or 0) for record in records if record.selected
        ),
        "completeness": build_completeness_summary(
            observations if observations is not None else Table(), records
        ),
        "products": [asdict(record) for record in records],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def download_selected_products(
    records: Sequence[ProductRecord],
    *,
    output_root: str | Path,
    mode: str,
    target: str,
    overrides: ColumnOverrides | None = None,
    normalize_direct: bool = True,
    overwrite: bool = False,
    observations_client: Any | None = None,
    manifest_callback: Any | None = None,
) -> None:
    """Download selected products and update each record in place."""

    if observations_client is None:
        from astroquery.mast import Observations as observations_client

    output_root = Path(output_root)
    normalized_root = output_root / "normalized"
    for record in records:
        if not record.selected:
            continue
        local_path = _local_product_path(output_root, record)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        record.local_path = str(local_path)

        if local_path.exists() and not overwrite:
            record.download_status = "cached"
        else:
            temporary = local_path.with_name(f".{local_path.name}.part")
            if temporary.exists():
                temporary.unlink()
            try:
                status, message, _url = observations_client.download_file(
                    record.data_uri,
                    local_path=str(temporary),
                    cache=False,
                )
            except Exception as exc:
                record.download_status = "ERROR"
                record.download_message = str(exc)
                if temporary.exists():
                    temporary.unlink()
                if manifest_callback is not None:
                    manifest_callback()
                continue
            record.download_status = str(status)
            record.download_message = None if message is None else str(message)
            if str(status).upper() == "COMPLETE" and temporary.exists():
                os.replace(temporary, local_path)
            elif temporary.exists():
                temporary.unlink()
        if not local_path.exists():
            if manifest_callback is not None:
                manifest_callback()
            continue

        record.sha256 = _sha256(local_path)
        classification, details, extracted = inspect_downloaded_product(
            local_path, mode=mode, overrides=overrides
        )
        record.content_classification = classification
        record.content_details = details
        if extracted is None:
            record.normalization_status = "reduction required"
        elif not normalize_direct:
            record.normalization_status = "direct product found; normalization disabled"
        else:
            stem = _clean_component(Path(record.filename).stem, "spectrum")
            product_relative = local_path.relative_to(output_root / "products")
            normalized_path = normalized_root / product_relative.parent / f"{stem}.tbl"
            normalized_existed = normalized_path.exists()
            try:
                if not normalized_existed or overwrite:
                    write_joint_spectrum_tbl(
                        extracted,
                        normalized_path,
                        target=target,
                        mode=mode,
                        instrument=record.instrument,
                        data_uri=record.data_uri,
                        overwrite=overwrite,
                    )
                record.normalized_path = str(normalized_path)
                record.normalization_status = (
                    "normalized (cached)" if normalized_existed and not overwrite else "normalized"
                )
                record.recommended_cli_flag = (
                    "--joint-spectrum-tbl"
                    if extracted.wavelength_angstrom.size >= 5
                    else "--bandpass-tbl"
                )
            except (OSError, ValueError, TypeError) as exc:
                record.normalization_status = "normalization error"
                record.content_details = dict(record.content_details or {})
                record.content_details["normalization_error"] = str(exc)
        if manifest_callback is not None:
            manifest_callback()


def _positive_limit_gb(value: str) -> int | None:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("size limits must be non-negative")
    if parsed == 0:
        return None
    return int(parsed * 1024**3)


def _positive_limit_count(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("count limits must be non-negative")
    return None if parsed == 0 else parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and fetch MAST low-resolution exoplanet spectroscopy products. "
            "Only explicit transit/eclipse depth tables are normalized automatically."
        )
    )
    parser.add_argument("--target", help="Target or host-star name resolved by MAST")
    parser.add_argument(
        "--planet",
        default=None,
        help=(
            "Planet name used for local paths and normalized metadata when the MAST "
            "target is the host star (default: reuse --target)"
        ),
    )
    parser.add_argument("--mode", required=True, choices=("transmission", "emission"))
    parser.add_argument(
        "--radius-deg",
        type=float,
        default=0.001,
        help="Cone-search radius in degrees (default: 0.001, or 3.6 arcsec)",
    )
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="MAST observation collection; repeat as needed (default: HST and JWST)",
    )
    parser.add_argument(
        "--archive-target-name",
        action="append",
        default=[],
        help=(
            "Exact MAST target_name label to union with object/cone discovery; "
            "repeat as needed"
        ),
    )
    parser.add_argument(
        "--alias-registry",
        type=Path,
        default=DEFAULT_ALIAS_REGISTRY,
        help=f"Curated exact target-name registry (default: {DEFAULT_ALIAS_REGISTRY})",
    )
    parser.add_argument(
        "--no-alias-registry",
        action="store_true",
        help="Do not add aliases from the local registry",
    )
    parser.add_argument("--proposal-id", action="append", default=[])
    parser.add_argument("--obs-id", action="append", default=[])
    parser.add_argument(
        "--data-uri",
        action="append",
        default=[],
        help="Exact mast: product URI; repeat for a preselected product list",
    )
    parser.add_argument(
        "--uri-file",
        type=Path,
        action="append",
        default=[],
        help="Text file containing one exact mast: product URI per line",
    )
    parser.add_argument("--instrument", action="append", default=[])
    parser.add_argument(
        "--product-profile",
        choices=PRODUCT_PROFILES,
        default="direct",
        help=(
            "direct downloads extracted/reduced spectra; reduction downloads calibrated "
            "time-series inputs; all includes all non-raw scientific products"
        ),
    )
    parser.add_argument(
        "--product-subgroup",
        action="append",
        default=[],
        help="Product subgroup to select; repeat to override the default spectral set",
    )
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--include-2d", action="store_true")
    parser.add_argument("--include-proprietary", action="store_true")
    parser.add_argument("--token-env", default="MAST_API_TOKEN")
    parser.add_argument("--max-products", type=_positive_limit_count, default=200)
    parser.add_argument("--max-file-gb", type=_positive_limit_gb, default=2 * 1024**3)
    parser.add_argument("--max-total-gb", type=_positive_limit_gb, default=20 * 1024**3)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--query-only", action="store_true", help="Write manifests without downloads")
    action.add_argument("--download", action="store_true", help="Download the selected products")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")

    parser.add_argument("--wavelength-column")
    parser.add_argument("--value-column")
    parser.add_argument("--uncertainty-column")
    parser.add_argument("--uncertainty-low-column")
    parser.add_argument("--uncertainty-high-column")
    parser.add_argument("--bandwidth-column")
    parser.add_argument("--wavelength-unit")
    parser.add_argument("--value-unit")
    parser.add_argument("--value-kind", choices=("depth", "radius_ratio", "flux_ratio"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    data_uris = list(args.data_uri)
    for uri_file in args.uri_file:
        data_uris.extend(uri_file.read_text().splitlines())
    data_uris = [uri.strip() for uri in data_uris if uri.strip() and not uri.lstrip().startswith("#")]
    if data_uris and (
        args.proposal_id
        or args.obs_id
        or args.instrument
        or args.product_subgroup
        or args.archive_target_name
    ):
        parser.error("--data-uri/--uri-file cannot be combined with archive query filters")
    if not (args.planet or args.target):
        parser.error("provide --planet when querying only by observation ID or exact data URI")

    archive_target_names: tuple[str, ...] = ()
    if not data_uris:
        archive_target_names = resolve_archive_target_names(
            target=args.target,
            planet=args.planet,
            explicit_names=args.archive_target_name,
            registry_path=args.alias_registry,
            use_registry=not args.no_alias_registry,
        )
        if not args.target and not args.obs_id and not archive_target_names:
            parser.error(
                "provide --target, --archive-target-name, --obs-id, --data-uri, or --uri-file"
            )

    collections = tuple(args.collections or DEFAULT_COLLECTIONS)
    query = MastQueryConfig(
        target=args.target,
        mode=args.mode,
        planet=args.planet,
        radius_deg=float(args.radius_deg),
        collections=collections,
        proposal_ids=tuple(args.proposal_id),
        observation_ids=tuple(args.obs_id),
        instruments=tuple(args.instrument),
        archive_target_names=archive_target_names,
    )
    output_root = args.output_dir or (
        Path("input") / "lrs" / args.mode / target_slug(args.planet or args.target) / "mast"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    from astroquery.mast import Observations

    if args.include_proprietary:
        token = os.getenv(args.token_env)
        if not token:
            raise RuntimeError(
                f"--include-proprietary requires a token in the {args.token_env} environment variable."
            )
        Observations.login(token=token, store_token=False)

    if data_uris:
        observations = Table()
        products = Table()
        records = build_uri_records(data_uris)
        if args.max_products is not None:
            for record in records[args.max_products :]:
                record.selected = False
                record.selection_reason = "excluded by product-count limit"
        print(f"Using {len(records)} exact MAST product URIs.")
    else:
        query_label = args.target or ", ".join(args.obs_id)
        print(f"Querying MAST for {query_label!r} in {', '.join(collections)}...")
        if archive_target_names and not args.obs_id:
            print("Adding exact target_name queries: " + ", ".join(archive_target_names))
        observations, products = query_mast(query, observations_client=Observations)
        observations.write(output_root / "observations.ecsv", format="ascii.ecsv", overwrite=True)
        if len(products) > 0:
            products.write(output_root / "products.ecsv", format="ascii.ecsv", overwrite=True)
        records = build_product_records(observations, products)
        select_products(
            records,
            product_profile=args.product_profile,
            product_subgroups=args.product_subgroup or None,
            include_raw=args.include_raw,
            include_2d=args.include_2d,
            include_proprietary=args.include_proprietary,
            max_products=args.max_products,
            max_file_bytes=args.max_file_gb,
            max_total_bytes=args.max_total_gb,
        )
    manifest_path = output_root / "manifest.json"

    def save_manifest() -> None:
        write_manifest(
            manifest_path,
            query=query,
            records=records,
            observations_count=len(observations),
            products_count=len(products),
            observations=observations,
            selection={
                "product_profile": args.product_profile,
                "product_subgroups": list(args.product_subgroup),
                "include_raw": bool(args.include_raw),
                "include_2d": bool(args.include_2d),
                "include_proprietary": bool(args.include_proprietary),
                "max_products": args.max_products,
                "max_file_bytes": args.max_file_gb,
                "max_total_bytes": args.max_total_gb,
            },
        )

    save_manifest()
    selected = [record for record in records if record.selected]
    selected_bytes = sum(record.size_bytes or 0 for record in selected)
    if data_uris:
        print(f"Selected {len(selected)} exact products (archive sizes unavailable before download).")
    else:
        print(
            f"Found {len(observations)} observations and {len(records)} unique products; "
            f"selected {len(selected)} products ({selected_bytes / 1024**3:.2f} GiB)."
        )
    print(f"Manifest: {manifest_path}")

    if not args.download:
        print("No files downloaded. Re-run with --download after reviewing the manifest.")
        return 0

    overrides = ColumnOverrides(
        wavelength_column=args.wavelength_column,
        value_column=args.value_column,
        uncertainty_column=args.uncertainty_column,
        uncertainty_low_column=args.uncertainty_low_column,
        uncertainty_high_column=args.uncertainty_high_column,
        bandwidth_column=args.bandwidth_column,
        wavelength_unit=args.wavelength_unit,
        value_unit=args.value_unit,
        value_kind=args.value_kind,
    )
    download_selected_products(
        records,
        output_root=output_root,
        mode=args.mode,
        target=args.planet or args.target,
        overrides=overrides,
        normalize_direct=not args.no_normalize,
        overwrite=args.overwrite,
        observations_client=Observations,
        manifest_callback=save_manifest,
    )
    save_manifest()
    normalized = [record for record in records if record.normalized_path]
    needs_reduction = [
        record
        for record in records
        if record.selected and record.normalization_status == "reduction required"
    ]
    print(f"Normalized {len(normalized)} direct depth products.")
    print(f"Downloaded products requiring reduction: {len(needs_reduction)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
