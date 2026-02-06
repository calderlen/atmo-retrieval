"""Load observed spectra and instrumental resolution data."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from exojax.utils.grids import wav2nu


def load_observed_spectrum(
    wav_path: str,
    spectrum_path: str,
    uncertainty_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    wav_obs = np.load(wav_path)
    spectrum = np.load(spectrum_path)
    uncertainty = np.load(uncertainty_path)

    # Wavelength in Angstroms, convert to wavenumber
    inst_nus = wav2nu(wav_obs, "AA")
    return wav_obs, spectrum, uncertainty, inst_nus


def load_resolution_curve(fits_path: str) -> np.ndarray:
    """Load instrumental resolution curve from FITS table."""
    with fits.open(fits_path) as hdul:
        data = np.asarray([list(row) for row in hdul[1].data])
    return data


def _parse_metadata_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text.startswith("\\"):
        return None
    text = text[1:].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    return key, value


def _parse_pipe_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def _parse_token(token: str) -> float | str | None:
    if token.lower() == "null":
        return None
    try:
        return float(token)
    except ValueError:
        return token


def _to_float_array(values: list[float | str | None]) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i, val in enumerate(values):
        if val is None:
            continue
        if isinstance(val, (float, int, np.floating, np.integer)):
            out[i] = float(val)
        else:
            try:
                out[i] = float(val)
            except (TypeError, ValueError):
                continue
    return out


def parse_nasa_archive_tbl(
    tbl_path: str | Path,
) -> tuple[dict[str, str], list[str], dict[str, list[float | str | None]], dict[str, str]]:
    """Parse NASA Exoplanet Archive .tbl format.

    Returns:
        metadata: key/value pairs from lines starting with '\\'
        columns: list of column names
        data_by_col: dict of column -> raw values (float/str/None)
        units_by_col: dict of column -> unit string (may be empty)
    """
    path = Path(tbl_path)
    lines = path.read_text().splitlines()

    metadata: dict[str, str] = {}
    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        meta = _parse_metadata_line(stripped)
        if meta is not None:
            key, value = meta
            metadata[key] = value
            continue

        if stripped.startswith("|"):
            header_rows.append(_parse_pipe_row(stripped))
            continue

        tokens = stripped.split()
        if not tokens:
            continue
        if all(token.lower() == "null" for token in tokens):
            continue
        data_rows.append(tokens)

    if not header_rows:
        raise ValueError(f"No header rows found in {path}")

    columns = header_rows[0]
    units_row = header_rows[2] if len(header_rows) >= 3 else []
    units_by_col = {
        col: (units_row[idx] if idx < len(units_row) else "")
        for idx, col in enumerate(columns)
    }

    data_by_col: dict[str, list[float | str | None]] = {col: [] for col in columns}
    n_cols = len(columns)

    for row in data_rows:
        if len(row) < n_cols:
            row = row + ["null"] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]

        for col, token in zip(columns, row):
            data_by_col[col].append(_parse_token(token))

    return metadata, columns, data_by_col, units_by_col


def _select_value_column(columns: list[str], mode: str | None) -> str:
    candidates = []
    if mode == "emission":
        candidates = [
            "ESPECLIPDEP",
            "SPECLIPDEP",
            "SPECDEP",
        ]
    elif mode == "transmission":
        candidates = [
            "SPECTRANSDEP",
            "SPECTRANDEP",
            "SPECTRANSDEPTH",
            "SPECDEP",
        ]

    for name in candidates:
        if name in columns:
            return name

    raise ValueError(
        "Could not infer value column from header. "
        f"Available columns: {columns}"
    )


def _select_error_columns(columns: list[str], value_col: str) -> tuple[str | None, str | None]:
    err1 = f"{value_col}ERR1"
    err2 = f"{value_col}ERR2"
    err1_col = err1 if err1 in columns else None
    err2_col = err2 if err2 in columns else None
    return err1_col, err2_col


def _convert_wavelength_to_angstrom(wavelength: np.ndarray, unit: str) -> np.ndarray:
    unit_norm = unit.lower().strip()
    if "micron" in unit_norm:
        return wavelength * 10000.0
    if unit_norm in {"nm", "nanometer", "nanometers"}:
        return wavelength * 10.0
    return wavelength


def _convert_spectrum_units(spectrum: np.ndarray, unit: str) -> np.ndarray:
    unit_norm = unit.lower().strip()
    if "ppm" in unit_norm:
        return spectrum / 1.0e6
    if "%" in unit_norm:
        return spectrum / 100.0
    return spectrum


def load_nasa_archive_spectrum(
    tbl_path: str | Path,
    *,
    mode: str | None = None,
    value_column: str | None = None,
    err1_column: str | None = None,
    err2_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Load a low-resolution spectrum from NASA Exoplanet Archive .tbl.

    Returns:
        wavelength_angstrom, spectrum, uncertainty, metadata
    """
    metadata, columns, data_by_col, units_by_col = parse_nasa_archive_tbl(tbl_path)

    spec_type = metadata.get("SPEC_TYPE", "").strip()
    if mode is None or mode == "auto":
        spec_type_lower = spec_type.lower()
        if "eclipse" in spec_type_lower:
            mode = "emission"
        elif "transit" in spec_type_lower or "transmission" in spec_type_lower:
            mode = "transmission"
        else:
            raise ValueError(
                "Could not infer mode from SPEC_TYPE; provide mode='emission' or 'transmission'."
            )

    if value_column is None:
        value_column = _select_value_column(columns, mode)

    if err1_column is None and err2_column is None:
        err1_column, err2_column = _select_error_columns(columns, value_column)

    if "CENTRALWAVELNG" not in data_by_col:
        raise ValueError("CENTRALWAVELNG column is required but missing.")

    wav_raw = _to_float_array(data_by_col["CENTRALWAVELNG"])
    spec_raw = _to_float_array(data_by_col[value_column])

    if err1_column is not None:
        err1 = _to_float_array(data_by_col[err1_column])
    else:
        err1 = np.full_like(spec_raw, np.nan)

    if err2_column is not None:
        err2 = _to_float_array(data_by_col[err2_column])
    else:
        err2 = np.full_like(spec_raw, np.nan)

    sigma = np.nanmax(np.vstack([np.abs(err1), np.abs(err2)]), axis=0)
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)

    wav_unit = units_by_col.get("CENTRALWAVELNG", "")
    val_unit = units_by_col.get(value_column, "")

    wav_angstrom = _convert_wavelength_to_angstrom(wav_raw, wav_unit)
    spectrum = _convert_spectrum_units(spec_raw, val_unit)
    sigma = _convert_spectrum_units(sigma, val_unit)

    mask = np.isfinite(wav_angstrom) & np.isfinite(spectrum) & np.isfinite(sigma)
    wav_angstrom = wav_angstrom[mask]
    spectrum = spectrum[mask]
    sigma = sigma[mask]

    return wav_angstrom, spectrum, sigma, metadata


class ResolutionInterpolator:
    """Interpolate instrumental resolving power vs wavelength."""

    def __init__(
        self,
        res_curve_path: str | None = None,
        constant_R: float | None = None,
    ) -> None:
        if constant_R is not None:
            self.constant_R = constant_R
            self.res_curve = None
        elif res_curve_path is not None:
            self.res_curve = load_resolution_curve(res_curve_path)
            self.constant_R = None
        else:
            raise ValueError("Must provide either res_curve_path or constant_R")

    def __call__(self, wavelength_nm: float | np.ndarray) -> float | np.ndarray:
        """Return resolving power R at given wavelength."""
        if self.constant_R is not None:
            if isinstance(wavelength_nm, np.ndarray):
                return np.full_like(wavelength_nm, self.constant_R)
            else:
                return self.constant_R
        else:
            return np.interp(
                wavelength_nm / 1000.0,
                self.res_curve[:, 0],
                self.res_curve[:, 2]
            )


def mask_telluric_regions(
    wav_obs: np.ndarray,
    spectrum: np.ndarray,
    uncertainty: np.ndarray,
    telluric_mask_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask heavily contaminated telluric regions."""
    if telluric_mask_file is None:
        # Default: mask strong telluric bands
        mask = np.ones(len(wav_obs), dtype=bool)

        # Mask O2 A-band (759-771 nm)
        mask &= (wav_obs < 759) | (wav_obs > 771)
    else:
        mask = np.load(telluric_mask_file)

    return wav_obs[mask], spectrum[mask], uncertainty[mask], mask
