"""Bandpass response loaders for joint retrievals."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np

import config


def _download_tess_bandpass(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = urlopen(config.TESS_BANDPASS_URL).read().decode("utf-8")
    path.write_text(text)


def load_tess_bandpass(
    path: str | Path | None = None,
    *,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load the TESS response function CSV.

    Returns wavelength in meters and dimensionless response.
    """
    bandpass_path = Path(path) if path is not None else config.TESS_BANDPASS_PATH

    if not bandpass_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Bandpass file not found: {bandpass_path}. "
                "Provide bandpass_path or allow download."
            )
        _download_tess_bandpass(bandpass_path)

    wavelength_nm: list[float] = []
    response: list[float] = []

    for raw_line in bandpass_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue

        left, right = line.split(",", 1)
        try:
            lam_nm = float(left.strip())
            rsp = float(right.strip())
        except ValueError:
            continue

        wavelength_nm.append(lam_nm)
        response.append(rsp)

    wavelength_nm_arr = np.asarray(wavelength_nm, dtype=float)
    response_arr = np.asarray(response, dtype=float)

    finite_mask = np.isfinite(wavelength_nm_arr) & np.isfinite(response_arr)
    positive_mask = response_arr > 0.0
    mask = finite_mask & positive_mask

    wavelength_m = wavelength_nm_arr[mask] * 1.0e-9
    response_clean = response_arr[mask]

    if wavelength_m.size < 2:
        raise ValueError(
            f"Bandpass file {bandpass_path} does not contain enough valid points."
        )

    order = np.argsort(wavelength_m)
    wavelength_m = wavelength_m[order]
    response_clean = response_clean[order]
    return wavelength_m, response_clean, bandpass_path
