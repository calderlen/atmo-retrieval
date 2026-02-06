#!/usr/bin/env python
"""Import NASA Exoplanet Archive low-res spectra (.tbl) into .npy files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataio.load import load_nasa_archive_spectrum, parse_nasa_archive_tbl


def _sanitize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "")


def _infer_mode(metadata: dict[str, str]) -> str:
    spec_type = metadata.get("SPEC_TYPE", "").lower()
    if "eclipse" in spec_type:
        return "emission"
    if "transit" in spec_type or "transmission" in spec_type:
        return "transmission"
    raise ValueError("Could not infer mode from SPEC_TYPE; use --mode.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert NASA Exoplanet Archive .tbl spectra to .npy files",
    )
    parser.add_argument("--tbl", required=True, help="Path to .tbl file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: input/spectra/nasa_archive/<planet>/<spec_num>)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "emission", "transmission"],
        default="auto",
        help="Spectrum mode (auto uses SPEC_TYPE)",
    )
    parser.add_argument(
        "--value-column",
        default=None,
        help="Override value column name (e.g., ESPECLIPDEP)",
    )
    parser.add_argument(
        "--err1-column",
        default=None,
        help="Override +error column name",
    )
    parser.add_argument(
        "--err2-column",
        default=None,
        help="Override -error column name",
    )

    args = parser.parse_args()

    tbl_path = Path(args.tbl)
    if not tbl_path.exists():
        raise FileNotFoundError(f"File not found: {tbl_path}")

    metadata, _columns, _data_by_col, _units_by_col = parse_nasa_archive_tbl(tbl_path)

    if args.mode == "auto":
        mode = _infer_mode(metadata)
    else:
        mode = args.mode

    wav_angstrom, spectrum, sigma, metadata = load_nasa_archive_spectrum(
        tbl_path,
        mode=mode,
        value_column=args.value_column,
        err1_column=args.err1_column,
        err2_column=args.err2_column,
    )

    if args.output_dir is None:
        planet_name = metadata.get("PL_NAME", tbl_path.stem)
        spec_num = metadata.get("SPEC_NUM", tbl_path.stem)
        planet_dir = _sanitize_name(planet_name)
        output_dir = Path("input/spectra/nasa_archive") / planet_dir / str(spec_num)
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "emission" if mode == "emission" else "transmission"

    np.save(output_dir / f"wavelength_{suffix}.npy", wav_angstrom)
    np.save(output_dir / f"spectrum_{suffix}.npy", spectrum)
    np.save(output_dir / f"uncertainty_{suffix}.npy", sigma)

    meta_out = {
        "source_tbl": str(tbl_path),
        "mode": mode,
        "n_points": int(len(wav_angstrom)),
        "value_column": args.value_column,
        "err1_column": args.err1_column,
        "err2_column": args.err2_column,
        "metadata": metadata,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved {len(wav_angstrom)} points to {output_dir}")
    print(f"  - wavelength_{suffix}.npy")
    print(f"  - spectrum_{suffix}.npy")
    print(f"  - uncertainty_{suffix}.npy")
    print("  - metadata.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
