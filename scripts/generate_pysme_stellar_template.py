#!/usr/bin/env python
"""Generate a frozen, intrinsic PySME spectrum for stellar LSD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataio.stellar_lsd import (  # noqa: E402
    STELLAR_TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_KIND,
    velocity_to_doppler_factor,
)
import config_utils  # noqa: E402


DEFAULT_SEGMENTS = ((4200.0, 5600.0), (6100.0, 7500.0))


def _vald_header_geometry(path: Path) -> tuple[float, float, float]:
    fields = path.read_text(encoding="utf-8", errors="strict").splitlines()[0].split(",")
    if len(fields) < 5:
        raise ValueError("Unexpected VALD Extract Stellar header.")
    lower = float(fields[0])
    upper = float(fields[1])
    vmicro = float(fields[4].strip().split(maxsplit=1)[0])
    return lower, upper, vmicro


def _segments_within_linelist(lower: float, upper: float) -> tuple[tuple[float, float], ...]:
    segments = tuple(
        (max(lower, segment_lower), min(upper, segment_upper))
        for segment_lower, segment_upper in DEFAULT_SEGMENTS
        if max(lower, segment_lower) < min(upper, segment_upper)
    )
    if not segments:
        raise ValueError(
            f"VALD wavelength range {lower:g}:{upper:g} A does not overlap the PEPSI arms."
        )
    return segments


def _parse_segment(value: str) -> tuple[float, float]:
    try:
        lower_text, upper_text = value.split(":", 1)
        lower = float(lower_text)
        upper = float(upper_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "segments must use LOWER:UPPER Angstrom syntax"
        ) from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise argparse.ArgumentTypeError("segment bounds must be finite, positive, and increasing")
    return lower, upper


def _velocity_sampled_wavelengths(
    lower: float,
    upper: float,
    step_kms: float,
) -> np.ndarray:
    ratio = float(velocity_to_doppler_factor(step_kms))
    count = int(np.floor(np.log(upper / lower) / np.log(ratio))) + 1
    wavelength = lower * ratio ** np.arange(count, dtype=float)
    if wavelength[-1] < upper:
        wavelength = np.append(wavelength, upper)
    return wavelength


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_vald_for_pysme(path: Path, vald_file_class: type) -> tuple[object, str]:
    """Normalize the emailed VALD header that PySME 1.0.2 parses strictly."""
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines(keepends=True)
    fields = lines[0].split(",", 4)
    if len(fields) != 5:
        raise ValueError("Unexpected VALD Extract Stellar header.")
    trailing = fields[4].strip()
    pieces = trailing.split(maxsplit=1)
    if len(pieces) != 2 or pieces[1] != "Wavelength region, lines selected, lines processed, Vmicro":
        return vald_file_class(str(path)), "none"
    float(pieces[0])
    fields[4] = f" {pieces[0]}, {pieces[1]}\n"
    lines[0] = ",".join(fields)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".vald",
            delete=False,
        ) as handle:
            handle.writelines(lines)
            temporary_path = Path(handle.name)
        return vald_file_class(str(temporary_path)), "inserted_missing_vmicro_header_comma"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a continuum-normalized, disk-integrated PySME template "
            "with no radial, rotational, macroturbulent, or instrumental shift."
        )
    )
    parser.add_argument("--planet", default="KELT-20b")
    parser.add_argument("--parameter-source", default="Recommended")
    parser.add_argument("--linelist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teff-k", type=float, default=None)
    parser.add_argument("--logg", type=float, default=None)
    parser.add_argument("--metallicity", type=float, default=None)
    parser.add_argument("--vmicro-kms", type=float, default=None)
    parser.add_argument("--atmosphere", default="atlas9_vmic2.0.sav")
    parser.add_argument("--wavelength-step-kms", type=float, default=0.5)
    parser.add_argument(
        "--segment",
        action="append",
        type=_parse_segment,
        default=None,
        help="Vacuum-Angstrom segment LOWER:UPPER; repeat for multiple segments.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    linelist_path = args.linelist.resolve()
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if not linelist_path.is_file():
        raise FileNotFoundError(f"VALD line list not found: {linelist_path}")
    params = config_utils.get_params(args.planet, args.parameter_source)
    linelist_lower, linelist_upper, linelist_vmicro = _vald_header_geometry(
        linelist_path
    )
    parameter_provenance: dict[str, str] = {}
    if args.teff_k is None:
        args.teff_k = float(params.get("T_star", np.nan))
        parameter_provenance["teff_k"] = f"config:{args.parameter_source}:T_star"
    else:
        parameter_provenance["teff_k"] = "command_line"
    if args.logg is None:
        configured_logg = float(params.get("logg_star", np.nan))
        if np.isfinite(configured_logg):
            args.logg = configured_logg
            parameter_provenance["logg"] = f"config:{args.parameter_source}:logg_star"
        else:
            stellar_mass = float(params.get("M_star", np.nan))
            stellar_radius = float(params.get("R_star", np.nan))
            if not np.isfinite(stellar_mass) or not np.isfinite(stellar_radius):
                raise ValueError(
                    "Template logg requires finite logg_star or M_star and R_star."
                )
            args.logg = 4.438067627 + np.log10(stellar_mass / stellar_radius**2)
            parameter_provenance["logg"] = "derived_from_config_M_star_R_star"
    else:
        parameter_provenance["logg"] = "command_line"
    if args.metallicity is None:
        args.metallicity = float(params.get("Fe_H", np.nan))
        parameter_provenance["metallicity"] = f"config:{args.parameter_source}:Fe_H"
    else:
        parameter_provenance["metallicity"] = "command_line"
    if args.vmicro_kms is None:
        args.vmicro_kms = linelist_vmicro
        parameter_provenance["vmicro_kms"] = "vald_extract_stellar_header"
    else:
        parameter_provenance["vmicro_kms"] = "command_line"
    if args.teff_k <= 0.0 or not np.isfinite(args.teff_k):
        raise ValueError("--teff-k must be finite and positive")
    if not np.isfinite(args.logg) or not np.isfinite(args.metallicity):
        raise ValueError("--logg and --metallicity must be finite")
    if args.vmicro_kms < 0.0 or not np.isfinite(args.vmicro_kms):
        raise ValueError("--vmicro-kms must be finite and non-negative")
    if args.wavelength_step_kms <= 0.0 or not np.isfinite(args.wavelength_step_kms):
        raise ValueError("--wavelength-step-kms must be finite and positive")

    header = linelist_path.read_text(encoding="utf-8", errors="replace")[:4096]
    if "WL_vac(A)" not in header:
        raise ValueError(
            "The VALD input must declare vacuum wavelengths with a WL_vac(A) column."
        )

    try:
        import pysme
        from pysme.abund import Abund
        from pysme.linelist.vald import ValdFile
        from pysme.sme import SME_Structure
        from pysme.synthesize import synthesize_spectrum
    except ImportError as exc:
        raise RuntimeError(
            "PySME is required. Install the pinned pysme-astro dependency in the retrieval environment."
        ) from exc

    segments = tuple(
        args.segment
        or _segments_within_linelist(linelist_lower, linelist_upper)
    )
    wavelength_segments = [
        _velocity_sampled_wavelengths(lower, upper, args.wavelength_step_kms)
        for lower, upper in segments
    ]
    sme = SME_Structure()
    sme.abund = Abund.solar()
    sme.teff = float(args.teff_k)
    sme.logg = float(args.logg)
    sme.monh = float(args.metallicity)
    sme.vmic = float(args.vmicro_kms)
    sme.vmac = 0.0
    sme.vsini = 0.0
    sme.vrad = 0.0
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    sme.normalize_by_continuum = True
    sme.specific_intensities_only = False
    sme.ipres = 0.0
    sme.atmo.source = str(args.atmosphere)
    sme.linelist, header_normalization = _load_vald_for_pysme(
        linelist_path,
        ValdFile,
    )
    sme.wave = wavelength_segments

    print(
        "Synthesizing intrinsic PySME template: "
        f"Teff={sme.teff:.0f} K, logg={sme.logg:.3f}, [M/H]={sme.monh:+.3f}, "
        f"vmicro={sme.vmic:.2f} km/s, atmosphere={sme.atmo.source}"
    )
    synthesized = synthesize_spectrum(sme)
    wavelength = np.concatenate(
        [np.asarray(segment, dtype=float) for segment in synthesized.wave]
    )
    flux = np.concatenate(
        [np.asarray(segment, dtype=float) for segment in synthesized.synth]
    )
    if wavelength.shape != flux.shape or wavelength.ndim != 1:
        raise RuntimeError("PySME returned incompatible wavelength and flux arrays.")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.diff(wavelength) > 0.0):
        raise RuntimeError("PySME returned invalid or unordered wavelengths.")
    if not np.all(np.isfinite(flux)) or float(np.max(1.0 - flux)) <= 1.0e-4:
        raise RuntimeError("PySME returned an invalid or featureless normalized spectrum.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelength_vacuum_angstrom=wavelength,
        normalized_flux=flux,
    )
    metadata = {
        "schema_version": STELLAR_TEMPLATE_SCHEMA_VERSION,
        "template_kind": TEMPLATE_KIND,
        "planet": str(args.planet),
        "stellar_parameter_source": str(args.parameter_source),
        "stellar_parameter_provenance": parameter_provenance,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": shlex.join(sys.argv),
        "pysme_version": str(pysme.__version__),
        "teff_k": float(args.teff_k),
        "logg": float(args.logg),
        "metallicity_dex": float(args.metallicity),
        "vmicro_kms": float(args.vmicro_kms),
        "radial_velocity_kms": 0.0,
        "vsini_kms": 0.0,
        "vmacro_kms": 0.0,
        "instrumental_broadening": "none",
        "atmosphere": str(args.atmosphere),
        "line_formation": "LTE",
        "continuum_normalized": True,
        "disk_integrated_flux": True,
        "wavelength_medium": "vacuum",
        "wavelength_frame": "stellar_rest",
        "wavelength_step_kms": float(args.wavelength_step_kms),
        "segments_vacuum_angstrom": [[float(a), float(b)] for a, b in segments],
        "n_template_pixels": int(wavelength.size),
        "linelist_path": str(linelist_path),
        "linelist_sha256": _sha256(linelist_path),
        "linelist_header_normalization": header_normalization,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {wavelength.size} template pixels: {output_path}")
    print(f"Saved provenance: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
