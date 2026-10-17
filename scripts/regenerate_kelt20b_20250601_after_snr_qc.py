#!/usr/bin/env python
"""Rebuild the KELT-20b 20250601 transmission products after exposure QC.

This is deliberately target-specific.  It recalibrates the two arm-edge
trims from the filtered raw exposures, requires an accepted post-SYSREM
manifest, overwrites the canonical time-series and collapse-source products
for this epoch, and rebuilds both collapsed spectra.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.edge_trim_calibration import (
    CalibrationSettings,
    print_summary,
    run_edge_trim_calibration,
)


PLANET = "KELT-20b"
PLANET_SLUG = "kelt20b"
EPHEMERIS = "Duck24"
EPOCH = "20250601"
CANONICAL_ROOT = (PROJECT_ROOT / "input" / "hrs" / "transmission" / PLANET_SLUG / EPOCH)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate and overwrite only the canonical KELT-20b 20250601 "
            "transmission products using the configured exposure selection."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the calibration and overwrite the canonical products.",
    )
    return parser


def _run(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _calibrate_edges() -> Path:
    print("\n=== Fresh 20250601 edge-trim calibration ===", flush=True)
    result = run_edge_trim_calibration(
        planet=PLANET,
        ephemeris=EPHEMERIS,
        mode="transmission",
        datasets=((EPOCH, "red"), (EPOCH, "blue")),
        settings=CalibrationSettings(),
    )
    print_summary(result)

    manifest = result["manifest"]
    manifest_path = Path(result["manifest_path"]).resolve()
    if manifest["overall_status"] != "accepted_post_sysrem":
        raise RuntimeError(
            "Fresh edge calibration was not accepted_post_sysrem. "
            f"Canonical products were not touched. Inspect {result['run_dir']}"
        )

    return manifest_path


def _regenerate(manifest_path: Path) -> None:
    common = [
        sys.executable,
        "-m",
        "dataio.prepare_retrieval_timeseries",
        "--planet",
        PLANET,
        "--ephemeris",
        EPHEMERIS,
        "--epoch",
        EPOCH,
        "--arm",
        "full",
        "--molecfit",
        "--regrid",
        "--run-sysrem",
        "--apply-stellar-rest",
        "--edge-trim-manifest",
        str(manifest_path),
    ]
    _run(
        "Canonical retrieval time series",
        common
        + [
            "--phase-bin",
            "full",
            "--product-kind",
            "timeseries",
            "--subtract-median",
        ],
    )
    _run(
        "Canonical collapse-source cubes",
        common
        + [
            "--phase-bin",
            "all",
            "--product-kind",
            "collapse-source",
            "--no-subtract-median",
        ],
    )
    _run(
        "Mandatory shared-basis LSD Doppler shadows",
        [
            sys.executable,
            "-m",
            "spectroscopy.doppler_shadow",
            "--planet",
            PLANET,
            "--ephemeris",
            EPHEMERIS,
            "--shadow-source",
            "Recommended",
            "--epoch",
            EPOCH,
            "--arm",
            "both",
        ],
    )
    _run(
        "Canonical collapsed spectra",
        [
            sys.executable,
            "-m",
            "dataio.collapse_transmission_timeseries_to_1d",
            "--planet",
            PLANET,
            "--ephemeris",
            EPHEMERIS,
            "--epoch",
            EPOCH,
            "--arm",
            "full",
            "--shadow-source",
            "Recommended",
            "--kp-kms",
            "169.0",
            "--bin-size",
            "1",
        ],
    )


def main() -> int:
    args = create_parser().parse_args()
    if not args.execute:
        print(
            "This script will recalibrate edge trims and overwrite only:\n"
            f"  {CANONICAL_ROOT / 'blue'}\n"
            f"  {CANONICAL_ROOT / 'red'}\n\n"
            "Run it with --execute to proceed."
        )
        return 0

    manifest_path = _calibrate_edges()
    _regenerate(manifest_path)
    print(
        "\nCOMPLETE: KELT-20b 20250601 blue/red time series, "
        "collapse sources, and collapsed spectra were regenerated."
    )
    print(f"Edge-trim manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
