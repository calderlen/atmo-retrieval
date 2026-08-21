#!/usr/bin/env python
"""Rebuild the KELT-20b 20250601 transmission products after exposure QC.

This is deliberately target-specific.  It recalibrates the two arm-edge
trims from the filtered raw exposures, requires an accepted post-SYSREM
manifest, overwrites the canonical time-series and collapse-source products
for this epoch, rebuilds both collapsed spectra, and verifies the permanent
red-ingress and paired-arm q_snr exclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.edge_trim_manifest import load_accepted_edge_trim_widths
from dataio.exposure_selection import (
    SCIENCE_EXPOSURE_SELECTION_POLICY,
    SNR_QUALITY_POLICY,
    canonical_exposure_id,
)
from pipeline.edge_trim_calibration import (
    CalibrationSettings,
    print_summary,
    run_edge_trim_calibration,
)


PLANET = "KELT-20b"
PLANET_SLUG = "kelt20b"
EPHEMERIS = "Duck24"
EPOCH = "20250601"
ARMS = ("blue", "red")
EXPECTED_USABLE_EXPOSURES = {
    "blue": 69,
    "red": 69,
}
EXPECTED_CONFIGURED_EXCLUDED_IDS = {
    "blue": {
        "pepsib.20250601.062",
        "pepsib.20250601.063",
        "pepsib.20250601.064",
        "pepsib.20250601.065",
    },
    "red": {
        "pepsir.20250601.041",
        "pepsir.20250601.042",
        "pepsir.20250601.043",
        "pepsir.20250601.044",
    },
}
EXPECTED_SNR_EXCLUDED_IDS = {
    "blue": {
        "pepsib.20250601.085",
        "pepsib.20250601.088",
        "pepsib.20250601.093",
        "pepsib.20250601.094",
        "pepsib.20250601.095",
    },
    "red": {
        "pepsir.20250601.064",
        "pepsir.20250601.067",
        "pepsir.20250601.072",
        "pepsir.20250601.073",
        "pepsir.20250601.074",
    },
}
CANONICAL_ROOT = (
    PROJECT_ROOT / "input" / "hrs" / "transmission" / PLANET_SLUG / EPOCH
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate and overwrite only the canonical KELT-20b 20250601 "
            "transmission products after applying permanent ingress-profile "
            "and paired-arm q_snr < 0.4 QC."
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
    if manifest.get("overall_status") != "accepted_post_sysrem":
        raise RuntimeError(
            "Fresh edge calibration was not accepted_post_sysrem. "
            f"Canonical products were not touched. Inspect {result['run_dir']}"
        )

    for arm in ARMS:
        selected_path, widths = load_accepted_edge_trim_widths(
            manifest_path,
            planet=PLANET,
            mode="transmission",
            epoch=EPOCH,
            arm=arm,
        )
        if selected_path.resolve() != manifest_path:
            raise RuntimeError(
                f"Edge manifest resolution changed unexpectedly for {arm}: "
                f"{selected_path}"
            )
        print(
            f"Accepted {arm} trim: left={widths[0]:.3f} A, "
            f"right={widths[1]:.3f} A"
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


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Expected output is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_selection(arm: str, product: str, manifest_path: Path) -> dict:
    metadata_path = CANONICAL_ROOT / arm / product / "timeseries_prep.json"
    metadata = _load_json(metadata_path)
    selection = metadata.get("science_exposure_selection", {})
    quality = selection.get("snr_quality", {})

    if selection.get("policy") != SCIENCE_EXPOSURE_SELECTION_POLICY:
        raise RuntimeError(f"{metadata_path}: stale exposure-selection policy")
    if quality.get("policy") != SNR_QUALITY_POLICY:
        raise RuntimeError(f"{metadata_path}: stale S/N policy")
    if not math.isclose(
        float(quality.get("hard_threshold", math.nan)),
        0.4,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(f"{metadata_path}: hard q_snr threshold is not 0.4")
    if quality.get("requires_both_arms_below_threshold") is not True:
        raise RuntimeError(f"{metadata_path}: paired-arm requirement is not active")
    if quality.get("missing_or_nonfinite_snr") != "pass":
        raise RuntimeError(f"{metadata_path}: missing S/N does not pass")
    if quality.get("fallback_estimator") is not None:
        raise RuntimeError(f"{metadata_path}: an unexpected fallback estimator is active")
    expected_usable = EXPECTED_USABLE_EXPOSURES[arm]
    if int(selection.get("n_usable_files", -1)) != expected_usable:
        raise RuntimeError(
            f"{metadata_path}: expected {expected_usable} usable files, "
            f"found {selection.get('n_usable_files')}"
        )

    configured_excluded = {
        canonical_exposure_id(name)
        for name in selection.get("configured_excluded_files", [])
    }
    if configured_excluded != EXPECTED_CONFIGURED_EXCLUDED_IDS[arm]:
        raise RuntimeError(
            f"{metadata_path}: unexpected configured exclusions; "
            f"expected={sorted(EXPECTED_CONFIGURED_EXCLUDED_IDS[arm])}, "
            f"found={sorted(configured_excluded)}"
        )

    snr_excluded = {
        canonical_exposure_id(name)
        for name in selection.get("snr_quality_excluded_files", [])
    }
    if snr_excluded != EXPECTED_SNR_EXCLUDED_IDS[arm]:
        raise RuntimeError(
            f"{metadata_path}: unexpected q_snr exclusions; "
            f"expected={sorted(EXPECTED_SNR_EXCLUDED_IDS[arm])}, "
            f"found={sorted(snr_excluded)}"
        )

    all_excluded = {
        canonical_exposure_id(name)
        for name in selection.get("excluded_files", [])
    }
    expected_all_excluded = configured_excluded | snr_excluded
    if all_excluded != expected_all_excluded:
        raise RuntimeError(
            f"{metadata_path}: combined exclusions do not match their sources; "
            f"expected={sorted(expected_all_excluded)}, "
            f"found={sorted(all_excluded)}"
        )

    edge_source = metadata.get("arm_edge_trim", {}).get("source")
    resolved_edge_source = None
    if edge_source is not None:
        resolved_edge_source = Path(edge_source)
        if not resolved_edge_source.is_absolute():
            resolved_edge_source = PROJECT_ROOT / resolved_edge_source
        resolved_edge_source = resolved_edge_source.resolve()
    if resolved_edge_source != manifest_path.resolve():
        raise RuntimeError(f"{metadata_path}: edge-trim manifest provenance mismatch")
    if metadata.get("run_sysrem") is not True:
        raise RuntimeError(f"{metadata_path}: SYSREM is not active")
    if metadata.get("stellar_velocity", {}).get("applied") is not True:
        raise RuntimeError(f"{metadata_path}: stellar-rest correction was not applied")

    print(
        f"VERIFIED {arm}/{product}: "
        f"usable={selection['n_usable_files']}, "
        f"exported={metadata['n_exposures']}, "
        f"configured_excluded={sorted(configured_excluded)}, "
        f"snr_excluded={sorted(snr_excluded)}"
    )
    return metadata


def _verify(manifest_path: Path) -> None:
    print("\n=== Verification ===", flush=True)
    for arm in ARMS:
        _verify_selection(arm, "timeseries", manifest_path)
        collapse_source = _verify_selection(arm, "collapse_source", manifest_path)
        collapsed_path = (
            CANONICAL_ROOT
            / arm
            / "collapsed"
            / "full_transit"
            / "collapse_metadata.json"
        )
        collapsed = _load_json(collapsed_path)
        if int(collapsed.get("n_source_exposures", -1)) != int(
            collapse_source.get("n_exposures", -2)
        ):
            raise RuntimeError(
                f"{collapsed_path}: collapsed source-row count does not match "
                "the regenerated collapse-source cube"
            )
        for filename in (
            "spectrum_transmission.npy",
            "uncertainty_transmission.npy",
            "wavelength_transmission.npy",
            "transmission_collapse_operator.npz",
        ):
            if not (collapsed_path.parent / filename).is_file():
                raise FileNotFoundError(
                    f"Expected collapsed product is missing: "
                    f"{collapsed_path.parent / filename}"
                )
        print(
            f"VERIFIED {arm}/collapsed/full_transit: "
            f"source={collapsed['n_source_exposures']}, "
            f"selected={collapsed['n_selected_exposures']}, "
            f"wavelengths={collapsed['n_output_wavelengths']}"
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
    _verify(manifest_path)
    print(
        "\nCOMPLETE: KELT-20b 20250601 blue/red time series, "
        "collapse sources, and collapsed spectra were regenerated and verified."
    )
    print(f"Edge-trim manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
