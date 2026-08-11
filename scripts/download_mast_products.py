#!/usr/bin/env python3
"""Download an exact set of MAST products from a checked-in manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.mast import download_mast_selection, load_mast_selection


DEFAULT_SELECTION = PROJECT_ROOT / "reference" / "mast_hst_selection.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = download_mast_selection(load_mast_selection(args.selection), args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
