#!/usr/bin/env python
"""Prepare emission spectra for retrieval.

This module will handle emission spectroscopy reduction:
- Secondary eclipse analysis
- Phase curve extraction
- Dayside/nightside separation

Currently not implemented.
"""

import argparse
import sys

import config


def main():
    parser = argparse.ArgumentParser(description='Prepare emission spectra for retrieval')
    parser.add_argument('--epoch', type=str, required=True, help='Observation epoch (YYYYMMDD)')
    parser.add_argument('--planet', type=str, default=config.DEFAULT_DATA_PLANET, help='Planet name')
    parser.add_argument('--arm', type=str, choices=['red', 'blue', 'full'], default=config.DEFAULT_DATA_ARM, help='Spectrograph arm')

    args = parser.parse_args()

    raise NotImplementedError(
        "Emission spectrum reduction is not yet implemented.\n"
        "For transmission spectra, use: python -m dataio.make_transmission"
    )


if __name__ == '__main__':
    sys.exit(main())
