#!/usr/bin/env python
"""Collapse high-resolution emission time series to a 1D spectrum.

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
    parser = argparse.ArgumentParser(description='Collapse emission time series to a 1D spectrum')
    parser.add_argument('--epoch', type=str, required=True, help='Observation epoch (YYYYMMDD)')
    parser.add_argument('--planet', type=str, default=config.DEFAULT_DATA_PLANET, help='Planet name')
    parser.add_argument('--arm', type=str, choices=['red', 'blue', 'full'], default=config.DEFAULT_DATA_ARM, help='Spectrograph arm')

    args = parser.parse_args()

    raise NotImplementedError(
        "Emission spectrum reduction is not yet implemented.\n"
        "For retrieval-ready emission time-series cubes, use: "
        "python -m dataio.prepare_emission_retrieval_timeseries\n"
        "For transmission spectra, use: python -m dataio.collapse_transmission_timeseries_to_1d"
    )


if __name__ == '__main__':
    sys.exit(main())
