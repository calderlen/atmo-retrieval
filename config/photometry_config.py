"""Bandpass and broadband-observation defaults."""

from .paths_config import REFERENCE_BANDPASS_DIR


TESS_BANDPASS_URL = "https://heasarc.gsfc.nasa.gov/docs/tess/data/tess-response-function-v2.0.csv"
TESS_BANDPASS_PATH = REFERENCE_BANDPASS_DIR / "tess-response-function-v2.0.csv"

# Physical constants in SI units used by broadband reflection calculations.
AU_M = 1.495978707e11
