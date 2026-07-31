"""Shared Matplotlib typography for retrieval figures."""

from __future__ import annotations

import matplotlib


COMPUTER_MODERN_BRIGHT_FONTS = (
    "Computer Modern Bright",
    "CMU Bright",
    "CM Bright",
    "sans-serif",
)


def configure_matplotlib() -> None:
    """Use Computer Modern Bright for plot text and Computer Modern for math.

    The alternate family names cover the common font distributions.  The final
    generic family keeps plots renderable in environments where the font has
    not yet been installed.
    """
    matplotlib.rcParams.update(
        {
            "font.family": COMPUTER_MODERN_BRIGHT_FONTS,
            "mathtext.fontset": "cm",
        }
    )
