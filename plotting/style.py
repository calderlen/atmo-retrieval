"""Shared Matplotlib typography for retrieval figures."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import matplotlib


COMPUTER_MODERN_BRIGHT_FONTS = (
    "CMU Bright",
)

COMPUTER_MODERN_BRIGHT_RCPARAMS = {
    "font.family": COMPUTER_MODERN_BRIGHT_FONTS,
    "mathtext.fontset": "custom",
    "mathtext.rm": "CMU Bright",
    "mathtext.it": "CMU Bright:style=oblique",
    "mathtext.bf": "CMU Bright:weight=bold",
    "mathtext.bfit": "CMU Bright:style=oblique:weight=bold",
    "mathtext.sf": "CMU Bright",
    "mathtext.tt": "CMU Bright",
    "mathtext.cal": "CMU Bright",
    # CMU Bright's TrueType face omits a few mathematical operator glyphs
    # (for example \odot and \star).  Keep CM only as a symbol fallback so
    # labels remain complete while all available alphanumerics stay Bright.
    "mathtext.fallback": "cm",
    # Saved figures are vector PDFs by default. Explicit paths are normalized
    # by save_figure_pdf() so a legacy .png suffix cannot override this.
    "savefig.format": "pdf",
    # Embed the installed CMU Bright TrueType outlines instead of Matplotlib's
    # default Type 3 glyphs.  Type 42/CID text remains vector, searchable, and
    # suitable for manuscript-production workflows.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def configure_matplotlib() -> None:
    """Use Computer Modern Bright and PNG for Jupyter's inline display.

    No generic family is included: if Computer Modern Bright is unavailable,
    Matplotlib emits a visible font warning instead of silently producing a
    different typeface.  This does not change the PDF-only saved-file policy.
    """
    matplotlib.rcParams.update(COMPUTER_MODERN_BRIGHT_RCPARAMS)
    try:
        from matplotlib_inline.backend_inline import InlineBackend
    except ImportError:
        pass
    else:
        InlineBackend.instance().figure_formats = {"png"}


def pdf_figure_path(path: str | Path) -> Path:
    """Return *path* with a mandatory ``.pdf`` suffix."""
    candidate = Path(path)
    if candidate.suffix.lower() == ".pdf":
        return candidate
    return candidate.with_suffix(".pdf")


def save_figure_pdf(figure: Any, path: str | Path, **kwargs: Any) -> Path:
    """Save a Matplotlib figure as PDF regardless of the requested suffix."""
    output_path = pdf_figure_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs["format"] = "pdf"
    figure.savefig(output_path, **kwargs)
    return output_path


def display_pdf_png(path: str | Path, *, dpi: int = 144) -> None:
    """Rasterize the first page of a saved PDF for PNG-only inline display."""
    from IPython.display import Image, display

    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF figure, found {pdf_path}.")
    rendered = subprocess.run(
        [
            "pdftoppm",
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(dpi),
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
    )
    if not rendered.stdout.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"pdftoppm did not render a PNG for {pdf_path}.")
    display(Image(data=rendered.stdout, format="png"))
