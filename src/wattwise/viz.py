"""One chart style for the whole project (matplotlib).

Palette: a colour-blind-validated categorical order (adjacent-pair CVD ΔE ≥ 8), a single-hue
blue ramp for magnitude, a blue↔red diverging pair with a grey midpoint, and reserved
status colours. Highlight-and-grey is the default: one colour carries the story, context
stays grey.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .io import FIGURES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c9c7c0"   # de-emphasised marks

SEQ_CMAP = LinearSegmentedColormap.from_list("wattwise_blue", ["#f4f8fd"] + BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list("wattwise_div", ["#184f95", "#6da7ec", "#f0efec", "#ef8a89", "#b8302f"])


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            # list = per-glyph fallback (Helvetica Neue lacks ✓ ✗ → ≈ etc.)
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 9.5,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "patch.linewidth": 0,
        }
    )


def title(ax, headline: str, sub: str | None = None) -> None:
    """Insight headline (bold) plus an optional grey subtitle with units / scope."""
    ax.set_title(headline, loc="left", pad=22 if sub else 12)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def source(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, color=MUTED, fontsize=7.5, ha="left", va="top")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / f"{name}.png"
    fig.savefig(p)
    return p


def fmt_thousands(ax, axis: str = "y", suffix: str = "") -> None:
    f = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}{suffix}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def kpi_tiles(items: list[tuple[str, str, str]], ncols: int | None = None):
    """Render headline numbers as stat tiles: (value, label, footnote)."""
    n = len(items)
    ncols = ncols or n
    fig, axes = plt.subplots(1, ncols, figsize=(2.75 * ncols, 1.75))
    for ax, (value, label, foot) in zip(axes, items):
        ax.axis("off")
        ax.add_patch(mpl.patches.FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle="round,pad=0,rounding_size=0.06",
                                                transform=ax.transAxes, facecolor="white", edgecolor=GRID, linewidth=1))
        ax.text(0.09, 0.80, label, transform=ax.transAxes, fontsize=9, color=INK_2, va="top")
        ax.text(0.09, 0.48, value, transform=ax.transAxes, fontsize=21, fontweight="bold", color=INK, va="center")
        ax.text(0.09, 0.16, foot, transform=ax.transAxes, fontsize=7.6, color=MUTED, va="center")
    fig.subplots_adjust(wspace=0.08)
    return fig
