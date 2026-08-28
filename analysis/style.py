"""Shared chart styling.

Charts from this project are meant to survive being pulled out of the repo and
posted as a standalone image, so they are built for that: large type, a title
that states the claim rather than naming the axes, no chartjunk, and a footnote
carrying n, source and date so the number cannot be quoted without its context.
"""

from __future__ import annotations

import datetime as _dt

import matplotlib as mpl
import matplotlib.pyplot as plt

# One accent for the "spent on resent history" idea, one for "your actual
# question", one neutral. Chosen to stay distinguishable in greyscale.
INK = "#14171C"
MUTED = "#6B7280"
GRID = "#E3E6EA"
ACCENT = "#B45309"        # amber: history / overhead
ACCENT_LIGHT = "#F0C88A"
CONTRAST = "#1D4ED8"      # blue: new user text
BG = "#FFFFFF"

FONT_STACK = ["Helvetica Neue", "Avenir Next", "Helvetica", "Arial", "DejaVu Sans"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 15,
        "axes.titlesize": 21,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def titles(ax, claim: str, subtitle: str | None = None) -> None:
    """Title states the finding; subtitle carries the measurement detail.

    Both are drawn as text above the axes rather than via set_title, so that a
    wrapped two-line subtitle pushes the title up instead of colliding with it.
    """
    import textwrap

    fig = ax.figure
    line_h = 0.052  # axes-fraction height of one text line at these font sizes

    y = 1.035
    if subtitle:
        wrapped = textwrap.fill(subtitle, 84)
        ax.text(0.0, y, wrapped, transform=ax.transAxes,
                fontsize=14.5, color=MUTED, va="bottom", linespacing=1.35)
        y += line_h * (wrapped.count("\n") + 1) + 0.022

    claim_wrapped = textwrap.fill(claim, 58)
    ax.text(0.0, y, claim_wrapped, transform=ax.transAxes,
            fontsize=20.5, fontweight="bold", color=INK, va="bottom",
            linespacing=1.18)

    total_lines = claim_wrapped.count("\n") + 1 + (wrapped.count("\n") + 1 if subtitle else 0)
    fig.subplots_adjust(top=0.90 - 0.045 * total_lines)


def footnote(fig, n: int, extra: str = "") -> None:
    date = _dt.date.today().isoformat()
    bits = [
        f"n = {n:,} conversations",
        "median, band = interquartile range",
        "allenai/WildChat-1M, English, non-coding, ≥10 turns",
        "tokenizer: tiktoken o200k_base",
    ]
    if extra:
        bits.append(extra)
    bits.append(date)
    fig.text(0.0, -0.045, "  ·  ".join(bits), fontsize=11.5, color=MUTED,
             ha="left", va="top")


def save(fig, path_stem) -> None:
    for ext in ("png", "svg"):
        fig.savefig(f"{path_stem}.{ext}")
    plt.close(fig)
    print(f"[chart] {path_stem}.png / .svg")
