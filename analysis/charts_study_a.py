"""Study A charts.

    fig1  cumulative input tokens vs turn index (median, IQR band)
    fig2  share of cumulative spend that is resent history  <- flagship
    fig3  cost of turn N relative to turn 1

Usage:
    python analysis/charts_study_a.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

import style


def fmt_thousands(x, _pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def fig1_cumulative(agg: pd.DataFrame, n: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.fill_between(agg.turn, agg.cum_input_tokens_p25, agg.cum_input_tokens_p75,
                    color=style.ACCENT, alpha=0.16, linewidth=0)
    ax.plot(agg.turn, agg.cum_input_tokens_p50, color=style.ACCENT)

    style.titles(
        ax,
        "A chat thread's cost grows with the square of its length",
        "Cumulative input tokens sent, by turn number.",
    )
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Cumulative input tokens")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_thousands))
    ax.set_xlim(1, agg.turn.max())
    ax.set_ylim(bottom=0)

    # Anchored to the axes, not to the curve, so it never collides with the band.
    last = agg.iloc[-1]
    ax.text(
        0.03, 0.95,
        f"By turn {int(last.turn)}, the median thread has sent\n"
        f"{last.cum_input_tokens_p50 / 1000:.0f}k input tokens in total",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=14.5, color=style.INK, linespacing=1.4,
    )

    style.footnote(fig, n)
    style.save(fig, out / "study_a_fig1_cumulative_tokens")


def fig2_history_share(agg: pd.DataFrame, n: int, out: Path) -> None:
    """The flagship chart: how much of the spend is re-transmitted thread."""
    fig, ax = plt.subplots(figsize=(11, 6.6))

    pct = agg.history_share_cum_p50 * 100
    ax.fill_between(agg.turn, agg.history_share_cum_p25 * 100,
                    agg.history_share_cum_p75 * 100,
                    color=style.ACCENT, alpha=0.22, linewidth=0)
    ax.plot(agg.turn, pct, color=style.ACCENT, label="Resent conversation history")

    # The stricter metric, shown so the headline cannot be accused of picking
    # the flattering definition.
    ax.plot(agg.turn, agg.repeat_share_cum_p50 * 100, color=style.MUTED,
            linewidth=2.2, linestyle=(0, (5, 3)),
            label="Stricter definition (see METHODOLOGY)")

    style.titles(
        ax,
        "Most of what a long chat costs is re-sending the thread",
        "Share of all input tokens spent so far that were not your new question.",
    )
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Share of cumulative input spend")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    ax.set_xlim(1, agg.turn.max())
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")

    for turn in (10, 20, 30):
        row = agg[agg.turn == turn]
        if row.empty:
            continue
        y = float(row.iloc[0].history_share_cum_p50) * 100
        ax.plot([turn], [y], "o", color=style.ACCENT, markersize=9,
                markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(f"{y:.0f}%", xy=(turn, y), xytext=(0, 12),
                    textcoords="offset points", ha="center",
                    fontsize=14.5, fontweight="bold", color=style.ACCENT)

    style.footnote(fig, n)
    style.save(fig, out / "study_a_fig2_history_share")


def fig3_relative_cost(agg: pd.DataFrame, n: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.fill_between(agg.turn, agg.relative_cost_vs_turn1_p25,
                    agg.relative_cost_vs_turn1_p75,
                    color=style.CONTRAST, alpha=0.14, linewidth=0)
    ax.plot(agg.turn, agg.relative_cost_vs_turn1_p50, color=style.CONTRAST)
    ax.axhline(1, color=style.MUTED, linewidth=1.4, linestyle=(0, (4, 4)))
    ax.annotate("cost of your first message", xy=(agg.turn.max(), 1),
                xytext=(-6, 8), textcoords="offset points", ha="right",
                fontsize=13.5, color=style.MUTED)

    style.titles(
        ax,
        "The same question costs more the later you ask it",
        "Input tokens for a single turn, as a multiple of the first turn's.",
    )
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Cost relative to turn 1")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}×"))
    ax.set_xlim(1, agg.turn.max())
    ax.set_ylim(bottom=0)

    last = agg.iloc[-1]
    ax.text(
        0.03, 0.95,
        f"A new question at turn {int(last.turn)} costs\n"
        f"{last.relative_cost_vs_turn1_p50:.0f}× what it costs in a fresh chat",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=14.5, color=style.CONTRAST, fontweight="bold", linespacing=1.4,
    )

    style.footnote(fig, n)
    style.save(fig, out / "study_a_fig3_relative_cost")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()

    tables, figures = Path(args.tables), Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(tables / "study_a_by_turn_index.csv")
    head = json.loads((tables / "study_a_headline.json").read_text())
    n = head["n_conversations"]

    style.apply()
    fig1_cumulative(agg, n, figures)
    fig2_history_share(agg, n, figures)
    fig3_relative_cost(agg, n, figures)


if __name__ == "__main__":
    main()
