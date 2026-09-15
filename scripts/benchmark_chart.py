"""
scripts/benchmark_chart.py — render docs/benchmark.png from the benchmark rows.
Called by benchmark_estimators.py --chart; or:  python3 scripts/benchmark_chart.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import FancyBboxPatch       # noqa: E402
import numpy as np                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# validated categorical slots (light surface): 1 blue, 2 orange, 3 aqua
METHODS = [
    ("orb_l1",        "ORB  (weighted L1)", "#2a78d6"),
    ("least_squares", "Least squares",      "#eb6834"),
]
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"


def _bar(ax, x, h, color, w=0.22, bottom=0.0):
    """Thin bar, 4px-rounded cap, square at the baseline."""
    if h <= 0:
        ax.plot([x - w / 2, x + w / 2], [bottom, bottom], color=color, lw=2, solid_capstyle="round")
        return
    ax.add_patch(FancyBboxPatch((x - w / 2, bottom), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                mutation_aspect=1, linewidth=0, facecolor=color, clip_on=False))


def _style(ax, title, ylabel=""):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=22)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK3, labelsize=9, length=0)
    ax.tick_params(axis="x", colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK3, fontsize=9)


def render(rows: list[dict], summary: dict, out: Path):
    plt.rcParams.update({"font.family": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
                         "figure.facecolor": SURFACE})
    n_sc = len(rows)
    fig = plt.figure(figsize=(15, 5.6), dpi=150)
    gs = fig.add_gridspec(1, 4, wspace=0.42, left=0.06, right=0.985, top=0.66, bottom=0.13)

    fig.text(0.06, 0.92, "ORB recovers the true state where least squares cannot",
             fontsize=17, fontweight="bold", color=INK)
    fig.text(0.06, 0.855, f"Same compiled system, two estimators, {n_sc} fixture scenarios with known ground truth. Bars are means over the {n_sc} scenarios.",
             fontsize=10.5, color=INK2)

    # ── legend row ──
    lx = 0.06
    for key, label, color in METHODS:
        fig.patches.append(FancyBboxPatch((lx, 0.775), 0.012, 0.022, boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, facecolor=color, linewidth=0))
        fig.text(lx + 0.017, 0.777, label, fontsize=9.5, color=INK2, va="bottom")
        lx += 0.017 + 0.0062 * len(label) + 0.02

    # ── top row: four one-axis panels ──
    panels = [
        ("Exact recovery", "exact_recovery_rate", lambda v: v * 100, "% of scenarios with every node within 1 unit", "{:.0f}%", 100),
        ("Mean max node error", "mean_max_node_error", lambda v: v, "units, lower is better", "{:.0f}", None),
        ("Corruption detection (F1)", "mean_f1", lambda v: v, "flagged rows vs planted rows", "{:.2f}", 1.0),
        ("Honest rows wrongly blamed", "mean_false_flags", lambda v: v, "false flags per run, lower is better", "{:.1f}", None),
    ]
    for pi, (title, key, tf, sub, fmt, ymax) in enumerate(panels):
        ax = fig.add_subplot(gs[0, pi])
        _style(ax, title)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=INK3, va="bottom")
        vals = [tf(summary[m]["mean_" + key.split("mean_")[-1]] if key.startswith("mean_") else summary[m][key]) for m, _, _ in METHODS]
        if key == "mean_seconds":
            ax.set_yscale("log")
            floor = 0.3
            vals_plot = [max(v, floor) for v in vals]
            ax.set_ylim(floor, max(vals_plot) * 4)
            for i, ((m, label, color), v) in enumerate(zip(METHODS, vals_plot)):
                ax.bar(i, v, width=0.22, bottom=floor, color=color, linewidth=0)
                lab = (f"{vals[i]:.1f} ms" if vals[i] < 10 else f"{vals[i]:,.0f} ms")
                ax.text(i, v * 1.25, lab, ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
        else:
            top = ymax if ymax is not None else max(vals) * 1.25
            ax.set_ylim(0, top * (1.18 if ymax is not None else 1.0))
            for i, ((m, label, color), v) in enumerate(zip(METHODS, vals)):
                _bar(ax, i, v, color)
                ax.text(i, v + top * 0.03, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(["ORB", "least squares"])

    fig.text(0.06, 0.035, "Least squares spreads one bad report across every honest row; the L1 objective concentrates it on the rows that are actually wrong. "
             "The two scenarios ORB misses are the ones its guarantee had already ruled out.",
             fontsize=9, color=INK3)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    data = json.loads((ROOT / "docs" / "benchmark.json").read_text())
    render(data["scenarios"], data["summary"], ROOT / "docs" / "benchmark.png")
    print("wrote docs/benchmark.png")
