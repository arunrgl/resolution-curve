"""Static figure for the Stage 1 report (matplotlib; light surface, print-friendly)."""

from __future__ import annotations

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 (validated: CVD dE 25.7, normal dE 34)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_report(rep: dict, path: str, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = rep["curve_all_segments"]
    x = rep["curve_excluding_last_segment"]
    edges = np.asarray(a["edges"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    # Left: cumulative share of outcome variance resolved vs relative position
    ax1.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=MUTED, linewidth=1.2)
    ax1.text(0.62, 0.55, "uniform resolution", color=MUTED, fontsize=8, rotation=0)
    for res, color, label in ((a, SERIES[0], "all segments"), (x, SERIES[1], "excluding final segment")):
        c = np.asarray(res["curve"])
        lo, hi = np.asarray(res["curve_ci"])
        ax1.fill_between(edges, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax1.plot(edges, c, color=color, linewidth=2, label=label, marker="o", markersize=4,
                 markeredgecolor=SURFACE, markeredgewidth=1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("relative position in response  t / N", color=INK_2, fontsize=9)
    ax1.set_ylabel("cumulative share of Var(R | x) resolved", color=INK_2, fontsize=9)
    ax1.set_title(
        f"F = {a['F']:+.3f} [{a['F_ci'][0]:+.3f}, {a['F_ci'][1]:+.3f}]  ·  excl. final: {x['F']:+.3f}",
        color=INK, fontsize=9, loc="left",
    )
    leg = ax1.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    for h in leg.legend_handles:
        h.set_markersize(0)

    # Right: per-bin share with bootstrap CI, uniform reference
    B = len(a["shares"])
    centers = (edges[:-1] + edges[1:]) / 2
    sh = np.asarray(a["shares"])
    lo, hi = np.asarray(a["shares_ci"])
    ax2.bar(centers, sh, width=(1 / B) - 0.012, color=SERIES[0], edgecolor=SURFACE, linewidth=0)
    ax2.errorbar(centers, sh, yerr=[sh - lo, hi - sh], fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
    ax2.axhline(1 / B, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax2.text(0.005, 1 / B, "uniform", color=MUTED, fontsize=8, va="bottom", ha="left")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("relative position in response  t / N", color=INK_2, fontsize=9)
    ax2.set_ylabel("share of Var(R | x) resolved in bin", color=INK_2, fontsize=9)
    ax2.set_title("Per-bin share, all segments (95% cluster-bootstrap CI)", color=INK, fontsize=9, loc="left")

    sup = title or "Where the outcome is decided"
    fig.suptitle(
        f"{sup}  ·  {rep['n_prompts']} prompts, {rep['n_traj']} trajectories  ·  "
        f"conservation {a['conservation_ratio']:.2f} [{a['conservation_ci'][0]:.2f}, {a['conservation_ci'][1]:.2f}]",
        color=INK, fontsize=10, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
