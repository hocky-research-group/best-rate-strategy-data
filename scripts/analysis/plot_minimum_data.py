"""Minimum data figure: hit rate and spread against cost for each method.

Usage:
    uv run python scripts/plot_minimum_data.py

Inputs : analysis/subsampling_imetad.csv, analysis/subsampling_flooding.csv
Output : figures/fig2_minimum_data.png
Row 1: fraction of independent subsets whose estimate lies within 1 ln unit of
       the reference (P1) against the cost of the subset, per CV. WT-MetaD
       grid curves are the mean of P1 over all conditions at each subset size
       (thin lines: individual conditions with bias factor 10, height 0.5).
       OPES and VES flooding curves are means over paces (OPES) or the single
       VES sweep.
Row 2: spread (sd over subsets) of ln k, same layout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7f7f7f"
CVS = ["Q", "RMSD", "Rg", "Ree"]


def main() -> None:
    """Draw the figure."""
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    fl = pd.read_csv(ROOT / "analysis/subsampling_flooding.csv")
    im = im[(im.n < im.n_total) & (im.bf >= 2)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey="row")
    for j, cv in enumerate(CVS):
        for est, col, lab in (("iMetaD CDF", VERM, "WT-MetaD: iMetaD"), ("EATR CDF", BLUE, "WT-MetaD: EATR")):
            g = im[(im.cv == cv) & (im.estimator == est)]
            if g.empty:
                continue
            a = g.groupby("n").agg(cost=("cost_us", "median"), p1=("frac_within_1", "mean"), sd=("sd", "median")).reset_index()
            axes[0, j].plot(a.cost, a.p1, "-o", color=col, ms=4, lw=1.8, label=lab)
            axes[1, j].plot(a.cost, a.sd, "-o", color=col, ms=4, lw=1.8, label=lab)
            for _, c in g[(g.bf == 10) & (g.height == 0.5)].groupby("pace"):
                c = c.sort_values("n")
                axes[0, j].plot(c.cost_us, c.frac_within_1, "-", color=col, lw=0.5, alpha=0.35)
        for fam, mk, famlab in (("opes", "o", "OPES"), ("ves", "s", "VES")):
            for est, col, lab in (("OPESf", VERM, "flooding γ=1"), ("EATRf", BLUE, "EATR-flooding")):
                g = fl[(fl.family == fam) & (fl.cv == cv) & (fl.estimator == est)]
                if g.empty:
                    continue
                a = g.groupby("n").agg(cost=("cost_us", "median"), p1=("frac_within_1", "mean"), sd=("sd", "median")).reset_index()
                fill = col if fam == "opes" else "none"
                axes[0, j].plot(a.cost, a.p1, "--", marker=mk, color=col, mfc=fill, ms=6, lw=1.2, label=f"{famlab}: {lab}")
                axes[1, j].plot(a.cost, a.sd, "--", marker=mk, color=col, mfc=fill, ms=6, lw=1.2, label=f"{famlab}: {lab}")
        axes[0, j].set_title(cv, fontsize=11)
    for ax in axes.ravel():
        ax.set_xscale("log"); ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)
    axes[0, 0].set_ylim(0, 1.05); axes[0, 0].axhline(0.9, color=GREY, lw=0.8, ls=":")
    axes[1, 0].set_yscale("log")
    axes[0, 0].set_ylabel("fraction of subsets within 1 ln unit", fontsize=9)
    axes[1, 0].set_ylabel("spread of ln k over subsets (sd)", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("total simulated time per estimate (μs)", fontsize=9)
    axes[0, 3].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("How much data for the protein G unfolding rate: subsampling sweeps (WT-MetaD grid: bias factors 2-20; flooding sweeps complete)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig2_minimum_data.png", dpi=160)
    print("wrote figures/fig2_minimum_data.png")


if __name__ == "__main__":
    main()
