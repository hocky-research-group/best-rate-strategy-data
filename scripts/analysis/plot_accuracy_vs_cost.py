"""Signed error of the rate estimate against total simulated time, per CV.

Usage:
    uv run python scripts/plot_accuracy_vs_cost.py [master_table.csv] [out.png]

Row 1: time-dependent bias (WT-MetaD grid): iMetaD CDF vs EATR CDF, one point
       per (bf, height, pace) condition with 200 runs (50 for bf 1, 2).
Row 2: quasi-static bias: OPES flooding (5 barriers, 50 runs each, per pace)
       and VES cutoff flooding (6 cutoffs, N runs each): gamma = 1 estimator
       (OPESf) vs EATR-flooding.
The grey band is +/- 1 ln unit around the unbiased reference; the black point
is the unbiased reference itself (400 runs, its own cost).
Colors: Okabe-Ito blue = EATR family, vermilion = gamma = 1 family, validated
with the dataviz palette checker; marker shape distinguishes OPES from VES.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as _paths
from paths import dataset
import pandas as pd

ROOT = _paths.ROOT
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7f7f7f"
CVS = ["Q", "RMSD", "Rg", "Ree"]


def main(table: str, out: str) -> None:
    """Draw the two-row, four-column figure.

    Args:
        table (str): path to master_table.csv.
        out (str): output image path.
    """
    df = pd.read_csv(table)
    df = df[df.valid]
    # replace the WT-MetaD rows by the regenerated grid (correct acceleration column per CV,
    # bias factors 2-20; bias factor 1 is degenerate in PLUMED) when it is available
    own = ROOT / "analysis/subsampling_imetad.csv"
    if own.exists():
        g = pd.read_csv(own)
        g = g[(g.n == g.n_total) & (g.bf >= 2)]
        g = g.rename(columns={"mean_err": "err_ln"})
        g["family"] = "imetad"; g["valid"] = True
        df = pd.concat([df[df.family != "imetad"], g[["family", "cv", "bf", "height", "pace", "estimator", "err_ln", "cost_us", "valid"]]], ignore_index=True)
    ref = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())
    ref_cost = ref["n_runs"] * ref["mean_fpt_us"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey="row")
    for j, cv in enumerate(CVS):
        ax = axes[0, j]
        g = df[(df.family == "imetad") & (df.cv == cv)]
        for est, col, lab, z in (("iMetaD CDF", VERM, "iMetaD", 1), ("EATR CDF", BLUE, "EATR", 2)):
            d = g[g.estimator == est]
            ax.scatter(d.cost_us, d.err_ln, s=14, c=col, alpha=0.65, lw=0, label=lab, zorder=z)
        ax.set_title(f"{cv}  (WT-MetaD, {len(g) // 2} conditions, bf 2-20)", fontsize=10)
        ax = axes[1, j]
        for fam, mk, famlab in (("opesf", "o", "OPES"), ("vesf", "s", "VES")):
            g = df[(df.family == fam) & (df.cv == cv)]
            for est, col, lab in (("OPESf", VERM, "flooding, γ=1"), ("EATRf", BLUE, "EATR-flooding")):
                d = g[g.estimator == est]
                ax.scatter(d.cost_us, d.err_ln, s=34, marker=mk, facecolors=col if fam == "opesf" else "none",
                           edgecolors=col, lw=1.4, label=f"{famlab}: {lab}", zorder=2)
        ax.set_title(f"{cv}  (OPES per pace; VES per N)", fontsize=10)
    for ax in axes.ravel():
        ax.axhspan(-1, 1, color=GREY, alpha=0.12, lw=0)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.errorbar(ref_cost, 0, yerr=ref["ln_k_mle_boot_std"], fmt="o", color="k", ms=5, capsize=2, zorder=3)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylim(-9, 4.5); axes[1, 0].set_ylim(-4, 2.5)
    axes[0, 0].set_ylabel("ln k_est − ln k_ref", fontsize=9); axes[1, 0].set_ylabel("ln k_est − ln k_ref", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("total simulated time per estimate (μs)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[1, 0].legend(fontsize=7, frameon=False, loc="lower right", ncol=1)
    axes[0, 3].annotate("black: unbiased reference\n(400 runs)", xy=(ref_cost, 0), xytext=(0.55, 0.85),
                        textcoords="axes fraction", fontsize=7, color="k")
    fig.suptitle("Protein G unfolding rate: signed error vs. total simulated time (all 200 runs per condition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else str(ROOT / "analysis/master_table.csv"),
         args[1] if len(args) > 1 else str(ROOT / "figures/fig1_accuracy_vs_cost.png"))
