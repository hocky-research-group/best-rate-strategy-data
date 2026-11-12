"""Regime diagnostics of the WT-MetaD grid: where does a single-gamma fit work?

Usage:
    uv run python scripts/plot_regime_diagnostics.py

Input : analysis/growth_law_conditions.csv (per condition: boost = k_obs/k0,
        coefficient of variation of escape times, EATR error, gamma, hills)
Output: figures/fig3_regime_diagnostics.png and a printed table of the EATR
        error binned by boost and by CV of times.
Theory (inefficiency manuscript, Sec. 2.4): thermally driven regime when the
boost approaches one (fitted gamma exceeds one); fill-limited regime when the
CV of escape times falls well below one (deposition-limited escape, k0
information lost); the gently driven regime in between is where a single gamma
describes the run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
COL = {"Q": "#0072B2", "RMSD": "#009E73", "Rg": "#E69F00", "Ree": "#D55E00"}


def main() -> None:
    """Draw the diagnostics figure and print binned errors."""
    df = pd.read_csv(ROOT / "analysis/growth_law_conditions.csv")
    df = df[np.isfinite(df.eatr_err)]
    df["boost_bin"] = pd.cut(df.boost, [0, 3, 10, 30, 100, 1e9], labels=["<3", "3-10", "10-30", "30-100", ">100"])
    df["cvt_bin"] = pd.cut(df.cv_times, [0, 0.4, 0.6, 0.8, 1.0, 10], labels=["<0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1"])
    print("EATR |error| median (fraction within 1) by boost bin:")
    print(df.groupby("boost_bin", observed=True).eatr_err.agg(n="size", med_abs=lambda x: x.abs().median(), frac1=lambda x: (x.abs() <= 1).mean()).round(2).to_string())
    print("by coefficient of variation of escape times:")
    print(df.groupby("cvt_bin", observed=True).eatr_err.agg(n="size", med_abs=lambda x: x.abs().median(), frac1=lambda x: (x.abs() <= 1).mean()).round(2).to_string())
    print("gamma > 1 (thermally driven signature):", int((df.gamma > 1).sum()), "of", len(df))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for cv, g in df.groupby("cv"):
        axes[0].scatter(g.boost, g.eatr_err, s=14, c=COL[cv], alpha=0.75, lw=0, label=cv)
        axes[1].scatter(g.cv_times, g.eatr_err, s=14, c=COL[cv], alpha=0.75, lw=0)
        axes[2].scatter(g.boost, g.cv_times, s=14, c=COL[cv], alpha=0.75, lw=0)
    axes[0].set_xscale("log"); axes[0].axhspan(-1, 1, color="#7f7f7f", alpha=0.12, lw=0); axes[0].axhline(0, color="#7f7f7f", lw=0.8)
    axes[0].set_xlabel("boost k_obs / k0"); axes[0].set_ylabel("EATR error, ln k_est − ln k_ref"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhspan(-1, 1, color="#7f7f7f", alpha=0.12, lw=0); axes[1].axhline(0, color="#7f7f7f", lw=0.8)
    axes[1].set_xlabel("coefficient of variation of escape times"); axes[1].set_ylabel("EATR error")
    axes[2].set_xscale("log"); axes[2].axhline(1, color="#7f7f7f", lw=0.8, ls=":")
    axes[2].set_xlabel("boost k_obs / k0"); axes[2].set_ylabel("coefficient of variation of escape times")
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.suptitle("WT-MetaD grid (bias factors 2-20): regime diagnostics computable before any fit", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig3_regime_diagnostics.png", dpi=160)
    print("wrote figures/fig3_regime_diagnostics.png")


if __name__ == "__main__":
    main()
