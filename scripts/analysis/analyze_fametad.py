"""Does adapting the deposition frequency move the error-cost curve?

Usage:
    uv run --group figures python scripts/analyze_fametad.py

Input : analysis/subsampling_fametad.csv (the theta scan) and the exported
        accuracy_cost.csv (the fixed-pace grid), wherever paths.py finds it.
Output: analysis/fametad_vs_pace.csv and analysis/fametad_vs_pace.pdf.

The comparison is controlled by construction. Frequency-adaptive runs start at PACE = 100
and saturate at FA_MAX_PACE = 1e5, and the grid already holds both of those paces at the
same bias factor and height with 200 runs each, so the fixed-pace ladder between them is
the baseline and theta alone decides where an adaptive run falls on it. A method that only
interpolates that ladder lands on the same curve; a method that helps lands above and to
the left of it, reaching an accuracy for less total simulated time.

Two numbers are reported per CV and estimator. The first is the cheapest condition whose
error is within one ln unit, which is the metric the main text already uses for how much
data a strategy needs. The second is the error at matched cost, interpolated along the
fixed-pace ladder at each adaptive condition's own cost, which isolates the shift from the
fact that the two families do not sample the same costs.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial"], "pdf.fonttype": 42,
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
import paths as _paths
from paths import dataset
ROOT = _paths.ROOT
CVS = ["Q", "RMSD", "Rg", "Ree"]
CVNAME = {"Q": "$Q$", "RMSD": "RMSD", "Rg": "$R_g$", "Ree": "$R_{ee}$"}
BF, HEIGHT = 10, 0.5          # the grid conditions the adaptive scan sits between
EATR, G1 = "#b02418", "#3f7fbf"


def baseline() -> pd.DataFrame:
    """Fixed-pace grid conditions at the bias factor and height of the adaptive scan."""
    d = pd.read_csv(dataset("figures.data") / "accuracy_cost.csv")
    d = d[(d.family == "wtmetad") & (d.bf == BF) & (d.height == HEIGHT)]
    return d.sort_values("pace")


def adaptive() -> pd.DataFrame:
    """Full-data rows of the theta scan, one per (CV, theta, estimator)."""
    d = pd.read_csv(ROOT / "analysis/subsampling_fametad.csv")
    d = d[d.n == d.n_total].copy()
    d["estimator"] = d.estimator.str.replace(" CDF", "", regex=False)
    d["err_ln"] = d.mean_err
    return d.sort_values("theta")


def matched(base: pd.DataFrame, cost: float) -> float:
    """Fixed-pace error interpolated in log cost, or nan outside the ladder's range.

    Args:
        base (pd.DataFrame): one CV and estimator of the fixed-pace ladder.
        cost (float): total simulated time to match, in us.
    """
    b = base.sort_values("cost_us")
    if cost < b.cost_us.min() or cost > b.cost_us.max():
        return np.nan
    return float(np.interp(np.log(cost), np.log(b.cost_us), b.err_ln))


def main() -> None:
    """Write the comparison table and figure."""
    base, fa = baseline(), adaptive()
    cvs = [c for c in CVS if c in set(fa.cv)]
    rows = []
    for cv in cvs:
        for est in ("iMetaD", "EATR"):
            b = base[(base.cv == cv) & (base.estimator == est)]
            a = fa[(fa.cv == cv) & (fa.estimator == est)]
            for r in a.itertuples():
                rows.append(dict(cv=cv, estimator=est, theta=r.theta, cost_us=r.cost_us,
                                 err_ln=r.err_ln, gamma=r.gamma,
                                 err_fixed_pace_same_cost=matched(b, r.cost_us)))
    t = pd.DataFrame(rows)
    t["shift_ln"] = t.err_ln.abs() - t.err_fixed_pace_same_cost.abs()
    t.to_csv(ROOT / "analysis/fametad_vs_pace.csv", index=False)

    fig, axes = plt.subplots(1, len(cvs), figsize=(1.75 * len(cvs) + 0.5, 2.4),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, cv in zip(axes, cvs):
        ax.axhspan(-1, 1, color="0.92", zorder=0)
        ax.axhline(0, color="0.4", lw=0.8, zorder=1)
        for est, col in (("iMetaD", G1), ("EATR", EATR)):
            b = base[(base.cv == cv) & (base.estimator == est)].sort_values("cost_us")
            ax.plot(b.cost_us, b.err_ln, "-", color=col, lw=1.0, alpha=0.55, zorder=2)
            ax.plot(b.cost_us, b.err_ln, "o", color=col, ms=3, mfc="white", mew=0.9, zorder=3)
            a = fa[(fa.cv == cv) & (fa.estimator == est)].sort_values("cost_us")
            ax.plot(a.cost_us, a.err_ln, "^-", color=col, ms=5, lw=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_title(CVNAME[cv])
        ax.set_xlabel(r"cost ($\mu$s)")
    axes[0].set_ylabel(r"$\varepsilon = \ln k_{\rm est} - \ln k_0$")
    handles = [plt.Line2D([], [], marker="o", ms=3, mfc="white", mew=0.9, color="0.35",
                          lw=1.0, label="fixed pace"),
               plt.Line2D([], [], marker="^", ms=5, color="0.35", lw=1.1, label="adaptive"),
               plt.Line2D([], [], ls="", marker="s", ms=5, color=G1, label="iMetaD"),
               plt.Line2D([], [], ls="", marker="s", ms=5, color=EATR, label="EATR")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.2,
               handletextpad=0.4)
    fig.savefig(ROOT / "analysis/fametad_vs_pace.pdf")

    print(f"{'CV':>5} {'est':>7} {'cheapest |e|<=1':>26} {'shift at matched cost':>24}")
    for cv in cvs:
        for est in ("iMetaD", "EATR"):
            b = base[(base.cv == cv) & (base.estimator == est)]
            a = fa[(fa.cv == cv) & (fa.estimator == est)]
            bo = b[b.err_ln.abs() <= 1].cost_us.min() if (b.err_ln.abs() <= 1).any() else np.nan
            ao = a[a.err_ln.abs() <= 1].cost_us.min() if (a.err_ln.abs() <= 1).any() else np.nan
            sh = t[(t.cv == cv) & (t.estimator == est)].shift_ln.dropna()
            print(f"{cv:>5} {est:>7}   fixed {bo:>8.2f}  adaptive {ao:>8.2f}   "
                  f"{sh.mean():>+8.2f} +- {sh.std(ddof=1) if len(sh) > 1 else np.nan:.2f} "
                  f"({len(sh)} matched)")
    print("wrote analysis/fametad_vs_pace.csv and .pdf")


if __name__ == "__main__":
    main()
