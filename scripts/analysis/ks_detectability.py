"""Does a failing Kolmogorov-Smirnov test flag the inaccurate naive rate estimates?

Usage (project root):
    uv run python scripts/ks_detectability.py [--fig <png path>]

Cross-tabulates the KS p value of the rescaled escape times (p < 0.05 is a failure)
against the accuracy of the estimate (|error| > 1 ln unit is inaccurate), for the
full-data fit of every WT-MetaD grid condition (iMetaD and EATR, bias factors 2-20, one
fit per condition) and for the naive per-level estimate of every OPES and VES level (one
entry per level, at its largest run count). Writes analysis/ks_detectability.csv with one
row per (family, estimator, CV): the number of accurate and inaccurate estimates, how many
of each fail the test, and the median p of each group; and
analysis/ks_detectability_by_pace.csv with the same counts per pace for the grid. With
--fig, draws p against the signed error for the naive estimates.

Inputs: the grid fits behind every grid figure of the paper (dataset fits.grid, read by
tradeoff_analysis.load_grid_full) and analysis/master_sets.csv (per-set observed rate,
acceleration and ks_p); data/reference_rate/reference_rate.json for the reference rate.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import paths as _paths
from paths import dataset
from tradeoff_analysis import load_grid_full

ROOT = _paths.ROOT
L6 = math.log(1e6)
CVS = ["Q", "RMSD", "Rg", "Ree"]
ALPHA = 0.05


def crosstab(d: pd.DataFrame, family: str, estimator: str) -> list[dict]:
    """Summarize KS failure against accuracy for one family and estimator.

    Args:
        d (pd.DataFrame): rows with columns cv, ks_p and err_ln.
        family (str): label written to the output.
        estimator (str): label written to the output.
    """
    d = d.dropna(subset=["ks_p", "err_ln"])
    rows = []
    for cv in CVS:
        g = d[d.cv == cv]
        bad = g[g.err_ln.abs() > 1]
        good = g[g.err_ln.abs() <= 1]
        rows.append(dict(family=family, estimator=estimator, cv=cv,
                         n_accurate=len(good), n_accurate_ks_fail=int((good.ks_p < ALPHA).sum()),
                         n_inaccurate=len(bad), n_inaccurate_ks_fail=int((bad.ks_p < ALPHA).sum()),
                         median_p_accurate=good.ks_p.median(), median_p_inaccurate=bad.ks_p.median()))
    return rows


def distinct_levels(ms: pd.DataFrame) -> pd.DataFrame:
    """One row per static bias level, at the largest number of runs it was fitted with.

    Args:
        ms (pd.DataFrame): per-set table with family, cv, pace, set_label (the level) and
            total (runs in the set). A VES level appears once per run count of the sweep.
    """
    keys = ["family", "cv", "pace", "set_label"]
    return ms.sort_values("total").drop_duplicates(keys, keep="last").sort_index()


def grid_fits() -> pd.DataFrame:
    """Full-data fits of the WT-MetaD grid in long form: cv, bf, height, pace, estimator, ks_p, err_ln."""
    g = load_grid_full()
    key = ["cv", "bf", "height", "pace"]
    assert not g.duplicated(key).any(), "a grid condition appears more than once"
    parts = [g[key].assign(estimator=est, ks_p=g[f"ks_{col}"], err_ln=g[f"err_{col}"])
             for est, col in (("iMetaD", "imetad"), ("EATR", "eatr"))]
    return pd.concat(parts, ignore_index=True)


def by_pace(grid: pd.DataFrame) -> pd.DataFrame:
    """KS failures among accurate and inaccurate grid fits, per estimator and pace.

    Args:
        grid (pd.DataFrame): output of grid_fits().
    """
    d = grid.dropna(subset=["ks_p", "err_ln"]).assign(bad=lambda x: x.err_ln.abs() > 1, fail=lambda x: x.ks_p < ALPHA)
    rows = []
    for (est, pace), g in d.groupby(["estimator", "pace"]):
        rows.append(dict(estimator=est, pace=pace, n_accurate=int((~g.bad).sum()), n_accurate_ks_fail=int((~g.bad & g.fail).sum()),
                         n_inaccurate=int(g.bad.sum()), n_inaccurate_ks_fail=int((g.bad & g.fail).sum())))
    return pd.DataFrame(rows)


def main() -> None:
    """Build the cross-tabulation and, optionally, the diagnostic figure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fig", type=Path, help="write a p-against-error figure to this PNG path")
    args = ap.parse_args()
    ref = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
    grid = grid_fits()
    ms = distinct_levels(pd.read_csv(ROOT / "analysis/master_sets.csv"))
    # the naive estimate of one static level rescales its observed rate by its mean acceleration
    ms = ms.assign(err_ln=ms.log_k_obs_s - L6 - ms.ln_exp_beta_v - ref)
    rows = (crosstab(grid[grid.estimator == "iMetaD"], "wtmetad", "iMetaD")
            + crosstab(grid[grid.estimator == "EATR"], "wtmetad", "EATR")
            + crosstab(ms[ms.family == "opesf"], "opesf", "naive per level")
            + crosstab(ms[ms.family == "vesf"], "vesf", "naive per level"))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "analysis/ks_detectability.csv", index=False)
    print(out.to_string(index=False))
    bp = by_pace(grid)
    bp.to_csv(ROOT / "analysis/ks_detectability_by_pace.csv", index=False)
    print("\nWT-MetaD grid by pace:"); print(bp.to_string(index=False))
    o = ms[ms.family == "opesf"].dropna(subset=["ks_p"])
    print("\nOPES levels failing the test, by BARRIER (kJ/mol):")
    print(o.groupby("set_label").apply(lambda g: pd.Series({"n": len(g), "ks_fail": int((g.ks_p < ALPHA).sum()),
                                                           "median_err": g.err_ln.median()})).round(2).to_string())
    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True, sharey=True)
        for j, cv in enumerate(CVS):
            ax = axes[0, j]
            g = grid[(grid.estimator == "iMetaD") & (grid.cv == cv)]
            ax.scatter(g.err_ln, g.ks_p.clip(1e-6), s=8, c="#D55E00", alpha=0.6, lw=0)
            ax.set_title(f"{cv}: WT-MetaD grid, iMetaD")
            ax = axes[1, j]
            for fam, mk in (("opesf", "o"), ("vesf", "s")):
                g = ms[(ms.family == fam) & (ms.cv == cv)]
                ax.scatter(g.err_ln, g.ks_p.clip(1e-6), s=14, marker=mk, facecolors="none", edgecolors="#D55E00", lw=0.8, label=fam)
            ax.set_title(f"{cv}: static levels, naive")
            ax.set_xlabel("error of ln k")
        for ax in axes.ravel():
            ax.set_yscale("log"); ax.axhline(ALPHA, color="k", lw=0.6, ls="--")
            ax.axvspan(-1, 1, color="0.8", alpha=0.4, lw=0)
        axes[0, 0].set_ylabel("KS p"); axes[1, 0].set_ylabel("KS p"); axes[1, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(args.fig, dpi=130)


if __name__ == "__main__":
    main()
