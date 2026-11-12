"""Three follow-up analyses on the completed benchmark.

Usage:
    uv run python scripts/tradeoff_analysis.py

1. Where EATR is worse than iMetaD (full-data fits on the WT-MetaD grid): is the
   difference statistical (the reference lies inside EATR's bootstrap interval, or the
   two estimates differ by less than the EATR subset spread at n = 80) or systematic?
   Output: analysis/eatr_worse_cases.csv, figures/fig8_eatr_vs_imetad_cases.png
2. A priori parameter choice for the poor CVs: which grid conditions the design rule
   recommends from sigma_A, barrier, k0 and the OPES pilot gamma, and the cost at which
   90% of subsets of those conditions land within 1 ln unit, against the rest; and the
   position of every flooding set on the boost axis.
   Output: analysis/apriori_choice.csv
3. Error-cost Pareto front per CV over strategies (protocol, estimator, subset size),
   error = RMSE pooled over the a priori recommended conditions (grid) or all paces
   (flooding); single full-set fits excluded.
   Output: analysis/pareto_front.csv, figures/fig9_pareto.png
"""
import gzip
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
L6 = math.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
PAT = re.compile(r"^(\w+)_bf([\d.]+)_height([\d.]+)_pace(\d+)\.json$")
CVS = ["Q", "RMSD", "Rg", "Ree"]
COL = {"Q": "#0072B2", "RMSD": "#009E73", "Rg": "#E69F00", "Ree": "#D55E00"}
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#7f7f7f"
MAX_POOL_FRACTION = 0.5      # largest subset, as a share of the runs available, that counts as a strategy


def pooled_rmse(d: pd.DataFrame) -> float:
    """Root-mean-square error over every subset fit of several conditions.

    Args:
        d (pd.DataFrame): one row per condition with n_fits (number of subset fits) and
            rmse (root-mean-square error over those fits).
    Weighting each condition's mean squared error by its fit count makes this the RMSE of
    all the fits pooled, which is the quantity the manuscript names.
    """
    return math.sqrt(float((d.n_fits * d.rmse ** 2).sum() / d.n_fits.sum()))


def iter_grid_records():
    """Yield ((cv, bf, height, pace), fit) for every WT-MetaD grid condition of bias factor 2 and above.

    fit is the content of the condition's eatr-analysis output JSON, as archived in fits.grid.
    """
    for pk in sorted((dataset("fits.grid")).rglob("*.jsonl.gz")):
        with gzip.open(pk, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                m = PAT.match(Path(rec["file"]).name)
                if not m:
                    continue
                key = (m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4)))
                if key[1] < 2:
                    continue
                yield key, rec["data"]


def load_grid_full() -> pd.DataFrame:
    """Full-fit rows of the WT-MetaD grid with bootstrap intervals, from the archive."""
    rows = []
    for (cv, bf, h, pace), d in iter_grid_records():
        eci = d.get("EATR CDF ln k CI") or [np.nan, np.nan]
        ici = d.get("iMetaD CDF ln k CI") or [np.nan, np.nan]
        sub = d.get("subsample", {}).get("summary", {})
        s80 = sub.get("80", sub.get("40", {}))
        rows.append(dict(cv=cv, bf=bf, height=h, pace=pace,
                         err_eatr=d["EATR CDF ln k"] - L6 - REF, err_imetad=d["iMetaD CDF ln k"] - L6 - REF,
                         eatr_lo=eci[0] - L6 - REF, eatr_hi=eci[1] - L6 - REF,
                         imetad_lo=ici[0] - L6 - REF, imetad_hi=ici[1] - L6 - REF,
                         gamma=d.get("EATR CDF gamma"), ks_eatr=d.get("EATR CDF p value"), ks_imetad=d.get("iMetaD CDF p value"),
                         sd_eatr_sub=s80.get("EATR CDF ln k std", np.nan), sd_imetad_sub=s80.get("iMetaD CDF ln k std", np.nan),
                         cost_us=sum(d["EATR CDF plot"]["time"]) * 1e6))
    return pd.DataFrame(rows)


def worse_class(worse: pd.DataFrame) -> np.ndarray:
    """Class of each case in which EATR is less accurate than iMetaD.

    Args:
        worse (pd.DataFrame): cases with abs_e and abs_i (|error| of EATR and of iMetaD),
            diff (their difference) and the flags both_within_1, ref_in_eatr_ci and
            diff_lt_spread.
    Returns one label per row, the first that applies: marginal (both within one ln unit and
    closer than 0.3), statistical (the reference inside EATR's bootstrap interval, or the
    difference below the subset spread), systematic (EATR outside one ln unit and iMetaD
    inside), other.
    """
    return np.select(
        [worse.both_within_1 & (worse["diff"] < 0.3), worse.ref_in_eatr_ci | worse.diff_lt_spread,
         (worse.abs_e > 1) & (worse.abs_i <= 1)],
        ["marginal (both within 1, diff < 0.3)", "statistical (reference inside EATR interval or diff < subset spread)", "systematic (EATR outside 1, iMetaD inside)"],
        default="other")


def part1(g: pd.DataFrame) -> None:
    """EATR-worse-than-iMetaD cases."""
    g = g.copy()
    g["abs_e"], g["abs_i"] = g.err_eatr.abs(), g.err_imetad.abs()
    worse = g[g.abs_e > g.abs_i].copy()
    worse["diff"] = worse.abs_e - worse.abs_i
    worse["ref_in_eatr_ci"] = (worse.eatr_lo <= 0) & (0 <= worse.eatr_hi)
    worse["diff_lt_spread"] = worse["diff"] < worse.sd_eatr_sub
    worse["both_within_1"] = (worse.abs_e <= 1) & (worse.abs_i <= 1)
    worse["class"] = worse_class(worse)
    worse.to_csv(ROOT / "analysis/eatr_worse_cases.csv", index=False)
    print(f"1. EATR worse than iMetaD in {len(worse)} of {len(g)} conditions ({len(worse)/len(g):.0%})")
    print(worse.groupby("class").size().to_string())
    print("   by CV and pace (count of worse cases / conditions):")
    t = worse.groupby(["cv", "pace"]).size().unstack(fill_value=0)
    n = g.groupby(["cv", "pace"]).size().unstack(fill_value=0)
    print((t.astype(str) + "/" + n.astype(str)).to_string())
    sysm = worse[worse["class"].str.startswith("systematic")]
    print(f"   systematic cases: {len(sysm)}; paces: {sorted(sysm.pace.unique())}; median gamma {sysm.gamma.median():.2f}; median EATR err {sysm.err_eatr.median():+.2f}; KS p<0.05 in {(sysm.ks_eatr < 0.05).mean():.0%}")
    print("   signed EATR error of systematic cases by pace:")
    print(sysm.groupby("pace").err_eatr.agg(["size", "median"]).round(2).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for cv in CVS:
        d = g[g.cv == cv]
        ax.scatter(d.abs_i, d.abs_e, s=12, c=COL[cv], alpha=0.7, lw=0, label=cv)
    lim = [0.01, 10]
    ax.plot(lim, lim, color=GREY, lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axhline(1, color=GREY, lw=0.6, ls=":"); ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("|error| iMetaD"); ax.set_ylabel("|error| EATR"); ax.set_title("full fits, 200 runs; above the diagonal EATR is worse", fontsize=9); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    order = ["marginal (both within 1, diff < 0.3)", "statistical (reference inside EATR interval or diff < subset spread)", "systematic (EATR outside 1, iMetaD inside)", "other"]
    ct = worse.groupby(["pace", "class"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    bottom = np.zeros(len(ct))
    for c, col in zip(order, ["#BBBBBB", "#56B4E9", VERM, "#000000"]):
        ax.bar([str(int(p)) for p in ct.index], ct[c], bottom=bottom, color=col, label=c.split(" (")[0]); bottom += ct[c].to_numpy()
    ax.set_xlabel("pace (steps)"); ax.set_ylabel("conditions where EATR is worse"); ax.legend(fontsize=7, frameon=False); ax.set_title("classification of the EATR-worse cases", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig8_eatr_vs_imetad_cases.png", dpi=160)


def part2(g: pd.DataFrame) -> None:
    """A priori choice for the poor CVs."""
    val = pd.read_csv(ROOT / "analysis/bias_design_validation.csv")
    sub = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    sub = sub[(sub.n < sub.n_total) & (sub.bf >= 5) & (sub.estimator == "EATR CDF")]
    rows = []
    print("\n2. A priori choice (design rule with sigma_A, barrier, k0, OPES pilot gamma):")
    for cv in CVS:
        v = val[val.cv == cv]
        rec = v[v.recommended][["bf", "height", "pace"]]
        s = sub[sub.cv == cv].merge(rec.assign(recommended=True), on=["bf", "height", "pace"], how="left").fillna({"recommended": False})
        for flag, lab in ((True, "recommended"), (False, "not recommended")):
            a = s[s.recommended == flag].groupby("n").agg(cost=("cost_us", "median"), p1=("frac_within_1", "mean"), sd=("sd", "median"), rmse=("rmse", "median")).reset_index().sort_values("n")
            ok = [bool((a.p1.iloc[i:] >= 0.9).all()) for i in range(len(a))]
            hit = a[ok].sort_values("cost")
            rows.append(dict(cv=cv, group=lab, n_conditions=int(s[s.recommended == flag].groupby(["bf", "height", "pace"]).ngroups),
                             min_cost_us_90pct=hit.cost.iloc[0] if len(hit) else np.nan, n_at_min=int(hit.n.iloc[0]) if len(hit) else -1,
                             rmse_n20=float(a[a.n == 20].rmse.iloc[0]) if (a.n == 20).any() else np.nan,
                             sd_n20=float(a[a.n == 20].sd.iloc[0]) if (a.n == 20).any() else np.nan))
        cheapest = v[v.recommended].sort_values("pred_t_ns").head(3)[["bf", "height", "pace", "pred_boost", "pred_t_ns", "eatr_err"]]
        print(f"   {cv}: gamma_pilot {v.gamma_pilot.iloc[0]:.2f}; cheapest recommended protocols (pred. escape ns, actual EATR err):")
        print("   " + cheapest.round(2).to_string(index=False).replace("\n", "\n   "))
    t = pd.DataFrame(rows); t.to_csv(ROOT / "analysis/apriori_choice.csv", index=False)
    print(t.round(2).to_string(index=False))
    # flooding sets on the boost axis
    ms = pd.read_csv(ROOT / "analysis/master_sets.csv")
    o = ms[(ms.family == "opesf")].copy(); o["boost"] = np.exp(o.log_k_obs_s - L6 - REF)
    print("   OPES sets: boost k_obs/k0 by barrier (median over paces):")
    print(o.pivot_table(index="cv", columns="set_label", values="boost", aggfunc="median").round(1).to_string())
    o["lnacc_per_kT"] = o.ln_exp_beta_v / (o.set_label / 2.594)
    print("   ln<e^{beta V}> per kT of BARRIER (median):", o.groupby("cv").lnacc_per_kT.median().round(2).to_dict())


def pareto_front(d: pd.DataFrame) -> pd.DataFrame:
    """The strategies no other strategy beats on both cost and RMSE, cheapest first.

    Args:
        d (pd.DataFrame): one row per strategy, with columns cost_us and rmse.
    """
    # two estimators fitted to the same runs cost exactly the same; ordering ties by RMSE puts
    # the better one first, so the other is dropped whatever order the rows arrive in
    d = d.sort_values(["cost_us", "rmse"], kind="mergesort")
    best = np.inf; keep = []
    for i, r in d.iterrows():
        if r.rmse < best:
            best = r.rmse; keep.append(i)
    return d.loc[keep]


def part3() -> None:
    """Pareto front of RMSE against cost per CV, at the level of strategies.

    A strategy is (protocol, estimator, runs per condition or per set). Its error is
    the RMSE of ln k pooled over independent subsets of every condition in its class:
    for WT-MetaD the conditions the design rule recommends a priori (bias factors
    5-20; the pool the rule would send a practitioner to), for flooding all paces.
    Single full-set fits are excluded: one fit per condition has no spread and picking
    the luckiest of 640 would be selection bias. Subset sizes above MAX_POOL_FRACTION of
    the runs available are excluded as well: draws without replacement from most of a pool
    share most of their runs, so their spread understates the error of that many
    independent runs (variance factor 1 - n/N). Per-condition subset points are drawn
    faintly behind the strategy points.
    Costs are production runs only: the median simulated time of a subset.
    """
    val = pd.read_csv(ROOT / "analysis/bias_design_validation.csv")
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    im = im[(im.n <= MAX_POOL_FRACTION * im.n_total) & (im.bf >= 5) & (im.n_fits >= 5)].copy()
    rec = val[val.recommended][["cv", "bf", "height", "pace"]].assign(recommended=True)
    im = im.merge(rec, on=["cv", "bf", "height", "pace"], how="left").fillna({"recommended": False})
    fl = pd.read_csv(ROOT / "analysis/subsampling_flooding.csv")
    pool = fl.groupby("family").n.transform("max")          # runs per set: the full-set row of each family
    fl = fl[fl.estimator.isin(["EATRf", "OPESf"]) & (fl.n_fits >= 3) & (fl.n <= MAX_POOL_FRACTION * pool)].copy()
    strat = []
    for cv in CVS:
        g = im[(im.cv == cv) & im.recommended]
        for (est, n), d in g.groupby(["estimator", "n"]):
            strat.append(dict(cv=cv, protocol="WT-MetaD (recommended set)", estimator=est.split()[0], n=int(n),
                              cost_us=float(d.cost_us.median()), rmse=pooled_rmse(d), p1=float(d.frac_within_1.mean()),
                              label=f"WT-MetaD {est.split()[0]} n={int(n)}"))
        for fam in ("opes", "ves"):
            g = fl[(fl.cv == cv) & (fl.family == fam)]
            for (est, n), d in g.groupby(["estimator", "n"]):
                strat.append(dict(cv=cv, protocol=fam.upper(), estimator=est, n=int(n), cost_us=float(d.cost_us.median()),
                                  rmse=pooled_rmse(d), p1=float(d.frac_within_1.mean()), label=f"{fam.upper()} {est} n={int(n)}/set"))
    st = pd.DataFrame(strat)
    fronts = []
    print("\n3. Pareto front over strategies (pooled RMSE vs median cost):")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    style = {("WT-MetaD (recommended set)", "iMetaD"): (VERM, "."), ("WT-MetaD (recommended set)", "EATR"): (BLUE, "."),
             ("OPES", "OPESf"): (VERM, "s"), ("OPES", "EATRf"): (BLUE, "o"), ("VES", "OPESf"): (VERM, "D"), ("VES", "EATRf"): (BLUE, "^")}
    for j, cv in enumerate(CVS):
        ax = axes[j]
        bg = im[(im.cv == cv)]
        ax.scatter(bg.cost_us, bg.rmse, s=4, c=[BLUE if e.startswith("EATR") else VERM for e in bg.estimator], alpha=0.12, lw=0)
        d = st[st.cv == cv].sort_values(["cost_us", "rmse"], kind="mergesort")
        f = pareto_front(d); fronts.append(f)
        for (prot, est), (col, mk) in style.items():
            dd = d[(d.protocol == prot) & (d.estimator == est)]
            if len(dd):
                ax.scatter(dd.cost_us, dd.rmse, s=40 if mk != "." else 60, marker=mk, c=col, edgecolors="k", lw=0.4, label=f"{prot.split()[0]} {est}", zorder=3)
        ax.plot(f.cost_us, f.rmse, "k-", lw=1.2, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(cv); ax.set_xlabel("cost (μs)"); ax.grid(True, alpha=0.25, lw=0.5)
        ax.axhline(1, color=GREY, lw=0.6, ls=":"); ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
        print(f"   {cv}:")
        for _, r in f.iterrows():
            print(f"      {r.cost_us:7.2f} us  RMSE {r.rmse:4.2f}  P1 {r.p1:.2f}  {r.label}")
    axes[0].set_ylabel("RMSE of ln k over subsets (pooled)"); axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Error-cost trade-off by strategy: WT-MetaD on the a priori recommended conditions, flooding over all paces; faint points: every grid condition", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig9_pareto.png", dpi=160)
    pd.concat(fronts).to_csv(ROOT / "analysis/pareto_front.csv", index=False)
    st.to_csv(ROOT / "analysis/pareto_strategies.csv", index=False)


if __name__ == "__main__":
    grid = load_grid_full()
    part1(grid)
    part2(grid)
    part3()
    print("wrote figures/fig8_eatr_vs_imetad_cases.png, figures/fig9_pareto.png")
