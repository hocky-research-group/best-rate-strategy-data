"""Rate against barrier across the temperature ladder, and what it takes to get gamma right.

Usage:
    uv run --group figures python scripts/analyze_temperature_rates.py

Input : data/subsampling/opes_temperature/results/opes_T*.json  (EATR-flooding, five fills
        per temperature), data/subsampling/wtmetad_temperature/results/wtmetad_T*.json
        (one well-tempered condition per temperature), and
        analysis/fes_temperature_scan_profiles.csv for the free energy the barriers are read
        from.
Output: analysis/temperature_rates.csv, analysis/temperature_convergence.csv (the fit with
        three, four and five bias sets per temperature) and analysis/temperature_rates.pdf.

The experiment the benchmark temperature cannot perform. Lowering the temperature raises the
barrier from 5.95 to 16.09 kT, so the rate falls by ten ln units and the acceleration needed
to see an escape rises from about 20 to about 1e5. Whether an estimator still recovers the
rate over that range is the question, and the barrier supplies the answer to compare against:
d ln k = -d(barrier), a line of slope one with no free parameter.

The figure shows, in order, where the barriers come from, whether the estimates follow them,
and what it takes to get gamma right:

(a) The free energy along Q at each temperature, with the barrier each one implies marked.
    Both scan seeds are drawn, so the spread that enters the barrier is visible rather than
    averaged away.

(b) Whether the estimate tracks the barrier. The unbiased rate is known only at 312 K, so
    the test is on the change relative to that anchor, which is what the ladder was built
    for.

(c) How many fills it takes to determine gamma. The flooding fit reports its own result
    using three, four and five bias sets, and the number needed grows with the acceleration:
    a gamma error moves the rate by (1 - gamma) ln(boost), which is small where the boost is
    small and several ln units where it is large.

Panel (b) also carries the single well-tempered condition per temperature at two deposition
paces, whose achieved residual barriers fall either side of the two-kT rule. They are the
direct measurement of what that rule buys: the same bias factors and heights, one variable
between them.

A fourth result is computed and written to the CSV but not drawn, since it shows no trend:
the rate each rung implies on its own, against the fill that rung achieved. Within a
temperature those scatter by about one ln unit with no dependence on the fill, which spans
+3.3 to -0.6 kT.

The estimator that assumes gamma = 1 is labelled by that assumption rather than as naive:
the acceleration it uses is measured from each run, and what distinguishes it is only that
it takes the measured acceleration at face value.
"""
import json
import re
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
KB = 0.0083144626
LN_US = np.log(1e6)
BARRIER_KT = {312: 5.95, 300: 8.26, 285: 11.00, 270: 16.09}
REF_LNK, REF_SD = 0.431, 0.050        # 312 K unbiased reference, ln(k / us^-1)
TEMPS = (312, 300, 285, 270)
COLORS = {270: "#1f3b73", 285: "#3f7fbf", 300: "#e08a1e", 312: "#b02418"}


def ladder(T: int) -> dict:
    """Flooding fit for one temperature.

    Args:
        T (int): temperature in K.
    """
    return json.load(open(dataset("fits.opes_temperature") / f"results/opes_T{T}.json"))


def fit_path(T: int, trees: tuple) -> Path | None:
    """The well-tempered fit of one temperature, from the first tree that holds it.

    Args:
        T (int): temperature in K.
        trees (tuple): subsampling trees in order of preference; a condition refitted with
            more runs sits in its own tree ahead of the original.
    """
    for tree in trees:
        p = dataset(f"fits.{tree}") / f"wtmetad_T{T}.json"
        if p.exists():
            return p
    return None


def wtmetad(T: int, trees: tuple = ("wtmetad_temperature",)) -> dict | None:
    """Well-tempered fit for one temperature, or None when that temperature has none.

    Args:
        T (int): temperature in K.
        trees (tuple): see fit_path.
    """
    p = fit_path(T, trees)
    return json.load(open(p)) if p else None


# the two single-fill campaigns: the same bias factors and heights at two paces, whose
# achieved residual barriers fall either side of the two-kT rule
SINGLE = (("wtmetad single, outside rule", ("wtmetad_temperature",),
           {300: 1.75, 285: 0.85, 270: 1.26}),
          # 270 K was extended from 50 to 200 runs; the larger fit stands in for the smaller
          ("wtmetad single, inside rule", ("wtmetad_temperature_slowpace_n200", "wtmetad_temperature_slowpace"),
           {300: 3.74, 285: 2.60, 270: 2.72}),
          # the same protocol along R_ee, 200 runs per temperature; its residual is not measured
          ("wtmetad single, R_ee", ("wtmetad_temperature_Ree",), {}))


def rows() -> pd.DataFrame:
    """One row per (temperature, estimator) with the rate, gamma, the barrier it implies, and
    the cost of the condition against the mean first-passage time that barrier implies."""
    out = []
    for T in TEMPS:
        d = ladder(T)
        dbar = BARRIER_KT[T] - BARRIER_KT[312]
        # the unbiased rate is known only at 312 K; below it the barrier sets the expectation
        mfpt_us = np.exp(dbar) / np.exp(REF_LNK)
        runs = sum(s["total"] for s in d["set_reports"])
        cost_us = sum(s["tau_obs"] * s["total"] for s in d["set_reports"]) * 1e6
        for est, k, sd, g in (("EATR-flooding", d["logk0"], d["bootstrap_logk0_std"], d["gamma"]),
                              ("gamma = 1", d["opes_logk0"], d["bootstrap_opes_logk0_std"], 1.0)):
            out.append(dict(temperature_K=T, family="opes ladder", estimator=est,
                            ln_k=k - LN_US, ln_k_sd=sd, gamma=g, d_barrier_kT=dbar,
                            offset=(k - LN_US - REF_LNK) + dbar, runs=runs, cost_us=cost_us,
                            mfpt_us=mfpt_us, cost_over_mfpt=cost_us / mfpt_us,
                            boost_obs=mfpt_us / (cost_us / runs)))
        for fam, trees, resid in SINGLE:
            w = wtmetad(T, trees)
            if not w:
                continue
            cmd = fit_path(T, trees).with_suffix(".cmd").read_text()
            n = int(re.search(r"committed=(\d+)", cmd).group(1))
            cost = n * np.exp(-w["ln(k_obs)"]) * 1e6
            for est, key, g in (("EATR", "EATR CDF ln k", w["EATR CDF gamma"]),
                                ("iMetaD", "iMetaD CDF ln k", 1.0)):
                out.append(dict(temperature_K=T, family=fam, estimator=est,
                                ln_k=w[key] - LN_US, ln_k_sd=np.nan, gamma=g,
                                residual_kT=resid.get(T), d_barrier_kT=dbar,
                                offset=(w[key] - LN_US - REF_LNK) + dbar, runs=n, cost_us=cost,
                                mfpt_us=mfpt_us, cost_over_mfpt=cost / mfpt_us,
                                boost_obs=mfpt_us / (cost / n)))
    df = pd.DataFrame(out)
    for fam, trees, _ in SINGLE:
        if not (df.family == fam).any():
            raise SystemExit(f"no well-tempered fits found under {[str(dataset(f'fits.{t}')) for t in trees]}")
    return df


def per_rung() -> pd.DataFrame:
    """One row per (temperature, rung): the fill it achieved and the rate it implies."""
    out = []
    for T in TEMPS:
        d = ladder(T)
        kT = KB * T
        bias = {s["barrier"]: s["avg_max_bias"] for s in d["set_reports"]}
        labels = d["flooding_diagnostics"]["convergence_analysis"]["sorted_barrier_labels"]
        for b, lk in zip(labels, d["flooding_diagnostics"]["ln_k0_per_set_best"]):
            out.append(dict(temperature_K=T, barrier_kJmol=b,
                            residual_kT=BARRIER_KT[T] - bias[b] / kT, ln_k=lk - LN_US,
                            ln_k_ladder=d["flooding_diagnostics"]["logk0_best"] - LN_US))
    return pd.DataFrame(out)


def convergence() -> pd.DataFrame:
    """One row per (temperature, number of bias sets used) from the fit's own report."""
    out = []
    for T in TEMPS:
        ca = ladder(T)["flooding_diagnostics"]["convergence_analysis"]
        for n, g, k in zip(ca["n_sets"], ca["gamma"], ca["logk0"]):
            out.append(dict(temperature_K=T, n_sets=n, gamma=g, ln_k=k - LN_US,
                            ln_k_all_sets=ladder(T)["flooding_diagnostics"]["logk0_best"] - LN_US))
    return pd.DataFrame(out)


def profiles() -> pd.DataFrame:
    """Reweighted free energy along Q, second half of each scan run, zeroed at the folded minimum."""
    d = pd.read_csv(ROOT / "analysis/fes_temperature_scan_profiles.csv")
    return d[(d.estimate == "reweight") & (d.window == "second half")]


def main() -> None:
    """Write the table and the three-panel figure."""
    r, pr, cv, prof = rows(), per_rung(), convergence(), profiles()
    r.to_csv(ROOT / "analysis/temperature_rates.csv", index=False)
    pr.to_csv(ROOT / "analysis/temperature_rates_per_rung.csv", index=False)
    cv.to_csv(ROOT / "analysis/temperature_convergence.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(6.69, 2.4), layout="constrained")

    # (a) the free energy the barriers are read from, with each barrier marked
    for T, g in prof.groupby("temperature_K"):
        for seed, sg in g.groupby("seed"):
            q, f = sg.q.values, sg.F_kT.values
            ax[0].plot(q, f, color=COLORS[T], lw=1.0, alpha=0.85,
                       label=f"{T:g} K" if seed == sorted(g.seed.unique())[0] else None)
        # the barrier of this temperature, from the top of the mean profile
        m = g.groupby("q").F_kT.mean()
        q, f = m.index.values, m.values
        fold = np.argmin(np.where(q > 0.65, f, np.inf))
        unf = np.argmin(np.where(q < 0.35, f, np.inf))
        lo, hi = sorted((fold, unf))
        top = lo + int(np.argmax(f[lo:hi + 1]))
        ax[0].plot(q[top], f[top], "v", ms=5, color=COLORS[T], mec="white", mew=0.6, zorder=5)
    ax[0].set_xlabel("$Q$")
    ax[0].set_ylabel(r"$F$ ($k_{\rm B}T$)")
    ax[0].set_xlim(0.15, 1.0)
    ax[0].set_ylim(-1, 21)
    ax[0].legend(frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.4,
                 labelspacing=0.25, borderaxespad=0.2)

    # (b) rate against barrier, with the parameter-free prediction
    x = np.array([0, 11.0])
    ax[1].plot(x, REF_LNK - x, color="0.4", lw=1.2, ls=(0, (4, 2)), zorder=1,
               label=r"$-\Delta$barrier")
    for est, mk, fc, col in (("EATR-flooding", "o", None, "#b02418"),
                             ("gamma = 1", "s", "none", "#3f7fbf")):
        g = r[(r.family == "opes ladder") & (r.estimator == est)]
        ax[1].errorbar(g.d_barrier_kT, g.ln_k, yerr=g.ln_k_sd, fmt=mk, ms=4.5, lw=0,
                       elinewidth=0.8, capsize=2, zorder=3, mfc=fc, color=col,
                       label="EATR-flooding" if est == "EATR-flooding" else r"$\gamma = 1$")
    for fam, mk, fc, lab in (("wtmetad single, outside rule", "^", "none",
                              r"one fill, residual $<2\,k_{\rm B}T$"),
                             ("wtmetad single, inside rule", "v", "#1f3b73",
                              r"one fill, residual $>2\,k_{\rm B}T$")):
        g = r[(r.family == fam) & (r.estimator == "EATR")]
        ax[1].plot(g.d_barrier_kT, g.ln_k, mk, ms=4.5, mfc=fc, color="#1f3b73",
                   zorder=3, label=lab)
    ax[1].set_xlabel(r"$\Delta$ barrier from 312 K ($k_{\rm B}T$)")
    ax[1].set_ylabel(r"$\ln (k / \mu{\rm s}^{-1})$")
    ax[1].legend(frameon=False, loc="lower left", handletextpad=0.4, borderaxespad=0.2,
                 labelspacing=0.3)

    # (c) how many fills it takes to pin gamma
    for T, g in cv.groupby("temperature_K"):
        ax[2].plot(g.n_sets, g.ln_k, "o-", ms=4, lw=1.1, color=COLORS[T], label=f"{T:g} K")
        ax[2].axhline(ladder(T)["flooding_diagnostics"]["logk0_best"] - LN_US,
                      color=COLORS[T], lw=0.7, ls=(0, (1, 1.5)))
    ax[2].set_xlabel("bias sets used in the fit")
    ax[2].set_ylabel(r"$\ln (k / \mu{\rm s}^{-1})$")
    ax[2].set_xticks([3, 4, 5])
    for i, a in enumerate(ax):
        a.text(0.02, 0.97, "(" + "abc"[i] + ")", transform=a.transAxes, va="top", weight="bold")
    fig.savefig(ROOT / "analysis/temperature_rates.pdf")

    # the numbers the text will quote
    for est in ("EATR-flooding", "gamma = 1"):
        g = r[(r.family == "opes ladder") & (r.estimator == est) & (r.temperature_K != 312)]
        slope = np.polyfit(g.d_barrier_kT, -g.ln_k, 1)[0]
        print(f"{est:>14}: slope {slope:.3f} (unit slope expected), "
              f"offset {g.offset.mean():+.2f} +- {g.offset.std(ddof=1):.2f} ln units")
    print("per-rung spread within a temperature, ln units: "
          + ", ".join(f"{T:g} K {g.ln_k.max() - g.ln_k.min():.2f}"
                      for T, g in pr.groupby("temperature_K"))
          + f"; fill spans {pr.residual_kT.min():+.2f} to {pr.residual_kT.max():+.2f} kT")
    print("wrote analysis/temperature_rates.csv, _per_rung.csv and temperature_rates.pdf")


if __name__ == "__main__":
    main()
