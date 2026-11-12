"""PMF with overlaid average applied bias, and VES validity checks.

Usage:
    uv run python scripts/plot_pmf_bias_overlays.py

Inputs : data/heuristics/pmf_unbiased/pmf_<cv>.dat (kJ/mol, zero at folded min)
         data/results/bias_profiles/profile_<family>_<cv>_<set>.dat
         data/results/bias_profiles/vesbias_<cv>_fc<fc>.dat, ves_stage1_<cv>_fc<fc>.dat
         analysis/master_sets.csv, analysis/master_table.csv
Outputs: figures/fig5_pmf_bias_overlays.png  (rows: WT-MetaD, OPES, VES;
         columns: Q = working, Ree = failing at gamma = 1, Rg for flooding)
         figures/fig6_ves_checks.png
Each overlay panel shows the unbiased PMF F (black, kT) and the biased landscape
F + <V> for each set (sequential colours, light = weakest bias), with the
committor boundary (Q < 0.35 defines unfolding) and the PMF barrier marked.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
KT = 0.0083144626 * 312.0
L6 = np.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
PMFKEY = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}
PROF = dataset("runs.bias_profiles")
SEQ = plt.cm.YlGnBu


def pmf(cv: str) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased PMF along cv in kT (x, F).

    Args:
        cv (str): Q | RMSD | Rg | Ree.
    """
    a = np.loadtxt(dataset("pmf_unbiased") / f"pmf_{PMFKEY[cv]}.dat")
    return a[:, 0], a[:, 1] / KT


def overlay(ax, cv: str, family: str, sets: list[tuple[str, str]], summary: dict) -> None:
    """Draw F and F + <V> for the listed sets on one axis.

    Args:
        ax: matplotlib axis.
        cv (str): biased CV.
        family (str): imetad | opes | ves.
        sets (list): (set name, legend label) pairs.
        summary (dict): PMF summary for the barrier position.
    """
    x, F = pmf(cv)
    ax.plot(x, F, color="k", lw=1.6, label="unbiased PMF")
    n = len(sets)
    for i, (name, lab) in enumerate(sets):
        f = PROF / f"profile_{family}_{cv}_{name}.dat"
        if not f.exists():
            continue
        p = np.loadtxt(f)
        ok = p[:, 2] >= 20
        V = np.interp(x, p[ok, 0], p[ok, 1] / KT, left=np.nan, right=np.nan)
        col = SEQ(0.35 + 0.6 * i / max(n - 1, 1))
        ax.plot(x, F + V, color=col, lw=1.3, label=lab)
        ax.plot(x, V, color=col, lw=0.8, ls=":")
    b = summary[PMFKEY[cv]]
    ax.axvline(b["barrier_pos"], color="#7f7f7f", lw=0.8, ls="--")
    if cv == "Q":
        ax.axvline(0.35, color="#D55E00", lw=0.8, ls="-.")
        ax.set_xlim(0, 1)
    elif cv == "Ree":
        ax.set_xlim(1.5, 8)
    elif cv == "Rg":
        ax.set_xlim(0.95, 2.6)
    ax.set_ylim(-1, 12)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def main() -> None:
    """Draw both figures."""
    summary = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    ms = pd.read_csv(ROOT / "analysis/master_sets.csv")

    def note(fam: str, cv: str, key: dict) -> str:
        r = mt[(mt.family == fam) & (mt.cv == cv)]
        for k, v in key.items():
            r = r[r[k] == v]
        e = r[r.estimator == "EATRf"]; o = r[r.estimator == "OPESf"]
        if len(e) == 0:
            return ""
        return f"EATRf err {e.err_ln.iloc[0]:+.2f}, γ {e.gamma.iloc[0]:.2f}; γ=1 err {o.err_ln.iloc[0]:+.2f}" if len(o) else f"EATRf err {e.err_ln.iloc[0]:+.2f}, γ {e.gamma.iloc[0]:.2f}"

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    # row 1: WT-MetaD (Q, Ree) at bf 10, h 0.5, three paces; third panel: iMetaD vs EATR errors
    for j, cv in enumerate(("Q", "Ree")):
        sets = [(f"bf10_height0.5_pace{p}", f"pace {p} steps") for p in (100, 1000, 100000)]
        overlay(axes[0, j], cv, "imetad", sets, summary)
        g = mt[(mt.family == "imetad") & (mt.cv == cv) & (mt.bf == 10) & (mt.height == 0.5) & mt.valid]
        txt = "\n".join(f"pace {int(p)}: iMetaD err {g[(g.pace == p) & (g.estimator == 'iMetaD CDF')].err_ln.iloc[0]:+.1f}, EATR err {g[(g.pace == p) & (g.estimator == 'EATR CDF')].err_ln.iloc[0]:+.1f}"
                        for p in (100, 1000, 100000) if len(g[(g.pace == p) & (g.estimator == 'iMetaD CDF')]))
        axes[0, j].set_title(f"WT-MetaD on {cv} (bias near escape; bf 10, h 0.5 kJ/mol)", fontsize=8)
        axes[0, j].text(0.98, 0.97, txt, transform=axes[0, j].transAxes, fontsize=7, ha="right", va="top")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.0, 0.95, "Reading the overlays\n\nblack: unbiased PMF (kT)\ncolour: F + <V>, light → dark = weaker → stronger bias\ndotted: <V> alone\ngrey dashed: PMF barrier; orange dash-dot: Q = 0.35 unfolding boundary\n\nWorking case: F + <V> is flat inside the folded basin and\nuntouched at the barrier (bias confined to the reactant).\nFailing case (Ree): the unfolded ensemble overlaps the\nfolded basin along Ree, so bias meant for the basin lands\non the transition region; γ = 1 then overcounts the boost.",
                    fontsize=8, va="top", family="monospace")
    # row 2: OPES at pace 1000 (Q, Ree, Rg)
    for j, cv in enumerate(("Q", "Ree", "Rg")):
        sets = [(f"barrier{b}_pace1000", f"ΔE = {b} kJ/mol") for b in (5, 7, 9, 11, 13)]
        overlay(axes[1, j], cv, "opes", sets, summary)
        axes[1, j].set_title(f"OPES flooding on {cv} (pace 1000)\n{note('opesf', cv, {'pace': 1000})}", fontsize=8)
    # row 3: VES cutoff flooding (Q, Ree, Rg)
    for j, cv in enumerate(("Q", "Ree", "Rg")):
        sets = [(f"fc{fc}", f"F_c = {fc} kJ/mol") for fc in (5, 7, 9, 11, 13, 15)]
        overlay(axes[2, j], cv, "ves", sets, summary)
        axes[2, j].set_title(f"VES cutoff flooding on {cv} (N = 200)\n{note('vesf', cv, {'barrier_label': 200})}", fontsize=8)
        axes[2, j].set_xlabel(cv + (" (nm)" if cv != "Q" else ""), fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("free energy (kT)", fontsize=9)
    axes[1, 0].legend(fontsize=7, frameon=False, loc="upper left"); axes[2, 0].legend(fontsize=7, frameon=False, loc="upper left"); axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Unbiased PMF with the average applied bias: working (Q) and failing (Ree) coordinates", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig5_pmf_bias_overlays.png", dpi=160)
    print("wrote figures/fig5_pmf_bias_overlays.png")

    # ---- VES checks ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    cols = {"Q": "#0072B2", "RMSD": "#009E73", "Rg": "#E69F00", "Ree": "#D55E00"}
    ax = axes[0, 0]
    for i, fc in enumerate((5, 9, 13, 15)):
        f = PROF / f"ves_stage1_Q_fc{fc}.dat"
        if f.exists():
            tr = np.loadtxt(f)
            ax.plot(tr[:, 0] / 1e3, tr[:, 1], lw=0.6, color=SEQ(0.35 + 0.6 * i / 3), label=f"F_c = {fc}")
            ax.axhline(fc, color=SEQ(0.35 + 0.6 * i / 3), lw=0.6, ls="--")
    ax.set_xlabel("stage-1 time (ns)"); ax.set_ylabel("ves.bias at the walker (kJ/mol)"); ax.set_title("(a) Q stage 1: bias fills up to the cutoff", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[0, 1]
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        mx = []
        for fc in (5, 7, 9, 11, 13, 15):
            f = PROF / f"vesbias_{cv}_fc{fc}.dat"
            mx.append(np.loadtxt(f)[:, 1].max() if f.exists() else np.nan)
        ax.plot([5, 7, 9, 11, 13, 15], mx, "o-", color=cols[cv], ms=4, label=cv)
    ax.plot([4, 16], [4, 16], color="#7f7f7f", lw=0.8); ax.set_xlabel("cutoff F_c (kJ/mol)"); ax.set_ylabel("max of static bias (kJ/mol)"); ax.set_title("(b) static stage-2 bias reaches F_c", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[0, 2]
    x, F = pmf("Q")
    ax.plot(x, F, color="k", lw=1.5, label="PMF")
    for i, fc in enumerate((5, 9, 13)):
        f = PROF / f"vesbias_Q_fc{fc}.dat"
        if f.exists():
            b = np.loadtxt(f); V = np.interp(x, b[:, 0], b[:, 1]) / KT
            ax.plot(x, F + V, color=SEQ(0.35 + 0.6 * i / 2), lw=1.2, label=f"F + V, F_c = {fc}")
            ax.axhline(fc / KT, color=SEQ(0.35 + 0.6 * i / 2), lw=0.6, ls="--")
    ax.set_xlim(0.3, 1); ax.set_ylim(-0.5, 9); ax.set_xlabel("Q"); ax.set_ylabel("kT"); ax.set_title("(c) Q: F + V is level at F_c inside the basin", fontsize=9); ax.legend(fontsize=7, frameon=False)
    v = ms[(ms.family == "vesf") & (ms.label == 200)]
    ax = axes[1, 0]
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        g = v[v.cv == cv].sort_values("set_label")
        ax.plot(g.set_label, g.ks_p, "o-", color=cols[cv], ms=4, label=cv)
    ax.axhline(0.05, color="#7f7f7f", lw=0.8, ls=":"); ax.set_yscale("log"); ax.set_xlabel("cutoff F_c (kJ/mol)"); ax.set_ylabel("KS p (exponential escape times)"); ax.set_title("(d) escape times stay Poissonian", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 1]
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        g = v[v.cv == cv].sort_values("ln_exp_beta_v")
        s = np.polyfit(g.ln_exp_beta_v, g.log_k_obs_s, 1)[0]
        ax.plot(g.ln_exp_beta_v, g.log_k_obs_s - L6, "o-", color=cols[cv], ms=4, label=f"{cv}: slope {s:.2f}")
        for _, r in g.iterrows():
            ax.annotate(f"{int(r.set_label)}", (r.ln_exp_beta_v, r.log_k_obs_s - L6), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("ln <e^{βV}>"); ax.set_ylabel("ln k_obs (μs⁻¹)"); ax.set_title("(e) observed rate vs acceleration (labels: F_c)", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 2]
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        g = v[v.cv == cv].sort_values("set_label")
        ax.plot(g.set_label, g.tau_obs_us * 1e3, "o-", color=cols[cv], ms=4, label=cv)
    ax.set_yscale("log"); ax.set_xlabel("cutoff F_c (kJ/mol)"); ax.set_ylabel("mean escape time (ns)"); ax.set_title("(f) escape time vs cutoff (non-monotonic = unconverged stage 1)", fontsize=9); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)
    fig.suptitle("VES cutoff flooding: is it done properly?", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig6_ves_checks.png", dpi=160)
    print("wrote figures/fig6_ves_checks.png")


if __name__ == "__main__":
    main()
