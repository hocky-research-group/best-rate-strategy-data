"""Censoring robustness of the flooding estimators (Supporting Information).

Usage: python scripts/censoring_robustness.py
Input : data/verification/flooding_bootstrap_fix/summary.json and fits.tar.gz (torch job
        17376566; EATR-rate-analysis 14cee55 installed; 100 bootstrap resamples, seed
        12345). The OPES ladders at pace 1000 (BARRIER 5, 7, 9, 11, 13 kJ/mol, 50 runs per
        set) for Q and Ree, complete and right-censored at one wall time T: the (1 - f)
        quantile of the escape times of the slowest set, f = 0.1, 0.25, 0.5. A run that
        has not escaped by T ends at T without an event.
Output: analysis/censoring_robustness.csv, one row per ladder;
        figures/figS_censoring.png, the errors of EATR-flooding and of the gamma = 1
        (OPES-flooding) estimate, and the fitted gamma, against the fraction of all runs
        censored, with bootstrap 95% intervals.
Sanity check (printed): the complete ladders reproduce the EATRf and OPESf errors of the
same data in analysis/master_table.csv (Triasha Pal's fits, 1000 resamples) to within
the bootstrap spread.
"""
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
VER = dataset("verification") / "flooding_bootstrap_fix"
L6 = np.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
BLUE, VERM, INK, GREY = "#0072B2", "#D55E00", "#222222", "#8a8a8a"
NAME = {"Q": "$Q$", "Ree": "$R_{ee}$"}


def ladder_rows() -> pd.DataFrame:
    """One row per OPES ladder (complete or censored) with errors, gamma and intervals."""
    summary = json.loads((VER / "summary.json").read_text())
    rows = []
    with tarfile.open(VER / "fits.tar.gz") as tf:
        for c in summary["cases"]:
            if c["family"] != "opes":
                continue
            b = json.load(tf.extractfile(f"json/{c['case']}__new_b.json"))
            ref = c["reference"]
            m, n = sum(r["transitioned"] for r in ref), sum(r["total"] for r in ref)
            oci = b["bootstrap_opes_logk0_ci"]
            rows.append(dict(cv=c["cv"], f_slowest=c["frac"], T_ns=np.nan if c["T_ps"] is None else c["T_ps"] / 1e3,
                             censored_all=1 - m / n, events=m, runs=n,
                             censored_per_set=";".join(f"{r['censored_frac']:.2f}" for r in ref),
                             eatrf_err=b["logk0"] - L6 - REF, eatrf_lo=b["bootstrap_logk0_ci"][0] - L6 - REF,
                             eatrf_hi=b["bootstrap_logk0_ci"][1] - L6 - REF,
                             gamma=b["gamma"], gamma_lo=b["bootstrap_gamma_ci"][0], gamma_hi=b["bootstrap_gamma_ci"][1],
                             opesf_err=b["opes_logk0"] - L6 - REF, opesf_lo=oci[0] - L6 - REF, opesf_hi=oci[1] - L6 - REF))
    return pd.DataFrame(rows).sort_values(["cv", "f_slowest"]).reset_index(drop=True)


def sanity(df: pd.DataFrame) -> None:
    """Print the complete-ladder errors next to the master-table values for the same data.

    Args:
        df (pd.DataFrame): output of ladder_rows.
    """
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    for cv in ("Q", "Ree"):
        full = df[(df.cv == cv) & (df.f_slowest == 0)].iloc[0]
        m = mt[(mt.family == "opesf") & (mt.cv == cv) & (mt.pace == 1000)]
        e = float(m[m.estimator == "EATRf"].err_ln.iloc[0]); o = float(m[m.estimator == "OPESf"].err_ln.iloc[0])
        print(f"sanity {cv}: EATRf {full.eatrf_err:+.2f} [{full.eatrf_lo:+.2f}, {full.eatrf_hi:+.2f}] vs master {e:+.2f}; "
              f"OPESf {full.opesf_err:+.2f} [{full.opesf_lo:+.2f}, {full.opesf_hi:+.2f}] vs master {o:+.2f}")


def figure(df: pd.DataFrame) -> None:
    """Two columns (Q, Ree): errors on top, fitted gamma below, against the censored fraction.

    Args:
        df (pd.DataFrame): output of ladder_rows.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex="col", gridspec_kw=dict(height_ratios=(1.6, 1)))
    for j, cv in enumerate(("Q", "Ree")):
        d = df[df.cv == cv]
        x = d.censored_all.to_numpy() * 100
        ax = axes[0, j]
        ax.axhline(0, color=INK, lw=0.8)
        for lim in (-1, 1):
            ax.axhline(lim, color=GREY, lw=0.6, ls=":")
        for col, key, mk, lab in ((BLUE, "eatrf", "o", "EATR-flooding"), (VERM, "opesf", "s", r"$\gamma = 1$ (OPES-flooding)")):
            y = d[f"{key}_err"].to_numpy()
            yerr = np.vstack([y - d[f"{key}_lo"].to_numpy(), d[f"{key}_hi"].to_numpy() - y])
            ax.errorbar(x, y, yerr=yerr, color=col, marker=mk, ms=6, lw=2, capsize=3, elinewidth=1, label=lab)
        ax.set_title(NAME[cv], color=INK)
        ax.grid(True, axis="y", alpha=0.25, lw=0.5)
        axes[1, j].errorbar(x, d.gamma, yerr=np.vstack([d.gamma - d.gamma_lo, d.gamma_hi - d.gamma]),
                            color=BLUE, marker="o", ms=6, lw=2, capsize=3, elinewidth=1)
        axes[1, j].set_xlabel("runs censored (% of all runs in the ladder)", color=INK)
        axes[1, j].grid(True, axis="y", alpha=0.25, lw=0.5)
    axes[0, 0].set_ylabel(r"error in $\ln k$", color=INK)
    axes[1, 0].set_ylabel(r"fitted $\gamma$", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/figS_censoring.png", dpi=160)


def main() -> None:
    """Write the table and the figure and print the sanity check."""
    df = ladder_rows()
    df.to_csv(ROOT / "analysis/censoring_robustness.csv", index=False)
    print(df.drop(columns=["censored_per_set"]).round(3).to_string(index=False))
    sanity(df)
    figure(df)
    print("wrote analysis/censoring_robustness.csv, figures/figS_censoring.png")


if __name__ == "__main__":
    main()
