"""Test the well-tempered growth law and escape heuristic on protein G.

Usage:
    uv run python scripts/heuristics_growth_law.py

Theory (Mazzaferro, Limmer, Cossio, Hocky, "Origin of biasing inefficiency",
eqs 10-13): before escape every hill lands in the reactant basin and the bias
at the deposition point after n hills is
    V_n = (b-1) kT ln(1 + n h_eff / ((b-1) kT)),
    h_eff = h0 sigma / sqrt(sigma^2 + 2 b sigma_A^2),
with b the bias factor, h0 the initial hill height, sigma the hill width and
sigma_A the unbiased basin width along the CV. Escape is expected when the
expected number of escapes on the biased landscape reaches one,
    k0 exp(beta gamma V_n) n tau = 1,
with tau the deposition pace and gamma the biasing efficiency of the CV.

Inputs
    data/heuristics/pmf_unbiased/pmf_summary.json  (sigma_A per CV)
    data/results/trajectory_summary/imetad_<CV>_trajectories.csv
        per replica: nhills, last_hill_height (PLUMED records well-tempered
        heights scaled by b/(b-1)), t_final_ps
    analysis/subsampling_imetad.csv  (full-set EATR gamma per condition)
    data/reference_rate/reference_rate.json
Outputs
    analysis/growth_law_conditions.csv  one row per (cv, bf, height, pace) with
        observed mean hills at escape, largest bias at any deposition
        point V_obs = -(b-1) kT ln(h_min (1-1/b) / h0) (minimum recorded hill
        height; the last hill sits at the escape point and underestimates the fill), the growth-law
        prediction V_pred(n_obs), the predicted escape hill count n_star and
        time t_star from the heuristic with the fitted gamma, the observed mean
        escape time, the boost k_obs/k0 and the coefficient of variation of the
        escape times (regime diagnostics).
    figures/fig_growth_law.png
Sanity check: for b -> large and small n, V_pred -> n h_eff; and V_obs must be
non-negative and below (1 - 1/b) times the apparent barrier for every
condition with all replicas committed (printed as a count).
"""
import json
import math
from pathlib import Path

import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
KT = 0.0083144626 * 312.0  # kJ/mol
DT_PS = 0.01
SIGMA = {"Q": 0.02, "RMSD": 0.02, "Rg": 0.02, "Ree": 0.06}
PMFKEY = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}


def h_eff(h0: float, sigma: float, b: float, sigma_a: float) -> float:
    """Mean bias increment per hill at the next deposition point (eq 11).

    Args:
        h0 (float): initial hill height (kJ/mol).
        sigma (float): hill width in CV units.
        b (float): bias factor.
        sigma_a (float): unbiased basin standard deviation along the CV.
    """
    return h0 * sigma / math.sqrt(sigma ** 2 + 2 * b * sigma_a ** 2)


def v_n(n: np.ndarray, heff: float, b: float) -> np.ndarray:
    """Bias at the deposition point after n hills (eq 10), kJ/mol.

    Args:
        n (np.ndarray): number of hills.
        heff (float): increment per hill (kJ/mol).
        b (float): bias factor.
    """
    return (b - 1) * KT * np.log1p(n * heff / ((b - 1) * KT))


def n_star(k0_per_ps: float, gamma: float, heff: float, b: float, tau_ps: float) -> float:
    """Hill count at which k0 exp(beta gamma V_n) n tau reaches one.

    Args:
        k0_per_ps (float): unbiased rate (1/ps).
        gamma (float): biasing efficiency.
        heff (float): increment per hill (kJ/mol).
        b (float): bias factor.
        tau_ps (float): deposition pace (ps).
    """
    n = np.logspace(0, 8, 4000)
    expected = k0_per_ps * np.exp(gamma * v_n(n, heff, b) / KT) * n * tau_ps
    idx = np.argmax(expected >= 1.0)
    return float(n[idx]) if expected[idx] >= 1.0 else float("inf")


def main() -> None:
    """Build the per-condition table and the figure."""
    pmf = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    ref = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())
    k0_ps = ref["k_mle_per_us"] * 1e-6
    sub = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    full = sub[(sub.estimator == "EATR CDF") & (sub.n == sub.n_total)][["cv", "bf", "height", "pace", "gamma", "mean_err"]]
    rows = []
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        f = dataset("runs.grid_summary") / f"imetad_{cv}_trajectories.csv"
        if not f.exists() or f.stat().st_size == 0:
            print("missing or empty", f); continue
        t = pd.read_csv(f)
        t = t[t.committed == 1]
        sa = pmf[PMFKEY[cv]]["basin_sigma"]
        barrier = pmf[PMFKEY[cv]]["barrier_kJmol"]
        for (bf, h0, pace), g in t.groupby(["bf", "height", "pace"]):
            b, tau = float(bf), pace * DT_PS
            if b <= 1:
                continue  # bias factor 1 is untempered in PLUMED; the growth law of eq 10 does not apply
            heff = h_eff(h0, SIGMA[cv], b, sa)
            heff_unbroadened = h0 * SIGMA[cv] / math.sqrt(SIGMA[cv] ** 2 + 2 * sa ** 2)
            n_obs = g["nhills"].to_numpy(dtype=float)
            hcol = "min_hill_height" if "min_hill_height" in g else "last_hill_height"
            h_last = g[hcol].to_numpy(dtype=float) * (1 - 1 / b)
            v_obs = -(b - 1) * KT * np.log(np.clip(h_last / h0, 1e-12, None))
            tf = g["t_final_ps"].to_numpy(dtype=float)
            gam = full[(full.cv == cv) & (full.bf == b) & (full.height == h0) & (full.pace == pace)]
            gamma = float(gam["gamma"].iloc[0]) if len(gam) else np.nan
            ns = n_star(k0_ps, gamma, heff, b, tau) if np.isfinite(gamma) else np.nan
            ns_unb = n_star(k0_ps, gamma, heff_unbroadened, b, tau) if np.isfinite(gamma) else np.nan
            # effective increment per hill implied by the observed fill and hill count (eq 10 inverted)
            n_at = g["n_at_min_height"].to_numpy(dtype=float) if "n_at_min_height" in g else n_obs
            heff_fit = (b - 1) * KT * np.expm1(v_obs / ((b - 1) * KT)) / np.clip(n_at, 1, None)
            rows.append(dict(cv=cv, bf=b, height=h0, pace=pace, tau_ps=tau, n_replicas=len(g),
                             sigma_A=sa, h_eff=heff, h_eff_fit=float(np.nanmedian(heff_fit)),
                             h_eff_fit_over_h0=float(np.nanmedian(heff_fit) / h0), h_eff_eq11_over_h0=heff / h0,
                             h_eff_unbroadened_over_h0=heff_unbroadened / h0, gamma=gamma,
                             n_hills_obs=n_obs.mean(), n_hills_sd=n_obs.std(ddof=1) if len(g) > 1 else np.nan,
                             V_obs_kJmol=np.nanmean(v_obs), V_pred_kJmol=float(np.mean(v_n(n_obs, heff, b))),
                             V_obs_over_barrier=np.nanmean(v_obs) / barrier,
                             n_star=ns, t_star_ps=ns * tau, n_star_unbroadened=ns_unb, t_star_unbroadened_ps=ns_unb * tau,
                             V_pred_unbroadened_kJmol=float(np.mean(v_n(n_obs, heff_unbroadened, b))), t_obs_mean_ps=tf.mean(),
                             boost=(1 / tf.mean()) / k0_ps, cv_times=tf.std(ddof=1) / tf.mean() if len(g) > 1 else np.nan,
                             eatr_err=float(gam["mean_err"].iloc[0]) if len(gam) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis/growth_law_conditions.csv", index=False)
    ok = df[(df.V_obs_kJmol >= 0)]
    print(f"conditions: {len(df)}; V_obs >= 0 in {len(ok)}; "
          f"V_obs below (1-1/b) x barrier in {(df.V_obs_over_barrier < 1 - 1 / df.bf).sum()}")
    print(df.groupby("cv")[["n_hills_obs", "V_obs_kJmol", "V_pred_kJmol", "boost", "cv_times"]].median().round(2).to_string())
    print("effective increment per hill / h0: fitted vs eq 11 (well-tempered fixed point) vs unbroadened overlap")
    print(df.groupby(["cv", "bf"])[["h_eff_fit_over_h0", "h_eff_eq11_over_h0", "h_eff_unbroadened_over_h0"]].median().round(3).to_string())
    print("ratio t_obs / t_star (median by cv), eq-11 h_eff vs unbroadened h_eff:")
    print(pd.DataFrame({"eq11": (df.t_obs_mean_ps / df.t_star_ps).groupby(df.cv).median(),
                        "unbroadened": (df.t_obs_mean_ps / df.t_star_unbroadened_ps).groupby(df.cv).median(),
                        "log-spread(unb)": np.log(df.t_obs_mean_ps / df.t_star_unbroadened_ps).groupby(df.cv).std()}).round(2).to_string())
    print("V_obs / V_pred (median by cv), eq11 vs unbroadened:")
    print(pd.DataFrame({"eq11": (df.V_obs_kJmol / df.V_pred_kJmol).groupby(df.cv).median(),
                        "unbroadened": (df.V_obs_kJmol / df.V_pred_unbroadened_kJmol).groupby(df.cv).median()}).round(2).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Q": "#0072B2", "RMSD": "#009E73", "Rg": "#E69F00", "Ree": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for cv, g in df.groupby("cv"):
        axes[0].scatter(g.V_pred_unbroadened_kJmol, g.V_obs_kJmol, s=16, c=colors[cv], label=cv, alpha=0.8, lw=0)
        axes[1].scatter(g.t_star_unbroadened_ps / 1e3, g.t_obs_mean_ps / 1e3, s=16, c=colors[cv], alpha=0.8, lw=0)
        axes[2].scatter(g.boost, g.cv_times, s=16, c=colors[cv], alpha=0.8, lw=0)
    lim = [0, max(df.V_obs_kJmol.max(), df.V_pred_unbroadened_kJmol.max()) * 1.05]
    axes[0].plot(lim, lim, color="#7f7f7f", lw=0.8); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("V_n from hill count, eq 10 with unbroadened h_eff (kJ/mol)"); axes[0].set_ylabel("largest bias at a deposition point, from min hill height (kJ/mol)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    l2 = [min(df.t_star_unbroadened_ps.min(), df.t_obs_mean_ps.min()) / 1e3, max(df.t_star_unbroadened_ps.max(), df.t_obs_mean_ps.max()) / 1e3]
    axes[1].plot(l2, l2, color="#7f7f7f", lw=0.8)
    axes[1].set_xlabel("predicted escape time t* (ns), eq 12 with fitted γ, unbroadened h_eff"); axes[1].set_ylabel("observed mean escape time (ns)")
    axes[2].set_xscale("log"); axes[2].axhline(1, color="#7f7f7f", lw=0.8)
    axes[2].set_xlabel("boost k_obs / k0"); axes[2].set_ylabel("coefficient of variation of escape times")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig_growth_law.png", dpi=160)
    print("wrote figures/fig_growth_law.png")


if __name__ == "__main__":
    main()
