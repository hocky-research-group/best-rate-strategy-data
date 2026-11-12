"""Bias-design recipe: choose WT-MetaD (bias factor, height, pace) before running.

Usage:
    uv run python scripts/bias_design_recipe.py recommend --sigma-a 0.05 --sigma 0.02 \
        --k0 1.54e6 --gamma 0.85 --barrier-kT 6 --tau-rel 300
    uv run python scripts/bias_design_recipe.py validate

Inputs (from a short unbiased run and one pilot):
    sigma_A     unbiased basin width along the CV (same units as the hill width)
    sigma       hill width
    k0          rough unbiased rate (s^-1), e.g. from a pilot EATR/EATRf fit or a guess
    gamma       biasing efficiency of the CV (pilot EATR or EATR-flooding fit)
    barrier_kT  apparent barrier along the CV (kT)
    tau_rel     relaxation time of the CV in the reactant basin (ps): the integral of its
                autocorrelation about the basin mean
Model (inefficiency manuscript eqs 10-12 with the unbroadened overlap, which is
what the protein G data follow):
    h_eff = h0 sigma / sqrt(sigma^2 + 2 sigma_A^2)
    V_n   = (b-1) kT ln(1 + n h_eff / ((b-1) kT))
    escape when k0 exp(gamma V_n / kT) n tau = 1  ->  n*, t* = n* tau, boost = 1/(k0 t*)
Rule, three conditions that must all hold (`verdict`):
    residual barrier  barrier_kT - gamma V_n* >= RESIDUAL_MIN_KT: the escape stays a thermal
                      fluctuation over a partly filled basin
    pace              tau / tau_rel >= PACE_MIN_OVER_RELAX: the CV relaxes between hills, so
                      the bias is quasi-static and the growth law holds
    boost             1/(k0 t*) >= BOOST_MIN: a bias so gentle that it barely accelerates is
                      accurate but wasteful

`validate` applies the rule to the protein G grid using only sigma_A, the barrier, the
relaxation time and a single pilot gamma per CV (the OPES EATR-flooding gamma at pace
1000, an independent data set), then reports the EATR hit rate inside and outside the
recommended set per CV. The relaxation times are those of the unbiased first-passage runs
where the CV was recorded there (Q, R_g, R_ee) and of nearly unbiased grid segments
otherwise (RMSD).
Outputs: analysis/bias_design_validation.csv (the one table every downstream script reads
the verdict from), analysis/bias_design_table.csv
"""
import argparse
import json
import math
from pathlib import Path

import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
KT_KJ = 0.0083144626 * 312.0
DT_PS = 0.01
RESIDUAL_MIN_KT = 2.0
PACE_MIN_OVER_RELAX = 0.3
BOOST_MIN = 3.0
BF_GRID = (5, 10, 15, 20)
H_GRID = (0.1, 0.25, 0.5, 1.0)
PACE_GRID = (100, 500, 1000, 5000, 10000, 30000, 50000, 100000)
CV_KEY = {"q": "Q", "rmsd": "RMSD", "Rg": "Rg", "Ree": "Ree"}


def predict(sigma_a: float, sigma: float, k0_per_s: float, gamma: float, b: float, h0_kj: float,
            tau_ps: float, kt_kj: float = KT_KJ) -> dict:
    """Predicted escape hill count, time, boost and fill for one parameter set.

    Args:
        sigma_a (float): unbiased basin width along the CV.
        sigma (float): hill width.
        k0_per_s (float): unbiased rate in s^-1.
        gamma (float): biasing efficiency.
        b (float): bias factor (> 1).
        h0_kj (float): initial hill height (kJ/mol).
        tau_ps (float): deposition pace (ps).
        kt_kj (float): kT in kJ/mol.
    """
    heff = h0_kj * sigma / math.sqrt(sigma ** 2 + 2 * sigma_a ** 2)
    k0_ps = k0_per_s * 1e-12
    n = np.logspace(0, 9, 6000)
    v = (b - 1) * kt_kj * np.log1p(n * heff / ((b - 1) * kt_kj))
    expected = k0_ps * np.exp(gamma * v / kt_kj) * n * tau_ps
    idx = int(np.argmax(expected >= 1.0))
    if expected[idx] < 1.0:
        return dict(n_star=np.inf, t_star_ns=np.inf, boost=np.nan, V_escape_kT=np.nan, h_eff=heff)
    t_star = n[idx] * tau_ps
    return dict(n_star=float(n[idx]), t_star_ns=t_star / 1e3, boost=1.0 / (k0_ps * t_star),
                V_escape_kT=float(v[idx] / kt_kj), h_eff=heff)


def verdict(p: dict, gamma: float, barrier_kT: float, tau_ps: float, tau_rel_ps: float) -> dict:
    """The design rule applied to one prediction.

    Args:
        p (dict): output of `predict` for the protocol.
        gamma (float): biasing efficiency supplied to the rule.
        barrier_kT (float): apparent barrier along the CV (kT).
        tau_ps (float): deposition pace (ps).
        tau_rel_ps (float): relaxation time of the CV in the basin (ps).
    Returns the predicted residual barrier (kT), the pace over the relaxation time and
    whether all three conditions hold. A protocol that never escapes fails the boost
    condition.
    """
    residual = barrier_kT - gamma * p["V_escape_kT"]
    ratio = tau_ps / tau_rel_ps
    ok = bool(residual >= RESIDUAL_MIN_KT) and ratio >= PACE_MIN_OVER_RELAX and bool(p["boost"] >= BOOST_MIN)
    return dict(residual_pred_kT=residual, tau_over_relax=ratio, recommended=ok)


def relaxation_table() -> pd.DataFrame:
    """Relaxation time of each CV in the folded basin, one row per CV with its source.

    Prefers the unbiased first-passage runs (analysis/cv_autocorrelation_unbiased.csv) and
    falls back to the nearly unbiased grid segments (analysis/cv_autocorrelation.csv) for a
    CV those runs did not record.
    """
    ac = pd.read_csv(ROOT / "analysis/cv_autocorrelation.csv")
    ac["cv"] = ac.cv.map(CV_KEY)
    ac["source"] = "nearly unbiased segments"
    ub = pd.read_csv(ROOT / "analysis/cv_autocorrelation_unbiased.csv")
    ub["cv"] = ub.cv.map(CV_KEY)
    ub["source"] = "unbiased runs"
    return pd.concat([ub, ac[~ac.cv.isin(ub.cv)]], ignore_index=True)


def recommend(args: argparse.Namespace) -> pd.DataFrame:
    """Table of every grid parameter set with its prediction and a verdict."""
    rows = []
    for b in BF_GRID:
        for h0 in H_GRID:
            for pace in PACE_GRID:
                p = predict(args.sigma_a, args.sigma, args.k0, args.gamma, b, h0, pace * DT_PS)
                v = verdict(p, args.gamma, args.barrier_kT, pace * DT_PS, args.tau_rel)
                rows.append(dict(bf=b, height=h0, pace=pace, **p, **v))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis/bias_design_table.csv", index=False)
    rec = df[df.recommended].sort_values("t_star_ns")
    print(f"recommended (residual >= {RESIDUAL_MIN_KT} kT, pace >= {PACE_MIN_OVER_RELAX} x relaxation, "
          f"boost >= {BOOST_MIN}): {len(rec)} of {len(df)} parameter sets; cheapest first:")
    print(rec.head(12).round(3).to_string(index=False))
    return df


def validate(_: argparse.Namespace) -> None:
    """Prospective test on the grid: EATR hit rate inside vs outside the recommended set."""
    pmf = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    ref = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())
    k0 = ref["k_mle_per_us"] * 1e6
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    grid = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    grid = grid[(grid.n == grid.n_total) & (grid.bf >= 5) & (grid.estimator == "EATR CDF")]
    tau_rel = relaxation_table().set_index("cv").tau_int_ps
    sig = {"Q": 0.02, "RMSD": 0.02, "Rg": 0.02, "Ree": 0.06}
    key = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}
    rows = []
    for cv in ("Q", "RMSD", "Rg", "Ree"):
        pilot = mt[(mt.family == "opesf") & (mt.cv == cv) & (mt.estimator == "EATRf") & (mt.pace == 1000)]
        gamma = float(pilot.gamma.iloc[0])
        sa = pmf[key[cv]]["basin_sigma"]; barrier_kT = pmf[key[cv]]["barrier_kT"]
        for _, r in grid[grid.cv == cv].iterrows():
            p = predict(sa, sig[cv], k0, gamma, r.bf, r.height, r.pace * DT_PS)
            v = verdict(p, gamma, barrier_kT, r.pace * DT_PS, tau_rel[cv])
            rows.append(dict(cv=cv, bf=r.bf, height=r.height, pace=r.pace, gamma_pilot=gamma,
                             pred_boost=p["boost"], pred_t_ns=p["t_star_ns"], pred_V_kT=p["V_escape_kT"],
                             barrier_kT=barrier_kT, tau_rel_ps=tau_rel[cv], **v,
                             eatr_err=r.mean_err, obs_cost_us=r.cost_us))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "analysis/bias_design_validation.csv", index=False)
    summ = df.groupby(["cv", "recommended"]).agg(n=("eatr_err", "size"),
                                                 within_1=("eatr_err", lambda x: (x.abs() <= 1).mean()),
                                                 within_0p5=("eatr_err", lambda x: (x.abs() <= 0.5).mean()),
                                                 median_abs=("eatr_err", lambda x: x.abs().median()),
                                                 median_cost_us=("obs_cost_us", "median")).round(2)
    print("EATR accuracy of grid conditions inside vs outside the recommended set (prediction uses sigma_A, "
          "barrier, relaxation time, k0 and the OPES pilot gamma only):")
    print(summ.to_string())
    print("pilot gamma per CV:", df.groupby("cv").gamma_pilot.first().round(2).to_dict())
    print("relaxation time (ps) per CV:", tau_rel.round(0).to_dict())


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recommend")
    r.add_argument("--sigma-a", dest="sigma_a", type=float, required=True)
    r.add_argument("--sigma", type=float, required=True)
    r.add_argument("--k0", type=float, required=True, help="unbiased rate, s^-1")
    r.add_argument("--gamma", type=float, required=True)
    r.add_argument("--barrier-kT", dest="barrier_kT", type=float, required=True, help="apparent barrier, kT")
    r.add_argument("--tau-rel", dest="tau_rel", type=float, required=True, help="relaxation time in the basin, ps")
    r.set_defaults(func=recommend)
    v = sub.add_parser("validate"); v.set_defaults(func=validate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
