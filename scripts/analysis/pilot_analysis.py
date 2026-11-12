"""Cost and informativeness of a two-level flooding pilot for gamma.

Usage: python scripts/pilot_analysis.py
Input : data/subsampling/flooding_pilot/task_*.jsonl.gz (scripts/flooding_subsets.py, family
        pilot): no-bootstrap EATR-flooding fits of 10 committed OPES runs at each of two
        BARRIER levels, (5, 9) or (9, 13) kJ/mol, pace 1000, 50 draws per CV and pair;
        analysis/bias_design_validation.csv (grid conditions of bias factors 5-20 with the
        reference verdicts and EATR errors); the unbiased PMF summary; the reference rate.
Output: analysis/pilot_draws.csv, one row per draw; analysis/pilot_recommended.csv, the grid
        conditions each draw of the charged pair recommends (one row per draw and condition,
        with the draw's pilot cost); and analysis/pilot_summary.csv, one row
        per CV and pair: the simulated time of the pilot, the pilot's gamma and ln k0 error
        (median, 10th and 90th percentiles), the fractions of draws with gamma >= 0.8 and
        < 0.5, and, with the design rule re-run on each draw's own gamma and k0 in place of
        the reference inputs, the agreement of its verdicts with the reference verdicts over
        the grid (median, 10th percentile), the median number of conditions it recommends
        and the fraction of those on which EATR was within one ln unit (median, 10th
        percentile).
The pair charged to the strategies of the error-cost front is CHARGED_PAIR, (5, 9) kJ/mol:
the two lowest levels of the ladder, whose pilot places gamma on the correct side of 0.8
most reliably. The cheaper (9, 13) pair is summarised alongside.
Sanity check (printed): the median pilot cost against 10 times the sum of the mean escape
times of the two levels in the trajectory summaries.
"""
import gzip
import json
import sys
from pathlib import Path

import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

ROOT = _paths.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from bias_design_recipe import DT_PS, predict, verdict  # noqa: E402

L6 = np.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
CHARGED_PAIR = "5-9"
SIGMA = {"Q": 0.02, "RMSD": 0.02, "Rg": 0.02, "Ree": 0.06}
PMF_KEY = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}


def pilot_draws() -> pd.DataFrame:
    """One row per pilot fit: CV, level pair, draw, cost, gamma, ln k0 and its errors."""
    rows = []
    for pk in sorted((dataset("fits.flooding_pilot")).glob("task_*.jsonl.gz")):
        with gzip.open(pk, "rt") as fh:
            for rec in map(json.loads, fh):
                if "data" not in rec:
                    print(f"WARNING: {pk.name}: failed pilot fit {rec['file']}", file=sys.stderr)
                    continue
                d = rec["data"]
                cv, pair = rec["file"].split("/")[1].split("_pace1000_b")
                rows.append(dict(cv=cv, pair=pair, draw=int(rec["file"].split("_rep")[1].split(".")[0]),
                                 cost_us=sum(s["tau_obs"] * s["transitioned"] for s in d["set_reports"]) * 1e6,
                                 gamma=d["gamma"], lnk0_s=d["logk0"], err=d["logk0"] - L6 - REF,
                                 opesf_err=d["opes_logk0"] - L6 - REF))
    if not rows:
        raise SystemExit("no pilot fits under data/subsampling/flooding_pilot")
    return pd.DataFrame(rows).sort_values(["cv", "pair", "draw"]).reset_index(drop=True)


def verdicts(cv: str, gamma: float, k0_per_s: float, conds: pd.DataFrame, pmf: dict) -> np.ndarray:
    """Design-rule verdict (recommended or not) for every grid condition of one CV.

    Args:
        cv (str): biased CV.
        gamma (float): biasing efficiency supplied to the rule.
        k0_per_s (float): unbiased rate supplied to the rule (s^-1).
        conds (pd.DataFrame): grid conditions with bf, height, pace and tau_rel_ps.
        pmf (dict): unbiased PMF summary (basin width and barrier per CV).
    """
    sa, barrier = pmf[PMF_KEY[cv]]["basin_sigma"], pmf[PMF_KEY[cv]]["barrier_kT"]
    out = []
    for r in conds.itertuples():
        p = predict(sa, SIGMA[cv], k0_per_s, gamma, r.bf, r.height, r.pace * DT_PS)
        out.append(verdict(p, gamma, barrier, r.pace * DT_PS, r.tau_rel_ps)["recommended"])
    return np.array(out)


def main() -> None:
    """Write the per-draw and summary tables and print the sanity check."""
    dr = pilot_draws()
    pmf = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    val = pd.read_csv(ROOT / "analysis/bias_design_validation.csv")
    agree, n_rec, acc, rec = [], [], [], []
    for r in dr.itertuples():
        v = val[val.cv == r.cv]
        ver = verdicts(r.cv, r.gamma, np.exp(r.lnk0_s), v, pmf)
        if r.pair == CHARGED_PAIR:
            rec.append(v.loc[ver, ["cv", "bf", "height", "pace"]].assign(draw=r.draw, pilot_cost_us=r.cost_us))
        agree.append(float((ver == v.recommended.to_numpy()).mean()))
        n_rec.append(int(ver.sum()))
        acc.append(float((v.eatr_err.abs().to_numpy()[ver] <= 1).mean()) if ver.any() else np.nan)
    dr["rule_agreement"], dr["rule_n_recommended"], dr["rule_eatr_within_1"] = agree, n_rec, acc
    dr.to_csv(ROOT / "analysis/pilot_draws.csv", index=False)
    # each charged pilot's own recommended conditions, for the error-cost front scored through the pilot
    pd.concat(rec).to_csv(ROOT / "analysis/pilot_recommended.csv", index=False)

    q = lambda s, x: float(np.nanquantile(s, x))  # noqa: E731
    summ = []
    for (cv, pair), g in dr.groupby(["cv", "pair"]):
        summ.append(dict(cv=cv, pair=pair, draws=len(g), charged=pair == CHARGED_PAIR,
                         cost_us_median=g.cost_us.median(), cost_us_p10=q(g.cost_us, 0.1), cost_us_p90=q(g.cost_us, 0.9),
                         gamma_median=g.gamma.median(), gamma_p10=q(g.gamma, 0.1), gamma_p90=q(g.gamma, 0.9),
                         err_median=g.err.median(), err_p10=q(g.err, 0.1), err_p90=q(g.err, 0.9),
                         gamma_ge_0p8=float((g.gamma >= 0.8).mean()), gamma_lt_0p5=float((g.gamma < 0.5).mean()),
                         rule_agreement_median=g.rule_agreement.median(), rule_agreement_p10=q(g.rule_agreement, 0.1),
                         rule_eatr_within_1_p10=q(g.rule_eatr_within_1, 0.1),
                         rule_n_recommended_median=g.rule_n_recommended.median(),
                         rule_eatr_within_1_median=g.rule_eatr_within_1.median(),
                         reference_n_recommended=int(val[val.cv == cv].recommended.sum())))
    summ = pd.DataFrame(summ)
    summ.to_csv(ROOT / "analysis/pilot_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    for cv in summ.cv.unique():
        t = pd.read_csv(dataset("runs.grid_summary") / f"opes_{cv}_trajectories.csv")
        mean_us = t[t.pace == 1000].groupby("barrier").t_final_ps.mean() / 1e6
        for pair in summ[summ.cv == cv].pair:
            a, b = (int(x) for x in pair.split("-"))
            got = float(summ[(summ.cv == cv) & (summ.pair == pair)].cost_us_median.iloc[0])
            print(f"sanity {cv} {pair}: median pilot cost {got:.2f} us vs 10 x mean escape {10 * (mean_us[a] + mean_us[b]):.2f} us")


if __name__ == "__main__":
    main()
