"""Register the predictions of the prospective test of the WT-MetaD design rule.

Usage: python scripts/prospective_predictions.py
Writes analysis/prospective_predictions.csv, which must exist (and be committed) before any
prospective run is analysed.

The rule may use only what a practitioner has before running: sigma_A and the apparent
barrier from the unbiased PMF, k0 from the reference, and a pilot gamma (the OPES
EATR-flooding gamma at pace 1000). Protocols are off the grid in bias factor, height and
pace, inside the design rule with margin from each of its three conditions. Registered per
protocol:
- predicted boost, escape time t* and fill V at escape (growth law, unbroadened overlap);
- the expected mean escape time: t* times the median observed/predicted ratio over the
  recommended grid conditions of the same CV (the rule's known calibration);
- EATR within one ln unit of the reference (both CVs), and iMetaD within one ln unit only
  if the pilot gamma is at least 0.8;
- the pooled RMSE of each estimator at the same run count on the recommended grid set
  (analysis/pareto_strategies.csv) as the expected scatter.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import paths as _paths
from paths import dataset
import pandas as pd

ROOT = _paths.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from bias_design_recipe import DT_PS, predict, verdict  # noqa: E402

# (CV, bias factor, height kJ/mol, pace steps, runs); must match setup_prospective_wtmetad.sh
PROTOCOLS = [("Q", 12, 0.35, 70000, 20), ("Ree", 8, 0.75, 70000, 40)]
SIGMA = {"Q": 0.02, "RMSD": 0.02, "Rg": 0.02, "Ree": 0.06}
PMF_KEY = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}
GRID = {"bf": (2, 5, 10, 15, 20), "height": (0.1, 0.25, 0.5, 1.0),
        "pace": (100, 500, 1000, 5000, 10000, 30000, 50000, 100000)}


def main() -> None:
    """Compute and write the registered predictions."""
    pmf = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    k0 = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["k_mle_per_us"] * 1e6
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    val = pd.read_csv(ROOT / "analysis/bias_design_validation.csv")
    grid = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    full = grid[(grid.n == grid.n_total) & (grid.estimator == "EATR CDF")][["cv", "bf", "height", "pace", "n", "cost_us"]]
    strat = pd.read_csv(ROOT / "analysis/pareto_strategies.csv")
    rows = []
    for cv, bf, h, pace, nruns in PROTOCOLS:
        assert bf not in GRID["bf"] and h not in GRID["height"] and pace not in GRID["pace"], (cv, bf, h, pace)
        gamma = float(mt[(mt.family == "opesf") & (mt.cv == cv) & (mt.estimator == "EATRf") & (mt.pace == 1000)].gamma.iloc[0])
        sa, barrier = pmf[PMF_KEY[cv]]["basin_sigma"], pmf[PMF_KEY[cv]]["barrier_kT"]
        p = predict(sa, SIGMA[cv], k0, gamma, bf, h, pace * DT_PS)
        tau_rel = float(val[val.cv == cv].tau_rel_ps.iloc[0])
        recommended = verdict(p, gamma, barrier, pace * DT_PS, tau_rel)["recommended"]
        rec = val[(val.cv == cv) & val.recommended].merge(full, on=["cv", "bf", "height", "pace"])
        ratio = float(((rec.cost_us / rec.n) * 1e3 / rec.pred_t_ns).median())
        s = strat[(strat.cv == cv) & (strat.protocol == "WT-MetaD (recommended set)") & (strat.n == nruns)]
        rmse = {e: float(s[s.estimator == e].rmse.iloc[0]) if (s.estimator == e).any() else float("nan") for e in ("EATR", "iMetaD")}
        rows.append(dict(cv=cv, bf=bf, height=h, pace=pace, n_runs=nruns, lammps_seed=f"1000+k, k = 1..{nruns}",
                         gamma_pilot=gamma, sigma_A=sa, barrier_kT=barrier, k0_per_s=k0,
                         pred_boost=p["boost"], pred_t_star_ns=p["t_star_ns"], pred_V_escape_kT=p["V_escape_kT"],
                         recommended=recommended, grid_obs_over_pred_escape=ratio,
                         expected_mean_escape_ns=p["t_star_ns"] * ratio,
                         expect_eatr_within_1=True, expect_imetad_within_1=gamma >= 0.8,
                         grid_rmse_eatr_same_n=rmse["EATR"], grid_rmse_imetad_same_n=rmse["iMetaD"],
                         registered=datetime.now().astimezone().isoformat(timespec="seconds")))
        assert recommended, (cv, bf, h, pace, p)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "analysis/prospective_predictions.csv", index=False)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
