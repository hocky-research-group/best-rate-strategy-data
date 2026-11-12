"""Compare the prospective WT-MetaD runs with their registered predictions.

Usage: python scripts/prospective_outcome.py
Input : analysis/prospective_predictions.csv, registered in commit REGISTERED_COMMIT before
        any prospective run existed (the script stops if the file differs from that
        commit); data/results/prospective_wtmetad/<CV>_bf<b>_height<h>_pace<p>.json and
        .cmd from scripts/prospective_fit.sbatch (eatr-analysis with the grid's options).
Output: analysis/prospective_outcome.csv, one row per protocol: observed boost and mean
        escape time against the predictions, EATR and iMetaD errors with 95% intervals, the
        fitted gamma against the pilot gamma, and whether each registered expectation held.
The escape-time check uses the draft's stated accuracy of the growth-law prediction, a
factor of two of t*; the ratio to the grid-calibrated expectation is reported alongside.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import paths as _paths
from paths import dataset
import pandas as pd

ROOT = _paths.ROOT
REGISTERED_COMMIT = "4f11947"
PREDICTIONS = "analysis/prospective_predictions.csv"
L6 = math.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
sys.path.insert(0, str(ROOT / "scripts"))
from bias_design_recipe import BOOST_MIN  # noqa: E402


def registered_predictions() -> pd.DataFrame:
    """The predictions table, checked against the version registered before the runs.

    The check is what makes the test prospective, so it is not skipped when it cannot be
    made: where the registered commit is at hand the table is compared against it, and
    elsewhere (the released archive without git, or a repository holding the release as one
    commit) against the copy of that commit's version shipped beside it. Neither present is
    an error rather than a pass.
    """
    registered = ROOT / "analysis/prospective_predictions_registered.csv"
    current = (ROOT / PREDICTIONS).read_text()
    try:
        show = subprocess.run(["git", "-C", str(ROOT), "show", f"{REGISTERED_COMMIT}:{PREDICTIONS}"],
                              capture_output=True, text=True)
    except FileNotFoundError:  # no git on this machine
        show = None
    if show is not None and show.returncode == 0:
        committed = show.stdout
        source = f"commit {REGISTERED_COMMIT}"
    elif registered.exists():
        committed = registered.read_text()
        source = registered.name
    else:
        raise SystemExit(f"cannot check {PREDICTIONS} against its registered version: "
                         f"no commit {REGISTERED_COMMIT} and no {registered.name}")
    if committed != current:
        raise SystemExit(f"{PREDICTIONS} differs from the version registered in {source}")
    return pd.read_csv(ROOT / PREDICTIONS)


def main() -> None:
    """Write the outcome table and print it."""
    rows = []
    for p in registered_predictions().itertuples():
        tag = f"{p.cv}_bf{p.bf}_height{p.height}_pace{p.pace}"
        folder = dataset("runs.prospective")
        fit_json, fit_cmd = folder / f"{tag}.json", folder / f"{tag}.cmd"
        if not fit_json.exists():
            raise SystemExit(f"missing prospective fit {fit_json}")
        d = json.loads(fit_json.read_text())
        cmd = fit_cmd.read_text()
        seeds, committed = (int(re.search(rf"{k}=(\d+)", cmd).group(1)) for k in ("seeds", "committed"))
        lnk_obs = d["ln(k_obs)"]
        obs_escape_ns = math.exp(-lnk_obs) * 1e9
        obs_boost = math.exp(lnk_obs) / p.k0_per_s
        e = d["EATR CDF ln k"] - L6 - REF
        e_lo, e_hi = (c - L6 - REF for c in d["EATR CDF ln k CI"])
        i = d["iMetaD CDF ln k"] - L6 - REF
        i_lo, i_hi = (c - L6 - REF for c in d["iMetaD CDF ln k CI"])
        g_lo, g_hi = d["EATR CDF gamma CI"]
        rows.append(dict(cv=p.cv, bf=p.bf, height=p.height, pace=p.pace, runs=seeds, committed=committed,
                         pred_boost=p.pred_boost, obs_boost=obs_boost,
                         obs_boost_ge_min=obs_boost >= BOOST_MIN,
                         pred_t_star_ns=p.pred_t_star_ns, expected_mean_escape_ns=p.expected_mean_escape_ns,
                         obs_mean_escape_ns=obs_escape_ns, obs_over_t_star=obs_escape_ns / p.pred_t_star_ns,
                         obs_over_expected=obs_escape_ns / p.expected_mean_escape_ns,
                         escape_within_factor_2_of_t_star=0.5 <= obs_escape_ns / p.pred_t_star_ns <= 2.0,
                         eatr_err=e, eatr_lo=e_lo, eatr_hi=e_hi, eatr_within_1=abs(e) <= 1,
                         eatr_expectation_held=(abs(e) <= 1) == bool(p.expect_eatr_within_1),
                         gamma=d["EATR CDF gamma"], gamma_lo=g_lo, gamma_hi=g_hi, gamma_pilot=p.gamma_pilot,
                         imetad_err=i, imetad_lo=i_lo, imetad_hi=i_hi, imetad_within_1=abs(i) <= 1,
                         imetad_expectation_held=(abs(i) <= 1) == bool(p.expect_imetad_within_1),
                         eatr_ks_p=d["EATR CDF p value"], imetad_ks_p=d["iMetaD CDF p value"]))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "analysis/prospective_outcome.csv", index=False)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
