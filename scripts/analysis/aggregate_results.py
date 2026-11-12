"""Aggregate every rate-estimate JSON into one tidy table.

Usage:
    uv run python scripts/aggregate_results.py

Inputs : data/triasha_results/** (JSON written by eatr-analysis and
         eatr-flooding-analysis), data/reference_rate/reference_rate.json
Outputs: analysis/master_table.csv  (one row per estimate)
         analysis/master_sets.csv   (one row per simulation set inside a
                                     flooding fit: cost and observed rate)

Columns of master_table.csv
    family     imetad | eatrf_imetad | opesf | vesf
    estimator  iMetaD CDF | EATR CDF | EATRf | OPESf
    cv, bf, height, pace, barrier_label, n_sets, source
    lnk_s      ln k0 with k0 in s^-1;  lnk_us = lnk_s - ln(1e6)
    lnk_lo, lnk_hi   bootstrap interval as stored in the JSON
    gamma, gamma_lo, gamma_hi, ks_p, converged
    n_total, n_events, cost_us  (total simulated time of the data behind the
                                 estimate, microseconds; see notes)
    err_ln     lnk_us - reference ln k (MLE, us^-1)
    valid      False where the fit is known to be wrong (iMetaD with the wrong
               acceleration column) so it can be filtered without deleting it

Cost notes. For single-condition fits the EATR CDF plot block stores the sorted
observed transition times of the n_events transitioned runs; cost is their sum.
When n_events < n_total the censored runs' lengths are not in the JSON, so the
cost is a lower bound and `cost_complete` is False. For flooding fits each set
reports tau_obs (mean observed transition time) and counts; cost is the sum over
sets of tau_obs * transitioned, with the same caveat.

Validity rule for iMetaD (acceleration column): bootstrap-1 used column 8 which
is metad.acc for Q, RMSD, Ree; bootstrap-2 used column 6 which is metad.acc only
for Rg. EATR uses the bias column and is valid in both.
"""
import json
import math
import re
from pathlib import Path

import paths as _paths
from paths import dataset
import pandas as pd

from flooding_io import full_data_view

ROOT = _paths.ROOT
DATA = dataset("source.triasha")
OUT = ROOT / "analysis"
LN_S_TO_US = math.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())
LNK_REF_US = REF["ln_k_mle_per_us"]

PAT_COND = re.compile(r"bf([\d.]+)_height([\d.]+)_pace(\d+)\.json$")
PAT_FLOOD = re.compile(r"bf([\d.]+)_height([\d.]+)_flooding\.json$")
PAT_OPES = re.compile(r"pace(\d+)_flooding\.json$")
PAT_VES = re.compile(r"combined_N(\d+)\.json$")


def opesf_ln_k0(d: dict) -> float | None:
    """OPESf ln k0 from a flooding JSON, correcting the halving defect of
    EATR-rate-analysis versions before the 2026-09-10 fix. Those versions averaged
    ln k0 with a converged flag, so the stored value is ln k0 / 2 and the bootstrap
    interval starts at exactly 0; that signature selects the correction.

    Args:
        d (dict): parsed flooding JSON.
    """
    v = d.get("opes_logk0")
    if v is None:
        return None
    ci = d.get("bootstrap_opes_logk0_ci")
    halved = ci is not None and len(ci) == 2 and abs(ci[0]) < 1e-12 and ci[1] > 0
    return 2.0 * v if halved else float(v)


def cost_from_times(block: dict) -> tuple[float, bool]:
    """Total simulated time (us) from an eatr-analysis CDF plot block.

    Args:
        block (dict): the "EATR CDF plot" dictionary (times in seconds).
    Returns (cost_us, complete) where complete is False if runs were censored.
    """
    times = block.get("time") or []
    n_total, n_events = block.get("n_total"), block.get("n_events")
    return sum(times) * 1e6, (n_events == n_total)


def single_condition_rows(path: Path, cv: str, source: str) -> list[dict]:
    """Rows for one eatr-analysis JSON (iMetaD CDF and EATR CDF estimates).

    Args:
        path (Path): JSON file named bf*_height*_pace*.json.
        cv (str): biased collective variable.
        source (str): bootstrap-1 or bootstrap-2.
    """
    m = PAT_COND.search(path.name)
    if not m:
        return []
    bf, height, pace = float(m.group(1)), float(m.group(2)), int(m.group(3))
    d = json.loads(path.read_text())
    eplot = d.get("EATR CDF plot", {})
    cost_us, complete = cost_from_times(eplot)
    times = eplot.get("time") or []
    mean_t = sum(times) / len(times) if times else float("nan")
    cv_t = (pd.Series(times).std() / mean_t) if len(times) > 1 else float("nan")
    rows = []
    for est in ("iMetaD CDF", "EATR CDF"):
        if f"{est} ln k" not in d:
            continue
        ci = d.get(f"{est} ln k CI") or [float("nan")] * 2
        gci = d.get(f"{est} gamma CI") or [float("nan")] * 2
        lnk = d[f"{est} ln k"]
        if est == "iMetaD CDF":
            valid = (source == "bootstrap-1" and cv != "Rg") or (source == "bootstrap-2" and cv == "Rg")
        else:
            valid = True
        rows.append(dict(
            family="imetad", estimator=est, cv=cv, bf=bf, height=height, pace=pace,
            barrier_label=float("nan"), n_sets=1, source=source,
            lnk_s=lnk, lnk_lo=ci[0], lnk_hi=ci[1],
            gamma=d.get(f"{est} gamma", float("nan")), gamma_lo=gci[0], gamma_hi=gci[1],
            ks_p=d.get(f"{est} p value", float("nan")), converged=d.get(f"{est} converged"),
            n_total=eplot.get("n_total"), n_events=eplot.get("n_events"),
            cost_us=cost_us, cost_complete=complete, mean_time_us=mean_t * 1e6, cv_times=cv_t,
            lnk_obs_s=d.get("ln(k_obs)", float("nan")),
            err_ln=lnk - LN_S_TO_US - LNK_REF_US, valid=valid, file=str(path.relative_to(ROOT)),
        ))
    return rows


def flooding_rows(path: Path, cv: str, family: str, source: str) -> tuple[list[dict], list[dict]]:
    """Rows for one eatr-flooding-analysis JSON (EATRf and, if present, OPESf).

    Args:
        path (Path): flooding JSON.
        cv (str): biased collective variable.
        family (str): eatrf_imetad | opesf | vesf.
        source (str): results directory name.
    Returns (estimate rows, per-set rows).
    """
    d = full_data_view(json.loads(path.read_text()), str(path.relative_to(ROOT / "data")))
    bf = height = pace = float("nan")
    label = float("nan")
    if family == "eatrf_imetad":
        m = PAT_FLOOD.search(path.name); bf, height = float(m.group(1)), float(m.group(2))
    elif family == "opesf":
        m = PAT_OPES.search(path.name); pace = int(m.group(1))
    elif family == "vesf":
        m = PAT_VES.search(path.name); label = int(m.group(1)) if m else float("nan")
    sets = d.get("set_reports") or []
    cost_us = sum((s.get("tau_obs") or 0) * (s.get("transitioned") or 0) for s in sets) * 1e6
    complete = all(s.get("transitioned") == s.get("total") for s in sets)
    n_total = sum(s.get("total") or 0 for s in sets)
    n_events = sum(s.get("transitioned") or 0 for s in sets)
    diag = d.get("flooding_diagnostics") or {}
    common = dict(family=family, cv=cv, bf=bf, height=height, pace=pace, barrier_label=label,
                  set_reports_source=d.get("_set_reports_source"),
                  n_sets=len(sets), source=source, n_total=n_total, n_events=n_events,
                  cost_us=cost_us, cost_complete=complete, mean_time_us=float("nan"),
                  cv_times=float("nan"), lnk_obs_s=float("nan"), file=str(path.relative_to(ROOT)),
                  selected_nsets=diag.get("convergence_analysis", {}).get("selected_nsets"))
    rows = []
    ci = d.get("bootstrap_logk0_ci") or [float("nan")] * 2
    gci = d.get("bootstrap_gamma_ci") or [float("nan")] * 2
    rows.append(dict(common, estimator="EATRf", lnk_s=d["logk0"], lnk_lo=ci[0], lnk_hi=ci[1],
                     gamma=d.get("gamma"), gamma_lo=gci[0], gamma_hi=gci[1], ks_p=float("nan"),
                     converged=None, err_ln=d["logk0"] - LN_S_TO_US - LNK_REF_US, valid=True))
    if d.get("opes_logk0") is not None:
        opes_lnk = opesf_ln_k0(d)  # corrects the pre-fix halving by its signature
        rows.append(dict(common, estimator="OPESf", lnk_s=opes_lnk, lnk_lo=float("nan"), lnk_hi=float("nan"),
                         gamma=1.0, gamma_lo=float("nan"), gamma_hi=float("nan"), ks_p=float("nan"),
                         converged=None, err_ln=opes_lnk - LN_S_TO_US - LNK_REF_US, valid=True))
        # per-set flooding estimate at gamma = 1 (ensemble-first exponential average)
        grid = diag.get("gamma_grid") or []
        per = diag.get("per_set_ln_k0") or []
        if grid and per and abs(grid[-1] - 1.0) < 1e-9:
            vals = per[-1]
            g1 = sum(vals) / len(vals)
            rows.append(dict(common, estimator="Flooding gamma=1", lnk_s=g1, lnk_lo=min(vals), lnk_hi=max(vals),
                             gamma=1.0, gamma_lo=float("nan"), gamma_hi=float("nan"), ks_p=float("nan"),
                             converged=None, err_ln=g1 - LN_S_TO_US - LNK_REF_US, valid=True))
    set_rows = [dict(family=family, cv=cv, bf=bf, height=height, pace=pace, label=label, source=source,
                     set_label=s.get("barrier"), transitioned=s.get("transitioned"), total=s.get("total"),
                     tau_obs_us=(s.get("tau_obs") or float("nan")) * 1e6, log_k_obs_s=s.get("log_k_obs"),
                     ln_exp_beta_v=s.get("ln_exp_beta_v"), avg_max_bias=s.get("avg_max_bias"),
                     ks_p=s.get("p_value"), file=str(path.relative_to(ROOT))) for s in sets]
    return rows, set_rows


def main() -> None:
    """Walk the results tree and write the two CSV tables."""
    rows, set_rows = [], []
    for cvdir in sorted(DATA.glob("prod_imetad/bias*")):
        cv = cvdir.name[4:]
        for source in ("eatr_results_bootstrap-1", "eatr_results_bootstrap-2"):
            for f in sorted((cvdir / source).glob("*.json")):
                rows += single_condition_rows(f, cv, source.replace("eatr_results_", ""))
        for f in sorted((cvdir / "eatr-flooding-results-bootstrap").glob("*.json")):
            r, s = flooding_rows(f, cv, "eatrf_imetad", "eatr-flooding-results-bootstrap"); rows += r; set_rows += s
    for cvdir in sorted(DATA.glob("prod_opes/bias*")):
        cv = cvdir.name[4:]
        for f in sorted((cvdir / "opes-flooding-results-bootstrap").glob("*.json")):
            r, s = flooding_rows(f, cv, "opesf", "opes-flooding-results-bootstrap"); rows += r; set_rows += s
    for cvdir in sorted(DATA.glob("prod_ves_cutoff/bias*")):
        cv = cvdir.name[4:]
        for f in sorted((cvdir / "nsim-sweep").glob("combined_N*.json")):
            r, s = flooding_rows(f, cv, "vesf", "nsim-sweep"); rows += r; set_rows += s
    df = pd.DataFrame(rows)
    df["lnk_us"] = df["lnk_s"] - LN_S_TO_US
    df["abs_err_ln"] = df["err_ln"].abs()
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "master_table.csv", index=False)
    pd.DataFrame(set_rows).to_csv(OUT / "master_sets.csv", index=False)
    print(f"reference ln k (us^-1) = {LNK_REF_US:.3f}")
    print(df.groupby(["family", "estimator", "cv"]).size().to_string())
    ok = df[df.valid]
    summ = ok.groupby(["family", "estimator", "cv"]).agg(
        n=("abs_err_ln", "size"), median_abs_err=("abs_err_ln", "median"),
        frac_within_1=("abs_err_ln", lambda x: (x <= 1).mean()),
        frac_within_0p5=("abs_err_ln", lambda x: (x <= 0.5).mean()),
        median_cost_us=("cost_us", "median"))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
