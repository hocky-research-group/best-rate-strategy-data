"""Check the no-bootstrap flooding refits against the original bootstrap fits.

Usage: python scripts/check_flooding_refits.py   (exits 1 if a hard check fails)

Inputs: data/refits_full_data/task_*.jsonl.gz (scripts/refit_flooding_full_data.py) and the
original fits under data/ (flooding_nsim originals inside
data/subsampling/flooding_nsim/flooding_nsim.jsonl.gz).

Hard checks:
- every refit record has data (no failed fits) and every original has a refit;
- set labels and run totals of each refit equal the original's;
- OPES ladders of Triasha Pal: each set's ln k_obs equals ln(M / sum t) computed from
  data/results/trajectory_summary/opes_<CV>_trajectories.csv (independent of the package).
Diagnostic (printed, not enforced): the original set_reports are one bootstrap resample, so
(original - refit) / bootstrap sd of the same set should scatter as N(0, 1) for ln k_obs and
ln<e^bV>; a systematic offset in ln<e^bV> would mean the refit used different options (for
example the row stride).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
DATA = ROOT / "data"


def load_refits() -> tuple[dict, list[str]]:
    """Read every refit record.

    Returns:
        tuple[dict, list[str]]: data-relative path -> record for successful fits, and the
        paths of failed fits.
    """
    ok, failed = {}, []
    for pk in sorted((DATA / "refits_full_data").glob("task_*.jsonl.gz")):
        with gzip.open(pk, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if "data" in rec:
                    ok[rec["file"]] = rec
                else:
                    failed.append(rec["file"])
    return ok, [f for f in failed if f not in ok]


def load_originals() -> dict:
    """Read every original flooding fit that the refit covers.

    Returns:
        dict: data-relative path -> original fit JSON.
    """
    out = {}
    for pattern in ("triasha_results/prod_opes/bias*/opes-flooding-results-bootstrap/pace*_flooding.json",
                    "triasha_results/prod_ves_cutoff/bias*/nsim-sweep/combined_N*.json",
                    "triasha_results/prod_imetad/bias*/eatr-flooding-results-bootstrap/bf*_flooding.json",
                    "results/opes_highbarrier/*_pace1000_barriers5-17.json"):
        for p in sorted(DATA.glob(pattern)):
            out[str(p.relative_to(DATA))] = json.loads(p.read_text())
    with gzip.open(DATA / "subsampling/flooding_nsim/flooding_nsim.jsonl.gz", "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            out["subsampling/flooding_nsim/" + rec["file"]] = rec["data"]
    return out


def family(rel: str) -> str:
    """Short family name of a data-relative fit path.

    Args:
        rel (str): path relative to data/.
    """
    for key, name in (("prod_opes", "opes_triasha"), ("prod_ves_cutoff", "ves_triasha"),
                      ("prod_imetad", "pace_ladder"), ("opes_highbarrier", "opes7"),
                      ("flooding_nsim/opes", "nsim_opes"), ("flooding_nsim/ves", "nsim_ves")):
        if key in rel:
            return name
    return "other"


def main() -> int:
    """Run the checks and print a per-family summary; return the exit status."""
    refits, failed = load_refits()
    originals = load_originals()
    problems = [f"no refit for original (fit failed): {f}" for f in failed if f in originals]
    problems += [f"no refit for original: {r}" for r in sorted(set(originals) - set(refits) - set(failed))]
    # Ladders whose original fit never existed (the original job failed) are not used downstream.
    unused = sorted((set(refits) | set(failed)) - set(originals))
    rows = []
    for rel in sorted(set(refits) & set(originals)):
        old, new = originals[rel], refits[rel]["data"]
        so, sn = old["set_reports"], new["set_reports"]
        if [s["barrier"] for s in so] != [s["barrier"] for s in sn] or [s["total"] for s in so] != [s["total"] for s in sn]:
            problems.append(f"set labels or totals differ: {rel}")
            continue
        sd_k = old.get("bootstrap_per_set_log_k_obs_std") or [np.nan] * len(so)
        sd_v = old.get("bootstrap_per_set_ln_exp_beta_v_std") or [np.nan] * len(so)
        for j, (a, b) in enumerate(zip(so, sn)):
            rows.append(dict(family=family(rel), file=rel, barrier=b["barrier"], total=b["total"],
                             transitioned=b["transitioned"], log_k_obs=b["log_k_obs"],
                             d_lnk=a["log_k_obs"] - b["log_k_obs"], d_lnv=a["ln_exp_beta_v"] - b["ln_exp_beta_v"],
                             sd_lnk=sd_k[j], sd_lnv=sd_v[j]))
    r = pd.DataFrame(rows)
    r["z_lnk"], r["z_lnv"] = r.d_lnk / r.sd_lnk, r.d_lnv / r.sd_lnv

    # Independent check of the refit rates: ln(M / sum t) from the trajectory summaries.
    n_exact = 0
    for cv in ("Q", "RMSD", "Ree", "Rg"):
        ts = pd.read_csv(DATA / f"results/trajectory_summary/opes_{cv}_trajectories.csv")
        sub = r[(r.family == "opes_triasha") & r.file.str.contains(f"/bias{cv}/")]
        for _, row in sub.iterrows():
            pace = int(row.file.split("pace")[-1].split("_")[0])
            g = ts[(ts.pace == pace) & (ts.barrier == row.barrier)]
            ref = np.log(g.committed.sum() / (g.t_final_ps.sum() * 1e-12))
            if len(g) != row.total or int(g.committed.sum()) != row.transitioned or abs(ref - row.log_k_obs) > 1e-9:
                problems.append(f"OPES rate differs from trajectory summary: {row.file} barrier {row.barrier}")
            else:
                n_exact += 1

    print(f"refits: {len(refits)} ok, {len(failed)} failed; originals: {len(originals)}; "
          f"refit ladders without an original (unused): {len(unused)}"
          f" ({sum(u in failed for u in unused)} of them failed)")
    print(f"OPES (Triasha) sets whose refit ln k_obs equals ln(M/sum t) of the trajectory summaries: "
          f"{n_exact} of {int((r.family == 'opes_triasha').sum())}")
    print("per family: sets, median |orig - refit| (ln k_obs, ln<e^bV>), z mean/sd (ln k_obs), z mean/sd (ln<e^bV>)")
    for fam, gr in r.groupby("family"):
        print(f"  {fam:13s} {len(gr):5d}  {gr.d_lnk.abs().median():.3f} {gr.d_lnv.abs().median():.3f}  "
              f"{gr.z_lnk.mean():+.2f}/{gr.z_lnk.std():.2f}  {gr.z_lnv.mean():+.2f}/{gr.z_lnv.std():.2f}")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
