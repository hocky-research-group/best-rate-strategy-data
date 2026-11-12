"""Aggregate trajectory-count subsampling fits into per-(condition, N) tables.

Usage:
    uv run python scripts/aggregate_subsampling.py

Inputs
    data/subsampling/imetad_grid/results/*.json   eatr-analysis output with a
        "subsample" block (sizes x replicates) plus the full-set fit
    data/subsampling/imetad_grid_intermediate_sizes/   further subset sizes of the same
        conditions (scripts/intermediate_subset_conditions.py); only the "subsample" block is read
    data/subsampling/untempered/results/*.json   the same for untempered metadynamics
    data/subsampling/flooding_subsets/task_*.jsonl.gz
        no-bootstrap eatr-flooding-analysis fits of N runs per bias set
        (scripts/flooding_subsets.py); failed fits are skipped with a warning
    data/reference_rate/reference_rate.json
Outputs
    analysis/subsampling_imetad.csv   one row per (cv, bf, height, pace, estimator, n)
    analysis/subsampling_untempered.csv  the same for untempered metadynamics (bf = inf),
        written only when data/subsampling/untempered holds fits
    analysis/subsampling_flooding.csv one row per (family, cv, pace, estimator, n)
    analysis/minimum_data_summary.csv smallest cost at a 90% hit rate per method, with the
        pooled subset draws and the Wilson 95% interval of the hit rate at that size
    columns: n_fits, mean_err, sd (spread of ln k across independent subsets),
    rmse, frac_within_1, frac_within_0p5, cost_us (mean total simulated time of
    the subsets; n x mean run length when per-run lengths are unavailable).

OPESf values from pre-fix versions of the flooding tool (interval starting at 0)
are doubled; see opesf_ln_k0.
"""
import gzip
import json
import math
import re
import warnings
from pathlib import Path

from paths import dataset
import numpy as np
import pandas as pd

from flooding_io import full_data_view

import paths as _paths
ROOT = _paths.ROOT
L6 = math.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
PAT = re.compile(r"^(\w+)_bf([\d.]+)_height([\d.]+)_pace(\d+)\.json$")
PAT_UT = re.compile(r"^(\w+)_untempered_height([\d.]+)_pace(\d+)\.json$")
FA_BF, FA_HEIGHT, FA_TAU0 = 10, 0.5, 100   # fixed across the theta scan
PAT_FA = re.compile(r"^(\w+)_fametad_theta([\d.]+)\.json$")


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


def iter_results(root: Path, suffix: str = ".json"):
    """Yield (relative name, parsed JSON) from packed *.jsonl.gz files under root,
    falling back to loose JSON files when no archive is present.

    Args:
        root (Path): results directory (local copy of the torch tree).
        suffix (str): file suffix to select among loose files.
    """
    packed = sorted(root.rglob("*.jsonl.gz"))
    if packed:
        for pk in packed:
            with gzip.open(pk, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if "data" not in rec:
                        warnings.warn(f"{pk.name}: skipping failed fit {rec.get('file')}", stacklevel=2)
                        continue
                    yield str(pk.parent.relative_to(root) / rec["file"]).replace("./", ""), rec["data"]
        return
    for f in sorted(root.rglob("*" + suffix)):
        try:
            yield str(f.relative_to(root)), json.loads(f.read_text())
        except json.JSONDecodeError:
            continue


def stats(errs: np.ndarray, costs: np.ndarray) -> dict:
    """Summary statistics of signed errors over independent subsets.

    Args:
        errs (np.ndarray): ln k_est - ln k_ref per subset.
        costs (np.ndarray): total simulated time (us) per subset.
    """
    errs = errs[np.isfinite(errs)]
    return dict(n_fits=len(errs), mean_err=errs.mean() if len(errs) else np.nan,
                sd=errs.std(ddof=1) if len(errs) > 1 else np.nan,
                rmse=np.sqrt((errs ** 2).mean()) if len(errs) else np.nan,
                frac_within_1=(np.abs(errs) <= 1).mean() if len(errs) else np.nan,
                frac_within_0p5=(np.abs(errs) <= 0.5).mean() if len(errs) else np.nan,
                frac_within_log10=(np.abs(errs) <= math.log(10)).mean() if len(errs) else np.nan,
                cost_us=float(np.nanmean(costs)))


def grid_condition(rel: str, d: dict) -> tuple[dict, np.ndarray] | None:
    """Condition of one WT-MetaD grid fit and its escape times.

    Args:
        rel (str): file name of the fit, <CV>_bf<b>_height<h>_pace<p>.json.
        d (dict): parsed eatr-analysis JSON.
    Returns (dict of cv, bf, height, pace, n_total; escape times in us), or None when the
    name is not a grid condition.
    """
    m = PAT.match(Path(rel).name)
    if not m:
        return None
    base = dict(cv=m.group(1), bf=float(m.group(2)), height=float(m.group(3)), pace=int(m.group(4)),
                n_total=d["EATR CDF plot"]["n_total"])
    return base, np.array(d["EATR CDF plot"]["time"]) * 1e6


def subset_rows(base: dict, sub: dict, mean_t: float) -> list[dict]:
    """Rows of one condition's subset fits, one per (subset size, estimator).

    Args:
        base (dict): condition columns from grid_condition.
        sub (dict): the "subsample" block of an eatr-analysis JSON.
        mean_t (float): mean escape time of the condition's runs (us), which prices a subset.
    """
    rows = []
    res = pd.DataFrame(sub["results"])
    for n, g in res.groupby("n"):
        for est in ("iMetaD CDF", "EATR CDF"):
            errs = g[f"{est} ln k"].to_numpy(dtype=float) - L6 - REF
            st = stats(errs, np.full(len(errs), n * mean_t))
            rows.append(dict(base, estimator=est, n=int(n), **st,
                             gamma=g["EATR CDF gamma"].mean() if est == "EATR CDF" else 1.0,
                             ks_p=g[f"{est} p value"].mean()))
    return rows


def imetad_rows() -> pd.DataFrame:
    """Rows for the WT-MetaD grid subsampling fits.

    The full-set fit and the first subset sizes of a condition come from fits.grid. Sizes
    fitted afterwards live in fits.grid_intermediate, whose full-set fits are not the
    bootstrap means of fits.grid and are not read; a size present in both raises.
    """
    rows, sizes = [], {}
    for rel, d in iter_results(dataset("fits.grid")):
        cond = grid_condition(rel, d)
        if cond is None:
            continue
        base, times = cond
        cv, bf, height, pace, n_total = (base[k] for k in ("cv", "bf", "height", "pace", "n_total"))
        mean_t = times.mean()
        for est in ("iMetaD CDF", "EATR CDF"):
            err = d[f"{est} ln k"] - L6 - REF
            rows.append(dict(base, estimator=est, n=n_total, n_fits=1, mean_err=err, sd=np.nan, rmse=abs(err),
                             frac_within_1=float(abs(err) <= 1), frac_within_0p5=float(abs(err) <= 0.5),
                             frac_within_log10=float(abs(err) <= math.log(10)), cost_us=times.sum(), gamma=d.get("EATR CDF gamma") if est == "EATR CDF" else 1.0,
                             ks_p=d.get(f"{est} p value")))
        sub = d.get("subsample")
        if not sub:
            continue
        new = subset_rows(base, sub, mean_t)
        sizes[(cv, bf, height, pace)] = {r["n"] for r in new}
        rows += new
    try:
        extra = dataset("fits.grid_intermediate")
    except FileNotFoundError:
        return pd.DataFrame(rows)
    for rel, d in iter_results(extra):
        cond = grid_condition(rel, d)
        if cond is None or not d.get("subsample"):
            continue
        base, times = cond
        new = subset_rows(base, d["subsample"], times.mean())
        both = sizes.get((base["cv"], base["bf"], base["height"], base["pace"]), set()) & {r["n"] for r in new}
        if both:
            raise ValueError(f"{rel}: subset sizes {sorted(both)} are in both fits.grid and fits.grid_intermediate")
        rows += new
    return pd.DataFrame(rows)


def untempered_rows() -> pd.DataFrame:
    """Rows for the untempered metadynamics conditions (METAD without BIASFACTOR).

    Same columns as imetad_rows, with bf = inf: untempered metadynamics is the
    b -> infinity limit of the well-tempered family.
    """
    rows = []
    for rel, d in iter_results(dataset("fits.untempered")):
        m = PAT_UT.match(Path(rel).name)
        if not m:
            continue
        cv, height, pace = m.group(1), float(m.group(2)), int(m.group(3))
        times = np.array(d["EATR CDF plot"]["time"]) * 1e6
        n_total = d["EATR CDF plot"]["n_total"]
        mean_t = times.mean()
        base = dict(cv=cv, bf=np.inf, height=height, pace=pace, n_total=n_total)
        for est in ("iMetaD CDF", "EATR CDF"):
            err = d[f"{est} ln k"] - L6 - REF
            rows.append(dict(base, estimator=est, n=n_total, n_fits=1, mean_err=err, sd=np.nan, rmse=abs(err),
                             frac_within_1=float(abs(err) <= 1), frac_within_0p5=float(abs(err) <= 0.5),
                             frac_within_log10=float(abs(err) <= math.log(10)), cost_us=times.sum(),
                             gamma=d.get("EATR CDF gamma") if est == "EATR CDF" else 1.0,
                             ks_p=d.get(f"{est} p value")))
        sub = d.get("subsample")
        if not sub:
            continue
        res = pd.DataFrame(sub["results"])
        for n, g in res.groupby("n"):
            for est in ("iMetaD CDF", "EATR CDF"):
                errs = g[f"{est} ln k"].to_numpy(dtype=float) - L6 - REF
                st = stats(errs, np.full(len(errs), n * mean_t))
                rows.append(dict(base, estimator=est, n=int(n), **st,
                                 gamma=g["EATR CDF gamma"].mean() if est == "EATR CDF" else 1.0,
                                 ks_p=g[f"{est} p value"].mean()))
    return pd.DataFrame(rows)


def fametad_rows() -> pd.DataFrame:
    """Rows for the frequency-adaptive conditions (METAD with FREQUENCY_ADAPTIVE).

    Same columns as imetad_rows. The deposition interval is not a constant here, so the
    condition is labelled by its acceleration threshold theta and pace records the interval
    the run started from, which is the fast endpoint of the scan.
    """
    rows = []
    for rel, d in iter_results(dataset("fits.frequency_adaptive")):
        m = PAT_FA.match(Path(rel).name)
        if not m:
            continue
        cv, theta = m.group(1), float(m.group(2))
        times = np.array(d["EATR CDF plot"]["time"]) * 1e6
        n_total = d["EATR CDF plot"]["n_total"]
        mean_t = times.mean()
        base = dict(cv=cv, bf=FA_BF, height=FA_HEIGHT, pace=FA_TAU0, theta=theta, n_total=n_total)
        for est in ("iMetaD CDF", "EATR CDF"):
            err = d[f"{est} ln k"] - L6 - REF
            rows.append(dict(base, estimator=est, n=n_total, n_fits=1, mean_err=err, sd=np.nan,
                             rmse=abs(err), frac_within_1=float(abs(err) <= 1),
                             frac_within_0p5=float(abs(err) <= 0.5),
                             frac_within_log10=float(abs(err) <= math.log(10)), cost_us=times.sum(),
                             gamma=d.get("EATR CDF gamma") if est == "EATR CDF" else 1.0,
                             ks_p=d.get(f"{est} p value")))
        sub = d.get("subsample")
        if not sub:
            continue
        res = pd.DataFrame(sub["results"])
        for n, g in res.groupby("n"):
            for est in ("iMetaD CDF", "EATR CDF"):
                errs = g[f"{est} ln k"].to_numpy(dtype=float) - L6 - REF
                st = stats(errs, np.full(len(errs), n * mean_t))
                rows.append(dict(base, estimator=est, n=int(n), **st,
                                 gamma=g["EATR CDF gamma"].mean() if est == "EATR CDF" else 1.0,
                                 ks_p=g[f"{est} p value"].mean()))
    return pd.DataFrame(rows)


def flooding_rows() -> pd.DataFrame:
    """Rows for the OPES and VES flooding N-per-set sweeps."""
    recs = []
    for rel, d in iter_results(dataset("fits.flooding_subsets")):
        d = full_data_view(d, "subsampling/flooding_subsets/" + rel)
        parts = Path(rel).parts
        m = re.match(r"N(\d+)_rep(\d+)\.json", parts[-1])
        if not m or len(parts) < 3:
            continue
        fam, cond = parts[-3], parts[-2]
        cv, pace = (cond.split("_pace") + [None])[:2]
        n, rep = int(m.group(1)), int(m.group(2))
        sets = d.get("set_reports") or []
        cost = sum((s.get("tau_obs") or 0) * (s.get("transitioned") or 0) for s in sets) * 1e6
        recs.append(dict(family=fam, cv=cv, pace=int(pace) if pace else np.nan, n=n, rep=rep,
                         estimator="EATRf", err=d["logk0"] - L6 - REF, gamma=d["gamma"], cost_us=cost))
        if d.get("opes_logk0") is not None:
            recs.append(dict(family=fam, cv=cv, pace=int(pace) if pace else np.nan, n=n, rep=rep,
                             estimator="OPESf", err=opesf_ln_k0(d) - L6 - REF, gamma=1.0, cost_us=cost))
    # overbiasing-screened and convergence-selected variants from flooding_set_selection.py;
    # choosing the retained sets requires simulating every set, so the cost is that of all sets
    sel = ROOT / "analysis/flooding_set_selection.csv"
    if sel.exists():
        ss = pd.read_csv(sel)
        ss = ss[ss.source == "subsets"]
        for _, r in ss.iterrows():
            for est, ecol in (("EATRf screened", "err_screen"), ("EATRf conv", "err_conv")):
                recs.append(dict(family=r.family, cv=r.cv, pace=r.pace, n=int(r.n_per_set), rep=int(r.rep),
                                 estimator=est, err=r[ecol], gamma=np.nan, cost_us=r.cost_all))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    rows = []
    for key, g in df.groupby(["family", "cv", "pace", "estimator", "n"], dropna=False):
        st = stats(g["err"].to_numpy(), g["cost_us"].to_numpy())
        rows.append(dict(zip(["family", "cv", "pace", "estimator", "n"], key), **st, gamma=g["gamma"].mean()))
    return pd.DataFrame(rows)


def wilson_interval(hits: float, draws: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion.

    Args:
        hits (float): subsets inside the criterion.
        draws (float): subsets drawn.
        z (float): standard-normal quantile of the interval (1.96 for 95%).
    """
    if not draws > 0:
        return float("nan"), float("nan")
    p = hits / draws
    den = 1 + z * z / draws
    centre = (p + z * z / (2 * draws)) / den
    half = z * math.sqrt(p * (1 - p) / draws + z * z / (4 * draws * draws)) / den
    return centre - half, centre + half


def minimum_cost(df: pd.DataFrame, keys: list[str], target: float = 0.9) -> pd.DataFrame:
    """Smallest cost at which the hit rate reaches ``target`` for each group.

    Args:
        df (pd.DataFrame): per-(condition, n) table with cost_us, n_fits and the frac_* columns.
        keys (list[str]): grouping columns identifying a method.
        target (float): required fraction of subsets inside the criterion.
    Returns one row per group with the minimum cost (us) and the subset size n
    for each of the three criteria, requiring the hit rate to hold at every
    larger size as well (NaN when the target is never reached that way), and at
    that size the number of subset draws pooled over the group's conditions with
    the Wilson 95% interval of their hit rate. max_hit_rate and hit_rate_at_largest_n are the
    largest hit rate over sizes and the hit rate at the largest size, for the one-ln-unit
    criterion.
    """
    out = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for crit in ("frac_within_1", "frac_within_0p5", "frac_within_log10"):
            a = (g.assign(_hits=g[crit] * g["n_fits"])
                 .groupby("n").agg(cost=("cost_us", "median"), p=(crit, "mean"), hits=("_hits", "sum"), draws=("n_fits", "sum"))
                 .reset_index().sort_values("n"))
            # the hit rate must reach the target and stay there for every larger size:
            # a biased estimator can pass at small n by spread alone and fail as the
            # subsets converge onto the biased value
            ok = [bool((a.p.iloc[i:] >= target).all()) for i in range(len(a))]
            hit = a[ok].sort_values("cost")
            row[f"min_cost_us[{crit}]"] = hit.cost.iloc[0] if len(hit) else np.nan
            row[f"n_at_min[{crit}]"] = int(hit.n.iloc[0]) if len(hit) else -1
            draws = float(hit.draws.iloc[0]) if len(hit) else float("nan")
            lo, hi = wilson_interval(float(hit.hits.iloc[0]), draws) if len(hit) else (float("nan"), float("nan"))
            row[f"draws_at_min[{crit}]"] = draws
            row[f"hit_rate_lo95[{crit}]"] = lo
            row[f"hit_rate_hi95[{crit}]"] = hi
            if crit == "frac_within_1":
                row["max_hit_rate"] = a.p.max()
                row["hit_rate_at_largest_n"] = a.p.iloc[-1]
        out.append(row)
    return pd.DataFrame(out)


def wtmetad_random_subsets(im: pd.DataFrame) -> pd.DataFrame:
    """Rows of the WT-MetaD table that enter the pooled minimum-data criterion.

    Args:
        im (pd.DataFrame): per-(condition, n) table from imetad_rows().
    Returns the random-subset rows (n < n_total) of bias factors 2 and above. Conditions hold
    different numbers of runs, so a full-set fit (n = n_total, one fit) is not a subset size
    shared across conditions: left in, the single fits of the 50-run conditions would stand
    as "size 50" between the pooled sizes 40 and 60. Bias factor 1 is degenerate in PLUMED
    (hill height h0 exp(-V/((b-1)kT)) vanishes after the first hill) and is left out.
    """
    return im[(im.bf >= 2) & (im.n < im.n_total)]


def main() -> None:
    """Write both tables and print a compact view."""
    im = imetad_rows()
    im.to_csv(ROOT / "analysis/subsampling_imetad.csv", index=False)
    fa = fametad_rows()
    if len(fa):
        fa.to_csv(ROOT / "analysis/subsampling_fametad.csv", index=False)
        print(f"fametad: {len(fa)} rows, "
              f"{fa[['cv', 'theta']].drop_duplicates().shape[0]} conditions")
    ut = untempered_rows()
    if len(ut):
        ut.to_csv(ROOT / "analysis/subsampling_untempered.csv", index=False)
        print(f"untempered: {ut.n_total.count()} rows, "
              f"{ut[['cv', 'height', 'pace']].drop_duplicates().shape[0]} conditions")
    fl = flooding_rows()
    fl.to_csv(ROOT / "analysis/subsampling_flooding.csv", index=False)
    parts = []
    if len(im):
        wt = wtmetad_random_subsets(im)
        a = minimum_cost(wt, ["cv", "estimator", "pace"]); a.insert(0, "family", "wtmetad_by_pace"); parts.append(a)
        slow = wt[wt.pace >= 10000]
        a = minimum_cost(slow, ["cv", "estimator"]); a.insert(0, "family", "wtmetad_pace>=10000"); parts.append(a)
    if len(fl):
        a = minimum_cost(fl, ["family", "cv", "estimator"]); parts.append(a)
    if parts:
        summ = pd.concat(parts, ignore_index=True)
        summ.to_csv(ROOT / "analysis/minimum_data_summary.csv", index=False)
        print("Minimum cost (us) to reach 90% of subsets within 1 ln unit / 0.5 / one log10:")
        print(summ.round(2).to_string())
    print(f"reference ln k (us^-1) = {REF:.3f}")
    if len(im):
        v = im[im.n < im.n_total].groupby(["cv", "estimator", "n"]).agg(
            conds=("cost_us", "size"), sd=("sd", "median"), rmse=("rmse", "median"),
            frac1=("frac_within_1", "mean"), cost=("cost_us", "median")).round(2)
        print("WT-MetaD grid (median over conditions so far):"); print(v.to_string())
    if len(fl):
        v = fl.groupby(["family", "cv", "estimator", "n"]).agg(
            conds=("cost_us", "size"), sd=("sd", "median"), rmse=("rmse", "median"),
            frac1=("frac_within_1", "mean"), cost=("cost_us", "median")).round(2)
        print("Flooding sweeps:"); print(v.to_string())


if __name__ == "__main__":
    main()
