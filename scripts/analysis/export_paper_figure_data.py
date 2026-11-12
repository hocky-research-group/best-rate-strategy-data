"""Export the plot-ready data behind every manuscript and SI figure to paper_figures/data/.

Usage:
    uv run python scripts/export_paper_figure_data.py

Reads the committed analysis tables (analysis/*.csv) and data/ inputs, applies the
selections the figures use, and writes one tidy CSV per figure element plus
manifest.json (git commit, source files and row counts of every output). The figure
notebook paper_figures/make_paper_figures.ipynb reads only this folder.

Rates are ln(k / us^-1); errors are ln k_est - ln k_ref in natural-log units; costs are
total simulated time in us; free energies are in kT at 312 K.

Outputs (paper_figures/data/):
    reference.csv          unbiased reference: ln k, bootstrap sd, cost
    cv_summary.csv         per CV: basin width, apparent barrier, barrier position,
                           unfolded fraction at the barrier bin, median fitted gamma
    pmf.csv                unbiased PMFs, cv, x, F_kT, fraction of frames unfolded per bin
    overlays.csv           F + <V> and <V> per set for WT-MetaD, OPES and VES
    overlay_errors.csv     full-data errors and gamma of the overlaid sets
    accuracy_cost.csv      signed error and cost of every full-data estimate, including
                           untempered metadynamics (family "untempered") and
                           frequency-adaptive metadynamics (family "fametad", carrying the
                           acceleration threshold in "theta") when fitted
    mindata_curves.csv     P1 and spread against cost per protocol, estimator and n
    mindata_points.csv     smallest cost at which nine subsets in ten land within one ln
                           unit, per protocol, estimator and CV
    pareto_strategies.csv  every strategy of the error-cost front
    pareto_front.csv       the front itself
    pareto_background.csv  per-condition subset points of the grid
    design_conditions.csv  growth-law and design-rule quantities per grid condition,
                           with the residual barrier and the deposition interval in units
                           of the CV relaxation time
    cv_autocorrelation.csv relaxation time of each CV in the folded basin, from the
                           unbiased runs where available
    prospective.csv        the two prospective WT-MetaD protocols
    highbarrier_sets.csv   per-set values of the pace-1000 OPES ladders run past the barrier (all CVs)
    highbarrier_conv.csv   EATR-flooding over the lowest n sets of the Q and Ree ladders
    eatr_vs_imetad.csv     |error| of both estimators for every full grid fit
    eatr_worse_cases.csv   classification of the conditions in which EATR is worse
    sets_scan.csv          errors against the number of sets retained, full ladders
    ves_stage1.csv         stage-1 bias at the walker for Q
    ves_static_max.csv     maximum of the static VES bias per CV and cutoff
    ves_sets.csv           per-set values of the VES ladders, N = 200
    opes_sets.csv          per-set values of the five-barrier OPES ladders, every pace
    ks_grid.csv            KS p and error of every full-data grid fit, iMetaD and EATR
    cdf_fits.csv           empirical and fitted CDF against each estimator's time variable, bf 10,
                           h0 0.25 kJ/mol, four paces, from the grid fits
    cdf_fits_meta.csv      KS p and statistic, error, gamma and n of those fits
    censoring.csv          OPES ladders with censored runs
    temperature_rates.csv  EATR-flooding and naive rates of the OPES levels at 270--312 K
    temperature_convergence.csv  those fits with three, four and five levels
    temperature_pmf.csv    free energy along Q at each temperature, reweighted, both seeds
Sanity checks (raise on failure): 639 full grid fits per estimator; 160 conditions per
CV except RMSD (159); reference cost 260 us to 1 us; every overlay CV has a PMF; the
front is a subset of the strategies.
"""
import json
import subprocess
import sys
from pathlib import Path

from paths import dataset
import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from flooding_io import full_data_view  # noqa: E402
from bias_design_recipe import relaxation_table  # noqa: E402
from tradeoff_analysis import MAX_POOL_FRACTION  # noqa: E402

OUT = dataset("figures.data")
KT = 0.0083144626 * 312.0
L6 = np.log(1e6)
CVS = ["Q", "RMSD", "Rg", "Ree"]
PMFKEY = {"Q": "q", "RMSD": "rmsd", "Rg": "rg", "Ree": "ree"}
PROF = dataset("runs.bias_profiles")
MANIFEST: dict = {}


def write(df: pd.DataFrame, name: str, sources: list[str]) -> None:
    """Write one output table and record it in the manifest.

    Args:
        df (pd.DataFrame): table to write.
        name (str): file name inside paper_figures/data.
        sources (list[str]): repository-relative inputs the table was built from.
    """
    df.to_csv(OUT / name, index=False)
    MANIFEST[name] = dict(rows=len(df), columns=list(df.columns), sources=sources)
    print(f"{name}: {len(df)} rows")


def reference() -> dict:
    """Unbiased reference rate; returns the parsed JSON."""
    ref = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())
    cost = ref["n_runs"] * ref["mean_fpt_us"]
    assert abs(cost - 260) < 1, cost
    write(pd.DataFrame([dict(ln_k=ref["ln_k_mle_per_us"], sd=ref["ln_k_mle_boot_std"], cost_us=cost,
                             n_runs=ref["n_runs"])]), "reference.csv", ["data/reference_rate/reference_rate.json"])
    return ref


def pmfs_and_summary() -> dict:
    """PMFs and per-CV summary; returns the PMF summary JSON."""
    summary = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    rows = []
    for cv in CVS:
        a = np.loadtxt(dataset("pmf_unbiased") / f"pmf_{PMFKEY[cv]}.dat")
        rows.append(pd.DataFrame(dict(cv=cv, x=a[:, 0], F_kT=a[:, 1] / KT, p_unfolded=a[:, 2])))
    write(pd.concat(rows), "pmf.csv", [f"data/heuristics/pmf_unbiased/pmf_{k}.dat" for k in PMFKEY.values()])
    sub = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    grid = sub[(sub.estimator == "EATR CDF") & (sub.n == sub.n_total) & (sub.bf >= 2) & (sub.pace >= 1e4)]
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    opes = mt[(mt.family == "opesf") & (mt.estimator == "EATRf")]
    rows = []
    for cv in CVS:
        s = summary[PMFKEY[cv]]
        rows.append(dict(cv=cv, sigma_A=s["basin_sigma"], barrier_kT=s["barrier_kT"], barrier_pos=s["barrier_pos"],
                         folded_min=s["folded_min"], p_unfolded_at_barrier=s["p_unfolded_at_barrier"],
                         gamma_grid_slow=grid[grid.cv == cv].gamma.median(),
                         gamma_opes_ladder=opes[opes.cv == cv].gamma.median()))
    write(pd.DataFrame(rows), "cv_summary.csv", ["data/heuristics/pmf_unbiased/pmf_summary.json",
                                                 "analysis/subsampling_imetad.csv", "analysis/master_table.csv"])
    return summary


def overlays() -> None:
    """F + <V> per set on the PMF grid, with the same row filter as the working figure."""
    sets = []
    for cv in ("Q", "Ree"):
        sets += [("imetad", cv, f"bf10_height0.5_pace{p}", p) for p in (100, 1000, 100000)]
    for cv in ("Q", "Ree", "Rg", "RMSD"):
        sets += [("opes", cv, f"barrier{b}_pace1000", b) for b in (5, 7, 9, 11, 13)]
        sets += [("ves", cv, f"fc{fc}", fc) for fc in (5, 7, 9, 11, 13, 15)]
    rows = []
    for fam, cv, name, val in sets:
        f = PROF / f"profile_{fam}_{cv}_{name}.dat"
        if not f.exists():
            continue
        a = np.loadtxt(dataset("pmf_unbiased") / f"pmf_{PMFKEY[cv]}.dat")
        x, F = a[:, 0], a[:, 1] / KT
        p = np.loadtxt(f)
        ok = p[:, 2] >= 20  # bins visited by at least 20 rows
        V = np.interp(x, p[ok, 0], p[ok, 1] / KT, left=np.nan, right=np.nan)
        rows.append(pd.DataFrame(dict(family=fam, cv=cv, set_value=val, x=x, F_kT=F, V_kT=V, F_plus_V_kT=F + V)))
    write(pd.concat(rows), "overlays.csv", ["data/results/bias_profiles/profile_*.dat", "data/heuristics/pmf_unbiased/"])
    # WT-MetaD from the regenerated grid (the source of every grid number in the paper);
    # flooding from master_table
    g = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    g = g[(g.n == g.n_total) & (g.bf == 10) & (g.height == 0.5) & g.pace.isin([100, 1000, 100000])]
    g = g.rename(columns={"mean_err": "err_ln", "pace": "key"}).assign(family="imetad")
    assert len(g) == 24 and not g.duplicated(["cv", "key", "estimator"]).any()
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    mt = mt[mt.valid]
    fl = pd.concat([mt[(mt.family == "opesf") & (mt.pace == 1000)].assign(key=1000),
                    mt[(mt.family == "vesf") & (mt.barrier_label == 200)].assign(key=200)])
    fl = fl[fl.estimator.isin(["OPESf", "EATRf"])]
    assert not fl.duplicated(["family", "cv", "estimator"]).any()
    write(pd.concat([g, fl])[["family", "cv", "key", "estimator", "err_ln", "gamma"]], "overlay_errors.csv",
          ["analysis/subsampling_imetad.csv", "analysis/master_table.csv"])


def accuracy_cost(ref: dict) -> None:
    """Every full-data estimate: WT-MetaD grid (bf 2-20) from the regenerated grid, flooding from master_table."""
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    mt = mt[mt.valid & mt.family.isin(["opesf", "vesf"]) & mt.estimator.isin(["OPESf", "EATRf"])]
    g = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    g = g[(g.n == g.n_total) & (g.bf >= 2)].rename(columns={"mean_err": "err_ln"})
    counts = g.groupby(["cv", "estimator"]).size()
    assert counts.groupby("estimator").sum().eq(639).all(), counts
    assert counts.loc["RMSD"].eq(159).all() and counts.drop("RMSD", level=0).eq(160).all(), counts
    g["family"] = "wtmetad"
    g["estimator"] = g.estimator.str.replace(" CDF", "")
    fl = mt.assign(n_label=mt.barrier_label)[["family", "cv", "estimator", "pace", "n_label", "err_ln", "cost_us", "gamma"]]
    parts = [g[["family", "cv", "estimator", "bf", "height", "pace", "err_ln", "cost_us", "gamma"]], fl]
    # untempered metadynamics (METAD without BIASFACTOR), when those fits exist
    ut_path = ROOT / "analysis/subsampling_untempered.csv"
    if ut_path.exists():
        u = pd.read_csv(ut_path)
        u = u[u.n == u.n_total].rename(columns={"mean_err": "err_ln"})
        u["family"] = "untempered"
        u["estimator"] = u.estimator.str.replace(" CDF", "", regex=False)
        parts.append(u[["family", "cv", "estimator", "bf", "height", "pace", "err_ln", "cost_us", "gamma"]])
        print(f"untempered conditions: {u[['cv', 'height', 'pace']].drop_duplicates().shape[0]}")
    # frequency-adaptive metadynamics, labelled by its acceleration threshold
    fa_path = ROOT / "analysis/subsampling_fametad.csv"
    if fa_path.exists():
        f = pd.read_csv(fa_path)
        f = f[f.n == f.n_total].rename(columns={"mean_err": "err_ln"})
        f["family"] = "fametad"
        f["estimator"] = f.estimator.str.replace(" CDF", "", regex=False)
        parts.append(f[["family", "cv", "estimator", "bf", "height", "pace", "theta",
                        "err_ln", "cost_us", "gamma"]])
        print(f"fametad conditions: {f[['cv', 'theta']].drop_duplicates().shape[0]}")
    out = pd.concat(parts, ignore_index=True)
    write(out, "accuracy_cost.csv", ["analysis/subsampling_imetad.csv", "analysis/master_table.csv"])


def mindata() -> None:
    """P1 and spread per subset size: mean P1 and median cost and spread over conditions or paces."""
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    im = im[(im.n < im.n_total) & (im.bf >= 2)]
    fl = pd.read_csv(ROOT / "analysis/subsampling_flooding.csv")
    fl = fl[fl.estimator.isin(["OPESf", "EATRf"])]
    rows = []
    for (cv, est), g in im.groupby(["cv", "estimator"]):
        a = g.groupby("n").agg(cost_us=("cost_us", "median"), p1=("frac_within_1", "mean"), sd=("sd", "median"),
                               n_conditions=("pace", "size")).reset_index()
        rows.append(a.assign(protocol="wtmetad", cv=cv, estimator=est.replace(" CDF", "")))
    for (fam, cv, est), g in fl.groupby(["family", "cv", "estimator"]):
        a = g.groupby("n").agg(cost_us=("cost_us", "median"), p1=("frac_within_1", "mean"), sd=("sd", "median"),
                               n_conditions=("n_fits", "size")).reset_index()
        rows.append(a.assign(protocol=fam, cv=cv, estimator=est))
    write(pd.concat(rows)[["protocol", "cv", "estimator", "n", "cost_us", "p1", "sd", "n_conditions"]],
          "mindata_curves.csv", ["analysis/subsampling_imetad.csv", "analysis/subsampling_flooding.csv"])


def mindata_points() -> None:
    """Minimum data per strategy: the entries of the manuscript's minimum-data table.

    One row per (protocol, estimator, CV) with the smallest cost at which nine subsets in
    ten land within one ln unit and stay there, the number of runs there, and a flag for
    the strategies that never reach it.
    """
    m = pd.read_csv(ROOT / "analysis/minimum_data_summary.csv")
    keep = pd.concat([m[(m.family == "wtmetad_by_pace") & m.pace.isin([30000.0, 100000.0])],
                      m[m.family.isin(["opes", "ves"])]])
    out = keep.rename(columns={"min_cost_us[frac_within_1]": "min_cost_us",
                               "n_at_min[frac_within_1]": "n_at_min",
                               "hit_rate_lo95[frac_within_1]": "lo95",
                               "hit_rate_hi95[frac_within_1]": "hi95"})
    out["never"] = out.min_cost_us.isna()
    out["estimator"] = out.estimator.str.replace(" CDF", "", regex=False)
    out["protocol"] = np.where(out.family == "wtmetad_by_pace",
                               "WT-MetaD, pace " + out.pace.fillna(0).astype(int).astype(str),
                               out.family.str.upper())
    write(out[["protocol", "family", "pace", "cv", "estimator", "min_cost_us", "n_at_min",
               "lo95", "hi95", "never"]], "mindata_points.csv", ["analysis/minimum_data_summary.csv"])


def pareto() -> None:
    """Strategies, front and background points of the error-cost front (production cost only)."""
    st = pd.read_csv(ROOT / "analysis/pareto_strategies.csv")
    fr = pd.read_csv(ROOT / "analysis/pareto_front.csv")
    assert set(fr.label + fr.cv) <= set(st.label + st.cv)
    write(st, "pareto_strategies.csv", ["analysis/pareto_strategies.csv"])
    write(fr, "pareto_front.csv", ["analysis/pareto_front.csv"])
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    # the same subset sizes as the strategies: none above the share of the pool they allow
    im = im[(im.n <= MAX_POOL_FRACTION * im.n_total) & (im.bf >= 5) & (im.n_fits >= 5)]
    bg = im.assign(estimator=im.estimator.str.replace(" CDF", ""))
    write(bg[["cv", "estimator", "bf", "height", "pace", "n", "cost_us", "rmse"]], "pareto_background.csv",
          ["analysis/subsampling_imetad.csv"])


def temperature() -> None:
    """The static bias levels repeated at lower temperature: rates against the barrier, and the fit by number of levels."""
    tr = pd.read_csv(ROOT / "analysis/temperature_rates.csv")
    write(tr, "temperature_rates.csv", ["analysis/temperature_rates.csv"])
    tc = pd.read_csv(ROOT / "analysis/temperature_convergence.csv")
    write(tc, "temperature_convergence.csv", ["analysis/temperature_convergence.csv"])
    pf = pd.read_csv(ROOT / "analysis/fes_temperature_scan_profiles.csv")
    pf = pf[(pf.estimate == "reweight") & (pf.window == "second half")][["temperature_K", "seed", "q", "F_kT"]]
    write(pf, "temperature_pmf.csv", ["analysis/fes_temperature_scan_profiles.csv"])


def design() -> None:
    """Growth-law quantities per grid condition joined with the a priori design rule, and the prospective runs."""
    gl = pd.read_csv(ROOT / "analysis/growth_law_conditions.csv")
    val = pd.read_csv(ROOT / "analysis/bias_design_validation.csv")
    # the rule's verdict and the quantities it was applied to, from the one table that
    # carries them (bias_design_recipe.py validate); bias factor 2 lies outside the rule
    d = gl.merge(val[["cv", "bf", "height", "pace", "gamma_pilot", "pred_boost", "pred_t_ns", "recommended",
                      "residual_pred_kT", "tau_rel_ps"]], on=["cv", "bf", "height", "pace"], how="left")
    assert len(d) == len(gl)
    summary = json.loads((dataset("pmf_unbiased") / "pmf_summary.json").read_text())
    ac = relaxation_table()
    d["barrier_kT"] = d.cv.map({cv: summary[PMFKEY[cv]]["barrier_kT"] for cv in CVS})
    # the retrospective diagnostic of Fig. 4b: barrier left after the fill each run reached
    d["residual_kT"] = d.barrier_kT - d.gamma * d.V_obs_kJmol / KT
    d["tau_ps"] = d.pace * 0.01                      # 10 fs per step
    d["tau_rel_ps"] = d.cv.map(ac.set_index("cv").tau_int_ps)
    d["tau_over_relax"] = d.tau_ps / d.tau_rel_ps
    cols = ["cv", "bf", "height", "pace", "gamma", "V_obs_kJmol", "V_pred_unbroadened_kJmol", "V_pred_kJmol",
            "t_star_unbroadened_ps", "t_obs_mean_ps", "boost", "cv_times", "eatr_err", "gamma_pilot", "pred_boost",
            "pred_t_ns", "recommended", "barrier_kT", "residual_kT", "residual_pred_kT", "tau_ps", "tau_rel_ps",
            "tau_over_relax"]
    write(d[cols], "design_conditions.csv", ["analysis/growth_law_conditions.csv",
                                             "analysis/bias_design_validation.csv",
                                             "analysis/cv_autocorrelation_unbiased.csv",
                                             "analysis/cv_autocorrelation.csv"])
    write(ac, "cv_autocorrelation.csv", ["analysis/cv_autocorrelation_unbiased.csv",
                                         "analysis/cv_autocorrelation.csv"])
    po = pd.read_csv(ROOT / "analysis/prospective_outcome.csv")
    write(po, "prospective.csv", ["analysis/prospective_outcome.csv"])


def highbarrier(ref: dict) -> None:
    """Per-set values of the pace-1000 OPES ladders run past the barrier (full-data per-set view).

    Q and Ree: 7 levels (5-17 kJ/mol), bootstrap fits with set_reports from the full-data
    refit, plus the n-set EATR-flooding convergence used by the ladder figure. RMSD and Rg:
    9 levels (5-21 kJ/mol) fitted once without bootstrap, whose set_reports describe the
    full data directly; the levels above 13 kJ/mol exist for the slope figures only and
    enter no rate estimate, so no convergence rows are written for them.
    """
    sets, conv = [], []
    files = {"Q": ("Q_pace1000_barriers5-17.json", True), "Ree": ("Ree_pace1000_barriers5-17.json", True),
             "RMSD": ("RMSD_pace1000_barriers5-21_nobootstrap.json", False), "Rg": ("Rg_pace1000_barriers5-21_nobootstrap.json", False)}
    for cv, (name, with_conv) in files.items():
        # rel stays relative to the canonical data layout because it is also the key of the
        # refit index; only the file itself is resolved through the layout in use
        rel = f"results/opes_highbarrier/{name}"
        src = dataset("runs.highbarrier") / name
        d = full_data_view(json.loads(src.read_text()), rel, strict=True)
        assert d["_set_reports_source"].startswith("full-data"), (rel, d["_set_reports_source"])
        ss = sorted(d["set_reports"], key=lambda s: s["ln_exp_beta_v"])
        for s in ss:
            lnk_obs = s["log_k_obs"] - L6
            sets.append(dict(cv=cv, barrier=int(s["barrier"]), ln_exp_beta_v=s["ln_exp_beta_v"], lnk_obs_us=lnk_obs,
                             err_gamma1=lnk_obs - s["ln_exp_beta_v"] - ref["ln_k_mle_per_us"], gamma_all=d["gamma"]))
        if not with_conv:
            continue
        ca = d["flooding_diagnostics"]["convergence_analysis"]
        for n, lk in zip(ca["n_sets"], ca["logk0"]):
            conv.append(dict(cv=cv, n_sets=n, max_barrier=int(ss[n - 1]["barrier"]), err_eatrf=lk - L6 - ref["ln_k_mle_per_us"]))
    src = ["data/results/opes_highbarrier/", "data/refits_full_data/"]
    write(pd.DataFrame(sets), "highbarrier_sets.csv", src)
    write(pd.DataFrame(conv), "highbarrier_conv.csv", src)


CDF_PACES = (100, 1000, 10000, 30000)


def cdf_fit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical and fitted CDF of both estimators: bias factor 10, h0 0.25 kJ/mol, four paces, every CV.

    The fits are the full-data fits of the grid (dataset fits.grid), the ones behind every
    grid figure and table. Each stores, per estimator, the empirical CDF at the sorted
    escape times and the fitted CDF at the same points. time_ns is each estimator's own
    time variable: the raw escape time for EATR, whose fit is a function of raw time through
    the per-run acceleration, and the rescaled time t * alpha for iMetaD, whose fit is the
    exponential in that variable. k_teff = -ln(1 - F_fit) is the fitted cumulative hazard,
    on which every correct fit is 1 - exp(-x).

    Returns the curves (cv, pace, estimator, time_ns, k_teff, ecdf, fit) and one row per
    curve with its KS p and statistic, signed error, gamma (EATR only) and number of runs.
    """
    from tradeoff_analysis import REF, iter_grid_records
    want = {(cv, 10.0, 0.25, pace) for cv in CVS for pace in CDF_PACES}
    fits = {key: d for key, d in iter_grid_records() if key in want}
    assert set(fits) == want, sorted(want - set(fits))
    rows, meta = [], []
    for cv in CVS:
        for pace in CDF_PACES:
            d = fits[(cv, 10.0, 0.25, pace)]
            for est, key in (("iMetaD", "iMetaD CDF"), ("EATR", "EATR CDF")):
                blk = d[f"{key} plot"]
                assert blk["n_events"] == blk["n_total"] == len(blk["ecdf"]), (cv, pace, est)
                fit = np.clip(np.asarray(blk["fit"], dtype=float), 0.0, 1.0 - 1e-12)
                rows += [dict(cv=cv, pace=pace, estimator=est, time_ns=t * 1e9, k_teff=x, ecdf=e, fit=f)
                         for t, x, e, f in zip(blk["time"], -np.log1p(-fit), blk["ecdf"], blk["fit"])]
                meta.append(dict(cv=cv, pace=pace, estimator=est, ks_p=d[f"{key} p value"], ks_stat=d[f"{key} KS stat"],
                                 err_ln=d[f"{key} ln k"] - L6 - REF, gamma=d.get(f"{key} gamma", np.nan), n=blk["n_total"]))
    return pd.DataFrame(rows), pd.DataFrame(meta)


def cdf_fits() -> None:
    """Write the CDF curves and their per-curve summary (cdf_fit_tables)."""
    rows, meta = cdf_fit_tables()
    src = ["data/subsampling/imetad_grid/"]
    write(rows, "cdf_fits.csv", src)
    write(meta, "cdf_fits_meta.csv", src)


def ks_grid() -> None:
    """KS p-value and signed error of every full-data WT-MetaD grid fit (bf 2-20), both estimators."""
    from ks_detectability import grid_fits
    g = grid_fits()
    assert g.groupby("estimator").size().eq(639).all(), g.groupby("estimator").size()
    write(g, "ks_grid.csv", ["data/subsampling/imetad_grid/"])


def eatr_vs_imetad() -> None:
    """|error| of EATR and iMetaD for every full grid fit, and the EATR-worse classification."""
    from tradeoff_analysis import load_grid_full
    g = load_grid_full()
    write(g[["cv", "bf", "height", "pace", "err_eatr", "err_imetad"]], "eatr_vs_imetad.csv", ["data/subsampling/imetad_grid/"])
    w = pd.read_csv(ROOT / "analysis/eatr_worse_cases.csv")
    write(w[["cv", "bf", "height", "pace", "err_eatr", "err_imetad", "gamma", "class"]], "eatr_worse_cases.csv",
          ["analysis/eatr_worse_cases.csv"])


def sets_scan() -> None:
    """Error against the number of lowest-acceleration sets retained, full ladders only."""
    sc = pd.read_csv(ROOT / "analysis/flooding_sets_scan.csv")
    sc = sc[(sc.source == "triasha") & (((sc.family == "opes") & (sc.n_per_set == 50)) | ((sc.family == "ves") & (sc.n_per_set == 200)))]
    write(sc[["family", "cv", "pace", "n", "err_eatrf", "gamma", "err_g1"]], "sets_scan.csv", ["analysis/flooding_sets_scan.csv"])


def ves() -> None:
    """Stage-1 fill for Q, maximum of the static bias, and per-set values of the N = 200 ladders."""
    rows = []
    for fc in (5, 9, 13, 15):
        tr = np.loadtxt(PROF / f"ves_stage1_Q_fc{fc}.dat")
        rows.append(pd.DataFrame(dict(cv="Q", fc=fc, time_ns=tr[:, 0] / 1e3, bias_kJmol=tr[:, 1])))
    write(pd.concat(rows), "ves_stage1.csv", ["data/results/bias_profiles/ves_stage1_Q_fc*.dat"])
    rows = [dict(cv=cv, fc=fc, max_bias_kJmol=np.loadtxt(PROF / f"vesbias_{cv}_fc{fc}.dat")[:, 1].max())
            for cv in CVS for fc in (5, 7, 9, 11, 13, 15) if (PROF / f"vesbias_{cv}_fc{fc}.dat").exists()]
    write(pd.DataFrame(rows), "ves_static_max.csv", ["data/results/bias_profiles/vesbias_*.dat"])
    ms = pd.read_csv(ROOT / "analysis/master_sets.csv")
    v = ms[(ms.family == "vesf") & (ms.label == 200)].rename(columns={"set_label": "fc"})
    v = v.assign(lnk_obs_us=v.log_k_obs_s - L6)
    write(v[["cv", "fc", "ln_exp_beta_v", "lnk_obs_us", "tau_obs_us", "ks_p"]], "ves_sets.csv", ["analysis/master_sets.csv"])
    o = ms[ms.family == "opesf"].rename(columns={"set_label": "barrier"})
    o = o.assign(lnk_obs_us=o.log_k_obs_s - L6)
    write(o[["cv", "pace", "barrier", "ln_exp_beta_v", "lnk_obs_us", "tau_obs_us", "ks_p"]], "opes_sets.csv", ["analysis/master_sets.csv"])


def censoring() -> None:
    """OPES ladders with runs censored at a common wall time."""
    from censoring_robustness import ladder_rows
    write(ladder_rows(), "censoring.csv", ["data/verification/flooding_bootstrap_fix/"])


def snapshots() -> None:
    """The unbiased-run frames rendered for the Q ladder in Fig. 1: target Q, run, frame and the frame's q."""
    d = pd.read_csv(dataset("unbiased.q_ladder") / "selected_frames.csv")
    assert d.target.is_monotonic_decreasing and len(d) == 8
    write(d[["target", "run", "frame", "time_ps", "q"]], "q_ladder_frames.csv",
          ["data/unbiased_runs_q_ladder/selected_frames.csv", "data/unbiased_runs_dumped_frame_q.tsv"])


def main() -> None:
    """Export every table and write the manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    ref = reference()
    summary = pmfs_and_summary()
    overlays()
    assert set(pd.read_csv(OUT / "overlays.csv").cv) <= {c for c in CVS if PMFKEY[c] in summary}
    accuracy_cost(ref)
    mindata()
    mindata_points()
    pareto()
    design()
    temperature()
    highbarrier(ref)
    eatr_vs_imetad()
    ks_grid()
    cdf_fits()
    sets_scan()
    ves()
    censoring()
    snapshots()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "analysis", "data"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    (OUT / "manifest.json").write_text(json.dumps(dict(commit=commit, analysis_or_data_modified=dirty,
                                                       script="scripts/export_paper_figure_data.py", files=MANIFEST), indent=1))
    print("manifest: commit", commit, "(analysis/ or data/ modified)" if dirty else "")


if __name__ == "__main__":
    main()
