"""Generate the Supporting Information tables (LaTeX, booktabs) from analysis/*.csv.

Usage:
    uv run python scripts/make_si_tables.py
Output: overleaf_best_rates_paper/si_tables/*.tex (override with SI_TABLES_DIR),
    one file per table, included by EfficientEATR_SI.tex.
Every number is read from the CSV tables of this repository; nothing is typed by hand.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
# the manuscript lives beside the working tree but not inside the released archive, so the
# supplementary tables go to the manuscript when it is there and to analysis/ when it is not
_DEFAULT_OUT = (ROOT / "overleaf_best_rates_paper/si_tables"
                if (ROOT / "overleaf_best_rates_paper").is_dir() else ROOT / "analysis/si_tables")
OUT = Path(os.environ.get("SI_TABLES_DIR", _DEFAULT_OUT))
OUT.mkdir(parents=True, exist_ok=True)
CVS = ["Q", "RMSD", "Rg", "Ree"]
NAME = {"Q": "$Q$", "RMSD": "RMSD", "Rg": "$R_g$", "Ree": "$R_{ee}$"}


def fmt(x, nd=2):
    """Format a number or NaN for a table cell."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    s = f"{x:.{nd}f}"
    return s[1:] if s == "-" + f"{0:.{nd}f}" else s


LONG = 34   # rows above which a table is set as a longtable so it can break across pages

# The supplementary document carries only the tables whose numbers are quoted in the
# manuscript. Everything else is drawn in a figure there (the strategies behind the front
# in Fig. 5, the EATR-worse classification in Fig. S4, the censored runs in Fig. S7,
# the minimum data in Fig. 3, every grid condition in Figs. S8-S11) or is detail rather
# than support. All tables are written to FULL_OUT for the repository that accompanies
# the paper; SI_TABLES are additionally written to the manuscript's si_tables directory.
SI_TABLES = {"s2_error_by_pace", "s8_pace_ladder", "s4_mindata_by_bf", "s5_flooding_allsets", "s6_set_selection", "s17_gamma_grid",
             "s13_prospective", "s15_mindata_intervals", "s15b_mindata_intervals_static",
             "s18_temperature_cost"}
FULL_OUT = ROOT / "analysis/si_tables_full"


def write(name: str, caption: str, header: str, rows: list[str], label: str, cols: str, fit: bool = False) -> None:
    """Write one booktabs table to si_tables/<name>.tex.

    Tables are set where they are included ([H], from the float package), so that each one
    stays in the section that discusses it rather than drifting to the end; a table too long
    for a page is written as a longtable, which is not a float and breaks across pages,
    repeating its header.

    Args:
        name (str): file name without extension.
        caption (str): table caption.
        header (str): header row, columns separated by &.
        rows (list[str]): body rows, columns separated by &.
        label (str): LaTeX label.
        cols (str): column specification.
        fit (bool): scale the tabular to the text width, for a table too wide at the text size
            (RevTeX resets the font inside its table environment, so a size command cannot).
    """
    body = " \\\\\n".join(rows)
    if len(rows) > LONG:
        text = ("\\begin{longtable}{" + cols + "}\n\\caption{" + caption + "}\\label{" + label + "}\\\\\n"
                "\\toprule\n" + header + "\\\\ \\midrule\n\\endfirsthead\n"
                "\\toprule\n" + header + "\\\\ \\midrule\n\\endhead\n"
                "\\bottomrule\n\\endfoot\n" + body + "\\\\ \\bottomrule\n\\end{longtable}\n")
    else:
        tab = ("\\begin{tabular}{" + cols + "}\\toprule\n" + header + "\\\\ \\midrule\n"
               + body + "\\\\ \\bottomrule\n\\end{tabular}")
        if fit:
            tab = "\\resizebox{\\textwidth}{!}{" + tab + "}"
        text = ("\\begin{table}[H]\\centering\n\\caption{" + caption + "}\\label{" + label + "}\n"
                + tab + "\n\\end{table}\n")
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    (FULL_OUT / f"{name}.tex").write_text(text)
    if name in SI_TABLES:
        (OUT / f"{name}.tex").write_text(text)


def main() -> None:
    """Build every SI table."""
    # S1: grid accuracy per bias factor
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    full = im[(im.n == im.n_total) & (im.bf >= 2)]
    rows = []
    for cv in CVS:
        for bf in (2, 5, 10, 15, 20):
            g = full[(full.cv == cv) & (full.bf == bf)]
            e, i = g[g.estimator == "EATR CDF"].mean_err, g[g.estimator == "iMetaD CDF"].mean_err
            rows.append(f"{NAME[cv]} & {bf} & {len(e)} & {fmt(e.abs().median())} & {fmt((e.abs() <= 1).mean())} & {fmt(i.abs().median())} & {fmt((i.abs() <= 1).mean())}")
    write("s1_grid_by_bf", "Well-tempered metadynamics grid, all runs per condition: median absolute error and fraction of conditions within one ln unit, by bias factor (32 conditions per CV and bias factor, fewer where the fit failed).",
          "CV & $b$ & $n$ & EATR med $|\\varepsilon|$ & EATR within 1 & iMetaD med $|\\varepsilon|$ & iMetaD within 1", rows, "tab:s1", "llccccc")
    # S2: iMetaD and EATR signed median error by pace (bf 5-20)
    rows = []
    for pace in sorted(full.pace.unique()):
        g = full[(full.pace == pace) & (full.bf >= 5)]
        cells = []
        for cv in CVS:
            cells.append(fmt(g[(g.cv == cv) & (g.estimator == "iMetaD CDF")].mean_err.median()))
            cells.append(fmt(g[(g.cv == cv) & (g.estimator == "EATR CDF")].mean_err.median()))
        rows.append(f"{int(pace)} & " + " & ".join(cells))
    write("s2_error_by_pace", "Median signed error by deposition pace, bias factors 5--20 (iMetaD / EATR for each CV).",
          "pace (steps) & \\multicolumn{2}{c}{$Q$} & \\multicolumn{2}{c}{RMSD} & \\multicolumn{2}{c}{$R_g$} & \\multicolumn{2}{c}{$R_{ee}$}", rows, "tab:s2", "lcccccccc")
    # S3: minimum data by pace (bf 2-20)
    m = pd.read_csv(ROOT / "analysis/minimum_data_summary.csv"); w = m[m.family == "wtmetad_by_pace"]
    rows = []
    for pace in sorted(w.pace.unique()):
        cells = []
        for cv in CVS:
            for est in ("EATR CDF", "iMetaD CDF"):
                r = w[(w.pace == pace) & (w.cv == cv) & (w.estimator == est)]
                if len(r) and not math.isnan(r["min_cost_us[frac_within_1]"].iloc[0]):
                    cells.append(f"{r['min_cost_us[frac_within_1]'].iloc[0]:.1f} ({int(r['n_at_min[frac_within_1]'].iloc[0])})")
                else:
                    cells.append("--")
        rows.append(f"{int(pace)} & " + " & ".join(cells))
    write("s3_mindata_by_pace", "Smallest cost ($\\mu$s; runs in parentheses) at which 90\\% of random subsets land within one ln unit and stay there at larger sizes, by pace, pooled over bias factors 2--20 (EATR / iMetaD per CV); -- never reached.",
          "pace & \\multicolumn{2}{c}{$Q$} & \\multicolumn{2}{c}{RMSD} & \\multicolumn{2}{c}{$R_g$} & \\multicolumn{2}{c}{$R_{ee}$}", rows, "tab:s3", "lcccccccc")
    # S4: minimum data by bias factor at pace >= 10000, bf 5-20 vs bf 2
    sub = im[(im.n < im.n_total) & (im.pace >= 10000)]
    rows = []
    for bfs, lab in (((2,), "2"), ((5, 10, 15, 20), "5--20")):
        for cv in CVS:
            cells = []
            for est in ("EATR CDF", "iMetaD CDF"):
                g = sub[(sub.cv == cv) & (sub.estimator == est) & sub.bf.isin(bfs)]
                a = g.groupby("n").agg(cost=("cost_us", "median"), p=("frac_within_1", "mean")).reset_index().sort_values("n")
                ok = [bool((a.p.iloc[i:] >= 0.9).all()) for i in range(len(a))]
                hit = a[ok].sort_values("cost")
                cells.append(f"{hit.cost.iloc[0]:.1f} ({int(hit.n.iloc[0])})" if len(hit) else "--")
            rows.append(f"{lab} & {NAME[cv]} & " + " & ".join(cells))
    write("s4_mindata_by_bf", "The same criterion at pace $\\ge 10^4$ steps, separately for bias factor 2 and for bias factors 5--20: the gentle bias-factor-2 protocols are where iMetaD converges cheaply.",
          "bias factor & CV & EATR & iMetaD", rows, "tab:s4", "llcc")
    # S5: flooding protocols, every bias level
    mt = pd.read_csv(ROOT / "analysis/master_table.csv")
    rows = []
    for fam, lab in (("opesf", "OPES (per pace)"), ("vesf", "VES (per $N$)")):
        for cv in CVS:
            g = mt[(mt.family == fam) & (mt.cv == cv)]
            e, o = g[g.estimator == "EATRf"], g[g.estimator == "OPESf"]
            rows.append(f"{lab} & {NAME[cv]} & {fmt(e.err_ln.median())} & {fmt(e.err_ln.min())} / {fmt(e.err_ln.max())} & {fmt(e.gamma.median())} & {fmt(o.err_ln.median())} & {fmt(o.err_ln.min())} / {fmt(o.err_ln.max())}")
    write("s5_flooding_allsets", "Flooding with every bias level: median (and range) of the signed error and the fitted $\\gamma$ for EATR-flooding, and the same error for the $\\gamma = 1$ estimate, over the eight paces for OPES (every run) and over the six numbers of runs per cutoff $N$ for VES (10 to 200).",
          "protocol & CV & EATRf med $\\varepsilon$ & range & $\\gamma$ & $\\gamma=1$ med $\\varepsilon$ & range", rows, "tab:s5", "llccccc")
    # S17: fitted gamma over the grid, by pace and by bias factor
    from tradeoff_analysis import load_grid_full
    gr = load_grid_full()
    gr = gr[gr.bf >= 5]
    slowg = gr[gr.pace >= 10000]
    rows = []
    for pace in sorted(gr.pace.unique()):
        d = gr[gr.pace == pace]
        rows.append(f"pace {int(pace)} & " + " & ".join(fmt(d[d.cv == cv].gamma.median()) for cv in CVS))
    for i, bf in enumerate(sorted(slowg.bf.unique())):
        d = slowg[slowg.bf == bf]
        rows.append(("\\midrule " if i == 0 else "") + f"$b = {int(bf)}$ & "
                    + " & ".join(fmt(d[d.cv == cv].gamma.median()) for cv in CVS))
    write("s17_gamma_grid", "Median fitted EATR efficiency $\\gamma$ over the well-tempered grid (bias factors 5--20), by deposition pace and, for the paces of $10^4$ steps and slower, by bias factor. The coordinate sets $\\gamma$; the bias factor barely moves it, and fast deposition lowers it for every CV but $R_{ee}$, whose $\\gamma$ is already the lowest.",
          "& $Q$ & RMSD & $R_g$ & $R_{ee}$", rows, "tab:s17", "lcccc")
    # S6: set-selection screen summary
    ss = pd.read_csv(ROOT / "analysis/flooding_set_selection.csv"); fullf = ss[ss.source == "triasha"]
    rows = []
    for fam in ("opes", "ves"):
        for cv in CVS:
            g = fullf[(fullf.family == fam) & (fullf.cv == cv)]
            rows.append(f"{fam.upper()} & {NAME[cv]} & {len(g)} & {int((g.n_screen < g.n_sets).sum())} & {fmt(g.slope_all.median())} & {fmt(g.err_all.abs().median())} & {fmt(g.err_screen.abs().median())} & {fmt(g.err_conv.abs().median())}")
    hb = ss[ss.source == "highbarrier"]
    for _, r in hb.iterrows():
        rows.append(f"OPES 5--17 & {NAME[r.cv]} & 1 & {int(r.n_screen < r.n_sets)} & {fmt(r.slope_all)} & {fmt(abs(r.err_all))} & {fmt(abs(r.err_screen))} & {fmt(abs(r.err_conv))}")
    write("s6_set_selection", "Overbiasing screen on the flooding fits of Table~\\ref{tab:s5} (eight OPES paces; six VES run counts $N$) and on the two ladders extended to 17~kJ/mol: number of fits, how often the screen removed a set, the slope of $\\ln k_{\\rm obs}$ against $\\ln\\langle e^{\\beta V}\\rangle$, and the median absolute error with all sets, with the screened sets, and with the convergence-selected sets.",
          "protocol & CV & fits & screened & slope & $|\\varepsilon|$ all & $|\\varepsilon|$ screened & $|\\varepsilon|$ conv.", rows, "tab:s6", "llcccccc")
    # S7: growth law per bias factor
    g = pd.read_csv(ROOT / "analysis/growth_law_conditions.csv")
    rows = []
    for cv in CVS:
        for bf in (2, 5, 10, 15, 20):
            d = g[(g.cv == cv) & (g.bf == bf)]
            rows.append(f"{NAME[cv]} & {bf} & {fmt(d.h_eff_fit_over_h0.median(), 3)} & {fmt(d.h_eff_eq11_over_h0.median(), 3)} & {fmt(d.h_eff_unbroadened_over_h0.median(), 3)} & {fmt((d.V_obs_kJmol / d.V_pred_unbroadened_kJmol).median())} & {fmt((d.t_obs_mean_ps / d.t_star_unbroadened_ps).median())}")
    write("s7_growth_law", "Growth-law test by bias factor: per-hill increment implied by the fill at escape ($h_{\\rm eff}/h_0$, fitted), the well-tempered fixed-point prediction, the unbroadened overlap, the ratio of observed to predicted fill with the unbroadened overlap, and the ratio of observed to predicted escape time (medians over heights and paces).",
          "CV & $b$ & fitted & fixed point & unbroadened & $V_{\\rm obs}/V_{\\rm pred}$ & $t_{\\rm obs}/t^\\ast$", rows, "tab:s7", "llccccc")
    # S8: the deposition-pace negative result
    sets = pd.read_csv(ROOT / "analysis/master_sets.csv"); pl = sets[sets.family == "eatrf_imetad"]
    rows = []
    for cv in CVS:
        for (bf, h), d in pl[pl.cv == cv].groupby(["bf", "height"]):
            if len(d) < 3:
                continue
            slope = np.polyfit(d.ln_exp_beta_v, d.log_k_obs_s, 1)[0]
            r = mt[(mt.family == "eatrf_imetad") & (mt.cv == cv) & (mt.bf == bf) & (mt.height == h) & (mt.estimator == "EATRf")]
            rows.append(f"{NAME[cv]} & {int(bf)} & {h} & {fmt(slope)} & {fmt(r.gamma.iloc[0]) if len(r) else '--'} & {fmt(r.err_ln.iloc[0]) if len(r) else '--'}")
    write("s8_pace_ladder_full", "EATR-flooding across the eight deposition paces of a growing bias at fixed bias factor and height: slope of $\\ln k_{\\rm obs}$ against the whole-trajectory acceleration, fitted $\\gamma$, and error.",
          "CV & $b$ & $h_0$ & slope & $\\gamma$ & $\\varepsilon$", rows, "tab:s8full", "lllccc")
    # the supplement carries the ranges the manuscript quotes, one row per CV
    summary = []
    for cv in CVS:
        d = [r.split(" & ") for r in rows if r.startswith(NAME[cv] + " &")]
        sl = [float(x[3]) for x in d if x[3] != "--"]
        ga = [float(x[4]) for x in d if x[4] != "--"]
        er = [float(x[5]) for x in d if x[5] != "--"]
        worst = max(er, key=abs)
        summary.append(f"{NAME[cv]} & {len(d)} & {min(sl):.2f}--{max(sl):.2f} & {min(ga):.2f}--{max(ga):.2f} & {worst:+.2f}")
    write("s8_pace_ladder", "EATR-flooding across the eight deposition paces of a growing bias, at each fixed bias factor and height: range over those series of the slope of $\\ln k_{\\rm obs}$ against the whole-trajectory acceleration, of the fitted $\\gamma$, and the largest error. Every series individually is in the analysis repository.",
          "CV & series & slope & $\\gamma$ & largest $\\varepsilon$", summary, "tab:s8", "lcccc")
    # S9: EATR-worse classification
    wc = pd.read_csv(ROOT / "analysis/eatr_worse_cases.csv")
    rows = []
    for cls, d in wc.groupby("class"):
        rows.append(f"{cls.split(' (')[0]} & {len(d)} & {fmt(d.err_eatr.median())} & {fmt(d.err_imetad.median())} & {fmt(d.gamma.median())} & {', '.join(str(int(p)) for p in sorted(d.pace.unique()))}")
    write("s9_eatr_worse", "The 90 conditions in which EATR was worse than iMetaD: class, count, median signed errors, median $\\gamma$, and the paces at which they occur.",
          "class & $n$ & EATR $\\varepsilon$ & iMetaD $\\varepsilon$ & $\\gamma$ & paces", rows, "tab:s9", "lccccp{0.3\\textwidth}")
    # S10: strategy table behind the front
    st = pd.read_csv(ROOT / "analysis/pareto_strategies.csv")
    for cv in CVS:
        d = st[st.cv == cv].sort_values(["cost_us", "rmse"], kind="mergesort")   # ties: lower RMSE first
        rows = [f"{r.label} & {r.cost_us:.2f} & {fmt(r.rmse)} & {fmt(r.p1)}" for _, r in d.iterrows()]
        write(f"s10_strategies_{cv}", f"Strategies behind the error--cost front for {NAME[cv]}: pooled RMSE over independent subsets and the fraction within one ln unit, at the median cost of the subsets.",
              "strategy & cost ($\\mu$s) & typical error & within 1", rows, f"tab:s10{cv}", "lccc")
    # S11: regime bins
    g2 = g[np.isfinite(g.eatr_err)].copy()
    g2["bin"] = pd.cut(g2.boost, [0, 3, 10, 30, 100, 1e9], labels=["$<3$", "3--10", "10--30", "30--100", "$>100$"])
    rows = [f"{b} & {len(d)} & {fmt(d.eatr_err.abs().median())} & {fmt((d.eatr_err.abs() <= 1).mean())} & {fmt(d.cv_times.median())}" for b, d in g2.groupby("bin", observed=True)]
    write("s11_regimes", "EATR accuracy by boost bin over the grid (bias factors 2--20): conditions, median absolute error, fraction within one ln unit, and the median spread of the escape times relative to their mean.",
          "boost & $n$ & med $|\\varepsilon|$ & within 1 & escape-time spread", rows, "tab:s11", "lcccc")
    untempered_table()
    print("wrote", len(list(FULL_OUT.glob("*.tex"))), "tables to", FULL_OUT,
          "of which", len(list(OUT.glob("*.tex"))), "to", OUT)


def untempered_table() -> None:
    """S16: every untempered metadynamics condition, both estimators side by side."""
    u = pd.read_csv(ROOT / "analysis/subsampling_untempered.csv")
    u = u[u.n == u.n_total]                                  # the full-data fit of each condition
    u["est"] = u.estimator.str.replace(" CDF", "", regex=False)
    rows = []
    for (cv, h, pace), g in u.groupby(["cv", "height", "pace"]):
        i = g[g.est == "iMetaD"], g[g.est == "EATR"]
        if not i[0].size or not i[1].size:
            continue
        im, ea = i[0].iloc[0], i[1].iloc[0]
        rows.append(f"{NAME[cv]} & {h:g} & $10^{{{int(np.log10(pace))}}}$ & {fmt(ea.cost_us, 1)} & "
                    f"{fmt(ea.gamma)} & {fmt(im.mean_err)} & {fmt(ea.mean_err)}")
    write("s16_untempered", "Untempered metadynamics, all 200 runs per condition: cost, fitted $\\gamma$ and the "
          "signed error of each estimator. The hill height never decays, so the bias keeps growing; every iMetaD "
          "error is negative, and the gap to EATR widens as $\\gamma$ falls.",
          "CV & $h_0$ & $\\tau$ & cost ($\\mu$s) & $\\gamma$ & $\\varepsilon$ iMetaD & $\\varepsilon$ EATR",
          rows, "tab:s16", "lccccrr")


def censoring_table() -> None:
    """S12: the flooding estimators on OPES bias levels whose runs stop at a common wall time."""
    c = pd.read_csv(ROOT / "analysis/censoring_robustness.csv")
    rows = []
    for _, r in c.iterrows():
        t = "--" if math.isnan(r.T_ns) else f"{r.T_ns:.0f}"
        rows.append(f"{NAME[r.cv]} & {100 * r.censored_all:.0f} & {100 * r.f_slowest:.0f} & {t} & "
                    f"{fmt(r.eatrf_err)} [{fmt(r.eatrf_lo)}, {fmt(r.eatrf_hi)}] & "
                    f"{fmt(r.gamma)} [{fmt(r.gamma_lo)}, {fmt(r.gamma_hi)}] & {fmt(r.opesf_err)}")
    write("s12_censoring", "OPES bias levels at pace 1000 (\\texttt{BARRIER} 5--13 kJ/mol, 50 runs per level) with every run stopped "
          "at a common wall time $T$, the $(1-f)$ quantile of the escape times of the slowest set; runs that have not "
          "escaped by $T$ are censored. Errors and $\\gamma$ are bootstrap means with 95\\% intervals (100 resamples); "
          "EATRf, EATR-flooding.",
          "CV & censored (\\%) & slowest (\\%) & $T$ (ns) & EATRf $\\varepsilon$ & $\\gamma$ & $\\gamma = 1$ $\\varepsilon$",
          rows, "tab:s12", "lcccccc")


def prospective_table() -> None:
    """S13: the prospective WT-MetaD protocols against their registered predictions."""
    o = pd.read_csv(ROOT / "analysis/prospective_outcome.csv").set_index("cv")
    cvs = [cv for cv in ("Q", "Ree") if cv in o.index]

    def yes(flag: bool) -> str:
        """Table word for a boolean."""
        return "yes" if flag else "no"

    def line(label: str, cell) -> str:
        """One quantity across the protocols."""
        return label + " & " + " & ".join(cell(o.loc[cv]) for cv in cvs)

    rows = [
        line("$b$ / $h_0$ (kJ/mol) / pace (steps)", lambda r: f"{int(r.bf)} / {r.height} / {int(r.pace)}"),
        line("runs escaped", lambda r: f"{int(r.committed)} of {int(r.runs)}"),
        line("boost: predicted / observed", lambda r: f"{fmt(r.pred_boost, 1)} / {fmt(r.obs_boost, 1)}"),
        line("escape (ns): $t^\\ast$ / expected / observed",
             lambda r: f"{r.pred_t_star_ns:.0f} / {r.expected_mean_escape_ns:.0f} / {r.obs_mean_escape_ns:.0f}"),
        line("EATR $\\varepsilon$ [95\\% interval]", lambda r: f"{fmt(r.eatr_err)} [{fmt(r.eatr_lo)}, {fmt(r.eatr_hi)}]"),
        line("EATR $\\gamma$ [95\\% interval] (rule input)", lambda r: f"{fmt(r.gamma)} [{fmt(r.gamma_lo)}, {fmt(r.gamma_hi)}] ({fmt(r.gamma_pilot)})"),
        line("iMetaD $\\varepsilon$ [95\\% interval]", lambda r: f"{fmt(r.imetad_err)} [{fmt(r.imetad_lo)}, {fmt(r.imetad_hi)}]"),
        line("EATR within one ln unit (registered expectation held)", lambda r: f"{yes(r.eatr_within_1)} ({yes(r.eatr_expectation_held)})"),
        line("iMetaD within one ln unit (registered expectation held)", lambda r: f"{yes(r.imetad_within_1)} ({yes(r.imetad_expectation_held)})"),
        line("KS $p$: EATR / iMetaD", lambda r: f"{r.eatr_ks_p:.2g} / {r.imetad_ks_p:.2g}"),
    ]
    ratios = ", ".join(f"{o.loc[cv].expected_mean_escape_ns / o.loc[cv].pred_t_star_ns:.2f} for {NAME[cv]}" for cv in cvs)
    write("s13_prospective", "Prospective test of the design rule: two protocols off the grid in bias factor, height and "
          "pace, chosen and predicted before any run (predictions committed to the repository first). The expected "
          "escape time is $t^\\ast$ times the median observed-to-predicted ratio over the grid conditions of the CV "
          "that the selection applied to the two protocols accepted, as computed at registration (" + ratios + "); "
          "errors and $\\gamma$ "
          "carry bootstrap 95\\% intervals, and the OPES-flooding $\\gamma$ given to the rule is in parentheses.",
          "& " + " & ".join(NAME[cv] for cv in cvs), rows, "tab:s13", "l" + "c" * len(cvs), fit=True)


def mindata_intervals_table() -> None:
    """S15: the minimum-data entries of the main text with subset draws and hit-rate intervals."""
    m = pd.read_csv(ROOT / "analysis/minimum_data_summary.csv")
    selection = [("wtmetad_by_pace", 30000, "WT-MetaD, pace $3\\times10^4$", ("EATR CDF", "iMetaD CDF")),
                 ("wtmetad_by_pace", 100000, "WT-MetaD, pace $10^5$", ("EATR CDF", "iMetaD CDF")),
                 ("opes", None, "OPES, 5 barriers", ("EATRf", "OPESf")),
                 ("ves", None, "VES, 6 cutoffs", ("EATRf", "OPESf"))]
    label = {"EATR CDF": "EATR", "iMetaD CDF": "iMetaD", "EATRf": "EATR-flooding", "OPESf": "$\\gamma = 1$"}
    # one table per kind of bias: together the 32 rows overflow a page of the supplement
    support = ("cost ($\\mu$s) and runs at the smallest size where 90\\% of random subsets land within one ln "
               "unit and stay there, how many subsets were drawn at that size, and the uncertainty on the fraction "
               "of them within one ln unit (a 95\\% range)")
    header = "protocol & estimator & CV & cost ($\\mu$s) & runs & draws & 95\\% interval"
    for name, families, caption in (
            ("s15_mindata_intervals", ("wtmetad_by_pace",),
             f"The minimum-data entries of the main text for the growing bias, with their support: {support}."),
            ("s15b_mindata_intervals_static", ("opes", "ves"),
             f"The same for the static bias levels: {support}.")):
        rows = []
        for fam, pace, protocol, estimators in selection:
            if fam not in families:
                continue
            for est in estimators:
                for cv in CVS:
                    r = m[(m.family == fam) & (m.cv == cv) & (m.estimator == est)]
                    if pace is not None:
                        r = r[r.pace == pace]
                    first = cv == CVS[0]
                    lead = f"{protocol if first and est == estimators[0] else ''} & {label[est] if first else ''} & {NAME[cv]}"
                    if len(r) and not math.isnan(r["min_cost_us[frac_within_1]"].iloc[0]):
                        x = r.iloc[0]
                        rows.append(f"{lead} & {x['min_cost_us[frac_within_1]']:.1f} & {int(x['n_at_min[frac_within_1]'])} & "
                                    f"{int(x['draws_at_min[frac_within_1]'])} & "
                                    f"{x['hit_rate_lo95[frac_within_1]']:.2f}--{x['hit_rate_hi95[frac_within_1]']:.2f}")
                    else:
                        rows.append(f"{lead} & never & -- & -- & --")
        write(name, caption, header, rows, "tab:s15" if name == "s15_mindata_intervals" else "tab:s15b", "lllcccc")


def temperature_cost_table() -> None:
    """Cost of every low-temperature condition against the mean first-passage time its barrier implies."""
    t = pd.read_csv(ROOT / "analysis/temperature_rates.csv")

    def sci(x: float) -> str:
        """A ratio in LaTeX scientific notation.

        Args:
            x (float): value.
        """
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$" if x < 0.01 else fmt(x, 2)

    rows = []
    for T in (312, 300, 285, 270):
        g = t[(t.temperature_K == T) & (t.family == "opes ladder")].set_index("estimator")
        e, n = g.loc["EATR-flooding"], g.loc["gamma = 1"]
        rows.append(f"{T} & OPES, five levels & {int(e.runs)} & {fmt(e.cost_us, 1)} & {fmt(e.mfpt_us, 2 if e.mfpt_us < 100 else 0)} & "
                    f"{sci(e.cost_over_mfpt)} & {fmt(e.boost_obs, 0)} & {fmt(e.offset)} & {fmt(n.offset)}")
        for fam, lab in (("wtmetad single, inside rule", "WT-MetaD, one condition"),
                         ("wtmetad single, R_ee", "WT-MetaD, one condition, $R_{ee}$")):
            w = t[(t.temperature_K == T) & (t.family == fam)].set_index("estimator")
            if len(w):
                a, b = w.loc["EATR"], w.loc["iMetaD"]
                rows.append(f"{T} & {lab} & {int(a.runs)} & {fmt(a.cost_us, 1)} & {fmt(a.mfpt_us, 2 if a.mfpt_us < 100 else 0)} & "
                            f"{sci(a.cost_over_mfpt)} & {fmt(a.boost_obs, 0)} & {fmt(a.offset)} & {fmt(b.offset)}")
    write("s18_temperature_cost",
          "Cost of the low-temperature conditions against the mean first-passage time (MFPT) the barrier implies, "
          "$e^{\\Delta(\\text{barrier})}$ times the 312~K value of $0.65~\\mu$s. The five OPES-flooding levels carry 50 runs each; "
          "the well-tempered condition is the one per temperature that kept more than $2~\\kB T$ of barrier, "
          "50 runs at 300 and 285~K and 200 at 270~K. The same protocol along $R_{ee}$ is listed at each temperature, with 200 runs. "
          "The boost is the first-passage time over the mean biased escape time. Errors $\\varepsilon$ are against the "
          "barrier prediction $\\Delta\\ln k = -\\Delta(\\text{barrier})$ through the 312~K reference, "
          "for the $\\gamma$-fitting estimator (EATR-flooding, EATR) and the naive one ($\\gamma = 1$).",
          "$T$ (K) & protocol & runs & cost ($\\mu$s) & MFPT ($\\mu$s) & cost/MFPT & boost & "
          "$\\varepsilon$ ($\\gamma$ fitted) & $\\varepsilon$ ($\\gamma = 1$)",
          rows, "tab:s18", "llrrrrrrr", fit=True)


if __name__ == "__main__":
    main()
    censoring_table()
    prospective_table()
    mindata_intervals_table()
    temperature_cost_table()
