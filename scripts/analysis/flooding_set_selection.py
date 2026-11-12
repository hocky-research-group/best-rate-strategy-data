"""Overbiasing screen and set-selection scan for every EATR-flooding fit.

Usage:
    uv run python scripts/flooding_set_selection.py

Follows Mazzaferro et al. JCTC 2026 (Sec. 4.2 and Fig. 5): when the system is
not overbiased, ln k_obs is linear in ln<e^{beta V}> across the bias ladder with
slope ~ gamma; sets at the top of the ladder that curve down are overbiased and
are excluded. The scan over the n lowest-acceleration sets (n = 3 .. N) is
stored by eatr-flooding-analysis in flooding_diagnostics.convergence_analysis
(ln k0 and gamma for each n), so no refit is needed.

Screening rule (documented, applied identically everywhere):
    sort sets by x = ln<e^{beta V}> (gamma = 1); starting from n = N, fit
    y = ln k_obs on the first n-1 sets by least squares (slope s); if the local
    slope between sets n-1 and n is below 0.5 s, set n is declared overbiased
    and n -> n-1; stop at n = 3 or when no downturn is found. n_screen is the
    number of sets retained. n_conv is the tool's convergence choice: the n
    (3 <= n < N) minimising |ln k0(n+1) - ln k0(n)|, or N if none is smaller.

Inputs : every flooding JSON under data/triasha_results/prod_opes,
         data/triasha_results/prod_ves_cutoff/*/nsim-sweep, and
         data/subsampling/flooding_subsets (no-bootstrap subset fits, packed jsonl.gz)
Outputs: analysis/flooding_set_selection.csv (one row per fit) and
         analysis/flooding_sets_scan.csv (one row per fit and n), plus
         figures/fig4_flooding_sets_retained.png
"""
import gzip
import json
import math
import re
import sys
from pathlib import Path

import paths as _paths
from paths import dataset
import numpy as np
import pandas as pd

from flooding_io import full_data_view

ROOT = _paths.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
L6 = math.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]


def iter_json(root: Path, pattern: str):
    """Yield (relative name, data) for JSON files matching pattern under root,
    reading packed *.jsonl.gz archives when present.

    Args:
        root (Path): directory to search.
        pattern (str): regex applied to the relative path.
    """
    rx = re.compile(pattern)
    packed = sorted(root.rglob("*.jsonl.gz"))
    if packed:
        for pk in packed:
            with gzip.open(pk, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if "data" not in rec:
                        print(f"WARNING: {pk.name}: skipping failed fit {rec.get('file')}", file=sys.stderr)
                        continue
                    rel = str(pk.parent.relative_to(root) / rec["file"])
                    if rx.search(rel):
                        yield rel, rec["data"]
        return
    for f in sorted(root.rglob("*.json")):
        rel = str(f.relative_to(root))
        if rx.search(rel):
            try:
                yield rel, json.loads(f.read_text())
            except json.JSONDecodeError:
                continue


def screen(x: np.ndarray, y: np.ndarray) -> int:
    """Number of sets retained by the downturn rule (sets sorted by x).

    Args:
        x (np.ndarray): ln<e^{beta V}> per set, ascending.
        y (np.ndarray): ln k_obs per set.
    """
    n = len(x)
    while n > 3:
        s = np.polyfit(x[: n - 1], y[: n - 1], 1)[0]
        local = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
        if local < 0.5 * s:
            n -= 1
        else:
            break
    return n


def analyse(label: dict, d: dict) -> tuple[dict, list[dict]]:
    """Screening result and per-n scan for one flooding JSON.

    Args:
        label (dict): identifying columns (family, cv, pace, n_per_set, rep).
        d (dict): parsed JSON.
    """
    sets = d["set_reports"]
    order = np.argsort([s["ln_exp_beta_v"] for s in sets])
    x = np.array([sets[i]["ln_exp_beta_v"] for i in order])
    y = np.array([sets[i]["log_k_obs"] for i in order])
    labels = [sets[i]["barrier"] for i in order]
    cost = np.array([(sets[i]["tau_obs"] or 0) * (sets[i]["transitioned"] or 0) * 1e6 for i in order])
    ca = d["flooding_diagnostics"]["convergence_analysis"]
    ns = list(ca["n_sets"])
    k0n = np.array(ca["logk0"]) - L6 - REF
    gn = np.array(ca["gamma"])
    if ca.get("sorted_barrier_labels") is not None:
        assert [float(b) for b in ca["sorted_barrier_labels"]] == [float(b) for b in labels], (label, ca["sorted_barrier_labels"], labels)
    N = len(sets)
    n_screen = screen(x, y)
    dk = np.abs(np.diff(k0n))
    n_conv = ns[int(np.argmin(dk))] if len(dk) else N
    slope_all = np.polyfit(x, y, 1)[0]

    def at(n: int) -> float:
        return float(k0n[ns.index(n)]) if n in ns else float("nan")

    def gam(n: int) -> float:
        return float(gn[ns.index(n)]) if n in ns else float("nan")

    row = dict(label, n_sets=N, n_screen=n_screen, n_conv=n_conv, slope_all=slope_all,
               excluded=";".join(str(b) for b in labels[n_screen:]),
               err_all=d["logk0"] - L6 - REF, gamma_all=d["gamma"],
               err_screen=at(n_screen), gamma_screen=gam(n_screen),
               err_conv=at(n_conv), gamma_conv=gam(n_conv),
               err_g1_all=float(np.mean(y - x)) - L6 - REF,
               err_g1_lowest=float(y[0] - x[0]) - L6 - REF,
               cost_all=float(cost.sum()), cost_screen=float(cost[:n_screen].sum()), cost_conv=float(cost[:n_conv].sum()))
    scan = [dict(label, n=n, err_eatrf=at(n), gamma=gam(n),
                 err_g1=float(np.mean(y[:n] - x[:n])) - L6 - REF, cost=float(cost[:n].sum())) for n in ns]
    return row, scan


def main() -> None:
    """Run the screen on every flooding fit and write the tables and figure."""
    rows, scans = [], []
    for rel, d in iter_json(dataset("source.prod_opes"), r"pace\d+_flooding\.json$"):
        d = full_data_view(d, "triasha_results/prod_opes/" + rel)
        cv = re.search(r"bias(\w+)/", rel).group(1); pace = int(re.search(r"pace(\d+)_", rel).group(1))
        r, s = analyse(dict(family="opes", cv=cv, pace=pace, n_per_set=50, rep=0, source="triasha"), d); rows.append(r); scans += s
    for rel, d in iter_json(dataset("source.prod_ves_cutoff"), r"nsim-sweep/combined_N\d+\.json$"):
        d = full_data_view(d, "triasha_results/prod_ves_cutoff/" + rel)
        cv = re.search(r"bias(\w+)/", rel).group(1); n = int(re.search(r"combined_N(\d+)", rel).group(1))
        r, s = analyse(dict(family="ves", cv=cv, pace=np.nan, n_per_set=n, rep=0, source="triasha"), d); rows.append(r); scans += s
    for rel, d in iter_json(dataset("runs.highbarrier"), r"_barriers5-17\.json$"):
        d = full_data_view(d, "results/opes_highbarrier/" + rel)
        cv = Path(rel).name.split("_")[0]
        r, s = analyse(dict(family="opes7", cv=cv, pace=1000, n_per_set=50, rep=0, source="highbarrier"), d); rows.append(r); scans += s
    for rel, d in iter_json(dataset("fits.flooding_subsets"), r"N\d+_rep\d+\.json$"):
        d = full_data_view(d, "subsampling/flooding_subsets/" + rel)
        parts = Path(rel).parts
        fam, cond = parts[-3], parts[-2]
        cv, pace = (cond.split("_pace") + [None])[:2]
        m = re.match(r"N(\d+)_rep(\d+)", parts[-1])
        r, s = analyse(dict(family=fam, cv=cv, pace=int(pace) if pace else np.nan, n_per_set=int(m.group(1)), rep=int(m.group(2)), source="subsets"), d)
        rows.append(r); scans += s
    df = pd.DataFrame(rows); sc = pd.DataFrame(scans)
    df.to_csv(ROOT / "analysis/flooding_set_selection.csv", index=False)
    sc.to_csv(ROOT / "analysis/flooding_sets_scan.csv", index=False)
    hb = df[df.source == "highbarrier"]
    if len(hb):
        print("7-barrier OPES ladders (5-17 kJ/mol):")
        print(hb[["cv", "n_sets", "n_screen", "n_conv", "excluded", "slope_all", "err_all", "gamma_all", "err_screen", "err_g1_all", "err_g1_lowest"]].round(2).to_string(index=False))
    full = df[df.source == "triasha"]
    print("Screening on the full-N fits (OPES per pace, VES per N):")
    print(full.groupby(["family", "cv"]).agg(fits=("n_sets", "size"), screened=("n_screen", lambda x: (x < df.loc[x.index, "n_sets"]).sum()),
                                             mean_n_screen=("n_screen", "mean"), mean_n_conv=("n_conv", "mean"),
                                             med_abs_err_all=("err_all", lambda x: x.abs().median()),
                                             med_abs_err_screen=("err_screen", lambda x: x.abs().median()),
                                             med_abs_err_conv=("err_conv", lambda x: x.abs().median()),
                                             med_slope=("slope_all", "median")).round(2).to_string())
    print("Screening on the subsampling sweeps:")
    sub = df[df.source == "subsets"]
    print(sub.groupby(["family", "cv"]).agg(fits=("n_sets", "size"), frac_screened=("n_screen", lambda x: (x < df.loc[x.index, "n_sets"]).mean()),
                                            med_abs_err_all=("err_all", lambda x: x.abs().median()),
                                            med_abs_err_screen=("err_screen", lambda x: x.abs().median()),
                                            med_abs_err_conv=("err_conv", lambda x: x.abs().median())).round(2).to_string())
    # figure: error vs number of lowest-acceleration sets retained (full-N fits)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, VERM = "#0072B2", "#D55E00"
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex="row", sharey=True)
    for j, cv in enumerate(["Q", "RMSD", "Rg", "Ree"]):
        for i, fam in enumerate(("opes", "ves")):
            ax = axes[i, j]
            g = sc[(sc.family == fam) & (sc.cv == cv) & (sc.n_per_set == (50 if fam == "opes" else 200))]
            for key, gg in g.groupby(["pace"] if fam == "opes" else ["n_per_set"]):
                gg = gg.sort_values("n")
                ax.plot(gg.n, gg.err_eatrf, "-o", color=BLUE, ms=3, lw=0.8, alpha=0.6)
                ax.plot(gg.n, gg.err_g1, "-s", color=VERM, ms=3, lw=0.8, alpha=0.6)
            ax.axhspan(-1, 1, color="#7f7f7f", alpha=0.12, lw=0); ax.axhline(0, color="#7f7f7f", lw=0.8)
            ax.set_title(f"{cv}  ({'OPES, one curve per pace' if fam == 'opes' else 'VES, N = 200'})", fontsize=9)
            ax.grid(True, alpha=0.25, lw=0.5)
    axes[0, 0].plot([], [], "-o", color=BLUE, label="EATR-flooding"); axes[0, 0].plot([], [], "-s", color=VERM, label="flooding γ=1 (mean of sets)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("number of lowest-acceleration sets retained")
    axes[0, 0].set_ylabel("ln k_est − ln k_ref"); axes[1, 0].set_ylabel("ln k_est − ln k_ref")
    fig.suptitle("Set-selection scan (JCTC 2026 Fig. 5 analysis) for every flooding ladder", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig4_flooding_sets_retained.png", dpi=160)
    print("wrote figures/fig4_flooding_sets_retained.png")


if __name__ == "__main__":
    main()
