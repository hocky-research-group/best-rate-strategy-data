"""The growth law in the b -> infinity limit, from the untempered conditions.

Usage:
    uv run python scripts/analyze_untempered_growth_law.py

Input : analysis/untempered_growth_law_runs.csv (fill and escape time per run, extracted on
        torch by untempered_growth_law.sbatch) and analysis/growth_law_conditions.csv for
        the basin widths and the tempered grid this is compared against.
Output: analysis/untempered_growth_law.csv, one row per (CV, height, pace).

Tempering makes the growth law a logarithm,
    V_n = (b - 1) kT ln(1 + n h_eff / ((b - 1) kT)),
which flattens as the bias accumulates. Without tempering the hills never shrink and the
same law becomes V_n = n h_eff, so the fill divided by the hill count is the per-hill
increment directly, with no logarithm to invert and no bias factor in the way.

That makes these conditions a sharper test of the increment than the tempered grid, because
the two candidate forms disagree most here. Equation 11's increment,
h0 sigma / sqrt(sigma^2 + 2 b sigma_A^2), vanishes as b grows and predicts no fill at all.
The unbroadened increment, h0 sigma / sqrt(sigma^2 + 2 sigma_A^2), does not depend on b, so
it predicts the same value here as on the grid. The tempered grid already prefers the
unbroadened form, with fitted increments that move by a factor of 1.4 to 2.6 across
b = 2 to 20 where equation 11 requires a factor of 3; if the untempered conditions return
the same increment again, that is the b axis closed at its far end.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import paths as _paths
ROOT = _paths.ROOT
DT_PS = 0.01
SIGMA = {"Q": 0.02, "RMSD": 0.02, "Rg": 0.02, "Ree": 0.06}


def basin_widths() -> dict:
    """Unbiased basin width along each CV, as used by the tempered growth-law test."""
    g = pd.read_csv(ROOT / "analysis/growth_law_conditions.csv")
    return g.groupby("cv").sigma_A.first().to_dict()


def main() -> None:
    """Write the per-condition table and print the comparison with the tempered grid."""
    runs = pd.read_csv(ROOT / "analysis/untempered_growth_law_runs.csv")
    sig_a = basin_widths()
    runs["n_hills"] = runs.t_ps / (runs.pace * DT_PS)
    runs["h_eff_obs"] = runs.V_max_kJmol / runs.n_hills

    rows = []
    for (cv, h, p), g in runs.groupby(["cv", "height", "pace"]):
        s, sa = SIGMA[cv], sig_a[cv]
        pred = h * s / np.sqrt(s ** 2 + 2 * sa ** 2)
        rows.append(dict(cv=cv, height=h, pace=int(p), n_runs=len(g),
                         t_mean_ps=g.t_ps.mean(), n_hills=g.n_hills.median(),
                         V_obs_kJmol=g.V_max_kJmol.median(),
                         h_eff_obs=g.h_eff_obs.median(), h_eff_obs_over_h0=g.h_eff_obs.median() / h,
                         h_eff_unbroadened_over_h0=pred / h,
                         ratio=g.h_eff_obs.median() / pred))
    t = pd.DataFrame(rows).sort_values(["cv", "height", "pace"])
    t.to_csv(ROOT / "analysis/untempered_growth_law.csv", index=False)

    tg = pd.read_csv(ROOT / "analysis/growth_law_conditions.csv")
    print(f"{'CV':>5} {'sigma_A':>8} {'unbroadened':>12} {'untempered obs':>22} "
          f"{'tempered grid (b=2..20)':>26}")
    for cv, g in t.groupby("cv"):
        w = tg[tg.cv == cv].groupby("bf").h_eff_fit_over_h0.median()
        print(f"{cv:>5} {sig_a[cv]:>8.4f} {g.h_eff_unbroadened_over_h0.iloc[0]:>12.3f} "
              f"{g.h_eff_obs_over_h0.median():>10.3f} "
              f"[{g.h_eff_obs_over_h0.min():.3f}-{g.h_eff_obs_over_h0.max():.3f}]   "
              f"{w.loc[2]:.3f} -> {w.loc[20]:.3f}")
    print("\nobserved increment divided by the unbroadened prediction, by pace:")
    piv = t.pivot_table(index="cv", columns="pace", values="ratio")
    print(piv.round(2).to_string())
    print(f"\noverall median ratio {t.ratio.median():.3f} over {len(t)} conditions; "
          f"equation 11 predicts an increment that vanishes as b grows and so no fill at all")
    print("wrote analysis/untempered_growth_law.csv")


if __name__ == "__main__":
    main()
