"""Free energy along Q at every temperature of the scan, on one axis.

Usage:
    uv run --group figures python scripts/plot_pmf_temperature.py [out.pdf]

Input : analysis/fes_temperature_scan_profiles.csv (reweighted, second half of each run,
        zeroed at the folded minimum) and data/heuristics/pmf_unbiased/pmf_q.dat.
Output: analysis/pmf_temperature.pdf, two panels, the same curves in kJ/mol and in kT.

The unbiased PMF at 312 K is drawn with the scan curves as the check on the method: the
312 K scan curve must lie on it. Seeds are drawn separately rather than averaged, so that
the disagreement between them at low temperature is visible rather than hidden.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import dataset
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial"], "pdf.fonttype": 42,
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
KB = 0.0083144626
COLORS = {270: "#1f3b73", 285: "#3f7fbf", 300: "#e08a1e", 312: "#b02418"}
Q_LO = 0.15          # below this the unfolded wall dominates the scale


def main(out: str) -> None:
    """Draw the scan profiles in energy and in thermal units.

    Args:
        out (str): output PDF path.
    """
    d = pd.read_csv("analysis/fes_temperature_scan_profiles.csv")
    d = d[(d.estimate == "reweight") & (d.window == "second half")]
    q, f_kj = np.loadtxt(dataset("pmf_unbiased") / "pmf_q.dat", usecols=(0, 1), unpack=True)
    f_kj = f_kj - f_kj[np.argmin(np.where(q > 0.65, f_kj, np.inf))]

    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.6), constrained_layout=True)
    for ax, col, unit, ref in ((axes[0], "F_kJmol", "kJ/mol", f_kj),
                               (axes[1], "F_kT", r"$k_{\rm B}T$", f_kj / (KB * 312))):
        ax.plot(q, ref, color="0.35", lw=1.6, ls=(0, (4, 2)), zorder=5,
                label="unbiased, 312 K")
        for T, g in d.groupby("temperature_K"):
            for i, (seed, s) in enumerate(g.groupby("seed")):
                ax.plot(s.q, s[col], color=COLORS[T], lw=1.1, alpha=0.9,
                        ls="-" if i == 0 else (0, (1, 1)),
                        label=f"{T:g} K" if i == 0 else None)
        ax.set_xlabel("$Q$")
        ax.set_ylabel(f"$F$ ({unit})")
        ax.set_xlim(Q_LO, 1.0)
    # the wall below Q = 0.15 runs off to 90 kJ/mol at the lowest temperature and would
    # flatten everything else; the barriers and both minima are what the scan is for
    hi = d[d.q >= Q_LO].F_kJmol.max()
    axes[0].set_ylim(-2, hi * 1.30)
    axes[1].set_ylim(-2 / (KB * 312), hi * 1.30 / (KB * 312))
    axes[1].legend(frameon=False, ncol=1, loc="upper right", handlelength=1.6,
                   borderaxespad=0.2, labelspacing=0.3)
    # the line style carries the seed, which is the point of the figure at low
    # temperature, so it needs its own key rather than a note in the caption
    seed_keys = [Line2D([], [], color="0.25", lw=1.1, ls="-", label="seed 1"),
                 Line2D([], [], color="0.25", lw=1.1, ls=(0, (1, 1)), label="seed 2")]
    axes[0].legend(handles=seed_keys, frameon=False, loc="upper right",
                   handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
    fig.savefig(out)

    # the check that licenses the rest: the 312 K scan curve against the unbiased PMF
    s = d[(d.temperature_K == 312)]
    for seed, g in s.groupby("seed"):
        y = np.interp(q, g.q, g.F_kJmol)
        m = (q > 0.35) & (q < 0.97)
        print(f"312 K {seed}: mean |difference| from the unbiased PMF over 0.35 < Q < 0.97, "
              f"{np.abs(y[m] - f_kj[m]).mean():.2f} kJ/mol "
              f"({np.abs(y[m] - f_kj[m]).mean() / (KB * 312):.2f} kT)")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "analysis/pmf_temperature.pdf")
