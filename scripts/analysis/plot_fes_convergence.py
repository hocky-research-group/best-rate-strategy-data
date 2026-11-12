"""Convergence of the temperature-scan free energy: basin gap against sampling time.

Usage:
    uv run --group figures python scripts/plot_fes_convergence.py \
        [analysis/fes_temperature_scan_basin_gap.csv] [analysis/fes_convergence.pdf]

Input is written by scripts/fes_temperature_scan.py. Each curve is the free energy of the
folded basin relative to the unfolded one, from the reweighted populations accumulated
from the start of a run, so a curve that has flattened and lies on its partner seed is a
converged estimate and one still falling is not.

Sanity check: at the folding temperature the two basins are equally populated and the gap
is zero; the 312 K curves must therefore sit near zero, and the gap must deepen as the
temperature falls.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial"], "pdf.fonttype": 42,
                     "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
COLORS = {270: "#1f3b73", 285: "#3f7fbf", 300: "#e08a1e", 312: "#b02418"}


def main(src: str, out: str) -> None:
    """Draw the basin gap against sampling time, one colour per temperature.

    Args:
        src (str): basin-gap CSV from fes_temperature_scan.py.
        out (str): output PDF.
    """
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(3.37, 2.4), constrained_layout=True)
    for T, g in d.groupby("temperature_K"):
        for i, (seed, s) in enumerate(g.groupby("seed")):
            ax.plot(s.t_ns / 1e3, s.gap_kT, color=COLORS[T], lw=1.2,
                    ls="-" if i == 0 else "--",
                    label=f"{T:g} K" if i == 0 else None)
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"sampling time ($\mu$s)")
    ax.set_ylabel(r"$F_{\rm folded} - F_{\rm unfolded}$ ($k_{\rm B}T$)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.savefig(out)
    last = d[d.frac == 1.0].pivot_table(index="temperature_K", columns="seed", values="gap_kT")
    print(last.round(2).to_string())
    print("seed difference (kT):", (last.s1 - last.s2).abs().round(2).to_dict())
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "analysis/fes_temperature_scan_basin_gap.csv",
         sys.argv[2] if len(sys.argv) > 2 else "analysis/fes_convergence.pdf")
