"""OPES ladder up to 17 kJ/mol: where the gamma = 1 estimate breaks and EATRf does not.

Usage:
    uv run python scripts/plot_highbarrier_ladder.py
Input : data/results/opes_highbarrier/<CV>_pace1000_barriers5-17.json
Output: figures/fig7_opes_highbarrier_ladder.png
Left : ln k_obs against ln<e^{beta V}> per set (labels: BARRIER in kJ/mol), with the
       least-squares line through the five lowest sets; a set falling below that
       line is the overbiasing signature of JCTC 2026 Sec. 4.2.
Right: per-set gamma = 1 estimate and the EATRf estimate with n lowest sets.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as _paths
from paths import dataset
import numpy as np

from flooding_io import full_data_view

ROOT = _paths.ROOT
L6 = np.log(1e6)
REF = json.loads((dataset("reference_rate") / "reference_rate.json").read_text())["ln_k_mle_per_us"]
COL = {"Q": "#0072B2", "Ree": "#D55E00"}
BLUE, VERM = "#0072B2", "#D55E00"


def main() -> None:
    """Draw the two-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cv in ("Q", "Ree"):
        # rel stays relative to the canonical data layout because it is also the key of the
        # refit index; only the file itself is resolved through the layout in use
        rel = f"results/opes_highbarrier/{cv}_pace1000_barriers5-17.json"
        src = dataset("runs.highbarrier") / f"{cv}_pace1000_barriers5-17.json"
        d = full_data_view(json.loads(src.read_text()), rel, strict=True)
        sets = sorted(d["set_reports"], key=lambda s: s["ln_exp_beta_v"])
        x = np.array([s["ln_exp_beta_v"] for s in sets]); y = np.array([s["log_k_obs"] for s in sets]) - L6
        lab = [int(s["barrier"]) for s in sets]
        ax = axes[0]
        ax.plot(x, y, "o", color=COL[cv], ms=6, label=f"{cv}: EATRf γ = {d['gamma']:.2f}")
        for xi, yi, li in zip(x, y, lab):
            ax.annotate(str(li), (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)
        p = np.polyfit(x[:5], y[:5], 1)
        xx = np.linspace(x.min() - 0.2, x.max() + 0.2, 10)
        ax.plot(xx, np.polyval(p, xx), "-", color=COL[cv], lw=0.8, alpha=0.7, label=f"{cv}: fit to barriers 5-13, slope {p[0]:.2f}")
        ax = axes[1]
        ax.plot(lab, y - x - REF, "s--", color=COL[cv], ms=5, lw=0.8, label=f"{cv}: γ = 1 per set")
        ca = d["flooding_diagnostics"]["convergence_analysis"]
        ax.plot([lab[n - 1] for n in ca["n_sets"]], np.array(ca["logk0"]) - L6 - REF, "o-", color=COL[cv], ms=5, lw=1.6,
                label=f"{cv}: EATRf with sets up to this barrier")
    axes[0].set_xlabel("ln <e^{βV}>  (acceleration)"); axes[0].set_ylabel("ln k_obs (μs⁻¹)")
    axes[0].set_title("OPES ladder, pace 1000: observed rate vs acceleration", fontsize=9)
    axes[1].axhspan(-1, 1, color="#7f7f7f", alpha=0.12, lw=0); axes[1].axhline(0, color="#7f7f7f", lw=0.8)
    axes[1].set_xlabel("highest BARRIER included (kJ/mol); apparent barrier along Q = 15.4"); axes[1].set_ylabel("ln k_est − ln k_ref")
    axes[1].set_title("γ = 1 per set vs EATR-flooding over the lowest n sets", fontsize=9)
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / "figures/fig7_opes_highbarrier_ladder.png", dpi=160)
    print("wrote figures/fig7_opes_highbarrier_ladder.png")


if __name__ == "__main__":
    main()
