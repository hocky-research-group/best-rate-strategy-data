"""PMF along each CV from the long unbiased protein G trajectory.

Usage (torch login node):
    module load anaconda3/2025.06
    python pmf_from_unbiased_colvar.py <cvs.colvar> <outdir>

Input : PLUMED COLVAR with header "#! FIELDS time q rmsd rg ree". The time column is
        a frame index and the frames are 100 ps apart, so the 1.2 M frames are 120 us;
        the histogram does not depend on the stride, but the transition count that
        sets the statistical weight of the result does.
Output: <outdir>/pmf_<cv>.dat  (two columns: cv value, F in kJ/mol, zero at
        the folded minimum) and <outdir>/pmf_summary.json with, per CV, the
        folded-minimum position, the basin standard deviation sigma_A (frames
        with q > 0.65, the OPES excluded region used for Q), the barrier-top
        position and the apparent barrier height in kJ/mol and kT.
The unfolded state is defined by the committor criterion q < 0.35 used to stop
every rate simulation; the unfolded minimum of each PMF is the lowest bin in
which more than half of the frames are unfolded, and the barrier is the PMF
maximum between the two minima. Sanity check: the barrier along Q must fall
between q = 0.35 and the folded minimum near q = 0.9.
"""
import json
import sys
from pathlib import Path

import numpy as np

KT = 0.0083144626 * 312.0  # kJ/mol at 312 K
NBINS = {"q": 100, "rmsd": 100, "rg": 100, "ree": 100}
FOLDED_MASK_Q = 0.65
UNFOLDED_Q = 0.35


def main(colvar: str, outdir: str) -> None:
    """Histogram every CV column and write PMFs plus a summary.

    Args:
        colvar (str): path to the COLVAR file.
        outdir (str): directory for outputs (created if missing).
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(colvar) as fh:
        header = fh.readline().split()[2:]
    data = np.loadtxt(colvar, comments="#")
    cols = {name: data[:, i] for i, name in enumerate(header)}
    folded = cols["q"] > FOLDED_MASK_Q
    unfolded = cols["q"] < UNFOLDED_Q  # committor-based unfolded state
    summary = {"n_frames": int(len(data)), "kT_kJmol": KT,
               "folded_mask": f"q>{FOLDED_MASK_Q}", "unfolded_mask": f"q<{UNFOLDED_Q}",
               "note": "time column is a frame index; frames are 100 ps apart (120 us total)"}
    for cv in ("q", "rmsd", "rg", "ree"):
        x = cols[cv]
        hist, edges = np.histogram(x, bins=NBINS[cv], density=True)
        hist_unf, _ = np.histogram(x[unfolded], bins=edges, density=False)
        hist_all, _ = np.histogram(x, bins=edges, density=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_unf = hist_unf / hist_all
        centers = 0.5 * (edges[1:] + edges[:-1])
        with np.errstate(divide="ignore"):
            F = -KT * np.log(hist)
        F[~np.isfinite(F)] = np.nan
        fmean, fstd = x[folded].mean(), x[folded].std()
        # folded minimum: PMF minimum among bins dominated by folded frames
        fold_bins = np.isfinite(F) & (p_unf < 0.5)
        i_fold = int(np.nanargmin(np.where(fold_bins, F, np.nan)))
        F = F - F[i_fold]
        np.savetxt(out / f"pmf_{cv}.dat", np.column_stack([centers, F, p_unf]),
                   header=f"{cv}  F_kJmol(zero at folded min)  p_unfolded(q<{UNFOLDED_Q}); {len(data)} frames, {NBINS[cv]} bins")
        # unfolded minimum: PMF minimum among bins dominated by unfolded frames
        unf_bins = np.isfinite(F) & (p_unf >= 0.5)
        if not unf_bins.any():
            summary[cv] = {"error": "no bins dominated by unfolded frames"}
            continue
        i_unf = int(np.nanargmin(np.where(unf_bins, F, np.nan)))
        lo, hi = sorted([i_fold, i_unf])
        seg = F[lo:hi + 1]
        i_bar = lo + int(np.nanargmax(seg))
        summary[cv] = {
            "folded_min": float(centers[i_fold]),
            "basin_mean": float(fmean),
            "basin_sigma": float(fstd),
            "unfolded_min": float(centers[i_unf]),
            "F_unfolded_min_kJmol": float(F[i_unf]),
            "barrier_pos": float(centers[i_bar]),
            "barrier_kJmol": float(F[i_bar]),
            "barrier_kT": float(F[i_bar] / KT),
            "p_unfolded_at_barrier": float(p_unf[i_bar]),
        }
    (out / "pmf_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
