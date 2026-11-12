"""Free energy along R_ee at each temperature of the scan, reweighted from the Q-biased runs.

Usage (torch): module load anaconda3/2025.06
               python3 fes_temperature_scan_ree.py <pmf_temperature dir> <out.csv>

The scan runs bias Q (fes_temperature_scan.py reads the free energy along Q from them).
Every COLVAR row also records R_ee, so the same rows reweighted by exp(beta * metad.rbias)
give the free energy along R_ee and the width of the folded basin along it, which are the
two inputs the growth law needs to design a well-tempered condition along R_ee at that
temperature.

Definitions follow the unbiased PMF (pmf_from_unbiased_colvar.py): the folded basin is the
set of frames with Q > 0.65 and its width is the (weighted) standard deviation of R_ee over
those frames; the folded and unfolded minima are the lowest bins among those dominated by
folded and by unfolded frames (Q < 0.35), and the barrier is the highest bin between them.
The histogram uses the 100 bins of the unbiased R_ee PMF. Each quantity is reported for the
second half of every run and for the whole run; the second half is the estimate used.

Sanity check (printed): at 312 K the second-half barrier must be near the 5.92 kT of the
unbiased PMF and the width near 0.152 nm; the reweighted estimates are expected a little
low in the barrier, as they are along Q (6% after 2 us), and the width a little high.
"""
import glob
import os
import sys

import numpy as np

KB = 0.0083144626  # kJ/(mol K)
FOLDED_MIN_Q = 0.65
UNFOLDED_MAX_Q = 0.35
REE_EDGES = np.linspace(0.322690, 10.478959, 101)   # the 100 bins of pmf_ree.dat (0.1016 nm wide)
UNBIASED_312 = dict(barrier_kT=5.92, sigma_A=0.152)


def weighted_std(x, w):
    """Weighted standard deviation.

    Args:
        x (np.ndarray): values.
        w (np.ndarray): non-negative weights.
    """
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def profile(colvar, kT, lo=0.0, hi=1.0):
    """Reweighted free energy along R_ee and the folded-basin width, from part of a run.

    Args:
        colvar (str): COLVAR file with columns q, Ree and metad.rbias.
        kT (float): thermal energy (kJ/mol).
        lo (float): first fraction of the run to use.
        hi (float): last fraction of the run to use.

    Returns:
        dict with t_ns, sigma_A (weighted, nm), sigma_A_unweighted, barrier_kT, folded_min,
        barrier_pos, unfolded_min (nm), folded_weight_fraction, and the profile F_kT on
        the bin centers (nan where empty).
    """
    head = open(colvar).readline().split()[2:]
    a = np.loadtxt(colvar, comments="#")
    a = a[int(lo * len(a)):int(hi * len(a))]
    q, ree, rb = a[:, head.index("q")], a[:, head.index("Ree")], a[:, head.index("metad.rbias")]
    w = np.exp((rb - rb.max()) / kT)
    folded, unfolded = q > FOLDED_MIN_Q, q < UNFOLDED_MAX_Q
    h_all, _ = np.histogram(ree, bins=REE_EDGES, weights=w)
    h_unf, _ = np.histogram(ree[unfolded], bins=REE_EDGES, weights=w[unfolded])
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -kT * np.log(h_all / h_all.sum())
        p_unf = h_unf / h_all
    f[~np.isfinite(f)] = np.nan
    centers = 0.5 * (REE_EDGES[1:] + REE_EDGES[:-1])
    fold_bins = np.isfinite(f) & (p_unf < 0.5)
    unf_bins = np.isfinite(f) & (p_unf >= 0.5)
    i_fold = int(np.nanargmin(np.where(fold_bins, f, np.nan)))
    i_unf = int(np.nanargmin(np.where(unf_bins, f, np.nan)))
    f = f - f[i_fold]
    lo_i, hi_i = sorted((i_fold, i_unf))
    i_bar = lo_i + int(np.nanargmax(f[lo_i:hi_i + 1]))
    return dict(t_ns=a[-1, 0] / 1e3, sigma_A=weighted_std(ree[folded], w[folded]),
                sigma_A_unweighted=float(ree[folded].std()), barrier_kT=float(f[i_bar] / kT),
                folded_min=float(centers[i_fold]), barrier_pos=float(centers[i_bar]),
                unfolded_min=float(centers[i_unf]), folded_weight_fraction=float(w[folded].sum() / w.sum()),
                F_kT=f / kT, centers=centers)


def main(root, out):
    """Write one row per seed and window, and the profiles beside it.

    Args:
        root (str): simulations/pmf_temperature directory.
        out (str): output CSV.
    """
    cols = ["t_ns", "sigma_A", "sigma_A_unweighted", "barrier_kT", "folded_min", "barrier_pos",
            "unfolded_min", "folded_weight_fraction"]
    lines = ["temperature_K,seed,window," + ",".join(cols)]
    profiles = ["temperature_K,seed,window,ree,F_kT"]
    check = []
    for d in sorted(glob.glob(os.path.join(root, "T*", "s*"))):
        T = int(os.path.basename(os.path.dirname(d))[1:])
        seed = os.path.basename(d)
        colvar = glob.glob(os.path.join(d, "*.colvar.dat"))[0]
        kT = KB * T
        for window, lo in (("second half", 0.5), ("all", 0.0)):
            r = profile(colvar, kT, lo=lo)
            lines.append(f"{T},{seed},{window}," + ",".join(f"{r[c]:.4f}" for c in cols))
            profiles += [f"{T},{seed},{window},{x:.4f},{y:.4f}" for x, y in zip(r["centers"], r["F_kT"])
                         if np.isfinite(y)]
            if T == 312 and window == "second half":
                check.append((seed, r["barrier_kT"], r["sigma_A"]))
            print(f"{T} K {seed} {window:11s} barrier {r['barrier_kT']:.2f} kT at {r['barrier_pos']:.2f} nm, "
                  f"folded min {r['folded_min']:.2f} nm, sigma_A {r['sigma_A']:.3f} nm "
                  f"(unweighted {r['sigma_A_unweighted']:.3f}), t {r['t_ns']:.0f} ns")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    prof_out = out.replace(".csv", "_profiles.csv")
    with open(prof_out, "w") as fh:
        fh.write("\n".join(profiles) + "\n")
    for seed, bar, sa in check:
        print(f"check 312 K {seed}: barrier {bar:.2f} kT (unbiased {UNBIASED_312['barrier_kT']}), "
              f"sigma_A {sa:.3f} nm (unbiased {UNBIASED_312['sigma_A']})")
    print("wrote", out, "and", prof_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fes_temperature_scan_ree.csv")
