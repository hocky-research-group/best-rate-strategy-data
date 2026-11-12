"""Free energy along Q at each temperature of the scan, and its convergence.

Usage (torch): module load anaconda3/2025.06
               python3 fes_temperature_scan.py <pmf_temperature dir> <out.csv>

Two estimates are reported per run. The primary one reweights the sampled trajectory with
exp(beta * metad.rbias), which converges faster and agrees between seeds; the hills sum is
kept as a cross-check. Both are computed for the first and second halves of the run, since
a difference between halves is what shows the estimate is still moving.

For the hills estimate it sums the recorded hills on a grid in Q. PLUMED writes
well-tempered hill heights already scaled by b/(b-1), so the sum of the hills is -F(Q)
up to a constant (the same convention the growth-law analysis uses, where the recorded
height is multiplied by (1 - 1/b) to recover the increment of the bias).

A second output file reports the free energy of the folded basin relative to the unfolded
one against sampling time, accumulated from the start of each run. That gap converges
faster than the profile between the basins, so a gap still drifting at the end of a run is
the clearest sign the free energy surface is not finished.

For each seed the FES is computed from the first half of the hills and from all of them,
so that the two can be compared; the barrier is the maximum of F between the folded
minimum (largest Q) and the unfolded minimum (smallest Q), the definition used for the
unbiased PMF.

Sanity check: at 312 K the barrier must agree with the 5.95 kT of the unbiased PMF; a
value off by a factor b/(b-1) = 20/19 would mean the height convention is the other one.
"""
import glob
import os
import sys

import numpy as np

KB = 0.0083144626  # kJ/(mol K)
FOLDED_MIN_Q = 0.65
UNFOLDED_MAX_Q = 0.35


def fes_from_hills(path, grid, frac=1.0):
    """Free energy on the grid from the first frac of the hills in a HILLS file.

    Args:
        path (str): HILLS file.
        grid (np.ndarray): Q values.
        frac (float): fraction of the hills to use.
    """
    rows = np.loadtxt(path, comments="#")
    rows = rows[: max(1, int(len(rows) * frac))]
    s, sigma, height = rows[:, 1], rows[:, 2], rows[:, 3]
    v = np.zeros_like(grid)
    for i in range(0, len(s), 20000):                      # in blocks, to bound memory
        j = slice(i, i + 20000)
        v += (height[j] * np.exp(-((grid[:, None] - s[j][None, :]) ** 2)
                                 / (2 * sigma[j][None, :] ** 2))).sum(axis=1)
    return -v, rows[-1, 0]


def barrier(grid, f, kT):
    """Barrier in kT between the folded and unfolded minima.

    Args:
        grid (np.ndarray): Q values.
        f (np.ndarray): free energy on the grid (kJ/mol).
        kT (float): thermal energy (kJ/mol).
    """
    folded = grid > FOLDED_MIN_Q
    unfolded = grid < UNFOLDED_MAX_Q
    i_f = np.argmin(np.where(folded, f, np.inf))
    i_u = np.argmin(np.where(unfolded, f, np.inf))
    lo, hi = sorted((i_f, i_u))
    top = f[lo:hi + 1].max()
    return (top - f[i_f]) / kT, grid[i_f], grid[i_u], (top - f[i_u]) / kT


def fes_reweighted(colvar, kT, edges, lo=0.0, hi=1.0):
    """Free energy on a histogram of Q, reweighted by the instantaneous bias.

    Args:
        colvar (str): COLVAR file with columns q and metad.rbias.
        kT (float): thermal energy (kJ/mol).
        edges (np.ndarray): histogram edges in Q.
        lo (float): first fraction of the run to use.
        hi (float): last fraction of the run to use.
    """
    head = open(colvar).readline().split()[2:]
    a = np.loadtxt(colvar, comments="#")
    a = a[int(lo * len(a)):int(hi * len(a))]
    q, rb = a[:, head.index("q")], a[:, head.index("metad.rbias")]
    w = np.exp(rb / kT)
    w /= w.max()
    h, _ = np.histogram(q, bins=edges, weights=w)
    with np.errstate(divide="ignore"):
        return -kT * np.log(h / h.sum()), a[-1, 0]


def basin_gap(colvar, kT, fracs):
    """Free energy of the folded basin relative to the unfolded one against sampling time.

    The gap between two states converges faster than the profile between them and is the
    cheaper convergence test: it is read from the reweighted populations of the two basins,
    accumulated from the start of the run to each fraction of it, so a value still drifting
    at the end of the run means the estimate is not finished.

    Args:
        colvar (str): COLVAR file with columns q and metad.rbias.
        kT (float): thermal energy (kJ/mol).
        fracs (iterable): fractions of the run at which to report the gap.

    Returns:
        list of (fraction, time in ps, gap in kJ/mol, folded weight fraction).
    """
    head = open(colvar).readline().split()[2:]
    a = np.loadtxt(colvar, comments="#")
    q, rb, t = a[:, head.index("q")], a[:, head.index("metad.rbias")], a[:, 0]
    w = np.exp((rb - rb.max()) / kT)
    folded, unfolded = q > FOLDED_MIN_Q, q < UNFOLDED_MAX_Q
    out = []
    for f in fracs:
        n = max(1, int(f * len(a)))
        wf, wu = w[:n][folded[:n]].sum(), w[:n][unfolded[:n]].sum()
        gap = np.nan if wf <= 0 or wu <= 0 else -kT * np.log(wf / wu)
        out.append((f, t[n - 1], gap, wf / w[:n].sum()))
    return out


def main(root, out):
    """Write one row per seed and half.

    Args:
        root (str): simulations/pmf_temperature directory.
        out (str): output CSV.
    """
    grid = np.linspace(0.0, 1.0, 201)
    edges = np.linspace(0.0, 1.0, 202)                       # centers coincide with grid
    lines = ["temperature_K,seed,estimate,window,t_ns,barrier_kT,barrier_kJmol,"
             "folded_min_q,unfolded_min_q,reverse_barrier_kT"]
    gaps = ["temperature_K,seed,frac,t_ns,gap_kJmol,gap_kT,folded_weight_fraction"]
    profiles = ["temperature_K,seed,estimate,window,q,F_kJmol,F_kT"]
    for d in sorted(glob.glob(os.path.join(root, "T*", "s*"))):
        hills = glob.glob(os.path.join(d, "*.hills"))
        colvar = glob.glob(os.path.join(d, "*.colvar.dat"))
        if not hills:
            continue
        T = float(os.path.basename(os.path.dirname(d))[1:])
        seed = os.path.basename(d)
        kT = KB * T

        def row(est, window, f, t_ps):
            """One output row from a free energy profile.

            Args:
                est (str): "reweight" or "hills".
                window (str): portion of the run used, e.g. "all" or "second half".
                f (np.ndarray): free energy on the grid (kJ/mol).
                t_ps (float): last time in the file (ps).
            """
            b_kT, qf, qu, rev = barrier(grid, f, kT)
            lines.append(f"{T:g},{seed},{est},{window},{t_ps / 1e3:.1f},{b_kT:.3f},"
                         f"{b_kT * kT:.3f},{qf:.3f},{qu:.3f},{rev:.3f}")
            print(lines[-1], flush=True)
            # the profile itself, zeroed at the folded minimum so that curves from
            # different temperatures can be drawn on one axis
            f0 = f - f[np.argmin(np.where(grid > FOLDED_MIN_Q, f, np.inf))]
            for x, y in zip(grid, f0):
                if np.isfinite(y):
                    profiles.append(f"{T:g},{seed},{est},{window},{x:.4f},{y:.4f},{y / kT:.4f}")

        # the reweighted estimate is the primary one; see the module docstring
        for window, (lo, hi) in (("all", (0.0, 1.0)), ("first half", (0.0, 0.5)),
                                 ("second half", (0.5, 1.0))):
            if colvar:
                f, t_ps = fes_reweighted(colvar[0], kT, edges, lo, hi)
                row("reweight", window, f, t_ps)
        for window, frac in (("all", 1.0), ("first half", 0.5)):
            f, t_ps = fes_from_hills(hills[0], grid, frac)
            row("hills", window, f, t_ps)
        if colvar:
            for f, t_ps, gap, pf in basin_gap(colvar[0], kT, np.arange(0.1, 1.01, 0.05)):
                gaps.append(f"{T:g},{seed},{f:.2f},{t_ps / 1e3:.1f},{gap:.3f},"
                            f"{gap / kT:.3f},{pf:.4f}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    prof_out = out.replace(".csv", "_profiles.csv")
    with open(prof_out, "w") as fh:
        fh.write("\n".join(profiles) + "\n")
    gap_out = out.replace(".csv", "_basin_gap.csv")
    with open(gap_out, "w") as fh:
        fh.write("\n".join(gaps) + "\n")
    print("wrote", out, prof_out, "and", gap_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fes_temperature_scan.csv")
