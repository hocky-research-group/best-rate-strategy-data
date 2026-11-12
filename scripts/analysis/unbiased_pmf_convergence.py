"""Convergence of the unbiased PMF: basin populations against trajectory length.

Usage (torch): module load anaconda3/2025.06
               python3 unbiased_pmf_convergence.py <cvs.colvar> <out.csv>

The PMF used throughout this work comes from one long unbiased trajectory, so its
statistical weight is set by the number of times that trajectory crosses between the
basins, not by the number of frames. This script counts those crossings and follows the
free energy of the folded basin relative to the unfolded one as the trajectory is
extended, which is the same test applied to the biased temperature scan in
fes_temperature_scan.py.

State is assigned by hysteresis: a frame belongs to the folded state once q rises above
0.65 and to the unfolded state once q falls below 0.35, keeping its previous label in
between, so that recrossing the boundary of a single basin is not counted as a transition.

Reported per window: the gap from the populations of the two basins, the number of
completed transitions, and the block standard error from splitting the trajectory into
blocks each holding about ten transitions.

Sanity check: the frames are 100 ps apart and the unfolding rate is 1.54 us^-1, so a
120 us trajectory must contain of order 180 unfolding events; far fewer would mean the
file is not what it is taken to be.
"""
import sys

import numpy as np

KT = 0.0083144626 * 312.0  # kJ/mol at 312 K
FRAME_PS = 100.0
FOLDED_MIN_Q = 0.65
UNFOLDED_MAX_Q = 0.35


def label_states(q: np.ndarray) -> np.ndarray:
    """Assign +1 (folded), -1 (unfolded) to every frame by hysteresis.

    Args:
        q (np.ndarray): fraction of native contacts per frame.
    """
    s = np.zeros(len(q), dtype=np.int8)
    s[q > FOLDED_MIN_Q] = 1
    s[q < UNFOLDED_MAX_Q] = -1
    # carry the last decided label forward across the barrier region
    idx = np.nonzero(s)[0]
    if len(idx) == 0:
        raise SystemExit("no frame is in either basin")
    out = np.empty(len(q), dtype=np.int8)
    out[: idx[0]] = s[idx[0]]
    fill = np.searchsorted(idx, np.arange(len(q)), side="right") - 1
    fill[fill < 0] = 0
    out[idx[0]:] = s[idx][fill[idx[0]:]]
    return out


def gap_and_transitions(state: np.ndarray) -> tuple[float, int]:
    """Folded-minus-unfolded free energy in kT and the number of transitions.

    Args:
        state (np.ndarray): +1/-1 labels from label_states.
    """
    nf, nu = int((state == 1).sum()), int((state == -1).sum())
    gap = np.nan if nf == 0 or nu == 0 else -np.log(nf / nu)
    return gap, int((np.diff(state) != 0).sum())


def main(colvar: str, out: str) -> None:
    """Write the gap, transition count and block error against trajectory length.

    Args:
        colvar (str): PLUMED COLVAR with a q column.
        out (str): output CSV.
    """
    head = open(colvar).readline().split()[2:]
    a = np.loadtxt(colvar, comments="#")
    q = a[:, head.index("q")]
    state = label_states(q)
    total_gap, total_tr = gap_and_transitions(state)
    print(f"{len(q)} frames = {len(q) * FRAME_PS / 1e6:.1f} us, {total_tr} transitions, "
          f"gap {total_gap:+.2f} kT ({total_gap * KT:+.2f} kJ/mol)")

    # block error: blocks holding about ten transitions each
    nblocks = max(2, total_tr // 10)
    blocks = np.array_split(state, nblocks)
    bg = np.array([gap_and_transitions(b)[0] for b in blocks], dtype=float)
    ok = np.isfinite(bg)
    sem = float(np.std(bg[ok], ddof=1) / np.sqrt(ok.sum()))
    print(f"{nblocks} blocks of {len(q) // nblocks * FRAME_PS / 1e3:.1f} ns: "
          f"gap {np.nanmean(bg):+.2f} +- {sem:.2f} kT (block standard error), "
          f"{ok.sum()} blocks visited both basins")

    lines = ["frac,t_us,gap_kT,gap_kJmol,transitions,folded_fraction"]
    for f in np.arange(0.1, 1.001, 0.05):
        n = max(1, int(f * len(q)))
        g, tr = gap_and_transitions(state[:n])
        lines.append(f"{f:.2f},{n * FRAME_PS / 1e6:.2f},{g:.3f},{g * KT:.3f},{tr},"
                     f"{(state[:n] == 1).mean():.4f}")
        print(lines[-1], flush=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unbiased_pmf_convergence.csv")
