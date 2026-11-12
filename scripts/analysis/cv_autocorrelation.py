"""Relaxation time of each CV in the folded basin, from nearly unbiased trajectory segments.

Usage:
    uv run --group figures python scripts/cv_autocorrelation.py <dir with *.dat> [out.csv]

Input: COLVAR segments written every 1 ps by the WT-MetaD runs at the slowest pace and
the smallest hill height (bias factor 10, h0 = 0.1 kJ/mol, pace 10^5 steps). Over the
first 10 ns such a run deposits at most ten hills of 0.1 kJ/mol, that is below 0.4 kT in
total, so the segment is effectively unbiased; the check below reports the bias reached.

For each CV the normalized autocorrelation function C(t) of the fluctuations about the
segment mean is averaged over segments. Two times are reported:
    tau_exp  the time at which C falls to 1/e (linear interpolation between samples)
    tau_int  the integrated correlation time, sum of C(t) dt up to the first zero crossing
Both are reported for the segment-averaged C; the spread of the per-segment estimates is
reported alongside, and a warning is printed when the correlation time exceeds 5% of the
segment length, where the estimate is no longer resolved.
Sanity check: C(0) = 1 by construction, and tau_exp must exceed the 1 ps sampling stride,
otherwise the stride is too coarse to resolve the decay (printed as a warning).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CVS = ["q", "rmsd", "Rg", "Ree"]


def acf(x: np.ndarray, nlag: int) -> np.ndarray:
    """Normalized autocorrelation of a series, lags 0..nlag.

    Args:
        x (np.ndarray): time series.
        nlag (int): largest lag in samples.
    """
    y = x - x.mean()
    n = len(y)
    full = np.correlate(y, y, mode="full")[n - 1:n + nlag]
    return full / full[0]


def times(c: np.ndarray, dt: float) -> tuple[float, float]:
    """Decay time to 1/e and integrated correlation time of an autocorrelation function.

    Args:
        c (np.ndarray): autocorrelation at lags 0, dt, 2dt, ...
        dt (float): sampling interval.
    """
    below = np.nonzero(c < 1 / np.e)[0]
    if len(below) == 0:
        t_exp = np.nan
    else:
        i = below[0]
        c0, c1 = c[i - 1], c[i]
        t_exp = dt * (i - 1 + (c0 - 1 / np.e) / (c0 - c1))
    zero = np.nonzero(c <= 0)[0]
    stop = zero[0] if len(zero) else len(c)
    return t_exp, float(np.sum(c[:stop]) * dt)


def main(indir: str, out: str, unbiased: bool = False) -> None:
    """Average the autocorrelation over segments and write one row per CV.

    Args:
        indir (str): directory of COLVAR segments.
        out (str): output CSV path.
        unbiased (bool): segments come from the unbiased first-passage runs, whose fields
            are "time q de d2 rg" (de is the end-to-end distance, rg the radius of
            gyration, and RMSD is not recorded). Each run is truncated at its first
            excursion below q = 0.65, the folded-basin boundary, so that only residence
            in the reactant basin enters the average.
    """
    files = sorted(Path(indir).glob("*.dat"))
    if not files:
        raise SystemExit(f"no segments in {indir}")
    frames, bias = [], []
    for f in files:
        head = f.read_text().splitlines()[0].split()[2:]
        d = pd.read_csv(f, sep=r"\s+", comment="#", names=head)
        if unbiased:
            d = d.rename(columns={"de": "Ree", "rg": "Rg"})
            left = np.nonzero((d.q < 0.65).to_numpy())[0]
            if len(left):
                d = d.iloc[:left[0]]
            if len(d) < 500:            # shorter than 5 ns at a 10 ps stride
                continue
            frames.append(d)
            bias.append(0.0)
            continue
        frames.append(d)
        bias.append(d["metad.bias"].max())
    dt = float(frames[0].time.iloc[1] - frames[0].time.iloc[0])
    kT = 0.0083144626 * 312.0
    lens = [len(d) for d in frames]
    print(f"{len(frames)} segments, {min(lens)}-{max(lens)} rows ({min(lens) * dt / 1e3:.0f}-"
          f"{max(lens) * dt / 1e3:.0f} ns), stride {dt} ps; "
          f"largest bias in a segment {max(bias):.2f} kJ/mol = {max(bias) / kT:.2f} kT")
    nlag = min(4000, min(len(d) for d in frames) // 4)
    rows = []
    for cv in [c for c in CVS if c in frames[0].columns]:
        per = [times(acf(d[cv].to_numpy(), nlag), dt) for d in frames]   # one estimate per segment
        cs = np.mean([acf(d[cv].to_numpy(), nlag) for d in frames], axis=0)
        assert abs(cs[0] - 1) < 1e-12
        t_exp, t_int = times(cs, dt)
        exp_each = np.array([t[0] for t in per], dtype=float)
        int_each = np.array([t[1] for t in per], dtype=float)
        if not np.isfinite(t_exp) or t_exp <= dt:
            print(f"WARNING: {cv} decays within one sampling interval ({dt} ps); "
                  "the stride is too coarse to resolve it")
        if t_int > 0.05 * np.median(lens) * dt:
            print(f"WARNING: {cv} correlation time {t_int:.0f} ps exceeds 5% of the "
                  f"{np.median(lens) * dt / 1e3:.0f} ns median segment; it is poorly resolved")
        rows.append(dict(cv=cv, tau_exp_ps=t_exp, tau_int_ps=t_int,
                         tau_int_sd_ps=np.nanstd(int_each, ddof=1),
                         tau_int_sem_ps=np.nanstd(int_each, ddof=1) / np.sqrt(np.isfinite(int_each).sum()),
                         tau_exp_sd_ps=np.nanstd(exp_each, ddof=1),
                         n_segments=len(frames), stride_ps=dt,
                         rows_per_segment=int(np.median(lens)),
                         segment_ns=float(np.median(lens)) * dt / 1e3))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--unbiased"]
    main(args[0], args[1] if len(args) > 1 else "analysis/cv_autocorrelation.csv",
         unbiased="--unbiased" in sys.argv)
