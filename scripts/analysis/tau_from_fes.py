"""Relaxation time of a CV in the folded basin from the shape of its free-energy profile.

Usage:
    uv run python scripts/tau_from_fes.py selftest
    uv run python scripts/tau_from_fes.py run <out.csv> <colvar> [<colvar> ...]
    uv run python scripts/tau_from_fes.py biased <out.csv> <rows> <colvar> [<colvar> ...]

Question: the design rule needs the relaxation time tau_rel of the biased CV, measured so
far from the autocorrelation of unbiased basin trajectories (scripts/cv_autocorrelation.py).
Can a free-energy profile supply it instead?

Theory (notes/tau_from_fes.tex). For overdamped diffusion along s with diffusion
coefficient D on a profile F(s), the integrated autocorrelation time of s is

    tau = (1 / var) * integral ds G(s)^2 / (D p(s)),   G(s) = integral_{-inf}^{s} (s' - <s>) p(s') ds',

with p ~ exp(-F / kT) and var its variance. For a harmonic basin this is var / D. The
profile fixes everything but D, which is dynamical: it is taken from the displacement over
a short lag, D(lag) = <(s(t + lag) - s(t))^2> / (2 lag), a local quantity that a slowly
changing bias does not alter to first order in the lag. Three estimates per lag:
    tau_harmonic  var / D(lag), var from the profile
    tau_ou        -lag / ln(1 - lag D(lag) / var), exact at any lag for a harmonic basin
    tau_shape     the integral above with D(lag)
They are compared with tau_int and tau_exp of the measured autocorrelation, and with a
two-component reading of it: C(t) = f_slow exp(-t / tau_slow) beyond a fast initial drop,
fitted on ln C between FIT_WINDOW lags, for which tau_int = f_slow tau_slow when the fast
part is unresolved.

`run` input: COLVARs of unbiased runs with fields "time q de d2 rg" (ps; de = R_ee,
rg = R_g), each truncated at its first excursion below q = 0.65 and dropped when shorter
than 5 ns, as in cv_autocorrelation.py. `biased` input: COLVARs of WT-MetaD runs, of which
the first <rows> rows are used (nearly unbiased when the bias is still small; the largest
bias in a segment is printed). The autocorrelation is averaged over segments, the
displacements and the histogram that stands for the profile are pooled, and the variance
is the mean over segments of the variance about the segment mean.
Output: one row per (cv, lag).

`selftest`: an exactly sampled Ornstein-Uhlenbeck series of known tau must return that tau
from tau_ou at every lag, and tau_shape of a Gaussian profile must equal var / D.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cv_autocorrelation import acf, times

LAGS = (1, 2, 3, 5, 10, 20, 50, 100)   # in samples
BASIN_Q = 0.65
FIT_WINDOW = (50.0, 500.0)             # ps, lags of the slow-component fit


def half_msd(segs: list[np.ndarray], lag: int) -> float:
    """Half the mean squared displacement over a lag, pooled over segments.

    Args:
        segs (list[np.ndarray]): time series, one per segment.
        lag (int): lag in samples.
    """
    d = np.concatenate([x[lag:] - x[:-lag] for x in segs])
    return float(0.5 * np.mean(d * d))


def tau_shape(s: np.ndarray, p: np.ndarray, diffusion: float) -> float:
    """Integrated autocorrelation time of s for diffusion on the profile -kT ln p.

    Args:
        s (np.ndarray): bin centres, uniformly spaced.
        p (np.ndarray): probability per bin (any normalization); empty bins are skipped.
        diffusion (float): diffusion coefficient, units of s^2 per time.
    """
    ds = s[1] - s[0]
    p = p / (p.sum() * ds)
    mean = np.sum(s * p) * ds
    var = np.sum((s - mean) ** 2 * p) * ds
    # G at the bin edges to the right of each bin; it vanishes at both ends of the support
    g = np.cumsum((s - mean) * p) * ds
    p_edge = 0.5 * (p[:-1] + p[1:])
    ok = p_edge > 0
    return float(np.sum(g[:-1][ok] ** 2 / p_edge[ok]) * ds / (diffusion * var))


def estimates(segs: list[np.ndarray], dt: float, lags: tuple[int, ...] = LAGS, bins: int = 60) -> pd.DataFrame:
    """The three profile-based estimates per lag, beside the measured correlation times.

    Args:
        segs (list[np.ndarray]): basin segments of one CV.
        dt (float): sampling interval (ps).
        lags (tuple[int, ...]): lags in samples.
        bins (int): bins of the histogram that stands for the profile.
    """
    nlag = min(4000, min(len(x) for x in segs) // 4)
    c = np.mean([acf(x, nlag) for x in segs], axis=0)
    t_exp, t_int = times(c, dt)
    t = np.arange(len(c)) * dt
    win = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1]) & (c > 0)
    slope, icpt = np.polyfit(t[win], np.log(c[win]), 1)
    f_slow, tau_slow = float(np.exp(icpt)), float(-1 / slope)
    hist, edges = np.histogram(np.concatenate([x - x.mean() for x in segs]), bins=bins)
    s = 0.5 * (edges[:-1] + edges[1:])
    var = float(np.mean([np.var(x) for x in segs]))
    rows = []
    for lag in [k for k in lags if k <= nlag]:
        d = half_msd(segs, lag) / (lag * dt)
        ratio = d * lag * dt / var
        rows.append(dict(lag_ps=lag * dt, sigma=np.sqrt(var), D=d, tau_harmonic_ps=var / d,
                         tau_ou_ps=-lag * dt / np.log(1 - ratio) if ratio < 1 else np.nan,
                         tau_shape_ps=tau_shape(s, hist.astype(float), d),
                         tau_int_ps=t_int, tau_exp_ps=t_exp, acf_at_lag=float(c[lag]),
                         f_slow=f_slow, tau_slow_ps=tau_slow, f_tau_slow_ps=f_slow * tau_slow))
    return pd.DataFrame(rows)


def selftest() -> None:
    """Known answers: an exact Ornstein-Uhlenbeck series and a Gaussian profile."""
    rng = np.random.default_rng(20261105)
    tau, sigma, dt, n = 300.0, 0.05, 10.0, 400_000
    a = np.exp(-dt / tau)
    x = np.empty(n)
    x[0] = sigma * rng.standard_normal()
    noise = sigma * np.sqrt(1 - a * a) * rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    e = estimates([x], dt)
    print(e.round(4).to_string(index=False))
    assert np.allclose(e.tau_ou_ps, tau, rtol=0.05), "tau_ou must return the OU time at every lag"
    assert abs(e.tau_int_ps.iloc[0] - tau) / tau < 0.1
    s = np.linspace(-6, 6, 2001)
    assert abs(tau_shape(s, np.exp(-0.5 * s * s), 2.0) - 0.5) < 1e-3, "Gaussian profile: tau = var / D"
    # the short-lag D of an OU process is sigma^2 / tau, so tau_shape at the shortest lag is tau
    assert abs(e.tau_shape_ps.iloc[0] - tau) / tau < 0.1
    print("selftest passed: OU tau", tau, "ps recovered; Gaussian profile gives var / D")


def run(out: str, colvars: list[str], rows: int | None) -> None:
    """Estimates for every CV recorded, from unbiased runs or from the start of biased ones.

    Args:
        out (str): CSV to write.
        colvars (list[str]): COLVAR paths.
        rows (int | None): None for unbiased runs (fields time q de d2 rg, truncated at the
            basin boundary); otherwise the number of leading rows kept of each biased run.
    """
    frames, bias = [], []
    for f in colvars:
        head = Path(f).read_text().splitlines()[0].split()[2:]
        d = pd.read_csv(f, sep=r"\s+", comment="#", names=head, nrows=rows)
        if rows is None:
            d = d.rename(columns={"de": "Ree", "rg": "Rg"})
            left = np.nonzero((d.q < BASIN_Q).to_numpy())[0]
            d = d.iloc[:left[0]] if len(left) else d
        else:
            bias.append(float(d["metad.bias"].max()))
        if len(d) * float(d.time.iloc[1] - d.time.iloc[0]) >= 5000:
            frames.append(d)
    dt = float(frames[0].time.iloc[1] - frames[0].time.iloc[0])
    lens = [len(d) for d in frames]
    print(f"{len(frames)} of {len(colvars)} segments kept, {min(lens) * dt / 1e3:.0f}-{max(lens) * dt / 1e3:.0f} ns, "
          f"stride {dt} ps" + (f", largest bias {max(bias):.2f} kJ/mol" if bias else ""))
    cvs = [c for c in ("q", "rmsd", "Ree", "Rg") if c in frames[0].columns]
    parts = [estimates([d[cv].to_numpy() for d in frames], dt).assign(cv=cv) for cv in cvs]
    df = pd.concat(parts)[["cv", "lag_ps", "sigma", "D", "acf_at_lag", "tau_harmonic_ps", "tau_ou_ps",
                           "tau_shape_ps", "tau_int_ps", "tau_exp_ps", "f_slow", "tau_slow_ps", "f_tau_slow_ps"]]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    df.to_csv(out, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "biased":
        run(sys.argv[2], sys.argv[4:], int(sys.argv[3]))
    else:
        run(sys.argv[2], sys.argv[3:], None)
