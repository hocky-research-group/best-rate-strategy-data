"""Recompute the unbiased protein G unfolding rate from first-passage times.

Usage:
    uv run python scripts/reference_rate.py

Input : data/reference_rate/unbiased_first_passage_times_ps.dat
        (columns: run, fpt_ps, q_at_commit; extracted from nm3787 p.log files)
Output: data/reference_rate/reference_rate.json

Every run transitioned, so the maximum-likelihood rate is N / sum(t). A
least-squares fit of the empirical CDF to 1 - exp(-k t) is reported as well,
matching the estimator used in Mazzaferro et al. JCTC 2024. Uncertainties are
bootstrap standard deviations and 2.5/97.5 percentiles over 2000 resamples.
Sanity check: the MLE must equal 1/mean(t), and the CDF-fit value should agree
with the published 1.4 +/- 0.1 us^-1.
"""
import json
import sys
from pathlib import Path

import paths as _paths
from paths import dataset
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest

ROOT = _paths.ROOT
INFILE = dataset("reference_rate") / "unbiased_first_passage_times_ps.dat"
OUTFILE = dataset("reference_rate") / "reference_rate.json"
SEED = 20260909
NBOOT = 2000


def load_times_us(path: Path) -> np.ndarray:
    """Return first-passage times in microseconds.

    Args:
        path (Path): whitespace table with columns run, fpt_ps, q_at_commit.
    """
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] == "NA":
            continue
        rows.append(float(parts[1]))
    return np.asarray(rows) / 1e6  # ps -> us


def cdf_fit_rate(t: np.ndarray) -> float:
    """Least-squares fit of the empirical CDF to 1 - exp(-k t).

    Args:
        t (np.ndarray): transition times (all uncensored), any unit.
    Returns the fitted rate in the inverse of that unit.
    """
    x = np.sort(t)
    y = np.arange(1, len(x) + 1) / len(x)
    k0 = len(t) / t.sum()
    popt, _ = curve_fit(lambda tt, k: 1 - np.exp(-k * tt), x, y, p0=[k0])
    return float(popt[0])


def main() -> None:
    """Compute MLE and CDF-fit rates with bootstrap uncertainties."""
    t = load_times_us(INFILE)
    n = len(t)
    k_mle = n / t.sum()
    assert np.isclose(k_mle, 1 / t.mean()), "MLE must equal 1/mean(t)"
    k_cdf = cdf_fit_rate(t)
    ks = kstest(t, "expon", args=(0, 1 / k_mle))
    rng = np.random.default_rng(SEED)
    boot_mle, boot_cdf = [], []
    for _ in range(NBOOT):
        tb = rng.choice(t, size=n, replace=True)
        boot_mle.append(np.log(n / tb.sum()))
        boot_cdf.append(np.log(cdf_fit_rate(tb)))
    boot_mle, boot_cdf = np.asarray(boot_mle), np.asarray(boot_cdf)
    out = {
        "n_runs": int(n),
        "mean_fpt_us": float(t.mean()),
        "k_mle_per_us": float(k_mle),
        "ln_k_mle_per_us": float(np.log(k_mle)),
        "ln_k_mle_boot_std": float(boot_mle.std(ddof=1)),
        "ln_k_mle_boot_ci95": [float(np.percentile(boot_mle, 2.5)), float(np.percentile(boot_mle, 97.5))],
        "k_cdf_per_us": float(k_cdf),
        "ln_k_cdf_per_us": float(np.log(k_cdf)),
        "ln_k_cdf_boot_std": float(boot_cdf.std(ddof=1)),
        "ln_k_cdf_boot_ci95": [float(np.percentile(boot_cdf, 2.5)), float(np.percentile(boot_cdf, 97.5))],
        "ln_k_mle_per_s": float(np.log(k_mle * 1e6)),
        "ks_stat_vs_exponential": float(ks.statistic),
        "ks_p_vs_exponential": float(ks.pvalue),
        "cv_of_times": float(t.std(ddof=1) / t.mean()),
        "seed": SEED,
        "nboot": NBOOT,
        "source": str(INFILE.relative_to(ROOT)),
    }
    OUTFILE.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
