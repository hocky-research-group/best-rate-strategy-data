"""full_data_view must splice in per-set values only from a refit of the same ladder, keep the
bootstrap headline, and never let one-resample per-set values through unannounced."""
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import flooding_io  # noqa: E402

REL = "results/opes_highbarrier/Q_pace1000_barriers5-17.json"


def _fit(labels, totals, lnk, logk0, gamma_best, numboots=100):
    """Minimal eatr-flooding-analysis payload.

    Args:
        labels (list[float]): set labels (BARRIER values).
        totals (list[int]): runs per set.
        lnk (list[float]): per-set ln k_obs.
        logk0 (float): headline ln k0.
        gamma_best (float): gamma of the diagnostics scan.
        numboots (int | None): bootstrap resamples; None for a fit run without --bootstrap,
            whose payload carries neither numboots nor bootstrap fields.
    """
    d = dict(logk0=logk0, gamma=0.9, opes_logk0=13.9,
             set_reports=[dict(barrier=b, total=t, transitioned=t, log_k_obs=k) for b, t, k in zip(labels, totals, lnk)],
             flooding_diagnostics=dict(gamma_best=gamma_best))
    if numboots is not None:
        d.update(numboots=numboots, bootstrap_iterations=list(range(numboots)), bootstrap_logk0_ci=[13.8, 14.2])
    return d


@pytest.fixture
def index(monkeypatch):
    """Install a one-record refit index in place of data/refits_full_data."""
    refit = _fit([5.0, 7.0], [50, 50], [15.9, 16.8], logk0=14.05, gamma_best=0.97, numboots=None)
    idx = {REL: dict(file=REL, data=refit)}
    monkeypatch.setattr(flooding_io, "_INDEX", idx)
    return idx


def test_view_takes_per_set_values_from_refit_and_keeps_bootstrap_headline(index):
    """Per-set values describe the full data; the headline stays the bootstrap mean the paper quotes."""
    orig = _fit([5.0, 7.0], [50, 50], [15.7, 17.1], logk0=14.08, gamma_best=0.99)
    v = flooding_io.full_data_view(orig, REL)
    assert [s["log_k_obs"] for s in v["set_reports"]] == [15.9, 16.8]
    assert v["flooding_diagnostics"]["gamma_best"] == 0.97
    assert (v["logk0"], v["gamma"], v["opes_logk0"], v["bootstrap_logk0_ci"]) == (14.08, 0.9, 13.9, [13.8, 14.2])
    assert v["_set_reports_source"] == "full-data refit (no bootstrap)"
    assert [s["log_k_obs"] for s in orig["set_reports"]] == [15.7, 17.1]


@pytest.mark.parametrize("labels,totals", [([5.0, 9.0], [50, 50]), ([5.0, 7.0], [50, 49])])
def test_view_refuses_refit_of_a_different_ladder(index, labels, totals):
    """A refit whose set labels or run counts differ describes other data and must not be spliced in."""
    with pytest.raises(ValueError):
        flooding_io.full_data_view(_fit(labels, totals, [15.7, 17.1], 14.08, 0.99), REL)


def test_missing_refit_strict_raises(index):
    """Strict callers (figures built only from per-set values) must stop without a refit."""
    with pytest.raises(KeyError):
        flooding_io.full_data_view(_fit([5.0], [50], [15.7], 14.08, 0.99), "results/other.json", strict=True)


def test_fit_without_bootstrap_is_used_as_is(index):
    """A fit run without --bootstrap already reports its own data, so it needs no refit:
    it passes through unchanged, silently, even in strict mode."""
    fit = _fit([5.0, 9.0], [10, 10], [15.7, 17.9], 14.1, 0.95, numboots=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = flooding_io.full_data_view(fit, "subsampling/flooding_subsets/pilot/Q_pace1000_b5-9/N10_rep1.json", strict=True)
    assert v["set_reports"] == fit["set_reports"]
    assert v["flooding_diagnostics"] == fit["flooding_diagnostics"]
    assert v["_set_reports_source"] == "full-data fit (no bootstrap)"


def test_missing_refit_warns_and_marks_the_resample_values(index):
    """Without a refit the per-set values are one bootstrap resample; that must be announced."""
    orig = _fit([5.0], [50], [15.7], 14.08, 0.99)
    with pytest.warns(UserWarning, match="results/other.json"):
        v = flooding_io.full_data_view(orig, "results/other.json")
    assert v["set_reports"] == orig["set_reports"]
    assert v["_set_reports_source"].startswith("bootstrap last resample")
