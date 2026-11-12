"""Counts and classes quoted in the paper are taken over distinct conditions and levels, with
the definitions the text gives."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate_subsampling as agg  # noqa: E402
import ks_detectability as ksd  # noqa: E402
import tradeoff_analysis as ta  # noqa: E402


def _subset_table():
    """Two conditions of one pace: random subsets of 5 and 10 runs, and full-set fits.

    The 20-run condition's full-set fit misses; every random subset of 10 runs hits.
    """
    rows = []
    for height, n_total, full_hit in ((0.5, 20, 0.0), (1.0, 40, 1.0)):
        base = dict(cv="Q", bf=2.0, height=height, pace=30000, estimator="EATR CDF", n_total=n_total,
                    frac_within_0p5=1.0, frac_within_log10=1.0)
        rows += [dict(base, n=5, n_fits=10, frac_within_1=0.5, cost_us=5.0),
                 dict(base, n=10, n_fits=10, frac_within_1=1.0, cost_us=10.0),
                 dict(base, n=n_total, n_fits=1, frac_within_1=full_hit, cost_us=float(n_total))]
    return pd.DataFrame(rows)


def test_minimum_data_for_wtmetad_counts_random_subsets_only():
    """A condition's single full-set fit is not one more subset size of the pooled criterion."""
    wt = agg.wtmetad_random_subsets(_subset_table())
    assert sorted(wt.n.unique()) == [5, 10]
    out = agg.minimum_cost(wt, ["cv", "estimator", "pace"]).iloc[0]
    assert out["n_at_min[frac_within_1]"] == 10
    # with the full-set rows left in, the miss at "size 20" would push the minimum to 40
    kept = agg.minimum_cost(_subset_table(), ["cv", "estimator", "pace"]).iloc[0]
    assert kept["n_at_min[frac_within_1]"] == 40


def test_wtmetad_random_subsets_drop_bias_factor_one():
    t = _subset_table()
    t.loc[t.height == 1.0, "bf"] = 1.0
    assert set(agg.wtmetad_random_subsets(t).height) == {0.5}


def _worse(abs_e, abs_i, ref_in_ci=False, diff_lt_spread=False):
    """One EATR-worse-than-iMetaD case.

    Args:
        abs_e (float): |error| of EATR.
        abs_i (float): |error| of iMetaD.
        ref_in_ci (bool): reference inside the EATR bootstrap interval.
        diff_lt_spread (bool): difference below the subset spread.
    """
    return dict(abs_e=abs_e, abs_i=abs_i, diff=abs_e - abs_i, ref_in_eatr_ci=ref_in_ci,
                diff_lt_spread=diff_lt_spread, both_within_1=abs_e <= 1 and abs_i <= 1)


def test_systematic_class_requires_imetad_within_one_ln_unit():
    w = pd.DataFrame([_worse(1.6, 0.4), _worse(1.6, 1.2), _worse(0.9, 0.8), _worse(1.6, 0.4, ref_in_ci=True)])
    cls = list(ta.worse_class(w))
    assert cls[0].startswith("systematic")
    assert cls[1] == "other"                 # EATR outside, but iMetaD is outside too
    assert cls[2].startswith("marginal")
    assert cls[3].startswith("statistical")


def test_static_levels_are_counted_once():
    """A level fitted at several run counts enters the KS tally at its largest run count only."""
    ms = pd.DataFrame([
        dict(family="vesf", cv="Q", pace=float("nan"), set_label=5.0, label=10.0, total=10, ks_p=0.01),
        dict(family="vesf", cv="Q", pace=float("nan"), set_label=5.0, label=200.0, total=200, ks_p=0.50),
        dict(family="vesf", cv="Q", pace=float("nan"), set_label=7.0, label=200.0, total=200, ks_p=0.40),
        dict(family="opesf", cv="Q", pace=100.0, set_label=5.0, label=float("nan"), total=50, ks_p=0.30),
        dict(family="opesf", cv="Q", pace=500.0, set_label=5.0, label=float("nan"), total=50, ks_p=0.20),
    ])
    out = ksd.distinct_levels(ms)
    assert len(out) == 4
    assert out[(out.family == "vesf") & (out.set_label == 5.0)].ks_p.tolist() == [0.50]


def test_grid_ks_table_holds_each_condition_once():
    """The KS tally and its figure count one full-data fit per grid condition and estimator."""
    g = ksd.grid_fits()
    assert g.groupby("estimator").size().to_dict() == {"EATR": 639, "iMetaD": 639}
    assert not g.duplicated(["cv", "bf", "height", "pace", "estimator"]).any()


def test_cdf_figure_shows_the_fits_of_the_grid_tables():
    """The CDF panels carry the error, gamma and KS p of the same fits as every grid figure."""
    import export_paper_figure_data as exp
    rows, meta = exp.cdf_fit_tables()
    full = ta.load_grid_full().set_index(["cv", "bf", "height", "pace"])
    assert len(meta) == 2 * 4 * len(exp.CDF_PACES)
    for m in meta.itertuples():
        ref = full.loc[(m.cv, 10.0, 0.25, m.pace)]
        col = "eatr" if m.estimator == "EATR" else "imetad"
        assert abs(m.err_ln - ref[f"err_{col}"]) < 1e-9
        assert abs(m.ks_p - ref[f"ks_{col}"]) < 1e-12
        if m.estimator == "EATR":
            assert abs(m.gamma - ref.gamma) < 1e-12
    assert set(rows.estimator) == {"EATR", "iMetaD"}


def test_hit_rate_summary_columns_belong_to_the_one_ln_unit_criterion():
    """max_hit_rate and hit_rate_at_largest_n describe the criterion the paper uses, not the last one looped over."""
    t = _subset_table()
    t = t[t.n < t.n_total].assign(frac_within_log10=0.25, frac_within_0p5=0.5)
    out = agg.minimum_cost(t, ["cv", "estimator", "pace"]).iloc[0]
    assert out["max_hit_rate"] == 1.0            # frac_within_1 at n = 10
    assert out["hit_rate_at_largest_n"] == 1.0
