"""The WT-MetaD subset table takes each condition's full-set fit from the grid fits alone and
its subset sizes from both the grid fits and the intermediate-size fits, never one size twice."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate_subsampling as agg  # noqa: E402

NAME = "Q_bf10_height0.5_pace1000.json"
SHIFT = agg.L6 + agg.REF          # stored ln k (s^-1) of a fit with zero error


def _fit(full_err, subsets):
    """Minimal eatr-analysis payload of one condition with four runs.

    Args:
        full_err (float): signed error of the full-set fit, both estimators.
        subsets (dict[int, list[float]]): subset size -> signed error of each draw.
    """
    results = [{"n": n, "rep": i + 1, "EATR CDF ln k": SHIFT + e, "iMetaD CDF ln k": SHIFT + e,
                "EATR CDF gamma": 0.5, "EATR CDF p value": 0.5, "iMetaD CDF p value": 0.5}
               for n, errs in subsets.items() for i, e in enumerate(errs)]
    return {"EATR CDF plot": {"time": [1e-6, 2e-6, 3e-6, 2e-6], "n_total": 4},
            "EATR CDF ln k": SHIFT + full_err, "iMetaD CDF ln k": SHIFT + full_err,
            "EATR CDF gamma": 0.5, "EATR CDF p value": 0.5, "iMetaD CDF p value": 0.5,
            "subsample": {"results": results}}


def _rows(monkeypatch, grid, extra):
    """imetad_rows() over synthetic records.

    Args:
        monkeypatch (pytest.MonkeyPatch): fixture.
        grid (dict): payload of the grid fit.
        extra (dict | None): payload of the intermediate-size fit; None when that dataset is absent.
    """
    def dataset(name):
        if name == "fits.grid_intermediate" and extra is None:
            raise FileNotFoundError(name)
        return name
    monkeypatch.setattr(agg, "dataset", dataset)
    monkeypatch.setattr(agg, "iter_results",
                        lambda root: iter([(NAME, grid if root == "fits.grid" else extra)]))
    return agg.imetad_rows()


def test_intermediate_fits_add_sizes_and_leave_the_full_fit_alone(monkeypatch):
    df = _rows(monkeypatch, _fit(0.2, {2: [0.0, 2.0]}), _fit(9.0, {3: [0.5, 0.5, 3.0]}))
    eatr = df[df.estimator == "EATR CDF"].set_index("n")
    assert sorted(eatr.index) == [2, 3, 4]
    assert eatr.loc[4, "mean_err"] == pytest.approx(0.2)          # not the 9.0 of the second file
    assert eatr.loc[2, "frac_within_1"] == pytest.approx(0.5)
    assert eatr.loc[3, "frac_within_1"] == pytest.approx(2 / 3)
    assert eatr.loc[3, "n_fits"] == 3
    assert eatr.loc[3, "cost_us"] == pytest.approx(3 * 2.0)       # n x mean run length (us)
    assert math.isclose(eatr.loc[3, "n_total"], 4)


def test_grid_fits_alone_still_aggregate(monkeypatch):
    df = _rows(monkeypatch, _fit(0.2, {2: [0.0, 2.0]}), None)
    assert sorted(df[df.estimator == "EATR CDF"].n) == [2, 4]


def test_a_size_fitted_in_both_datasets_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="both"):
        _rows(monkeypatch, _fit(0.2, {2: [0.0, 2.0]}), _fit(0.2, {2: [0.1, 0.1]}))
