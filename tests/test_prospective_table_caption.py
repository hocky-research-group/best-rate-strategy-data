"""The prospective table's expected escape time is the registered one, so its caption has
to name the calibration that produces it: the ratio fixed at registration over the grid
conditions the selection accepted, not a ratio a reader would recompute from the grid."""
import importlib

import pandas as pd
import pytest


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """The table module, writing into a temporary directory.

    tmp_path (Path): directory that receives the tables.
    monkeypatch (MonkeyPatch): redirects both output directories.
    """
    monkeypatch.setenv("SI_TABLES_DIR", str(tmp_path / "si"))
    module = importlib.import_module("make_si_tables")
    monkeypatch.setattr(module, "OUT", tmp_path / "si")
    monkeypatch.setattr(module, "FULL_OUT", tmp_path / "full")
    (tmp_path / "si").mkdir(exist_ok=True)
    return module


def test_caption_states_the_registered_calibration(tables, tmp_path):
    tables.prospective_table()
    text = (tmp_path / "full" / "s13_prospective.tex").read_text()
    caption = text[text.index("\\caption{"):text.index("\\label{")]
    registered = pd.read_csv(tables.ROOT / "analysis/prospective_predictions_registered.csv").set_index("cv")
    for cv in ("Q", "Ree"):
        ratio = registered.loc[cv, "grid_obs_over_pred_escape"]
        assert f"{ratio:.2f} for {tables.NAME[cv]}" in caption
    assert "at registration" in caption
    assert "accepted" in caption
    assert "the grid's median" not in caption
