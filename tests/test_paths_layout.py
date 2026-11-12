"""A dataset is found by name in either layout, so a script that asks for it by name runs
unchanged in the working tree and in the released archive."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paths  # noqa: E402

# datasets the figure export and the KS table read, beyond the fits
NEEDED = ["unbiased.q_ladder", "runs.relaxation_times", "reference_rate", "figures.data"]


@pytest.mark.parametrize("name", NEEDED)
def test_dataset_is_known_and_present(name):
    assert name in paths.LAYOUT
    assert paths.dataset(name).exists()


def test_released_layout_wins_under_a_data_root(tmp_path, monkeypatch):
    """With RATES_DATA_ROOT set, the released path of a dataset is the one returned."""
    released, _ = paths.LAYOUT["unbiased.q_ladder"]
    (tmp_path / released).mkdir(parents=True)
    monkeypatch.setenv("RATES_DATA_ROOT", str(tmp_path))
    assert paths.dataset("unbiased.q_ladder") == tmp_path / released


@pytest.mark.parametrize("module", ["ks_detectability", "export_paper_figure_data"])
def test_scripts_of_the_chain_ask_for_data_by_name(module):
    """No script the release runs in place spells out a working-tree data path in code."""
    for d in ("", "analysis"):
        sys.path.insert(0, str(Path(paths.__file__).parent / d))
    src = Path(importlib.util.find_spec(module).origin).read_text()
    code = [ln for ln in src.splitlines() if "ROOT / \"data/" in ln]
    assert code == []
