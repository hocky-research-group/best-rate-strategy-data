"""The prospective test rests on the predictions being those registered before the runs.
The check against the registered version holds wherever the scripts run: with the working
repository's history, in the released archive without git, and in a repository that holds
the release as a single commit."""
import subprocess

import pytest

import prospective_outcome as po

REGISTERED_COPY = "analysis/prospective_predictions_registered.csv"


def _repository_without_history(tmp_path, current: str, registered: str):
    """A git repository holding the two tables and no registered commit.

    tmp_path (Path): directory to build in.
    current (str): text of the predictions table.
    registered (str): text of the shipped registered copy.
    """
    (tmp_path / "analysis").mkdir()
    (tmp_path / po.PREDICTIONS).write_text(current)
    (tmp_path / REGISTERED_COPY).write_text(registered)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_registered_copy_is_checked_where_the_commit_is_absent(tmp_path, monkeypatch):
    text = (po.ROOT / po.PREDICTIONS).read_text()
    monkeypatch.setattr(po, "ROOT", _repository_without_history(tmp_path, text, text))
    assert len(po.registered_predictions()) == 2


def test_changed_predictions_are_refused_where_the_commit_is_absent(tmp_path, monkeypatch):
    text = (po.ROOT / po.PREDICTIONS).read_text()
    changed = text.replace("\n", "\n#", 1)
    monkeypatch.setattr(po, "ROOT", _repository_without_history(tmp_path, changed, text))
    with pytest.raises(SystemExit, match="differs"):
        po.registered_predictions()


def test_shipped_copy_is_the_registered_commit():
    """Where the history is at hand, the copy shipped in its place is that commit's file."""
    show = subprocess.run(["git", "-C", str(po.ROOT), "show", f"{po.REGISTERED_COMMIT}:{po.PREDICTIONS}"],
                          capture_output=True, text=True)
    if show.returncode != 0:
        pytest.skip("the registered commit is not in this repository")
    assert show.stdout == (po.ROOT / REGISTERED_COPY).read_text()
