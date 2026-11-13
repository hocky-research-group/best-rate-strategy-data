"""The files too large for the repository are fetched by hand from the full-data folder, so
the check has to tell a reader three things apart: the file is in place and is the one the
list names, it is absent, or something else sits at its path."""
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_large_files.sh"
LARGE = "runs/unbiased/first_passage/first_pass_runs.tar.gz"
CONTENT = b"the first-passage runs\n"


@pytest.fixture
def tree(tmp_path):
    """A stand-in repository holding the script and a list naming one large file.

    tmp_path (Path): receives scripts/ and LARGE_FILES.tsv.
    """
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts")
    digest = hashlib.sha256(CONTENT).hexdigest()
    (tmp_path / "LARGE_FILES.tsv").write_text(f"path\tbytes\tsha256\n{LARGE}\t{len(CONTENT)}\t{digest}\n")
    return tmp_path


def _check(tree: Path) -> subprocess.CompletedProcess:
    """Run the check inside the stand-in repository.

    tree (Path): directory made by the tree fixture.
    """
    return subprocess.run(["bash", str(tree / "scripts/check_large_files.sh")], capture_output=True, text=True)


def test_the_listed_file_in_place_passes(tree):
    (tree / LARGE).parent.mkdir(parents=True)
    (tree / LARGE).write_bytes(CONTENT)
    run = _check(tree)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"present  {LARGE}" in run.stdout


def test_an_absent_file_fails_and_says_where_it_goes(tree):
    run = _check(tree)
    assert run.returncode != 0
    assert f"absent   {LARGE}" in run.stdout and "first_pass_runs.tar.gz" in run.stdout and "README.md" in run.stdout


def test_another_file_at_the_path_fails(tree):
    (tree / LARGE).parent.mkdir(parents=True)
    (tree / LARGE).write_bytes(b"THE FIRST-PASSAGE RUNS\n")
    run = _check(tree)
    assert run.returncode != 0
    assert f"FAILED   {LARGE}" in run.stdout


def test_the_readme_links_the_folder_the_check_points_to():
    readme = ROOT / ("release_README.md" if (ROOT / "release_README.md").exists() else "README.md")
    assert "https://nyu.box.com/s/z1skc8raekdjmmho2eugp5thn7qfskf3" in readme.read_text()
    assert "scripts/check_large_files.sh" in readme.read_text()
