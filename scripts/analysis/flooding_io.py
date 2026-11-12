"""Full-data view of eatr-flooding-analysis JSON files.

Before EATR-rate-analysis 14cee55, a run with --bootstrap wrote the set_reports and
flooding_diagnostics of its LAST bootstrap resample, while logk0, gamma and opes_logk0
were bootstrap means and are correct. The no-bootstrap refits collected in
data/refits_full_data/task_*.jsonl.gz describe the full data. `full_data_view` returns a
copy of a fit whose set_reports and flooding_diagnostics come from the matching refit and
whose headline bootstrap estimates are the original's; `_set_reports_source` records
which applied. Files are matched by their path relative to data/.
"""
import copy
import gzip
import json
import warnings
from pathlib import Path

import paths as _paths
from paths import dataset

ROOT = _paths.ROOT
_INDEX: dict | None = None


def refit_index() -> dict:
    """Map data-relative path -> refit record ({"data": fit JSON, "n_per_set": [...]})."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for pk in sorted((dataset("refits.flooding")).glob("task_*.jsonl.gz")):
            with gzip.open(pk, "rt") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if "data" in rec:
                        _INDEX[rec["file"]] = rec
    return _INDEX


def full_data_view(d: dict, rel: str, strict: bool = False) -> dict:
    """Fit JSON with set_reports and flooding_diagnostics from the full-data refit.

    Args:
        d (dict): parsed flooding JSON as originally written.
        rel (str): its path relative to data/ (e.g. "results/opes_highbarrier/Q_...json").
        strict (bool): raise when no refit exists instead of warning and returning d unchanged.
    Raises ValueError if the refit's sets differ from the original's (labels or run totals).
    """
    if d.get("numboots") is None and not d.get("bootstrap_iterations"):
        # a fit run without --bootstrap reports its own data and needs no refit
        out = copy.copy(d)
        out["_set_reports_source"] = "full-data fit (no bootstrap)"
        return out
    rec = refit_index().get(rel)
    if rec is None:
        if strict:
            raise KeyError(f"no full-data refit for {rel}")
        warnings.warn(f"no full-data refit for {rel}: its set_reports are one bootstrap resample", stacklevel=2)
        out = copy.copy(d)
        out["_set_reports_source"] = "bootstrap last resample (no refit available)"
        return out
    r = rec["data"]
    old_labels = [s["barrier"] for s in d["set_reports"]]
    new_labels = [s["barrier"] for s in r["set_reports"]]
    old_tot = [s["total"] for s in d["set_reports"]]
    new_tot = [s["total"] for s in r["set_reports"]]
    if old_labels != new_labels or old_tot != new_tot:
        raise ValueError(f"{rel}: refit sets {new_labels}/{new_tot} differ from original {old_labels}/{old_tot}")
    out = copy.copy(d)
    out["set_reports"] = r["set_reports"]
    out["flooding_diagnostics"] = r["flooding_diagnostics"]
    out["logk0_full_point"] = r["logk0"]
    out["gamma_full_point"] = r["gamma"]
    out["_set_reports_source"] = "full-data refit (no bootstrap)"
    return out
