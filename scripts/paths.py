"""Where each dataset lives, in either the working tree or the released archive.

The working tree keeps everything this study downloaded under `data/`, mirroring the layout
of the directories it came from. The released archive groups runs by the kind of simulation
they are, so that a reader can find "untempered metadynamics" without knowing which
experiment needed it. The same scripts have to read both, so neither layout is written into
them: a dataset is asked for by name and this module says where it is.

    from paths import dataset
    for path in sorted(dataset("fits.untempered").glob("*.json")):
        ...

Resolution order for a name:

1. `RATES_DATA_ROOT`, if set, is treated as the root of a released archive.
2. The released layout, relative to the repository root.
3. The working layout.

The first that exists wins, so a checkout with neither layout populated fails at the point
the missing dataset is read, naming it, rather than returning an empty glob that silently
becomes an empty table three steps later.
"""
import os
from pathlib import Path


def _find_root() -> Path:
    """The archive or repository root, from wherever this file happens to sit.

    The working tree keeps scripts one level below the root, the released archive keeps them
    two, so the depth cannot be assumed. The root is the nearest ancestor holding a directory
    that only a root holds.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "runs").is_dir() or (parent / "data").is_dir():
            return parent
    return here.parents[1]


ROOT = _find_root()

# logical name -> (path in the released archive, path in the working tree)
LAYOUT = {
    "fits.grid":             ("runs/wtmetad_grid/fits",                   "data/subsampling/imetad_grid"),
    "fits.grid_intermediate": ("runs/wtmetad_grid/fits_intermediate_sizes", "data/subsampling/imetad_grid_intermediate_sizes"),
    "fits.untempered":       ("runs/wtmetad_untempered/fits",             "data/subsampling/untempered"),
    "fits.frequency_adaptive": ("runs/wtmetad_frequency_adaptive/fits",   "data/subsampling/fametad"),
    "fits.flooding_subsets": ("runs/opes_flooding/flooding_subsets",      "data/subsampling/flooding_subsets"),
    "fits.flooding_nsim":    ("runs/opes_flooding/flooding_nsim",         "data/subsampling/flooding_nsim"),
    "fits.flooding_pilot":   ("runs/opes_flooding/flooding_pilot",        "data/subsampling/flooding_pilot"),
    "fits.opes_temperature": ("runs/opes_flooding/opes_temperature",      "data/subsampling/opes_temperature"),
    "fits.wtmetad_temperature": ("runs/wtmetad_temperature/wtmetad_temperature_fits",
                                 "data/subsampling/wtmetad_temperature/results"),
    "fits.wtmetad_temperature_slowpace": ("runs/wtmetad_temperature/wtmetad_temperature_slowpace_fits",
                                          "data/subsampling/wtmetad_temperature_slowpace/results"),
    "fits.wtmetad_temperature_slowpace_n200": ("runs/wtmetad_temperature/wtmetad_temperature_slowpace_n200_fits",
                                               "data/subsampling/wtmetad_temperature_slowpace_n200/results"),
    "fits.wtmetad_temperature_Ree": ("runs/wtmetad_temperature/wtmetad_temperature_Ree_fits",
                                     "data/subsampling/wtmetad_temperature_Ree/results"),
    "fits.pilot":            ("runs/wtmetad_prospective/pilot_fits",      "data/subsampling/pilot"),
    "refits.flooding":       ("runs/opes_flooding/refits_full_data",      "data/refits_full_data"),
    "runs.grid_summary":     ("runs/wtmetad_grid/trajectory_summary",     "data/results/trajectory_summary"),
    "runs.study_summary":    ("runs/_per_run_summaries",                  "data/results/study_run_summaries"),
    "runs.prospective":      ("runs/wtmetad_prospective/outcome",         "data/results/prospective_wtmetad"),
    "runs.highbarrier":      ("runs/opes_flooding/opes_highbarrier",      "data/results/opes_highbarrier"),
    "runs.bias_profiles":    ("runs/metad_pmf/bias_profiles",             "data/results/bias_profiles"),
    "source.prod_opes":      ("runs/opes_flooding/prod_opes",             "data/triasha_results/prod_opes"),
    "source.prod_ves_cutoff": ("runs/ves_flooding/prod_ves_cutoff",       "data/triasha_results/prod_ves_cutoff"),
    "source.triasha":        ("runs",                                     "data/triasha_results"),
    "reference_rate":        ("runs/unbiased/reference_rate",             "data/reference_rate"),
    "pmf_unbiased":          ("runs/metad_pmf/heuristics/pmf_unbiased",   "data/heuristics/pmf_unbiased"),
    "unbiased.q_ladder":     ("runs/unbiased/q_ladder_snapshots",         "data/unbiased_runs_q_ladder"),
    "runs.relaxation_times": ("runs/unbiased/relaxation_times",           "data/results/tau_from_fes"),
    "verification":          ("analysis/verification",                    "data/verification"),
    "figures.data":          ("figures/data",                             "paper_figures/data"),
}


def dataset(name: str) -> Path:
    """Directory holding the named dataset, in whichever layout is present.

    Args:
        name (str): key of LAYOUT, for example "fits.untempered".

    Returns:
        Path to the directory.

    Raises:
        KeyError: the name is not a known dataset.
        FileNotFoundError: no layout on this machine holds it.
    """
    if name not in LAYOUT:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(LAYOUT))}")
    released, working = LAYOUT[name]
    env = os.environ.get("RATES_DATA_ROOT")
    candidates = ([Path(env) / released] if env else []) + [ROOT / released, ROOT / working]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"dataset {name!r} not found; looked in " + ", ".join(str(c) for c in candidates))


def optional(name: str) -> Path | None:
    """The dataset's directory, or None when it is absent.

    For datasets a run of the analysis can legitimately do without, such as a campaign whose
    fits have not been produced yet.

    Args:
        name (str): key of LAYOUT.
    """
    try:
        return dataset(name)
    except FileNotFoundError:
        return None
