# Efficient strategies for obtaining accurate kinetics from biased simulations — data and scripts

Everything behind the figures and tables of the paper: the inputs each simulation was
generated from, the scripts that generated, ran and analysed them, the fits and per-run
summaries, and the notebook that draws every figure from those.

The analysis runs on this archive in place. Nothing points outside it.

## Layout

```
inputs/          templates every run was generated from
  structure/     the Go model: go.pdb and the LAMMPS data file
  lammps/        the LAMMPS input, with placeholders for the per-run values
  plumed/        one PLUMED input per collective variable and biasing scheme
runs/            per-run data, grouped by the kind of simulation
  unbiased/                     the 120 us trajectory, the first-passage set, the reference
                                rate, CV relaxation times, and the frames drawn in Fig. 1(a)
  wtmetad_grid/                 the 640-condition well-tempered grid
  wtmetad_untempered/           hills that never decay, the b -> infinity limit
  wtmetad_frequency_adaptive/   deposition interval that grows with the acceleration
  wtmetad_prospective/          the two conditions predicted before they were run
  wtmetad_temperature/          one condition per temperature at two paces along Q, and
                                along R_ee at 300, 285 and 270 K
  opes_flooding/                static-bias ladders at 312 K, levels past the barrier, and
                                the ladders down the temperature scan
  ves_flooding/                 variationally enhanced cutoff flooding
  metad_pmf/                    free energy scans and bias profiles
  _per_run_summaries/           one row per run of the campaigns produced for this paper
campaigns/       symlinks that regroup runs/ by experiment, storing nothing twice
scripts/
  setup/         design a campaign and generate its run tree from the inputs
  run/           submit trajectories
  fit/           rate fits per condition
  analysis/      aggregate fits into tables, and draw the figures
analysis/        the derived tables, and the verification checks behind them
figures/         the notebook, the data it reads, and the figures it writes
tests/           the definitions the paper's counts and criteria rest on
```

Runs are grouped by what kind of simulation they are rather than by the experiment that
needed them or the figure they appear in, since the same runs serve several of each. The
experiments are reassembled in `campaigns/` as symlinks.

The LAMMPS input in `inputs/lammps/` defines two variables, `e` (0.97) and `epNC`, that none
of its `pair_coeff` lines uses. The pair coefficients are those of the Gō-model topology,
converted from kJ/mol and nm by `econv` and `dconv`, with no further scaling; the bonded
terms in `inputs/structure/new_data_file.lammps` carry the same unit conversion and nothing
else.

`SOURCE_COMMIT` names the commit of the authors' working repository the scripts, tables and
figures were taken from. `SHA256SUMS` lists every file.

## Reproducing the figures

The environment is pinned in `pyproject.toml` and `uv.lock`, and `uv sync` builds it. Without
`uv`, any Python 3.11 or later with numpy, pandas, scipy and matplotlib runs the scripts, and
the notebook needs jupyter as well; `scripts/verify_release.sh` runs the whole chain and
takes the interpreter from `PY`.

```
uv sync --group figures --group dev
uv run python scripts/analysis/aggregate_subsampling.py      # fits      -> analysis/*.csv
uv run python scripts/analysis/export_paper_figure_data.py   # analysis  -> figures/data/*.csv
uv run --group figures jupyter nbconvert --to notebook --execute --inplace \
        figures/make_paper_figures.ipynb                     # data      -> figures/pdf/*.pdf
uv run python scripts/analysis/make_si_tables.py             # the supplementary tables
uv run pytest tests                                          # the definitions behind the counts
```

Each step reads only what the step before it wrote, so any of them can be run alone against
the files already here. `scripts/verify_release.sh` lists every step in dependency order.

The scripts under `scripts/analysis` that the verification runs work in place. The scripts
under `scripts/setup`, `scripts/run` and `scripts/fit`, and those analysis scripts that read
trajectories, record how the data here were produced; they name the cluster paths they ran
against and need the raw runs.

Sixteen tables under `analysis/` are carried as they are and not rewritten by the
verification. Eleven are computed from trajectories or from the source tree's own fits:
`cv_autocorrelation*.csv`, `fes_temperature_scan*.csv`, `unbiased_pmf_convergence.csv`,
`untempered_growth_law_runs.csv`, `master_table.csv` and `master_sets.csv`. Three are designs
made before a campaign ran: `bias_design_table.csv` and `wtmetad_temperature_design*.csv`.
`prospective_predictions.csv` and its copy `prospective_predictions_registered.csv` are the
predictions as committed before the prospective runs; `prospective_outcome.py` refuses to
run if the two differ, so the first is never regenerated.

## What is not here

**The first-passage trajectories.** `runs/unbiased/first_passage/first_pass_runs.tar.gz`,
the 400 unbiased runs the reference rate rests on, is too large for a git host. It is listed
with its size and checksum in `LARGE_FILES.tsv` and held in the data deposit named in the
paper's data availability statement; `scripts/fetch_large_files.sh` downloads and checks it.
No figure or table needs it: the first-passage times extracted from it are in
`runs/unbiased/reference_rate/`.

**Raw trajectories of the biased runs.** Each run's COLVAR, HILLS or KERNELS and PLUMED log,
tarred one archive per condition, come to 294 GB in 919 archives and are held
separately; see the data availability statement. This archive holds what the analysis reads
instead: one escape time, acceleration, maximum bias and hill count per run, in
`runs/_per_run_summaries` and `runs/wtmetad_grid/trajectory_summary`, and the fit of every
condition.

Two sets of runs are not in the raw archive either. Bias factor 1 is PLUMED's degenerate
case, where every hill after the first has zero height and the run is effectively unbiased;
those trajectories are 920 GB on their own and no figure in this paper uses them, though
their fits are here. The `withwork` and uncut VES variants of the source tree are likewise
absent, as no analysis in this paper reads them.

**The software environment.** The LAMMPS build, the PLUMED build and the container images
are archived with the raw trajectories rather than here. Versions are recorded in the paper;
every fit records the git hash of the analysis code that produced it, in the `.cmd` file
beside it.

## Licence

Everything in this repository is released under the MIT licence; see `LICENSE`.

## Provenance

The metadynamics grid, the OPES and VES flooding ladders, and the unbiased reference data
were produced by Triasha Pal. The unbiased trajectory and the LAMMPS and PLUMED build are
from Nicodemo Mazzaferro. The untempered, frequency-adaptive, temperature, prospective and
past-barrier campaigns were produced for this paper.

Within a condition every run has its own velocity seed. In the
grid and in the 312 K OPES and VES ladders the seed is the run index, so run k starts from
the same velocities in every condition and differs only in the bias applied. The campaigns
produced for this paper add a base to the index, set in the campaign's setup script
(`scripts/setup/`); a temperature campaign uses one seed per run index at every temperature,
and the two paces along Q share theirs.
