#!/bin/bash
# Run every analysis and plotting step against the archive, in order, in place.
#
# The point of the release layout is that someone who downloads it can reproduce the tables
# and figures without editing a path. That claim is only worth making if it has been tested
# from inside the archive, with nothing from the working tree on the path, so this script is
# run from the archive root and reports each step rather than stopping at the first failure:
# a report saying which six of twenty steps fail is more useful than one saying that step
# three does.
#
# Steps run in dependency order. The fits and per-run summaries are the archive's inputs;
# every step below derives from them and from the steps before it.
#
# Usage (from the archive root, inside the environment of pyproject.toml):
#   bash scripts/verify_release.sh            # the whole chain
#   bash scripts/verify_release.sh --quick    # tables only, skipping the figures
#   PY=/path/to/python bash scripts/verify_release.sh   # an interpreter other than python3
set -u
PY=${PY:-python3}
# a verification run should leave the tree as it found it, bytecode caches included
export PYTHONDONTWRITEBYTECODE=1
QUICK=${1:-}
cd "$(dirname "$0")/.." || exit 1
pass=0; fail=0; failed=""

# An interpreter without the packages fails every step for the same reason; say so once.
if ! "$PY" -c "import numpy, pandas, scipy, matplotlib" > /dev/null 2>&1; then
    echo "$PY lacks numpy, pandas, scipy or matplotlib: build the environment (uv sync --group figures)"
    echo "or point PY at an interpreter that has them"; exit 2
fi
if [ "$QUICK" != "--quick" ] && ! command -v jupyter > /dev/null 2>&1; then
    echo "jupyter is not on the path and the notebook step needs it: use the figures group"
    echo "of the environment, or pass --quick to verify the tables only"; exit 2
fi

# Logs go beside the archive, not to /tmp. A shared login node's /tmp fills up with other
# people's work, and a step whose log cannot be written fails on the redirect alone: every
# step then reports FAIL while every script it runs is fine. The directory is removed on the
# way out, so a finished run leaves nothing behind.
LOGDIR=$(mktemp -d "$PWD/.verify.XXXXXX") || { echo "cannot create a log directory in $PWD"; exit 1; }
trap 'rm -rf "$LOGDIR"' EXIT

step() {   # step <description> <command...>
    local name=$1; shift
    if "$@" > "$LOGDIR/step.log" 2>&1; then
        printf '  ok    %s\n' "$name"; pass=$((pass + 1))
    else
        printf '  FAIL  %s\n' "$name"; sed 's/^/          /' "$LOGDIR/step.log" | tail -3
        fail=$((fail + 1)); failed="$failed $name"
    fi
}

echo "verifying in $(pwd)"
echo
echo "derived inputs"
step "reference rate from the first-passage times" $PY scripts/analysis/reference_rate.py

echo "tables"
step "aggregate the fits"          $PY scripts/analysis/aggregate_subsampling.py
step "flooding set selection"      $PY scripts/analysis/flooding_set_selection.py
step "growth law on the grid"      $PY scripts/analysis/heuristics_growth_law.py
step "growth law at b to infinity" $PY scripts/analysis/analyze_untempered_growth_law.py
step "design rule on the grid"     $PY scripts/analysis/bias_design_recipe.py validate
step "gamma pilot"                 $PY scripts/analysis/pilot_analysis.py
step "censoring robustness"        $PY scripts/analysis/censoring_robustness.py
step "prospective outcome"         $PY scripts/analysis/prospective_outcome.py
step "error-cost tradeoff"         $PY scripts/analysis/tradeoff_analysis.py
step "KS test against accuracy"    $PY scripts/analysis/ks_detectability.py
# an analysis whose figure is a by-product: the export reads its tables
step "temperature ladder"          $PY scripts/analysis/analyze_temperature_rates.py
step "figure data export"          $PY scripts/analysis/export_paper_figure_data.py
step "supplementary tables"        $PY scripts/analysis/make_si_tables.py

[ "$QUICK" = "--quick" ] && { echo; echo "$pass passed, $fail failed (figures skipped)"; exit $((fail > 0)); }

echo "figures"
step "frequency adaptive"       $PY scripts/analysis/analyze_fametad.py
step "PMF and bias overlays"    $PY scripts/analysis/plot_pmf_bias_overlays.py
step "PMF by temperature"       $PY scripts/analysis/plot_pmf_temperature.py
step "high-barrier ladder"      $PY scripts/analysis/plot_highbarrier_ladder.py
step "accuracy against cost"    $PY scripts/analysis/plot_accuracy_vs_cost.py
step "minimum data"             $PY scripts/analysis/plot_minimum_data.py
step "regime diagnostics"       $PY scripts/analysis/plot_regime_diagnostics.py
step "FES convergence"          $PY scripts/analysis/plot_fes_convergence.py
# executed to a copy, not in place: the notebook is part of what the archive ships, and a
# check should not rewrite the artifact it is checking. The figures are still written to
# figures/pdf by the notebook itself, which is what this step is really testing.
step "paper figures notebook"   jupyter nbconvert --to notebook --execute \
                                    --output-dir "$LOGDIR" --output verify_executed.ipynb \
                                    --ExecutePreprocessor.timeout=1800 figures/make_paper_figures.ipynb

echo
echo "$pass passed, $fail failed"
[ $fail -gt 0 ] && echo "failed:$failed"
exit $((fail > 0))
