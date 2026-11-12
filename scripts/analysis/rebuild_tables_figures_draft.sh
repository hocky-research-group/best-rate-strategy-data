#!/bin/bash
# Rebuild every analysis table and figure, and the manuscript and SI PDFs, from the data in
# this repository, stopping at the first failed gate.
#
# Usage:
#   bash scripts/rebuild_tables_figures_draft.sh           # from the committed data/
#   FETCH=1 bash scripts/rebuild_tables_figures_draft.sh   # rsync torch results first
#
# Gates, in order:
#   1. (FETCH=1 only) rsync of the flooding subset, pilot and prospective results from the
#      torch study directory (ssh host alias "torch"). Once results are complete, the torch
#      copies are archived (study directory README) and FETCH is no longer needed.
#   2. Completeness: 7636 subset fits, 400 pilot fits and 2 prospective fits.
#   3. scripts/check_flooding_refits.py exits 0.
#   4. The test suite passes.
#   5. Every regeneration step exits 0 and prints no full-data-refit fallback or
#      failed-fit warning.
#   6. The manuscript and SI compile without LaTeX errors or undefined references,
#      the supplement first so the manuscript can resolve its references into it,
#      and no supplementary cross-reference is left unresolved.
# Python runs through uv when uv is installed, otherwise through the project .venv.
# Per-step logs go to a temporary directory printed at the start.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
LOG=$(mktemp -d "${TMPDIR:-/tmp}/rebuild_tables_figures_draft.XXXXXX")
if command -v uv >/dev/null 2>&1; then PY=(uv run python); else PY=("$ROOT/.venv/bin/python"); fi
cd "$ROOT" || exit 1
echo "logs: $LOG; python: ${PY[*]}"

if [ "${FETCH:-0}" = 1 ]; then
  echo "== fetch"
  for d in subsampling/flooding_subsets subsampling/flooding_pilot results/prospective_wtmetad; do
    mkdir -p "data/$d"
    rsync -a "torch:$P/$d/" "data/$d/" || { echo "GATE FAILED: rsync $d"; exit 1; }
  done
fi

echo "== completeness"
"${PY[@]}" - <<'EOF' || { echo "GATE FAILED: completeness"; exit 1; }
import glob, gzip, json, sys
def count(pattern):
    ok = failed = 0
    for p in glob.glob(pattern):
        for rec in map(json.loads, gzip.open(p, "rt")):
            ok += "data" in rec
            failed += "data" not in rec
    return ok, failed
s, p = count("data/subsampling/flooding_subsets/task_*.jsonl.gz"), count("data/subsampling/flooding_pilot/task_*.jsonl.gz")
n_pro = len(glob.glob("data/results/prospective_wtmetad/*.json"))
print(f"subset fits ok {s[0]} failed records {s[1]}; pilot fits ok {p[0]} failed records {p[1]}; prospective fits {n_pro}")
sys.exit(0 if (s[0] == 7636 and p[0] == 400 and n_pro == 2) else 1)
EOF

echo "== refit check"
"${PY[@]}" scripts/check_flooding_refits.py > "$LOG/check_flooding_refits.log" 2>&1 || { echo "GATE FAILED: check_flooding_refits"; grep -v "^  " "$LOG/check_flooding_refits.log"; exit 1; }
grep -v "^  " "$LOG/check_flooding_refits.log"

echo "== tests"
"${PY[@]}" -m pytest -q -p no:cacheprovider tests > "$LOG/tests.log" 2>&1 || { echo "GATE FAILED: tests"; tail -5 "$LOG/tests.log"; exit 1; }
tail -1 "$LOG/tests.log"

echo "== regenerate"
: > "$LOG/warnings.txt"
for s in aggregate_results flooding_set_selection aggregate_subsampling "bias_design_recipe validate" pilot_analysis \
         plot_highbarrier_ladder plot_pmf_bias_overlays plot_accuracy_vs_cost plot_minimum_data \
         tradeoff_analysis ks_detectability censoring_robustness prospective_outcome analyze_temperature_rates \
         make_si_tables; do
  script=${s%% *}; args=""; [ "$script" != "$s" ] && args=${s#* }
  "${PY[@]}" -W default "scripts/$script.py" $args > "$LOG/$script.log" 2>&1 || { echo "GATE FAILED: $script"; tail -8 "$LOG/$script.log"; exit 1; }
  grep -hiE "no full-data refit|skipping failed fit|failed pilot fit" "$LOG/$script.log" >> "$LOG/warnings.txt"
  echo "ok $script"
done
if [ -s "$LOG/warnings.txt" ]; then echo "GATE FAILED: warnings"; cat "$LOG/warnings.txt"; exit 1; fi

echo "== compile"
# The supplement is compiled first and the manuscript second, and the order is not a
# preference: the manuscript reads EfficientEATR_SI.aux through xr-hyper to resolve its
# references into the supplement, so a missing or stale supplementary aux file renders every
# one of them as ?? without failing anything. Compiling the supplement twice first settles
# its own figure numbers before the manuscript reads them.
cd overleaf_best_rates_paper || exit 1
for doc in EfficientEATR_SI EfficientEATR EfficientEATR_SI; do
  pdflatex -interaction=nonstopmode $doc.tex > /dev/null 2>&1; bibtex $doc > /dev/null 2>&1
  pdflatex -interaction=nonstopmode $doc.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode $doc.tex > /dev/null 2>&1
  n=$(grep -c "^! " $doc.log)
  # "Font shape ... undefined" is a font substitution notice, not a missing reference
  u=$(grep -i "undefined" $doc.log | grep -vc "Font")
  q=$(grep -c "??" $doc.log 2>/dev/null || echo 0)
  echo "$doc: $(grep -o '[0-9]* pages' $doc.log | tail -1); LaTeX errors $n, undefined $u"
  [ "$n" -eq 0 ] && [ "$u" -eq 0 ] || { echo "GATE FAILED: LaTeX $doc"; exit 1; }
done
# every cross-reference into the supplement must have resolved to a number
if pdftotext EfficientEATR.pdf - 2>/dev/null | grep -q "Fig\. ??\|Figs\. ??"; then
  echo "GATE FAILED: unresolved supplementary cross-reference in the manuscript"; exit 1
fi
echo "REBUILD OK"
