#!/bin/bash
#SBATCH --job-name=flood_temp
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_temp_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_temp_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR-flooding fits for the OPES temperature ladder, one array task per temperature.
#
# The whole ladder at one temperature is a single fit, not five: EATR-flooding determines
# gamma from how the observed escape rate moves as the fill deepens across the bias sets,
# so the sets have to be handed to it together, each with its own --barrier. A rung on its
# own carries no information about gamma.
#
# This is the instrument for the question a single well-tempered condition per temperature
# cannot answer. The rungs span residual barriers of 0.94 to 4.02 kT, which brackets the
# two-kT rule this work recommends, so the fit reports both a rate and whether that rate
# still depends on how much of the barrier was filled.
#
# --temp is the temperature of the ladder: the estimator turns the recorded bias into an
# acceleration through beta, so the benchmark temperature would misscale every bias.
# --vcol 6 is opes.bias, the layout the excluded-region CUSTOM keeps stable across this
# campaign; the globs start at run_ so PLUMED's bck.0. backups of restarted runs, which
# hold truncated trajectories, can never enter a fit.
#
# Usage: sbatch --array=0-3 fit_opes_temperature.sbatch
# Skips a temperature whose JSON already exists (re-submit safe).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
ROOT=$P/simulations/opes_temperature
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
OVL=$SW/torch-eatr-python-15GB-500K.ext3
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OUTDIR=$P/subsampling/opes_temperature/results
mkdir -p $OUTDIR

TEMPS=(312 300 285 270)
T=${TEMPS[$SLURM_ARRAY_TASK_ID]}
TAG=opes_T${T}
OUT=$OUTDIR/$TAG.json
if [ -s "$OUT" ]; then echo "exists, skipping $OUT"; exit 0; fi

# one -i group per rung, each with the BARRIER it was run at and its own logs
ARGS=()
NSETS=0
TOTAL=0
for d in $(ls -d $ROOT/T$T/barrier*/ | sed 's#/$##' | sort -t r -k3 -g); do
    B=$(basename $d); B=${B#barrier}
    mapfile -t COL < <(ls $d/s[0-9]*/run_*.colvar.dat 2>/dev/null)
    [ ${#COL[@]} -eq 0 ] && { echo "no COLVAR files in $d"; exit 1; }
    LOGS=()
    for c in "${COL[@]}"; do LOGS+=("${c%.colvar.dat}.plumed.log"); done
    ARGS+=("-i" "${COL[@]}" "--barrier" "$B" "--logfiles" "${LOGS[@]}")
    NSETS=$((NSETS + 1))
    TOTAL=$((TOTAL + ${#COL[@]}))
    echo "  set BARRIER=$B: ${#COL[@]} runs"
done
echo "start $(date -Iseconds) host=$(hostname) T=$T sets=$NSETS runs=$TOTAL"

CMD="eatr-flooding-analysis ${ARGS[*]} --temp $T --energyunit 1.0 --timeunit 1e-12 --tcol 0 \
  --vcol 6 --opesf --stride 1000 --subsample-min-points 200 --no-plots -q \
  -b --numboots 100 --seed 42 --threads 4 -o $OUT"
printf 'temp=%s sets=%s runs=%s date=%s\n' "$T" "$NSETS" "$TOTAL" "$(date -Iseconds)" > $OUTDIR/$TAG.cmd
echo "$CMD" >> $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $OVL:ro --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup \
  $SIF /bin/bash -c "source /ext3/env.sh; $CMD"
RC=$?
echo "end $(date -Iseconds) rc=$RC"
exit $RC
