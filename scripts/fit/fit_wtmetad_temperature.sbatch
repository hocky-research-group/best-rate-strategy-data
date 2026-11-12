#!/bin/bash
#SBATCH --job-name=eatr_wt_temp
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_wt_temp_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_wt_temp_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR and iMetaD fits for the well-tempered temperature campaign, one array task per
# temperature. The fit options match the grid and the untempered conditions, so the errors
# are comparable across families; the subset sizes are smaller because these conditions hold
# 50 runs rather than 200.
#
# --temp is the temperature of the condition, not 312: the estimator converts the recorded
# bias to an acceleration with beta, and using the benchmark temperature here would rescale
# every bias by T/312 and bias the rate by orders of magnitude at the bottom of the ladder.
#
# The acceleration column is read from the COLVAR header, since the templates differ in
# whether they carry CALC_RCT. The COLVAR glob starts at run_, which excludes the bck.0.
# files PLUMED writes when a cancelled run is restarted; those hold a truncated trajectory
# and must never enter a fit.
#
# RUNROOT and OUTDIR may be overridden so a second pace is fitted into its own tree, and
# SUBS sets the subset sizes: 5,10,20,40 for the 50-run conditions, 5,10,20,40,80 once a
# condition holds 200 runs, as on the grid.
#
# Usage: sbatch --array=0-2 fit_wtmetad_temperature.sbatch
# Skips a temperature whose JSON already exists (re-submit safe).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
RUNROOT=${RUNROOT:-$P/simulations/wtmetad_temperature}
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OUTDIR=${OUTDIR:-$P/subsampling/wtmetad_temperature/results}
SUBS=${SUBS:-5,10,20,40}
mkdir -p $OUTDIR

TEMPS=(300 285 270)
T=${TEMPS[$SLURM_ARRAY_TASK_ID]}
RUNDIR=$RUNROOT/T$T
HDR=$(head -1 $(ls $RUNDIR/s1/run_*.colvar.dat | head -1))
ACOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.acc") print i-3}')
[ -z "$ACOL" ] && { echo "no metad.acc in header: $HDR"; exit 1; }
VCOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.bias") print i-3}')
[ -z "$VCOL" ] && { echo "no metad.bias in header: $HDR"; exit 1; }

TAG=wtmetad_T${T}
OUT=$OUTDIR/$TAG.json
NCOMMIT=$(grep -l COMMITTED $RUNDIR/s[0-9]*/run_*.plumed.log 2>/dev/null | wc -l)
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) T=$T committed=$NCOMMIT vcol=$VCOL acol=$ACOL eatr_git=$GITHASH"
if [ -s "$OUT" ]; then echo "exists, skipping $OUT"; exit 0; fi

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp $T --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol $VCOL --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME -b --numboots 100 --threads ${THREADS:-8} --seed ${SEED:-42} \
  --subsample-runs $SUBS --subsample-reps 10 -q -o $OUT"
printf '%s\neatr_git=%s seed=42 temp=%s committed=%s date=%s\n' \
  "$CMD" "$GITHASH" "$T" "$NCOMMIT" "$(date -Iseconds)" > $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
  /bin/bash -c "source /ext3/env.sh; $CMD"
RC=$?
echo "end $(date -Iseconds) rc=$RC"
exit $RC
