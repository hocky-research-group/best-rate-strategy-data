#!/bin/bash
#SBATCH --job-name=eatr_untempered
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_untempered_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_untempered_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR + iMetaD CDF fits with trajectory-count subsampling for the untempered
# metadynamics conditions, with exactly the options used for the grid
# (scripts/eatr_grid_subsample.sbatch): stride 1000 with a 200-row floor,
# binomially weighted CDF, -ME, 100 bootstrap resamples, seed 42, subset sizes
# 5,10,20,40,80 with 10 repetitions.
# Conditions come from simulations/untempered_metad/conditions.txt, one line per
# condition: CV HEIGHT PACE (written by this script's --write-list mode).
# Acceleration column: read from the COLVAR header, as the Rg template has no
# CALC_RCT and so a different layout.
# Usage:
#   bash fit_untempered.sbatch --write-list      # refresh the condition list
#   sbatch --array=0-<N-1>%12 fit_untempered.sbatch
#   sbatch --export=ALL,LIST=<file> --array=0-<N-1>%12 fit_untempered.sbatch   # subset
# Skips a condition whose JSON already exists (re-submit safe).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
RUNROOT=$P/simulations/untempered_metad
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LIST=${LIST:-$RUNROOT/conditions.txt}   # override to fit a subset, e.g. conditions_complete.txt
OUTDIR=$P/subsampling/untempered/results
mkdir -p $OUTDIR

if [ "${1:-}" = "--write-list" ]; then
  : > $LIST
  for d in $RUNROOT/bias*/height*/pace*; do
    cv=$(basename $(dirname $(dirname $d))); cv=${cv#bias}
    h=$(basename $(dirname $d)); h=${h#height}
    p=$(basename $d); p=${p#pace}
    n=$(grep -l COMMITTED $d/s[0-9]*/*.plumed.log 2>/dev/null | wc -l)
    echo "$cv $h $p" >> $LIST
    echo "$cv h$h p$p: $n committed runs"
  done
  echo "wrote $LIST with $(wc -l < $LIST) conditions"
  exit 0
fi

read CV HEIGHT PACE <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)"
RUNDIR=$RUNROOT/bias${CV}/height${HEIGHT}/pace${PACE}
HDR=$(head -1 $(ls $RUNDIR/s1/run_*.colvar.dat | head -1))
ACOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.acc") print i-3}')
[ -z "$ACOL" ] && { echo "no metad.acc in header: $HDR"; exit 1; }
TAG=${CV}_untempered_height${HEIGHT}_pace${PACE}
OUT=$OUTDIR/$TAG.json
NSEEDS=$(ls -d $RUNDIR/s[0-9]* 2>/dev/null | wc -l)
NCOMMIT=$(grep -l COMMITTED $RUNDIR/s[0-9]*/*.plumed.log 2>/dev/null | wc -l)
if [ "$NSEEDS" -ge 150 ]; then SIZES=5,10,20,40,80; else SIZES=5,10,20,40; fi
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID cond=$TAG nseeds=$NSEEDS committed=$NCOMMIT acol=$ACOL sizes=$SIZES eatr_git=$GITHASH"
if [ -s "$OUT" ]; then echo "exists, skipping $OUT"; exit 0; fi
if [ ! -d "$RUNDIR" ]; then echo "missing $RUNDIR"; exit 1; fi

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol 5 --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME -b --numboots 100 --threads ${THREADS:-8} --seed ${SEED:-42} \
  --subsample-runs $SIZES --subsample-reps 10 -q -o $OUT"
printf '%s\neatr_git=%s seed=42 committed=%s date=%s\n' "$CMD" "$GITHASH" "$NCOMMIT" "$(date -Iseconds)" > $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
  /bin/bash -c "source /ext3/env.sh; $CMD"
RC=$?
echo "end $(date -Iseconds) rc=$RC"
exit $RC
