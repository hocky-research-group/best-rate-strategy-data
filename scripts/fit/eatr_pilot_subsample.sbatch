#!/bin/bash
#SBATCH --job-name=eatr_pilot
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_pilot_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_pilot_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# Timing pilot: EATR + iMetaD CDF fits with trajectory-count subsampling on
# four iMetaD conditions (good CV Q, poor CV Ree; fastest and slowest pace).
# Corrects the acceleration column per CV: metad.acc is column 8 for Q, RMSD,
# Ree (CALC_RCT adds rbias and rct) and column 6 for Rg.
# Usage: sbatch eatr_pilot_subsample.sbatch
# ---------------------------------------------------------------------------
set -u
CONDS=("Q 10 0.5 100" "Q 10 0.5 100000" "Ree 10 0.5 100" "Ree 10 0.5 100000")
read CV BF HEIGHT PACE <<< "${CONDS[$SLURM_ARRAY_TASK_ID]}"

SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

case $CV in Rg) ACOL=6;; *) ACOL=8;; esac
RUNDIR=$SRC/prod_imetad/bias${CV}/bf${BF}/height${HEIGHT}/pace${PACE}
OUTDIR=$P/subsampling/pilot
mkdir -p $OUTDIR
TAG=${CV}_bf${BF}_height${HEIGHT}_pace${PACE}
OUT=$OUTDIR/$TAG.json

GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID cond=$TAG eatr_git=$GITHASH"
ls $RUNDIR/s*/run_*.colvar.dat | wc -l

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol 5 --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME -b --numboots 100 --threads 8 --seed 42 \
  --subsample-runs 5,10,20,40,80 --subsample-reps 10 -q -o $OUT"
echo "$CMD" > $OUTDIR/$TAG.cmd
echo "eatr_git=$GITHASH seed=42 date=$(date -Iseconds)" >> $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
  /bin/bash -c "source /ext3/env.sh; $CMD"
echo "end $(date -Iseconds) rc=$?"
