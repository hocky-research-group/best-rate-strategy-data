#!/bin/bash
#SBATCH --job-name=eatr_grid
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_grid_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_grid_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR + iMetaD CDF fits with trajectory-count subsampling for every iMetaD
# condition listed in subsampling/imetad_grid/conditions.txt
# (one line per condition: CV BF HEIGHT PACE [SIZES]; a fifth field, a comma-separated
# list, replaces the default subset sizes for that condition).
# Acceleration column: read from the COLVAR header (metad.acc), because the
# templates differ (bias factors 1-2 add metad.work; Rg lacks CALC_RCT).
# Usage: sbatch --array=0-<N-1>%100 eatr_grid_subsample.sbatch
# Skips a condition whose JSON already exists (re-submit safe).
# Overrides (sbatch --export=ALL,NAME=value,...): LIST conditions file, OUTDIR output
# directory, SEED subset-draw seed (default 42), BOOT=0 drops the bootstrap of the
# full-set fit (the subset fits never bootstrap, so their values do not depend on it).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LIST=${LIST:-$P/subsampling/imetad_grid/conditions.txt}
OUTDIR=${OUTDIR:-$P/subsampling/imetad_grid/results}
SEED=${SEED:-42}
BOOTARGS="-b --numboots 100"; [ "${BOOT:-1}" = 0 ] && BOOTARGS=""
mkdir -p $OUTDIR

read CV BF HEIGHT PACE SIZES_LINE <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)"
RUNDIR=$SRC/prod_imetad/bias${CV}/bf${BF}/height${HEIGHT}/pace${PACE}
# acceleration column from the COLVAR header (0-based index of metad.acc); the
# layout differs between templates (CALC_RCT adds rbias/rct, CALC_WORK adds work)
HDR=$(head -1 $(ls $RUNDIR/s1/run_*.colvar.dat | head -1))
ACOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.acc") print i-3}')
[ -z "$ACOL" ] && { echo "no metad.acc in header: $HDR"; exit 1; }
TAG=${CV}_bf${BF}_height${HEIGHT}_pace${PACE}
OUT=$OUTDIR/$TAG.json
NSEEDS=$(ls -d $RUNDIR/s[0-9]* 2>/dev/null | wc -l)
if [ -n "$SIZES_LINE" ]; then SIZES=$SIZES_LINE; elif [ "$NSEEDS" -ge 150 ]; then SIZES=5,10,20,40,80; else SIZES=5,10,20,40; fi
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID cond=$TAG nseeds=$NSEEDS acol=$ACOL sizes=$SIZES seed=$SEED boot=${BOOT:-1} eatr_git=$GITHASH"
if [ -s "$OUT" ]; then echo "exists, skipping $OUT"; exit 0; fi
if [ ! -d "$RUNDIR" ]; then echo "missing $RUNDIR"; exit 1; fi

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol 5 --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME $BOOTARGS --threads ${THREADS:-8} --seed $SEED \
  --subsample-runs $SIZES --subsample-reps 10 -q -o $OUT"
printf '%s\neatr_git=%s seed=%s date=%s\n' "$CMD" "$GITHASH" "$SEED" "$(date -Iseconds)" > $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
  /bin/bash -c "source /ext3/env.sh; $CMD"
RC=$?
echo "end $(date -Iseconds) rc=$RC"
exit $RC
