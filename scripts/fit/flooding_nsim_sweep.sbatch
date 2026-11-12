#!/bin/bash
#SBATCH --job-name=flood_nsim
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_nsim_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_nsim_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR-flooding fits with N replicas per bias set, drawn at random (seeded)
# from the committed replicas, for the OPES-flooding data (sets = BARRIER
# 5,7,9,11,13 kJ/mol at fixed pace) and the VES cutoff-flooding data (sets =
# F_c 5..15 kJ/mol). Explicit file lists are passed (no symlinks).
# Task list: subsampling/flooding_nsim/tasks.txt, one line per task:
#   FAMILY CV PACE      (FAMILY = opes | ves; PACE = - for ves)
# Usage: sbatch --array=0-<N-1> flooding_nsim_sweep.sbatch
# Output: subsampling/flooding_nsim/<family>/<cv>[_pace<pace>]/N<n>_rep<r>.json
#         plus <same>.files listing the replicas used and the seed.
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
LIST=$P/subsampling/flooding_nsim/tasks.txt
read FAMILY CV PACE <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)"
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
BASE_SEED=12345
REPS=5

if [ "$FAMILY" = opes ]; then
  LABELS="5 7 9 11 13"; VCOL=6; OFFSET=""; SIZES="5 10 20 50"
  OUTDIR=$P/subsampling/flooding_nsim/opes/${CV}_pace${PACE}
  setdir() { echo $SRC/prod_opes/bias${CV}/barrier$1/pace${PACE}; }
  colvar_of() { echo $1/run_proteinG_opes_bias${CV}_barrier$2_pace${PACE}_$(basename $1).colvar.dat; }
  log_of() { echo ${1%.colvar.dat}.plumed.log; }
else
  LABELS="5 7 9 11 13 15"; VCOL=5; OFFSET="--nooffset"; SIZES="5 10 20 50 100 200"
  OUTDIR=$P/subsampling/flooding_nsim/ves/${CV}
  setdir() { echo $SRC/prod_ves_cutoff/bias${CV}/fc$1; }
  colvar_of() { echo $1/vesf_${CV}_fc$2_flood_$(basename $1).colvar.dat; }
  log_of() { echo ${1%.colvar.dat}.plumed.log; }
fi
mkdir -p $OUTDIR
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID family=$FAMILY cv=$CV pace=$PACE eatr_git=$GITHASH"

# pool of committed replicas per set
declare -A POOL
for L in $LABELS; do
  pf=$OUTDIR/pool_set$L.txt; : > $pf
  for s in $(setdir $L)/s[0-9]*; do
    c=$(colvar_of $s $L); l=$(log_of $c)
    [ -f "$c" ] && [ -f "$l" ] && grep -q COMMITTED "$l" && echo "$c" >> $pf
  done
  POOL[$L]=$pf; echo "set $L: $(wc -l < $pf) committed of $(ls -d $(setdir $L)/s[0-9]* | wc -l)"
done
MINAVAIL=$(for L in $LABELS; do wc -l < ${POOL[$L]}; done | sort -n | head -1)

seeded_source() { openssl enc -aes-256-ctr -pass "pass:$1" -nosalt </dev/zero 2>/dev/null; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for N in $SIZES; do
  [ "$N" -gt "$MINAVAIL" ] && { echo "N=$N > available $MINAVAIL, skip"; continue; }
  NREP=$REPS; [ "$N" -ge "$MINAVAIL" ] && NREP=1
  for R in $(seq 1 $NREP); do
    OUT=$OUTDIR/N${N}_rep${R}.json
    [ -s "$OUT" ] && { echo "exists $OUT"; continue; }
    SEED=$((BASE_SEED + 1000 * N + R))
    ARGS=""; : > $OUT.files; echo "# seed=$SEED N=$N rep=$R eatr_git=$GITHASH" >> $OUT.files
    for L in $LABELS; do
      FILES=$(shuf --random-source=<(seeded_source $SEED$L) -n $N ${POOL[$L]} | sort)
      LOGS=$(for f in $FILES; do log_of $f; done)
      printf '%s\n' $FILES >> $OUT.files
      ARGS="$ARGS -i $(echo $FILES) --barrier $L --logfiles $(echo $LOGS)"
    done
    CMD="eatr-flooding-analysis $ARGS $OFFSET --opesf --temp 312 --energyunit 1.0 --timeunit 1e-12 \
      --tcol 0 --vcol $VCOL --stride 1000 --subsample-min-points 200 --seed $SEED \
      -b --numboots 100 --threads 8 --no-plots -q -o $OUT"
    echo "N=$N rep=$R seed=$SEED $(date -Iseconds)"
    # the command can exceed the single-argument limit (200 files x 6 sets), so
    # it is written to a script and executed from inside the container
    printf '#!/bin/bash\nsource /ext3/env.sh\n%s\n' "$CMD" > $OUT.sh
    singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
      --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
      /bin/bash $OUT.sh || echo "FAILED N=$N rep=$R"
    rm -f $OUT.sh
  done
done
echo "end $(date -Iseconds)"
