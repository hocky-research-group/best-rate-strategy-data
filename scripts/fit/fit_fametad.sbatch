#!/bin/bash
#SBATCH --job-name=eatr_fametad
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_fametad_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatr_fametad_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR and iMetaD fits for the frequency-adaptive conditions, with the options used for the
# grid and the untempered conditions so the errors are comparable across families.
#
# FREQUENCY_ADAPTIVE adds metad.pace to the PRINT ARG=* columns. It is appended after
# metad.acc, so the bias and acceleration columns keep their usual places, but both are
# read from the COLVAR header anyway rather than assumed: the Rg template carries no
# CALC_RCT and so a shorter row.
#
# Conditions come from a list of "CV THETA" lines, written by --write-list over the run
# tree; a condition whose runs are not all committed is left out, since fitting a partly
# finished condition samples the fast tail and reports a rate that is too high.
#
# Usage:
#   bash fit_fametad.sbatch --write-list <list file>
#   sbatch --export=ALL,LIST=<list file> --array=0-<N-1>%12 fit_fametad.sbatch
# Skips a condition whose JSON already exists (re-submit safe).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
RUNROOT=$P/simulations/fametad
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OUTDIR=$P/subsampling/fametad/results
LIST=${LIST:-$RUNROOT/conditions.txt}
mkdir -p $OUTDIR

if [ "${1:-}" = "--write-list" ]; then
    out=${2:-$LIST}
    : > $out
    for d in $RUNROOT/bias*/theta*; do
        cv=$(basename $(dirname $d)); cv=${cv#bias}
        th=$(basename $d); th=${th#theta}
        n=$(ls -d $d/s[0-9]* 2>/dev/null | wc -l)
        c=$(grep -l COMMITTED $d/s[0-9]*/run_*.plumed.log 2>/dev/null | wc -l)
        if [ "$c" -eq "$n" ] && [ "$n" -gt 0 ]; then
            echo "$cv $th" >> $out
            echo "  $cv theta=$th: $c/$n committed, listed"
        else
            echo "  $cv theta=$th: $c/$n committed, SKIPPED"
        fi
    done
    echo "wrote $out with $(wc -l < $out) complete conditions"
    exit 0
fi

read CV TH <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)"
RUNDIR=$RUNROOT/bias${CV}/theta${TH}
HDR=$(head -1 $(ls $RUNDIR/s1/run_*.colvar.dat | head -1))
ACOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.acc") print i-3}')
VCOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.bias") print i-3}')
[ -z "$ACOL" ] && { echo "no metad.acc in header: $HDR"; exit 1; }
[ -z "$VCOL" ] && { echo "no metad.bias in header: $HDR"; exit 1; }

TAG=${CV}_fametad_theta${TH}
OUT=$OUTDIR/$TAG.json
NSEEDS=$(ls -d $RUNDIR/s[0-9]* 2>/dev/null | wc -l)
NCOMMIT=$(grep -l COMMITTED $RUNDIR/s[0-9]*/run_*.plumed.log 2>/dev/null | wc -l)
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) cond=$TAG nseeds=$NSEEDS committed=$NCOMMIT vcol=$VCOL acol=$ACOL eatr_git=$GITHASH"
if [ -s "$OUT" ]; then echo "exists, skipping $OUT"; exit 0; fi
[ "$NCOMMIT" -eq "$NSEEDS" ] || { echo "$NCOMMIT of $NSEEDS committed, not fitting a partial condition"; exit 1; }

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol $VCOL --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME -b --numboots 100 --threads ${THREADS:-8} --seed ${SEED:-42} \
  --subsample-runs 5,10,20,40,80 --subsample-reps 10 -q -o $OUT"
printf '%s\neatr_git=%s seed=42 committed=%s date=%s\n' \
  "$CMD" "$GITHASH" "$NCOMMIT" "$(date -Iseconds)" > $OUTDIR/$TAG.cmd

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF \
  /bin/bash -c "source /ext3/env.sh; $CMD"
RC=$?
echo "end $(date -Iseconds) rc=$RC"
exit $RC
