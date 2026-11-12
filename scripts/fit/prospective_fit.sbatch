#!/bin/bash
#SBATCH --job-name=prospective_fit
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/prospective_fit_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/prospective_fit_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# EATR and iMetaD fits of the prospective WT-MetaD runs (setup_prospective_wtmetad.sh),
# with the estimator options of the grid (eatr_grid_subsample.sbatch) and the installed
# package. Array index selects the condition in CONDITIONS; run after every seed has
# committed or ended. Usage: sbatch --array=0-1 prospective_fit.sbatch
# Output: results/prospective_wtmetad/<CV>_bf<b>_height<h>_pace<p>.json and .cmd (command,
# git hash of the clone, blob of the installed rates_cmd.py, seed and committed counts).
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CONDITIONS=("Q 12 0.35 70000" "Ree 8 0.75 70000")
read CV BF HEIGHT PACE <<< "${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
RUNDIR=$P/simulations/prospective_wtmetad/bias$CV/bf$BF/height$HEIGHT/pace$PACE
OUTDIR=$P/results/prospective_wtmetad
mkdir -p $OUTDIR
TAG=${CV}_bf${BF}_height${HEIGHT}_pace${PACE}
HDR=$(head -1 $(ls $RUNDIR/s1/run_*.colvar.dat | head -1))
ACOL=$(echo "$HDR" | awk '{for(i=3;i<=NF;i++) if($i=="metad.acc") print i-3}')
[ -z "$ACOL" ] && { echo "no metad.acc in header: $HDR"; exit 1; }
NSEEDS=$(ls -d $RUNDIR/s[0-9]* | wc -l)
NCOMMIT=$(grep -l COMMITTED $RUNDIR/s*/*.plumed.log 2>/dev/null | wc -l)
GITHASH=$(cd $SW/EATR-rate-analysis && git rev-parse --short HEAD)
echo "start $(date -Iseconds) host=$(hostname) cond=$TAG seeds=$NSEEDS committed=$NCOMMIT acol=$ACOL eatr_git=$GITHASH"

CMD="eatr-analysis -g '$RUNDIR/s*/run_*.colvar.dat' --logfiles-glob '$RUNDIR/s*/run_*.plumed.log' \
  --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol 5 --acol $ACOL \
  --stride 1000 --subsample-min-points 200 --cdf-weights=binomial \
  -ME -b --numboots 100 --threads 8 --seed 42 -q -o $OUTDIR/$TAG.json"
BLOB='python -c "import hashlib, eatr_rates.rates_cmd as m; s = open(m.__file__, \"rb\").read(); print(\"rates_cmd_blob=\" + hashlib.sha1(b\"blob %d\" % len(s) + bytes([0]) + s).hexdigest())"'

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '#!/bin/bash\nsource /ext3/env.sh >/dev/null 2>&1\ncd /tmp\n%s\n%s\n' "$BLOB" "$CMD" > $OUTDIR/$TAG.sh
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro \
  --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF /bin/bash $OUTDIR/$TAG.sh > $OUTDIR/$TAG.log 2>&1
RC=$?
{ printf '%s\n' "$CMD"; echo "eatr_git=$GITHASH seeds=$NSEEDS committed=$NCOMMIT seed=42 date=$(date -Iseconds) rc=$RC"; grep "^rates_cmd_blob=" $OUTDIR/$TAG.log; } > $OUTDIR/$TAG.cmd
rm -f $OUTDIR/$TAG.sh
echo "end $(date -Iseconds) rc=$RC"
exit $RC
