#!/bin/bash
#SBATCH --job-name=eatrf_highbarrier
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatrf_highbarrier_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/eatrf_highbarrier_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_157_chemistry
# EATR-flooding across the extended OPES ladder (5..13 from Triasha, NEW levels from the
# study) for one CV at pace 1000. Usage: sbatch --export=ALL,CV=Q,NEW="15 17" opes_highbarrier_eatrf.sbatch
# BOOT=0 fits the full data once without bootstrap (set_reports then describe the full data,
# see refit_flooding_full_data.py) and names the output *_nobootstrap.json.
set -u
CV=${CV:?set CV}
NEW=${NEW:-"15 17"}
LAST=${NEW##* }
BOOT=${BOOT:-1}
BOOTARGS="-b --numboots 200"; SUFFIX=""
[ "$BOOT" = 0 ] && { BOOTARGS=""; SUFFIX="_nobootstrap"; }
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/prod_opes
SW=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OUT=$P/results/opes_highbarrier/${CV}_pace1000_barriers5-${LAST}${SUFFIX}.json
mkdir -p $(dirname $OUT)
ARGS=""
for B in 5 7 9 11 13; do
  ARGS="$ARGS --input-glob '$SRC/bias$CV/barrier$B/pace1000/s*/run_*.colvar.dat' --barrier $B --logfiles-glob '$SRC/bias$CV/barrier$B/pace1000/s*/run_*.plumed.log'"
done
for B in $NEW; do
  D=$P/simulations/opes_highbarrier/bias$CV/barrier$B/pace1000
  ARGS="$ARGS --input-glob '$D/s*/run_*.colvar.dat' --barrier $B --logfiles-glob '$D/s*/run_*.plumed.log'"
done
CMD="eatr-flooding-analysis $ARGS --opesf --temp 312 --energyunit 1.0 --timeunit 1e-12 --tcol 0 --vcol 6 \
  --stride 1000 --subsample-min-points 200 --seed 12345 $BOOTARGS --threads 8 --no-plots -q -o $OUT"
echo "$CMD" > ${OUT%.json}.cmd
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '#!/bin/bash\nsource /ext3/env.sh\n%s\n' "$CMD" > ${OUT%.json}.sh
singularity exec --overlay $SW/torch-eatr-python-15GB-500K.ext3:ro --bind /projects/rps/gmh4/hockygroup:/projects/rps/gmh4/hockygroup $SIF /bin/bash ${OUT%.json}.sh
echo "end $(date -Iseconds) rc=$?"; rm -f ${OUT%.json}.sh
