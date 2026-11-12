#!/bin/bash
#SBATCH --job-name=refit_flood
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/refit_flood_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/refit_flood_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_157_chemistry
# No-bootstrap refits of every flooding ladder (see refit_flooding_full_data.py).
# Usage: N=$(python refit_flooding_full_data.py list); sbatch --array=0-$(( (N+CHUNK-1)/CHUNK - 1 )) --export=ALL,CHUNK=25 refit_flooding_full_data.sbatch
# Re-submit safe: fits already recorded in task_<k>.jsonl.gz are skipped.
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
TMP=/tmp/gmh4_refit_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
module load anaconda3/2025.06
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID chunk=$CHUNK"
python -u $P/scripts/refit_flooding_full_data.py run $SLURM_ARRAY_TASK_ID $CHUNK $TMP
rc=$?
rm -rf $TMP
echo "end $(date -Iseconds) rc=$rc"
