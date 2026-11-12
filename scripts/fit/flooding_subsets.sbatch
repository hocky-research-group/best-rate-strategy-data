#!/bin/bash
#SBATCH --job-name=flood_subsets
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_subsets_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/flood_subsets_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_157_chemistry
# Seeded no-bootstrap subset refits of the flooding ladders and two-level pilots
# (see flooding_subsets.py).
# Usage: N=$(python flooding_subsets.py list | head -1); sbatch --array=0-$((N - 1)) flooding_subsets.sbatch
# Re-submit safe: fits already recorded in a task's file are skipped.
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
TMP=/tmp/gmh4_subsets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
module load anaconda3/2025.06
echo "start $(date -Iseconds) host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
python -u $P/scripts/flooding_subsets.py run $SLURM_ARRAY_TASK_ID $TMP
rc=$?
rm -rf $TMP
echo "end $(date -Iseconds) rc=$rc"
