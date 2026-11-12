#!/bin/bash
#SBATCH --job-name=traj_summary
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/traj_summary_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/traj_summary_%A_%a.err
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_157_chemistry
# Per-trajectory summaries (cost, final bias, hills) for every family x CV.
# Usage: sbatch trajectory_summary.sbatch
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
FAM=(imetad imetad imetad imetad opes opes opes opes ves ves ves ves)
CVS=(Q RMSD Ree Rg Q RMSD Ree Rg Q RMSD Ree Rg)
module load anaconda3/2025.06
echo "start $(date -Iseconds) ${FAM[$SLURM_ARRAY_TASK_ID]} ${CVS[$SLURM_ARRAY_TASK_ID]}"
python $P/scripts/trajectory_summary.py ${FAM[$SLURM_ARRAY_TASK_ID]} ${CVS[$SLURM_ARRAY_TASK_ID]} $P/results/trajectory_summary
echo "end $(date -Iseconds) rc=$?"
