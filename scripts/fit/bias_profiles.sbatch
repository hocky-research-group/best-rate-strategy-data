#!/bin/bash
#SBATCH --job-name=bias_profiles
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/bias_profiles_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/bias_profiles_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# Average applied bias profiles for selected sets. Usage: sbatch bias_profiles.sbatch
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
module load anaconda3/2025.06
echo "start $(date -Iseconds)"
python $P/scripts/bias_profiles.py $P/results/bias_profiles
echo "end $(date -Iseconds) rc=$?"
