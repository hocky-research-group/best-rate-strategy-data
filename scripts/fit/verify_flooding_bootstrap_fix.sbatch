#!/bin/bash
#SBATCH --job-name=verify_boot_fix
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/verify_boot_fix_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/verify_boot_fix_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_157_chemistry
# Verify the flooding bootstrap fix on protein G ladders (see verify_flooding_bootstrap_fix.py).
# Scratch is node-local and removed at the end. Usage: sbatch verify_flooding_bootstrap_fix.sbatch
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
TMP=/tmp/gmh4_verify_${SLURM_JOB_ID}
mkdir -p $TMP
module load anaconda3/2025.06
echo "start $(date -Iseconds) host=$(hostname)"
python -u $P/scripts/verify_flooding_bootstrap_fix.py $P/verification/flooding_bootstrap_fix $TMP
rc=$?
rm -rf $TMP
echo "end $(date -Iseconds) rc=$rc"
