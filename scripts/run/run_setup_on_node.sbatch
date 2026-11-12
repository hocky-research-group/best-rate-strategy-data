#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/setup_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# Run a campaign setup script on a compute node.
#
# A setup creates one directory per trajectory and copies the starting structure into each,
# which for a campaign of thousands of runs is tens of thousands of filesystem operations.
# On the login node that is slow enough to look hung and it is shared with everyone else's
# interactive work, so the setup belongs on a node like any other job. The setup submits the
# trajectory array itself when it finishes, and submitting from inside a job is fine.
#
# Usage: sbatch --job-name <name> run_setup_on_node.sbatch <setup script> [VAR=value ...]
# Any VAR=value arguments are exported before the script runs, so the setup's own
# environment knobs (DRY, CVS, THETAS, NSEEDS, THROTTLE, ...) work unchanged.
# ---------------------------------------------------------------------------
set -u
script=$1
shift
for kv in "$@"; do export "$kv"; done
echo "start $(date -Iseconds) host=$(hostname) script=$script args=$*"
bash "$script"
rc=$?
echo "end $(date -Iseconds) rc=$rc"
exit $rc
