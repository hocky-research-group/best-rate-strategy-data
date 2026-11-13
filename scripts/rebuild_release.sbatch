#!/bin/bash
#SBATCH --job-name=rebuild_release
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/rebuild_release_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/rebuild_release_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# Bring both release archives up to date with the study tree, in one job.
#
# Run after scripts/sync_release_code.sh has pushed the current code into the public archive,
# since every step below uses the archive's own copy of the scripts:
#   1. per-run summaries of the campaigns named in SUMMARISE: those with new runs, whose
#      run trees are still under simulations/; none by default
#   2. the public archive's data layer, and the campaign links over it
#   3. the raw archive's manifest, then a tarball for every condition that has none
#   4. the archive's own verification, logged to release/verify_<date>.log
#   5. the contents table and checksums
# A finished run leaves release/rebuild_<jobid>.done holding the exit status of each step.
# The job exits non-zero when any step did. Verification runs only on a tree every earlier
# step built, and the checksums are written only after it passes: a checksum list over a
# partial or unverified tree would certify it.
#
# The number of conditions without a tarball is checked against MAX_NEW before any is
# tarred: a manifest that suddenly lists hundreds of untarred conditions is a wrong pattern
# or a moved directory, not new data.
#
# Usage: sbatch rebuild_release.sbatch
#        SUMMARISE="<campaign> ..." MAX_NEW=10 sbatch --export=ALL rebuild_release.sbatch
# ---------------------------------------------------------------------------
set -u
P=${STUDY:-/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study}
PUB=$P/release/best-rate-strategy-data
RAW=$P/release/raw-runs
B=$PUB/scripts/build_release.sh
SUMMARISE=${SUMMARISE:-}
MAX_NEW=${MAX_NEW:-10}
STATUS=$P/release/rebuild_${SLURM_JOB_ID:-manual}.done
module load anaconda3/2025.06
cd $P || exit 1
: > $STATUS.partial
FAILED=0
note() { echo "$1 rc=$2" >> $STATUS.partial; echo "== $1 rc=$2 $(date -Iseconds)"; [ "$2" -eq 0 ] || FAILED=1; }
skip() { echo "$1 not run: an earlier step failed" | tee -a $STATUS.partial; }

echo "== start $(date -Iseconds) on $(hostname), code $(cat $PUB/SOURCE_COMMIT 2>/dev/null)"
if [ -n "$SUMMARISE" ]; then
    python $PUB/scripts/study_run_summary.py $P/results/study_run_summaries $SUMMARISE
    note summaries $?
else
    note summaries_none_requested 0
fi

bash $B public;  note public $?
bash $B links;   note links $?
bash $B raw;     note raw_manifest $?

n=0; missing=()
while IFS=$'\t' read -r name root cond; do
    n=$((n + 1))
    tag=$(printf '%s' "${cond#$root/}" | tr '/' '_')
    [ -s "$RAW/$name/$tag.tar.gz" ] || missing+=("$n")
done < $RAW/conditions.txt
echo "== $n conditions listed, ${#missing[@]} without a tarball: ${missing[*]:-none}"
rc=0
if [ ${#missing[@]} -gt $MAX_NEW ]; then
    echo "more than MAX_NEW=$MAX_NEW untarred conditions; none tarred"; rc=3
else
    for i in "${missing[@]:-}"; do
        [ -n "$i" ] || continue
        SLURM_ARRAY_TASK_ID=$i RAW=$RAW bash $PUB/scripts/run/tar_condition.sbatch || rc=$?
    done
fi
note tar_new_conditions $rc

if [ $FAILED -eq 0 ]; then
    (cd $PUB && PY=python bash scripts/verify_release.sh) > $P/release/verify_$(date +%F).log 2>&1
    note verify $?
    tail -4 $P/release/verify_$(date +%F).log
else
    skip verify
fi

if [ $FAILED -eq 0 ]; then bash $B contents; note contents $?; else skip contents; fi
mv $STATUS.partial $STATUS
echo "== end $(date -Iseconds)"
exit $FAILED
