#!/bin/bash
#SBATCH --job-name=rate_seed
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/seed_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/seed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# One escape trajectory per array task, for any campaign in this study whose run
# directories hold a LAMMPS input driven by PLUMED with a COMMITTOR.
#
# Running the seeds of a condition in sequence makes the wall time of a campaign the
# wall time of its slowest condition times its seed count, which is hours when the
# same work spread one seed per task takes as long as a single trajectory. Escape
# times vary by orders of magnitude across a ladder of residual barriers, so the
# sequential lanes are also badly unbalanced: the rungs that matter most, the ones
# with the most barrier left standing, are always the slowest.
#
# The log globs match run_*.plumed.log, not *.plumed.log: PLUMED keeps bck.N. copies of
# its log, and those sort before the live one, so a glob piped through head -1 returns a
# backup. A finished trajectory then looks unfinished, is run again, and overwrites its
# own completed output.
#
# LIST holds one line per trajectory, "<temperature K> <run directory>". Build it
# with --write-list over any set of run directories; directories whose PLUMED log
# already records a COMMITTED escape are left out, so a partially finished campaign
# can be relisted and resubmitted without repeating work.
#
# GROUP sets how many consecutive lines one task runs, so that a campaign of thousands of
# short trajectories does not need thousands of queued tasks: array tasks count one by one
# against the per-user submit limit, and a queue full of ten-second jobs serves nobody.
# Group only trajectories of the same condition, whose run lengths are comparable; grouping
# across conditions is what made the sequential lanes slow. Task n runs lines
# (n-1)*GROUP+1 through n*GROUP, so the array runs 1 to ceil(N/GROUP).
#
# Usage:
#   bash run_seed_array.sbatch --write-list <list file> <temperature> <run dir>...
#   sbatch --export=ALL,LIST=<list file>,GROUP=<g> --array=1-<ceil(N/g)>%250 run_seed_array.sbatch
# ---------------------------------------------------------------------------
set -u
LMP=/projects/rps/gmh4/hockygroup/nm3787/lammps-nico-new/src/lammps_plumed_mpi.sh

if [ "${1:-}" = "--write-list" ]; then
    out=$2
    T=$3
    shift 3
    : > $out
    for d in "$@"; do
        log=$(ls $d/run_*.plumed.log 2>/dev/null | head -1)
        if [ -n "$log" ] && grep -qs COMMITTED "$log"; then continue; fi
        echo "$T $d" >> $out
    done
    echo "$(wc -l < $out) trajectories listed in $out"
    exit 0
fi

GROUP=${GROUP:-1}
first=$(( (SLURM_ARRAY_TASK_ID - 1) * GROUP + 1 ))
last=$(( first + GROUP - 1 ))
ran=0; failed=0
for n in $(seq $first $last); do
    line=$(sed -n "${n}p" $LIST)
    [ -n "$line" ] || continue          # the last task is short when GROUP does not divide N
    T=${line%% *}
    seed_dir=${line#* }

    # a seed committed between listing and running is left alone rather than repeated
    log=$(ls $seed_dir/run_*.plumed.log 2>/dev/null | head -1)
    if [ -n "$log" ] && grep -qs COMMITTED "$log"; then echo "skip $seed_dir"; continue; fi

    echo "start $(date -Iseconds) line=$n T=$T $seed_dir"
    ( cd $seed_dir && SLURM_NTASKS=1 $LMP -var temp $T -in run_proteinG_plumed.lammps.in \
        > lammps_stdout.txt 2>&1 )
    rc=$?
    # the launcher can return 0 when LAMMPS itself stopped at once, so the escape marker in
    # the PLUMED log, not the exit code, decides whether the trajectory was run
    log=$(ls $seed_dir/run_*.plumed.log 2>/dev/null | head -1)
    if [ -n "$log" ] && grep -qs COMMITTED "$log"; then
        echo "done $(date -Iseconds) line=$n rc=$rc $seed_dir"
    else
        echo "FAILED $(date -Iseconds) line=$n rc=$rc $seed_dir: $(grep -m1 ERROR $seed_dir/lammps_stdout.txt 2>/dev/null | cut -c1-120)"
        failed=$((failed + 1))
    fi
    ran=$((ran + 1))
done
echo "task ${SLURM_ARRAY_TASK_ID} ran $ran of lines $first-$last, $failed without an escape"
[ "$failed" -eq 0 ] || exit 1
