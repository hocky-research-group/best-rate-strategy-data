#!/bin/bash
#SBATCH --job-name=tar_cond
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/tar_cond_%A_%a.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/tar_cond_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# Tar one condition of the raw archive per array task.
#
# The manifest is written by build_release.sh raw, one line per condition:
#   <archive name><TAB><root the condition sits under><TAB><condition directory>
#
# Only COLVAR, the deposited bias and the PLUMED log are kept. The bias file is HILLS for
# metadynamics and KERNELS for OPES, and neither is excluded below. Everything else in a run directory is the
# LAMMPS stdout log, the slurm output, and one copy per run of the input files, which is 38%
# of the bulk and is in the public archive's inputs/ once. The stdout is named log.lammps in
# the source tree and lammps_stdout.txt in the campaigns run for this paper, so both are
# named here: excluding one and not the other would leave a reader wondering why some
# campaigns carry a file the rest do not. PLUMED's bck. backups of restarted
# runs hold truncated trajectories and are excluded everywhere in this work.
#
# A condition whose tarball already exists is skipped, so the array can be resubmitted after
# a timeout or a node failure and will only do what is left.
#
# Usage: sbatch --array=1-<N>%200 tar_condition.sbatch
#        with N the number of lines in the manifest.
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
RAW=${RAW:-$P/release/raw-runs}
LIST=${LIST:-$RAW/conditions.txt}

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $LIST)
[ -n "$line" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $LIST"; exit 1; }
name=$(printf '%s' "$line" | cut -f1)
root=$(printf '%s' "$line" | cut -f2)
cond=$(printf '%s' "$line" | cut -f3)

tag=$(printf '%s' "${cond#$root/}" | tr '/' '_')
out=$RAW/$name/$tag.tar.gz
mkdir -p $RAW/$name
if [ -s "$out" ]; then echo "exists, skipping $out"; exit 0; fi

echo "start $(date -Iseconds) $name/$tag"
# to a temporary name first, so an interrupted task never leaves a truncated tarball that
# the next run would mistake for a finished one
tar -czf "$out.partial" -C "$(dirname $cond)" \
    --exclude='log.lammps' --exclude='lammps_stdout.txt' --exclude='*.lammps.in' \
    --exclude='go.pdb' \
    --exclude='new_data_file.lammps' --exclude='*.plumed.dat' --exclude='slurm*' \
    --exclude='bck.*' --exclude='*.dcd' --warning=no-file-changed \
    "$(basename $cond)"
rc=$?
if [ $rc -le 1 ]; then          # 1 is the "file changed as we read it" notice on this filesystem
    mv "$out.partial" "$out"
    echo "done $(date -Iseconds) $(du -h $out | cut -f1) $name/$tag"
else
    rm -f "$out.partial"
    echo "FAILED rc=$rc $name/$tag"
fi
exit $([ $rc -le 1 ] && echo 0 || echo $rc)
