#!/bin/bash
# Untempered (standard) metadynamics runs: the BIASFACTOR keyword is dropped from the
# METAD line, so every hill keeps its initial height. The grid of Triasha Pal covers
# bias factors 2-20 only; PLUMED's BIASFACTOR=1 is degenerate (hill heights vanish after
# the first hill) and is not untempered metadynamics.
#
# Usage (torch): DRY=1 bash setup_untempered_metad.sh   # create and check inputs only
#                bash setup_untempered_metad.sh         # create (if missing) and submit
#
# Output tree: best-rate-strategy-study/simulations/untempered_metad/bias<CV>/height<h>/pace<p>/s<k>
# The LAMMPS velocity seed of seed directory s<k> is 5000 + k, distinct from the grid
# (Triasha's seeds) and from the prospective runs (1000 + k).
# Everything else (force field, thermostat, committor criterion, hill width, grid) comes
# from the same templates as the grid, so the only difference is the missing BIASFACTOR.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
ROOT=$P/simulations/untempered_metad
CHUNK=${CHUNK:-25}
mkdir -p $ROOT
cat > $ROOT/run_chunk.sbatch <<'SB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=20:00:00
#SBATCH --account=torch_pr_157_chemistry
# Runs the seed directories given as arguments in sequence, skipping seeds whose PLUMED
# log already records a committed escape, so the job can be resubmitted safely.
# Usage (from the condition directory): sbatch --job-name <name> run_chunk.sbatch s1 s2 ...
for seed_dir in "$@"; do
    log=$(ls $seed_dir/*.plumed.log 2>/dev/null)
    if [ -f "$log" ] && grep -qs COMMITTED $log; then echo "skip $seed_dir"; continue; fi
    pushd $seed_dir > /dev/null
    SLURM_NTASKS=1 /projects/rps/gmh4/hockygroup/nm3787/lammps-nico-new/src/lammps_plumed_mpi.sh -var temp 312 -in run_proteinG_plumed.lammps.in > lammps_stdout.txt 2>&1
    echo "$seed_dir rc=$? $(date -Iseconds)"
    popd > /dev/null
done
echo "done $(date -Iseconds)"
SB
# columns: CV HEIGHT_KJ PACE_STEPS NSEEDS
while read CV HEIGHT PACE NSEEDS; do
  plumed_file=$SRC/imetad_${CV}.plumed.dat
  plumed_name=$(basename $plumed_file)
  lammps_name=$(basename $input_file)
  outdir=$ROOT/bias$CV/height$HEIGHT/pace$PACE
  base_prefix=run_proteinG_untempered_bias${CV}_height${HEIGHT}_pace${PACE}
  for k in $(seq 1 $NSEEDS); do
    seeddir=$outdir/s$k
    [ -d $seeddir ] && continue
    prefix=${base_prefix}_s$k
    mkdir -p $seeddir
    sed -e "s/__PREFIX__/$prefix/" -e "s/__HEIGHT__/$HEIGHT/" -e "s/__PACE__/$PACE/" \
        -e "s/ BIASFACTOR=__BF__//" $plumed_file > $seeddir/$plumed_name
    sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" -e "s/__PLUMEDFILE__/${plumed_name}/" \
        -e "s/__SEED__/$((5000 + k))/" $input_file > $seeddir/$lammps_name
    cp $start_file $pdb_file $seeddir/
  done
  cp $ROOT/run_chunk.sbatch $outdir/
  # checks on the first and last seed: no placeholder left, no BIASFACTOR, parameters and seed as intended
  for k in 1 $NSEEDS; do
    s=$outdir/s$k
    if grep -q "__" $s/$plumed_name $s/$lammps_name; then echo "PLACEHOLDER LEFT in $s"; exit 1; fi
    m=$(grep "^METAD" $s/$plumed_name)
    echo "$m" | grep -q "BIASFACTOR" && { echo "BIASFACTOR still set in $s: $m"; exit 1; }
    echo "$m" | grep -q "ACCELERATION" || { echo "METAD line lacks ACCELERATION in $s: $m"; exit 1; }
    for want in "HEIGHT=$HEIGHT " "PACE=$PACE "; do
      echo "$m" | grep -q "$want" || { echo "METAD line lacks '$want' in $s: $m"; exit 1; }
    done
    grep -q "create \${temp} $((5000 + k)) " $s/$lammps_name || { echo "wrong velocity seed in $s"; exit 1; }
    ls $s/go.pdb $s/$start_name > /dev/null || exit 1
  done
  echo "$outdir: $(ls -d $outdir/s[0-9]* | wc -l) seed dirs; METAD: $(grep '^METAD' $outdir/s1/$plumed_name)"
  [ "${DRY:-0}" = 1 ] && continue
  (cd $outdir && seeds=($(ls -d s[0-9]* | sort -V)) && \
   for ((i = 0; i < ${#seeds[@]}; i += CHUNK)); do
     sbatch --job-name ut_${CV}_h${HEIGHT}_p${PACE}_c$((i / CHUNK)) run_chunk.sbatch "${seeds[@]:i:CHUNK}"
   done)
done <<'EOF'
Q 0.1 1000 200
Q 0.1 10000 200
Q 0.1 100000 200
Q 0.5 1000 200
Q 0.5 10000 200
Q 0.5 100000 200
RMSD 0.1 1000 200
RMSD 0.1 10000 200
RMSD 0.1 100000 200
RMSD 0.5 1000 200
RMSD 0.5 10000 200
RMSD 0.5 100000 200
Rg 0.1 1000 200
Rg 0.1 10000 200
Rg 0.1 100000 200
Rg 0.5 1000 200
Rg 0.5 10000 200
Rg 0.5 100000 200
Ree 0.1 1000 200
Ree 0.1 10000 200
Ree 0.1 100000 200
Ree 0.5 1000 200
Ree 0.5 10000 200
Ree 0.5 100000 200
EOF
