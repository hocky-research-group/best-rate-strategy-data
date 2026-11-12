#!/bin/bash
# Set up and submit OPES-flooding runs above the apparent barrier at pace 1000 steps,
# 50 seeds each, reusing Triasha's templates unchanged.
# Usage (torch): bash setup_opes_highbarrier.sh ["CV ..."] ["BARRIER ..."]
# Defaults: CVs "Q Ree", BARRIER "15 17" (kJ/mol); existing seed directories are skipped.
# Output tree: best-rate-strategy-study/simulations/opes_highbarrier/bias<CV>/barrier<B>/pace1000/s<seed>/
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
PACE=1000
CVS=${1:-"Q Ree"}
BARRIERS=${2:-"15 17"}
NSEEDS=50
mkdir -p $P/simulations/opes_highbarrier
cat > $P/simulations/opes_highbarrier/run_seeds.sbatch <<'SB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_157_chemistry
# Runs every seed directory below the submission directory in sequence,
# skipping seeds whose PLUMED log already records a committed escape.
# Usage: sbatch --job-name <prefix>_allseeds run_seeds.sbatch <prefix>
prefix=$1
for seed_dir in $(ls -d s[0-9]*/); do
    plumed_log=$(ls $seed_dir/${prefix}*.plumed.log 2>/dev/null)
    if [ -f "$plumed_log" ] && grep -qs COMMITTED $plumed_log; then echo "skip $seed_dir"; continue; fi
    pushd $seed_dir > /dev/null
    SLURM_NTASKS=1 /projects/rps/gmh4/hockygroup/nm3787/lammps-nico-new/src/lammps_plumed_mpi.sh -var temp 312 -in run_proteinG_plumed.lammps.in > lammps_stdout.txt 2>&1
    popd > /dev/null
done
echo "done $(date -Iseconds)"
SB
for CV in $CVS; do
  plumed_file=$SRC/opes_${CV}.plumed.dat
  plumed_name=$(basename $plumed_file)
  for barrier in $BARRIERS; do
    outdir=$P/simulations/opes_highbarrier/bias$CV/barrier$barrier/pace$PACE
    base_prefix=run_proteinG_opes_bias${CV}_barrier${barrier}_pace${PACE}
    for seed in $(seq 1 $NSEEDS); do
      seeddir=$outdir/s$seed
      [ -d $seeddir ] && continue
      prefix=${base_prefix}_s$seed
      mkdir -p $seeddir
      sed -e "s/__PREFIX__/$prefix/" -e "s/__BARRIER__/$barrier/" -e "s/__PACE__/$PACE/" $plumed_file > $seeddir/$plumed_name
      sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" -e "s/__PLUMEDFILE__/${plumed_name}/" -e "s/__SEED__/$seed/" $input_file > $seeddir/$(basename $input_file)
      cp $start_file $pdb_file $seeddir/
    done
    cp $P/simulations/opes_highbarrier/run_seeds.sbatch $outdir/
    # dry check: no placeholder left, inputs present
    if grep -q "__" $outdir/s1/$plumed_name $outdir/s1/$(basename $input_file); then echo "PLACEHOLDER LEFT in $outdir"; exit 1; fi
    ls $outdir/s1/go.pdb $outdir/s1/$start_name > /dev/null || exit 1
    (cd $outdir && sbatch --job-name ${base_prefix}_allseeds run_seeds.sbatch $base_prefix)
  done
done
echo "seed dirs: $(ls -d $P/simulations/opes_highbarrier/bias*/barrier*/pace*/s[0-9]* | wc -l)"
