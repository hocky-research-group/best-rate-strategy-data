#!/bin/bash
# Converged free energy surfaces along Q at several temperatures, to measure how the
# apparent barrier scales with temperature. This tests the transferable form of the
# design rule (leave part of the barrier standing) without any unbiased runs: the rule
# is stated in units of the barrier, so the barrier is what has to be measured.
#
# Ordinary well-tempered metadynamics, not flooding: the COMMITTOR block is removed so a
# run keeps recrossing, and the run length is capped instead of ending at first escape.
# The FES is obtained afterwards from the HILLS file with `plumed sum_hills`.
#
# Usage (torch): DRY=1 bash setup_pmf_temperature_scan.sh   # create and check inputs only
#                bash setup_pmf_temperature_scan.sh         # create (if missing) and submit
#
# Output tree: best-rate-strategy-study/simulations/pmf_temperature/T<temp>/s<k>/
# Velocity seeds are 7000 + k, distinct from every other campaign in this study.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
plumed_file=$SRC/imetad_Q.plumed.dat
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
plumed_name=$(basename $plumed_file)
lammps_name=$(basename $input_file)
ROOT=$P/simulations/pmf_temperature
STEPS=${STEPS:-200000000}        # 2 us at 10 fs
BF=${BF:-20}                     # bias factor: high enough to cross at the lowest T
HEIGHT=${HEIGHT:-1.0}            # kJ/mol
PACE=${PACE:-500}                # steps; kinetics are not measured here, so filling can be fast
NSEEDS=${NSEEDS:-2}
mkdir -p $ROOT
cat > $ROOT/run_pmf.sbatch <<'SB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_157_chemistry
# Usage (from the seed directory): sbatch --job-name <name> run_pmf.sbatch <temperature>
set -u
T=$1
SLURM_NTASKS=1 /projects/rps/gmh4/hockygroup/nm3787/lammps-nico-new/src/lammps_plumed_mpi.sh \
  -var temp $T -in run_proteinG_plumed.lammps.in > lammps_stdout.txt 2>&1
echo "rc=$? $(date -Iseconds)"
SB
for T in $TEMPS; do
  for k in $(seq 1 $NSEEDS); do
    seeddir=$ROOT/T$T/s$k
    [ -d $seeddir ] && continue
    prefix=pmfscan_Q_T${T}_s$k
    mkdir -p $seeddir
    # PLUMED: drop the COMMITTOR block (the run must keep recrossing), set the metadynamics
    # parameters and write hills to a named file
    awk -v drop=0 '/^COMMITTOR \.\.\./{drop=1} drop==0{print} /^\.\.\. COMMITTOR/{drop=0}' $plumed_file \
      | sed -e "s/__PREFIX__/$prefix/" -e "s/__HEIGHT__/$HEIGHT/" -e "s/__BF__/$BF/" -e "s/__PACE__/$PACE/" \
            -e "s/TEMP=312/TEMP=$T/" \
            -e "s|^METAD \(.*\)LABEL=metad|METAD \1FILE=$prefix.hills LABEL=metad|" \
            -e "s/^PRINT ARG=\* FILE=\(.*\) STRIDE=100$/PRINT ARG=* FILE=\1 STRIDE=1000/" > $seeddir/$plumed_name
    sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" -e "s/__PLUMEDFILE__/${plumed_name}/" \
        -e "s/__SEED__/$((7000 + k))/" -e "s/^run             1000000000/run             $STEPS/" \
        $input_file > $seeddir/$lammps_name
    cp $start_file $pdb_file $seeddir/
  done
  # checks on the first seed of each temperature
  s=$ROOT/T$T/s1
  grep -q "__" $s/$plumed_name $s/$lammps_name && { echo "PLACEHOLDER LEFT in $s"; exit 1; }
  grep -q "COMMITTOR" $s/$plumed_name && { echo "COMMITTOR still present in $s"; exit 1; }
  m=$(grep "^METAD" $s/$plumed_name)
  for want in "HEIGHT=$HEIGHT " "BIASFACTOR=$BF " "PACE=$PACE " "TEMP=$T " ".hills"; do
    echo "$m" | grep -q "$want" || { echo "METAD line lacks '$want' in $s: $m"; exit 1; }
  done
  grep -q "^run             $STEPS" $s/$lammps_name || { echo "run length not set in $s"; exit 1; }
  grep -q "create \${temp} 7001 " $s/$lammps_name || { echo "wrong velocity seed in $s"; exit 1; }
  echo "T=$T: $(ls -d $ROOT/T$T/s[0-9]* | wc -l) seeds; METAD: $m"
  [ "${DRY:-0}" = 1 ] && continue
  for k in $(seq 1 $NSEEDS); do
    cp $ROOT/run_pmf.sbatch $ROOT/T$T/s$k/
    (cd $ROOT/T$T/s$k && sbatch --job-name pmf_T${T}_s$k run_pmf.sbatch $T)
  done
done
