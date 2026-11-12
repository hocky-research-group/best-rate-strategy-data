#!/bin/bash
# One well-tempered metadynamics condition per temperature along Q, chosen by the design
# rule rather than scanned: bias factor, height and pace are solved from the growth law so
# that the bias reaches 2.5 kT below the apparent barrier at about 80 ns, at every
# temperature. The residual barrier is therefore held fixed while the boost the condition
# implies runs from 32 at 312 K to 8e5 at 270 K.
#
# These give the iMetaD and EATR estimates, which the static ladders of
# setup_opes_temperature_ladder.sh do not: a growing bias is what the design rule is
# stated for, and iMetaD has no meaning without one.
#
# Parameters, from scripts/design_wtmetad_temperature.py (growth law with the unbroadened
# hill overlap, h_eff = h0 sigma / sqrt(sigma^2 + 2 sigma_A^2), hill SIGMA 0.02):
#
#   T (K)   barrier (kT)   sigma_A   b    h0 (kJ/mol)   pace     predicted boost
#   300      8.26          0.0449    12   0.55          70000        316
#   285     11.00          0.0403    14   0.75          70000       4939
#   270     16.09          0.0325    16   1.09          70000     799579
#
# The same procedure at 312 K returns b = 12, h0 = 0.34, pace 70000 and an escape near
# 80 ns, which is the prospective condition already run and validated there (b = 12,
# h0 = 0.35, observed escape 81 ns, EATR error 0.17, gamma 1.00). That agreement is the
# check on the procedure; 312 K is therefore not repeated here.
#
# Usage (torch): DRY=1 bash setup_wtmetad_temperature.sh   # create and check inputs only
#                bash setup_wtmetad_temperature.sh         # create (if missing) and submit
#
# Every trajectory is its own array task, through scripts/run_seed_array.sbatch; THROTTLE
# caps how many run at once.
#
# The escape residual is set by how fast the bias fills, not by the target the parameters
# were solved for: the bias spends ln f longer at every residual when the fill rate is
# divided by f, so the run escapes at a residual higher by ln f. The first campaign here
# reached 0.85 to 1.75 kT against the 2.5 it was designed for, because escapes took 109 to
# 171 ns rather than the 80 ns assumed and the bias kept growing. Dividing the fill rate by
# 5.71 (PACE 400000 in place of 70000) raises those to 2.6 to 3.5 kT, inside the rule this
# work recommends, and raises the deposition interval from 0.7 to 4 ns, which is the rule's
# other condition. ROOT can be overridden so a second pace lands in its own tree.
#
# Output tree: best-rate-strategy-study/simulations/wtmetad_temperature/T<temp>/s<k>
# Velocity seeds are SEED_BASE + k, 9000 + k by default, distinct from every other campaign
# in this study; a campaign along another CV sets CV (the imetad_<CV> template of the source
# inputs, whose escape criterion is Q < 0.35 whichever CV is biased) and its own SEED_BASE.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
CV=${CV:-Q}
SEED_BASE=${SEED_BASE:-9000}
plumed_file=$SRC/imetad_$CV.plumed.dat
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
plumed_name=$(basename $plumed_file)
lammps_name=$(basename $input_file)
ROOT=${ROOT:-$P/simulations/wtmetad_temperature}
SCRIPTS=$P/scripts
THROTTLE=${THROTTLE:-250}
GROUP=${GROUP:-10}    # trajectories per array task; array tasks count against the submit limit
PACE=${PACE:-70000}
NSEEDS=${NSEEDS:-50}
# temperature:bias factor:height
CONDITIONS=${CONDITIONS:-"300:12:0.55 285:14:0.75 270:16:1.09"}

mkdir -p $ROOT
LIST=$ROOT/pending_trajectories.txt
: > $LIST

for cond in $CONDITIONS; do
  T=${cond%%:*}; rest=${cond#*:}; BF=${rest%%:*}; HEIGHT=${rest#*:}
  outdir=$ROOT/T$T
  base_prefix=run_proteinG_wtmetad_bias${CV}_T${T}_bf${BF}_h${HEIGHT}_pace${PACE}
  for k in $(seq 1 $NSEEDS); do
    seeddir=$outdir/s$k
    [ -d $seeddir ] && continue
    prefix=${base_prefix}_s$k
    mkdir -p $seeddir
    sed -e "s/__PREFIX__/$prefix/" -e "s/__HEIGHT__/$HEIGHT/" -e "s/__BF__/$BF/" \
        -e "s/__PACE__/$PACE/" -e "s/TEMP=312/TEMP=$T/" $plumed_file > $seeddir/$plumed_name
    sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" \
        -e "s/__PLUMEDFILE__/${plumed_name}/" -e "s/__SEED__/$((SEED_BASE + k))/" \
        $input_file > $seeddir/$lammps_name
    cp $start_file $pdb_file $seeddir/
  done
  # checks on the first seed of the condition
  s=$outdir/s1
  grep -q "__" $s/$plumed_name $s/$lammps_name && { echo "PLACEHOLDER LEFT in $s"; exit 1; }
  m=$(grep "^METAD" $s/$plumed_name)
  for want in "HEIGHT=$HEIGHT " "BIASFACTOR=$BF " "PACE=$PACE " "TEMP=$T "; do
    echo "$m" | grep -q "$want" || { echo "METAD line lacks '$want' in $s: $m"; exit 1; }
  done
  grep -q "BASIN_UL1=0.35" $s/$plumed_name || { echo "escape criterion changed in $s"; exit 1; }
  grep -q "^COMMITTOR" $s/$plumed_name || { echo "COMMITTOR missing in $s: runs would never stop"; exit 1; }
  grep -q "create \${temp} $((SEED_BASE + 1)) " $s/$lammps_name || { echo "wrong velocity seed in $s"; exit 1; }
  ls $s/go.pdb $s/$start_name > /dev/null || exit 1
  echo "CV=$CV T=$T b=$BF h0=$HEIGHT pace=$PACE: $(ls -d $outdir/s[0-9]* | wc -l) seeds; METAD: $m"
  [ "${DRY:-0}" = 1 ] && continue
  # the per-condition list is written beside the campaign, not to /tmp: the login node's
  # /tmp is shared and fills up, and a list that cannot be written submits nothing
  bash $SCRIPTS/run_seed_array.sbatch --write-list $ROOT/cond.$$ $T $outdir/s[0-9]* > /dev/null
  cat $ROOT/cond.$$ >> $LIST && rm -f $ROOT/cond.$$
done
echo "seed dirs: $(ls -d $ROOT/T*/s[0-9]* 2>/dev/null | wc -l)"
[ "${DRY:-0}" = 1 ] && exit 0
N=$(wc -l < $LIST)
NTASK=$(( (N + GROUP - 1) / GROUP ))
echo "submitting $N trajectories as $NTASK tasks of $GROUP, at most $THROTTLE at a time"
sbatch --export=ALL,LIST=$LIST,GROUP=$GROUP --array=1-$NTASK%$THROTTLE $SCRIPTS/run_seed_array.sbatch
