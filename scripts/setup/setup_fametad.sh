#!/bin/bash
# Frequency-adaptive metadynamics: the deposition interval grows with the acceleration the
# run has already achieved, so the bias fills the basin quickly and then slows before it can
# reach the transition state,
#
#   tau_dep(t) = min[ tau_0 max(alpha(t)/theta, 1), tau_c ],
#
# with tau_0 = PACE, theta = FA_MIN_ACCELERATION and tau_c = FA_MAX_PACE (Wang, Valsson,
# Tiwary, Parrinello and Lindorff-Larsen, J. Chem. Phys. 149, 072309 (2018); PLUMED 2.8
# implements this equation directly, and adds a metad.pace component that records the
# schedule).
#
# The experiment is a scan of theta between two conditions of the existing grid. At bias
# factor 10 and height 0.5 the grid already holds pace 100 and pace 1e5 with 200 runs each,
# and they bracket the effect this method claims to remove: the fast pace costs a hundredth
# of the slow one and misses by 2.5 to 7.4 ln units, while the slow pace is accurate. Fixing
# tau_0 = 100 and tau_c = 1e5 makes those two conditions the endpoints of the scan, so
# nothing else changes between them and theta alone decides where a run falls.
#
# The ladder is placed from the acceleration a run actually reaches. Once alpha exceeds
# theta the interval is tau_0 alpha/theta, so the interval at escape is about
# tau_0 alpha_esc/theta, and it is alpha_esc that sets which theta matter. A pilot run at
# theta = 10 escaped at alpha = 110 with the interval at 1700 steps, far short of tau_c, so
# reaching the slow endpoint needs theta near tau_0 alpha_esc/tau_c = 0.11 and every theta
# above 1 is a variation on the fast endpoint. Hence 0.03 to 10: 0.03 saturates at tau_c
# almost at once and should reproduce the pace 1e5 condition, 10 stays close to pace 100.
# The prediction under test is a monotone crossover between the two endpoints; the useful
# outcome is a theta that keeps the accuracy of the slow pace at a cost near the fast one.
#
# Every trajectory is its own array task through scripts/run_seed_array.sbatch; THROTTLE
# caps how many run at once.
#
# Usage (torch): DRY=1 bash setup_fametad.sh   # create and check inputs only
#                bash setup_fametad.sh         # create (if missing) and submit
#
# Output tree: best-rate-strategy-study/simulations/fametad/bias<CV>/theta<theta>/s<k>
# Velocity seeds are 20000 + k, distinct from every other campaign in this study. As on the
# grid, the seed set is shared across conditions, so conditions differ only in the bias.
#
# Note for the analysis: FREQUENCY_ADAPTIVE adds metad.pace to the PRINT ARG=* columns, so
# the acceleration column moves. Read it from the COLVAR header rather than assuming the
# grid's layout.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
lammps_name=$(basename $input_file)
ROOT=$P/simulations/fametad
SCRIPTS=$P/scripts
THROTTLE=${THROTTLE:-250}
GROUP=${GROUP:-10}    # trajectories per array task; array tasks count against the submit limit
TEMP=312

CVS=${CVS:-"Q RMSD Rg Ree"}
THETAS=${THETAS:-"0.03 0.1 0.3 1 3 10"}
BF=${BF:-10}
HEIGHT=${HEIGHT:-0.5}
TAU0=${TAU0:-100}            # PACE, the interval before the acceleration threshold is met
TAUC=${TAUC:-100000}         # FA_MAX_PACE, the interval it saturates at
FAUPD=${FAUPD:-100}          # FA_UPDATE_FREQUENCY; PLUMED defaults this to PACE
NSEEDS=${NSEEDS:-200}

mkdir -p $ROOT
LIST=$ROOT/pending_trajectories.txt
: > $LIST

for CV in $CVS; do
  plumed_file=$SRC/imetad_${CV}.plumed.dat
  plumed_name=$(basename $plumed_file)
  for TH in $THETAS; do
    outdir=$ROOT/bias$CV/theta$TH
    base_prefix=run_proteinG_fametad_bias${CV}_bf${BF}_h${HEIGHT}_tau0${TAU0}_tauc${TAUC}_theta${TH}
    for k in $(seq 1 $NSEEDS); do
      seeddir=$outdir/s$k
      [ -d $seeddir ] && continue
      prefix=${base_prefix}_s$k
      mkdir -p $seeddir
      # the frequency-adaptive keywords go on the METAD line, which ends with LABEL=metad
      # in every one of the four CV templates
      sed -e "s/__PREFIX__/$prefix/" -e "s/__HEIGHT__/$HEIGHT/" -e "s/__BF__/$BF/" \
          -e "s/__PACE__/$TAU0/" \
          -e "s/ LABEL=metad/ FREQUENCY_ADAPTIVE FA_UPDATE_FREQUENCY=$FAUPD FA_MAX_PACE=$TAUC FA_MIN_ACCELERATION=$TH LABEL=metad/" \
          $plumed_file > $seeddir/$plumed_name
      sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" \
          -e "s/__PLUMEDFILE__/${plumed_name}/" -e "s/__SEED__/$((20000 + k))/" \
          $input_file > $seeddir/$lammps_name
      cp $start_file $pdb_file $seeddir/
    done
    # checks on the first and last seed of the condition
    for k in 1 $NSEEDS; do
      s=$outdir/s$k
      grep -q "__" $s/$plumed_name $s/$lammps_name && { echo "PLACEHOLDER LEFT in $s"; exit 1; }
      m=$(grep "^METAD" $s/$plumed_name)
      for want in "HEIGHT=$HEIGHT " "BIASFACTOR=$BF " "PACE=$TAU0 " "FREQUENCY_ADAPTIVE " \
                  "FA_UPDATE_FREQUENCY=$FAUPD " "FA_MAX_PACE=$TAUC " "FA_MIN_ACCELERATION=$TH " \
                  "ACCELERATION"; do
        echo "$m" | grep -q "$want" || { echo "METAD line lacks '$want' in $s: $m"; exit 1; }
      done
      # FREQUENCY_ADAPTIVE is driven by the acceleration factor, so ACCELERATION must be a
      # keyword of this METAD and not merely a substring of FA_MIN_ACCELERATION
      echo "$m" | grep -qE "(^| )ACCELERATION( |$)" || { echo "ACCELERATION not set in $s: $m"; exit 1; }
      grep -q "^COMMITTOR" $s/$plumed_name || { echo "COMMITTOR missing in $s: runs would never stop"; exit 1; }
      grep -q "BASIN_UL1=0.35" $s/$plumed_name || { echo "escape criterion changed in $s"; exit 1; }
      grep -q "create \${temp} $((20000 + k)) " $s/$lammps_name || { echo "wrong velocity seed in $s"; exit 1; }
      ls $s/go.pdb $s/$start_name > /dev/null || exit 1
    done
    echo "$CV theta=$TH: $(ls -d $outdir/s[0-9]* | wc -l) seeds; METAD: $(grep '^METAD' $outdir/s1/$plumed_name)"
    [ "${DRY:-0}" = 1 ] && continue
    bash $SCRIPTS/run_seed_array.sbatch --write-list $ROOT/fa.$$ $TEMP $outdir/s[0-9]* > /dev/null
    cat $ROOT/fa.$$ >> $LIST && rm -f $ROOT/fa.$$
  done
done
echo "seed dirs: $(ls -d $ROOT/bias*/theta*/s[0-9]* 2>/dev/null | wc -l)"
[ "${DRY:-0}" = 1 ] && exit 0
N=$(wc -l < $LIST)
NTASK=$(( (N + GROUP - 1) / GROUP ))
echo "submitting $N trajectories as $NTASK tasks of $GROUP, at most $THROTTLE at a time"
sbatch --export=ALL,LIST=$LIST,GROUP=$GROUP --array=1-$NTASK%$THROTTLE $SCRIPTS/run_seed_array.sbatch
