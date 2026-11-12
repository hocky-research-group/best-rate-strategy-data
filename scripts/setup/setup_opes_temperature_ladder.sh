#!/bin/bash
# OPES-flooding ladders along Q at 312, 300, 285 and 270 K, with the residual barrier held
# fixed across temperatures and the boost therefore varying by more than two orders of
# magnitude. This is the experiment the benchmark temperature cannot perform on its own: at
# 312 K every usable fill gives a boost below about 150, where "keep the boost small" and
# "leave part of the barrier standing" predict the same thing, so the two cannot be told
# apart.
#
# The bias is deposited everywhere along Q, with no excluded region. The 312 K ladder of the
# main study excludes Q < 0.65 so that nothing is deposited beyond the barrier top, but the
# top moves to lower Q as the temperature falls -- 0.59 at 312 K against 0.43 at 270 K --
# while the cut does not, so a fixed cut strands a growing part of the barrier: at 270 K only
# 11.3 kT of the 15.1 kT barrier can be reached and no rung of this ladder is attainable.
# Depositing everywhere removes that ceiling. It is expected to cost biasing efficiency,
# since bias that reaches the transition region perturbs the escape it is meant to time, and
# gamma is the quantity that registers the cost. The fills here stop well short of the
# barrier top, so the perturbation is small.
#
# 312 K is therefore repeated in this campaign rather than taken from the main study: those
# runs used the excluded region and are not a like-for-like control. Run against them, the
# new 312 K ladder also measures what dropping the excluded region costs at fixed
# temperature, which is the one place both variants exist.
#
# The rungs are the residual barriers of the existing 312 K ladder (BARRIER 5, 7, 9, 11 and
# 13 kJ/mol against the apparent barrier along Q, 5.95 kT = 15.44 kJ/mol), namely 4.02,
# 3.25, 2.48, 1.71 and 0.94 kT. At each temperature BARRIER = (barrier_kT - residual_kT) *
# kB T, so at 312 K this reproduces 5, 7, 9, 11 and 13 kJ/mol exactly and the only
# difference from the main-study ladder is the excluded region.
#
# The residual a run actually achieves is measured from its own bias afterwards; the rungs
# only decide where the ladder is placed.
#
# Barriers per temperature are the second-half reweighted values of the temperature scan
# (analysis/fes_temperature_scan.csv), scaled onto the 312 K unbiased PMF by the one factor
# that anchor supplies; scripts/design_wtmetad_temperature.py prints them. They are stated
# here so a run directory records the number its BARRIER was derived from. The achieved
# fill is measured from each run afterwards, so an error in these barriers shifts the rungs
# but does not corrupt the analysis.
#
# Every trajectory is its own array task, through scripts/run_seed_array.sbatch: escape
# times across a ladder of residual barriers differ by orders of magnitude, so running a
# rung's seeds in sequence makes the campaign as slow as its slowest rung and leaves the
# machine idle. THROTTLE caps how many run at once.
#
# Usage (torch): DRY=1 bash setup_opes_temperature_ladder.sh    # create and check only
#                PILOT=1 bash setup_opes_temperature_ladder.sh  # one seed per rung
#                bash setup_opes_temperature_ladder.sh          # create and submit
#
# Output tree: best-rate-strategy-study/simulations/opes_temperature/T<temp>/barrier<B>/s<k>
# Velocity seeds are 8000 + k, distinct from every other campaign in this study.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
SRC=/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension/inputs
input_file=$SRC/run_proteinG_plumed.lammps.in
plumed_file=$SRC/opes_Q.plumed.dat
start_file=$SRC/new_data_file.lammps
pdb_file=$SRC/go.pdb
start_name=$(basename $start_file)
plumed_name=$(basename $plumed_file)
lammps_name=$(basename $input_file)
ROOT=$P/simulations/opes_temperature
SCRIPTS=$P/scripts
THROTTLE=${THROTTLE:-250}
GROUP=${GROUP:-10}    # trajectories per array task; array tasks count against the submit limit
PACE=${PACE:-1000}
NSEEDS=${NSEEDS:-50}
[ "${PILOT:-0}" = 1 ] && NSEEDS=1
KB=0.0083144626
RESIDUALS="4.02 3.25 2.48 1.71 0.94"      # kT, matched to the 312 K ladder
BARRIER_KT_312=5.95      # the unbiased PMF, which is the anchor for the others
BARRIER_KT_300=8.26
BARRIER_KT_285=11.00
BARRIER_KT_270=16.09
TEMPS=${TEMPS:-"312 300 285 270"}

mkdir -p $ROOT
LIST=$ROOT/pending_trajectories.txt
: > $LIST

for T in $TEMPS; do
  eval bkt=\$BARRIER_KT_$T
  for res in $RESIDUALS; do
    barrier=$(awk -v b=$bkt -v r=$res -v kb=$KB -v t=$T 'BEGIN{printf "%.1f", (b-r)*kb*t}')
    outdir=$ROOT/T$T/barrier$barrier
    base_prefix=run_proteinG_opes_biasQ_T${T}_barrier${barrier}_pace${PACE}
    for k in $(seq 1 $NSEEDS); do
      seeddir=$outdir/s$k
      [ -d $seeddir ] && continue
      prefix=${base_prefix}_s$k
      mkdir -p $seeddir
      # OPES: set the fill target, the deposition pace and the temperature of this rung, and
      # drop the excluded region so the whole barrier can be filled. The excl CUSTOM stays
      # defined and so stays in the PRINT ARG=* columns, keeping the COLVAR layout of the
      # 312 K runs this ladder is compared against.
      sed -e "s/__PREFIX__/$prefix/" -e "s/__BARRIER__/$barrier/" -e "s/__PACE__/$PACE/" \
          -e "s/TEMP=312/TEMP=$T/" -e "s/ EXCLUDED_REGION=excl//" $plumed_file > $seeddir/$plumed_name
      sed -e "s/__INPUT__/${start_name}/" -e "s/__PREFIX__/$prefix/" \
          -e "s/__PLUMEDFILE__/${plumed_name}/" -e "s/__SEED__/$((8000 + k))/" \
          $input_file > $seeddir/$lammps_name
      cp $start_file $pdb_file $seeddir/
    done
    # checks on the first seed of each rung
    s=$outdir/s1
    grep -q "__" $s/$plumed_name $s/$lammps_name && { echo "PLACEHOLDER LEFT in $s"; exit 1; }
    m=$(grep "^OPES_METAD" $s/$plumed_name)
    for want in "BARRIER=$barrier " "PACE=$PACE " "TEMP=$T "; do
      echo "$m" | grep -q "$want" || { echo "OPES line lacks '$want' in $s: $m"; exit 1; }
    done
    echo "$m" | grep -q "EXCLUDED_REGION" && { echo "excluded region still set in $s: $m"; exit 1; }
    grep -q "^excl:" $s/$plumed_name || { echo "excl dropped in $s: COLVAR columns would shift"; exit 1; }
    grep -q "BASIN_UL1=0.35" $s/$plumed_name || { echo "escape criterion changed in $s"; exit 1; }
    grep -q "^COMMITTOR" $s/$plumed_name || { echo "COMMITTOR missing in $s: runs would never stop"; exit 1; }
    grep -q "create \${temp} 8001 " $s/$lammps_name || { echo "wrong velocity seed in $s"; exit 1; }
    ls $s/go.pdb $s/$start_name > /dev/null || exit 1
    echo "T=$T residual=$res kT -> BARRIER=$barrier kJ/mol, $(ls -d $outdir/s[0-9]* | wc -l) seeds"
    [ "${DRY:-0}" = 1 ] && continue
    bash $SCRIPTS/run_seed_array.sbatch --write-list $ROOT/rung.$$ $T $outdir/s[0-9]* > /dev/null
    cat $ROOT/rung.$$ >> $LIST && rm -f $ROOT/rung.$$
  done
done
echo "seed dirs: $(ls -d $ROOT/T*/barrier*/s[0-9]* 2>/dev/null | wc -l)"
[ "${DRY:-0}" = 1 ] && exit 0
N=$(wc -l < $LIST)
NTASK=$(( (N + GROUP - 1) / GROUP ))
echo "submitting $N trajectories as $NTASK tasks of $GROUP, at most $THROTTLE at a time"
sbatch --export=ALL,LIST=$LIST,GROUP=$GROUP --array=1-$NTASK%$THROTTLE $SCRIPTS/run_seed_array.sbatch
