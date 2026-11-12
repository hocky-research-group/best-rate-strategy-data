#!/bin/bash
#SBATCH --job-name=ut_growth
#SBATCH --output=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/ut_growth_%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study/logs/ut_growth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_157_chemistry
# ---------------------------------------------------------------------------
# Fill reached before escape in every untempered run, for the b -> infinity limit of the
# growth law.
#
# The well-tempered test recovers the fill from the smallest recorded hill height, inverting
# h_n = h0 exp(-V_n/((b-1)kT)). Untempered hills never decay, so that inversion does not
# exist and the fill has to be read from the bias itself: metad.bias is the bias at the
# current position, so its maximum over a trajectory is the depth the basin was filled to
# before the system left. The last hill is not used for the same reason as in the tempered
# case, that it sits at the escape point and understates the fill.
#
# The growth law V_n = (b-1) kT ln(1 + n h_eff/((b-1) kT)) becomes V_n = n h_eff as b grows,
# so untempered runs test the per-hill increment with no logarithm and no tempering: the
# fill divided by the hill count is h_eff directly. Equation 11's increment,
# h0 sigma/sqrt(sigma^2 + 2 b sigma_A^2), vanishes in this limit and makes no prediction;
# the unbroadened increment, h0 sigma/sqrt(sigma^2 + 2 sigma_A^2), is independent of b and
# is the one the tempered grid selects.
#
# The bias column is read from each COLVAR header, since the Rg template carries no
# CALC_RCT and so a shorter row; the glob starts at run_ so restart backups are excluded.
#
# Usage: sbatch untempered_growth_law.sbatch
# Output: analysis/untempered_growth_law_runs.csv, one row per run.
# ---------------------------------------------------------------------------
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
ROOT=$P/simulations/untempered_metad
OUT=$P/analysis/untempered_growth_law_runs.csv
mkdir -p $P/analysis

echo "cv,height,pace,seed,t_ps,V_max_kJmol" > $OUT
n=0
for d in $ROOT/bias*/height*/pace*; do
    cv=$(basename $(dirname $(dirname $d))); cv=${cv#bias}
    h=$(basename $(dirname $d)); h=${h#height}
    p=$(basename $d); p=${p#pace}
    first=$(ls $d/s1/run_*.colvar.dat 2>/dev/null | head -1)
    [ -z "$first" ] && { echo "no COLVAR in $d/s1"; continue; }
    # bias column index (1-based awk field) from this condition's own header
    vcol=$(head -1 $first | awk '{for(i=3;i<=NF;i++) if($i=="metad.bias") print i-2}')
    [ -z "$vcol" ] && { echo "no metad.bias in $first"; exit 1; }
    for f in $d/s[0-9]*/run_*.colvar.dat; do
        s=$(basename $(dirname $f))
        awk -v cv=$cv -v h=$h -v p=$p -v s=$s -v c=$vcol \
            'NR>1 { if ($c > m) m = $c; t = $1 }
             END { if (NR > 1) printf "%s,%s,%s,%s,%.1f,%.4f\n", cv, h, p, s, t, m }' $f >> $OUT
        n=$((n + 1))
    done
    echo "$cv h$h p$p: vcol=$vcol, $(ls -d $d/s[0-9]* | wc -l) runs"
done
echo "wrote $OUT with $(( $(wc -l < $OUT) - 1 )) runs of $n files, $(date -Iseconds)"
