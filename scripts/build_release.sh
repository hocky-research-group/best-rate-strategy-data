#!/bin/bash
# Assemble the release tree from the three directories this study drew on.
#
# Two archives come out of one build:
#
#   release/best-rate-strategy-data   the public archive, for GitHub and Zenodo. Inputs,
#                                     scripts, every fit and per-run summary, and the derived
#                                     tables and figures. The analysis and plotting scripts
#                                     run on it in place.
#   release/raw-runs                  the raw per-run output, tarred by condition, for Box or
#                                     institutional storage. Hundreds of GB, referenced from
#                                     the public archive's README rather than included.
#
# Runs are grouped by the kind of simulation, not by the campaign that produced them or the
# figure they end up in, because a reader looking for "untempered metadynamics along Q" should
# not have to know which experiment happened to need it. campaigns/ then holds symlinks that
# reassemble the experiments, so both views exist without duplicating a byte.
#
# Only COLVAR, HILLS and the PLUMED log are kept from each run. The rest of a run directory is
# the LAMMPS stdout log and 200 identical copies of the input files, which is 38% of the bulk
# and is already in inputs/ once.
#
# Sources, two of which are read-only and are never written to:
#   $STUDY   this study's own campaigns
#   $TRIASHA the metadynamics grid, the flooding ladders and the unbiased reference data
#   $NICO    the unbiased first-passage runs, and the LAMMPS and PLUMED build
#
# Usage (torch):
#   bash build_release.sh public    # inputs, scripts, fits, summaries, figures
#   bash build_release.sh raw       # write the manifest the tarring array works through
#   bash build_release.sh links     # campaign symlinks over the run tree
#   bash build_release.sh env       # the software environment, for the raw archive only
#   bash build_release.sh contents  # checksums and a table of contents for both archives
set -u
STUDY=${STUDY:-/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study}
TRIASHA=${TRIASHA:-/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension}
NICO=${NICO:-/projects/rps/gmh4/hockygroup/nm3787}
REL=${REL:-$STUDY/release}
PUB=$REL/best-rate-strategy-data
RAW=$REL/raw-runs
MODE=${1:-public}

# kind of run -> where its fits and summaries live in the study tree
say() { printf '  %-34s %s\n' "$1" "$2"; }

# cp -r copies INTO an existing destination, so a second build nests the source inside the
# first copy and every file is then read twice by anything that globs recursively. Replacing
# the destination makes the build idempotent, which it has to be: it is re-run whenever one
# campaign is corrected.
put() {   # put <source> <destination>
    [ -e "$1" ] || return 0
    rm -rf "$2"
    mkdir -p "$(dirname $2)"
    cp -r "$1" "$2"
}

build_public() {
    mkdir -p $PUB/runs $PUB/analysis $PUB/figures

    # --- per-run summaries and fits, by kind of simulation -------------------------------
    # one escape time, acceleration and fill per run: the layer the analysis actually reads
    for kind in wtmetad_grid wtmetad_untempered wtmetad_frequency_adaptive \
                wtmetad_prospective wtmetad_temperature opes_flooding ves_flooding \
                metad_pmf unbiased; do
        mkdir -p $PUB/runs/$kind
    done

    put $STUDY/results/trajectory_summary $PUB/runs/wtmetad_grid/trajectory_summary
    # per-run summaries of this paper's own campaigns (scripts/study_run_summary.py)
    put $STUDY/results/study_run_summaries $PUB/runs/_per_run_summaries
    put $STUDY/subsampling/imetad_grid $PUB/runs/wtmetad_grid/fits
    put $STUDY/subsampling/imetad_grid_intermediate_sizes $PUB/runs/wtmetad_grid/fits_intermediate_sizes
    say "wtmetad_grid" "$(find $PUB/runs/wtmetad_grid -type f | wc -l) files"

    put $STUDY/subsampling/untempered/results $PUB/runs/wtmetad_untempered/fits
    cp $STUDY/analysis/untempered_growth_law_runs.csv $PUB/runs/wtmetad_untempered/ 2>/dev/null
    say "wtmetad_untempered" "$(find $PUB/runs/wtmetad_untempered -type f | wc -l) files"

    put $STUDY/subsampling/fametad/results $PUB/runs/wtmetad_frequency_adaptive/fits
    say "wtmetad_frequency_adaptive" "$(find $PUB/runs/wtmetad_frequency_adaptive -type f | wc -l) files"

    put $STUDY/subsampling/pilot $PUB/runs/wtmetad_prospective/pilot_fits
    put $STUDY/results/prospective_wtmetad $PUB/runs/wtmetad_prospective/outcome
    say "wtmetad_prospective" "$(find $PUB/runs/wtmetad_prospective -type f | wc -l) files"

    for t in wtmetad_temperature wtmetad_temperature_slowpace wtmetad_temperature_slowpace_n200 \
             wtmetad_temperature_Ree; do
        put $STUDY/subsampling/$t/results $PUB/runs/wtmetad_temperature/${t}_fits
    done
    say "wtmetad_temperature" "$(find $PUB/runs/wtmetad_temperature -type f | wc -l) files"

    for t in flooding_subsets flooding_nsim flooding_pilot opes_temperature; do
        put $STUDY/subsampling/$t $PUB/runs/opes_flooding/$t
    done
    put $STUDY/results/opes_highbarrier $PUB/runs/opes_flooding/opes_highbarrier
    put $STUDY/refits_full_data $PUB/runs/opes_flooding/refits_full_data
    # the study tree packs per-task fits into one tar to spend fewer inodes on a shared
    # filesystem, but the analysis globs for the per-task archives, so the release holds them
    # loose. Data in an archive the scripts cannot read is data the archive does not really
    # contain. Every uncompressed tar under runs/ is such a pack and is unpacked, wherever it
    # sits and whatever it is called: a pattern on the directory or on the file name leaves
    # the next pack sealed, and the analysis then runs without error on part of the fits.
    for t in $(find $PUB/runs -name '*.tar'); do
        [ -e "$t" ] || continue
        tar -xf "$t" -C "$(dirname $t)" && rm -f "$t"
    done
    say "opes_flooding" "$(find $PUB/runs/opes_flooding -type f | wc -l) files"

    put $STUDY/results/ves_stage1_convergence_plots_triasha $PUB/runs/ves_flooding/stage1_convergence

    # The per-set flooding fits that the set-selection analysis reads sit inside the source
    # run tree rather than with the other fits. They are 57 small JSON files; copied with
    # their directory structure, since the analysis matches on the path, not the name alone.
    rsync -am --exclude='.ipynb_checkpoints/' --include='*/' --include='pace*_flooding.json' --exclude='*' \
        $TRIASHA/prod_opes/ $PUB/runs/opes_flooding/prod_opes/ 2>/dev/null
    rsync -am --exclude='.ipynb_checkpoints/' --include='*/' --include='combined_N*.json' --exclude='*' \
        $TRIASHA/prod_ves_cutoff/ $PUB/runs/ves_flooding/prod_ves_cutoff/ 2>/dev/null
    say "source per-set fits" "$(find $PUB/runs/opes_flooding/prod_opes $PUB/runs/ves_flooding/prod_ves_cutoff -name '*.json' 2>/dev/null | wc -l) files"
    say "ves_flooding" "$(find $PUB/runs/ves_flooding -type f | wc -l) files"

    # free energy scans: the profiles, not the trajectories that made them
    cp $STUDY/analysis/fes_temperature_scan*.csv $PUB/runs/metad_pmf/ 2>/dev/null
    put $STUDY/results/bias_profiles $PUB/runs/metad_pmf/bias_profiles
    put $STUDY/heuristics $PUB/runs/metad_pmf/heuristics
    say "metad_pmf" "$(find $PUB/runs/metad_pmf -type f | wc -l) files"

    # --- unbiased reference: the long trajectory and the first-passage set ---------------
    # the primary data behind both the PMF and the reference rate, so it is included whole
    U=$TRIASHA/unbiased_traj_data_nico
    mkdir -p $PUB/runs/unbiased/long_trajectory $PUB/runs/unbiased/first_passage
    # the long trajectory's CV file, not the 861 MB DCD beside it: the free energy surface is
    # built from the CVs and nothing here reads coordinates
    if [ -f $U/colvar_from_unbiased_traj/cvs.colvar ]; then
        gzip -c $U/colvar_from_unbiased_traj/cvs.colvar \
            > $PUB/runs/unbiased/long_trajectory/cvs.colvar.gz
    fi
    # first_pass is a symlink into the directory the runs were produced in, so the real path
    # is resolved before tarring; tarring the link archives one line of text
    FP=$(readlink -f $U/first_pass)
    if [ -d "$FP" ]; then
        # no coordinates: the rate comes from the CV time series and the PLUMED log, and a
        # DCD per run would multiply this archive severalfold for nothing the analysis reads.
        # --warning=no-file-changed because the parallel filesystem reports attribute changes
        # on files nothing is writing, which would otherwise make tar exit non-zero.
        tar -czf $PUB/runs/unbiased/first_passage/first_pass_runs.tar.gz \
            -C "$(dirname $FP)" --exclude='slurm*' --exclude='*.ipynb' --exclude='*.dcd' \
            --warning=no-file-changed "$(basename $FP)"
        # the rate analysis that sits beside the runs, kept loose so it can be read directly
        for f in comm.dat avg_comm.dat distr.dat getRate.py contact_plumed.dat; do
            [ -f "$FP/$f" ] && cp "$FP/$f" $PUB/runs/unbiased/first_passage/
        done
    fi
    put $STUDY/reference_rate $PUB/runs/unbiased/reference_rate
    # relaxation times of the CVs in the folded basin, from the first-passage runs
    put $STUDY/results/tau_from_fes $PUB/runs/unbiased/relaxation_times
    say "unbiased" "$(find $PUB/runs/unbiased -type f | wc -l) files"

    put $STUDY/verification $PUB/analysis/verification

    # reference_rate.json is derived from the first-passage times rather than copied, so the
    # archive holds the primary measurement and reproduces the number the whole paper is
    # scaled against. If this step is skipped every later script fails on the missing file,
    # which is the right failure: the reference rate is not an optional input.
    # the version of the predictions registered before the prospective runs, so the archive
    # can make the same check the repository makes against its own history
    if [ -d "$STUDY/../../.git" ] || [ -n "${PAPER_REPO:-}" ]; then
        git -C "${PAPER_REPO:-$STUDY}" show 4f11947:analysis/prospective_predictions.csv \
            > $PUB/analysis/prospective_predictions_registered.csv 2>/dev/null \
            && say "registered predictions" "extracted from commit 4f11947"
    fi
    module load anaconda3/2025.06 2>/dev/null || true
    if (cd $PUB && python3 scripts/analysis/reference_rate.py > /dev/null 2>&1); then
        say "reference rate" "derived from the first-passage times"
    else
        say "reference rate" "NOT DERIVED - run scripts/analysis/reference_rate.py"
    fi
    # bytecode caches and notebook checkpoints are by-products of running things in the tree,
    # not part of what it ships
    find $PUB \( -name __pycache__ -o -name .ipynb_checkpoints \) -type d -prune -exec rm -rf {} +
    echo "public archive: $(find $PUB -type f | wc -l) files, $(du -sh $PUB | cut -f1)"
}

build_raw() {
    # Write the manifest of conditions to tar, one per line, for the array in
    # tar_condition.sbatch to work through. Tarring 640 conditions in one job takes days;
    # one task per condition takes hours and retries only what failed.
    #
    # Bias factor 1 is excluded. PLUMED's BIASFACTOR=1 makes every hill after the first
    # vanish, so those runs are effectively unbiased: they take enormously longer to escape,
    # they are 920 GB of the grid's 1.7 TB, and no figure in the paper uses them. Their fits
    # are in the public archive; the trajectories are not retained.
    mkdir -p $RAW
    # the manifest being replaced: a campaign whose run tree is no longer on this filesystem
    # takes its lines from here
    local previous=$RAW/conditions.txt.previous
    if [ -s $RAW/conditions.txt ]; then cp $RAW/conditions.txt $previous; else : > $previous; fi
    : > $RAW/conditions.txt
    # A condition is pinned by depth AND shape. Depth alone counts the analysis output the
    # source trees keep beside the runs (eatr_results_bootstrap, nseeds_50) as conditions;
    # shape alone is worse, because find's -path wildcards match across / , so
    # "bias*/barrier*/pace*" also matches every run directory below a condition. Sorting is
    # left out: the manifest order does not matter, and sorting a million paths fills the
    # login node's /tmp.
    list() {   # list <destination name> <root> <depth> <pattern relative to root>
        local name=$1 root=$2 depth=$3 pattern=$4 n=0
        # The manifest is the record of where each tarball came from, and the tarballs outlive
        # the run trees. A root that is absent is a tree moved away after tarring, or a
        # filesystem that is briefly unreachable; either way its conditions are kept.
        if [ ! -d "$root" ]; then
            awk -F'\t' -v r="$root" '$2 == r' $previous >> $RAW/conditions.txt
            n=$(awk -F'\t' -v r="$root" '$2 == r' $previous | wc -l | tr -d ' ')
            echo "  $name: $n conditions kept from the previous manifest, source tree absent"
            return
        fi
        while read -r c; do
            case "$c" in *"/bf1/"*) continue ;; esac
            printf '%s\t%s\t%s\n' "$name" "$root" "$c" >> $RAW/conditions.txt
            n=$((n + 1))
        done < <(find $root -mindepth $depth -maxdepth $depth -type d -path "$root/$pattern" 2>/dev/null)
        echo "  $name: $n conditions"
    }
    list wtmetad_grid               $TRIASHA/prod_imetad 4 "bias*/bf*/height*/pace*"
    list opes_flooding_312K         $TRIASHA/prod_opes 3 "bias*/barrier*/pace*"
    list ves_flooding_cutoff        $TRIASHA/prod_ves_cutoff 2 "bias*/fc*"
    list wtmetad_untempered         $STUDY/simulations/untempered_metad 3 "bias*/height*/pace*"
    list wtmetad_frequency_adaptive $STUDY/simulations/fametad 2 "bias*/theta*"
    list opes_temperature           $STUDY/simulations/opes_temperature 2 "T*/barrier*"
    list wtmetad_temperature        $STUDY/simulations/wtmetad_temperature 1 "T*"
    list wtmetad_temperature_slow   $STUDY/simulations/wtmetad_temperature_slowpace 1 "T*"
    list wtmetad_temperature_Ree    $STUDY/simulations/wtmetad_temperature_Ree 1 "T*"
    list metad_pmf                  $STUDY/simulations/pmf_temperature 1 "T*"
    # Two campaigns were tarred per condition when they finished, inputs and LAMMPS logs
    # included, and have no loose run directories left to list. Their tarballs are taken as
    # they are, under the name the condition would have had.
    adopt() {   # adopt <destination name> <directory holding <condition>_runs.tar.gz>
        local name=$1 src=$2 n=0 t
        mkdir -p $RAW/$name
        for t in $src/*_runs.tar.gz; do
            [ -e "$t" ] || continue
            local out=$RAW/$name/$(basename ${t%_runs.tar.gz}).tar.gz
            if [ ! -s "$out" ] || [ "$(stat -c %s "$t")" != "$(stat -c %s "$out")" ]; then
                cp "$t" "$out.partial" && mv "$out.partial" "$out"
            fi
            n=$((n + 1))
        done
        echo "  $name: $n conditions, tarred at source"
    }
    adopt opes_flooding_312K_past_barrier $STUDY/simulations/opes_highbarrier
    adopt wtmetad_prospective             $STUDY/simulations/prospective_wtmetad
    rm -f $previous
    echo "manifest: $(wc -l < $RAW/conditions.txt) conditions -> $RAW/conditions.txt"
    echo "submit with: sbatch --array=1-$(wc -l < $RAW/conditions.txt)%200 scripts/run/tar_condition.sbatch"
}

build_env() {
    # The environment goes with the raw archive, never the public one: these are 46 GB of
    # filesystem images, and a public repository wants the inputs and the analysis, not a
    # copy of the operating system they ran on.
    #
    # The two .sif images are NYU's shared base images rather than ours. They are copied so
    # the archive stands alone, since a shared image can be retired or rebuilt; the overlays
    # hold what was installed on top of them and are the part that cannot be reconstructed.
    local E=$RAW/environment
    mkdir -p $E
    copy() {   # copy <source> <name> ; skipped when already present and the same size
        [ -e "$1" ] || { echo "  missing, skipped: $1"; return 0; }
        if [ -e "$E/$2" ] && [ "$(stat -c %s "$1")" = "$(stat -c %s "$E/$2")" ]; then
            echo "  already present: $2"; return 0
        fi
        cp "$1" "$E/$2" && echo "  $2 ($(du -h $E/$2 | cut -f1))"
    }
    copy $NICO/lammps-eten.ext3 lammps_plumed_overlay.ext3
    copy $STUDY/../eatr-software/torch-eatr-python-15GB-500K.ext3 eatr_analysis_overlay.ext3
    copy /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif simulation_base.sif
    copy /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif analysis_base.sif
    # the LAMMPS build the overlay is driven through, which is source and scripts, not an image
    if [ -d $NICO/lammps-nico-new ] && [ ! -s $E/lammps_build.tar.gz ]; then
        tar -czf $E/lammps_build.tar.gz -C $NICO --warning=no-file-changed lammps-nico-new \
            && echo "  lammps_build.tar.gz ($(du -h $E/lammps_build.tar.gz | cut -f1))"
    fi
    cat > $E/README.md <<'ENV'
# Software environment

The runs and the fits were executed inside Singularity containers. Each base image is NYU's
shared image; each overlay holds what this work installed on top of it, and is the part that
cannot be rebuilt from a package list.

| file | what it is |
|------|------------|
| `simulation_base.sif` | base image the trajectories ran in |
| `lammps_plumed_overlay.ext3` | LAMMPS with PLUMED 2.8.1, the build every trajectory used |
| `lammps_build.tar.gz` | that build's source and launch scripts |
| `analysis_base.sif` | base image the rate fits ran in |
| `eatr_analysis_overlay.ext3` | Python with the EATR analysis package |

A run is reproduced as the run scripts do it:

    singularity exec --overlay lammps_plumed_overlay.ext3:ro simulation_base.sif \
        /bin/bash -c "mpirun -np 1 <lammps>/lmp_mpi -var temp 312 -in run_proteinG_plumed.lammps.in"

The figure and table scripts in the public archive need none of this. They are ordinary
Python and run against the environment pinned in its pyproject.toml.
ENV
    echo "environment: $(find $E -type f | wc -l) files, $(du -sh $E | cut -f1)"
}

build_contents() {
    # A table saying what each directory of the raw archive holds, and a checksum for every
    # file of both archives. Both are written from the files as they are, so neither can go
    # stale against a README: the README points here for counts and sizes.
    {
        printf 'directory\ttarballs\tsize_GB\tfirst\n'
        for d in $(cd $RAW && ls -d */); do
            d=${d%/}
            printf '%s\t%s\t%s\t%s\n' "$d" "$(find $RAW/$d -maxdepth 1 -type f | wc -l)" \
                "$(du -s --apparent-size --block-size=1 $RAW/$d | awk '{printf "%.2f", $1 / 1e9}')" \
                "$(ls $RAW/$d | head -1)"
        done
    } > $RAW/CONTENTS.tsv
    echo "raw archive: $(($(wc -l < $RAW/CONTENTS.tsv) - 1)) directories -> $RAW/CONTENTS.tsv"
    cat > $RAW/README.md <<'RAWREADME'
# Raw runs

Per-run output of every biased simulation behind the paper, one tarball per condition. The
public archive (`best-rate-strategy-data`) holds the inputs, the scripts, one summary row per
run and the fit of every condition; this archive holds what those were computed from.

`CONTENTS.tsv` gives the number of tarballs and the size of each directory, `SHA256SUMS` a
checksum for every file, and `conditions.txt` the source directory each tarball was made from.

| directory | simulation | tarball name |
|-----------|------------|--------------|
| `wtmetad_grid` | well-tempered metadynamics grid at 312 K | `bias<CV>_bf<bias factor>_height<h0, kJ/mol>_pace<steps>` |
| `wtmetad_untempered` | metadynamics with hills that never decay | `bias<CV>_height<h0>_pace<steps>` |
| `wtmetad_frequency_adaptive` | deposition interval growing with the acceleration | `bias<CV>_theta<threshold>` |
| `wtmetad_prospective` | the two conditions predicted before they were run | `bias<CV>_bf<b>_height<h0>_pace<steps>` |
| `wtmetad_temperature` | one condition per temperature along Q, pace 70000 | `T<kelvin>` |
| `wtmetad_temperature_slow` | the same along Q at pace 400000 | `T<kelvin>` |
| `wtmetad_temperature_Ree` | one condition per temperature along R_ee, pace 400000 | `T<kelvin>` |
| `opes_flooding_312K` | OPES-flooding ladders, BARRIER 5 to 13 kJ/mol | `bias<CV>_barrier<kJ/mol>_pace<steps>` |
| `opes_flooding_312K_past_barrier` | levels of 15 to 21 kJ/mol at pace 1000 | `bias<CV>_barrier<kJ/mol>_pace1000` |
| `opes_temperature` | OPES-flooding ladders along Q down the temperature scan | `T<kelvin>_barrier<kJ/mol>` |
| `ves_flooding_cutoff` | VES cutoff flooding, static stage | `bias<CV>_fc<cutoff, kJ/mol>` |
| `metad_pmf` | free-energy scans along Q at each temperature | `T<kelvin>` |
| `environment` | container images, overlays and the LAMMPS build | see its own README |

CV is one of Q, RMSD, Rg, Ree. A tarball unpacks to the condition's directory with one
subdirectory `s<k>` per run, k being the run index.

Each run directory holds the COLVAR time series (`*.colvar.dat`), the deposited bias (`HILLS`
for metadynamics, `KERNELS` for OPES) and the PLUMED log, which records whether and when the
run committed. The LAMMPS log and the per-run copies of the input files are left out: the
inputs are in the public archive once, under `inputs/`. The tarballs of
`wtmetad_prospective` and `opes_flooding_312K_past_barrier` were made when those campaigns
finished and keep the inputs and LAMMPS logs as well.

The velocity seed of a run follows its index and is not stored in the tarball. In the grid
and in the 312 K OPES and VES ladders it equals k. In the campaigns generated for this paper
it is a campaign base plus k, the base being set in that campaign's setup script in the
public archive (`scripts/setup/`).

Runs at bias factor 1 are not here; see the public archive's README.
RAWREADME

    # Hashing the raw archive reads every byte of it, so a file already listed and not
    # modified since the list was written keeps its line.
    local sums=$RAW/SHA256SUMS new=$RAW/SHA256SUMS.new todo=$RAW/SHA256SUMS.todo
    : > $new; : > $todo
    (cd $RAW && find . -type f ! -name 'SHA256SUMS*' ! -name '*.partial' | sed 's|^\./||' | sort) | while read -r f; do
        if [ -s "$sums" ] && [ "$RAW/$f" -ot "$sums" ] && awk -v f="$f" '$2 == f {found = 1} END {exit !found}' "$sums"; then
            awk -v f="$f" '$2 == f' "$sums" >> $new
        else
            echo "$f" >> $todo
        fi
    done
    (cd $RAW && xargs -a $todo -r -d '\n' -P ${SLURM_CPUS_PER_TASK:-1} -n 8 sha256sum) >> $new
    sort -k2 $new > $sums && rm -f $new $todo
    echo "raw archive: $(wc -l < $sums) files -> $sums"

    (cd $PUB && find . -type f ! -name 'SHA256SUMS*' ! -path './.git/*' | sed 's|^\./||' | sort \
        | xargs -d '\n' -r sha256sum) > $PUB/SHA256SUMS.new && mv $PUB/SHA256SUMS.new $PUB/SHA256SUMS
    echo "public archive: $(wc -l < $PUB/SHA256SUMS) files hashed -> $PUB/SHA256SUMS"
}

build_links() {
    # the campaign view, as symlinks into the by-kind tree: no byte is stored twice
    C=$PUB/campaigns
    rm -rf $C; mkdir -p $C
    link() { mkdir -p $C/$1; ln -sfn ../../runs/$2 $C/$1/$(basename $2); }
    link estimator_comparison  wtmetad_grid
    link estimator_comparison  opes_flooding
    link estimator_comparison  ves_flooding
    link design_rule           wtmetad_prospective
    link design_rule           metad_pmf
    link growth_law            wtmetad_untempered
    link growth_law            wtmetad_grid
    link temperature_ladder    wtmetad_temperature
    link temperature_ladder    opes_flooding
    link temperature_ladder    metad_pmf
    link frequency_adaptive    wtmetad_frequency_adaptive
    link frequency_adaptive    wtmetad_grid
    link reference             unbiased
    echo "campaigns: $(find $C -type l | wc -l) links in $(ls $C | wc -l) groups"
}

case $MODE in
    public) build_public ;;
    raw)    build_raw ;;
    links)  build_links ;;
    env)    build_env ;;
    contents) build_contents ;;
    *)      echo "usage: build_release.sh [public|raw|links|env|contents]"; exit 1 ;;
esac
