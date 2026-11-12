"""No-bootstrap refits of every flooding ladder used in the paper (full-data set reports).

Usage (compute node, anaconda3/2025.06):
    python refit_flooding_full_data.py list                 # print the number of fits
    python refit_flooding_full_data.py run <task> <chunk> <tmpdir>

Before EATR-rate-analysis 14cee55, eatr-flooding-analysis with --bootstrap wrote the
set_reports and flooding_diagnostics of its LAST bootstrap resample; the headline
logk0, gamma and opes_logk0 were bootstrap means and are unaffected. A fit without
--bootstrap has always described the full data, so each ladder is refitted once without
it, on the same inputs and options as the original run, and only its set_reports and
flooding_diagnostics are used downstream.
Fits (in this order): Triasha Pal's OPES ladders (4 CVs x 8 paces), her VES subset
ladders (4 CVs x N = 10..200, via her subset_links), her pace ladders of the WT-MetaD
grid (4 CVs x 16 bias factor/height pairs), the 7-barrier OPES ladders (Q, Ree), and the
flooding_nsim subsampling fits (file lists recovered from flooding_nsim_raw.tar.gz).
Each task writes <study>/refits_full_data/task_<k>.jsonl.gz with one line
{"file": <path relative to the local data/ tree>, "data": <fit JSON>, "cmd": [...]}.
"""
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import tarfile

SRC = "/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension"
P = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study"
SW = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software"
OVL = f"{SW}/torch-eatr-python-15GB-500K.ext3"
SIF = "/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
BASE = ["--temp", "312", "--energyunit", "1.0", "--timeunit", "1e-12", "--tcol", "0", "--no-plots", "-q"]
THIN = ["--stride", "1000", "--subsample-min-points", "200"]
CVS = ("Q", "RMSD", "Ree", "Rg")
PACES = (100, 500, 1000, 5000, 10000, 30000, 50000, 100000)


def logs_of(colvars: list[str]) -> list[str]:
    """PLUMED log paths matching COLVAR paths."""
    return [c[:-len(".colvar.dat")] + ".plumed.log" for c in colvars]


def g(pattern: str) -> list[str]:
    """Sorted glob without PLUMED backups."""
    return sorted(f for f in glob.glob(pattern) if not os.path.basename(f).startswith("bck"))


def fit_list() -> list[dict]:
    """Every fit as {rel, sets: [(label, colvar_glob_or_list)], opts, needs}."""
    fits = []
    for cv in CVS:
        for pace in PACES:
            fits.append(dict(rel=f"triasha_results/prod_opes/bias{cv}/opes-flooding-results-bootstrap/pace{pace}_flooding.json",
                             sets=[(b, f"{SRC}/prod_opes/bias{cv}/barrier{b}/pace{pace}/s[0-9]*/run_*.colvar.dat") for b in (5, 7, 9, 11, 13)],
                             opts=["--vcol", "6", "--opesf"] + THIN))
    for cv in CVS:
        for n in (10, 25, 50, 100, 150, 200):
            fits.append(dict(rel=f"triasha_results/prod_ves_cutoff/bias{cv}/nsim-sweep/combined_N{n}.json",
                             sets=[(fc, f"{SRC}/prod_ves_cutoff/bias{cv}/nsim-sweep/subset_links/N{n}/fc{fc}/*.colvar.dat") for fc in (5, 7, 9, 11, 13, 15)],
                             opts=["--vcol", "5", "--nooffset", "--opesf"] + THIN))
    for cv in CVS:
        for bf in (5, 10, 15, 20):
            for h in ("0.1", "0.25", "0.5", "1.0"):
                fits.append(dict(rel=f"triasha_results/prod_imetad/bias{cv}/eatr-flooding-results-bootstrap/bf{bf}_height{h}_flooding.json",
                                 sets=[(p, f"{SRC}/prod_imetad/bias{cv}/bf{bf}/height{h}/pace{p}/s[0-9]*/run_*.colvar.dat") for p in PACES],
                                 opts=["--vcol", "5", "--nooffset"]))
    for cv in ("Q", "Ree"):
        fits.append(dict(rel=f"results/opes_highbarrier/{cv}_pace1000_barriers5-17.json",
                         sets=[(b, f"{SRC}/prod_opes/bias{cv}/barrier{b}/pace1000/s[0-9]*/run_*.colvar.dat") for b in (5, 7, 9, 11, 13)]
                         + [(b, f"@HB/bias{cv}/barrier{b}/pace1000/s[0-9]*/run_*.colvar.dat") for b in (15, 17)],
                         opts=["--vcol", "6", "--opesf"] + THIN, needs=f"hb_{cv}"))
    with tarfile.open(f"{P}/subsampling/flooding_nsim/flooding_nsim_raw.tar.gz") as tf:
        members = sorted(m.name for m in tf.getmembers() if m.name.endswith(".json.files"))
    for m in members:
        fam, cond, fname = m.split("/")[-3:]
        opts = (["--vcol", "6"] if fam == "opes" else ["--vcol", "5", "--nooffset"]) + ["--opesf"] + THIN
        fits.append(dict(rel=f"subsampling/flooding_nsim/{fam}/{cond}/{fname[:-len('.files')]}", sets=None, filelist=m, family=fam, opts=opts))
    return fits


def resolve_sets(fit: dict, tmp: str) -> list[tuple[int, list[str]]]:
    """Concrete (label, colvars) per set for one fit."""
    if fit.get("filelist"):
        with tarfile.open(f"{P}/subsampling/flooding_nsim/flooding_nsim_raw.tar.gz") as tf:
            lines = tf.extractfile(fit["filelist"]).read().decode().splitlines()
        paths = [l for l in lines if l and not l.startswith("#")]
        rx = re.compile(r"/barrier(\d+)/pace" if fit["family"] == "opes" else r"/fc(\d+)/")
        order, groups = [], {}
        for p in paths:
            lab = int(rx.search(p).group(1))
            if lab not in groups:
                order.append(lab); groups[lab] = []
            groups[lab].append(p)
        return [(lab, groups[lab]) for lab in order]
    return [(lab, g(pat.replace("@HB", os.path.join(tmp, "hb")))) for lab, pat in fit["sets"]]


def run_one(fit: dict, tmp: str) -> dict:
    """Run one no-bootstrap fit with the installed package; return a record."""
    if fit.get("needs", "").startswith("hb_") and not os.path.isdir(os.path.join(tmp, "hb", f"bias{fit['needs'][3:]}")):
        cv = fit["needs"][3:]
        for b in (15, 17):
            with tarfile.open(f"{P}/simulations/opes_highbarrier/bias{cv}_barrier{b}_pace1000_runs.tar.gz") as tf:
                tf.extractall(os.path.join(tmp, "hb"))
    sets = resolve_sets(fit, tmp)
    args = []
    for lab, cols in sets:
        if not cols:
            raise RuntimeError(f"no files for set {lab} of {fit['rel']}")
        args += ["-i", *cols, "--barrier", str(lab), "--logfiles", *logs_of(cols)]
    out = os.path.join(tmp, "out.json")
    script = os.path.join(tmp, "cmd.sh")
    cmd = ["eatr-flooding-analysis", *args, *BASE, *fit["opts"], "--threads", "4", "-o", out]
    with open(script, "w") as fh:
        fh.write("source /ext3/env.sh >/dev/null 2>&1\ncd " + tmp + "\n" + " ".join(cmd) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(["singularity", "exec", "--overlay", f"{OVL}:ro", "--bind", "/projects/rps/gmh4/hockygroup",
                        "--bind", f"{tmp}:{tmp}", SIF, "/bin/bash", script], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return dict(file=fit["rel"], error=(r.stdout + r.stderr)[-2000:])
    data = json.load(open(out))
    os.remove(out)
    return dict(file=fit["rel"], data=data, n_per_set=[len(c) for _, c in sets],
                opts=fit["opts"], package_git="14cee55 (installed)")


def main() -> None:
    """Entry point."""
    fits = fit_list()
    if sys.argv[1] == "list":
        print(len(fits)); return
    task, chunk, tmp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(tmp, exist_ok=True)
    todo = fits[task * chunk:(task + 1) * chunk]
    outdir = f"{P}/refits_full_data"; os.makedirs(outdir, exist_ok=True)
    path = f"{outdir}/task_{task:03d}.jsonl.gz"
    done = set()
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            done = {json.loads(l)["file"] for l in fh if '"data"' in l}
    recs = []
    for fit in todo:
        if fit["rel"] in done:
            continue
        rec = run_one(fit, tmp)
        print(("ok   " if "data" in rec else "FAIL ") + fit["rel"], flush=True)
        recs.append(rec)
    old = []
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            old = [json.loads(l) for l in fh]
    with gzip.open(path + ".tmp", "wt") as fh:
        for r in [o for o in old if "data" in o] + recs:
            fh.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print(f"task {task}: {sum('data' in r for r in recs)} ok, {sum('error' in r for r in recs)} failed", flush=True)


if __name__ == "__main__":
    main()
