"""Seeded subset refits of the flooding ladders, without bootstrap.

Usage (compute node or login node for one fit, anaconda3/2025.06):
    python flooding_subsets.py list                              # tasks and fits
    python flooding_subsets.py run <task> <tmpdir> [limit] [outdir]

Purpose: the minimum-data criterion (90% of subsets within one ln unit) with enough
independent draws to put intervals on it, and the cost and gamma spread of a two-level
flooding pilot. Each fit is a no-bootstrap eatr-flooding-analysis run on N committed runs
per set, drawn without replacement, so its set reports and diagnostics describe the drawn
data. Options are those of the earlier sweep (flooding_nsim_sweep.sbatch) without the
bootstrap. A task stops before fitting unless the installed rates_eatr_opes.py is the
fixed version (EXPECTED_BLOB, EATR-rate-analysis 14cee55 = main dfa7df1).

Tasks, one per (family, CV, pace or level pair, N):
    opes   BARRIER 5, 7, 9, 11, 13 kJ/mol; 4 CVs x 8 paces; N = 5, 10, 20 (25 draws), 50 (1)
    ves    F_c 5, 7, ..., 15 kJ/mol; 4 CVs; N = 5, 10, 20, 50, 100 (50 draws), 200 (1)
    pilot  OPES pace 1000, levels (5, 9) and (9, 13) kJ/mol; 4 CVs; N = 10 (50 draws)
    opes   as above with N = 7, 15, 30, 40 (25 draws)
    ves    as above with N = 7, 15, 30, 75, 150 (50 draws)
A task is addressed by its position in this order, so new tasks are appended at the end.
A set with fewer committed runs than N raises. Draw d of a task uses
numpy default_rng([SEED, family code, CV index, pace, N, first level, d]).
Output: <study>/subsampling/flooding_subsets/task_<k>.jsonl.gz (opes, ves) or
<study>/subsampling/flooding_pilot/task_<k>.jsonl.gz (pilot), one gzip member per fit:
{"file": "<family>/<condition>/N<n>_rep<r>.json", "data": <fit JSON>, "seed": [...],
 "files": {"<level>": [...]}, "package_blob": <sha1>}; a failed fit writes "error" instead
of "data". A rerun of a task skips the fits already recorded with data.
"""
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

SRC = "/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension"
P = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study"
SW = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software"
OVL = f"{SW}/torch-eatr-python-15GB-500K.ext3"
SIF = "/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
EXPECTED_BLOB = "ac3d11ce87912464381754a1f4072907e8fd1dc5"
BASE = ["--temp", "312", "--energyunit", "1.0", "--timeunit", "1e-12", "--tcol", "0", "--no-plots", "-q"]
THIN = ["--stride", "1000", "--subsample-min-points", "200"]
CVS = ("Q", "RMSD", "Ree", "Rg")
PACES = (100, 500, 1000, 5000, 10000, 30000, 50000, 100000)
SEED = 20260912
FAMILY_CODE = {"opes": 1, "ves": 2, "pilot": 3}


def task_list() -> list[dict]:
    """Every task as a dict with family, cv, pace, levels, n, draws and condition name."""
    tasks = []
    for cv in CVS:
        for pace in PACES:
            for n, draws in ((5, 25), (10, 25), (20, 25), (50, 1)):
                tasks.append(dict(family="opes", cv=cv, pace=pace, levels=(5, 7, 9, 11, 13), n=n, draws=draws, cond=f"{cv}_pace{pace}"))
    for cv in CVS:
        for n, draws in ((5, 50), (10, 50), (20, 50), (50, 50), (100, 50), (200, 1)):
            tasks.append(dict(family="ves", cv=cv, pace=0, levels=(5, 7, 9, 11, 13, 15), n=n, draws=draws, cond=cv))
    for cv in CVS:
        for pair in ((5, 9), (9, 13)):
            tasks.append(dict(family="pilot", cv=cv, pace=1000, levels=pair, n=10, draws=50, cond=f"{cv}_pace1000_b{pair[0]}-{pair[1]}"))
    # sizes between those above; appended so that the earlier task numbers keep their meaning
    for cv in CVS:
        for pace in PACES:
            for n in (7, 15, 30, 40):
                tasks.append(dict(family="opes", cv=cv, pace=pace, levels=(5, 7, 9, 11, 13), n=n, draws=25, cond=f"{cv}_pace{pace}"))
    for cv in CVS:
        for n in (7, 15, 30, 75, 150):
            tasks.append(dict(family="ves", cv=cv, pace=0, levels=(5, 7, 9, 11, 13, 15), n=n, draws=50, cond=cv))
    return tasks


def committed_pool(task: dict, level: int) -> list[str]:
    """Sorted COLVAR paths of the committed runs of one set.

    Args:
        task (dict): task from task_list.
        level (int): BARRIER (OPES) or bias cutoff F_c (VES) in kJ/mol.
    """
    cv = task["cv"]
    if task["family"] == "ves":
        pattern = f"{SRC}/prod_ves_cutoff/bias{cv}/fc{level}/s[0-9]*/vesf_{cv}_fc{level}_flood_s*.colvar.dat"
    else:
        pattern = f"{SRC}/prod_opes/bias{cv}/barrier{level}/pace{task['pace']}/s[0-9]*/run_*.colvar.dat"
    pool = []
    for c in sorted(glob.glob(pattern)):
        log = c[:-len(".colvar.dat")] + ".plumed.log"
        if os.path.basename(c).startswith("bck") or not os.path.exists(log):
            continue
        with open(log, errors="replace") as fh:
            if "COMMITTED" in fh.read():
                pool.append(c)
    return pool


def container(script_lines: list[str], tmp: str) -> subprocess.CompletedProcess:
    """Run bash lines inside the EATR container with the read-only overlay.

    Args:
        script_lines (list[str]): commands, run after sourcing the overlay environment.
        tmp (str): scratch directory (bound into the container and used as cwd).
    """
    script = os.path.join(tmp, "cmd.sh")
    with open(script, "w") as fh:
        fh.write("source /ext3/env.sh >/dev/null 2>&1\ncd " + tmp + "\n" + "\n".join(script_lines) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(["singularity", "exec", "--overlay", f"{OVL}:ro", "--bind", "/projects/rps/gmh4/hockygroup",
                           "--bind", f"{tmp}:{tmp}", SIF, "/bin/bash", script], capture_output=True, text=True, env=env)


def installed_blob(tmp: str) -> str:
    """Git blob sha1 of the rates_eatr_opes.py module the container imports.

    Args:
        tmp (str): scratch directory.
    """
    code = ("import hashlib, eatr_rates.rates_eatr_opes as m; s = open(m.__file__, 'rb').read(); "
            "print(hashlib.sha1(b'blob %d' % len(s) + bytes([0]) + s).hexdigest())")
    r = container([f'python -c "{code}"'], tmp)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else "unknown: " + r.stderr[-500:]


def fit_one(task: dict, draw: int, pools: dict, tmp: str) -> dict:
    """Draw N runs per set, fit without bootstrap, and return the record.

    Args:
        task (dict): task from task_list.
        draw (int): draw index (1-based).
        pools (dict): level -> committed COLVAR paths.
        tmp (str): scratch directory.
    """
    seed = [SEED, FAMILY_CODE[task["family"]], CVS.index(task["cv"]), task["pace"], task["n"], task["levels"][0], draw]
    rng = np.random.default_rng(seed)
    files, args = {}, []
    for level in task["levels"]:
        pool = pools[level]
        if len(pool) < task["n"]:
            raise RuntimeError(f"set {level} of {task['cond']} has {len(pool)} committed runs < N = {task['n']}")
        chosen = [pool[i] for i in sorted(rng.choice(len(pool), size=task["n"], replace=False))]
        files[str(level)] = chosen
        args += ["-i", *chosen, "--barrier", str(level), "--logfiles", *[c[:-len(".colvar.dat")] + ".plumed.log" for c in chosen]]
    opts = ["--vcol", "5", "--nooffset", "--opesf"] if task["family"] == "ves" else ["--vcol", "6", "--opesf"]
    out = os.path.join(tmp, "out.json")
    r = container([" ".join(["eatr-flooding-analysis", *args, *BASE, *opts, *THIN, "--threads", "4", "-o", out])], tmp)
    rel = f"{task['family']}/{task['cond']}/N{task['n']}_rep{draw}.json"
    rec = dict(file=rel, seed=seed, files=files, opts=opts + THIN)
    if r.returncode != 0 or not os.path.exists(out):
        rec["error"] = (r.stdout + r.stderr)[-2000:]
        return rec
    with open(out) as fh:
        rec["data"] = json.load(fh)
    os.remove(out)
    return rec


def main() -> None:
    """Entry point."""
    tasks = task_list()
    if sys.argv[1] == "list":
        print(len(tasks)); print(sum(t["draws"] for t in tasks), "fits")
        return
    k, tmp = int(sys.argv[2]), sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    task = tasks[k]
    outdir = sys.argv[5] if len(sys.argv) > 5 else f"{P}/subsampling/{'flooding_pilot' if task['family'] == 'pilot' else 'flooding_subsets'}"
    os.makedirs(tmp, exist_ok=True); os.makedirs(outdir, exist_ok=True)
    blob = installed_blob(tmp)
    print(f"task {k}: {task['family']} {task['cond']} N={task['n']} draws={task['draws']} installed blob {blob}", flush=True)
    if blob != EXPECTED_BLOB:
        raise SystemExit(f"installed rates_eatr_opes.py is {blob}, not the fixed version {EXPECTED_BLOB}")
    path = f"{outdir}/task_{k:03d}.jsonl.gz"
    done = set()
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            done = {rec["file"] for rec in map(json.loads, fh) if "data" in rec}
    pools = {level: committed_pool(task, level) for level in task["levels"]}
    print("committed runs per set:", {lv: len(p) for lv, p in pools.items()}, flush=True)
    n_ok = n_fail = 0
    for draw in range(1, task["draws"] + 1):
        if limit is not None and n_ok + n_fail >= limit:
            break
        rel = f"{task['family']}/{task['cond']}/N{task['n']}_rep{draw}.json"
        if rel in done:
            continue
        rec = fit_one(task, draw, pools, tmp)
        rec["package_blob"] = blob
        with gzip.open(path, "at") as fh:
            fh.write(json.dumps(rec) + "\n")
        ok = "data" in rec
        n_ok += ok; n_fail += not ok
        print(("ok   " if ok else "FAIL ") + rel, flush=True)
    print(f"task {k}: {n_ok} ok, {n_fail} failed, {len(done)} already recorded", flush=True)


if __name__ == "__main__":
    main()
