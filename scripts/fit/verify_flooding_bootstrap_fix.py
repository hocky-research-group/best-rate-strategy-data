"""Verify the eatr-flooding-analysis bootstrap fix (EATR-rate-analysis 14cee55) on protein G data.

Usage (compute node, `module load anaconda3/2025.06`):
    python verify_flooding_bootstrap_fix.py <outdir> <tmpdir>

For every case three fits are run on identical inputs with identical options:
    old_b : ea9d483 (exported from the torch clone into <tmpdir>/old), --bootstrap
    new_b : the installed package (14cee55), --bootstrap
    new_nb: the installed package, no bootstrap
plus an independent NumPy reference for each set: the censored MLE ln(M / sum t) and the
ensemble-first exponential average ln<e^{beta V}> with the package's row thinning.
Cases: full-data OPES ladders (Q, Ree at pace 1000; 5 barriers x 50 runs), the 7-barrier
Q ladder (adds BARRIER 15, 17), the VES Rg ladder (6 cutoffs x 200 runs), and the Q and
Ree OPES ladders right-censored at a wall-time T equal to the (1-f) quantile of the
slowest set's escape times, f = 0.1, 0.25, 0.5 (runs escaping after T are truncated at T
and given an empty PLUMED log, so the package counts them as censored).
Checks: headline keys of old_b and new_b identical; set_reports and flooding_diagnostics
of new_b equal to new_nb; new_b per-set values equal to the reference; how far old_b's
per-set values were from the full data.
Outputs: <outdir>/summary.json, <outdir>/summary.txt, <outdir>/fits.tar.gz (all JSONs
and run logs), with code versions recorded.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np

SRC = "/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension"
P = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study"
SW = "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/eatr-software"
CLONE = f"{SW}/EATR-rate-analysis"
OVL = f"{SW}/torch-eatr-python-15GB-500K.ext3"
SIF = "/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
COMMON = ["--temp", "312", "--energyunit", "1.0", "--timeunit", "1e-12", "--tcol", "0",
          "--stride", "1000", "--subsample-min-points", "200", "--opesf", "--no-plots", "-q"]
BOOT = ["-b", "--numboots", "100", "--seed", "12345"]
HEADLINE = ["beta", "logk0", "gamma", "opes_logk0", "bootstrap_logk0_std", "bootstrap_logk0_ci",
            "bootstrap_gamma_std", "bootstrap_gamma_ci", "bootstrap_opes_logk0_std", "bootstrap_opes_logk0_ci",
            "bootstrap_per_set_log_k_obs_std", "bootstrap_per_set_ln_exp_beta_v_std", "bootstrap_iterations"]
REPORT_FIELDS = ["transitioned", "total", "avg_max_bias", "tau_obs", "k_obs", "log_k_obs", "ks_stat", "p_value", "ln_exp_beta_v"]


def committed(log: str) -> bool:
    """True if a PLUMED log records a committed basin (the package's event rule).

    Args:
        log (str): PLUMED log path.
    """
    with open(log, errors="replace") as fh:
        return any("SET COMMIT" in line for line in fh)


def last_time_ps(colvar: str) -> float:
    """Final time (ps) of a COLVAR file.

    Args:
        colvar (str): COLVAR path.
    """
    with open(colvar, "rb") as fh:
        fh.seek(0, os.SEEK_END); fh.seek(max(0, fh.tell() - 4096))
        lines = [l for l in fh.read().decode(errors="replace").splitlines() if l.strip() and not l.startswith("#")]
    return float(lines[-1].split()[0])


def opes_set(cv: str, barrier: int, pace: int = 1000, root: str | None = None) -> dict:
    """File lists for one OPES set (sorted, backups excluded).

    Args:
        cv (str): biased CV.
        barrier (int): OPES BARRIER in kJ/mol.
        pace (int): deposition pace in steps.
        root (str | None): directory holding bias<CV>/barrier<B>/pace<P>; default Triasha's tree.
    """
    base = f"{root or SRC + '/prod_opes'}/bias{cv}/barrier{barrier}/pace{pace}"
    colvars = sorted(f for f in glob.glob(f"{base}/s[0-9]*/run_*.colvar.dat") if not os.path.basename(f).startswith("bck"))
    return dict(label=barrier, colvars=colvars, logs=[c[:-len(".colvar.dat")] + ".plumed.log" for c in colvars])


def ves_set(cv: str, fc: int) -> dict:
    """File lists for one VES cutoff set (all replicas).

    Args:
        cv (str): biased CV.
        fc (int): bias cutoff in kJ/mol.
    """
    base = f"{SRC}/prod_ves_cutoff/bias{cv}/fc{fc}"
    colvars = sorted(glob.glob(f"{base}/s[0-9]*/vesf_{cv}_fc{fc}_flood_s*.colvar.dat"))
    return dict(label=fc, colvars=colvars, logs=[c[:-len(".colvar.dat")] + ".plumed.log" for c in colvars])


def censor_ladder(sets: list[dict], frac: float, tmp: str, tag: str) -> tuple[list[dict], float]:
    """Right-censor a ladder at one wall-time T for every set.

    T is the (1 - frac) quantile of the escape times of the set with the longest median
    escape time. Runs escaping before T are linked unchanged; later runs are truncated to
    rows with t <= T and get an empty log, which the package reads as censored.

    Args:
        sets (list[dict]): ladder as returned by opes_set.
        frac (float): target censored fraction of the slowest set.
        tmp (str): scratch directory.
        tag (str): subdirectory name.
    Returns the censored ladder and T in ps.
    """
    times = [np.array([last_time_ps(c) for c in s["colvars"]]) for s in sets]
    slowest = int(np.argmax([np.median(t) for t in times]))
    T = float(np.floor(np.quantile(times[slowest], 1 - frac)))
    out = []
    for s, t in zip(sets, times):
        d = os.path.join(tmp, tag, f"set{s['label']}")
        os.makedirs(d, exist_ok=True)
        cols, logs = [], []
        for i, (c, lg, tf) in enumerate(zip(s["colvars"], s["logs"], t)):
            nc, nl = os.path.join(d, f"run_{i:04d}.colvar.dat"), os.path.join(d, f"run_{i:04d}.plumed.log")
            if tf < T:
                os.symlink(c, nc); os.symlink(lg, nl)
            else:
                with open(c) as src, open(nc, "w") as dst:
                    for line in src:
                        if line.startswith("#") or float(line.split()[0]) <= T:
                            dst.write(line)
                        else:
                            break
                open(nl, "w").close()
            cols.append(nc); logs.append(nl)
        out.append(dict(label=s["label"], colvars=cols, logs=logs))
    return out, T


def reference(sets: list[dict], vcol: int, offset: bool, beta: float) -> list[dict]:
    """Independent per-set values: censored MLE and ln<e^{beta V}> with the package's thinning.

    Args:
        sets (list[dict]): ladder.
        vcol (int): bias column.
        offset (bool): add the set label (OPES BARRIER) to the bias.
        beta (float): inverse temperature in mol/kJ.
    """
    res = []
    for s in sets:
        trajs = [np.atleast_2d(np.loadtxt(c, comments="#", usecols=(0, vcol))) for c in s["colvars"]]
        trajs = [np.column_stack([t[:, 0] * 1e-12, t[:, 1]]) for t in trajs]
        event = np.array([committed(l) for l in s["logs"]])
        tf = np.array([t[-1, 0] for t in trajs])
        step = min(1000, max(1, min(len(t) for t in trajs) // 200))
        if step > 1:
            def thin(t):
                keep = list(range(0, len(t), step))
                if keep[-1] != len(t) - 1:
                    keep.append(len(t) - 1)
                return t[keep]
            trajs = [thin(t) for t in trajs]
        L = max(len(t) for t in trajs)
        v = np.full((len(trajs), L), np.nan)
        for i, t in enumerate(trajs):
            v[i, :len(t)] = t[:, 1] + (s["label"] if offset else 0.0)
        res.append(dict(label=s["label"], transitioned=int(event.sum()), total=len(trajs),
                        log_k_obs=math.log(event.sum() / tf.sum()),
                        ln_exp_beta_v=float(np.log(np.mean(np.nanmean(np.exp(beta * v), axis=0)))),
                        censored_frac=float(1 - event.mean()), effective_stride=step))
    return res


def run_fit(version: str, sets: list[dict], extra: list[str], out: str, tmp: str, name: str) -> float:
    """Run one eatr-flooding-analysis fit inside the container; return wall seconds.

    Args:
        version (str): "old" (ea9d483 export) or "new" (installed package).
        sets (list[dict]): ladder.
        extra (list[str]): options beyond COMMON (vcol, offset, bootstrap).
        out (str): output JSON path.
        tmp (str): scratch directory.
        name (str): tag for the command script and log.
    """
    args = []
    for s in sets:
        args += ["-i", *s["colvars"], "--barrier", str(s["label"]), "--logfiles", *s["logs"]]
    if version == "old":
        cwd, exe = os.path.join(tmp, "old"), "python -m eatr_rates.rates_eatr_opes"
    else:
        cwd, exe = os.path.join(tmp, "run"), "eatr-flooding-analysis"
    script = os.path.join(tmp, f"{name}.sh")
    with open(script, "w") as fh:
        fh.write("source /ext3/env.sh >/dev/null 2>&1\n")
        fh.write(f"cd {cwd}\n")
        fh.write("python -c \"import eatr_rates.rates_eatr_opes as m, rate_methods_library as r; print('CODE', m.__file__, r.__file__)\"\n")
        fh.write(exe + " " + " ".join(args + COMMON + extra + ["--threads", "8", "-o", out]) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import time
    t0 = time.time()
    with open(os.path.join(tmp, "logs", f"{name}.log"), "w") as log:
        rc = subprocess.run(["singularity", "exec", "--overlay", f"{OVL}:ro", "--bind", "/projects/rps/gmh4/hockygroup",
                             "--bind", f"{tmp}:{tmp}", SIF, "/bin/bash", script], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"{name} failed, see {tmp}/logs/{name}.log")
    return time.time() - t0


def compare(case: str, meta: dict, fits: dict, ref: list[dict]) -> dict:
    """Checks for one case.

    Args:
        case (str): case name.
        meta (dict): case description.
        fits (dict): parsed JSON per version.
        ref (list[dict]): reference per set.
    """
    ob, nb, nn = fits["old_b"], fits["new_b"], fits["new_nb"]
    head_same = {k: ob.get(k) == nb.get(k) for k in HEADLINE}
    rep_eq = max(abs(a[f] - b[f]) / max(1e-300, abs(b[f])) for a, b in zip(nb["set_reports"], nn["set_reports"]) for f in REPORT_FIELDS)
    d_nb, d_nn = nb["flooding_diagnostics"], nn["flooding_diagnostics"]
    diag_eq = (d_nb["gamma_best"] == d_nn["gamma_best"] and d_nb["logk0_best"] == d_nn["logk0_best"]
               and d_nb["var_ln_k0"] == d_nn["var_ln_k0"] and d_nb["convergence_analysis"] == d_nn["convergence_analysis"])
    new_vs_ref = dict(
        max_abs_log_k_obs=max(abs(r["log_k_obs"] - x["log_k_obs"]) for r, x in zip(ref, nb["set_reports"])),
        max_abs_ln_exp_beta_v=max(abs(r["ln_exp_beta_v"] - x["ln_exp_beta_v"]) for r, x in zip(ref, nb["set_reports"])),
        transitioned_match=all(r["transitioned"] == x["transitioned"] for r, x in zip(ref, nb["set_reports"])))
    old_vs_ref = dict(
        max_abs_log_k_obs=max(abs(r["log_k_obs"] - x["log_k_obs"]) for r, x in zip(ref, ob["set_reports"])),
        max_abs_ln_exp_beta_v=max(abs(r["ln_exp_beta_v"] - x["ln_exp_beta_v"]) for r, x in zip(ref, ob["set_reports"])),
        transitioned_mismatch_sets=sum(r["transitioned"] != x["transitioned"] for r, x in zip(ref, ob["set_reports"])),
        gamma_best_old=ob["flooding_diagnostics"]["gamma_best"], gamma_best_full=d_nn["gamma_best"],
        slope_old=float(np.polyfit([x["ln_exp_beta_v"] for x in ob["set_reports"]], [x["log_k_obs"] for x in ob["set_reports"]], 1)[0]),
        slope_full=float(np.polyfit([r["ln_exp_beta_v"] for r in ref], [r["log_k_obs"] for r in ref], 1)[0]))
    L6, REF = math.log(1e6), 0.4306353231386764
    return dict(case=case, **meta, headline_identical=all(head_same.values()),
                headline_keys_differing=[k for k, v in head_same.items() if not v],
                new_b_reports_equal_nobootstrap_maxrel=rep_eq, new_b_diagnostics_equal_nobootstrap=diag_eq,
                new_b_vs_reference=new_vs_ref, old_b_vs_reference=old_vs_ref,
                err_eatrf_bootmean=nb["logk0"] - L6 - REF, err_eatrf_point=nn["logk0"] - L6 - REF,
                err_opesf_bootmean=nb["opes_logk0"] - L6 - REF,
                ci_eatrf=[c - L6 - REF for c in nb["bootstrap_logk0_ci"]], gamma=nb["gamma"],
                censored_frac_per_set=[r["censored_frac"] for r in ref])


def main(outdir: str, tmp: str) -> None:
    """Build every case, run the fits, compare, and package the outputs."""
    os.makedirs(outdir, exist_ok=True)
    for d in ("old", "run", "logs", "json"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    subprocess.run(f"git -C {CLONE} archive ea9d483 eatr_rates rate_methods_library.py ks_censored.py | tar -x -C {tmp}/old", shell=True, check=True)
    for B in (15, 17):
        with tarfile.open(f"{P}/simulations/opes_highbarrier/biasQ_barrier{B}_pace1000_runs.tar.gz") as tf:
            tf.extractall(os.path.join(tmp, "hb"))
    versions = dict(old="ea9d483 (git archive of the torch clone)",
                    new_clone_head=subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    beta = 1.0 / (8.314462618e-3 * 312.0)
    cases = {}
    for cv in ("Q", "Ree"):
        ladder = [opes_set(cv, b) for b in (5, 7, 9, 11, 13)]
        cases[f"opes_{cv}_pace1000_full"] = (ladder, 6, True, dict(family="opes", cv=cv, frac=0.0, T_ps=None))
        for f in (0.1, 0.25, 0.5):
            cl, T = censor_ladder(ladder, f, tmp, f"cens_{cv}_{f}")
            cases[f"opes_{cv}_pace1000_censored{f}"] = (cl, 6, True, dict(family="opes", cv=cv, frac=f, T_ps=T))
    cases["opes7_Q_pace1000_full"] = ([opes_set("Q", b) for b in (5, 7, 9, 11, 13)] + [opes_set("Q", b, root=os.path.join(tmp, "hb")) for b in (15, 17)],
                                      6, True, dict(family="opes7", cv="Q", frac=0.0, T_ps=None))
    cases["ves_Rg_N200_full"] = ([ves_set("Rg", fc) for fc in (5, 7, 9, 11, 13, 15)], 5, False, dict(family="ves", cv="Rg", frac=0.0, T_ps=None))
    summary = []
    for name, (sets, vcol, offset, meta) in cases.items():
        extra = ["--vcol", str(vcol)] + ([] if offset else ["--nooffset"])
        fits, secs = {}, {}
        for tag, version, boot in (("old_b", "old", True), ("new_b", "new", True), ("new_nb", "new", False)):
            out = os.path.join(tmp, "json", f"{name}__{tag}.json")
            secs[tag] = run_fit(version, sets, extra + (BOOT if boot else []), out, tmp, f"{name}__{tag}")
            fits[tag] = json.load(open(out))
        ref = reference(sets, vcol, offset, beta)
        row = compare(name, dict(meta, n_sets=len(sets), runs_per_set=[len(s["colvars"]) for s in sets], wall_s=secs), fits, ref)
        row["reference"] = ref
        summary.append(row)
        print(json.dumps({k: row[k] for k in ("case", "headline_identical", "new_b_reports_equal_nobootstrap_maxrel",
                                              "new_b_diagnostics_equal_nobootstrap", "new_b_vs_reference", "old_b_vs_reference")}), flush=True)
    code = {}
    for tag in ("old_b", "new_b"):
        with open(os.path.join(tmp, "logs", f"{list(cases)[0]}__{tag}.log")) as fh:
            code[tag] = [l.strip() for l in fh if l.startswith("CODE")]
    json.dump(dict(versions=versions, code_paths=code, options=dict(common=COMMON, bootstrap=BOOT), cases=summary),
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(f"code: {code}\n")
        fh.write("case | headline identical old_b/new_b | new_b==no-bootstrap (reports maxrel, diagnostics) | new_b vs ref (dlnk, dlnacc, events) | old_b vs ref (dlnk, dlnacc, event-mismatch sets, gamma_best old/full, slope old/full) | EATRf err boot/point | OPESf err | censored frac\n")
        for r in summary:
            nv, ov = r["new_b_vs_reference"], r["old_b_vs_reference"]
            fh.write(f"{r['case']} | {r['headline_identical']} {r['headline_keys_differing']} | {r['new_b_reports_equal_nobootstrap_maxrel']:.1e}, {r['new_b_diagnostics_equal_nobootstrap']} | "
                     f"{nv['max_abs_log_k_obs']:.1e}, {nv['max_abs_ln_exp_beta_v']:.1e}, {nv['transitioned_match']} | "
                     f"{ov['max_abs_log_k_obs']:.3f}, {ov['max_abs_ln_exp_beta_v']:.3f}, {ov['transitioned_mismatch_sets']}, {ov['gamma_best_old']:.3f}/{ov['gamma_best_full']:.3f}, {ov['slope_old']:.3f}/{ov['slope_full']:.3f} | "
                     f"{r['err_eatrf_bootmean']:+.3f}/{r['err_eatrf_point']:+.3f} | {r['err_opesf_bootmean']:+.3f} | {[round(x, 2) for x in r['censored_frac_per_set']]}\n")
    with tarfile.open(os.path.join(outdir, "fits.tar.gz"), "w:gz") as tf:
        tf.add(os.path.join(tmp, "json"), arcname="json")
        tf.add(os.path.join(tmp, "logs"), arcname="logs")
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
