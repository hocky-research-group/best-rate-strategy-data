"""Average applied bias along the biased CV for selected flooding/MetaD sets.

Usage (torch, anaconda3/2025.06):
    python bias_profiles.py <outdir>

For each selected set the COLVAR rows of every committed replica are binned
by the biased CV and the bias column is averaged per bin:
    VES   : all rows (static bias), ves.bias (kJ/mol)
    OPES  : rows in the last 50% of each run (bias converged), opes.bias + BARRIER
    MetaD : rows in the last 20% of each run (bias near escape), metad.bias
Outputs <outdir>/profile_<family>_<cv>_<set>.dat with columns
    cv_center  mean_bias  n_rows
Also copies the static VES bias on its grid (bias.ves.data of replica s1) to
<outdir>/vesbias_<cv>_fc<fc>.dat and a downsampled stage-1 trace of ves.bias
versus time to <outdir>/ves_stage1_<cv>_fc<fc>.dat (every 100th row).
Sanity check: for VES the trajectory-binned average must agree with the static
bias file where the bins are populated (printed max deviation, kJ/mol).
"""
import glob
import os
import sys

import numpy as np

SRC = "/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension"
CVCOL = {"Q": 1, "RMSD": 2, "Rg": 3, "Ree": 4}
RANGE = {"Q": (0.0, 1.0), "RMSD": (0.0, 3.6), "Rg": (0.9, 3.8), "Ree": (0.0, 10.5)}
NB = 100
SETS = (
    [("ves", cv, f"fc{fc}", f"{SRC}/prod_ves_cutoff/bias{cv}/fc{fc}/s[0-9]*/vesf_{cv}_fc{fc}_flood_s*.colvar.dat", 5, 0.0, 1.0)
     for cv in ("Q", "RMSD", "Rg", "Ree") for fc in (5, 7, 9, 11, 13, 15)]
    + [("opes", cv, f"barrier{b}_pace1000", f"{SRC}/prod_opes/bias{cv}/barrier{b}/pace1000/s[0-9]*/run_*.colvar.dat", 6, float(b), 0.5)
       for cv in ("Q", "Rg", "Ree") for b in (5, 7, 9, 11, 13)]
    + [("imetad", cv, f"bf10_height0.5_pace{p}", f"{SRC}/prod_imetad/bias{cv}/bf10/height0.5/pace{p}/s[0-9]*/run_*.colvar.dat", 5, 0.0, 0.2)
       for cv in ("Q", "Ree") for p in (100, 1000, 100000)]
)


def committed(colvar: str) -> bool:
    """True if the PLUMED log next to the COLVAR records a committed escape.

    Args:
        colvar (str): COLVAR path.
    """
    log = colvar[:-len(".colvar.dat")] + ".plumed.log"
    if not os.path.exists(log):
        return False
    with open(log, "rb") as fh:
        fh.seek(0, os.SEEK_END); fh.seek(max(0, fh.tell() - 4096))
        return b"COMMITTED" in fh.read()


def profile(files: list[str], cvcol: int, vcol: int, offset: float, tail_frac: float, rng: tuple) -> np.ndarray:
    """Binned mean bias along the CV over the last tail_frac of every run.

    Args:
        files (list[str]): COLVAR files.
        cvcol (int): CV column index.
        vcol (int): bias column index.
        offset (float): added to the bias column (OPES stores bias - BARRIER).
        tail_frac (float): fraction of rows at the end of each run to use.
        rng (tuple): (min, max) of the CV binning range.
    """
    edges = np.linspace(rng[0], rng[1], NB + 1)
    ssum = np.zeros(NB); cnt = np.zeros(NB)
    for f in files:
        a = np.loadtxt(f, comments="#", usecols=(cvcol, vcol))
        if a.ndim < 2 or len(a) == 0:
            continue
        a = a[int(len(a) * (1 - tail_frac)):]
        idx = np.digitize(a[:, 0], edges) - 1
        ok = (idx >= 0) & (idx < NB)
        np.add.at(ssum, idx[ok], a[ok, 1] + offset); np.add.at(cnt, idx[ok], 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(invalid="ignore"):
        return np.column_stack([centers, ssum / cnt, cnt])


def main(outdir: str) -> None:
    """Write all profiles and the VES static bias and stage-1 traces.

    Args:
        outdir (str): output directory.
    """
    os.makedirs(outdir, exist_ok=True)
    for fam, cv, name, pat, vcol, offset, tail in SETS:
        files = [f for f in sorted(glob.glob(pat)) if "bck." not in os.path.basename(f) and committed(f)]
        prof = profile(files, CVCOL[cv], vcol, offset, tail, RANGE[cv])
        out = os.path.join(outdir, f"profile_{fam}_{cv}_{name}.dat")
        np.savetxt(out, prof, header=f"{fam} {cv} {name}: {len(files)} committed runs; cv_center mean_bias_kJmol n_rows")
        msg = f"{fam} {cv} {name}: {len(files)} runs"
        if fam == "ves":
            fc = name[2:]
            d = f"{SRC}/prod_ves_cutoff/bias{cv}/{name}"
            static = np.loadtxt(f"{d}/s1/bias.ves.data", comments="#")
            np.savetxt(os.path.join(outdir, f"vesbias_{cv}_{name}.dat"), static[:, :2], header="cv  ves.bias_kJmol (static, replica s1)")
            interp = np.interp(prof[:, 0], static[:, 0], static[:, 1])
            pop = prof[:, 2] > 50
            dev = np.nanmax(np.abs(prof[pop, 1] - interp[pop])) if pop.any() else float("nan")
            msg += f"; max |binned - static| = {dev:.3f} kJ/mol; static max {static[:, 1].max():.2f}"
            s1 = glob.glob(f"{d}/stage_I_opt/vesf_{cv}_{name}_stage1.colvar.dat")
            if s1:
                tr = np.loadtxt(s1[0], comments="#", usecols=(0, 5))[::100]
                np.savetxt(os.path.join(outdir, f"ves_stage1_{cv}_{name}.dat"), tr, header="time_ps  ves.bias_kJmol (stage 1, every 100th row)")
        print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
