"""Per-trajectory summary of every biased run (cost, final bias, hills).

Usage (torch, anaconda3/2025.06):
    python trajectory_summary.py <family> <CV> <outdir>
        family: imetad | opes | ves

Walks every condition directory of the given family and CV under the read-only
tp2353 tree and writes <outdir>/<family>_<CV>_trajectories.csv with one row per
replica: condition labels, seed, committed flag (from the PLUMED log),
t_final (ps), bias at the last row, maximum bias, acceleration column at the
last row when present, number of deposited hills (HILLS rows, iMetaD only) and
the height of the last hill (well-tempered decay -> bias at the deposition
point). Only the header and the last 8 kB of each COLVAR are read.
Sanity check: for iMetaD the hill count must equal floor(t_final / (pace * dt))
to within one hill, which the script asserts on 1% of the runs.
"""
import csv
import glob
import os
import re
import sys

SRC = "/projects/rps/gmh4/hockygroup/tp2353/projects/project-VIII-EATR-extension"
DT_PS = 0.01  # 10 fs time step


def last_line(path: str, tail: int = 8192) -> str:
    """Return the last non-comment line of a text file reading only its tail.

    Args:
        path (str): file path.
        tail (int): bytes to read from the end.
    """
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - tail))
        chunk = fh.read().decode("utf-8", "replace")
    for line in reversed(chunk.splitlines()):
        if line.strip() and not line.startswith("#"):
            return line
    return ""


def header_fields(path: str) -> list[str]:
    """Return the COLVAR FIELDS names.

    Args:
        path (str): COLVAR path.
    """
    with open(path) as fh:
        return fh.readline().split()[2:]


def hills_info(path: str) -> tuple[int, float, float, float, int]:
    """Return (number of hills, last hill height, last hill time, minimum hill
    height, hill index of that minimum) from a HILLS file. The minimum height
    marks the largest bias at any deposition point, i.e. the fill level of the
    basin, whereas the last hill is deposited at the escape point.

    Args:
        path (str): HILLS file path.
    """
    n = 0
    last = ""
    hmin, imin = float("inf"), -1
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            n += 1
            last = line
            h = float(line.split()[3])
            if h < hmin:
                hmin, imin = h, n
    if not last:
        return 0, float("nan"), float("nan"), float("nan"), -1
    parts = last.split()
    return n, float(parts[3]), float(parts[0]), hmin, imin


def main(family: str, cv: str, outdir: str) -> None:
    """Summarize every replica of one family/CV.

    Args:
        family (str): imetad | opes | ves.
        cv (str): Q | RMSD | Ree | Rg.
        outdir (str): output directory.
    """
    os.makedirs(outdir, exist_ok=True)
    if family == "imetad":
        pat = f"{SRC}/prod_imetad/bias{cv}/bf*/height*/pace*/s[0-9]*/run_*.colvar.dat"
        rx = re.compile(r"bf([\d.]+)/height([\d.]+)/pace(\d+)/s(\d+)/")
    elif family == "opes":
        pat = f"{SRC}/prod_opes/bias{cv}/barrier*/pace*/s[0-9]*/run_*.colvar.dat"
        rx = re.compile(r"barrier(\d+)/pace(\d+)/s(\d+)/")
    else:
        pat = f"{SRC}/prod_ves_cutoff/bias{cv}/fc*/s[0-9]*/vesf_*_flood_s*.colvar.dat"
        rx = re.compile(r"fc(\d+)/s(\d+)/")
    files = sorted(f for f in glob.glob(pat) if "bck." not in os.path.basename(f))
    out = os.path.join(outdir, f"{family}_{cv}_trajectories.csv")
    checked = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "cv", "bf", "height", "pace", "barrier", "fc", "seed", "committed",
                    "t_final_ps", "bias_final", "acc_final", "nhills", "last_hill_height", "last_hill_time_ps",
                    "min_hill_height", "n_at_min_height"])
        for i, f in enumerate(files):
            m = rx.search(f)
            if not m:
                continue
            if family == "imetad":
                bf, height, pace, seed = m.groups(); barrier = fc = ""
            elif family == "opes":
                barrier, pace, seed = m.groups(); bf = height = fc = ""
            else:
                fc, seed = m.groups(); bf = height = pace = barrier = ""
            log = f[:-len(".colvar.dat")] + ".plumed.log"
            committed = False
            if os.path.exists(log):
                with open(log, "rb") as lf:
                    lf.seek(0, os.SEEK_END); lf.seek(max(0, lf.tell() - 4096))
                    committed = b"COMMITTED" in lf.read()
            fields = header_fields(f)
            last = last_line(f).split()
            try:
                t_final = float(last[0])
                bias = float(last[fields.index("metad.bias")]) if "metad.bias" in fields else \
                    float(last[fields.index("opes.bias")]) if "opes.bias" in fields else float(last[fields.index("ves.bias")])
                acc = float(last[fields.index("metad.acc")]) if "metad.acc" in fields else ""
            except (ValueError, IndexError):
                t_final, bias, acc = float("nan"), float("nan"), ""
            nh, lh, lt, hm, im = "", "", "", "", ""
            hills = os.path.join(os.path.dirname(f), "HILLS")
            if family == "imetad" and os.path.exists(hills):
                nh, lh, lt, hm, im = hills_info(hills)
                if i % 100 == 0 and t_final == t_final and nh:
                    expected = int(t_final / (int(pace) * DT_PS))
                    assert abs(nh - expected) <= 1, (f, nh, expected)
                    checked += 1
            w.writerow([family, cv, bf, height, pace, barrier, fc, seed, int(committed),
                        t_final, bias, acc, nh, lh, lt, hm, im])
    print(f"{family} {cv}: {len(files)} files -> {out}; hill-count checks passed: {checked}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
