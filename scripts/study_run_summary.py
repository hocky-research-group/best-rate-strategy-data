"""Per-run summary of every campaign run for this paper, one CSV per campaign.

Usage (torch, anaconda3/2025.06):
    python study_run_summary.py <outdir> [campaign ...]

The companion to trajectory_summary.py, which covers the grid and the flooding ladders in
the read-only source tree. This covers the campaigns produced for this paper, whose run
directories are laid out differently in each case.

One row per run: the condition it belongs to, its velocity seed, whether it committed, the
time it ended, the bias at the end and the largest bias it reached, the acceleration factor
at the end, and the number of hills deposited. That is everything the analysis reads out of
a trajectory, so the archive can carry these tables instead of the PLUMED logs and COLVAR
files themselves, which are two orders of magnitude larger.

Only the header and the last 8 kB of each COLVAR are read for the final row; the maximum
bias needs the whole file and is taken in one pass without loading it. The committed flag
comes from the PLUMED log, where the record sits within the last twenty lines, so only the
tail is read there too.

Run directories whose name begins with bck. are PLUMED's backups of a restarted run and
hold a truncated trajectory; they are skipped, as they are everywhere else in this work.

The prospective campaign is absent because its runs were tarred per condition when it
finished, so there are no loose directories to walk; its two conditions are covered by the
fits and the outcome table already in the archive.
"""
import csv
import os
import re
import sys
from pathlib import Path

STUDY = Path(os.environ.get(
    "STUDY", "/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study"))
DT_PS = 0.01

# campaign -> (path under simulations/, regex over the path from the campaign root to the
# seed directory, capturing the condition labels that campaign varies)
CAMPAIGNS = {
    "wtmetad_untempered": ("untempered_metad",
                           r"bias(?P<cv>\w+)/height(?P<height>[\d.]+)/pace(?P<pace>\d+)/s(?P<seed>\d+)"),
    "wtmetad_frequency_adaptive": ("fametad",
                                   r"bias(?P<cv>\w+)/theta(?P<theta>[\d.]+)/s(?P<seed>\d+)"),
    "opes_temperature": ("opes_temperature",
                         r"T(?P<temperature_K>\d+)/barrier(?P<barrier>[\d.]+)/s(?P<seed>\d+)"),
    "wtmetad_temperature": ("wtmetad_temperature", r"T(?P<temperature_K>\d+)/s(?P<seed>\d+)"),
    "wtmetad_temperature_slowpace": ("wtmetad_temperature_slowpace",
                                     r"T(?P<temperature_K>\d+)/s(?P<seed>\d+)"),
    "wtmetad_temperature_Ree": ("wtmetad_temperature_Ree", r"T(?P<temperature_K>\d+)/s(?P<seed>\d+)"),
    "metad_pmf": ("pmf_temperature", r"T(?P<temperature_K>\d+)/s(?P<seed>\d+)"),
}
FIELDS = ["campaign", "cv", "temperature_K", "height", "pace", "theta", "barrier",
          "condition", "seed", "committed", "t_final_ps", "bias_final", "bias_max",
          "acc_final", "nhills"]


def tail_text(path: Path, nbytes: int = 8192) -> str:
    """Decode the last nbytes of a file.

    Args:
        path (Path): file to read.
        nbytes (int): how many bytes from the end.
    """
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - nbytes))
        return fh.read().decode("utf-8", "replace")


def colvar_summary(path: Path) -> dict:
    """Final row, maximum bias and column layout of one COLVAR.

    Args:
        path (Path): COLVAR file written by PLUMED's PRINT.
    """
    with open(path) as fh:
        header = fh.readline().split()[2:]
    idx = {name: i for i, name in enumerate(header)}
    bcol = idx.get("metad.bias", idx.get("opes.bias"))
    acol = idx.get("metad.acc")
    bias_max, last = float("nan"), ""
    if bcol is not None:
        bias_max = -float("inf")
        with open(path) as fh:                      # one pass, nothing held in memory
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) > bcol:
                    v = float(parts[bcol])
                    if v > bias_max:
                        bias_max = v
                last = line
    if not last:
        for line in reversed(tail_text(path).splitlines()):
            if line.strip() and not line.startswith("#"):
                last = line
                break
    f = last.split()
    return dict(t_final_ps=float(f[0]) if f else float("nan"),
                bias_final=float(f[bcol]) if bcol is not None and len(f) > bcol else float("nan"),
                bias_max=bias_max if bias_max != -float("inf") else float("nan"),
                acc_final=float(f[acol]) if acol is not None and len(f) > acol else float("nan"))


def summarise(campaign: str, outdir: Path) -> int:
    """Write one CSV for a campaign and return the number of runs it covered.

    Args:
        campaign (str): key of CAMPAIGNS.
        outdir (Path): directory the CSV is written to.
    """
    sub, pattern = CAMPAIGNS[campaign]
    root = STUDY / "simulations" / sub
    if not root.is_dir():
        print(f"  {campaign}: {root} missing, skipped")
        return 0
    rx = re.compile(pattern + "$")
    rows = []
    for seed_dir in sorted(root.rglob("s[0-9]*")):
        if not seed_dir.is_dir():
            continue
        m = rx.search(str(seed_dir.relative_to(root)))
        if not m:
            continue
        # the prefix differs by campaign (run_ for the escape runs, pmfscan_ for the free
        # energy scans), so match on the suffix and drop PLUMED's restart backups
        colvar = sorted(f for f in seed_dir.glob("*.colvar.dat") if not f.name.startswith("bck."))
        log = sorted(f for f in seed_dir.glob("*.plumed.log") if not f.name.startswith("bck."))
        if not colvar:
            continue
        row = dict.fromkeys(FIELDS, "")
        row.update(campaign=campaign, **m.groupdict())
        row.update(colvar_summary(colvar[0]))
        row["committed"] = int(bool(log) and "COMMITTED" in tail_text(log[0], 4096))
        hills = sorted(seed_dir.glob("*.hills")) or sorted(seed_dir.glob("HILLS"))
        row["nhills"] = sum(1 for line in open(hills[0]) if not line.startswith("#")) if hills else ""
        rows.append(row)
    out = outdir / f"{campaign}_runs.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {campaign}: {len(rows)} runs, {sum(r['committed'] for r in rows)} committed -> {out.name}")
    return len(rows)


def main() -> None:
    """Summarise the campaigns named on the command line, or all of them."""
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or list(CAMPAIGNS)
    total = sum(summarise(c, outdir) for c in names)
    print(f"{total} runs summarised into {outdir}")


if __name__ == "__main__":
    main()
