"""Pick unbiased-run frames at a ladder of Q values and tile their rendered snapshots.

Usage (project root):
    uv run python scripts/q_ladder_snapshots.py select
    uv run python scripts/q_ladder_snapshots.py compose

select reads data/unbiased_runs_dumped_frame_q.tsv (run, frame, time_ps, q for every 1-ns
dumped frame of the 400 unbiased runs) and, for each target Q, takes the frame whose q is
nearest, breaking ties by the smallest trajectory file. It writes
data/unbiased_runs_q_ladder/selected_frames.csv and prints the tar paths to fetch from torch.

compose expects, for each selected row, a cropped render at
paper_figures/snapshots/q_ladder/go_q<q>_run<run>_frame<frame>.png (made with
scripts/render_go_snapshots.tcl and scripts/crop_png_whitespace.py) and writes the 2x4 strip
paper_figures/snapshots/go_q_ladder.png.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data/unbiased_runs_dumped_frame_q.tsv"
SELECTED = ROOT / "data/unbiased_runs_q_ladder/selected_frames.csv"
RENDERS = ROOT / "paper_figures/snapshots/q_ladder"
STRIP = ROOT / "paper_figures/snapshots/go_q_ladder.png"
TARGETS = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35]
DCD_HEADER, DCD_FRAME = 276, 752  # bytes for the 56-bead model: 3 x (4 + 224 + 4) + 56-byte cell block


def select() -> None:
    """Choose the dumped frame nearest each target Q and write selected_frames.csv."""
    d = pd.read_csv(TABLE, sep=r"\s+")
    d["nframes"] = (d.nframes_dcd_bytes - DCD_HEADER) // DCD_FRAME
    bad = d[d.frame >= d.nframes]
    assert bad.empty, f"frame index beyond the DCD:\n{bad.head()}"
    rows = []
    for t in TARGETS:
        c = d.assign(dq=(d.q - t).abs()).sort_values(["dq", "nframes_dcd_bytes"]).iloc[0]
        rows.append(dict(target=t, run=int(c.run), frame=int(c.frame), time_ps=c.time_ps,
                         q=round(c.q, 4), nframes=int(c.nframes), dcd_bytes=int(c.nframes_dcd_bytes)))
    sel = pd.DataFrame(rows)
    SELECTED.parent.mkdir(exist_ok=True)
    sel.to_csv(SELECTED, index=False)
    print(sel.to_string(index=False))
    print(" ".join(f"run_{r}/traj_312.dcd run_{r}/cvs.colvar" for r in sel.run.unique()))


def compose() -> None:
    """Tile the cropped renders of the selected frames into one labeled strip."""
    sel = pd.read_csv(SELECTED)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), layout="constrained")
    for ax, r in zip(axes.flat, sel.itertuples()):
        im = Image.open(RENDERS / f"go_q{r.q}_run{int(r.run)}_frame{int(r.frame)}.png")
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(f"$Q = {r.q:.2f}$", fontsize=14)
        ax.text(0.5, -0.03, f"run {int(r.run)}, {r.time_ps / 1000:.0f} ns", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="0.4")
    fig.savefig(STRIP, dpi=200)
    print(f"wrote {STRIP}")


if __name__ == "__main__":
    {"select": select, "compose": compose}[sys.argv[1]]()
