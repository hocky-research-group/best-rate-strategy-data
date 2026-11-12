"""Conditions file for WT-MetaD subset sizes between those of the first subsampling pass.

Usage:
    uv run python scripts/intermediate_subset_conditions.py

Inputs
    data/subsampling/imetad_grid/conditions.txt         one line per condition: CV BF HEIGHT PACE
    data/subsampling/imetad_grid/other_missing_idx.txt  0-based lines that needed more memory
    analysis/subsampling_imetad.csv                     n_total, the runs each condition has
Output
    data/subsampling/imetad_grid_intermediate_sizes/conditions_intermediate_sizes.txt
        CV BF HEIGHT PACE SIZES, the format eatr_grid_subsample.sbatch reads through LIST.
        SIZES holds the members of NEW_SIZES below the condition's run count. Bias factor 1 is
        left out (degenerate, excluded from every summary). The conditions that need more
        memory (bias factor 2 and the listed lines) come last, so each group is one array
        range; the two ranges are printed.
"""
import pandas as pd

from paths import ROOT

NEW_SIZES = (7, 15, 30, 60, 120, 160)


def main() -> None:
    """Write the conditions file and print the array ranges of the two memory groups."""
    grid = ROOT / "data/subsampling/imetad_grid"
    lines = (grid / "conditions.txt").read_text().split("\n")[:-1]
    heavy_idx = {int(i) for i in (grid / "other_missing_idx.txt").read_text().split(",")}
    im = pd.read_csv(ROOT / "analysis/subsampling_imetad.csv")
    n_total = im.drop_duplicates(["cv", "bf", "height", "pace"]).set_index(["cv", "bf", "height", "pace"]).n_total
    light, heavy = [], []
    for i, line in enumerate(lines):
        cv, bf, height, pace = line.split()
        key = (cv, float(bf), float(height), int(pace))
        if float(bf) < 2 or key not in n_total.index:
            continue
        sizes = ",".join(str(n) for n in NEW_SIZES if n < n_total[key])
        (heavy if float(bf) == 2 or i in heavy_idx else light).append(f"{line} {sizes}")
    out = ROOT / "data/subsampling/imetad_grid_intermediate_sizes"
    out.mkdir(exist_ok=True)
    (out / "conditions_intermediate_sizes.txt").write_text("\n".join(light + heavy) + "\n")
    print(f"{len(light)} conditions at the default memory: array 0-{len(light) - 1}")
    print(f"{len(heavy)} conditions needing more: array {len(light)}-{len(light) + len(heavy) - 1}")


if __name__ == "__main__":
    main()
