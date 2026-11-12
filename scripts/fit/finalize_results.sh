#!/bin/bash
# Pack the per-condition result files on torch into one jsonl.gz + one tar per
# family (file-number quota), leaving only packed archives and the 12 trajectory
# summary CSVs. Usage (torch login node): bash finalize_results.sh
# Safe to rerun: skips a family whose loose files are already gone.
set -u
P=/projects/rps/gmh4/hockygroup/gmh4/projects/rates/best-rate-strategy-study
for spec in "subsampling/pilot pilot" "subsampling/imetad_grid/results imetad_grid" "subsampling/flooding_nsim flooding_nsim"; do
  set -- $spec
  loose=$(find $P/$1 -type f ! -name "*.jsonl.gz" ! -name "*_raw.tar.gz" | wc -l)
  if [ "$loose" -eq 0 ]; then echo "$1: already packed"; continue; fi
  python3 $P/scripts/pack_results.py $P/$1 $2 --delete
done
echo "file count under $P: $(find $P -type f | wc -l)"
