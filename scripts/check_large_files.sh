#!/bin/bash
# Check the files this repository leaves to the full-data folder against the SHA-256 recorded
# in LARGE_FILES.tsv, and say where each absent one goes.
#
# The files sit at the top of the folder README.md links to under "Full simulation output",
# under their own names. A shared folder cannot be read by a script, so each is downloaded by
# hand and placed at the path this prints.
#
# Nothing in the figure and table chain needs them: the reference rate is derived from the
# first-passage times in runs/unbiased/reference_rate, which are here. They are the primary
# data behind that table.
#
# Usage: bash scripts/check_large_files.sh
set -u
cd "$(dirname "$0")/.." || exit 1
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fail=0
while IFS=$'\t' read -r path bytes want; do
    [ "$path" = "path" ] && continue
    if [ ! -f "$path" ]; then
        echo "absent   $path (download $(basename "$path") from the folder linked in README.md to this path)"; fail=1
    elif [ "$(sha "$path")" = "$want" ]; then
        echo "present  $path"
    else
        echo "FAILED   $path (checksum differs from LARGE_FILES.tsv)"; fail=1
    fi
done < LARGE_FILES.tsv
exit $fail
