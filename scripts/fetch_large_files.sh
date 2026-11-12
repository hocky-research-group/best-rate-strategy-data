#!/bin/bash
# Download the files this repository leaves to the data deposit, and check each against the
# size and SHA-256 recorded in LARGE_FILES.tsv.
#
# Nothing in the figure and table chain needs them: the reference rate is derived from the
# first-passage times in runs/unbiased/reference_rate, which are here. They are the primary
# data behind that table.
#
# Usage: ZENODO_RECORD=<record number> bash scripts/fetch_large_files.sh
#        The record number is the last part of the DOI given in the README.
set -u
cd "$(dirname "$0")/.." || exit 1
: "${ZENODO_RECORD:?set ZENODO_RECORD to the number of the Zenodo record (see README.md)}"
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fail=0
while IFS=$'\t' read -r path bytes want; do
    [ "$path" = "path" ] && continue
    if [ -f "$path" ] && [ "$(sha "$path")" = "$want" ]; then echo "present  $path"; continue; fi
    mkdir -p "$(dirname "$path")"
    curl -fL --retry 3 -o "$path.partial" \
        "https://zenodo.org/records/$ZENODO_RECORD/files/$(basename "$path")?download=1" \
        || { echo "FAILED   $path (download)"; rm -f "$path.partial"; fail=1; continue; }
    if [ "$(sha "$path.partial")" = "$want" ]; then
        mv "$path.partial" "$path"; echo "fetched  $path"
    else
        echo "FAILED   $path (checksum differs from LARGE_FILES.tsv)"; rm -f "$path.partial"; fail=1
    fi
done < LARGE_FILES.tsv
exit $fail
