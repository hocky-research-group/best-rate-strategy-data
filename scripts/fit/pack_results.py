"""Pack a directory tree of small result files into two files (file-count quota).

Usage (torch login node, stdlib only):
    python3 pack_results.py <dir> <name> [--delete]

Creates in <dir>:
    <name>.jsonl.gz   one line per *.json file: {"file": <relative path>,
                      "data": <parsed JSON>}; readable with gzip + json without
                      extracting anything (see read_jsonl_gz below)
    <name>_raw.tar.gz every regular file under <dir> (json, cmd, files, txt)
                      as an archive for provenance
With --delete the loose files are removed after both archives are verified
(member count and JSON count re-read from the archives). Rerunning on a
directory that already holds <name>.jsonl.gz merges the old records with the
new loose files (newer file wins) and writes the raw tar with a numbered suffix
so earlier archives are never overwritten.
"""
import gzip
import json
import os
import sys
import tarfile


def read_jsonl_gz(path: str):
    """Yield (relative file name, parsed JSON) from a packed jsonl.gz file.

    Args:
        path (str): the .jsonl.gz archive.
    """
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rec = json.loads(line)
            yield rec["file"], rec["data"]


def main(root: str, name: str, delete: bool) -> None:
    """Pack <root> into <root>/<name>.jsonl.gz and <root>/<name>_raw.tar.gz.

    Args:
        root (str): directory to pack.
        name (str): archive base name.
        delete (bool): remove loose files after verification.
    """
    root = os.path.abspath(root)
    jsonl = os.path.join(root, name + ".jsonl.gz")
    k = 1
    tarp = os.path.join(root, name + "_raw.tar.gz")
    while os.path.exists(tarp):
        k += 1
        tarp = os.path.join(root, f"{name}_raw{k}.tar.gz")
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if p == jsonl or (fn.startswith(name + "_raw") and fn.endswith(".tar.gz")):
                continue
            files.append(p)
    files.sort()
    jsons = [f for f in files if f.endswith(".json")]
    old = {}
    if os.path.exists(jsonl):
        old = {rel: data for rel, data in read_jsonl_gz(jsonl)}
        print(f"merging {len(old)} existing records")
    new_rel = {os.path.relpath(f, root) for f in jsons}
    n_written = 0
    tmp = jsonl + ".tmp"
    with gzip.open(tmp, "wt") as out:
        for rel, data in old.items():
            if rel not in new_rel:
                out.write(json.dumps({"file": rel, "data": data}) + "\n")
                n_written += 1
        for f in jsons:
            with open(f) as fh:
                try:
                    data = json.load(fh)
                except json.JSONDecodeError:
                    print("skipping unreadable", f)
                    continue
            out.write(json.dumps({"file": os.path.relpath(f, root), "data": data}) + "\n")
            n_written += 1
    os.replace(tmp, jsonl)
    with tarfile.open(tarp, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=os.path.relpath(f, root))
    n_back = sum(1 for _ in read_jsonl_gz(jsonl))
    with tarfile.open(tarp) as tar:
        n_members = len(tar.getmembers())
    print(f"{root}: {len(files)} files, {len(jsons)} json -> {n_written} records "
          f"(re-read {n_back}), tar members {n_members}")
    assert n_back == n_written and n_members == len(files)
    if delete:
        for f in files:
            os.remove(f)
        for dp, dns, fns in os.walk(root, topdown=False):
            for d in dns:
                try:
                    os.rmdir(os.path.join(dp, d))
                except OSError:
                    pass
        print("loose files removed")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], "--delete" in sys.argv)
