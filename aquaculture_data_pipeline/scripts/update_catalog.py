#!/usr/bin/env python3
import csv, hashlib, os, glob, datetime, sys, pandas as pd
from pathlib import Path

CAT = Path("data_catalog.csv")
if not CAT.exists():
    print("data_catalog.csv not found.", file=sys.stderr); sys.exit(1)

def file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def summarize_glob(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return "", 0, 0
    total = 0
    parts = []
    for p in paths:
        if os.path.isfile(p):
            total += os.path.getsize(p)
            parts.append(file_checksum(p))
    ch = hashlib.sha256(("".join(parts)).encode()).hexdigest()[:16] if parts else ""
    return ch, len(paths), total

df = pd.read_csv(CAT)

now = datetime.datetime.utcnow().strftime("%Y-%m-%d")
for i, row in df.iterrows():
    path = str(row.get("path","")).strip()
    if not path:
        continue
    checksum = ""
    if "*" in path or "?" in path:
        checksum, nfiles, total = summarize_glob(path)
        df.loc[i, "preprocessing_notes"] = (row.get("preprocessing_notes","") or "") + f" | matched {nfiles} files, total {total/1e6:.1f} MB"
    else:
        p = Path(path)
        if p.exists() and p.is_file():
            checksum = file_checksum(path)
            df.loc[i, "preprocessing_notes"] = (row.get("preprocessing_notes","") or "") + f" | size {p.stat().st_size/1e6:.1f} MB"
        elif p.exists() and p.is_dir():
            checksum, nfiles, total = summarize_glob(os.path.join(path, '**', '*'))
            df.loc[i, "preprocessing_notes"] = (row.get("preprocessing_notes","") or "") + f" | dir with {nfiles} files, total {total/1e6:.1f} MB"
        else:
            pass
    if checksum:
        df.loc[i, "checksum"] = checksum
        df.loc[i, "date_downloaded"] = df.loc[i, "date_downloaded"] or now

df.to_csv(CAT, index=False)
print("Updated", CAT)
