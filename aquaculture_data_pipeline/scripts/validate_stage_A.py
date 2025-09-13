#!/usr/bin/env python3
import sys, glob, json, os, re, datetime, pandas as pd
from pathlib import Path

ok = True
notes = []

def check_exists(pattern, min_count=1, kind="file"):
    global ok
    matches = glob.glob(pattern)
    if len(matches) >= min_count:
        notes.append(f"[OK] {pattern} -> {len(matches)} found")
        return matches
    else:
        notes.append(f"[MISS] {pattern} -> found {len(matches)}")
        ok = False
        return []

check_exists("data/raw/fishbase/species.*")
check_exists("data/raw/fishbase/synonyms.*")
check_exists("data/processed/fishbase_species_standardized.parquet")

adm = check_exists("data/aux/vn_provinces.gpkg")
if adm:
    try:
        import geopandas as gpd
        gdf = gpd.read_file(adm[0])
        crs_ok = (gdf.crs is not None and "4326" in str(gdf.crs).lower())
        notes.append(f"   CRS: {gdf.crs} ({'OK' if crs_ok else 'WARN'}) | provinces={len(gdf)}")
        if len(gdf) < 40:
            notes.append("   [WARN] Less than 40 features; expected ~63 provinces.")
    except Exception as e:
        notes.append(f"   [ERR] Could not read GPKG: {e}")
        ok = False

check_exists("data/raw/bio_oracle/present/*.tif", min_count=1)
check_exists("data/raw/bio_oracle/mid_2050_ssp245/*.tif", min_count=1)
check_exists("data/raw/bio_oracle/mid_2050_ssp585/*.tif", min_count=1)
check_exists("data/raw/bio_oracle/end_2100_ssp245/*.tif", min_count=1)
check_exists("data/raw/bio_oracle/end_2100_ssp585/*.tif", min_count=1)

check_exists("data/raw/gebco/gebco_vietnam.tif")

buf = check_exists("data/processed/vn_coastal_buffer_50km.gpkg")
if buf:
    try:
        import geopandas as gpd
        g = gpd.read_file(buf[0])
        area = g.to_crs(3857).area.sum()
        notes.append(f"   Buffer area ~ {area/1e6:.2f} km^2")
    except Exception as e:
        notes.append(f"   [ERR] Could not read buffer gpkg: {e}")
        ok = False

print("\nStage A Validation Report (" + datetime.datetime.utcnow().strftime("%Y-%m-%d") + " UTC)\n" + "-"*60)
for n in notes: print(n)
print("-"*60)
print("RESULT:", "PASS ✅" if ok else "NEEDS FIXES ❌")
if not ok:
    sys.exit(2)
