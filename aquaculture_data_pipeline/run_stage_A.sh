#!/usr/bin/env bash
# set -euo pipefail
# LOGDIR="reports/logs"
# mkdir -p "$LOGDIR"

echo "[1/7] FishBase via rfishbase"
Rscript scripts/fishbase_download_R.R | tee "$LOGDIR/01_fishbase_r.log" || true

echo "[2/7] FishBase via HuggingFace"
python3 scripts/fishbase_download_hf.py | tee "$LOGDIR/02_fishbase_hf.log" || true

echo "[3/7] Standardize species names"
python3 scripts/fishbase_clean.py | tee "$LOGDIR/03_fishbase_clean.log"

echo "[4/7] VN ADM1 boundaries"
python3 scripts/get_admin_boundaries.py | tee "$LOGDIR/04_admin.log"

echo "[5/7] List Bio-ORACLE layers (index)"
Rscript scripts/list_bio_oracle_layers.R | tee "$LOGDIR/05_bio_layers.log" || true

echo "[6/7] Download Bio-ORACLE rasters"
Rscript scripts/download_bio_oracle.R | tee "$LOGDIR/06_bio_oracle.log"

echo "[7/7] Coastal datasets (GEBCO clip + 50km buffer)"
python3 scripts/clip_gebco_vietnam.py | tee "$LOGDIR/07_gebco_clip.log" || true
python3 scripts/make_coastal_buffer.py | tee "$LOGDIR/08_buffer.log"

echo "[Validate] Stage A"
python3 scripts/validate_stage_A.py | tee "$LOGDIR/09_validate.log"

echo "Done Stage A."
