#!/bin/bash
# Intraday projection of tonight's list from today's partial bars (run during the session, typically 15:25 New York).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PVV_TARGET=smooth_42
export PVV_CACHE="${PVV_CACHE:-$PWD/pvv_score/.cache}"
mkdir -p "$PVV_CACHE"
cp -n pvv_score/results/model/factor_registry.csv "$PVV_CACHE/factor_registry.csv" 2>/dev/null || true
python -c "import requests, multitasking, platformdirs, peewee, bs4, curl_cffi, websockets, pandas, pyarrow, sklearn, scipy, lightgbm" 2>/dev/null || python -m pip install -q -e . pyarrow scikit-learn scipy lightgbm 2>&1 | tail -1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
if [ ! -f "$PVV_CACHE/Close.parquet" ]; then log "prices (no cache)"; python -m pvv_score.data_io; fi
log "vol indices"; python -c "from pvv_score.volindex import load_vol_indices; load_vol_indices(refresh=True)" > /dev/null
log "intraday projection"; python -m pvv_score.intraday "$@"
git add pvv_score/results/intraday
git -c user.name="pvv intraday" -c user.email="pvv-intraday@users.noreply.github.com" commit -q -m "Intraday projection $(date -u +%Y-%m-%dT%H:%M)Z" || log "nothing to commit"
git push -q -u origin "$(git rev-parse --abbrev-ref HEAD)"
log "done"
