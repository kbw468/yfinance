#!/bin/bash
# Nightly scoring from the frozen model. Refits nothing. Run after 21:30 UTC so the session's bar is final.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PVV_TARGET=smooth_42
export PVV_CACHE="${PVV_CACHE:-$PWD/pvv_score/.cache}"
mkdir -p "$PVV_CACHE"
# dependencies: the image carries pandas/numpy/pyarrow/scikit-learn; yfinance (this repo) needs requests & co. Install if missing.
python -c "import requests, multitasking, platformdirs, peewee, bs4, curl_cffi, websockets, pandas, pyarrow, sklearn, scipy, lightgbm" 2>/dev/null || pip install -q -e . pyarrow scikit-learn scipy lightgbm 2>&1 | tail -1
cp -n pvv_score/results/model/factor_registry.csv "$PVV_CACHE/factor_registry.csv" 2>/dev/null || true   # frozen copy of the factor registry for a fresh cache
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "prices"
python -m pvv_score.data_io
log "vol indices"
python -c "from pvv_score.volindex import load_vol_indices; print(load_vol_indices(refresh=True).tail(1))"
log "recent-window features"
python -m pvv_score.recent_rows
log "score (frozen model)"
python -m pvv_score.score_frozen --rows "$PVV_CACHE/recent_rows.parquet"
log "history / movers / scorecard"
python -m pvv_score.history
log "page and PDFs"
python -m pvv_score.list_page
python -m pvv_score.pdfs
log "checks"
python -m pvv_score.checks --frozen
ASOF=$(python -c "import pandas as pd; print(pd.read_csv('pvv_score/results/universe_scores_smooth.csv')['asof'].iloc[0])")
log "commit $ASOF"
git add pvv_score/results
git -c user.name="pvv nightly" -c user.email="pvv-nightly@users.noreply.github.com" commit -q -m "Nightly list $ASOF (frozen model $(python -c "import json; print(json.load(open('pvv_score/results/model/model.json'))['sha256'])"))" || log "nothing to commit"
git push -q -u origin "$(git rev-parse --abbrev-ref HEAD)"
log "done $ASOF"
