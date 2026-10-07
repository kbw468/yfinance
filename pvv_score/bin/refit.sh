#!/bin/bash
# Weekly refit: rebuild the research table, re-estimate composites, rediscover and reconfirm signatures, recalibrate,
# freeze, then score tonight from the new frozen model and verify it matches the full chain's own list.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PVV_TARGET=smooth_42
export PVV_CACHE="${PVV_CACHE:-$PWD/pvv_score/.cache}"
mkdir -p "$PVV_CACHE"
# dependencies: the image carries pandas/numpy/pyarrow/scikit-learn; yfinance (this repo) needs requests & co. Install if missing.
python -c "import requests, multitasking, platformdirs, peewee, bs4, curl_cffi, websockets, pandas, pyarrow, sklearn" 2>/dev/null || pip install -q -e . 2>&1 | tail -1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "prices";            python -m pvv_score.data_io
log "vol indices";       python -c "from pvv_score.volindex import load_vol_indices; load_vol_indices(refresh=True)"
log "research table";    python -m pvv_score.build
log "composite state";   python -m pvv_score.composite --state
log "composite level";   python -m pvv_score.composite
log "live scores";       python -m pvv_score.score_now; python -m pvv_score.score_state
log "calibrate";         python -m pvv_score.calibrate
log "signatures";        python -m pvv_score.signatures
log "signature rule";    python -m pvv_score.signature_rule
log "buylist";           python -m pvv_score.buylist
log "final list (reference)"; python -m pvv_score.final_list
cp pvv_score/results/THE_LIST.csv "$PVV_CACHE/THE_LIST_fullchain.csv"
log "freeze";            python -m pvv_score.freeze
log "score from frozen model"; python -m pvv_score.score_frozen --rows research
python - <<'PY'
import pandas as pd, numpy as np, os
a = pd.read_csv(os.environ["PVV_CACHE"] + "/THE_LIST_fullchain.csv"); b = pd.read_csv("pvv_score/results/THE_LIST.csv")
m = a.merge(b, on="ticker", suffixes=("_a", "_b")); assert len(m) == len(a) == len(b)
for c in ["P_topq_42d", "P_topq_63d", "avg_score"]: assert np.allclose(m[c + "_a"], m[c + "_b"]), c
for c in ["rank", "tier", "n_signatures", "basis"]: assert (m[c + "_a"].astype(str) == m[c + "_b"].astype(str)).all(), c
print("frozen scorer reproduces the full chain's list exactly")
PY
log "recent rows (so the frozen checks have tonight's rows)"; python -m pvv_score.recent_rows
log "history";           python -m pvv_score.history
log "page and PDFs";     python -m pvv_score.list_page; python -m pvv_score.pdfs
log "checks";            python -m pvv_score.checks
ASOF=$(python -c "import pandas as pd; print(pd.read_csv('pvv_score/results/universe_scores_smooth.csv')['asof'].iloc[0])")
git add pvv_score/results
git -c user.name="pvv refit" -c user.email="pvv-refit@users.noreply.github.com" commit -q -m "Weekly refit and list $ASOF (model $(python -c "import json; print(json.load(open('pvv_score/results/model/model.json'))['sha256'])"))" || log "nothing to commit"
git push -q -u origin "$(git rev-parse --abbrev-ref HEAD)"
log "done $ASOF"
