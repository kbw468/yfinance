#!/usr/bin/env bash
# Daily v2 chain: fresh data, feature store, multifractal layer, labels, walk-forward models, decision
# layer, ticker layer, today's reading, dashboard and PDF. Usage: bash risk_regime/v2/run_v2.sh (repo root).
# Research-only steps (audit_v1, screen, nulltest, the 15% layer's permutation test) are not rerun daily; their outputs are kept in results/.
# big.py runs with 0 null draws and carries the stored permutation result forward.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; S="$ROOT/risk_regime/scripts"; V="$ROOT/risk_regime/v2"; W="$ROOT/risk_regime/work"; R="$ROOT/risk_regime/results"
mkdir -p "$W/data" "$W/data_ew" "$R"; cd "$W"
python3 -c "import yfinance, sklearn, scipy, playwright" 2>/dev/null || pip install -q -e "$ROOT" scikit-learn scipy playwright
for s in fetch fetch_ew; do echo "== $s"; python3 "$S/$s.py" > "$R/${s}_out.txt" 2>&1 || { echo "FAILED: $s"; tail -20 "$R/${s}_out.txt"; exit 1; }; done
# the v1 build step is kept only for its universe guard and the audit pickles the dashboard shows
python3 "$S/build.py" > "$R/build_out.txt" 2>&1 || { echo "FAILED: build"; tail -20 "$R/build_out.txt"; exit 1; }
for s in features fractal labels model2 decision path2020 tickers big onset snapshot; do echo "== $s"; a=""; [ "$s" = "big" ] && a="0"; python3 -u "$V/$s.py" $a > "$R/v2_${s}_out.txt" 2>&1 || { echo "FAILED: $s"; tail -20 "$R/v2_${s}_out.txt"; exit 1; }; done
cp v2_data.json manifest.json "$R/"
python3 "$V/build_dashboard_v2.py" "$W/v2_data.json" "$ROOT/risk_regime/dashboard_v2.html"
python3 "$V/build_action.py" "$W/v2_data.json" "$ROOT/risk_regime/action.html"
python3 "$V/pdf_v2.py" "$ROOT/risk_regime/dashboard_v2.html" "$R/VOL_TAPE_READING_v2.pdf"
echo "done: $(date -u +%F) $(python3 -c "import json;print(json.load(open('v2_data.json'))['asof'])")"
