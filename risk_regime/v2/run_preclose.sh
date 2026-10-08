#!/usr/bin/env bash
# Pre-close estimate: fresh prices with today's partial bar, frozen production models, the action page with a
# PRE-CLOSE banner. Separate working directory so the evening chain's data stay untouched. Usage (repo root):
#   bash risk_regime/v2/run_preclose.sh            # refuses if today's bar is not in the download
#   bash risk_regime/v2/run_preclose.sh --allow-stale   # test outside market hours
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; S="$ROOT/risk_regime/scripts"; V="$ROOT/risk_regime/v2"; W="$ROOT/risk_regime/work"; P="$ROOT/risk_regime/work_pre"; R="$ROOT/risk_regime/results"
mkdir -p "$P/data"; [ -e "$P/data_ew" ] || ln -s "$W/data_ew" "$P/data_ew"; cd "$P"
python3 -c "import yfinance, sklearn" 2>/dev/null || pip install -q -e "$ROOT" scikit-learn
for f in LAB.pkl PRED2.pkl LASTFIT2.pkl DECISION.pkl VVSIG.pkl v2_data.json; do [ -f "$W/$f" ] || { echo "FAILED: $W/$f missing (run the evening chain once first)"; exit 1; }; done
echo "== fetch"; python3 "$S/fetch.py" > "$R/pre_fetch_out.txt" 2>&1 || { echo "FAILED: fetch"; tail -20 "$R/pre_fetch_out.txt"; exit 1; }
echo "== features"; python3 -u "$V/features.py" > "$R/pre_features_out.txt" 2>&1 || { echo "FAILED: features"; tail -20 "$R/pre_features_out.txt"; exit 1; }
echo "== estimate"; python3 -u "$V/preclose.py" "$@" | tee "$R/preclose_out.txt"
python3 "$V/build_action.py" "$P/preclose.json" "$ROOT/risk_regime/action.html"
