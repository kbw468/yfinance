#!/usr/bin/env bash
# Regenerate the whole risk-regime pipeline and dashboard from fresh Yahoo data.
# Usage: bash risk_regime/scripts/run_all.sh   (from the repo root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S="$ROOT/risk_regime/scripts"; W="$ROOT/risk_regime/work"; R="$ROOT/risk_regime/results"
mkdir -p "$W/data" "$R"; cd "$W"
python3 -c "import yfinance, sklearn, statsmodels, scipy" 2>/dev/null || pip install -q -e "$ROOT" scikit-learn statsmodels scipy
for s in fetch build rebuild features_roc events sig tickers scorecard playbook snapshot tick_roc tick_snapshot rv rv_snapshot score analog coarse; do
  echo "== $s"; python3 "$S/$s.py" > "$R/${s}_out.txt" 2>&1 || { echo "FAILED: $s"; tail -20 "$R/${s}_out.txt"; exit 1; }
done
cp dash_data.json tick_data.json rv_data.json score_data.json "$R/"
python3 "$S/build_dashboard.py" "$W/dash_data.json" "$ROOT/risk_regime/dashboard.html"
echo "done: $(date -u +%F) $(python3 -c "import json;print(json.load(open('dash_data.json'))['asof'])")"
