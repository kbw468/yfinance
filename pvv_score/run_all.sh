#!/bin/bash
# Full pipeline after a factor-library change. Logs to $LOG.
set -u
LOG=${LOG:-/tmp/claude-0/-home-user-yfinance/6b0db6a0-0646-53b0-950e-60d46b6034fa/scratchpad/roc}
mkdir -p $LOG
cd /home/user/yfinance
step(){ echo "[$(date +%H:%M:%S)] $1"; }
step "build"; python -m pvv_score.build > $LOG/build.log 2>&1 || { echo BUILD_FAILED; exit 1; }
step "stage A (parallel): run_eval, run_conditional, composite --state"
python -m pvv_score.run_eval > $LOG/run_eval.log 2>&1 &
python -m pvv_score.run_conditional > $LOG/run_conditional.log 2>&1 &
python -m pvv_score.composite --state > $LOG/composite_state.log 2>&1 &
wait
step "stage B (parallel): composite level, events, model variants"
python -m pvv_score.composite > $LOG/composite_level.log 2>&1 &
python -m pvv_score.events > $LOG/events.log 2>&1 &
python -m pvv_score.run_model_variants > $LOG/variants.log 2>&1 &
wait
step "stage C: identity x2, score_now, score_state"
python -m pvv_score.run_identity level > $LOG/identity_level.log 2>&1
python -m pvv_score.run_identity state > $LOG/identity_state.log 2>&1
python -m pvv_score.score_now > $LOG/score_now.log 2>&1
python -m pvv_score.score_state > $LOG/score_state.log 2>&1
python -m pvv_score.ranking_page > $LOG/page.log 2>&1
step "ALL_DONE"
