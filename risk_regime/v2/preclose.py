#!/usr/bin/env python3
"""Pre-close estimate of today's instruction, from live prices. Run from risk_regime/work_pre after fetch.py and
features.py have been run there (run_preclose.sh does this). Usage: python3 preclose.py [--allow-stale]

Yahoo's daily download carries today's bar while the market is open: the last trade stands in for the close and the
high and low are the session's so far. This script treats that partial bar as today's close and applies the frozen
production pieces, nothing refit beyond them:
  - the drawdown, rally and vol-expansion models: the current year's production fit (same features, monotone signs,
    training rows and parameters as model2.py's last window, so the same model), applied to today's features;
  - the reference rule: yesterday's state, OUT if today's drawdown probability is at or above the trailing-756 95th
    percentile of the production history through yesterday, IN if below the 80th;
  - VIX outrunning VVIX: yesterday's state, live if VIX's 21-day rate of change and the VIX/VVIX ratio's are both in
    their top 10% of the trailing 504 sessions including today, quiet once VIX's falls below its median.
The backtests act on a session's close at that close, which is what a pre-close reading plus a market-on-close order
does. Writes preclose.json (a copy of the evening v2_data.json with today's values) for build_action.py.
"""
import sys, json, datetime as dt, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from zoneinfo import ZoneInfo
from sklearn.ensemble import HistGradientBoostingClassifier
W = '../work'; allow_stale = '--allow-stale' in sys.argv
now = dt.datetime.now(ZoneInfo('America/New_York'))
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index
L = pd.read_pickle(f'{W}/LAB.pkl'); PRED = pd.read_pickle(f'{W}/PRED2.pkl'); LAST = pd.read_pickle(f'{W}/LASTFIT2.pkl')
DEC = pd.read_pickle(f'{W}/DECISION.pkl'); VS = pd.read_pickle(f'{W}/VVSIG.pkl'); D = json.load(open(f'{W}/v2_data.json'))
today = idx[-1]; prev = PRED.index[-1]
if today.date() != now.date() and not allow_stale:
    print(f'NO LIVE BAR: the last session in the download is {today.date()}, not today {now.date()}; the market may be closed or Yahoo has not posted the bar'); sys.exit(2)
if today <= prev and not allow_stale:
    print(f'NOTHING NEW: the evening data already covers {prev.date()}'); sys.exit(2)
PAR = dict(learning_rate=0.05, max_iter=200, max_bins=64, random_state=0, max_depth=2, min_samples_leaf=300, l2_regularization=2.0)
EX = (L.index >= pd.Timestamp('2020-02-01')) & (L.index <= pd.Timestamp('2020-07-31'))
cut = pd.Timestamp(f'{now.year}-01-01') - pd.Timedelta(days=100)
P_NOW = {}
for p, lab in (('P_off', 'riskoff21'), ('P_on', 'riskon21'), ('P_vol', 'volexp21')):
    feats, signs = LAST[p]; missing = [f for f in feats if f not in F.columns]
    if missing: print('missing features', missing); sys.exit(1)
    y = L[lab]; ok = (L['ok21'] & ~EX & y.notna() & (L.index < cut) & (L.index.year >= 1993))
    rows = L.index[ok]; Xtr = F.loc[rows, feats].to_numpy(np.float32)
    clf = HistGradientBoostingClassifier(monotonic_cst=list(signs), **PAR).fit(Xtr, y[ok].to_numpy())
    P_NOW[p] = float(clf.predict_proba(F.loc[[today], feats].to_numpy(np.float32))[:, 1][0])
    # consistency check: the refit reproduces yesterday's production value
    chk = float(clf.predict_proba(F.loc[[prev], feats].to_numpy(np.float32))[:, 1][0]); P_NOW[p + '_check'] = (round(chk, 4), round(float(PRED[p].iloc[-1]), 4))
hist = PRED['P_off'].dropna().iloc[-756:]; q95 = float(hist.quantile(0.95)); q80 = float(hist.quantile(0.8))
ref_prev = int(DEC['states']['C: OUT P_off>q0.95, IN P_off<q0.8'].iloc[-1]); pf = P_NOW['P_off']
ref_now = 0 if (ref_prev == 1 and pf >= q95) else (1 if (ref_prev == 0 and pf < q80) else ref_prev)
vix = O['I']['VIX']['Close'].reindex(idx); vvix_raw = O['I']['VVIX']['Close'].reindex(idx)
VVIX_NOTE = '' if not np.isnan(vvix_raw.iloc[-1]) else f'VVIX live bar missing at this run; used its last close {vvix_raw.dropna().index[-1].date()}'
vvix = vvix_raw.ffill(limit=3); vix = vix.ffill(limit=3); ratio = np.log(vix / vvix)
if VVIX_NOTE: print('NOTE:', VVIX_NOTE)
rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True)
V21 = rk(np.log(vix / vix.shift(21))); R21 = rk(ratio - ratio.shift(21)); vr, rr = float(V21.iloc[-1]), float(R21.iloc[-1])
vv_prev = int(VS['state'].iloc[-1]); vv_now = 0 if (vv_prev == 1 and vr >= 0.9 and rr >= 0.9) else (1 if (vv_prev == 0 and vr < 0.5) else vv_prev)
def tpct(p): s = PRED[p].dropna().iloc[-755:]; return float((s <= P_NOW[p]).mean())
def dec_of(p, v):
    d = D['probs'][p]['deciles']; his = [x['p_hi'] for x in d]; return int(min(9, np.searchsorted(his, v)))
for p in ('P_off', 'P_on', 'P_vol'):
    D['probs'][p]['today'] = round(P_NOW[p], 3); D['probs'][p]['trailing_pct'] = round(tpct(p), 3); D['probs'][p]['today_decile'] = dec_of(p, P_NOW[p])
D['state'] = 'IN' if ref_now == 1 else 'OUT'
if ref_now != ref_prev: D['state_since'] = str(today.date())
vt = D['vvsig']['today']; vt.update(live=bool(vv_now == 0), vix_roc21_rank=round(vr, 3), ratio_roc21_rank=round(rr, 3), vix=round(float(vix.iloc[-1]), 2), vvix=round(float(vvix.iloc[-1]), 2),
    vix_roc21_pct=round(float(np.log(vix.iloc[-1] / vix.iloc[-22]) * 100), 1), ratio_roc21_pct=round(float((ratio.iloc[-1] - ratio.iloc[-22]) * 100), 1))
if vv_now == 0 and vv_prev == 1: vt['since'] = str(today.date())
D['asof'] = str(today.date()); D['preclose'] = f"{now.strftime('%H:%M')} ET, prices delayed about 15 minutes" + (f'; {VVIX_NOTE}' if VVIX_NOTE else '')
spy = O['T']['SPY']['Close'].reindex(idx); D['preclose_spy'] = {'last': round(float(spy.iloc[-1]), 2), 'chg_pct': round(float((spy.iloc[-1] / spy.iloc[-2] - 1) * 100), 2)}
json.dump(D, open('preclose.json', 'w'))
print(f"pre-close {now.strftime('%Y-%m-%d %H:%M')} ET | bar {today.date()} | SPY {D['preclose_spy']['last']} ({D['preclose_spy']['chg_pct']:+.2f}%) | P_off {pf:.3f} pct {tpct('P_off'):.2f} (q95 {q95:.3f}, q80 {q80:.3f}) ref {ref_prev}->{ref_now} | VIX {vt['vix']} VVIX {vt['vvix']} V21 {vr:.2f} R21 {rr:.2f} vv {vv_prev}->{vv_now} | check (refit, production) yesterday: {P_NOW['P_off_check']} {P_NOW['P_on_check']} {P_NOW['P_vol_check']}")
