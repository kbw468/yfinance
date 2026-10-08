#!/usr/bin/env python3
"""v2 Stage 2b/3a: forward labels and objective vol episodes. Run from risk_regime/work.

Labels (on SPY sessions, Feb-Jul 2020 masked for both t and t+h):
  fwd{h}, fwdDD{h}, fwdUP{h} for h in 5,10,21,42,63 (log)
  riskoff21: 5% drawdown within 21 sessions; riskoff63: 10% within 63; riskon21: 5% up within 21
  volexp21: SPY 21d realized vol 21 sessions ahead at least 1.5x today's
  epstart21: a realized or implied vol episode starts within 21 sessions (not already inside one)
Episodes: realized = SPY cc21 trailing-504 rank crossing above 0.8, ending below 0.5; implied = VIX level rank
likewise. Outputs LAB.pkl, EPISODES.pkl, and a signature table of key features around episode starts.
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = O['idx']
spy = O['T']['SPY']['Close']; EX0, EX1 = pd.Timestamp('2020-02-01'), pd.Timestamp('2020-07-31')
L = pd.DataFrame(index=idx)
for h in (5, 10, 21, 42, 63):
    L[f'fwd{h}'] = np.log(spy.shift(-h) / spy)
    L[f'fwdDD{h}'] = np.log(spy[::-1].rolling(h).min()[::-1].shift(-1) / spy)
    L[f'fwdUP{h}'] = np.log(spy[::-1].rolling(h).max()[::-1].shift(-1) / spy)
    tf = pd.Series(idx, index=idx).shift(-h)
    bad = ((idx >= EX0) & (idx <= EX1)) | ((tf >= EX0) & (tf <= EX1)).values
    L[f'ok{h}'] = ~bad & L[f'fwd{h}'].notna()
L['riskoff21'] = (L['fwdDD21'] <= np.log(0.95)).astype(float); L['riskoff63'] = (L['fwdDD63'] <= np.log(0.90)).astype(float); L['riskon21'] = (L['fwdUP21'] >= np.log(1.05)).astype(float)
cc21 = F['SPY:cc21'].astype(float); L['volexp21'] = (np.log(cc21.shift(-21) / cc21) >= np.log(1.5)).astype(float)
L.loc[cc21.shift(-21).isna(), 'volexp21'] = np.nan
# ---- episodes
def episodes(rank, hi=0.8, lo=0.5, name=''):
    inside = False; rows = []; start = None; st = pd.Series(False, index=idx)
    for d, v in rank.items():
        if np.isnan(v): continue
        if not inside and v >= hi: inside = True; start = d; st[d] = True
        elif inside and v <= lo: inside = False; rows.append((start, d)); start = None
    if inside: rows.append((start, idx[-1]))
    tab = pd.DataFrame(rows, columns=['start', 'end']); tab['sessions'] = [idx.get_loc(b) - idx.get_loc(a) + 1 for a, b in rows]
    tab['spy_dd'] = [np.log(spy.loc[a:b].min() / spy.loc[:a].iloc[-1]) * 100 for a, b in rows]
    tab['spy_ret'] = [np.log(spy.loc[b] / spy.loc[a]) * 100 for a, b in rows]; tab['kind'] = name
    return st, tab
rv_rank = cc21.rolling(504, min_periods=300).rank(pct=True); st_rv, ep_rv = episodes(rv_rank, name='realized')
st_iv, ep_iv = episodes(F['VIX:lvl_rank504'].astype(float), name='implied')
EP = pd.concat([ep_rv, ep_iv]).sort_values('start').reset_index(drop=True)
starts = (st_rv | st_iv)
inside_any = pd.Series(False, index=idx)
for _, r in EP.iterrows(): inside_any.loc[r.start:r.end] = True
fut = starts[::-1].rolling(21).max()[::-1].shift(-1)   # any start in next 21 sessions
L['epstart21'] = ((fut > 0) & ~inside_any).astype(float); L.loc[fut.isna(), 'epstart21'] = np.nan
L.to_pickle('LAB.pkl'); EP.to_pickle('EPISODES.pkl')
ok = L['ok21'] & ~((idx >= EX0) & (idx <= EX1))
print('labels on', len(L), 'sessions; base rates (ok21, ex-2020):', {k: round(float(L.loc[ok, k].mean()), 3) for k in ('riskoff21','riskoff63','riskon21','volexp21','epstart21')})
print(f'episodes: {len(ep_rv)} realized, {len(ep_iv)} implied; median length {EP.sessions.median():.0f} sessions; median SPY drawdown inside {EP.spy_dd.median():.1f}%')
print(EP.tail(12).to_string())
# ---- signatures of key features around episode starts (median rank, sessions -10..+10)
KEY = ['VIX:roc5_rank','VIX:roc21_rank','VVIX/VIX:roc5_rank','VIX/VIX3M:roc5_rank','MOVE:roc5_rank','TNX:roc5_rank','TNX:roc21_rank','SKEW:roc10_rank','SPY:cc5_21_rank','SPY:cc21_63_rank','SPY:pk5_21_rank','SPY:cc21_roc5_rank','SPY:volofvol21_rank','SPY:vol_rank63','SPY:vroc5_rank','SPY:volvol21_rank','SPY:volvol21_roc5_rank','SPY:compress5_63_rank','SPY:amihud21_rank','SPY:VIX/cc21_rank','SPY:VIX_prem_chg5_rank','HYG:rel5_rank','HYG:VIX/cc21_rank','IWM:rel5_rank','XLU:rel5_rank','TLT:rel5_rank','XS:disp1_rank','XS:disp5_rank','XS:avgcorr21_rank','XS:avgcorr21_chg5_rank','XS:rv_breadth_rank','XS:vol_breadth_rank','XS:volvol_median_rank','XS:compress_breadth','XS:breadth_up21','RSP/SPY:roc21_rank']
KEY = [k for k in KEY if k in F.columns]; OFFS = list(range(-10, 11))
anchors = [d for d in idx[starts] if d >= pd.Timestamp('2000-01-01') and not ((d >= EX0) & (d <= EX1))]
rows = {}
for k in KEY:
    s = F[k].astype(float); pos = [idx.get_loc(a) for a in anchors]
    M = np.array([[s.iloc[p + o] if 0 <= p + o < len(s) else np.nan for o in OFFS] for p in pos]); rows[k] = np.nanmedian(M, axis=0)
sig = pd.DataFrame(rows, index=OFFS).T.round(2); sig.to_pickle('EP_SIG.pkl')
print(f'\n=== median trailing-rank of key features around {len(anchors)} vol-episode starts (0.5 = typical) ===')
print(sig.to_string())
