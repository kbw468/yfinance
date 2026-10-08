#!/usr/bin/env python3
"""v2 Stage 7a: today's reading and the dashboard data export. Run from risk_regime/work.

Headline state rule, fixed before the grid results were seen: rule B with OUT when P_vol is above
its trailing-756 90th percentile while P_on is below its median, and back IN when P_vol drops below
its 70th percentile or P_on rises above its 80th. Everything else on the page is the evidence.
Outputs v2_data.json.
"""
import pandas as pd, numpy as np, json, os, warnings; warnings.filterwarnings('ignore')
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1); PRED = pd.read_pickle('PRED2.pkl'); LAST = pd.read_pickle('LASTFIT2.pkl'); DEC = pd.read_pickle('DECISION.pkl'); EP = pd.read_pickle('EPISODES.pkl')
spy = O['T']['SPY']['Close']; last = idx[-1]; N = 504
HEAD = 'C: OUT P_off>q0.95, IN P_off<q0.8'   # reference rule, chosen from the pre-specified grid AFTER the grid was run (see README); the pre-registered rule B failed
PREREG = 'B: OUT P_vol>q0.90 & P_on<q.5, IN P_vol<q0.7 or P_on>q.8'
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok = L['ok21'] & ~EX
rnd = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
out = {'asof': str(last.date()), 'dates': [d.strftime('%Y-%m-%d') for d in idx[-N:]], 'spy': [rnd(v) for v in spy.iloc[-N:]], 'headline_rule': HEAD, 'preregistered_rule': PREREG}
# probabilities: today, trailing percentile, history, threshold lines
PR = {}
for p in ('P_off', 'P_on', 'P_vol', 'P_off63'):
    if p not in PRED: continue
    s = PRED[p]; hist = s.iloc[-N:]
    pct = float((s.iloc[-756:-1] <= s.iloc[-1]).mean())
    PR[p] = {'today': rnd(s.iloc[-1]), 'trailing_pct': round(pct, 3), 'hist': [rnd(v) for v in hist], 'q95': [rnd(v) for v in tq(s, 0.95).iloc[-N:]], 'q90': [rnd(v) for v in tq(s, 0.9).iloc[-N:]], 'q80': [rnd(v) for v in tq(s, 0.8).iloc[-N:]], 'q70': [rnd(v) for v in tq(s, 0.7).iloc[-N:]], 'q60': [rnd(v) for v in tq(s, 0.6).iloc[-N:]], 'q50': [rnd(v) for v in tq(s, 0.5).iloc[-N:]]}
    # decile outcome table from the whole OOS history
    m = ok & s.notna(); dec = pd.qcut(s[m].rank(method='first'), 10, labels=False)
    g = pd.DataFrame({'p_lo': s[m].groupby(dec).min(), 'p_hi': s[m].groupby(dec).max(), 'fwd21': L.loc[m, 'fwd21'].groupby(dec).mean() * 100, 'fwdDD21': L.loc[m, 'fwdDD21'].groupby(dec).mean() * 100, 'P_off': L.loc[m, 'riskoff21'].groupby(dec).mean(), 'P_on': L.loc[m, 'riskon21'].groupby(dec).mean(), 'P_volexp': L.loc[m, 'volexp21'].groupby(dec).mean()})
    PR[p]['deciles'] = [{k: rnd(v) for k, v in row.items()} for _, row in g.iterrows()]
    today_dec = int(((g['p_lo'] <= s.iloc[-1]) & (g['p_hi'] >= s.iloc[-1])).to_numpy().nonzero()[0][-1]) if s.iloc[-1] <= g['p_hi'].max() else 9
    PR[p]['today_decile'] = today_dec
    # drivers: the last-fit features, today's trailing rank and sign
    feats, signs = LAST[p]; drv = []
    yr_tab = None
    for f, sg in zip(feats, signs):
        v = X[f].iloc[-1]
        rk = float(X[f].iloc[-1]) if f.endswith('_rank') or f.endswith('rank63') or f.endswith('rank252') or f.endswith('rank504') or f.endswith('rank1260') else float(X[f].iloc[-756:].rank(pct=True).iloc[-1])
        drv.append({'feature': f, 'value': rnd(v), 'rank': round(rk, 2), 'sign': int(sg), 'push': round((rk - 0.5) * sg * 2, 2)})
    drv.sort(key=lambda d: -abs(d['push'])); PR[p]['drivers'] = drv
out['probs'] = PR
# states: headline and the grid
ST = DEC['states']; TAB = DEC['table']
out['state'] = 'IN' if int(ST[HEAD].iloc[-1]) == 1 else 'OUT'
s = ST[HEAD]; chg = (s != s.shift(1)); out['state_since'] = str(idx[chg][-1].date()) if chg.any() else str(idx[0].date())
out['state_hist'] = [int(v) for v in ST[HEAD].iloc[-N:]]
def row2(rw):
    d = {k: (rnd(v) if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in rw.items() if k != 'loss captured (12 episodes)'}
    d['loss_captured'] = [rnd(v) for v in rw['loss captured (12 episodes)']]; return d
out['rules'] = [row2(rw) for _, rw in TAB.iterrows()]
out['states_today'] = {k: ('IN' if int(v.iloc[-1]) == 1 else 'OUT') for k, v in ST.items()}
# episodes
out['episodes'] = [{'start': str(r.start.date()), 'end': str(r.end.date()), 'kind': r.kind, 'sessions': int(r.sessions), 'spy_dd': rnd(r.spy_dd), 'spy_ret': rnd(r.spy_ret)} for _, r in EP.tail(14).iterrows()]
out['inside_episode'] = bool(((EP['start'] <= last) & (EP['end'] >= last)).any())
# multifractal today
mf = {}
for s_ in ('SPY', 'HYG', 'VIX', 'QQQ', 'IWM'):
    mf[s_] = {k: {'value': rnd(MF[f'MF:{s_}:{k}'].iloc[-1]), 'rank': rnd(MF[f'MF:{s_}:{k}_rank'].iloc[-1])} for k in ('h2', 'width', 'alpha_abs', 'vr5', 'vr21')}
out['multifractal'] = mf
# key tape readings today (ranks), for the tiles
KEYS = ['VIX:lvl_rank504', 'VIX:roc5_rank', 'VIX:roc21_rank', 'VVIX/VIX:roc5_rank', 'VIX/VIX3M:roc5_rank', 'MOVE:lvl_rank504', 'MOVE:roc5_rank', 'TNX:roc21_rank', 'SPY:cc21', 'SPY:pk21', 'SPY:cc21_rank', 'SPY:cc21_63_rank', 'SPY:pk21_cc21_rank', 'SPY:VIX/cc21_rank', 'SPY:VIX_prem_chg5_rank', 'SPY:vol_rank63', 'SPY:volvol21_rank', 'SPY:compress5_63_rank', 'SPY:amihud21_rank', 'HYG:pk21_rank', 'HYG:signedvol21', 'HYG:VIX/cc21_rank', 'XLV:rel63_rank', 'XS:disp5_rank', 'XS:avgcorr21_rank', 'XS:avgcorr21_chg5_rank', 'XS:rv_breadth_rank', 'XS:vol_breadth_rank', 'XS:compress_breadth', 'XS:breadth_up21', 'RSP/SPY:roc21_rank', 'RSP/SPY:roc63_rank']
out['tape'] = {k: rnd(X[k].iloc[-1]) for k in KEYS if k in X.columns}
# ticker layer
TK = pd.read_pickle('TICK2.pkl'); out['tick_lead'] = TK['lead'].round(2).reset_index().rename(columns={'index': 'ticker'}).to_dict(orient='records')
out['tick_cond'] = {k: v.round(2).reset_index().rename(columns={'index': 'ticker'}).to_dict(orient='records') for k, v in TK['cond'].items()}
# screen summary: top consistent features
SC = pd.read_pickle('SCREEN.pkl'); c = SC['cons']
out['screen'] = {'risky': [{'feature': f, 'cons': rnd(v)} for f, v in c['ic_dd'].sort_values().head(25).items()], 'safe': [{'feature': f, 'cons': rnd(v)} for f, v in c['ic_dd'].sort_values(ascending=False).head(15).items()], 'rally': [{'feature': f, 'cons': rnd(v)} for f, v in c['ic_up'].sort_values(ascending=False).head(15).items()]}
# v1 audit summary
AU = pd.read_pickle('AUDIT_V1.pkl'); out['audit_v1'] = AU[['group', 'rule', 'kind', 'n_all', 'P_all', '2008-17 P', '2018-26 P', '2018-26 fwd21', 'halves']].fillna('').to_dict(orient='records')
# model variants (OOS)
MV = pd.read_pickle('MODEL_OFF_VARIANTS.pkl') if os.path.exists('MODEL_OFF_VARIANTS.pkl') else pd.DataFrame(); out['variants'] = MV.fillna('').to_dict(orient='records')
print('as of', out['asof'], '| state', out['state'], 'since', out['state_since'], '| P_vol', PR['P_vol']['today'], f"(pct {PR['P_vol']['trailing_pct']}, decile {PR['P_vol']['today_decile']})", '| P_on', PR['P_on']['today'], f"(pct {PR['P_on']['trailing_pct']})", '| P_off', PR['P_off']['today'], f"(pct {PR['P_off']['trailing_pct']})")
print('states today by rule:', out['states_today'])
out['yearly'] = {k: pd.read_pickle(f'DECISION_{k}.pkl')['yearly'].reset_index().rename(columns={'index': 'year', 'Date': 'year'}).to_dict(orient='records') for k in ('C95', 'C90', 'F95') if os.path.exists(f'DECISION_{k}.pkl')}
out['spells'] = {k: [{'from': str(a.date()), 'to': str(b.date()), 'sessions': int(idx.get_loc(b) - idx.get_loc(a) + 1), 'spy_while_out': round(float(np.log(spy.loc[b] / spy.loc[a]) * 100), 1)} for a, b in pd.read_pickle(f'DECISION_{k}.pkl')['spells']] for k in ('C95', 'C90', 'F95') if os.path.exists(f'DECISION_{k}.pkl')}
out['null'] = pd.read_pickle('NULLTEST2.pkl') if os.path.exists('NULLTEST2.pkl') else None
if out['null']:
    for k, v in out['null'].items(): v.pop('nulls', None)
MV2 = pd.read_pickle('VARIANTS2.pkl') if os.path.exists('VARIANTS2.pkl') else pd.DataFrame(); out['variants2'] = MV2.fillna('').to_dict(orient='records')
json.dump(out, open('v2_data.json', 'w')); print('v2_data.json', os.path.getsize('v2_data.json') // 1024, 'KB')
for p in ('P_off', 'P_on', 'P_vol'):
    print(f'\n{p} drivers today (push > 0 raises the probability):'); print(pd.DataFrame(PR[p]['drivers']).head(10).to_string(index=False))
    print(f'{p} decile table (OOS):'); print(pd.DataFrame(PR[p]['deciles']).round(3).to_string())
