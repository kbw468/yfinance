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
# ---------- 15% layer (Stage 8): the loss that matters
import re as _re, datetime as _dt
def jsafe(v):
    if isinstance(v, dict): return {str(k): jsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)): return [jsafe(x) for x in v]
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (float, np.floating)): return rnd(v)
    if isinstance(v, (pd.Timestamp, _dt.date)): return str(v)[:10]
    return v
if os.path.exists('BIG.pkl') and os.path.exists('PRED15.pkl') and os.path.exists('LASTFIT15.pkl'):
    BIG = pd.read_pickle('BIG.pkl'); P15 = pd.read_pickle('PRED15.pkl'); L15 = pd.read_pickle('LASTFIT15.pkl'); FT = {}
    EX0, EX1 = pd.Timestamp('2020-02-01'), pd.Timestamp('2020-07-31')
    def fwd15(N):
        fmin = spy[::-1].rolling(N, min_periods=N).min()[::-1].shift(-1); y = (fmin / spy - 1 <= -0.15).astype(float)
        touches = pd.Series(False, index=idx); pos0 = idx.get_loc(idx[idx >= EX0][0]); touches.iloc[max(0, pos0 - N):] = idx[max(0, pos0 - N):] <= EX1
        return y, fmin.notna() & spy.notna() & ~touches, (fmin / spy - 1) * 100, np.log(spy.shift(-N) / spy) * 100
    isrank = lambda f: f.endswith('_rank') or f.endswith(('rank63', 'rank252', 'rank504', 'rank1260'))
    for key, N in (('dd15_63', 63), ('dd15_126', 126)):
        s_ = P15[key]; y15, ok15, worst, fwd = fwd15(N); pct = float((s_.iloc[-756:-1] <= s_.iloc[-1]).mean())
        mm = ok15 & s_.notna() & (idx.year >= 2005); dec = pd.qcut(s_[mm].rank(method='first'), 10, labels=False)
        g = pd.DataFrame({'p_lo': s_[mm].groupby(dec).min(), 'p_hi': s_[mm].groupby(dec).max(), 'hit': y15[mm].groupby(dec).mean(), 'worst': worst[mm].groupby(dec).mean(), 'fwd': fwd[mm].groupby(dec).mean()})
        td = int(min(9, np.searchsorted(g['p_hi'].to_numpy(), s_.iloc[-1])))
        feats, signs = L15[key]; drv = []
        for f, sg in zip(feats, signs):
            if f not in X.columns: continue
            rk = float(X[f].iloc[-1]) if isrank(f) else float(X[f].iloc[-756:].rank(pct=True).iloc[-1])
            drv.append({'feature': f, 'value': rnd(X[f].iloc[-1]), 'rank': round(rk, 2), 'sign': int(sg), 'push': round((rk - 0.5) * sg * 2, 2)})
        drv.sort(key=lambda d: -abs(d['push']))
        FT[key] = {'today': rnd(s_.iloc[-1]), 'trailing_pct': round(pct, 3), 'today_decile': td, 'deciles': [{k: rnd(v) for k, v in row.items()} for _, row in g.iterrows()], 'drivers': drv, 'base': rnd(y15[mm].mean())}
    ET = BIG['event_table']; FT['events'] = jsafe(ET.to_dict(orient='records'))
    summ = {}
    for k in [c[:-len(' first>95th')] for c in ET.columns if c.endswith(' first>95th')]:
        at_peak = before5 = after5 = never = 0
        for v in ET[f'{k} first>95th']:
            if v == 'never': never += 1; continue
            mt = _re.search(r'SPY (-?[0-9.]+)%', str(v)); d = float(mt.group(1)) if mt else 0.0; sess = int(_re.search(r'\+(\d+)s', str(v)).group(1))
            if sess == 0: at_peak += 1
            elif d > -5: before5 += 1
            else: after5 += 1
        summ[k] = {'at_peak_already_extreme': at_peak, 'crossed_before_5pct': before5, 'crossed_after_5pct': after5, 'never': never, 'events': int(len(ET))}
    FT['timing'] = summ
    FT['rules'] = jsafe(BIG['rules15'].to_dict(orient='records')); FT['events_scored'] = [f"{a.date()} to {b.date()} {d}%" for a, b, d in zip(BIG['events_scored'].peak, BIG['events_scored'].trough, BIG['events_scored']['depth%'])]
    FT['models'] = jsafe(BIG['models15'].fillna('').to_dict(orient='records')); FT['existing'] = jsafe(BIG['existing_vs_labels'].to_dict(orient='records'))
    FT['null'] = jsafe({k: v for k, v in BIG['null15'].items() if k != 'nulls'}) if 'null15' in BIG else None
    if os.path.exists('ONSET.pkl'):
        ON = pd.read_pickle('ONSET.pkl')['table']; pairs = []; rowd = lambda rw: {'ann': rnd(rw.ann), 'sharpe': rnd(rw.sharpe), 'maxDD': rnd(rw.maxDD), 'spells': int(rw.spells), 'false': int(rw['spells w/o any 10% decline']), 'taken': rnd(rw['mean taken'])}
        for x_ in (3, 5, 7, 10):
            for iname in ('IN calm (P_off<q.8 & P_vol<q.7)', 'IN new 42d high', 'IN calm or new 42d high'):
                c_ = ON[ON.rule == f'control: SPY {x_}% below 63d high | {iname}']; v_ = ON[ON.rule == f'vol alert q0.9/42s & SPY {x_}% below 63d high | {iname}']
                if len(c_) and len(v_): pairs.append({'trigger': f'{x_}% below 63d high', 'reentry': iname, 'control': rowd(c_.iloc[0]), 'vol': rowd(v_.iloc[0])})
        FT['onset'] = pairs; FT['onset_best'] = jsafe(ON.sort_values('ann', ascending=False).head(6)[['rule', 'ann', 'sharpe', 'maxDD', 'exposure', 'spells', 'spells w/o any 10% decline', 'mean taken', 'worst taken']].to_dict(orient='records'))
    out['fifteen'] = FT
    print('P(15% decline / 63 sessions) today', FT['dd15_63']['today'], f"(pct {FT['dd15_63']['trailing_pct']}, decile {FT['dd15_63']['today_decile']})", '| /126', FT['dd15_126']['today'], f"(pct {FT['dd15_126']['trailing_pct']})", '| timing', summ.get('P15/63 (new)'))
if os.path.exists('EARLY.pkl'):
    EA = pd.read_pickle('EARLY.pkl'); out['early'] = {'yearly': EA['yearly'], 'ties': EA['ties'], 'auc_early': EA['auc_early'], 'tables': {k: jsafe(v.to_dict(orient='records')) for k, v in EA['tables'].items()}}
if os.path.exists('PATH2020.pkl'): out['path2020'] = jsafe(pd.read_pickle('PATH2020.pkl')['table'].to_dict(orient='records'))
if os.path.exists('ADDBOOK.pkl'):
    AB = pd.read_pickle('ADDBOOK.pkl'); out['addbook'] = {'sleeve': jsafe(AB['sleeve'].to_dict(orient='records')), 'book': jsafe(AB['book'].to_dict(orient='records'))}
if os.path.exists('VVSIG.pkl'):
    VS = pd.read_pickle('VVSIG.pkl'); out['vvsig'] = {'today': jsafe(VS['today']), 'split': jsafe(VS['split'].to_dict(orient='records')), 'books': jsafe(VS['books'].to_dict(orient='records')), 'spells': VS['spells'][-12:], 'n_spells': len(VS['spells'])}
    out['instruction'] = VS['today']['instruction']
if os.path.exists('LHLL.pkl'):
    LH = pd.read_pickle('LHLL.pkl')
    out['roc'] = {'today': jsafe(LH['today']), 'deciles': {k: jsafe(v.reset_index(drop=True).to_dict(orient='records')) for k, v in LH['deciles'].items()}, 'books': jsafe(LH['books'].to_dict(orient='records')), 'spells': LH['spells'][-12:], 'n_spells': len(LH['spells']), 'last': {k: [list(v[0]), list(v[1])] for k, v in LH['last'].items()}}
    out['instruction'] = LH['today']['instruction']
json.dump(out, open('v2_data.json', 'w')); print('v2_data.json', os.path.getsize('v2_data.json') // 1024, 'KB')
for p in ('P_off', 'P_on', 'P_vol'):
    print(f'\n{p} drivers today (push > 0 raises the probability):'); print(pd.DataFrame(PR[p]['drivers']).head(10).to_string(index=False))
    print(f'{p} decile table (OOS):'); print(pd.DataFrame(PR[p]['deciles']).round(3).to_string())
