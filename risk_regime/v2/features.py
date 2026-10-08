#!/usr/bin/env python3
"""v2 Stage 1: feature store. Run from risk_regime/work (needs data/*.csv from fetch.py).

Every feature is a rate of change, a ratio, or a trailing percentile rank. No moving averages of
price, no z-scores: normalization is the trailing-window percentile rank of the raw quantity
against its own history (distribution-free). Realized vol uses close-to-close, Parkinson and
Garman-Klass estimators from OHLC. Volume features include the volatility of volume.
Outputs: F.pkl (float32 DataFrame on SPY sessions), OHLCV.pkl (raw panels), manifest.json.
"""
import pandas as pd, numpy as np, json, os, warnings, time; warnings.filterwarnings('ignore')
t0 = time.time()
D = 'data'
IDX = ['VIX','MOVE','VXN','VVIX','TNX','TYX','GVZ','OVX','VIX1D','VIX9D','VIX3M','VIX6M','SKEW','FVX','IRX']
TICK = ['QQQ','TLT','IEF','HYG','IWM','GLD','SPY','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','RWR','XAR','KBE','XBI','KCE','XHE','XHS','XHB','KIE','XME','XES','XOP','XPH','KRE','XRT','XSD','XSW','XTL','XTN','XLK','XLU','BNO']
EXTRA = ['LQD','SHY']
RANK_W = 504          # trailing window for percentile ranks (two years)
def load(s): return pd.read_csv(f'{D}/{s}.csv', index_col=0, parse_dates=True)
spy = load('SPY'); idx = spy.index[spy['Close'].notna()]
def panel(syms, fill=False):
    out = {}
    for s in syms:
        d = load(s).reindex(idx)
        if fill: d = d.ffill(limit=3)
        out[s] = d
    return out
I = panel(IDX, fill=True); T = panel(TICK + EXTRA)
pd.to_pickle({'I': I, 'T': T, 'idx': idx}, 'OHLCV.pkl')
F = {}; FAM = {}
def add(name, s, fam):
    F[name] = s.astype('float32'); FAM.setdefault(fam, []).append(name)
def rrank(x, w=RANK_W): return x.rolling(w, min_periods=int(w*0.6)).rank(pct=True)
def roc(x, h): return np.log(x).diff(h)
ann = np.sqrt(252) * 100   # realized vol in percent, same units as the vol indices

# ---------------- implied / rate indices
for s in IDX:
    c = I[s]['Close']; israte = s in ('TNX','TYX','FVX','IRX')
    add(f'{s}:lvl_rank252', c.rolling(252, min_periods=150).rank(pct=True), 'idx_level')
    add(f'{s}:lvl_rank504', rrank(c), 'idx_level')
    add(f'{s}:lvl_rank1260', c.rolling(1260, min_periods=600).rank(pct=True), 'idx_level')
    for h in (1, 2, 3, 5, 10, 21, 42, 63):
        ch = c.diff(h) if israte else roc(c, h)
        add(f'{s}:roc{h}', ch, 'idx_roc')
        if h in (1, 3, 5, 10, 21, 63): add(f'{s}:roc{h}_rank', rrank(ch), 'idx_roc_rank')
    acc = (c.diff(5) if israte else roc(c, 5)); add(f'{s}:accel5_rank', rrank(acc - acc.shift(5)), 'idx_roc_rank')
    if {'High','Low'} <= set(I[s].columns) and I[s]['High'].notna().mean() > 0.5 and not israte:
        rng = np.log(I[s]['High'] / I[s]['Low']).replace([np.inf, -np.inf], np.nan)
        add(f'{s}:range_rank', rrank(rng), 'idx_range'); add(f'{s}:range5_rank', rrank(rng.rolling(5).median()), 'idx_range')
# term structure and cross-asset ratios
RAT = {'VIX/VIX3M': ('VIX','VIX3M'), 'VIX9D/VIX': ('VIX9D','VIX'), 'VIX1D/VIX9D': ('VIX1D','VIX9D'), 'VIX3M/VIX6M': ('VIX3M','VIX6M'), 'VIX/VIX6M': ('VIX','VIX6M'),
       'VVIX/VIX': ('VVIX','VIX'), 'VVIX/VIX3M': ('VVIX','VIX3M'), 'MOVE/VIX': ('MOVE','VIX'), 'VXN/VIX': ('VXN','VIX'), 'GVZ/VIX': ('GVZ','VIX'), 'OVX/VIX': ('OVX','VIX'), 'SKEW/VIX': ('SKEW','VIX')}
for k, (a, b) in RAT.items():
    r = np.log(I[a]['Close'] / I[b]['Close'])
    add(f'{k}:lvl_rank', rrank(r), 'ratio_level')
    for h in (1, 3, 5, 10, 21):
        add(f'{k}:roc{h}', r.diff(h), 'ratio_roc'); add(f'{k}:roc{h}_rank', rrank(r.diff(h)), 'ratio_roc_rank')
# curve
for k, (a, b) in {'TYX-TNX': ('TYX','TNX'), 'TNX-FVX': ('TNX','FVX'), 'TNX-IRX': ('TNX','IRX')}.items():
    sp = I[a]['Close'] - I[b]['Close']; add(f'{k}:lvl_rank', rrank(sp), 'curve')
    for h in (5, 21): add(f'{k}:chg{h}_rank', rrank(sp.diff(h)), 'curve')

# ---------------- ETFs: returns, realized vol (3 estimators), volume, range
RET = {}; RV21 = {}; RV10 = {}; VOLR = {}; CMP = {}; VVOL = {}
spyc = T['SPY']['Close']; spyr = np.log(spyc).diff()
for s in TICK + EXTRA:
    d = T[s]; c, h_, l_, o_, v = d['Close'], d['High'], d['Low'], d['Open'], d['Volume']
    r = np.log(c).diff(); RET[s] = r
    # price ROC and relative ROC
    for h in (1, 3, 5, 10, 21, 63):
        add(f'{s}:roc{h}', roc(c, h), 'etf_roc'); add(f'{s}:roc{h}_rank', rrank(roc(c, h)), 'etf_roc_rank')
        if s != 'SPY':
            rel = roc(c, h) - roc(spyc, h); add(f'{s}:rel{h}', rel, 'etf_rel'); add(f'{s}:rel{h}_rank', rrank(rel), 'etf_rel_rank')
    add(f'{s}:dd63', np.log(c / c.rolling(63).max()), 'etf_dd'); add(f'{s}:dd252', np.log(c / c.rolling(252).max()), 'etf_dd')
    # realized vol: close-to-close, Parkinson, Garman-Klass
    lhl2 = np.log(h_ / l_) ** 2; lco2 = np.log(c / o_) ** 2
    cc = {w: r.rolling(w).std() * ann for w in (5, 10, 21, 63, 126)}
    pk = {w: np.sqrt(lhl2.rolling(w).mean() / (4 * np.log(2))) * ann for w in (5, 10, 21)}
    gk = {w: np.sqrt((0.5 * lhl2 - (2 * np.log(2) - 1) * lco2).rolling(w).mean().clip(lower=0)) * ann for w in (5, 10, 21)}
    for w, x in cc.items(): add(f'{s}:cc{w}', x, 'etf_rv'); add(f'{s}:cc{w}_rank', rrank(x), 'etf_rv_rank')
    for w, x in pk.items(): add(f'{s}:pk{w}', x, 'etf_rv'); add(f'{s}:pk{w}_rank', rrank(x), 'etf_rv_rank')
    for w, x in gk.items(): add(f'{s}:gk{w}', x, 'etf_rv')
    RV21[s] = cc[21]; RV10[s] = cc[10]
    for nm, x in {'cc5_21': cc[5] / cc[21], 'cc10_21': cc[10] / cc[21], 'cc21_63': cc[21] / cc[63], 'cc21_126': cc[21] / cc[126], 'pk5_21': pk[5] / pk[21], 'gk10_21': gk[10] / gk[21], 'pk21_cc21': pk[21] / cc[21]}.items():
        add(f'{s}:{nm}', x, 'etf_rv_ratio'); add(f'{s}:{nm}_rank', rrank(x), 'etf_rv_ratio_rank')
    for w in (5, 21):
        x = np.log(cc[21]).diff(w); add(f'{s}:cc21_roc{w}', x, 'etf_rv_roc'); add(f'{s}:cc21_roc{w}_rank', rrank(x), 'etf_rv_roc_rank')
    x = np.log(cc[10]).diff(5); add(f'{s}:cc10_roc5_rank', rrank(x), 'etf_rv_roc_rank')
    vov = np.log(cc[10]).diff().rolling(21).std(); add(f'{s}:volofvol21_rank', rrank(vov), 'etf_vov'); add(f'{s}:volofvol21_roc5_rank', rrank(np.log(vov).diff(5)), 'etf_vov')
    # volume
    if v.notna().mean() > 0.5 and (v > 0).mean() > 0.5:
        lv = np.log(v.replace(0, np.nan))
        add(f'{s}:vol_rank63', v.rolling(63, min_periods=40).rank(pct=True), 'etf_volume'); add(f'{s}:vol_rank252', v.rolling(252, min_periods=150).rank(pct=True), 'etf_volume')
        v5 = v.rolling(5).sum(); v21 = v.rolling(21).sum()
        add(f'{s}:vroc5', np.log(v5 / v5.shift(5)), 'etf_volume'); add(f'{s}:vroc5_rank', rrank(np.log(v5 / v5.shift(5))), 'etf_volume')
        add(f'{s}:vroc21', np.log(v21 / v21.shift(21)), 'etf_volume'); add(f'{s}:vroc21_rank', rrank(np.log(v21 / v21.shift(21))), 'etf_volume')
        vv = lv.diff().rolling(21).std(); VVOL[s] = vv
        add(f'{s}:volvol21', vv, 'etf_volvol'); add(f'{s}:volvol21_rank', rrank(vv), 'etf_volvol'); add(f'{s}:volvol21_roc5_rank', rrank(np.log(vv).diff(5)), 'etf_volvol'); add(f'{s}:volvol21_roc21_rank', rrank(np.log(vv).diff(21)), 'etf_volvol')
        dv = v * c; ami = (r.abs() / dv * 1e9).rolling(21).mean(); add(f'{s}:amihud21_rank', rrank(ami), 'etf_liquidity'); add(f'{s}:amihud_roc5_rank', rrank(np.log(ami).diff(5)), 'etf_liquidity')
        sv = (np.sign(r) * v).rolling(5).sum() / v5; add(f'{s}:signedvol5', sv, 'etf_volume'); add(f'{s}:signedvol21', (np.sign(r) * v).rolling(21).sum() / v21, 'etf_volume')
        VOLR[s] = F[f'{s}:vol_rank63']
    # range / compression
    rng = np.log(h_ / l_); rng5 = rng.rolling(5).median(); rng63 = rng.rolling(63).median()
    add(f'{s}:range_rank', rrank(rng), 'etf_range'); add(f'{s}:compress5_63', rng5 / rng63, 'etf_range'); add(f'{s}:compress5_63_rank', rrank(rng5 / rng63), 'etf_range')
    CMP[s] = rng5 / rng63
    q20 = rng.rolling(63).quantile(0.2); add(f'{s}:nr5', (rng < q20).rolling(5).sum(), 'etf_range')
    clv = ((c - l_) / (h_ - l_)).replace([np.inf, -np.inf], np.nan); add(f'{s}:clv5', clv.rolling(5).mean(), 'etf_range')
    gap = np.log(o_ / c.shift(1)); add(f'{s}:gap5abs_rank', rrank(gap.abs().rolling(5).mean()), 'etf_range')
    # realized vol relative to SPY and ROC of that
    if s != 'SPY':
        rel_rv = np.log(cc[21] / (spyr.rolling(21).std() * ann)); add(f'{s}:relrv21', rel_rv, 'etf_relrv'); add(f'{s}:relrv21_roc5_rank', rrank(rel_rv.diff(5)), 'etf_relrv')

# ---------------- implied vs realized (VRP) on native pairs
PAIR = {'SPY': 'VIX', 'QQQ': 'VXN', 'IWM': 'VIX', 'HYG': 'VIX', 'TLT': 'MOVE', 'IEF': 'MOVE', 'GLD': 'GVZ', 'XOP': 'OVX', 'XLE': 'OVX', 'BNO': 'OVX'}
for s, iv in PAIR.items():
    ivs = I[iv]['Close'] / (100 if iv != 'MOVE' else 1)
    for nm, rv in {'cc21': F[f'{s}:cc21'].astype(float), 'pk21': F[f'{s}:pk21'].astype(float), 'cc5': F[f'{s}:cc5'].astype(float)}.items():
        ratio = np.log(I[iv]['Close'] / rv)   # implied (vol points) over realized (percent); MOVE is in bp so its ratio sits near 10
        add(f'{s}:{iv}/{nm}', ratio, 'vrp'); add(f'{s}:{iv}/{nm}_rank', rrank(ratio), 'vrp'); add(f'{s}:{iv}/{nm}_roc5_rank', rrank(ratio.diff(5)), 'vrp')
    if iv != 'MOVE':
        prem = I[iv]['Close'] - F[f'{s}:cc21'].astype(float)
        add(f'{s}:{iv}_prem_chg5_rank', rrank(prem.diff(5)), 'vrp'); add(f'{s}:{iv}_prem_chg21_rank', rrank(prem.diff(21)), 'vrp')

# ---------------- cross-sectional structure of the 38
Rm = pd.DataFrame({s: RET[s] for s in TICK}); Rm = Rm.loc[:, Rm.notna().mean() > 0.3]
for h in (1, 5, 21):
    rh = pd.DataFrame({s: roc(T[s]['Close'], h) for s in TICK}); disp = rh.std(axis=1)
    add(f'XS:disp{h}', disp, 'xs'); add(f'XS:disp{h}_rank', rrank(disp), 'xs'); add(f'XS:disp{h}_roc5_rank', rrank(np.log(disp).diff(5)), 'xs')
    add(f'XS:breadth_up{h}', (rh > 0).mean(axis=1), 'xs_breadth'); add(f'XS:breadth_beatspy{h}', (rh.sub(roc(spyc, h), axis=0) > 0).mean(axis=1), 'xs_breadth')
def avg_corr(w):
    X = Rm.values; n = len(X); out = np.full(n, np.nan)
    for i in range(w, n + 1):
        blk = X[i - w:i]; m = ~np.isnan(blk).any(axis=0)
        if m.sum() < 10: continue
        C = np.corrcoef(blk[:, m].T); k = m.sum(); out[i - 1] = (C.sum() - k) / (k * (k - 1))
    return pd.Series(out, index=Rm.index)
for w in (21, 63):
    ac = avg_corr(w); add(f'XS:avgcorr{w}', ac, 'xs'); add(f'XS:avgcorr{w}_rank', rrank(ac), 'xs'); add(f'XS:avgcorr{w}_chg5_rank', rrank(ac.diff(5)), 'xs')
rvb = pd.DataFrame({s: RV10[s] / RV21[s] for s in TICK}); add('XS:rv_breadth', (rvb > 1.2).mean(axis=1), 'xs_breadth'); add('XS:rv_breadth_rank', rrank((rvb > 1.2).mean(axis=1)), 'xs_breadth')
add('XS:rv21_median_rank', rrank(pd.DataFrame({s: RV21[s] for s in TICK}).median(axis=1)), 'xs')
if VOLR: vb = pd.DataFrame(VOLR); add('XS:vol_breadth', (vb > 0.8).mean(axis=1), 'xs_breadth'); add('XS:vol_breadth_rank', rrank((vb > 0.8).mean(axis=1)), 'xs_breadth')
if VVOL: vvb = pd.DataFrame(VVOL); add('XS:volvol_median_rank', rrank(vvb.median(axis=1)), 'xs'); add('XS:volvol_median_roc5_rank', rrank(np.log(vvb.median(axis=1)).diff(5)), 'xs')
cb = pd.DataFrame(CMP); add('XS:compress_breadth', (cb < 0.8).mean(axis=1), 'xs_breadth'); add('XS:compress_median_rank', rrank(cb.median(axis=1)), 'xs')
# equal weight vs cap weight (if the files exist from fetch_ew)
if os.path.exists('data_ew/RSP.csv'):
    for ew, cw in (('RSP','SPY'), ('QQQE','QQQ')):
        e = pd.read_csv(f'data_ew/{ew}.csv', index_col=0, parse_dates=True)['Close'].reindex(idx); rat = np.log(e / T[cw]['Close'])
        for h in (5, 21, 63): add(f'{ew}/{cw}:roc{h}_rank', rrank(rat.diff(h)), 'breadth_ew')

F = pd.DataFrame(F, index=idx)
F = F.loc[:, F.notna().mean() > 0.05]
F.to_pickle('F.pkl')
man = {'rows': len(F), 'cols': F.shape[1], 'first': str(idx[0].date()), 'last': str(idx[-1].date()), 'rank_window': RANK_W, 'families': {k: len([c for c in v if c in F.columns]) for k, v in FAM.items()}, 'feature_names': list(F.columns)}
json.dump(man, open('manifest.json', 'w'))
print(f'F {F.shape} float32 {F.memory_usage().sum()/1e6:.0f} MB, {time.time()-t0:.0f}s'); print({k: v for k, v in man['families'].items()})
nan = F.iloc[-1].isna(); print('NaN on last row:', int(nan.sum()), 'e.g.', list(F.columns[nan])[:12])
