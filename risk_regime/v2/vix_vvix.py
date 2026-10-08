#!/usr/bin/env python3
"""The VIX/VVIX ratio: level and rate of change, rank-based throughout. Run from risk_regime/work.
VIX/VVIX rises when spot implied vol climbs faster than the vol of vol (a stress already being priced), and falls
when VVIX is bid while VIX stays low. Every series is a trailing 504-session percentile rank of the ratio level or
of its log rate of change. For each, at both tails (top 10% and bottom 10%): fires after 10 quiet sessions since
2006 and 2023, then the share followed by a 5% SPY drop within 21 sessions, a 10% drop within 63, a 5% rally within
21, and the mean next month; out-of-sample AUC of the rank itself by era; first fire in the 2024, 2025, 2026
drawdowns; today's rank. Then: does the ratio add anything to VIX's own rate of change? VIX 21-day ROC fires, split
by whether VIX/VVIX 21-day ROC is in its top 10% at the same time."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 330); pd.set_option('display.max_rows', 200)
from sklearn.metrics import roc_auc_score
O = pd.read_pickle('OHLCV.pkl'); F = pd.read_pickle('F.pkl'); L = pd.read_pickle('LAB.pkl'); idx = F.index; Y = idx.year.to_numpy()
vix = O['I']['VIX']['Close'].reindex(idx); vvix = O['I']['VVIX']['Close'].reindex(idx); spy = O['T']['SPY']['Close'].reindex(idx)
ratio = np.log(vix / vvix); rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True)
S = {'VIX/VVIX level': rk(ratio)}
for n in (1, 3, 5, 10, 21, 63): S[f'VIX/VVIX {n}d ROC'] = rk(ratio - ratio.shift(n))
S['reference: VIX 21d ROC'] = rk(np.log(vix / vix.shift(21))); S['reference: VIX 5d ROC'] = rk(np.log(vix / vix.shift(5))); S['reference: VIX level'] = rk(np.log(vix))
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok = L['ok21'].to_numpy(bool) & ~EXM; y = L['riskoff21'].to_numpy(float)
f21 = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); dn5 = (f21 / spy - 1 <= -0.05)
x21 = spy[::-1].rolling(21, min_periods=21).max()[::-1].shift(-1); up5 = (x21 / spy - 1 >= 0.05)
f63 = spy[::-1].rolling(63, min_periods=63).min()[::-1].shift(-1); dn10 = (f63 / spy - 1 <= -0.10); fwd = np.log(spy.shift(-21) / spy) * 100
EV = [('2024-07-16', '2024-08-05'), ('2025-02-19', '2025-04-08'), ('2026-01-27', '2026-03-30')]
def fires(on):
    on = on.fillna(False).to_numpy(); out = np.zeros(len(on), bool); q = 10
    for i in range(len(on)):
        if on[i] and q >= 10: out[i] = True
        q = 0 if on[i] else q + 1
    return pd.Series(out, index=idx)
rows = []
for name, s in S.items():
    for tail, on in (('top 10%', s >= 0.9), ('bottom 10%', s <= 0.1)):
        fr = fires(on); row = {'reading': name, 'tail': tail, 'today': round(float(s.iloc[-1]), 2)}
        for a0, lab in (('2006-01-01', '06+'), ('2023-01-01', '23+')):
            m = fr & (idx >= pd.Timestamp(a0)) & ~EXM & dn5.notna() & up5.notna()
            row[f'fires {lab}'] = int(m.sum())
            if m.sum(): row[f'-5% {lab}'] = f'{dn5[m].mean() * 100:.0f}%'; row[f'+5% {lab}'] = f'{up5[m].mean() * 100:.0f}%'
            if lab == '06+' and (m & f63.notna()).sum(): row['-10%/63 06+'] = f'{dn10[m & f63.notna()].mean() * 100:.0f}%'; row['next month 06+'] = f'{fwd[m].mean():+.1f}%'
        for a, b in EV:
            a = pd.Timestamp(a); b = pd.Timestamp(b); w0 = idx[max(0, idx.get_loc(a) - 10)]; h = on.fillna(False).loc[w0:b]; h = h[h]
            if len(h): t = h.index[0]; row[str(a.year)] = f'{idx.get_loc(t) - idx.get_loc(a):+d}s {spy.loc[t] / spy.loc[a:b].loc[:t].max() * 100 - 100 if t >= a else 0:.1f}%'
            else: row[str(a.year)] = 'never'
        rows.append(row)
R = pd.DataFrame(rows).fillna('')
m6 = (idx >= pd.Timestamp('2006-01-01')) & ~EXM & dn5.notna() & up5.notna(); m23 = (idx >= pd.Timestamp('2023-01-01')) & dn5.notna() & up5.notna()
print(f'base rates, any day: 5% drop within 21 sessions {dn5[m6].mean() * 100:.0f}% since 2006, {dn5[m23].mean() * 100:.0f}% since 2023; 5% rally {up5[m6].mean() * 100:.0f}% / {up5[m23].mean() * 100:.0f}%; 10% drop within 63 {dn10[m6 & f63.notna()].mean() * 100:.0f}%; next month {fwd[m6].mean():+.1f}%')
print(R.to_string(index=False))
print('\nout-of-sample AUC of the rank itself against a 5% drop within 21 sessions (above 0.5 = high rank, more risk):')
for name, s in S.items():
    v = s.to_numpy(float); out = {}
    for e, (a, b) in (('06-15', (2006, 2015)), ('16-26', (2016, 2026)), ('23-26', (2023, 2026))):
        m = ok & ~np.isnan(v) & (Y >= a) & (Y <= b); out[e] = round(float(roc_auc_score(y[m], v[m])), 2)
    print(f'  {name:26s} {out}')
print('\ndoes the ratio add to VIX 21d ROC? VIX 21d ROC in its top 10% (fires), split by VIX/VVIX 21d ROC at the same session:')
fv = fires(S['reference: VIX 21d ROC'] >= 0.9)
for a0 in ('2006-01-01', '2023-01-01'):
    for lab, c in (('ratio 21d ROC also top 10% (VIX outrunning VVIX)', S['VIX/VVIX 21d ROC'] >= 0.9), ('ratio 21d ROC below top 10% (VVIX keeping up)', S['VIX/VVIX 21d ROC'] < 0.9)):
        m = fv & c.fillna(False) & (idx >= pd.Timestamp(a0)) & ~EXM & dn5.notna()
        print(f'  from {a0[:4]}, {lab}: {int(m.sum())} fires | 5% drop {dn5[m].mean() * 100:.0f}% | 5% rally {up5[m].mean() * 100:.0f}%' if m.sum() else f'  from {a0[:4]}, {lab}: none')
print('\ntoday: VIX', round(float(vix.iloc[-1]), 2), 'VVIX', round(float(vvix.iloc[-1]), 2), 'VIX/VVIX', round(float((vix / vvix).iloc[-1]), 3), '| two-year range of VIX/VVIX', round(float((vix / vvix).iloc[-504:].min()), 3), 'to', round(float((vix / vvix).iloc[-504:].max()), 3))
pd.to_pickle(R, 'VIX_VVIX.pkl')
