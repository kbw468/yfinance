"""Forward score 1-10: walk-forward-validated model of forward 63d (primary) and 21d risk-adjusted return,
built only from price, volume and volatility. Data: Nasdaq API. Usage: python fwd_rate.py TICKER [TICKER ...]"""
import sys, os, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwdfeat import features
from rate import history, _get, _num
M = pickle.load(open(os.path.join(HERE,'fwd_model.pkl'),'rb'))
FEATS, REF = M['feats'], M['ref']
LG = {h: lgb.Booster(model_file=os.path.join(HERE, f'lgbm{h}.txt')) for h in (21,63)}
REFSORT = {f: np.sort(REF[f].dropna().values) for f in FEATS}
NASDAQ_TO_FINVIZ = {'Technology':'Technology','Health Care':'Healthcare','Finance':'Financial','Industrials':'Industrials',
  'Consumer Discretionary':'Consumer Cyclical','Consumer Staples':'Consumer Defensive','Energy':'Energy','Basic Materials':'Basic Materials',
  'Utilities':'Utilities','Real Estate':'Real Estate','Telecommunications':'Communication Services'}
def _universe_sectors():
    out = []
    for f in ('universe_finviz.csv','universe_finviz_smallmid.csv'):
        try: out.append(pd.read_csv(os.path.join(os.path.dirname(HERE), f), index_col='Ticker')['Sector'])
        except Exception: pass
    s = pd.concat(out); return s[~s.index.duplicated()]
USEC = _universe_sectors()
def _ref_scores():
    z = np.nan_to_num(REF[FEATS].rank(pct=True).sub(0.5).values, nan=0.0)
    b63 = pd.Series(z@M['ridge63']).rank(pct=True).values + pd.Series(LG[63].predict(z)).rank(pct=True).values
    return pd.DataFrame({'b63': b63, 'r21': z@M['ridge21'], 'sector': USEC.reindex(REF.index).fillna('Other').values}, index=REF.index)
REFSC = _ref_scores()
def sector_of(t):
    if t in USEC.index: return USEC[t]
    from rate import sector as nasdaq_sector
    return NASDAQ_TO_FINVIZ.get(nasdaq_sector(t) or '', 'Other')
def pct_in(sorted_arr, x):
    if x is None or np.isnan(x) or len(sorted_arr)==0: return np.nan
    lo = np.searchsorted(sorted_arr, x, 'left'); hi = np.searchsorted(sorted_arr, x, 'right')
    return (lo + 0.5*(hi-lo) + 0.5)/(len(sorted_arr)+1)
def shares_out(t, last_close):
    d = _get(f'https://api.nasdaq.com/api/quote/{t}/summary?assetclass=stocks')
    try: return _num(d['summaryData']['MarketCap']['value'])/last_close
    except Exception: return np.nan
def score_vector(x, sec=None):
    """x: Series of raw features -> dict of scores, ranked within sector when sec is given."""
    z = np.array([[pct_in(REFSORT[f], x.get(f, np.nan)) - 0.5 for f in FEATS]]); z = np.nan_to_num(z, nan=0.0)
    out = {}
    for h in (21,63):
        pr = float((z@M[f'ridge{h}'])[0]); pl = float(LG[h].predict(z)[0])
        b = pct_in(M[f'ref_pred_ridge{h}'], pr) + pct_in(M[f'ref_pred_lgbm{h}'], pl)
        peers = REFSC[REFSC.sector==sec] if sec is not None and (REFSC.sector==sec).sum() >= 20 else REFSC
        p = pct_in(np.sort(peers['b63'].values), b) if h==63 else pct_in(np.sort(peers['r21'].values), pr)
        out[h] = round(1 + 9*p, 1)
    return out
def rate(T, asof=None):
    frames = {t: history(t, years=4) for t in (T,'SPY')}
    d = pd.concat(frames, axis=1).swaplevel(0,1,axis=1).sort_index(axis=1).dropna(subset=[('Close','SPY')])
    if asof is not None: d = d.loc[:asof]
    elif d['Volume'][T].iloc[-1] < 0.6*d['Volume'][T].iloc[-22:-1].mean(): d = d.iloc[:-1]
    O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
    f = features(O,H,L,C,V, {T: shares_out(T, C[T].iloc[-1])}).loc[T]
    sec = sector_of(T)
    s = score_vector(f, sec)
    flag = '' if (C[T].iloc[-1]>=3 and np.exp(f['e_logadv'])>=1e6) else ' | below price/liquidity floor used in training'
    if f['rv21'] < 0.08: flag += ' | possible deal pin: 21d vol under 8%'
    return C.index[-1].date(), s, flag + f' | sector {sec}' 
if __name__=='__main__':
    for t in sys.argv[1:]:
        asof, s, flag = rate(t.upper())
        print(f'{t.upper()} asof {asof} forward63 {s[63]} forward21 {s[21]} ref {M["ref_date"]}{flag}')
