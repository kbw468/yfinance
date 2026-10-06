# 1-10 volume/volatility backdrop rating. 10 components, each 0-1, linear between bad and good anchors.
import pandas as pd, numpy as np, yfinance as yf, warnings, sys, time
warnings.filterwarnings('ignore')
SECTOR_ETF = {'Industrials':'XLI','Technology':'XLK','Healthcare':'XLV','Financial Services':'XLF','Consumer Cyclical':'XLY',
 'Consumer Defensive':'XLP','Energy':'XLE','Basic Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE','Communication Services':'XLC'}
def lin(x, bad, good):
    if x is None or np.isnan(x): return 0.5
    return float(np.clip((x-bad)/(good-bad), 0, 1))
def rate(T, show=False):
    sec = None
    for i in range(4):
        try:
            sec = yf.Ticker(T).info.get('sector')
            if sec: break
        except Exception: pass
        time.sleep(3*(i+1))
    if not sec:  # fall back to the finviz universe file
        try:
            import os; u = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'universe_finviz.csv'), index_col='Ticker')
            sec = {'Financial':'Financial Services'}.get(u.loc[T,'Sector'], u.loc[T,'Sector'])
        except Exception: sec = None
    etf = SECTOR_ETF.get(sec, 'SPY')
    d = yf.download([T,'SPY',etf], period='3y', auto_adjust=True, progress=False, group_by='column')
    # drop today's partial bar if volume is <60% of 21d avg pace
    if d['Volume'][T].iloc[-1] < 0.6*d['Volume'][T].iloc[-22:-1].mean(): d = d.iloc[:-1]
    O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
    def f(t):
        o,h,l,c,v = O[t],H[t],L[t],C[t],V[t].replace(0,np.nan); r=np.log(c/c.shift(1)); W=21
        rr=r.iloc[-W:]; vv=v.iloc[-W:]; x={}
        x['updown']=(vv[rr>0].sum()-vv[rr<0].sum())/vv.sum()
        x['breakout_vol']=vv[(c>h.shift(1).rolling(W).max()).iloc[-W:]].sum()/vv.sum()
        rv=lambda w: r.iloc[-w:].std()
        x['rv21_63']=rv(21)/rv(63); s21=r.rolling(21).std(); x['rv_pct']=(s21.iloc[-252:]<=s21.iloc[-1]).mean()
        x['cc_park']=rv(21)*np.sqrt(252)/(np.sqrt((np.log(h/l)**2).iloc[-W:].mean()/(4*np.log(2)))*np.sqrt(252))
        x['clv']=((c-l)/(h-l)).iloc[-W:].mean()
        x['maxdown_var']=(rr.clip(upper=0)**2).max()/(rr**2).sum()
        dn=rr<0; idx=rr.sort_values().index[:3]; x['bigdown_conc']=vv[idx].sum()/vv[dn].sum() if dn.sum() else 0
        q=rr.abs()<0.005; x['quiet_vol']=vv[q].mean()/v.iloc[-252:].mean() if q.sum()>=2 else np.nan
        x['dist_hi']=c.iloc[-1]/h.iloc[-252:].max()-1
        x['ret63']=np.log(c.iloc[-1]/c.iloc[-64]); x['ret21']=np.log(c.iloc[-1]/c.iloc[-22])
        x['absret_vol_corr']=rr.abs().corr(vv); x['vov']=(r.rolling(5).std().iloc[-W:].std())/rv(21)
        x['lo_rising']= float(l.iloc[-10:].min() > l.iloc[-21:-10].min() > l.iloc[-42:-21].min())
        return x
    a, s, e = f(T), f('SPY'), f(etf)
    comp = {
     'accumulation vs SPY':     lin(a['updown']-s['updown'], -0.15, 0.35),
     'breakout-day volume':     lin(a['breakout_vol'], 0.0, 0.30),
     'vol compression':         0.5*lin(a['rv21_63'], 1.25, 0.70) + 0.5*lin(a['rv_pct'], 0.85, 0.10),
     'range vs gap vol':        lin(a['cc_park'], 1.50, 0.85),
     'close location':          lin(a['clv'], 0.40, 0.62),
     'no forced seller':        0.5*lin(a['maxdown_var'], 0.55, 0.15) + 0.5*lin(a['bigdown_conc'], 0.60, 0.25),
     'quiet-day volume':        lin(a['quiet_vol'], 0.60, 1.30),
     'proximity to 52w high':   lin(a['dist_hi'], -0.30, -0.02),
     'relative strength 63d':   0.5*lin(a['ret63']-s['ret63'], -0.10, 0.15) + 0.5*lin(a['ret63']-e['ret63'], -0.10, 0.15),
     'participation / structure': 0.4*lin(a['absret_vol_corr'], -0.1, 0.5) + 0.3*lin(a['vov'], 0.6, 0.25) + 0.3*a['lo_rising'],
    }
    score = round(sum(comp.values()), 1)
    if show:
        print(T, sec, etf, 'asof', C.index[-1].date(), 'score', score)
        for k,v in comp.items(): print(f'  {k:28s} {v:.2f}')
        print({k: round(v,3) for k,v in a.items()})
    return score, comp, a
if __name__=='__main__':
    for t in sys.argv[1:]: rate(t, show=True)
