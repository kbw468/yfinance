"""Footprint screen. Usage: python footprint_screen.py [YYYY-MM-DD]
Universe: US-listed, mcap >= $5B (yfinance screener), 1y return < +100%, 20d $vol > $20M.
Ranks by two validated composites (percentile-of-rank sums):
  ENERGY = +RVrel250 +beta60 -dsince+3
  B      = ENERGY +clv20 +stoch20 +px/AVWAPlow -px/2yH -RS12 -RV10/RV60
"""
import sys, pandas as pd, numpy as np, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from yfinance import EquityQuery
date = sys.argv[1] if len(sys.argv)>1 else None
syms=[]
for lo,hi in [(1.26e10,1e15),(5e9,1.26e10)]:
    q=EquityQuery('and',[EquityQuery('btwn',['intradaymarketcap',lo,hi]),EquityQuery('eq',['region','us'])])
    for off in range(0,2000,250):
        r=yf.screen(q,offset=off,size=250,sortField='intradaymarketcap',sortAsc=False); qs=r.get('quotes',[])
        if not qs: break
        syms+=[x['symbol'] for x in qs if x.get('exchange') in ('NMS','NYQ','NGM','NCM','ASE','BTS') and '.' not in x['symbol']]
syms=sorted(set(syms)|{'SPY'})
D=yf.download(syms,period='3y',auto_adjust=True,group_by='ticker',threads=True,progress=False); sc=D['SPY'].Close
def feat(s):
    s=s.dropna(); 
    if date: s=s[:date]
    if len(s)<260: return None
    c,h,l,v=s.Close,s.High,s.Low,s.Volume; r=c.pct_change(); L=len(s); scl=sc.reindex(c.index).ffill(); m={}
    m['ret_12m']=c.iloc[-1]/c.iloc[-253]-1; m['RS12']=(1+m['ret_12m'])/(scl.iloc[-1]/scl.iloc[-253])-1
    m['px/52wH']=c.iloc[-1]/c.iloc[-252:].max(); m['px/2yH']=c.iloc[-1]/c.iloc[-504:].max()
    rv=lambda n: r.rolling(n).std().iloc[-1]; m['RV10/RV60']=rv(10)/rv(60); m['RVrel250']=rv(250)/scl.pct_change().rolling(250).std().iloc[-1]
    lo_i=c.iloc[-252:].idxmin(); seg=s[lo_i:]; tp=(seg.High+seg.Low+seg.Close)/3; m['px/AVWAPlow']=c.iloc[-1]/((tp*seg.Volume).sum()/seg.Volume.sum())
    m['stoch20']=(c.iloc[-1]-l.iloc[-20:].min())/(h.iloc[-20:].max()-l.iloc[-20:].min()+1e-9)
    m['clv20']=(((c-l)-(h-c))/(h-l)).fillna(0).iloc[-20:].mean()
    i3=np.where(r.values>0.03)[0]; m['dsince+3']=L-1-i3[-1] if len(i3) else 999
    j=pd.concat([r,scl.pct_change()],axis=1).dropna().iloc[-60:]; m['beta60']=j.cov().iloc[0,1]/j.iloc[:,1].var()
    m['dollarvol20M']=(c*v).iloc[-20:].mean()/1e6; return m
T=pd.DataFrame({t:feat(D[t]) for t in syms if t!='SPY' and t in D.columns.get_level_values(0)}).T.dropna()
T=T[(T.dollarvol20M>20)]
z=lambda x: x.astype(float).rank(pct=True)-0.5
T['ENERGY']=(z(T.RVrel250)+z(T.beta60)-z(T['dsince+3'])).rank(pct=True)*100
T['B']=(z(T.RVrel250)+z(T.beta60)-z(T['dsince+3'])+z(T.clv20)+z(T.stoch20)+z(T['px/AVWAPlow'])-z(T['px/2yH'])-z(T.RS12)-z(T['RV10/RV60'])).rank(pct=True)*100
T['AVG']=(T.ENERGY+T.B)/2
E=T[T.ret_12m<1.0].sort_values('AVG',ascending=False)
pd.set_option('display.width',250); print(E.head(40).round(2).to_string()); E.to_csv(f'footprint_{date or "today"}.csv')
