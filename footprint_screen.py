"""Footprint screen: three buckets, one run.

    python footprint_screen.py              # today
    python footprint_screen.py 2025-05-01   # as of a past date

Universe: US-listed, mcap >= $5B (yfinance screener), 1y return < +100%, 20d $vol > $20M.

Bucket 1  OCTANE TURN   (OKTA, CRWD, TXG, NEM, NVDA-2019, VLO, HII, VIK)
          Rank by composite B = +RVrel250 +beta60 -dsince+3 +clv20 +stoch20 +px/AVWAPlow -px/2yH -RS12 -RV10/RV60
          Validated 2024-26 (22 dates) and 2018-20 (20 dates): top decile ~2.5x base rate of +50% in 250d.
          Downside tail is regime-dependent: run it 3-6 weeks after an index correction low.
Bucket 2  STEADY LEADER (TD, CSX, GWRE-2024, DBA)
          Rank by STEADY = -RV250 -RV60 -beta250 -big_up_days -big_dn_days -skew +mhit12 +UVDV250 +pctup250 +px/52wH
          Validated 2024-25 (21 dates): top decile 2.4x base rate of +30% with path drawdown < 15%; median path DD -17% vs -28%.
          Did NOT replicate on hit rate in 2020-22 (drawdown benefit held).
Bucket 3  LOW-VOL BASE  (DBA template, no backtested edge) -> watchlist only.
TURN flag (all buckets): stoch20 > 0.8 and px/AVWAPlow > 1.0 and UVDV60 > 1.0. Every case flipped this ~1 month after its low.
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
    s=s.dropna()
    if date: s=s[:date]
    if len(s)<260: return None
    c,h,l,v=s.Close,s.High,s.Low,s.Volume; r=c.pct_change(); L=len(s); scl=sc.reindex(c.index).ffill(); sr=scl.pct_change(); m={}
    m['ret_12m']=c.iloc[-1]/c.iloc[-253]-1; m['RS12']=(1+m['ret_12m'])/(scl.iloc[-1]/scl.iloc[-253])-1; m['ret_3m']=c.iloc[-1]/c.iloc[-64]-1; m['ret_1m']=c.iloc[-1]/c.iloc[-22]-1
    m['px/52wH']=c.iloc[-1]/c.iloc[-252:].max(); m['px/2yH']=c.iloc[-1]/c.iloc[-504:].max()
    rv=lambda n: r.rolling(n).std().iloc[-1]; m['RV10/RV60']=rv(10)/rv(60); m['RVrel250']=rv(250)/sr.rolling(250).std().iloc[-1]; m['RV250']=rv(250)*np.sqrt(252); m['RV60']=rv(60)*np.sqrt(252)
    lo_i=c.iloc[-252:].idxmin(); seg=s[lo_i:]; tp=(seg.High+seg.Low+seg.Close)/3; m['px/AVWAPlow']=c.iloc[-1]/((tp*seg.Volume).sum()/seg.Volume.sum())
    m['stoch20']=(c.iloc[-1]-l.iloc[-20:].min())/(h.iloc[-20:].max()-l.iloc[-20:].min()+1e-9); m['clv20']=(((c-l)-(h-c))/(h-l)).fillna(0).iloc[-20:].mean()
    i3=np.where(r.values>0.03)[0]; m['dsince+3']=L-1-i3[-1] if len(i3) else 999
    r60=r.iloc[-60:]; v60=v.iloc[-60:]; m['UVDV60']=v60[r60>0].sum()/max(v60[r60<0].sum(),1); r250=r.iloc[-250:]; v250=v.iloc[-250:]; m['UVDV250']=v250[r250>0].sum()/max(v250[r250<0].sum(),1)
    m['n+3/n-3_60']=((r60>0.03).sum()+1)/((r60<-0.03).sum()+1)
    j=pd.concat([r,sr],axis=1).dropna(); j60=j.iloc[-60:]; m['beta60']=j60.cov().iloc[0,1]/j60.iloc[:,1].var(); m['corr60']=j60.corr().iloc[0,1]; j250=j.iloc[-250:]; m['beta250']=j250.cov().iloc[0,1]/j250.iloc[:,1].var()
    m['pctup250']=(r250>0).mean(); mc=c.resample('ME').last().pct_change().dropna().iloc[-12:]; m['mhit12']=(mc>0).mean(); m['skew250']=r250.skew(); m['big_up_days250']=int((r250>0.05).sum()); m['big_dn_days250']=int((r250<-0.05).sum())
    lc=np.log(c.values)
    for n in [60,120,250]:
        y=lc[-n:]; x=np.arange(n); b_,a_=np.polyfit(x,y,1); m[f'slope_{n}']=b_*252; m[f'resid_{n}']=np.std(y-(a_+b_*x)); cn=c.iloc[-n:]; ddn=cn/cn.cummax()-1; m[f'maxDD_{n}']=ddn.min(); m[f'within5_{n}']=(ddn>-0.05).mean()
    wk=s.iloc[-130:].resample('W-FRI').agg({'Low':'min'}); m['wkHL_26']=(wk.Low.iloc[-26:].diff()>0).mean(); dd250=c.iloc[-250:]/c.iloc[-250:].cummax()-1; m['maxDD_days_ago']=L-1-c.index.get_loc(dd250.idxmin())
    m['days_since_60dlow']=L-1-c.index.get_loc(c.iloc[-60:].idxmin()); m['dd_from_20dH']=c.iloc[-1]/h.iloc[-20:].max()-1
    cp=((c-l)/(h-l)).fillna(0.5).iloc[-60:]; m['top30-bot30_60']=(cp>0.7).mean()-(cp<0.3).mean(); va=v.rolling(20).mean(); m['dryup120']=va.iloc[-120:].min()/v.iloc[-250:].mean()
    o=s.Open; on=(o/c.shift(1)-1).iloc[-120:]; idd=(c/o-1).iloc[-120:]; m['ID-ON_120']=(np.prod(1+idd)-1)-(np.prod(1+on)-1)
    m['vol20/vol250']=v.iloc[-20:].mean()/v.iloc[-250:].mean(); m['corr60_chg']=m['corr60']-(pd.concat([r,sr],axis=1).dropna().iloc[-120:-60].corr().iloc[0,1])
    hi252=c.rolling(252).max().shift(1); brk=(c>hi252).iloc[-15:]; m['52wH_break_15d']=bool(brk.any()); m['days_since_52wL']=int(L-1-np.argmin(c.iloc[-252:].values)-(L-252))
    m['dollarvol20M']=(c*v).iloc[-20:].mean()/1e6; return m
T=pd.DataFrame({t:feat(D[t]) for t in syms if t!='SPY' and t in D.columns.get_level_values(0)}).T.dropna(); T=T[T.dollarvol20M>20]
z=lambda x: x.astype(float).rank(pct=True)-0.5
T['B']=(z(T.RVrel250)+z(T.beta60)-z(T['dsince+3'])+z(T.clv20)+z(T.stoch20)+z(T['px/AVWAPlow'])-z(T['px/2yH'])-z(T.RS12)-z(T['RV10/RV60'])).rank(pct=True)*100
T['STEADY']=(-z(T.RV250)-z(T.RV60)-z(T.beta250)-z(T.big_up_days250)-z(T.big_dn_days250)-z(T.skew250)+z(T.mhit12)+z(T.UVDV250)+z(T.pctup250)+z(T['px/52wH'])).rank(pct=True)*100
T['PATH']=(z(T.maxDD_60)+z(T.maxDD_120)+z(T.maxDD_250)+z(T.within5_120)+z(T.within5_250)-z(T.resid_120)-z(T.resid_250)-z(T.RV250)-z(T.big_dn_days250)+z(T['px/52wH'])+z(T.wkHL_26)+z(T.maxDD_days_ago)).rank(pct=True)*100
T['GATE']=(T.slope_60>0.03)&(T.slope_120>0)&(T.ret_12m>=0.15)&(T['px/52wH']>=0.85)&((T.maxDD_250>-0.15)|(T.maxDD_days_ago>100))&(T.stoch20>0.5)&(T.days_since_60dlow>=5)&(T.dd_from_20dH>-0.06)
T['SMOOTH']=(-z(T.RVrel250)-z(T.RV250)-z(T.RV60)-z(T.big_up_days250)-z(T.beta250)-z(T.big_dn_days250)-z(T.beta60)-z(T.skew250)+z(T['top30-bot30_60'])+z(T.dryup120)+z(T.UVDV250)+z(T.mhit12)+z(T.pctup250)+z(T['px/52wH'])+z(T['px/2yH'])+z(T['dsince+3'])).rank(pct=True)*100
T['NEARTERM']=(z(T['n+3/n-3_60'])+z(T.UVDV60)).rank(pct=True)*100
T['TURN']=(T.stoch20>0.8)&(T['px/AVWAPlow']>1.0)&(T.UVDV60>1.0)
T['LOWVOL_BASE']=(T.RVrel250<1.3)&(T.beta60.abs()<0.8)&(T['px/2yH']<0.92)
T['FLOW']=np.where(T['ID-ON_120']>0.05,'accum',np.where(T['ID-ON_120']<-0.05,'gap','mixed'))
T['LEADER_ENTRY']=(T.STEADY>=85)&(T['52wH_break_15d'])&(T.days_since_52wL<=40)
E=T[T.ret_12m<1.0].copy()
pd.set_option('display.width',250); pd.set_option('display.max_rows',80)
cols=['PATH','GATE','SMOOTH','STEADY','B','NEARTERM','TURN','FLOW','ret_12m','ret_3m','ret_1m','px/52wH','px/2yH','RVrel250','beta60','corr60','corr60_chg','ID-ON_120','vol20/vol250','dsince+3','stoch20','px/AVWAPlow','UVDV60','dollarvol20M']
print('\n=== PRIMARY: PATH >= 80 + trend gate (slope60>3%, slope120>0, 1y>=15%, within 15% of high, no drawdown >15% in last 100 sessions). Validated: smooth next-120d at 2x base, 25%+ drawdowns at 1/5 base ===')
print(E[(E.PATH>=80)&E.GATE].sort_values('PATH',ascending=False)[cols].round(2).to_string())
print('\n=== SMOOTH composite (reference) — top 30 by SMOOTH (validated: +15% in 120d with path DD<10% at 2.2x base; 25%+ drawdowns at 1/6 base) ===')
print(E.sort_values('SMOOTH',ascending=False).head(30)[cols].round(2).to_string())
print('\n=== SMOOTH>=80, pulled back 4-15% off the high, close turning up ===')
P=E[(E.SMOOTH>=80)&(E['px/52wH'].between(0.85,0.96))&(E.stoch20>0.6)&(E['px/AVWAPlow']>1.0)].sort_values('SMOOTH',ascending=False); print(P[cols].round(2).to_string() if len(P) else '  none')
print('\n=== BUCKET 1: OCTANE TURN — B top 30, early stage (3m<+25%, 1m<+15%, below 52wH), turned ===')
print(E[(E.ret_3m<0.25)&(E.ret_1m<0.15)&(E['px/52wH']<0.95)&E.TURN].sort_values('B',ascending=False).head(30)[cols].round(2).to_string())
print('\n=== LEADER ENTRY TRIGGER (steady>=85, 52wH break in last 15 sessions, within 40 sessions of the 52w low) ===')
print(E[E.LEADER_ENTRY].sort_values('STEADY',ascending=False)[cols].round(2).to_string() if E.LEADER_ENTRY.any() else '  none')
print('\n=== BUCKET 2: STEADY LEADER — STEADY top 30 ===')
print(E.sort_values('STEADY',ascending=False).head(30)[cols].round(2).to_string())
print('\n=== BUCKET 3: LOW-VOL BASE watchlist (no backtested edge) — turned first ===')
print(E[E.LOWVOL_BASE].sort_values(['TURN','stoch20'],ascending=False)[cols].round(2).to_string())
E.sort_values('B',ascending=False).round(4).to_csv(f'footprint_{date or "today"}.csv'); print('\nsaved', f'footprint_{date or "today"}.csv')
