import yfinance as yf, pandas as pd, time, sys
SIG = ['^VIX','^MOVE','^VXN','^VVIX','^TNX','^TYX','^GVZ','^OVX','^VIX1D','^VIX9D','^VIX3M','^VIX6M','^VIX1Y','^SKEW','^FVX','^IRX','^RVX']
TICK = ['QQQ','TLT','IEF','HYG','IWM','GLD','SPY','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','RWR','XAR','KBE','XBI','KCE','XHE','XHS','XHB','KIE','XME','XES','XOP','XPH','KRE','XRT','XSD','XSW','XTL','XTN','XLK','XLU','BNO','LQD','SHY']
OPTIONAL={'^VIX1Y','^RVX'}  # not served by Yahoo (VIX1Y returns a single row, RVX is refused); excluded downstream
out={}; got={}
for s in SIG+TICK:
    for attempt in range(3):
        try:
            d=yf.download(s,start='1990-01-01',progress=False,auto_adjust=True,threads=False)
            if d is None or len(d)==0: raise Exception('empty')
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            d.to_csv(f'data/{s.replace("^","")}.csv')
            print(f'{s:8s} {len(d):6d} {d.index.min().date()} -> {d.index.max().date()}  nonNaN close {d["Close"].notna().sum()}')
            got[s]=d['Close'].dropna().index.max().date(); break
        except Exception as e:
            print(s,'retry',attempt,e); time.sleep(2)

# --- guard: every required series must have come back, and all should end on the same session
missing=[s for s in SIG+TICK if s not in got and s not in OPTIONAL]
latest=max(got.values()); lag={s:str(d) for s,d in got.items() if d<latest and s not in OPTIONAL}
print(f'latest session {latest}; {len(got)} series fetched; lagging: {lag if lag else "none"}')
if missing:
    print('FETCH FAILED, missing after 3 attempts:',missing); sys.exit(1)
