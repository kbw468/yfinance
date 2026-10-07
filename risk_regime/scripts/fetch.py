import yfinance as yf, pandas as pd, time
SIG = ['^VIX','^MOVE','^VXN','^VVIX','^TNX','^TYX','^GVZ','^OVX','^VIX1D','^VIX9D','^VIX3M','^VIX6M','^VIX1Y','^SKEW','^FVX','^IRX','^RVX']
TICK = ['QQQ','TLT','IEF','HYG','IWM','GLD','SPY','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','RWR','XAR','KBE','XBI','KCE','XHE','XHS','XHB','KIE','XME','XES','XOP','XPH','KRE','XRT','XSD','XSW','XTL','XTN','XLK','XLU','BNO','LQD','SHY']
out={}
for s in SIG+TICK:
    for attempt in range(3):
        try:
            d=yf.download(s,start='1990-01-01',progress=False,auto_adjust=True,threads=False)
            if d is None or len(d)==0: raise Exception('empty')
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            d.to_csv(f'data/{s.replace("^","")}.csv')
            print(f'{s:8s} {len(d):6d} {d.index.min().date()} -> {d.index.max().date()}  nonNaN close {d["Close"].notna().sum()}')
            break
        except Exception as e:
            print(s,'retry',attempt,e); time.sleep(2)
