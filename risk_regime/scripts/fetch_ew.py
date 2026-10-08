import yfinance as yf, pandas as pd, time
EW=['RSP','QQQE','RSPT','RSPS','RSPH','RSPF','RSPD','RSPG','RSPU','RSPM','RSPN','RSPR','RSPC','RSPE']
for s in EW:
    for a in range(3):
        try:
            d=yf.download(s,start='2000-01-01',progress=False,auto_adjust=True,threads=False)
            if d is None or len(d)==0: raise Exception('empty')
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            d.to_csv(f'data/{s}.csv'); print(f'{s:6s} {len(d):5d} {d.index.min().date()} -> {d.index.max().date()}'); break
        except Exception as e: print(s,'retry',a,e); time.sleep(2)
