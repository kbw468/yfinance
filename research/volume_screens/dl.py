import pandas as pd, yfinance as yf, csv
rows = list(csv.DictReader(open('/root/.claude/uploads/92055679-7ca0-50d0-b0a3-0d1e31e44b3b/cda84299-finviz-5.csv')))
tick = [r['Ticker'] for r in rows]
meta = pd.DataFrame(rows).set_index('Ticker')
meta.to_pickle('meta.pkl')
d = yf.download(tick+['SPY'], period='7y', auto_adjust=True, progress=False, threads=True, group_by='column')
print(d.shape, d.index[0], d.index[-1])
d.to_pickle('px.pkl')
cl = d['Close']
bad = cl.columns[cl.notna().sum() < 300].tolist()
print('short/missing:', bad)
print('last row NA count:', cl.iloc[-1].isna().sum())
