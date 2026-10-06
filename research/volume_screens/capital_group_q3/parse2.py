import re, pandas as pd, yfinance as yf, time
txt = open('/root/.claude/uploads/92055679-7ca0-50d0-b0a3-0d1e31e44b3b/cg.txt').read().splitlines()
df = pd.read_pickle('cg.pkl')
pat = re.compile(r'^\s*(\d+)\s+(.*?)\s*\b([A-Z][A-Z0-9.\-]{0,6})\s+0\s+([\d,]+)\s+\+\$([\d.,]+)M\s+\$0\.0M\s+\$([\d.,]+)M\s+([\d.]+)%\s+([\d.]+)%\s+([+\-]?\d+)\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+(\d+)\s*$')
rows=[]
for ln in txt[242:]:
    m = pat.match(ln)
    if m:
        g=m.groups()
        rows.append(dict(rank=int(g[0]), name=g[1].strip(), ticker=g[2], sh0=0, sh1=int(g[3].replace(',','')), dpct='NEW', active_pct='NEW',
            active_usd=float(g[4].replace(',','')), mv0=0.0, mv1=float(g[5].replace(',','')), wt0=0.0, wt1=float(g[7]), dbps=int(g[8]),
            maxfd=float(g[9]), own=int(g[10]), add=int(g[11]), cut=int(g[12]), active_pct_num=999))
new = pd.DataFrame(rows).set_index('ticker'); print('NEW rows:', len(new), new.index.tolist())
df = pd.concat([df, new]).sort_values('rank'); print('total', len(df), 'missing ranks', sorted(set(range(1,369))-set(df['rank'])))
df.to_pickle('cg.pkl')
d = pd.read_pickle('px.pkl')
need = [t for t in new.index if t not in d['Close'].columns]
x = yf.download(need, period='7y', auto_adjust=True, progress=False, threads=True, group_by='column')
for col in ['Open','High','Low','Close','Volume']:
    for t in need:
        if t in x[col].columns: d[(col,t)] = x[col][t].reindex(d.index)
d = d.sort_index(axis=1); d.to_pickle('px.pkl')
cl=d['Close']; print('still missing:', [t for t in df.index if t not in cl.columns or cl[t].notna().sum()<300])
