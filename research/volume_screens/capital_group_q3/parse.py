import re, pandas as pd, yfinance as yf, time
txt = open('/root/.claude/uploads/92055679-7ca0-50d0-b0a3-0d1e31e44b3b/cg.txt').read().splitlines()
pat = re.compile(r'^\s*(\d+)\s+(.*?)\s*\b([A-Z][A-Z0-9.\-]{0,6})\s+([\d,]+)\s+([\d,]+)\s+([+\-]?[\d.,]+%|NEW|EXIT)\s+([+\-]?[\d.,]+%|NEW|EXIT)\s+([+\-])\$([\d.,]+)M\s+\$([\d.,]+)M\s+\$([\d.,]+)M\s+([\d.]+)%\s+([\d.]+)%\s+([+\-]?\d+)\s+(?:([\d.]+)%\s+)?(\d+)\s+(\d+)\s+(\d+)\s*$')
rows=[]
for ln in txt[242:]:
    m = pat.match(ln)
    if m:
        g = m.groups()
        rows.append(dict(rank=int(g[0]), name=g[1].strip(), ticker=g[2], sh0=int(g[3].replace(',','')), sh1=int(g[4].replace(',','')),
            dpct=g[5], active_pct=g[6], active_usd=float(g[8].replace(',',''))*(1 if g[7]=='+' else -1),
            mv0=float(g[9].replace(',','')), mv1=float(g[10].replace(',','')), wt0=float(g[11]), wt1=float(g[12]), dbps=int(g[13]),
            maxfd=float(g[14]) if g[14] else 0.0, own=int(g[15]), add=int(g[16]), cut=int(g[17])))
df = pd.DataFrame(rows).set_index('ticker')
print(len(df), 'rows parsed; ranks missing:', sorted(set(range(1,369))-set(df['rank'])))
df['active_pct_num'] = pd.to_numeric(df.active_pct.str.replace('%','').str.replace(',','').str.replace('+',''), errors='coerce')
df.loc[df.active_pct=='NEW','active_pct_num']=999; df.loc[df.active_pct=='EXIT','active_pct_num']=-100
df.to_pickle('cg.pkl')
buys = df[(df.active_usd>0)&(df.sh1>0)]
print('net active buys:', len(buys), ' sells:', (df.active_usd<0).sum())
tick = df.index.tolist()
d = yf.download(tick+['SPY'], period='7y', auto_adjust=True, progress=False, threads=True, group_by='column')
cl = d['Close']; bad = [t for t in tick if t not in cl.columns or cl[t].notna().sum()<300]
print('short/missing first pass:', bad)
# retry missing
retry = [t for t in bad if t in cl.columns and cl[t].iloc[-5:].isna().all() or t not in cl.columns]
for i in range(0,len(retry),25):
    ch = retry[i:i+25]
    try:
        x = yf.download(ch, period='7y', auto_adjust=True, progress=False, threads=False, group_by='column')
        for col in ['Open','High','Low','Close','Volume']:
            for t in ch:
                if t in x[col].columns and x[col][t].notna().sum()>0: d[(col,t)] = x[col][t].reindex(d.index)
    except Exception as e: print('err',e)
    time.sleep(2)
d.to_pickle('px.pkl')
cl = d['Close']; print('still short/missing:', [t for t in tick if t not in cl.columns or cl[t].notna().sum()<300])
meta = pd.DataFrame({'Market Cap': (df.mv1/ (df.wt1/100)).where(df.wt1>0)}); meta.index.name='Ticker'
