import re, pandas as pd, yfinance as yf, time, json
txt = open('/root/.claude/uploads/92055679-7ca0-50d0-b0a3-0d1e31e44b3b/cg.txt').read().splitlines()
# buys: any row in all-positions with +$ active and sh 9/30 > 0
tick=[]
for ln in txt[242:]:
    m = re.match(r'^\s*\d+\s+.*?\b([A-Z][A-Z0-9.\-]{0,6})\s+[\d,]+\s+([\d,]+)\s+.*?\+\$([\d.,]+)M', ln)
    if m and int(m.group(2).replace(',',''))>0: tick.append(m.group(1))
tick = sorted(set(tick)); print(len(tick), 'buy tickers')
out={}
for i,t in enumerate(tick):
    rec={}
    for attempt in range(3):
        try:
            tk = yf.Ticker(t); inf = tk.info
            rec = {k: inf.get(k) for k in ['sector','industry','marketCap','sharesOutstanding','floatShares','shortPercentOfFloat','shortRatio','heldPercentInstitutions','trailingPE','forwardPE','earningsTimestamp','earningsTimestampStart']}
            try:
                cal = tk.calendar; ed = cal.get('Earnings Date') if isinstance(cal, dict) else None
                rec['earnings_date'] = str(ed[0]) if ed else None
            except Exception as e: rec['earnings_date']=None
            break
        except Exception as e:
            rec={'err':str(e)[:80]}; time.sleep(5*(attempt+1))
    out[t]=rec; time.sleep(0.4)
    if i%25==0: print(i, t, rec.get('sector'), rec.get('earnings_date'), flush=True)
pd.DataFrame(out).T.to_pickle('info.pkl'); print('done', sum('err' in v for v in out.values()), 'errors')
