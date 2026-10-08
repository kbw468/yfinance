import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); S=pd.read_pickle('S.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); base=~EX&(idx>=pd.Timestamp('2008-01-01'))&L['ok21']
r=R['SPY']; spy=P['SPY']; ls=np.log(spy)
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
RV21=r.rolling(21).std()*np.sqrt(252)*100; RV63=r.rolling(63).std()*np.sqrt(252)*100; RV10=r.rolling(10).std()*np.sqrt(252)*100; RAT=RV21/RV63
RVroc5=z(np.log(RV21).diff(5)); RV10roc5=z(np.log(RV10).diff(5)); RATroc=z(np.log(RAT).diff(5))
D21=L['fwdDD21']*100; D10=np.log(spy[::-1].rolling(10).min()[::-1].shift(-1)/spy)*100; F21=L['fwd21']*100; OFF=L['riskoff21']
VRP=S['VIX']-RV21; VRPz=z(VRP)
X={'VIX pct252':G['VIX_pct252'],'VIX roc5z':G['VIX_roc5_z252'],'VIX roc21z':G['VIX_roc21_z252'],'VVIX/VIX roc5z':G['X_VVIX_VIX_roc5_z252'],'VVIX pct':G['VVIX_pct252'],'MOVE roc5z':G['MOVE_roc5_z252'],'MOVE pct':G['MOVE_pct252'],
   'TNX chg5z':G['TNX_chg5_z252'],'TNX chg21z':G['TNX_chg21_z252'],'VIX/VIX3M roc3z':G['TS_VIX_VIX3M_roc3_z252'],'VIX9D/VIX roc5z':G['TS_VIX9D_VIX_roc5_z252'],'VIX-RV21 (VRP) z':VRPz,'GVZ roc5z':G['GVZ_roc5_z252'],'OVX roc5z':G['OVX_roc5_z252'],'SKEW roc10z':G['SKEW_roc10_z252']}
Y={'RV21 level':(RV21,[0,13,16,20,99],['<13','13-16','16-20','>20']),'RV21 roc5z':(RVroc5,[-9,-1,0,1,9],['<-1','-1..0','0..1','>1']),'21/63 ratio':(RAT,[0,0.85,1.0,1.15,9],['<.85','.85-1','1-1.15','>1.15'])}
hits=[]
for yn,(ys,yb,yl) in Y.items():
    for xn,xs in X.items():
        m=base&ys.notna()&xs.notna()
        A=pd.cut(ys[m],yb,labels=yl); B=pd.cut(xs[m],[-9,-1,0,1,9] if 'pct' not in xn else [-1,0.25,0.5,0.75,1.01],labels=['<-1','-1..0','0..1','>1'] if 'pct' not in xn else ['q1','q2','q3','q4'])
        d=pd.DataFrame({'A':A,'B':B,'off':OFF[m],'f':F21[m],'d10':D10[m],'d21':D21[m]})
        g=d.groupby(['A','B']).agg(n=('off','count'),P=('off','mean'),fwd21=('f','mean'),DD10=('d10','mean'),DD21=('d21','mean')).reset_index()
        for _,row in g.iterrows():
            if row.n>=40 and (row.P>=0.30 or row.P<=0.07): hits.append({'realized':f'{yn} {row.A}','index':f'{xn} {row.B}','n':int(row.n),'P5_21':round(row.P,2),'fwd21':round(row.fwd21,2),'DD10':round(row.DD10,2),'DD21':round(row.DD21,2)})
H=pd.DataFrame(hits).sort_values('P5_21',ascending=False)
print('base P(5% DD/21d)',round(OFF[base].mean(),3),' fwd21',round(F21[base].mean(),2),' DD21',round(D21[base].mean(),2))
print('\n=== CELLS with n>=40 and P>=0.30 (danger) ==='); print(H[H.P5_21>=0.30].to_string(index=False))
print('\n=== CELLS with n>=40 and P<=0.07 (safe) ==='); print(H[H.P5_21<=0.07].to_string(index=False))
# who moves first: realized ROC vs implied ROC
print('\n=== WHO MOVES FIRST: RV21 5d ROC z (rows) x VIX 5d ROC z (cols) : P | fwd21 | DD21 | n ===')
m=base&RVroc5.notna()
A=pd.cut(RVroc5[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B=pd.cut(G['VIX_roc5_z252'][m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
d=pd.DataFrame({'A':A,'B':B,'off':OFF[m],'f':F21[m],'d':D21[m]})
print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='off',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='off',aggfunc='count')},axis=1).to_string())
print('\n=== RV21 LEVEL (rows) x VIX-RV21 premium z (cols): the implied/realized gap at each realized level ===')
A=pd.cut(RV21[m],[0,13,16,20,99],labels=['<13','13-16','16-20','>20']); B=pd.cut(VRPz[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
d=pd.DataFrame({'A':A,'B':B,'off':OFF[m],'f':F21[m],'d':D21[m]})
print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='off',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='off',aggfunc='count')},axis=1).to_string())
# candidate rules, first-fire
def first(mm): mm=mm.fillna(False); return mm&~mm.shift(1,fill_value=False)&base
cands={
 'RV21 crosses 16 with VIX roc5z<0 (realized leads)':((RV21>16)&(RV21.shift(1)<=16))&(G['VIX_roc5_z252']<0),
 'RV21 crosses 16 with VIX roc5z>1 (implied confirms)':((RV21>16)&(RV21.shift(1)<=16))&(G['VIX_roc5_z252']>1),
 'RV21 crosses 20 with VVIX/VIX roc5z<-1':((RV21>20)&(RV21.shift(1)<=20))&(G['X_VVIX_VIX_roc5_z252']<-1),
 'RV21 crosses 20 with VVIX/VIX roc5z>0':((RV21>20)&(RV21.shift(1)<=20))&(G['X_VVIX_VIX_roc5_z252']>0),
 'RV21 roc5z>1 & VIX roc5z<0 & VIX pct<0.5 (realized up, implied asleep, calm)':(RVroc5>1)&(G['VIX_roc5_z252']<0)&(G['VIX_pct252']<0.5),
 'RV21 roc5z>1 & TNX chg5z>1':(RVroc5>1)&(G['TNX_chg5_z252']>1),
 'RV21 roc5z>1 & MOVE roc5z>1':(RVroc5>1)&(G['MOVE_roc5_z252']>1),
 'RV21>16 & VRP z<-1 (implied below realized)':(RV21>16)&(VRPz<-1),
 'RV21>16 & VRP z>1 (implied well above realized)':(RV21>16)&(VRPz>1),
 'RV21<13 & VIX roc5z>1 (spike from quiet realized)':(RV21<13)&(G['VIX_roc5_z252']>1),
 'RV21<13 & VIX roc5z>1 & VVIX/VIX roc5z<-1':(RV21<13)&(G['VIX_roc5_z252']>1)&(G['X_VVIX_VIX_roc5_z252']<-1),
 'RV21<13 & TNX chg21z>1.5 & VIX pct<0.3 (today-like)':(RV21<13)&(G['TNX_chg21_z252']>1.5)&(G['VIX_pct252']<0.3),
 '21/63 roc5z>1 & VIX roc5z<0':(RATroc>1)&(G['VIX_roc5_z252']<0),
 '21/63 roc5z>1 & VIX/VIX3M roc3z>1':(RATroc>1)&(G['TS_VIX_VIX3M_roc3_z252']>1),
}
print('\n=== candidate conjunction rules, first-fires ===')
for k,mm in cands.items():
    f=first(mm); print(f'{k:70s} n={f.sum():4d} P5/21 {OFF[f].mean():.2f} fwd21 {F21[f].mean():5.2f} DD10 {D10[f].mean():5.2f} DD21 {D21[f].mean():5.2f} P(fwd21<0) {(F21[f]<0).mean():.2f}')
print('\ntoday: RV21',round(RV21.iloc[-1],1),'RV21 roc5z',round(RVroc5.iloc[-1],2),'21/63',round(RAT.iloc[-1],2),'VRP z',round(VRPz.iloc[-1],2),'VIX',S['VIX'].iloc[-1])
