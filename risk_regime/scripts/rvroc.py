import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); S=pd.read_pickle('S.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); base=~EX&(idx>=pd.Timestamp('2008-01-01'))&L['ok21']
r=R['SPY']; spy=P['SPY']
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
RV5=r.rolling(5).std(); RV10=r.rolling(10).std(); RV21=r.rolling(21).std(); RV63=r.rolling(63).std()
RVr={'RV21 roc5z':z(np.log(RV21).diff(5)),'RV21 roc10z':z(np.log(RV21).diff(10)),'RV21 roc21z':z(np.log(RV21).diff(21)),'RV10 roc5z':z(np.log(RV10).diff(5)),'RV5 roc5z':z(np.log(RV5).diff(5)),
     'RV accel (roc5 - roc5 lag5) z':z(np.log(RV21).diff(5)-np.log(RV21).diff(5).shift(5)),'21/63 roc5z':z(np.log(RV21/RV63).diff(5)),'10/21 roc5z':z(np.log(RV10/RV21).diff(5)),
     'VRP chg5z (VIX-RV21 change)':z((S['VIX']-RV21*np.sqrt(252)*100).diff(5))}
X={'VIX roc5z':G['VIX_roc5_z252'],'VIX roc21z':G['VIX_roc21_z252'],'VVIX roc5z':G['VVIX_roc5_z252'],'VVIX/VIX roc5z':G['X_VVIX_VIX_roc5_z252'],'MOVE roc5z':G['MOVE_roc5_z252'],'MOVE roc21z':G['MOVE_roc21_z252'],
   'TNX chg5z':G['TNX_chg5_z252'],'TNX chg21z':G['TNX_chg21_z252'],'TYX chg5z':G['TYX_chg5_z252'],'VIX/VIX3M roc3z':G['TS_VIX_VIX3M_roc3_z252'],'VIX9D/VIX roc5z':G['TS_VIX9D_VIX_roc5_z252'],'MOVE/VIX roc5z':G['X_MOVE_VIX_roc5_z252'],
   'GVZ roc5z':G['GVZ_roc5_z252'],'OVX roc5z':G['OVX_roc5_z252'],'SKEW roc10z':G['SKEW_roc10_z252'],'VIX1D roc3z':G['VIX1D_roc3_z252']}
D21=L['fwdDD21']*100; D10=np.log(spy[::-1].rolling(10).min()[::-1].shift(-1)/spy)*100; F21=L['fwd21']*100; F10=(np.log(spy).shift(-10)-np.log(spy))*100; OFF=L['riskoff21']
lab=['<-1','-1..0','0..1','>1']; cut=[-9,-1,0,1,9]
hits=[]
for yn,ys in RVr.items():
    for xn,xs in X.items():
        m=base&ys.notna()&xs.notna()
        if m.sum()<500: continue
        d=pd.DataFrame({'A':pd.cut(ys[m],cut,labels=lab),'B':pd.cut(xs[m],cut,labels=lab),'off':OFF[m],'f21':F21[m],'f10':F10[m],'d10':D10[m],'d21':D21[m]})
        g=d.groupby(['A','B']).agg(n=('off','count'),P=('off','mean'),fwd10=('f10','mean'),fwd21=('f21','mean'),DD10=('d10','mean'),DD21=('d21','mean')).reset_index()
        for _,row in g.iterrows():
            if row.n>=40 and (row.P>=0.30 or row.P<=0.07): hits.append({'realized ROC':f'{yn} {row.A}','index ROC':f'{xn} {row.B}','n':int(row.n),'P5_21':round(row.P,2),'fwd10':round(row.fwd10,2),'fwd21':round(row.fwd21,2),'DD10':round(row.DD10,2),'DD21':round(row.DD21,2)})
H=pd.DataFrame(hits).sort_values('P5_21',ascending=False)
print('base P',round(OFF[base].mean(),3),'fwd21',round(F21[base].mean(),2),'DD21',round(D21[base].mean(),2))
print('\n=== ROC x ROC DANGER cells (n>=40, P>=0.30) ==='); print(H[H.P5_21>=0.30].to_string(index=False))
print('\n=== ROC x ROC SAFE cells (n>=40, P<=0.07) ==='); print(H[H.P5_21<=0.07].to_string(index=False))
# three-way: realized ROC x VIX ROC x VVIX/VIX ROC (the full "who moves first" with vol-of-vol)
print('\n=== THREE-WAY: RV21 roc5z >1 (realized expanding) split by VIX roc5z and VVIX/VIX roc5z ===')
m=base&RVr['RV21 roc5z'].notna()&(RVr['RV21 roc5z']>1)
d=pd.DataFrame({'A':pd.cut(G['VIX_roc5_z252'][m],cut,labels=lab),'B':pd.cut(G['X_VVIX_VIX_roc5_z252'][m],cut,labels=lab),'off':OFF[m],'f':F21[m],'d':D21[m]})
print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='off',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='off',aggfunc='count')},axis=1).to_string())
print('\n=== THREE-WAY: RV21 roc5z <-1 (realized compressing) split by VIX roc5z and VVIX/VIX roc5z ===')
m=base&RVr['RV21 roc5z'].notna()&(RVr['RV21 roc5z']<-1)
d=pd.DataFrame({'A':pd.cut(G['VIX_roc5_z252'][m],cut,labels=lab),'B':pd.cut(G['X_VVIX_VIX_roc5_z252'][m],cut,labels=lab),'off':OFF[m],'f':F21[m],'d':D21[m]})
print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='off',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='off',aggfunc='count')},axis=1).to_string())
# multi-horizon realized ROC: 5d vs 21d (acceleration of realized)
print('\n=== RV21 roc21z (rows, month trend) x RV21 roc5z (cols, week impulse) ===')
m=base&RVr['RV21 roc5z'].notna()&RVr['RV21 roc21z'].notna()
d=pd.DataFrame({'A':pd.cut(RVr['RV21 roc21z'][m],cut,labels=lab),'B':pd.cut(RVr['RV21 roc5z'][m],cut,labels=lab),'off':OFF[m],'f':F21[m],'d':D21[m]})
print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='off',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='off',aggfunc='count')},axis=1).to_string())
# first-fire ROC-only rules
def first(mm): mm=mm.fillna(False); return mm&~mm.shift(1,fill_value=False)&base
C={
 'RV21 roc5z>1 & VIX roc5z>1 & VVIX/VIX roc5z<-1 (all three, VVIX lagging)':(RVr['RV21 roc5z']>1)&(G['VIX_roc5_z252']>1)&(G['X_VVIX_VIX_roc5_z252']<-1),
 'RV21 roc5z>1 & VIX roc5z>1 & VVIX/VIX roc5z>0 (VVIX keeping up)':(RVr['RV21 roc5z']>1)&(G['VIX_roc5_z252']>1)&(G['X_VVIX_VIX_roc5_z252']>0),
 'RV21 roc5z>1 & TNX chg5z>1 & MOVE roc5z>1 (realized + rates + bond vol)':(RVr['RV21 roc5z']>1)&(G['TNX_chg5_z252']>1)&(G['MOVE_roc5_z252']>1),
 'RV21 roc21z>1 & RV21 roc5z>1 (month and week both expanding)':(RVr['RV21 roc21z']>1)&(RVr['RV21 roc5z']>1),
 'RV21 roc21z>1 & VIX roc21z<0 (month of realized up, implied not)':(RVr['RV21 roc21z']>1)&(G['VIX_roc21_z252']<0),
 'RV21 roc21z<-1 & VIX roc21z<-1 & VVIX roc21z<-1 (everything compressing)':(RVr['RV21 roc21z']<-1)&(G['VIX_roc21_z252']<-1)&(G['VVIX_roc21_z252']<-1),
 'RV accel z>1 & VIX roc5z>1':(RVr['RV accel (roc5 - roc5 lag5) z']>1)&(G['VIX_roc5_z252']>1),
 'VRP chg5z<-1 & RV21 roc5z>1 (realized rising faster than VIX)':(RVr['VRP chg5z (VIX-RV21 change)']<-1)&(RVr['RV21 roc5z']>1),
 'VRP chg5z>1 & RV21 roc5z>1 (VIX rising faster than realized)':(RVr['VRP chg5z (VIX-RV21 change)']>1)&(RVr['RV21 roc5z']>1),
 '10/21 roc5z>1 & VIX9D/VIX roc5z>1 (short realized + front-end implied)':(RVr['10/21 roc5z']>1)&(G['TS_VIX9D_VIX_roc5_z252']>1),
 'RV21 roc5z<-1 & TNX chg5z>1 (realized compressing into a yield rip)':(RVr['RV21 roc5z']<-1)&(G['TNX_chg5_z252']>1),
 'RV21 roc5z<-1 & MOVE roc5z>1':(RVr['RV21 roc5z']<-1)&(G['MOVE_roc5_z252']>1),
}
print('\n=== ROC-only conjunction rules, first-fires ===')
for k,mm in C.items():
    f=first(mm); print(f'{k:72s} n={f.sum():4d} P5/21 {OFF[f].mean():.2f} fwd10 {F10[f].mean():5.2f} fwd21 {F21[f].mean():5.2f} DD10 {D10[f].mean():5.2f} DD21 {D21[f].mean():5.2f}')
print('\ntoday:',{k:round(float(v.iloc[-1]),2) for k,v in RVr.items()})
