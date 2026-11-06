import numpy as np, sys, gzip, collections, csv
from pathlib import Path
sys.path.insert(0,"downstream")
from peak_coverage import load_per_bin_signal
r=list(csv.reader(open("data/TF1000cells.meta.csv")));i=r[0].index("TF")
cc=collections.Counter(x[i].lower() for x in r[1:])
# ENCODE CTCF summit bins (1 kb), label by ChromHMM state at summit
st=collections.defaultdict(list)
for ln in gzip.open("data/HEK293T_chromHMM18.bed.gz","rt"):
    p=ln.split("\t")
    if len(p)>=4: st[p[0]].append((int(p[1]),int(p[2]),p[3]))
S={c:(np.array([x[0] for x in v]),np.array([x[1] for x in v]),[x[2] for x in v]) for c,v in ((c,sorted(v)) for c,v in st.items())}
sites={}
for ln in open("downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"):
    p=ln.split("\t"); c=p[0]; s=int(p[1]); e=int(p[2])
    summ=s+int(p[9]) if len(p)>9 and p[9].strip().lstrip("-").isdigit() and int(p[9])>=0 else (s+e)//2
    if c not in S: continue
    a,b,lab=S[c]; k=np.searchsorted(a,summ,side="right")-1
    state=lab[k] if k>=0 and summ<b[k] else "NA"
    sites[(c,summ//1000*1000)]=state
print("sites(bins)",len(sites), collections.Counter(sites.values()).most_common(5))
cand=["ctcf","stag2"]+[t for t,n in cc.items() if 2400<=n<=5200 and t not in("ctcf","stag2")]+["rbbp4","nfya","xpa","e2f3","safb","znf367","pa2g4"]
keys=list(sites); idx={k:j for j,k in enumerate(keys)}
X={}
for tf in cand:
    sig,regions,kind,_=load_per_bin_signal(Path("unified/work")/tf/"mm")
    v=np.zeros(len(keys))
    for val,name in zip(sig,regions):
        if val==0: continue
        c,rest=name.split(":"); s0=int(rest.split("-")[0])
        j=idx.get((c,s0//1000*1000))
        if j is not None: v[j]+=val
    X[tf]=v
dist=np.array([sites[k] in("Quies","ZNF/Rpts","Het","ReprPCWk","ReprPC","EnhWk","TxWk","Tx") for k in keys])
for lab,m in [("all CTCF sites",np.ones(len(keys),bool)),("CTCF sites outside TSS/active-enhancer states",dist)]:
    out=[]
    for tf in cand:
        if tf=="ctcf": continue
        out.append((np.corrcoef(X["ctcf"][m],X[tf][m])[0,1],tf,cc[tf]))
    out.sort(reverse=True)
    rk=[t for _,t,_ in out].index("stag2")+1
    dm=[r_ for r_,t,n in out if 2400<=n<=5200 and t!="stag2"]
    print(f"== {lab} (n={m.sum()}): r(CTCF,STAG2)={[r_ for r_,t,_ in out if t=='stag2'][0]:.3f}, rank {rk}/{len(out)}; depth-matched TFs (2.4-5.2k cells, n={len(dm)}) median {np.median(dm):.3f}, max {max(dm):.3f}, frac >= STAG2 {np.mean(np.array(dm)>=[r_ for r_,t,_ in out if t=='stag2'][0]):.3f}")
    print("   top:",[(t,round(r_,3)) for r_,t,_ in out[:8]])
    # also: signal enrichment at CTCF sites relative to TF total
for tf in ["ctcf","stag2","rbbp4","hes1","wiz"]:
    if tf in X:
        sig,regions,kind,_=load_per_bin_signal(Path("unified/work")/tf/"mm") if False else (None,None,None,None)
