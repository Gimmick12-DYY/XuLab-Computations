import numpy as np, sys, glob, os, csv, collections
from pathlib import Path
sys.path.insert(0,"downstream")
def readbed(f,maxcol=3):
    out=collections.defaultdict(list)
    for ln in open(f):
        if ln.startswith(("#","track")): continue
        p=ln.split("\t")
        out[p[0]].append((int(p[1]),int(p[2])))
    return {c:np.array(sorted(v)) for c,v in out.items()}
def near(a,b,pad=500):
    """fraction of intervals in a within pad of an interval in b"""
    hit=tot=0
    for c,A in a.items():
        tot+=len(A)
        if c not in b: continue
        B=b[c]; s=B[:,0]-pad; e=B[:,1]+pad
        emax=np.maximum.accumulate(e)
        k=np.searchsorted(s,A[:,1],side="left")-1
        ok=(k>=0)&(emax[np.clip(k,0,None)]>A[:,0])
        hit+=ok.sum()
    return hit/tot
# ChromHMM elements
import gzip
st=collections.defaultdict(list)
for ln in gzip.open("data/HEK293T_chromHMM18.bed.gz","rt"):
    p=ln.split("\t")
    if len(p)<4: continue
    st[p[0]].append((int(p[1]),int(p[2]),p[3]))
for c in st: st[c].sort()
def states(peaks):
    cnt=collections.Counter(); lens=[]
    for c,A in peaks.items():
        if c not in st: continue
        S=np.array([x[0] for x in st[c]]); E=[x[1] for x in st[c]]
        mids=(A[:,0]+A[:,1])//2; k=np.searchsorted(S,mids,side="right")-1
        for kk,m in zip(k,mids):
            if kk>=0 and m<E[kk]: cnt[st[c][kk][2]]+=1; lens.append(E[kk]-st[c][kk][0])
    n=sum(cnt.values()); return {k:round(v/n,3) for k,v in cnt.most_common(6)}, int(np.median(lens)), n
P={Path(f).stem:readbed(f) for f in glob.glob("tf_complex/work/peaks/*.bed")}
enc=readbed("downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed")
print("chrom names enc sample",list(enc)[:3],"n enc",sum(len(v) for v in enc.values()))
for name,pk in [("ctcf",P["ctcf"]),("stag2",P["stag2"]),("ENCODE CTCF",enc)]:
    print("STATES",name,states(pk))
allp=[]
for t,pk in P.items(): allp.append((len(states(pk)) and 0, t))
med_len=[]
for t,pk in P.items():
    _,ml,_=states(pk); med_len.append(ml)
print("panel median element length containing peaks: median over TFs",int(np.median(med_len)))
npk={t:sum(len(v) for v in pk.values()) for t,pk in P.items()}
print("peaks: ctcf",npk["ctcf"],"stag2",npk["stag2"],"panel median",int(np.median(list(npk.values()))))
print("STAG2 peaks within 500bp of ENCODE CTCF: %.3f"%near(P["stag2"],enc))
print("CTCF(scATAC-TAG) peaks within 500bp of ENCODE CTCF: %.3f"%near(P["ctcf"],enc))
res=[]
for t,pk in P.items():
    if t=="stag2": continue
    res.append((near(P["stag2"],pk),t,npk[t]))
res.sort(reverse=True)
print("STAG2 peaks within 500bp of TF X peaks (top 10):",[(t,round(f,3),n) for f,t,n in res[:10]])
print("rank of ctcf:",[t for f,t,n in res].index("ctcf")+1,"of",len(res))
# null: other TFs with similar peak count to ctcf (within 2x)
sim=[f for f,t,n in res if t!="ctcf" and npk["ctcf"]/2<=n<=npk["ctcf"]*2]
print("STAG2->ctcf %.3f vs TFs with 6-25k peaks: median %.3f max %.3f n=%d"%([f for f,t,n in res if t=="ctcf"][0],np.median(sim),max(sim),len(sim)))
# depth-matched: for each TF X with similar peak count to stag2 (0.5-2x), frac of X peaks near ctcf
sm=[(near(pk,P["ctcf"]),t,npk[t]) for t,pk in P.items() if t not in("stag2","ctcf") and npk["stag2"]/2<=npk[t]<=npk["stag2"]*2]
print("frac of peaks near CTCF peak: stag2 %.3f ; similar-peak-count TFs:"%near(P["stag2"],P["ctcf"]),sorted([(t,round(f,3)) for f,t,n in sm],key=lambda x:-x[1]))
smE=[(near(pk,enc),t) for t,pk in P.items() if t not in("stag2","ctcf") and npk["stag2"]/2<=npk[t]<=npk["stag2"]*2]
print("frac of peaks near ENCODE CTCF: stag2 %.3f ; similar-peak-count TFs:"%near(P["stag2"],enc),sorted([(t,round(f,3)) for f,t in smE],key=lambda x:-x[1]))
allE=sorted([(round(near(pk,enc),3),t) for t,pk in P.items()],reverse=True)
print("ALL TFs frac peaks near ENCODE CTCF top12:",allE[:12])
