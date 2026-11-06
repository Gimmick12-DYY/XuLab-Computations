import glob,os,re,sys,itertools
import numpy as np,pandas as pd
os.chdir(sys.argv[1]); OUT=sys.argv[2]
CUT=float(sys.argv[3]); MINOV=int(sys.argv[4]); PMAX=-23.; TOP=int(sys.argv[5]) if len(sys.argv)>5 else 0
ALU="GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTCAGGAGATCGAGACCATCCCGGCTAAAACGGTGAAACCCCGTCTCTACTAAAAATACAAAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
IUP={"A":"A","C":"C","G":"G","T":"T","R":"AG","Y":"CT","S":"CG","W":"AT","K":"GT","M":"AC","B":"CGT","D":"AGT","H":"ACT","V":"ACG","N":"ACGT"}
rc=lambda s:s[::-1].translate(str.maketrans("ACGT","TGCA"))
ALUS=ALU+"N"+rc(ALU)
GEN=re.compile(r"^(Sp\d|SP\d|KLF|Klf|Maz|MAZ|ZNF148|Znf148|Zfp281|ZNF281|Egr|EGR|WT1|Wt1|ZNF263|Znf263|MBD|Mbd|E2F|E2f|Tfdp|TFDP|TFAP2|Tfap2|ZBTB14|ZBTB17|ZBTB33|NRF1|Nrf1|YY1|GFX|FLYWCH|ZNF740|CTCFL|BORIS|PLAG|PLAGL|VEZF|Vezf|SALL|RREB|THAP|ZFP|GABP)",)
def parse(run,bg,t):
    L=[]
    for f in glob.glob(f"{run}/{t}/{bg}/homerResults/motif*.motif"):
        if f.endswith("RV.motif"): continue
        ls=open(f).read().strip().split("\n"); h=ls[0].split("\t"); M=np.array([[float(x) for x in l.split("\t")] for l in ls[1:]],dtype=np.float32)
        lp=float(h[3]); 
        if lp>PMAX: continue
        m=re.search(r"T:[\d.]+\(([\d.]+)%\),B:[\d.]+\(([\d.]+)%\)",h[5]); bg_=re.search(r"BestGuess:([^/(]+)[^(]*\(([\d.]+)\)",h[1]); cons=h[0].lstrip(">")
        L.append(dict(tf=t,cons=cons,M=M,logp=lp,T=float(m[1]),B=float(m[2]),guess=bg_[1] if bg_ else "",gsc=float(bg_[2]) if bg_ else 0,gc=float(M[:,1:3].mean())))
    return sorted(L,key=lambda m:m["logp"])[:TOP] if TOP else L
def isalu(c):
    if len(c)<8: return False
    return re.search("".join("["+IUP.get(x,"ACGT")+"]" for x in c),ALUS) is not None
def norm(X):
    X=X-X.mean(1,keepdims=True); n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1; return X/n
def simmat(ms):
    N=len(ms); S=np.zeros((N,N),np.float32); by={}
    for i,m in enumerate(ms): by.setdefault(len(m["M"]),[]).append(i)
    fw={l:np.stack([ms[i]["M"] for i in ix]) for l,ix in by.items()}
    for la,ia in by.items():
      for lb,ib in by.items():
        if lb<la: continue
        A=fw[la];B0=fw[lb]
        for B in (B0,B0[:,::-1,::-1]):
          for off in range(-(la-MINOV),lb-MINOV+1):
            sa=max(0,-off); sb=max(0,off); L=min(la-sa,lb-sb)
            if L<MINOV: continue
            X=norm(A[:,sa:sa+L].reshape(len(A),-1)); Y=norm(B[:,sb:sb+L].reshape(len(B),-1))
            s=X@Y.T
            S[np.ix_(ia,ib)]=np.maximum(S[np.ix_(ia,ib)],s)
    return np.maximum(S,S.T)
SL="mbd3-rbbp4 mbd3-rbbp7 wiz-znf644 rbbp4-znf512b dnmt1-hp1 dnmt1-ezh2 ezh2-mtf2 mef2a-mef2d maz-znf143 hp1-pcgf6 adnp-hp1 ctcf-stag2 rbbp4-smarca1 hoxd10-srebf2 foxk1-mlx ddit3-rbbp4 mlx-thap6 ddit3-znf367 hoxb8-hp1 thap6-znf746 hp1-znf143 hes1-znf367 mlx-srebf2 hoxa5-znf581 hoxc9-rbbp4 rbbp7-srcap".split()
allr=[]; det=[]
for run,bg in itertools.product(("motif","motif_bins"),("homer_ocr","homer")):
    tfs=sorted(f.split("/")[1] for f in glob.glob(f"{run}/*/{bg}/knownResults.txt"))
    ms=[]; 
    for t in tfs: ms+=parse(run,bg,t)
    S=simmat(ms); tf=np.array([m["tf"] for m in ms]); utf=sorted(set(tf)); ix={t:np.where(tf==t)[0] for t in utf}
    # number of TFs carrying a similar motif
    ntf=np.array([len({t for t in tf[S[i]>=CUT]}) for i in range(len(ms))])
    for i,m in enumerate(ms):
        m["gen"]=bool(GEN.match(m["guess"]) and m["gsc"]>=.6) or m["gc"]>=.7 ; m["alu"]=isalu(m["cons"]); m["ntf"]=int(ntf[i])
    rare=max(2,int(.1*len(utf)))
    print(f"\n#### {run} {bg}: TFs with results {len(utf)}/{len(tfs)}, motifs(P<=1e-10) {len(ms)}; median per TF {np.median([len(ix[t]) for t in utf])}; generic {sum(m['gen'] for m in ms)}, Alu {sum(m['alu'] for m in ms)}; rare = in <= {rare} TFs: {int((ntf<=rare).sum())}")
    rec=[]
    for a,b in itertools.combinations(utf,2):
        s=S[np.ix_(ix[a],ix[b])]; ij=np.argwhere(s>=CUT)
        any_=len(ij)>0; ng=[];rr=[]
        for i,j in ij:
            x=ms[ix[a][i]];y=ms[ix[b][j]]
            if x["gen"] or y["gen"] or x["alu"] or y["alu"]: continue
            ng.append((i,j))
            if x["ntf"]<=rare and y["ntf"]<=rare: rr.append((i,j))
        rec.append(dict(a=a,b=b,any=any_,nongeneric=len(ng)>0,rare=len(rr)>0,n=len(ij),n_ng=len(ng),n_rare=len(rr)))
        if f"{a}-{b}" in SL or f"{b}-{a}" in SL:
            for lab,lst in (("rare",rr),("nongeneric",ng)):
                for i,j in lst:
                    x=ms[ix[a][i]];y=ms[ix[b][j]]
                    det.append(dict(run=run,bg=bg,pair=f"{a}-{b}",cls=lab,sim=round(float(s[i,j]),3),a_cons=x["cons"],a_logp=round(x["logp"]),a_T=x["T"],a_B=x["B"],a_guess=f"{x['guess']}({x['gsc']})",a_ntf=x["ntf"],b_cons=y["cons"],b_logp=round(y["logp"]),b_T=y["T"],b_B=y["B"],b_guess=f"{y['guess']}({y['gsc']})",b_ntf=y["ntf"]))
    R=pd.DataFrame(rec); R["run"]=run; R["bg"]=bg; allr.append(R)
    print("background across",len(R),"pairs: any",round(R["any"].mean(),3),"non-generic",round(R.nongeneric.mean(),3),"rare",round(R.rare.mean(),3))
    for p in SL:
        a,b=sorted(p.split("-")); q=R[(R.a==a)&(R.b==b)]
        if len(q)==0: print(" ",p,"NOT TESTED (missing:",[t for t in(a,b) if t not in utf],")")
        else:
            q=q.iloc[0]; print(" ",p,"any",q["any"],"n",q.n,"non-generic",q.n_ng,"rare",q.n_rare)
pd.concat(allr).to_csv(f"{OUT}/denovo_pairs_all_{CUT}_{MINOV}_{TOP}.tsv",sep="\t",index=False); pd.DataFrame(det).to_csv(f"{OUT}/denovo_pairs_detail_{CUT}_{MINOV}_{TOP}.tsv",sep="\t",index=False); print(len(det))
