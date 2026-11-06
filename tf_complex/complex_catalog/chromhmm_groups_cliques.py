import numpy as np, sys, itertools
sys.path.insert(0,"scripts")
from plot_correlation_matrix import load_similarity_tsv
M={}
for tag,f in [("all","results_chromhmm_elements_counts_spqn/tf_similarity_genome.tsv"),
              ("noq","results_chromhmm_elements_counts_spqn_noquies/tf_similarity_noquies.tsv"),
              ("enh","results_chromhmm_elements_counts_spqn_enhancer/tf_similarity_enhancer.tsv"),
              ("tss","results_chromhmm_elements_counts_spqn_tss/tf_similarity_tss.tsv"),
              ("noqS","results_chromhmm_elements_counts_spqn_noquies/spqn_similarity_noquies.tsv")]:
    L,R=load_similarity_tsv(f); M[tag]=(L,R)
L=M["all"][0]; n=len(L); ix={t:i for i,t in enumerate(L)}
rng=np.random.default_rng(0)
def gstat(R,idx):
    s=R[np.ix_(idx,idx)]; return s[~np.eye(len(idx),dtype=bool)].mean()
def perm_p(R,idx,B=20000):
    obs=gstat(R,idx); k=len(idx)
    null=np.array([gstat(R,rng.choice(n,k,replace=False)) for _ in range(B)])
    return obs,(1+(null>=obs).sum())/(B+1)
known={
 "NuRD core (RBBP4/7+MBD3)":["rbbp4","rbbp7","mbd3"],
 "NuRD + ZNF512B (NuRD-recruiting ZF)":["rbbp4","rbbp7","mbd3","znf512b"],
 "PRC2 (EZH2/MTF2/RBBP4/RBBP7)":["ezh2","mtf2","rbbp4","rbbp7"],
 "PRC2 core subunits only (EZH2/MTF2)":["ezh2","mtf2"],
 "RBBP4/RBBP7 paralogs":["rbbp4","rbbp7"],
 "CTCF-cohesin (CTCF/STAG2)":["ctcf","stag2"],
 "CTCF-cohesin + ZNF143":["ctcf","stag2","znf143"],
 "G9a/GLP-WIZ-ZNF644":["wiz","znf644"],
 "ChAHP (ADNP/HP1)":["adnp","hp1"],
 "HP1/SETDB1 heterochromatin":["hp1","setdb1"],
 "HP1/CHAMP1/ADNP (HP1 partners)":["hp1","champ1","adnp"],
 "DNMT1/HP1 maintenance":["dnmt1","hp1"],
 "PRC1.6 / PCGF6 with E2F-MGA sites (PCGF6/E2F3)":["pcgf6","e2f3"],
 "MEF2A/MEF2D heterodimer":["mef2a","mef2d"],
 "SoxC (SOX4/SOX12)":["sox4","sox12"],
 "ZNF512/ZNF512B paralogs":["znf512","znf512b"],
 "ZBTB40/ZBTB43":["zbtb40","zbtb43"],
 "HOX (all 8)":["hoxa5","hoxa10","hoxb6","hoxb8","hoxb9","hoxc4","hoxc9","hoxd10"],
 "Posterior HOX9-10":["hoxa10","hoxb9","hoxc9","hoxd10"],
 "Notch: RBPJ/HES1":["rbpj","hes1"],
 "Promoter CCAAT/E-box: NFYA/MLX/SREBF2":["nfya","mlx","srebf2"],
}
def fmt(tag,idx):
    Lt,R=M[tag]; ii=[Lt.index(L[i]) for i in idx]
    o,p=perm_p(R,ii,5000); return f"{tag}={o:.3f}(p={p:.4f})"

class _G:
    def __init__(s): s.adj={}
    def add_edge(s,a,b,**k):
        s.adj.setdefault(a,set()).add(b); s.adj.setdefault(b,set()).add(a)
class nx:
    Graph=_G
    @staticmethod
    def connected_components(G):
        seen=set()
        for v in G.adj:
            if v in seen: continue
            st=[v]; comp=set()
            while st:
                u=st.pop()
                if u in comp: continue
                comp.add(u); st.extend(G.adj[u]-comp)
            seen|=comp; yield comp
    @staticmethod
    def find_cliques(G):
        out=[]
        def bk(R,P,X):
            if not P and not X: out.append(list(R)); return
            for v in list(P):
                bk(R|{v},P&G.adj[v],X&G.adj[v]); P=P-{v}; X=X|{v}
        bk(set(),set(G.adj),set()); return out
print("KNOWN")
for name,g in known.items():
    miss=[t for t in g if t not in ix]
    idx=[ix[t] for t in g if t in ix]
    print(name,"|",",".join(g),"| missing:",miss," | "," ".join(fmt(t,idx) for t in ["all","noq","enh","tss","noqS"]))
# per-TF percentile ranks of pairs (noq)
print("DATA-DRIVEN: mutual top-3 neighbours, noquies raw")
Lq,Rq=M["noq"]
Rz=Rq.copy(); np.fill_diagonal(Rz,-9)
top={i:set(np.argsort(-Rz[i])[:3]) for i in range(n)}
G=nx.Graph()
for i in range(n):
    for j in top[i]:
        if i in top[j]: G.add_edge(Lq[i],Lq[j],r=Rq[i,j])
for c in sorted(nx.connected_components(G),key=len,reverse=True):
    idx=[Lq.index(t) for t in c]
    print(f"  n={len(c)} meanr={gstat(Rq,idx):.3f}: "+" ".join(sorted(c)))
for tag in ["all","noq","enh","tss"]:
    Lt,R=M[tag]; iu=np.triu_indices(n,1); thr=np.quantile(R[iu],0.98)
    G=nx.Graph(); 
    for a,b in zip(*iu):
        if R[a,b]>=thr: G.add_edge(Lt[a],Lt[b])
    print(f"DATA-DRIVEN: top2% edges {tag} (r>={thr:.3f}) maximal cliques size>=3")
    cl=sorted([c for c in nx.find_cliques(G) if len(c)>=3],key=len,reverse=True)
    for c in cl[:15]:
        idx=[Lt.index(t) for t in c]; print(f"  n={len(c)} meanr={gstat(R,idx):.3f}: "+" ".join(sorted(c)))
