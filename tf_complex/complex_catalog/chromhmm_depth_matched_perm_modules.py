import numpy as np, sys, csv, collections
sys.path.insert(0,"scripts")
from plot_correlation_matrix import load_similarity_tsv
exec(open("/dev/stdin").read()) if False else None
r=list(csv.reader(open("../data/TF1000cells.meta.csv")));i=r[0].index("TF")
cc=collections.Counter(x[i].lower() for x in r[1:])
rng=np.random.default_rng(1)
mats={"noq":"results_chromhmm_elements_counts_spqn_noquies/tf_similarity_noquies.tsv",
      "noqS":"results_chromhmm_elements_counts_spqn_noquies/spqn_similarity_noquies.tsv",
            "enhS":"results_chromhmm_elements_counts_spqn_enhancer/spqn_similarity_enhancer.tsv"}
known={"PCGF6/HP1":["pcgf6","hp1"],"PRC2-adjacent":["mtf2","ezh2","dr1","thap6","champ1","wiz","znf644","srcap"],"low-depth repressive":["neurod4","setdb1","znf79","sox12","zbtb43","znf777"],"STAG2/ZNF581/DNMT1":["stag2","znf581","dnmt1"],"MAZ/ZNF143":["maz","znf143"],"RBBP4/E2F3/XPA hub":["rbbp4","rbbp7","e2f3","xpa","ddit3","znf703"],"NFYA/SOX4 hub":["nfya","sox4","mlx","mbd3"]}
def g(R,idx):
    s=R[np.ix_(idx,idx)]; return s[~np.eye(len(idx),dtype=bool)].mean()
for tag,f in mats.items():
    L,R=load_similarity_tsv(f); n=len(L)
    order=np.argsort([-cc[t] for t in L]); rank=np.empty(n,int); rank[order]=np.arange(n)
    print("==",tag)
    for name,grp in known.items():
        idx=[L.index(t) for t in grp]; obs=g(R,idx); null=[]
        for _ in range(5000):
            pick=[]
            for t in idx:  # depth-matched: random TF within +-4 ranks, no repeats
                cand=[order[k] for k in range(max(0,rank[t]-4),min(n,rank[t]+5)) if order[k] not in pick]
                pick.append(rng.choice(cand))
            null.append(g(R,pick))
        null=np.array(null); print(f"  {name}: r={obs:.3f} depth-matched p={(1+(null>=obs).sum())/5001:.4f} null_mean={null.mean():.3f}")
    continue
    iu=np.triu_indices(n,1); v=R[iu]; o=np.argsort(-v)[:30]
    print("  TOP30: "+"; ".join(f"{L[iu[0][k]]}-{L[iu[1][k]]} {v[k]:.3f}" for k in o))
