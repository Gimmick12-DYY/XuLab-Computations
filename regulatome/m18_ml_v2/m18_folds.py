# m18_ml_v2 folds (SPEC Amendment 2): 5 folds over whole chromosomes, balanced on gene count, seed 20261112.
import hashlib
import numpy as np, pandas as pd
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{RG}/m18_ml_v2"
ug = np.load(f"{RG}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str)
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name"]).drop_duplicates("name", keep="last").set_index("name")
G = pd.DataFrame(dict(gene=ug, chr=GL.chr.reindex(ug).values)); assert G.chr.notna().all()
n = G.chr.value_counts(); rng = np.random.default_rng(20261112); ch = list(n.index); rng.shuffle(ch); ch = sorted(ch, key=lambda c: -n[c])
tot = np.zeros(5, int); F = {}
for c in ch: k = int(np.argmin(tot)); F[c] = k; tot[k] += n[c]
G["fold"] = G.chr.map(F); G.to_csv(f"{O}/folds.tsv", sep="\t", index=False)
h = hashlib.sha256(open(f"{O}/folds.tsv", "rb").read()).hexdigest(); open(f"{O}/folds.tsv.sha256", "w").write(f"{h}  folds.tsv\n")
print(G.groupby("fold").agg(n_genes=("gene", "size"), chroms=("chr", lambda v: ",".join(sorted(set(v), key=lambda x: (len(x), x))))).to_string()); print(h)
