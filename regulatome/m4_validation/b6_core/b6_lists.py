# B6 lists (a)-(e) per TF from R5 core (manifest v7 verified first). No Perturb-seq input; reads r5_core lists, the m2_abc links and the Gate 1
# cellmats (binding reads) only. (a) core top_1000; (b) promoter-only and (c) distal-only top 1,000 by the promoter / distal parts of S (N_e >= 2, as in
# r5_core/checks); (d) S_atac top 1,000; (e) binding-only null: top 1,000 by summed raw CPM over the gene's linked elements, no ABC weighting.
# Writes m4_validation/b6_core/lists/<TF>.tsv (gene, list) and lists_summary.tsv. usage: b6_lists.py
import os, subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; R5 = f"{X}/regulatome/r5_core"; O = f"{X}/regulatome/m4_validation/b6_core"; G = f"{X}/regulatome/gate1/cellmats"
assert not os.path.exists(f"{O}/manifest_b6.sha256"), "manifest_b6 exists; B6 outputs are frozen"
v = subprocess.run("sha256sum manifest_v7.sha256 && sha256sum -c --quiet manifest_v7.sha256", shell=True, cwd=R5, capture_output=True, text=True)
assert v.returncode == 0 and v.stdout.startswith("b89b483bba64b4e54cd39ed4f116510528702e3d35ebd9b282b7fe01f45f76ac"), "manifest v7 check failed: " + v.stdout + v.stderr
print("manifest v7 verified", v.stdout.split()[0], flush=True); os.makedirs(f"{O}/lists", exist_ok=True)
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None, names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
Ea = E[E.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "ABC.Score"])
coord = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(Ea.chrom, Ea.start, Ea.end))}; A["e"] = (A.chr + ":" + A.start.astype(str) + "-" + A.end.astype(str)).map(coord).astype(int)
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values); atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float); atac_cpm = atac / atac.sum() * 1e6
SA = A.assign(c=atac_cpm[A.e.values] * A["ABC.Score"].values).groupby("TargetGene").c.sum().rename("S_atac").reset_index().rename(columns={"TargetGene": "gene"})  # as in r5_core
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; rows = []
def top(df, col, ok=None, n=1000):
    d = df if ok is None else df[ok]; d = d[d[col] > 0]; return list(d.sort_values(col, ascending=False, kind="stable").gene.head(n))
for t in tfs:
    L = pd.read_csv(f"{R5}/lists/{t}.tsv.gz", sep="\t"); L["S_prom"] = L.S * L.promoter_share; L["S_dist"] = L.S * L.distal_share
    y = np.asarray(sp.load_npz(f"{G}/{t}.raw.npz").sum(0)).ravel(); cpm = y / np.load(f"{G}/{t}.cells.npz")["reads"].astype(float).sum() * 1e6
    B = A.assign(c=cpm[A.e.values]).groupby("TargetGene").c.sum().rename("S_bind").reset_index().rename(columns={"TargetGene": "gene"})
    ok = L.N_e >= 2
    lists = {"a_core": list(L.gene[L.top_1000]), "b_promoter_only": top(L, "S_prom", ok), "c_distal_only": top(L, "S_dist", ok), "d_atac_only": top(SA, "S_atac"), "e_binding_only": top(B, "S_bind")}
    pd.DataFrame([(g, k) for k, v_ in lists.items() for g in v_], columns=["gene", "list"]).to_csv(f"{O}/lists/{t}.tsv", sep="\t", index=False)
    rows.append(dict(tf=t, **{k: len(v_) for k, v_ in lists.items()}))
pd.DataFrame(rows).to_csv(f"{O}/lists_summary.tsv", sep="\t", index=False); print(pd.DataFrame(rows).describe().to_string())
