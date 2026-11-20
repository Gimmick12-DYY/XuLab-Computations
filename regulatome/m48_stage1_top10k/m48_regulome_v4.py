# m48: Regulome v4 = top 10,000 lab HITAG peaks per TF, summit in the gene's promoter element, an ABC-linked element, or within 10 kb of the TSS; 79 TFs; Perturb-seq columns are annotations only.
import os
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"
src = open(f"{R}/m38_link_rules/m38.py").read(); exec(src[:src.index("T25 = pd.read_csv")])  # UG, gi, rule_abc, window, cicero, elem_of, Le, Lg, Ls
O = f"{R}/m48_stage1_top10k/out"; os.makedirs(O, exist_ok=True)
tfs79 = [l.strip() for l in open(f"{R}/m1_binding/v2/tfs.txt") if l.strip()]; rows = []; coord_only = 0
for t in tfs79:
    f = f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    if not os.path.exists(f) or os.path.getsize(f) == 0: continue
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P = P[P.sig.rank(ascending=False, method="first") <= 10000]; c5, p5 = P.c.values.astype(str), (P.s + P.off).values
    e = elem_of(c5, p5); v = np.zeros(NE, bool); v[e[e >= 0]] = True
    prom = np.zeros(NG, bool); prom[Lg[v[Le] & Ls]] = True; A, _ = rule_abc(c5, p5); W = window(c5, p5, 10000); ci, has = cicero(t)
    ltype = np.where(prom, "promoter", np.where(A, "ABC", np.where(W, "within_10kb", "")))
    for j in np.flatnonzero(A | W): rows.append((t, UG[j], ltype[j], bool(ci[j])))
    coord_only += int((ci & ~(A | W)).sum())
E = pd.DataFrame(rows, columns=["tf", "gene", "link_type", "also_coordinated_binding_link"])
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); KD = set(T25.tf)
zmap = T25.set_index(["tf", "gene"]).z_top25; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
z = zmap.reindex(list(zip(E.tf, E.gene))).values.astype(float)
for k, (t, g) in enumerate(zip(E.tf, E.gene)):
    if g in OWN.get(t, [t]): z[k] = np.nan
a = np.abs(np.nan_to_num(z)); E["knockdown_available"] = E.tf.isin(KD); E["knockdown_z (annotation)"] = z
E["direction (|z| >= 2)"] = np.where(a >= 2, np.where(z < 0, "down on knockdown", "up on knockdown"), "none"); E["response class"] = np.where(a >= 4, "|z| >= 4", np.where(a >= 3, "|z| >= 3", np.where(a >= 2, "|z| >= 2", "none")))
E["label"] = "Regulome v4: top 10,000 HITAG peaks per TF, promoter / ABC-linked element / within 10 kb of TSS; Perturb-seq columns are annotations (top-25% sgRNA cells), never define a link"
E.to_csv(f"{O}/regulome_v4_top10k_10kb.tsv.gz", sep="\t", index=False, float_format="%.4g")
import sys; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps"); E.to_excel(f"{O}/regulome_v4_top10k_10kb.xlsx", index=False)
tpg = E.groupby("gene").tf.nunique(); gpt = E.groupby("tf").gene.nunique()
S = pd.DataFrame([{"links": len(E), "TFs": E.tf.nunique(), "genes": E.gene.nunique(), "median TFs per gene": float(tpg.median()), "median genes per TF": float(gpt.median()),
                   "promoter links": int((E.link_type == "promoter").sum()), "ABC links": int((E.link_type == "ABC").sum()), "within 10 kb links": int((E.link_type == "within_10kb").sum()),
                   "links also reached by coordinated binding": int(E.also_coordinated_binding_link.sum()), "coordinated-binding-only links (not in v4)": coord_only}])
S.to_csv(f"{O}/regulome_v4_summary.tsv", sep="\t", index=False); print(S.T.to_string()); print("done")
