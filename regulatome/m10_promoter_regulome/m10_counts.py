# m10_promoter_regulome step 1: per-TF promoter-window binding (diagnostic / development, benchmark already read).
# usage: m10_counts.py <task index 0-78>  -> work/counts/<TF>.npz with raw 5'-end counts in TSS +/- 500 bp and +/- 1 kb for the m9 promoter genes,
# the TF's total reads, reads within TSS +/- 1 kb of any GeneList gene, reads in canonical candidate elements, and imputed signal transferred
# by overlap fraction from panel elements to the same windows.
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m10_promoter_regulome"; os.makedirs(f"{O}/work/counts", exist_ok=True)
tfs = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; t = tfs[int(sys.argv[1])]
P = pd.read_csv(f"{RG}/m9_abc_diagnosis/work/promoters.bed", sep="\t", header=None, names=["c", "s", "e", "gene", "x", "strand"]); P["tss"] = P.s + 500
GL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "tss"])
E = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"])
F = pd.read_csv(f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "st": str})
pos = np.where(F.st.values == "+", F.s.values, F.e.values - 1); POS = {c: np.sort(pos[ix]) for c, ix in F.groupby("c", sort=False).indices.items()}
def win(half):
    out = np.zeros(len(P))
    for c, G in P.groupby("c"):
        if c not in POS: continue
        p = POS[c]; out[G.index.values] = np.searchsorted(p, G.tss.values + half, side="left") - np.searchsorted(p, G.tss.values - half, side="left")
    return out
w500, w1k = win(500), win(1000)
# reads within TSS +/- 1 kb of any GeneList gene, and within canonical candidate elements (union intervals)
def union_count(df):
    n = 0
    for c, G in df.groupby("c"):
        if c not in POS: continue
        ivl = G.sort_values("s")[["s", "e"]].values; ms = []
        for s, e in ivl:
            if ms and s <= ms[-1][1]: ms[-1][1] = max(ms[-1][1], e)
            else: ms.append([s, e])
        ms = np.array(ms); p = POS[c]; n += int((np.searchsorted(p, ms[:, 1], side="left") - np.searchsorted(p, ms[:, 0], side="left")).sum())
    return n
prom_all = union_count(pd.DataFrame(dict(c=GL.chr, s=GL.tss - 1000, e=GL.tss + 1000)))
elem_all = union_count(pd.DataFrame(dict(c=E.chr, s=E.start, e=E.end)))
# imputed: panel elements (m0 elements_v2 active), signal summed over cells, transferred to windows by overlap fraction of each panel element
E_ = pd.read_csv(f"{RG}/m0_universe/elements_v2.bed", sep="\t", header=None, names=["c", "s", "e", "eid", "type", "pid", "ptype", "cls"]); act = E_.cls.isin(["promoter", "distal_active", "bivalent"]).values; Pn = E_[act].reset_index(drop=True)
iv = np.asarray(sp.load_npz(f"{RG}/gate1/cellmats/{t}.imp.npz").sum(0)).ravel().astype(float); iv = iv / max(iv.sum(), 1e-12) * 1e6
def imp_win(half):
    out = np.zeros(len(P))
    for c, G in P.groupby("c"):
        H = Pn[Pn.c == c]
        if not len(H): continue
        o = np.argsort(H.s.values); hs, he, hv = H.s.values[o], H.e.values[o], iv[H.index.values[o]]
        for i, tss in zip(G.index.values, G.tss.values):
            a, b = tss - half, tss + half; k0 = max(np.searchsorted(hs, a, side="right") - 1, 0); k = k0; v = 0.0
            while k < len(hs) and hs[k] < b:
                ov = min(he[k], b) - max(hs[k], a)
                if ov > 0: v += hv[k] * ov / (he[k] - hs[k])
                k += 1
            out[i] = v
    return out
np.savez(f"{O}/work/counts/{t}.npz", genes=P.gene.values, w500=w500, w1k=w1k, imp500=imp_win(500), imp1k=imp_win(1000), total=len(F), in_tss1k_any=prom_all, in_canonical_elements=elem_all)
print(t, len(F), int(w500.sum()), int(w1k.sum()), prom_all, elem_all, flush=True)
