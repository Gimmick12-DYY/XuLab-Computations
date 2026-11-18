# m45: Perturb-seq gene and TF counts per cutoff (all 63 / 55 non-broad / 8 broad), and Regulome size per padding over the 63 TFs.
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{R}/m45_stage1_summary/out"
UG = np.load(f"{R}/m9_abc_diagnosis/work/responses_arm2.npz", allow_pickle=True)["genes"].astype(str); gi = {g: k for k, g in enumerate(UG)}
T25 = pd.read_csv(f"{R}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); LS = pd.read_csv(f"{O}/list_sizes.tsv", sep="\t", dtype={"padding": str})
TFS = sorted(set(T25.tf) & set(LS.TF)); assert len(TFS) == 63
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float); OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFS):
    for o in OWN.get(t, [t]):
        if o in gi: Z[i, gi[o]] = np.nan
OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z)); BROAD = set(LS[(LS.padding == "0") & LS["broad TF"]].TF)
rows = []
for vn, sel in (("all 63 TFs", np.ones(63, bool)), ("without broad TFs (55)", np.array([t not in BROAD for t in TFS])), ("broad TFs only (8)", np.array([t in BROAD for t in TFS]))):
    for c in (2, 3, 4, 5):
        n = (OK & (AZ >= c)).sum(1)[sel]
        rows.append({"TFs": vn, "cutoff": f"|z| >= {c}", "Perturb-seq genes (TF-gene pairs)": int(n.sum()), "TFs with >= 1 gene": int((n >= 1).sum()), "TFs with >= 10 genes": int((n >= 10).sum()), "median Perturb-seq genes per TF": float(np.median(n))})
C = pd.DataFrame(rows); C.to_csv(f"{O}/9_counts_per_cutoff.tsv", sep="\t", index=False); print(C.to_string(index=False))
P1 = pd.read_csv(f"{O}/1_recall_pooled.tsv", sep="\t", dtype={"padding": str}); g = P1[(P1.TFs == "all 63 TFs") & (P1.cutoff == "|z| >= 2")]
RS = pd.DataFrame({"padding": g.padding, "Regulome TF-gene links over the 63 TFs": g["our genes"], "median Regulome genes per TF": g["median our genes per TF"]}); RS.to_csv(f"{O}/10_regulome_size.tsv", sep="\t", index=False); print(RS.to_string(index=False))
