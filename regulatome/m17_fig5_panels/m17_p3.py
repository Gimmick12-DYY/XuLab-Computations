# m17_fig5_panels P3 (SPEC.md): gene example candidates from m15 confident pairs. Diagnostic, benchmark already read.
# usage: m17_p3.py -> out/p3_multi_tf_genes_<reading>_K<k>.tsv, out/p3_counts.tsv, out/p3_top20.tsv, out/p3_declared_genes.tsv
import glob
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; M15 = f"{RG}/m15_gene_to_tf/out"; OUT = f"{RG}/m17_fig5_panels/out"; LAB = "diagnostic, benchmark already read"
Ws = pd.read_csv(f"{RG}/m11_peak_regulome/out/wide_score_q0.05.tsv.gz", sep="\t", index_col=0); SH = (Ws > 0).sum(0)  # shared_by: TFs (of 79) with a q005 peak linked to the gene
cos = [f for f in glob.glob(f"{X}/data/**/*", recursive=True) if "cosmic" in f.lower() or "census" in f.lower()]; COSMIC = None if not cos else set(pd.read_csv(cos[0], sep=None, engine="python").iloc[:, 0].astype(str))
print("COSMIC local copy:", cos[:1] or "none")
CNT = []; TOP = None
for rd in ("dc", "raw"):
    for K in (2000, 10000):
        P = pd.read_csv(f"{M15}/{rd}_pairs_K{K}.tsv", sep="\t").drop_duplicates(["gene", "tf"])
        P["direction_on_kd"] = np.where(P.z_all < 0, "down", "up"); P["half_agree"] = np.sign(P.z_half1) == np.sign(P.z_half2)
        P["shared_by"] = P.gene.map(SH).fillna(0).astype(int); P["cosmic_cgc"] = "not available locally" if COSMIC is None else P.gene.isin(COSMIC)
        P["raw_ge75"] = P.pct_b_raw_promoter_reads > .75; P["peak_ge75"] = P.pct_a_peak_score_q005 > .75
        g = P.groupby("gene").agg(n_tfs=("tf", "nunique"), n_raw_ge75=("raw_ge75", "sum"), n_peak_ge75=("peak_ge75", "sum"), sum_absz=("z_all", lambda v: np.abs(v).sum()), tfs=("tf", ",".join))
        multi = g[g.n_tfs >= 2]
        CNT.append(dict(label=LAB, reading=rd, K=K, n_pairs=len(P), n_genes=P.gene.nunique(), genes_1tf=int((g.n_tfs == 1).sum()), genes_2tf=int((g.n_tfs == 2).sum()), genes_3tf=int((g.n_tfs == 3).sum()), genes_4plus=int((g.n_tfs >= 4).sum()),
                        multi_with_ge2_raw_ge75=int((multi.n_raw_ge75 >= 2).sum()), multi_with_ge2_peak_ge75=int((multi.n_peak_ge75 >= 2).sum())))
        cols = ["gene", "tf", "z_half1", "z_half2", "z_all", "direction_on_kd", "half_agree", "pct_b_raw_promoter_reads", "pct_a_peak_score_q005", "promoter_peak_q005", "own_motif_at_promoter", "n_linked_elements_with_peak_q005", "shared_by", "cosmic_cgc"]
        MT = P[P.gene.isin(multi.index)][cols].merge(multi[["n_tfs", "n_raw_ge75", "n_peak_ge75", "sum_absz"]], left_on="gene", right_index=True).sort_values(["n_raw_ge75", "sum_absz", "gene", "z_all"], ascending=[False, False, True, True])
        MT.insert(0, "label", LAB); MT.to_csv(f"{OUT}/p3_multi_tf_genes_{rd}_K{K}.tsv", sep="\t", index=False, float_format="%.4g")
        if rd == "dc" and K == 2000:
            top = multi.sort_values(["n_raw_ge75", "sum_absz"], ascending=False).head(20).index; TOP = MT[MT.gene.isin(top)]
C = pd.DataFrame(CNT); C.to_csv(f"{OUT}/p3_counts.tsv", sep="\t", index=False); print(C.drop(columns="label").to_string(index=False))
TOP.to_csv(f"{OUT}/p3_top20.tsv", sep="\t", index=False, float_format="%.4g")
print(TOP[["gene", "tf", "z_half1", "z_half2", "pct_b_raw_promoter_reads", "pct_a_peak_score_q005", "promoter_peak_q005", "own_motif_at_promoter", "shared_by", "n_tfs", "n_raw_ge75"]].round(2).to_string(index=False))
DECL = ["MYC", "EZH2", "CCNE1", "TXNIP", "LDLR", "CDKN1A"]
# the m15 reverse lookup holds only confident-pair genes, so rebuild the full per-gene view from m15's own matrix code (dc reading)
import sys; sys.argv = ["m15.py", "dc"]; src = open(f"{RG}/m15_gene_to_tf/m15.py").read(); _OUT = OUT; exec(src[:src.index("# ---- matched-gene strata")]); OUT = _OUT
CONF = {K: set(map(tuple, pd.read_csv(f"{M15}/dc_pairs_K{K}.tsv", sep="\t")[["gene", "tf"]].values)) for K in (2000, 10000)}; RR = []
for gname in DECL:
    if gname not in ug: RR.append(dict(label=LAB, gene=gname, note="not in the gene universe (>= 10% NTC detection and ABC gene list)")); continue
    j = ug[gname]; za = ZA[:, j]; o = np.argsort(-np.nan_to_num(np.abs(za), nan=-1))
    for r, i in enumerate(o):
        RR.append(dict(label=LAB, gene=gname, kd_rank=r + 1, tf=TF[i], z_all=za[i], z_half1=Z1[i, j], z_half2=Z2[i, j], half_agree=bool(np.sign(Z1[i, j]) == np.sign(Z2[i, j])), pct_b_raw_promoter_reads=PCT["b_raw_promoter_reads"][i, j],
                       pct_a_peak_score_q005=PCT["a_peak_score_q005"][i, j], peak_linked_q005=BIND["a_peak_score_q005"][i, j] > 0, confirmed_K2000=(gname, TF[i]) in CONF[2000], confirmed_K10000=(gname, TF[i]) in CONF[10000], shared_by=int(SH.get(gname, 0))))
R = pd.DataFrame(RR); R.to_csv(f"{OUT}/p3_declared_genes.tsv", sep="\t", index=False, float_format="%.4g")
for g_, H in R.groupby("gene", sort=False):
    print(H.head(6).drop(columns="label").round(2).to_string(index=False))
