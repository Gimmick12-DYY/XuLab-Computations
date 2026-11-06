# R2 module pooled lists, built like the TF lists (r2_scores.py): y = members' reads pooled, panel leaves all members out; kappa 1 uses the union of
# the members' Cicero links. Writes top1000_modules/<module>.tsv and module_summary.tsv with overlap vs the ATAC-only top 1,000. Also writes
# sources/<TF>.tsv: the 7 specific TFs' propagation-source lists; ZNF777 is capped to genes at expected >= 5 (source_flag shallow_source), and
# top1000_marked/ZNF777.tsv keeps its full direct list with low_support marks.
import os, sys
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_scores.py")).read().split("y = Y[i];")[0]
sys.argv = ["r2_scores.py", "tf", "MBD3"]; exec(src)
for d in ("top1000_modules", "sources", "top1000_marked"): os.makedirs(f"{O}/{d}", exist_ok=True)
MODS = {"NuRD_MBD3core": (["MBD3"], "."), "WIZ_ZNF644_broad_domain": (["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"], "."),
        "PRC2_core": (["EZH2", "MTF2", "PCGF6"], "not_PRC2_enriched")}
Sa = W.T @ np.log1p(atac_cpm); rows = []
for name, (mem, flag) in MODS.items():
    ix = [tfs.index(m) for m in mem]; y = Y[ix].sum(0); N = rd[ix].sum()
    pc = (Y.sum(0) - y) / (rd.sum() - N) * 1e6
    Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp])
    mu = glm_mu(y, np.full(nE, np.log(N / 1e6)), Xd)
    B = np.log2((y + 0.5) / (mu + 0.5)); S0 = W.T @ B; Cu = None
    for m in mem:
        t = m; c = cicero_W()
        if c is not None: Cu = c if Cu is None else Cu + c
    if Cu is None: S1, nc = S0.copy(), 0
    else:
        Cu.data[:] = 1.0; gain = W.multiply(Cu).tocsr(); S1 = S0 + gain.T @ B; nc = gain.nnz
    T = pd.DataFrame(dict(gene=genes, S_k1=S1, S_k0=S0, expected=L.T @ mu, reads=L.T @ y, S_atac=Sa)); T["count_ok"] = T.expected >= 5
    T["rank_k1"] = T.S_k1.rank(ascending=False, method="first").astype(int); T["rank_k0"] = T.S_k0.rank(ascending=False, method="first").astype(int); T["flag"] = flag
    a = T.nsmallest(1000, "rank_k1"); a[["gene", "rank_k1", "S_k1", "rank_k0", "S_k0", "expected", "count_ok", "flag"]].to_csv(f"{O}/top1000_modules/{name}.tsv", sep="\t", index=False, float_format="%.5g")
    at = set(T.nlargest(1000, "S_atac").gene); k0 = set(T.nsmallest(1000, "rank_k0").gene)
    rows.append(dict(module=name, members=",".join(mem), reads=int(N), flag=flag, cicero_links=nc, S_pos=int((S1 > 0).sum()), count_ok=int(T.count_ok.sum()),
                     list_pos_countok=int(((S1 > 0) & T.count_ok).sum()), top1000_count_ok=int(a.count_ok.sum()), top1000_ovl_atac=len(set(a.gene) & at),
                     chance=round(1e6 / len(T), 1), k1_vs_k0_shared=len(set(a.gene) & k0), spearman_S_atac=round(T.S_k1.corr(T.S_atac, method="spearman"), 3)))
M = pd.DataFrame(rows); M.to_csv(f"{O}/module_summary.tsv", sep="\t", index=False); pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); print(M.to_string(index=False))
for q in ["RBBP4", "NFYA", "SOX4", "ZNF507", "MLX", "E4F1", "ZNF777"]:
    a = pd.read_csv(f"{O}/top1000/{q}.tsv", sep="\t"); a["source_flag"] = "."
    if q == "ZNF777":
        d = a.copy(); d["support"] = np.where(d.count_ok, ".", "low_support"); d.to_csv(f"{O}/top1000_marked/ZNF777.tsv", sep="\t", index=False, float_format="%.5g")
        a = a[a.count_ok].copy(); a["source_flag"] = "shallow_source"
    a.to_csv(f"{O}/sources/{q}.tsv", sep="\t", index=False, float_format="%.5g"); print(q, "source genes", len(a))
