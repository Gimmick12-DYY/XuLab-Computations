# PREVIEW only, outside the frozen manifest: what a cis-coordinated transfer class would add. No frozen table is changed, and no knockdown data is read.
# Cis-coordinated pair: Cicero shared-edge excess z at stratified percentile >= 0.90 AND >= 1 other view-1 metric (domain-corrected Pearson, non-quiescent SpQN, compartment SpQN) >= 0.90; STAG2-wrap and NEUROD4 edges excluded.
# Transfer: w 0.25 from an ATAC-filtered source: a specific TF partner, or a module pool when the partner is a pool member and the recipient is not. Required: the recipient's own residual Btilde > 0 on >= 1 ABC element of the gene.
# Looser check: fraction of transferred genes in the recipient's direct top 1,000, against ~5% and against the fraction of ALL genes passing the required filter that are in that top 1,000 (the filter alone enriches for it).
import os, sys
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2_scores.py")).read().split("y = Y[i];")[0]
sys.argv = ["r2_scores.py", "tf", "MBD3"]; exec(src)
OUT = f"{O}/propagation_ciscoord_preview"; os.makedirs(f"{OUT}/lists", exist_ok=True)
F = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""]); M_ = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t", usecols=["a", "b", "pearson_domain_pct", "spqn_noquies_pct", "cicero_excess_z_pct", "compartment_spqn_pct"])
D = F.merge(M_, on=["a", "b"], validate="one_to_one"); oth = D[["pearson_domain_pct", "spqn_noquies_pct", "compartment_spqn_pct"]]
D["cis"] = (D.cicero_excess_z_pct >= 0.90) & (oth >= 0.90).any(axis=1) & ~D["class"].str.startswith("STAG2") & ~(D.a.eq("NEUROD4") | D.b.eq("NEUROD4"))
C = D[D.cis].copy(); C[["a", "b", "class", "w", "flags", "cicero_excess_z_pct", "pearson_domain_pct", "spqn_noquies_pct", "compartment_spqn_pct", "gpair"]].to_csv(f"{OUT}/cis_coordinated_pairs.tsv", sep="\t", index=False, float_format="%.3f")
Tier = pd.read_csv(f"{O}/r2_summary.tsv", sep="\t").set_index("tf").tier
TF_SRC = ["RBBP4", "NFYA", "SOX4", "ZNF507", "MLX", "E4F1", "ZNF777"]; POOL = {"NuRD_MBD3core": ["MBD3"], "WIZ_ZNF644_broad_domain": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"], "PRC2_core": ["EZH2", "MTF2", "PCGF6"]}
E = []
for r in C.itertuples():
    for t, p in ((r.a, r.b), (r.b, r.a)):
        if p in TF_SRC: E.append(dict(target=t, source=p, via=p, source_kind="TF", pair_class=r._3))
        for m, mem in POOL.items():
            if p in mem and t not in mem: E.append(dict(target=t, source=m, via=p, source_kind="module", pair_class=r._3))
E = pd.DataFrame(E).drop_duplicates(["target", "source"]); E.to_csv(f"{OUT}/preview_edges.tsv", sep="\t", index=False)
def own_pos(t):
    i = tfs.index(t); y = Y[i]; pc = (Y.sum(0) - y) / (rd.sum() - rd[i]) * 1e6
    Xd = np.column_stack([np.ones(nE), np.log1p(atac_cpm), np.log1p(pc), logw, gc, gc ** 2, mp]); mu = glm_mu(y, np.full(nE, np.log(rd[i] / 1e6)), Xd); B = np.log2((y + 0.5) / (mu + 0.5))
    return np.asarray(L.T @ (B > 0).astype(float)).ravel() > 0
rows = []
for scope, Ex in (("TF_and_module", E), ("TF_only", E[E.source_kind == "TF"])):
    for t, G in Ex.groupby("target"):
        ok = pd.Series(own_pos(t), index=genes); d = pd.read_csv(f"{O}/per_tf/{t}.tsv.gz", sep="\t", usecols=["gene", "rank_k1"]).set_index("gene"); top = set(d.index[d.rank_k1 <= 1000])
        cand = set()
        for s in G.source: cand |= set(pd.read_csv(f"{O}/sources_atacfiltered/{s}.tsv", sep="\t").gene)
        cand -= {t}; keep = sorted(g for g in cand if ok.get(g, False)); pas = set(ok.index[ok.values])
        if scope == "TF_and_module": pd.DataFrame(dict(target=t, gene=keep, w=0.25, in_direct_top1000=[g in top for g in keep])).to_csv(f"{OUT}/lists/{t}.tsv", sep="\t", index=False)
        rows.append(dict(scope=scope, target=t, tier=Tier[t], n_sources=G.source.nunique(), sources=",".join(sorted(G.source)), candidates=len(cand), kept=len(keep), frac_kept=round(len(keep) / max(len(cand), 1), 3), genes_passing_filter=len(pas), frac_in_direct_top1000=round(np.mean([g in top for g in keep]), 3) if keep else np.nan, baseline_top1000_given_filter=round(len(top & pas) / max(len(pas), 1), 3)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/preview_summary.tsv", sep="\t", index=False)
fz = set(pd.read_csv(f"{O}/propagation/propagation_summary.tsv", sep="\t").target); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 120); pd.set_option("display.max_colwidth", 46)
print("cis-coordinated pairs:", len(C), "| by class:", C["class"].value_counts().to_dict(), "| already w > 0:", int((C.w > 0).sum()), "| with a preview edge:", E.shape[0], "edges")
for sc in ("TF_and_module", "TF_only"):
    S = R[(R.scope == sc) & (R.kept > 0)]; w = S.kept
    print(sc, "TFs gaining:", len(S), "median list:", S.kept.median(), "range", S.kept.min(), S.kept.max(), "| by tier:", S.tier.value_counts().to_dict(), "| sparse:", ",".join(sorted(S.target[S.tier == "undetermined"])), "| frac in direct top1000:", round(np.average(S.frac_in_direct_top1000, weights=w), 3), "baseline given filter:", round(np.average(S.baseline_top1000_given_filter, weights=w), 3), "| combined with frozen:", len(set(S.target) | fz), "of 79; new TFs:", ",".join(sorted(set(S.target) - fz)))
print(R[R.scope == "TF_and_module"].sort_values(["tier", "target"]).drop(columns="scope").to_string(index=False))
print(C[["a", "b", "class", "w", "gpair"]].to_string(index=False)[:3500])
