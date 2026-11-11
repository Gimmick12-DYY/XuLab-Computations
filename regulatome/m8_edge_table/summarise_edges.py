# m8_edge_table: section 8 summaries and section 9 lookups from edges/edge_table_<link>.tsv.gz for both link sets.
# usage: summarise_edges.py -> summaries.tsv (section, link, key, value), per_tf_summary.tsv, lookups.tsv
import numpy as np, pandas as pd
from scipy.stats import spearmanr
O = "/work/users/d/y/dyy12/XuLab/regulatome/m8_edge_table"; S = []; PT = []; LK = []
def add(link, sec, key, val): S.append(dict(section=sec, link=link, key=key, value=val))
for link in ("hicfixed", "powerlaw"):
    E = pd.read_csv(f"{O}/edges/edge_table_{link}.tsv.gz", sep="\t", keep_default_na=False, na_values=[""])
    E["called"] = E.called.astype(str).eq("True"); E["in_v8"] = E.in_v8.astype(str).eq("True")
    gs = E.drop_duplicates("gene").set_index("gene"); sb = gs.shared_by
    T = E.groupby("tf").agg(n_called=("called", "sum"), cells=("cells", "first"), tertile=("tertile", "first"), d=("d", "first"), tier=("tier", "first")).reset_index()
    T["frac_called_shared_le5"] = [float((E[(E.tf == t) & E.called].shared_by <= 5).mean()) if n else np.nan for t, n in zip(T.tf, T.n_called)]
    T["v8_overlap_n"] = [int((E[(E.tf == t)].called & E[(E.tf == t)].in_v8).sum()) for t in T.tf]; T["v8_list_in_universe"] = [int(E[E.tf == t].in_v8.sum()) for t in T.tf]
    T["v8_overlap_frac_of_called"] = np.where(T.n_called > 0, T.v8_overlap_n / T.n_called.clip(lower=1), np.nan); T.insert(0, "link", link); PT.append(T)
    r, p = spearmanr(T.n_called, T.cells); add(link, "a", "spearman_ncalled_vs_cells_all79", f"{r:.3f} (p {p:.2g})")
    H = T[T.tertile == "high"]; r, p = spearmanr(H.n_called, H.cells); add(link, "a", f"spearman_ncalled_vs_cells_high_tertile_n{len(H)}", f"{r:.3f} (p {p:.2g})")
    add(link, "a", "n_called_per_tf_median_min_max", f"{T.n_called.median():.0f} ({T.n_called.min()}-{T.n_called.max()})"); add(link, "a", "tfs_with_0_called", int((T.n_called == 0).sum())); add(link, "a", "total_called_edges", int(T.n_called.sum()))
    q = np.percentile(sb, [0, 10, 25, 50, 75, 90, 99, 100]); add(link, "b", "shared_by_quantiles_0_10_25_50_75_90_99_100", " ".join(f"{x:.0f}" for x in q)); add(link, "b", "genes_shared_by_ge40", int((sb >= 40).sum()))
    add(link, "b", "genes_shared_by_1to5", int(((sb >= 1) & (sb <= 5)).sum())); add(link, "b", "genes_shared_by_0", int((sb == 0).sum())); add(link, "b", "genes_total", int(len(sb)))
    f = T.frac_called_shared_le5.dropna(); add(link, "c", "frac_called_shared_le5_median_min_max", f"{f.median():.3f} ({f.min():.3f}-{f.max():.3f}) over {len(f)} TFs with >=1 called")
    v = T.v8_overlap_n; add(link, "d", "v8_overlap_n_median_min_max", f"{v.median():.0f} ({v.min()}-{v.max()})"); vf = T.v8_overlap_frac_of_called.dropna(); add(link, "d", "v8_overlap_frac_of_called_median_min_max", f"{vf.median():.3f} ({vf.min():.3f}-{vf.max():.3f})")
    # section 9 lookups
    TG = {"E4F1": ["DLAT", "DLD", "MPC1", "PDPR", "CHEK1"], "MLX": ["TXNIP", "ARRDC4"], "NFYA": ["CCNB1", "CDK1"], "E2F3": ["CCNE1", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"]}
    for t, gl in TG.items():
        for g in gl:
            q_ = E[(E.tf == t) & (E.gene == g)]
            if not len(q_): LK.append(dict(link=link, kind="tf_to_gene", tf=t, gene=g, note="gene not in universe")); continue
            x = q_.iloc[0]; LK.append(dict(link=link, kind="tf_to_gene", tf=t, gene=g, promoter=x.promoter, distal=x.distal, total=x.total, rank_in_tf=x.rank_in_tf, p_total=x.p_total, q_total=x.q_total, called=x.called, shared_by=x.shared_by, shared_by_deep=x.shared_by_deep))
    for g in ["EZH2", "MYC", "CCNE1", "CDK1", "CCNB1", "TXNIP", "LDLR", "HMGCR", "CDKN1A", "CDKN2A", "ACTB", "GAPDH"]:
        q_ = E[(E.gene == g)]
        if not len(q_): LK.append(dict(link=link, kind="gene_to_tfs", gene=g, note="gene not in universe")); continue
        c = q_[q_.called].sort_values("q_total"); LK.append(dict(link=link, kind="gene_to_tfs", gene=g, shared_by=int(q_.shared_by.iloc[0]), shared_by_deep=int(q_.shared_by_deep.iloc[0]), called_tfs="; ".join(f"{r.tf} ({r.tier}, rank {r.rank_in_tf:.1f}, q {r.q_total:.3g})" for r in c.itertuples())))
pd.DataFrame(S).to_csv(f"{O}/summaries.tsv", sep="\t", index=False); P = pd.concat(PT); P.to_csv(f"{O}/per_tf_summary.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(LK).to_csv(f"{O}/lookups.tsv", sep="\t", index=False, float_format="%.4g")
print(pd.DataFrame(S).to_string(index=False))
