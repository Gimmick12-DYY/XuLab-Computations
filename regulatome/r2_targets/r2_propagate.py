# R2 propagation step 2. Sources: sources_atacfiltered/ (7 specific TFs, ZNF777 capped; 3 module pools). A target gains inferred targets from qualifying neighbours in frozen R1:
# complex, DNA-guided w 1.0; complex_weak, AF3-strong physical-only (structure), physical-only (DNA) w 0.5 provisional, capped at the 3 best by ipSAE per source (AF3 ipSAE where run, else Boltz); others 0.
# Module pools transfer at w 0.5 to members only, with members = the R1 module (PRC2 core pool to EZH2, MTF2, PCGF6, RBBP4, RBBP7; NuRD pool to RBBP4, RBBP7; WIZ/ZNF644 pool to its 6). MBD3 is left out of the NuRD transfer because the pool is MBD3 alone.
# w is normalised over each target's qualifying neighbours; inferred score = w_norm x source S_k1. locally_supported = the gene is in the target's own direct top 1,000; target_S_pos_count_ok is the looser check.
# A specific TF keeps its direct list as primary (direct_list_primary). Direct lists are not changed; inferred lists are written apart.
import glob, hashlib, os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R2 = f"{X}/regulatome/r2_targets"; O = f"{R2}/propagation"; F = f"{X}/regulatome/r1_tftf/v2_final"; os.makedirs(f"{O}/inferred_lists", exist_ok=True)
P = pd.read_csv(f"{F}/r1_pairs_frozen.tsv", sep="\t", keep_default_na=False, na_values=[""]); Tier = pd.read_csv(f"{R2}/r2_summary.tsv", sep="\t").set_index("tf").tier; tfs = list(Tier.index)
TF_SRC = ["RBBP4", "NFYA", "SOX4", "ZNF507", "MLX", "E4F1", "ZNF777"]
MOD = {"NuRD_MBD3core": ["RBBP4", "RBBP7"], "WIZ_ZNF644_broad_domain": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"], "PRC2_core": ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"]}
SRCFLAG = {"ZNF777": "shallow_source", "PRC2_core": "not_PRC2_enriched"}; UNC = ["complex_weak", "physical-only (structure)", "physical-only (DNA)"]
E = []
for s in TF_SRC:
    D = P[((P.a == s) | (P.b == s)) & (P.w > 0)].copy(); D["target"] = np.where(D.a == s, D.b, D.a); D["source"] = s; D["ips"] = D.af3_ipsae.fillna(D.boltz_ipsae).fillna(0.0)
    keep = D[D["class"].isin(UNC)].sort_values("ips", ascending=False).head(3); D2 = pd.concat([D[~D["class"].isin(UNC)], keep]); dropped = len(D) - len(D2)
    if dropped: print(s, "provisional edges over the cap dropped:", dropped)
    E.append(D2[["target", "source", "class", "w", "ips", "flags", "prc2_gene_edge_flag"]])
for m, mem in MOD.items():
    E.append(pd.DataFrame(dict(target=mem, source=m, **{"class": "module"}, w=0.5, ips=np.nan, flags=".", prc2_gene_edge_flag="not_PRC2_enriched" if m == "PRC2_core" else ".")))
E = pd.concat(E, ignore_index=True); E["w_norm"] = E.w / E.groupby("target").w.transform("sum"); E.to_csv(f"{O}/propagation_edges.tsv", sep="\t", index=False, float_format="%.4f")
def direct(t): return pd.read_csv(f"{R2}/per_tf/{t}.tsv.gz", sep="\t", usecols=["gene", "S_k1", "rank_k1", "count_ok"]).set_index("gene")
rows, summ = [], []
for t, D in E.groupby("target"):
    d = direct(t); top = set(d.index[d.rank_k1 <= 1000]); pos = set(d.index[(d.S_k1 > 0) & d.count_ok]); L = []
    for r in D.itertuples():
        f = f"{R2}/sources_atacfiltered/{r.source}.tsv"; S = pd.read_csv(f, sep="\t")
        if r.source == "MBD3": continue
        fl = ";".join(x for x in (r.flags, r.prc2_gene_edge_flag, SRCFLAG.get(r.source, ".")) if x != ".") or "."
        L.append(pd.DataFrame(dict(target=t, source=r.source, edge_class=getattr(r, "_3"), w=r.w, w_norm=round(r.w_norm, 4), gene=S.gene, inferred_score=np.round(r.w_norm * S.S_k1, 5), locally_supported=S.gene.isin(top), target_S_pos_count_ok=S.gene.isin(pos), flags_carried=fl, direct_list_primary=Tier[t] == "specific", inferred_only=True)))
    A = pd.concat(L); rows.append(A)
    G = A.groupby("gene").agg(inferred_score=("inferred_score", "sum"), n_sources=("source", "nunique"), sources=("source", lambda s: ",".join(sorted(set(s)))), locally_supported=("locally_supported", "first"), target_S_pos_count_ok=("target_S_pos_count_ok", "first")).reset_index().sort_values("inferred_score", ascending=False); G.to_csv(f"{O}/inferred_lists/{t}.tsv", sep="\t", index=False, float_format="%.5g")
    summ.append(dict(target=t, tier=Tier[t], direct_top1000_count_ok=int(d.loc[list(top)].count_ok.sum()), n_sources=D.source.nunique(), sources=",".join(D.source), inferred_targets=len(G), frac_locally_supported=round(G.locally_supported.mean(), 3), frac_S_pos_count_ok=round(G.target_S_pos_count_ok.mean(), 3)))
T = pd.concat(rows); T[["target", "source", "edge_class", "w", "w_norm", "gene", "inferred_score", "locally_supported", "target_S_pos_count_ok", "flags_carried", "direct_list_primary"]].to_csv(f"{O}/propagation_table.tsv.gz", sep="\t", index=False)
Sm = pd.DataFrame(summ).sort_values(["tier", "target"]); Sm.to_csv(f"{O}/propagation_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 100); pd.set_option("display.max_colwidth", 60); print(E.drop(columns=["ips"]).round(3).to_string(index=False)); print(Sm.to_string(index=False))
print("targets:", len(Sm), "by tier:", Sm.tier.value_counts().to_dict(), "of", Tier.value_counts().to_dict()); print("gene-level fraction locally supported:", round(np.average(Sm.frac_locally_supported, weights=Sm.inferred_targets), 3), "chance about", round(1000 / 19807, 3)); print("median inferred targets:", Sm.inferred_targets.median(), "rows", len(T))
fs = [f"{F}/r1_pairs_frozen.tsv", f"{F}/r1_edges_frozen.tsv", f"{F}/af3_vs_boltz.tsv", f"{R2}/propagation/propagation_edges.tsv", f"{R2}/propagation/propagation_table.tsv.gz", f"{R2}/propagation/propagation_summary.tsv", f"{R2}/r2_summary.tsv", f"{R2}/module_summary.tsv", f"{R2}/sources_atacfiltered_summary.tsv"]
for d in ("top1000", "top1000_marked", "top1000_modules", "sources", "sources_atacfiltered", "propagation/inferred_lists"): fs += sorted(glob.glob(f"{R2}/{d}/*.tsv"))
fs += sorted(glob.glob(f"{R2}/per_tf/*.tsv.gz"))
lines = [f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}" for p in fs]; open(f"{F}/frozen_manifest.sha256", "w").write("\n".join(lines) + "\n"); print("manifest", len(lines), "files; sha256 of manifest:", hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest())
