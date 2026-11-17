# m34 (SPEC.md): frozen Regulome table and Figure 5 panel tables. Descriptive.
import sys, json
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA
X34 = "/work/users/d/y/dyy12/XuLab"; R34 = f"{X34}/regulatome"; O34 = f"{R34}/m34_fig5_panels_v3/out"; LAB = "descriptive"
TFL = list(TF); gi = {g: k for k, g in enumerate(UG)}; ti = {t: i for i, t in enumerate(TFL)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFL):
    for o in OWN.get(t, [t]):
        if o in gi: ZA[i, gi[o]] = Z1[i, gi[o]] = Z2[i, gi[o]] = np.nan
E = pd.read_csv(f"{R34}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", low_memory=False)
keep = ["tf", "gene", "promoter_peak", "promoter_signalValue", "n_distal_elements_with_peak", "best_element", "best_abc", "best_dist_to_tss", "score_signal_x_abc", "within_tf_percentile", "shared_by", "co_bound_tfs", "coordinated_binding_module", "own_motif_promoter", "own_motif_best_element", "peak_definition"]
E = E[keep].copy(); r = E.tf.map(ti); c = E.gene.map(gi); okk = r.notna() & c.notna()
z = np.full(len(E), np.nan); z1 = z.copy(); z2 = z.copy()
z[okk.values] = ZA[r[okk].astype(int).values, c[okk].astype(int).values]; z1[okk.values] = Z1[r[okk].astype(int).values, c[okk].astype(int).values]; z2[okk.values] = Z2[r[okk].astype(int).values, c[okk].astype(int).values]
E["knockdown_available"] = E.tf.isin(ti); E["knockdown_z"] = z; E["z_half1"] = z1; E["z_half2"] = z2
resp = np.abs(np.nan_to_num(z)) >= 2; rep = resp & (np.abs(np.nan_to_num(z1)) >= 2) & (np.abs(np.nan_to_num(z2)) >= 2) & (np.sign(z1) == np.sign(z2)) & (np.sign(z1) == np.sign(z))
E["tier"] = np.where(rep, 1, np.where(resp, 2, 3)); E["sign"] = np.where(resp & (z < 0), "activates", np.where(resp & (z > 0), "represses", "none"))
pp = E.promoter_peak.astype(str).str.lower().eq("true"); dd = E.n_distal_elements_with_peak > 0; E["peak_location"] = np.where(pp & dd, "promoter and distal", np.where(pp, "promoter", "distal"))
E["n_other_tfs_bound"] = E.shared_by - 1
# per-TF evidence
M = pd.read_csv(f"{R34}/m25_peak_threshold/out/choice_motif.tsv", sep="\t"); M = M[M.definition == "E2_top5000"].set_index("tf")
B = pd.read_csv(f"{R34}/m20_raw_binding_audit/out/part2_bulk_overlap.tsv", sep="\t").sort_values("own_rank_recall").drop_duplicates("own_tf").set_index("own_tf")
PB = pd.read_csv(f"{R34}/m33_abc_pipeline_audit/out/partB_bulk_vs_ours.tsv", sep="\t")
F2 = {"CTCF": .45, "ZIC2": .10, "ZNF282": .10, "ZNF777": .03, "ZBTB7A": 0.0, "MAZ": -.11}
M12 = {t: json.load(open(f"{R34}/m12_percell_ontarget/out/{t}_summary.json")) for t in ("CTCF", "NFYA", "NRF1", "ZNF143")}
tfs79 = sorted(set(E.tf)); EV = []
for t in tfs79:
    mx = M.own_excess.get(t, np.nan); b = B.loc[t] if t in B.index else None; m12 = M12.get(t)
    ours = PB[(PB.TF == t) & PB.source.str.startswith("our")]; bulk = PB[(PB.TF == t) & PB.source.str.startswith("bulk")]
    lines = [bool(np.isfinite(mx) and mx >= .05), bool(b is not None and min(b.own_rank_recall, b.own_rank_precision) <= 10),
             bool(m12 is not None and m12["motif_own_pooled_enrichment"] - m12["motif_swap_pooled_enrichment"] >= .1), bool(F2.get(t, -9) >= .3)]
    n = sum(lines)
    EV.append(dict(tf=t, tf_motif_excess_points=100 * mx if np.isfinite(mx) else np.nan, tf_bulk_chip_ref=(b.chip if b is not None else ""), tf_bulk_recall_rank_of79=(b.own_rank_recall if b is not None else np.nan),
                   tf_bulk_precision_rank_of79=(b.own_rank_precision if b is not None else np.nan), tf_ours_fold_same_path=(float(ours.fold_resp.iloc[0]) if len(ours) else np.nan),
                   tf_bulk_fold_same_path=(";".join(f"{s.replace('bulk ChIP ', '')}:{f:.2f}" for s, f in zip(bulk.source, bulk.fold_resp)) if len(bulk) else ""),
                   tf_own_motif_fragment_enrichment=(f"{m12['motif_own_pooled_enrichment']:.2f} vs swap {m12['motif_swap_pooled_enrichment']:.2f}" if m12 else ""), tf_fig2_r=F2.get(t, np.nan),
                   tf_evidence_lines=n, tf_evidence="multiple lines" if n >= 2 else ("one line" if n == 1 else "binding only")))
EV = pd.DataFrame(EV); EV.to_csv(f"{O34}/tf_evidence.tsv", sep="\t", index=False, float_format="%.3g")
R = E.merge(EV, on="tf", how="left"); R.insert(0, "label", "Regulome v1 (frozen); Perturb-seq columns external, descriptive")
R.to_csv(f"{O34}/regulome_v1.tsv.gz", sep="\t", index=False, float_format="%.4g")
print("Regulome v1:", len(R), "edges", R.tf.nunique(), "TFs", R.gene.nunique(), "genes | tiers", R.tier.value_counts().sort_index().to_dict(), "| signs", R.sign.value_counts().to_dict(), flush=True)
print(EV[EV.tf_evidence != "binding only"][["tf", "tf_evidence_lines", "tf_motif_excess_points", "tf_bulk_recall_rank_of79", "tf_bulk_precision_rank_of79"]].to_string(index=False))
# b
pb = R.groupby("tf").agg(n_targets=("gene", "size"), promoter=("peak_location", lambda v: int((v == "promoter").sum())), distal=("peak_location", lambda v: int((v == "distal").sum())),
                         both=("peak_location", lambda v: int((v == "promoter and distal").sum()))).reset_index().sort_values("n_targets", ascending=False)
pb.to_csv(f"{O34}/panel_b_targets_per_tf.tsv", sep="\t", index=False)
tpg = R.groupby("gene").tf.nunique(); pd.DataFrame({"n_tfs_bound": tpg.value_counts().sort_index().index, "n_genes": tpg.value_counts().sort_index().values}).to_csv(f"{O34}/panel_b_tfs_per_gene.tsv", sep="\t", index=False)
print("b: median TFs per gene", tpg.median(), "genes", len(tpg), "| location shares", R.peak_location.value_counts(normalize=True).round(3).to_dict(), flush=True)
# c
PB2 = PB.copy(); PB2["reference_cell_line"] = PB2.source.str.replace("bulk ChIP ", "").str.replace("our HITAG (top 5,000, ABC)", "HEK293T (ours, HITAG)")
PB2[["TF", "source", "reference_cell_line", "n_targets", "n_resp", "shared_resp", "expected_resp", "fold_resp", "top200_hits_resp", "top200_expected_resp", "n_rep", "shared_rep", "fold_rep", "top200_hits_rep", "top200_expected_rep"]].to_csv(f"{O34}/panel_c_ours_vs_bulk.tsv", sep="\t", index=False, float_format="%.3f")
# d NRF1
N = R[R.tf == "NRF1"]; sup = N[N.tier <= 2]
fun = dict(bound=len(N), measured=int(N.knockdown_z.notna().sum()), supported=len(sup), down_activates=int((sup.sign == "activates").sum()), up_represses=int((sup.sign == "represses").sum()), tier1=int((N.tier == 1).sum()),
           tier1_down=int(((N.tier == 1) & (N.sign == "activates")).sum()), supported_promoter=int(sup.peak_location.isin(["promoter", "promoter and distal"]).sum()), supported_distal_only=int((sup.peak_location == "distal").sum()),
           supported_with_own_motif=int((sup.own_motif_promoter.astype(str).eq("True") | sup.own_motif_best_element.astype(str).eq("True")).sum()))
pd.DataFrame([fun]).to_csv(f"{O34}/panel_d_nrf1_funnel.tsv", sep="\t", index=False); sup.sort_values("knockdown_z").to_csv(f"{O34}/panel_d_nrf1_supported_edges.tsv", sep="\t", index=False, float_format="%.4g")
loc = sup[(sup.tier == 1) & (sup.sign == "activates") & sup.peak_location.isin(["promoter", "promoter and distal"])].sort_values("knockdown_z").head(5)
print("d:", fun, "| locus candidates:", loc[["gene", "knockdown_z", "best_element", "promoter_signalValue", "own_motif_promoter"]].to_string(index=False), flush=True)
# e
for g in ("HSPB1", "ATP5MC2", "MDK"):
    x = R[R.gene == g][["gene", "tf", "peak_location", "promoter_signalValue", "best_element", "best_abc", "knockdown_available", "knockdown_z", "z_half1", "z_half2", "tier", "sign", "tf_evidence"]].sort_values(["tier", "knockdown_z"])
    x.to_csv(f"{O34}/panel_e_gene_{g}.tsv", sep="\t", index=False, float_format="%.3g"); print("e", g, "bound TFs", len(x), "changing", int((x.tier <= 2).sum()), ",".join(f"{a}:{b:.1f}" for a, b in zip(x.tf[x.tier <= 2], x.knockdown_z[x.tier <= 2])), flush=True)
# f
S = R[R.tier <= 2]; pf = S.groupby("tf").agg(supported=("gene", "size"), activates=("sign", lambda v: int((v == "activates").sum())), represses=("sign", lambda v: int((v == "represses").sum())),
                                            tier1=("tier", lambda v: int((v == 1).sum()))).reset_index()
t1 = R[R.tier == 1].groupby("tf").sign.value_counts().unstack(fill_value=0); pf["tier1_activates"] = pf.tf.map(t1.get("activates", pd.Series(dtype=int))).fillna(0).astype(int); pf["tier1_represses"] = pf.tf.map(t1.get("represses", pd.Series(dtype=int))).fillna(0).astype(int)
pf["pct_activates"] = 100 * pf.activates / pf.supported; pf = pf.merge(EV[["tf", "tf_evidence"]], on="tf").sort_values("pct_activates", ascending=False); pf.to_csv(f"{O34}/panel_f_activator_repressor.tsv", sep="\t", index=False, float_format="%.1f")
print("f: TFs", len(pf), "| pct activates range", round(pf.pct_activates.min(), 1), "-", round(pf.pct_activates.max(), 1), flush=True)
# g
C = pd.read_csv(f"{X34}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); strong = [(a, b) for a, b, k in zip(C.tf_a, C.tf_b, C.af3_5seed_tier) if k == "strong"]
pairs = [("MLX", "SREBF2"), ("ETS2", "RBBP7"), ("DR1", "NFYA")] + [p for p in strong if p not in [("MLX", "SREBF2"), ("ETS2", "RBBP7"), ("DR1", "NFYA")]]
BG = {t: set(R.gene[R.tf == t]) for t in tfs79}; FEW = {"SREBF2": "knockdown has 18 cells: too few, not used"}; G7 = []
for a, b in pairs:
    if a not in BG or b not in BG: continue
    sh = sorted(BG[a] & BG[b]); row = dict(pair=f"{a}-{b}", figure4_pair=(a, b) in [("MLX", "SREBF2"), ("ETS2", "RBBP7"), ("DR1", "NFYA")], af3_5seed_strong=(a, b) in strong or (b, a) in strong, n_bound_a=len(BG[a]), n_bound_b=len(BG[b]), n_shared_targets=len(sh))
    for side, t in (("a", a), ("b", b)):
        if t in FEW or t not in ti: row[f"kd_{side}"] = FEW.get(t, "no knockdown"); continue
        zz = np.array([ZA[ti[t], gi[g]] if g in gi else np.nan for g in sh]); zo = np.array([ZA[ti[t], gi[g]] if g in gi else np.nan for g in BG[t] - set(sh)])
        row[f"kd_{side}"] = f"{t} knockdown"; row[f"n_shared_responding_{side}"] = int((np.abs(np.nan_to_num(zz)) >= 2).sum()); row[f"pct_shared_responding_{side}"] = 100 * np.nanmean(np.abs(zz) >= 2) if np.isfinite(zz).any() else np.nan
        row[f"pct_own_only_targets_responding_{side}"] = 100 * np.nanmean(np.abs(zo) >= 2) if np.isfinite(zo).any() else np.nan; row[f"n_shared_down_{side}"] = int((zz <= -2).sum()); row[f"n_shared_up_{side}"] = int((zz >= 2).sum())
    G7.append(row)
G7 = pd.DataFrame(G7); G7.to_csv(f"{O34}/panel_g_pairs_shared_targets.tsv", sep="\t", index=False, float_format="%.2f"); print(G7.round(1).to_string(index=False))
gene_rows = []
for a, b in [("MLX", "SREBF2"), ("ETS2", "RBBP7"), ("DR1", "NFYA")]:
    for g in sorted(BG[a] & BG[b]):
        gene_rows.append(dict(pair=f"{a}-{b}", gene=g, z_a=(ZA[ti[a], gi[g]] if a in ti and a not in FEW and g in gi else np.nan), z_b=(ZA[ti[b], gi[g]] if b in ti and b not in FEW and g in gi else np.nan)))
pd.DataFrame(gene_rows).to_csv(f"{O34}/panel_g_figure4_pairs_genes.tsv", sep="\t", index=False, float_format="%.3f")
print("done")
