# m34 v2 (m35 SPEC Amendment 2): regulome_v2 and panel tables on the top-25% sgRNA matrix; recovery_top25. Descriptive.
import sys, json
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])
R2 = "/work/users/d/y/dyy12/XuLab/regulatome"; O2 = f"{R2}/m34_fig5_panels_v3/out_v2"; import os; os.makedirs(O2, exist_ok=True); S35 = f"{R2}/m35_threshold_sweep/out"
gi = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
T25 = pd.read_csv(f"{R2}/m23_sgrna_top25/out/observed_kd_top25.tsv.gz", sep="\t", usecols=["tf", "gene", "z_top25"]); TFS = sorted(set(T25.tf)); ti = {t: i for i, t in enumerate(TFS)}
Z = T25.pivot(index="tf", columns="gene", values="z_top25").reindex(index=TFS, columns=UG).values.astype(float)
for t in TFS:
    for o in OWN.get(t, [t]):
        if o in gi: Z[ti[t], gi[o]] = np.nan
TFA = list(TF); tia = {t: i for i, t in enumerate(TFA)}
for i, t in enumerate(TFA):
    for o in OWN.get(t, [t]):
        if o in gi: ZA[i, gi[o]] = np.nan
cells = pd.read_csv(f"{R2}/m23_sgrna_top25/out/part2_per_tf.tsv", sep="\t"); cells = cells[cells.version == "top25_own"].set_index("tf").cells
V1 = pd.read_csv(f"{R2}/m34_fig5_panels_v3/out/regulome_v1.tsv.gz", sep="\t", low_memory=False)
drop = ["knockdown_available", "knockdown_z", "z_half1", "z_half2", "tier", "sign"]; R = V1.drop(columns=[c for c in drop if c in V1.columns]).copy()
r = R.tf.map(ti); c = R.gene.map(gi); ok = r.notna() & c.notna(); z = np.full(len(R), np.nan); z[ok.values] = Z[r[ok].astype(int).values, c[ok].astype(int).values]
ra = R.tf.map(tia); oka = ra.notna() & c.notna(); za = np.full(len(R), np.nan); za[oka.values] = ZA[ra[oka].astype(int).values, c[oka].astype(int).values]
a = np.abs(np.nan_to_num(z)); R["knockdown_available"] = R.tf.isin(ti) & R.tf.map(cells).fillna(0).ge(25); R["top25_cells"] = R.tf.map(cells)
R["knockdown_z"] = z; R["supported"] = a >= 2; R["direction"] = np.where(a >= 2, np.where(z < 0, "activates", "represses"), "none")
R["confidence"] = np.where(a >= 4, "|z| >= 4", np.where(a >= 3, "|z| >= 3", np.where(a >= 2, "|z| >= 2", "none"))); R["knockdown_z_all_cells"] = za
R["label"] = "Regulome v2 (frozen binding; Perturb-seq columns from top-25% sgRNA cells, external, descriptive)"
R.to_csv(f"{O2}/regulome_v2.tsv.gz", sep="\t", index=False, float_format="%.4g")
print("v2:", len(R), "edges | supported", int(R.supported.sum()), "| activates", int((R.direction == "activates").sum()), "represses", int((R.direction == "represses").sum()), "| |z|>=4", int((R.confidence == "|z| >= 4").sum()), "| TFs with knockdown", R[R.knockdown_available].tf.nunique(), flush=True)
# internal aliases used by the plotting code: tier 1 = |z| >= 4, 2 = 2 <= |z| < 4, 3 = not supported; sign
Q = R.copy(); Q["tier"] = np.where(a >= 4, 1, np.where(a >= 2, 2, 3)); Q["sign"] = Q.direction.where(Q.direction != "none", "none"); Q.to_csv(f"{O2}/regulome_v2_plotting.tsv.gz", sep="\t", index=False, float_format="%.4g")
# d NRF1
N = Q[Q.tf == "NRF1"]; sup = N[N.tier <= 2]
fun = dict(bound=len(N), measured=int(N.knockdown_z.notna().sum()), supported=len(sup), down_activates=int((sup.sign == "activates").sum()), up_represses=int((sup.sign == "represses").sum()), tier1=int((N.tier == 1).sum()))
pd.DataFrame([fun]).to_csv(f"{O2}/panel_d_nrf1_funnel.tsv", sep="\t", index=False); sup.sort_values("knockdown_z").to_csv(f"{O2}/panel_d_nrf1_supported_edges.tsv", sep="\t", index=False, float_format="%.4g")
loc = sup[sup.sign.eq("activates") & sup.peak_location.isin(["promoter", "promoter and distal"])].sort_values("knockdown_z").head(5); print("d:", fun, loc[["gene", "knockdown_z", "knockdown_z_all_cells"]].to_string(index=False), flush=True)
# e
for g in ("HSPB1", "ATP5MC2", "MDK", "PGAM1"):
    x = Q[Q.gene == g][["gene", "tf", "peak_location", "promoter_signalValue", "best_element", "best_abc", "knockdown_available", "knockdown_z", "knockdown_z_all_cells", "tier", "sign", "confidence", "tf_evidence"]].sort_values(["tier", "knockdown_z"])
    x.to_csv(f"{O2}/panel_e_gene_{g}.tsv", sep="\t", index=False, float_format="%.3g"); print("e", g, "bound", len(x), "changing", int((x.tier <= 2).sum()), ",".join(f"{a_}:{b_:.1f}" for a_, b_ in zip(x.tf[x.tier <= 2], x.knockdown_z[x.tier <= 2])), flush=True)
# f
S = Q[Q.tier <= 2]; pf = S.groupby("tf").agg(supported=("gene", "size"), activates=("sign", lambda v: int((v == "activates").sum())), represses=("sign", lambda v: int((v == "represses").sum())), tier1=("tier", lambda v: int((v == 1).sum()))).reset_index()
t1 = Q[Q.tier == 1].groupby("tf").sign.value_counts().unstack(fill_value=0); pf["tier1_activates"] = pf.tf.map(t1.get("activates", pd.Series(dtype=int))).fillna(0).astype(int); pf["tier1_represses"] = pf.tf.map(t1.get("represses", pd.Series(dtype=int))).fillna(0).astype(int)
pf["pct_activates"] = 100 * pf.activates / pf.supported; pf.sort_values("pct_activates", ascending=False).to_csv(f"{O2}/panel_f_activator_repressor.tsv", sep="\t", index=False, float_format="%.1f"); print("f TFs", len(pf), flush=True)
# g
G1 = pd.read_csv(f"{R2}/m34_fig5_panels_v3/out/panel_g_pairs_shared_targets.tsv", sep="\t")[["pair", "figure4_pair", "af3_5seed_strong", "n_bound_a", "n_bound_b", "n_shared_targets"]]
BG = {t: set(V1.gene[V1.tf == t]) for t in set(V1.tf)}; rows = []
for rr in G1.itertuples():
    a_, b_ = rr.pair.split("-"); sh = BG[a_] & BG[b_]; d = rr._asdict(); d.pop("Index")
    for side, t in (("a", a_), ("b", b_)):
        if t not in ti or cells.get(t, 0) < 25: d[f"kd_{side}"] = "no usable top-25% knockdown"; continue
        zz = np.array([Z[ti[t], gi[g]] if g in gi else np.nan for g in sh]); zo = np.array([Z[ti[t], gi[g]] if g in gi else np.nan for g in BG[t] - sh])
        d[f"kd_{side}"] = f"{t} knockdown"; d[f"n_shared_responding_{side}"] = int((np.abs(np.nan_to_num(zz)) >= 2).sum()); d[f"pct_shared_responding_{side}"] = 100 * np.nanmean(np.abs(zz) >= 2) if np.isfinite(zz).any() else np.nan
        d[f"pct_own_only_targets_responding_{side}"] = 100 * np.nanmean(np.abs(zo) >= 2) if np.isfinite(zo).any() else np.nan
    rows.append(d)
G2 = pd.DataFrame(rows); G2.to_csv(f"{O2}/panel_g_pairs_shared_targets.tsv", sep="\t", index=False, float_format="%.2f"); print(G2[G2.figure4_pair].round(1).to_string(index=False), flush=True)
# recovery_top25
T = pd.read_csv(f"{S35}/per_tf_threshold_sweep_top25_sgRNA.tsv", sep="\t"); P = pd.read_csv(f"{S35}/pooled_threshold_sweep_top25_sgRNA.tsv", sep="\t"); TH = ["abs2", "abs3", "abs4", "abs5", "top50", "top100", "top200"]
T = T[T.threshold.isin(TH)].copy(); T["top25_cells"] = T.TF.map(cells); T["usable_knockdown_ge25_cells"] = T.top25_cells.ge(25)
T = T.rename(columns={"n_responders": "links", "n_shared": "found", "pct_recovered": "pct_found", "reference_pct": "expected_pct", "enough_responders": "ge10_links"})
P = P[P.threshold.isin(TH)].rename(columns={"total_responders": "links", "total_recovered": "found", "pooled_pct_recovered": "pct_found", "pooled_reference_pct": "expected_pct", "n_tfs_ge10_responders": "n_tfs_ge10_links", "n_tfs_fold_gt_1_2": "n_tfs_fold_gt_1_2"})
P["set"] = P.set.str.replace("ours, all 75 TFs", "ours, all TFs with a usable top-25% knockdown")
T.to_csv(f"{S35}/recovery_top25_per_tf.tsv", sep="\t", index=False, float_format="%.3f"); P.to_csv(f"{S35}/recovery_top25_pooled.tsv", sep="\t", index=False, float_format="%.3f")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps"); T.to_excel(f"{S35}/recovery_top25_per_tf.xlsx", index=False); P.to_excel(f"{S35}/recovery_top25_pooled.xlsx", index=False)
print("usable TFs:", int(T[T.source.str.startswith("our")].drop_duplicates("TF").usable_knockdown_ge25_cells.sum()))
print(P[["set", "threshold_label", "n_tfs", "n_tfs_ge10_links", "links", "found", "pct_found", "expected_pct", "pooled_fold", "median_fold_tfs_ge10"]].round(2).to_string(index=False))
