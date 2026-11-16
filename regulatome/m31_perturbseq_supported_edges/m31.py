# m31 (SPEC.md): Perturb-seq-supported Regulome edges, per-TF and per-gene summaries, evidence inventory. Descriptive tables only.
import sys, json
import numpy as np, pandas as pd
X_ = "/work/users/d/y/dyy12/XuLab"; RG_ = f"{X_}/regulatome"; OUT_ = f"{RG_}/m31_perturbseq_supported_edges/out"; LAB_ = "descriptive"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])  # TF, UG, Z1, Z2, ZA
TFL, N = list(TF), len(UG); gi_ = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFL):
    for o in OWN.get(t, [t]):
        if o in gi_: ZA[i, gi_[o]] = Z1[i, gi_[o]] = Z2[i, gi_[o]] = np.nan
E = pd.read_csv(f"{RG_}/m22_edge_table/out/regulome_edges.tsv.gz", sep="\t", low_memory=False, usecols=["tf", "gene", "promoter_peak", "promoter_signalValue", "n_distal_elements_with_peak", "best_abc", "within_tf_percentile", "shared_by"])
E = E[E.gene.isin(gi_)]; ti = {t: i for i, t in enumerate(TFL)}; Ek = E[E.tf.isin(ti)].copy()
r, c = Ek.tf.map(ti).values, Ek.gene.map(gi_).values; Ek["z"] = ZA[r, c]; Ek["z1"] = Z1[r, c]; Ek["z2"] = Z2[r, c]
pp = Ek.promoter_peak.astype(str).str.lower().eq("true"); dd = Ek.n_distal_elements_with_peak > 0
Ek["peak_location"] = np.where(pp & dd, "both", np.where(pp, "promoter", "distal"))
S = Ek[np.abs(Ek.z) >= 2].copy()
S["direction"] = np.where(S.z < 0, "down on knockdown (TF activates)", "up on knockdown (TF represses)")
S["reproducible_both_halves"] = np.where(np.isfinite(S.z1) & np.isfinite(S.z2) & (np.abs(S.z1) >= 2) & (np.abs(S.z2) >= 2) & (np.sign(S.z1) == np.sign(S.z2)), "yes", "no")
S["n_other_panel_tfs_bound"] = S.shared_by - 1
S = S.rename(columns={"tf": "TF", "within_tf_percentile": "score_percentile_within_tf", "best_abc": "ABC", "z": "knockdown_z"})
S = S[["TF", "gene", "peak_location", "promoter_signalValue", "score_percentile_within_tf", "ABC", "knockdown_z", "direction", "reproducible_both_halves", "n_other_panel_tfs_bound"]].sort_values(["TF", "knockdown_z"])
S.insert(0, "label", LAB_); S.to_csv(f"{OUT_}/supported_edges.tsv", sep="\t", index=False, float_format="%.4g"); print("supported edges", len(S), "TFs", S.TF.nunique(), flush=True)
P = []
for i, t in enumerate(TFL):
    b = set(Ek.gene[Ek.tf == t]); resp = np.isfinite(ZA[i]) & (np.abs(np.nan_to_num(ZA[i])) >= 2); s = S[S.TF == t]
    P.append(dict(label=LAB_, TF=t, n_regulome_targets=len(b), n_responding_genes=int(resp.sum()), n_supported_edges=len(s), n_down=int((s.knockdown_z < 0).sum()), n_up=int((s.knockdown_z > 0).sum()),
                  n_reproducible=int((s.reproducible_both_halves == "yes").sum()), pct_of_all_genes_bound=100 * len(b) / N, pct_of_responding_genes_bound=100 * len(s) / resp.sum() if resp.sum() else np.nan))
P = pd.DataFrame(P).sort_values("n_supported_edges", ascending=False); P.to_csv(f"{OUT_}/per_tf_summary.tsv", sep="\t", index=False, float_format="%.3g")
BND = np.zeros((len(TFL), N), bool); BND[r, c] = True; RSP = np.isfinite(ZA) & (np.abs(np.nan_to_num(ZA)) >= 2); BOTH = BND & RSP
allb = E.groupby("gene").tf.nunique().reindex(UG).fillna(0).astype(int).values
G = pd.DataFrame(dict(label=LAB_, gene=UG, n_tfs_bound_of_79=allb, n_tfs_bound_with_knockdown=BND.sum(0), n_tfs_knockdown_changes_it=RSP.sum(0), n_tfs_bound_and_changing=BOTH.sum(0),
                      bound_and_changing=[";".join(f"{TFL[i]}:{ZA[i, j]:.2f}" for i in np.flatnonzero(BOTH[:, j])) for j in range(N)])).sort_values(["n_tfs_bound_and_changing", "n_tfs_knockdown_changes_it"], ascending=False)
G.to_csv(f"{OUT_}/per_gene_summary.tsv", sep="\t", index=False)
# evidence inventory
I = []
B = pd.read_csv(f"{RG_}/m20_raw_binding_audit/out/part2_bulk_overlap.tsv", sep="\t")
for x in B.itertuples():
    I.append(dict(TF=x.own_tf, evidence="bulk ChIP overlap (q005 summits)", reference=x.chip, value=f"{100 * x.own_recall:.1f}% of ChIP peaks have an own summit (rank {x.own_rank_recall:.0f}/79; other-TF median {100 * x.others_recall_median:.1f}%); {100 * x.own_precision:.1f}% of own summits in ChIP (rank {x.own_rank_precision:.0f}/79; other-TF median {100 * x.others_precision_median:.1f}%)", source="regulatome/m20_raw_binding_audit/out/part2_bulk_overlap.tsv"))
C = pd.read_csv(f"{RG_}/m25_peak_threshold/out/choice_chip.tsv", sep="\t"); C = C[C.definition == "E2_top5000"]
for x in C.itertuples(): I.append(dict(TF=x.tf, evidence="bulk ChIP overlap (top 5,000 peaks, edge elements)", reference=x.chip, value=f"own {100 * x.own_frac_in_chip:.1f}% in ChIP vs other-TF median {100 * x.other_tf_median:.1f}%", source="regulatome/m25_peak_threshold/out/choice_chip.tsv"))
M = pd.read_csv(f"{RG_}/m25_peak_threshold/out/choice_motif.tsv", sep="\t"); M = M[M.definition == "E2_top5000"]
for x in M.itertuples(): I.append(dict(TF=x.tf, evidence="own motif in edge elements (top 5,000 peaks)", reference="HOCOMOCO v11, FIMO p<1e-4", value=f"own motif in {100 * x.own_motif_frac:.1f}% of {x.n_edge_elements} elements vs {100 * x.own_motif_bg:.1f}% background (excess {100 * x.own_excess:+.1f} points)", source="regulatome/m25_peak_threshold/out/choice_motif.tsv"))
for t in ("CTCF", "NFYA", "NRF1", "ZNF143"):
    j = json.load(open(f"{RG_}/m12_percell_ontarget/out/{t}_summary.json"))
    I.append(dict(TF=t, evidence="fragments on own motif sites (pooled cells)", reference=f"swap TF {j.get('swap_tf')}", value=f"own-motif enrichment {j.get('motif_own_pooled_enrichment', float('nan')):.2f}x vs swap TF {j.get('motif_swap_pooled_enrichment', float('nan')):.2f}x; {100 * j.get('frac_in_q005', float('nan')):.1f}% of fragments in own q005 peaks; {100 * j.get('frac_open', float('nan')):.1f}% in open chromatin", source=f"regulatome/m12_percell_ontarget/out/{t}_summary.json"))
for t, rv, ref in (("CTCF", "+0.45", "HEK293/HEK293T 2-of-3"), ("ZIC2", "+0.10", "HEK293 ENCODE"), ("ZNF282", "+0.10", "HEK293T"), ("ZNF777", "+0.03", "HEK293 ENCODE"), ("ZBTB7A", "0.00", "HEK293 ENCODE"), ("MAZ", "-0.11", "HEK293 ENCODE")):
    I.append(dict(TF=t, evidence="Fig 2 QC: sc vs own bulk reference, summit-group Pearson", reference=ref, value=f"r = {rv} (only CTCF stands out from off-diagonal values)", source="RESEARCH_JOURNAL.md 2026-11-10; results_293T_analysis/fig2_qc/"))
I = pd.DataFrame(I).sort_values(["TF", "evidence"]); I.insert(0, "label", LAB_); I.to_csv(f"{OUT_}/evidence_inventory.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
print(P.drop(columns="label").round(1).to_string(index=False)); print(G.head(30).drop(columns="label").to_string(index=False)); print(I.drop(columns="label").to_string(index=False)); print("done")
