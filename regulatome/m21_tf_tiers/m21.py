# m21_tf_tiers: per-TF evidence table from existing results (SPEC.md). Diagnostic. No new fits, no new benchmark reads.
import hashlib, os, subprocess
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; OUT = f"{RG}/m21_tf_tiers/out"; LAB = "diagnostic"
def src(path):
    c = subprocess.run(["git", "-C", X, "log", "--format=%h", "-1", "--", path], capture_output=True, text=True).stdout.strip()
    return c if c else ("untracked directory" if os.path.isdir(path) else "untracked sha256:" + hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])
F = {}
tfs = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]; T = pd.DataFrame(index=pd.Index(tfs, name="tf"))
p = f"{RG}/m6_improve/repro_v1/per_tf_floor_test.tsv"; FL = pd.read_csv(p, sep="\t").set_index("tf"); F["cells,depth_tertile,noise_floor_d,noise_floor_clears"] = (p, src(p))
T["cells"] = FL.cells; T["depth_tertile"] = FL.tertile; T["noise_floor_d"] = FL.d; T["noise_floor_clears"] = FL.clears; T["line1_d_ge_0.04"] = FL.d.reindex(T.index) >= .04
p = f"{RG}/m7_step2/features/canon_counts/read_totals.tsv"; RT = pd.read_csv(p, sep="\t").set_index("tf"); T["fragments"] = RT.reads; F["fragments"] = (p, src(p))
p = f"{RG}/m11_peak_regulome/inventory_per_tf.tsv"; IV = pd.read_csv(p, sep="\t"); T["q005_peaks"] = IV[IV.set == "b_lab_q005"].set_index("tf").n_q05; F["q005_peaks"] = (p, src(p))
p = f"{RG}/m11_peak_regulome/out/long_q0.05.tsv.gz"; LG = pd.read_csv(p, sep="\t", usecols=["tf", "any_peak"]); T["genes_with_linked_peak"] = LG.groupby("tf").any_peak.sum(); F["genes_with_linked_peak"] = (p, src(p))
mot = sorted(f[:-5] for f in os.listdir(f"{RG}/m9_abc_diagnosis/work/motifs") if f.endswith(".meme")); T["own_motif_hocomoco_v11"] = T.index.isin(mot); F["own_motif_hocomoco_v11"] = (f"{RG}/m9_abc_diagnosis/work/motifs/", src(f"{RG}/m9_abc_diagnosis/work/motifs"))
p = f"{X}/complex_screen/pair_motif_composite/per_tf_raw/own_motif_check.tsv"; OM = pd.read_csv(p, sep="\t").set_index("tf"); T["heldout_own_motif_rank_of_472"] = OM.val_known_rank; T["heldout_motif_name"] = OM.val_known_motif; F["heldout_own_motif_rank"] = (p, src(p))
p = f"{RG}/m10_promoter_regulome/G1_signal_fraction.tsv"; G1 = pd.read_csv(p, sep="\t"); G1 = G1[(G1.site_list == "motif_" + G1.tf) & (G1.sites == "all")].set_index("tf")
T["G1_enrich_vs_pooled78"] = G1.enrichment_vs_pooled_other78; T["G1_lo"] = G1.ep_lo; T["G1_hi"] = G1.ep_hi; F["G1"] = (p, src(p))
T["line2_motif"] = (T.heldout_own_motif_rank_of_472 <= 10) | (T.G1_lo > 1); T["line2_testable"] = T.heldout_own_motif_rank_of_472.notna() | T.G1_lo.notna()
p = f"{RG}/m20_raw_binding_audit/out/part1d_motif_label_check.tsv"; PD = pd.read_csv(p, sep="\t").set_index("motif_of"); T["m20_promoter_read_motif_rank_of_79"] = PD.rank_own_among_79_D; F["m20_part1d"] = (p, src(p))
p = f"{RG}/m20_raw_binding_audit/out/part2_all_tfs_vs_each_chip.tsv"; P2 = pd.read_csv(p, sep="\t"); P2 = P2[P2.own]; F["m20_part2"] = (p, src(p))
best = P2.sort_values("rank_auroc").drop_duplicates("tf").set_index("tf")
T["bulk_chip"] = P2.groupby("tf").chip.apply(lambda v: "; ".join(v)); T["chip_auroc_open_chromatin"] = best.auroc_union_atac; T["chip_rank_of_79"] = best.rank_auroc
T["line3_chip"] = T.chip_rank_of_79 <= 10; T["line3_testable"] = T.chip_rank_of_79.notna()
p = f"{RG}/m16_strong_phenotype/out/strong_phenotype_per_tf.tsv"; SP = pd.read_csv(p, sep="\t").set_index("panel_tf"); T["kd_status"] = SP.status; T["strong_phenotype"] = SP.status.reindex(T.index) == "strong phenotype"; F["m16"] = (p, src(p))
p = f"{RG}/m7_step2/splits/tf_folds.tsv"; T["kd_reliability"] = pd.read_csv(p, sep="\t").set_index("TF").ceiling; F["reliability"] = (p, src(p))
p = f"{RG}/m19_binary_matrices/out/responder_counts.tsv"; RC = pd.read_csv(p, sep="\t", index_col=0); T["reproducible_responders_P4"] = RC["P4_reproducible_halves|all"]; F["m19_counts"] = (p, src(p))
p = f"{RG}/m19_binary_matrices/out/all_combinations_per_tf.tsv"; A = pd.read_csv(p, sep="\t"); F["m19"] = (p, src(p)); fs = []
for b, pp, tag in (("B2_promoter_peak", "P1_top5pct", "B2xP1"), ("B1_any_linked_peak", "P4_reproducible_halves", "B1xP4")):
    d = A[(A.B == b) & (A.P == pp) & (A.direction == "all")].set_index("tf")
    T[f"{tag}_status"] = d.status; T[f"{tag}_OR"] = d.OR; T[f"{tag}_lo"] = d.OR_lo; T[f"{tag}_hi"] = d.OR_hi; T[f"{tag}_OR_swapped"] = d.OR_swapped_tf
    fs.append((T[f"{tag}_lo"] > 1) & (T[f"{tag}_OR"] > T[f"{tag}_OR_swapped"].fillna(-np.inf)))
T["function_supported"] = fs[0] | fs[1]
T["n_lines_met"] = T["line1_d_ge_0.04"].fillna(False).astype(int) + T.line2_motif.fillna(False).astype(int) + T.line3_chip.fillna(False).astype(int)
T["n_lines_testable"] = T.noise_floor_d.notna().astype(int) + T.line2_testable.astype(int) + T.line3_testable.astype(int)
T["tier"] = np.where(T.n_lines_met >= 2, "Tier 1", np.where(T.n_lines_met == 1, "Tier 2", np.where(T.n_lines_testable <= 1, "Tier 3 (not testable: noise floor only, not met)", "Tier 3 (tested, none met)")))
T.insert(0, "label", LAB); T.reset_index().to_csv(f"{OUT}/tf_evidence_table.tsv", sep="\t", index=False, float_format="%.4g")
pd.DataFrame([dict(column=k, file=v[0], commit=v[1]) for k, v in F.items()]).to_csv(f"{OUT}/sources.tsv", sep="\t", index=False)
miss = T.drop(columns="label").isna().sum(); miss.to_frame("n_tfs_missing").to_csv(f"{OUT}/missing_per_column.tsv", sep="\t")
print(T.tier.value_counts().to_string()); 
for t_, g in T.groupby("tier"): print(t_, ":", ",".join(g.index))
print("function supported:", ",".join(T.index[T.function_supported]))
print(T[T.n_lines_met >= 1][["noise_floor_d", "heldout_own_motif_rank_of_472", "G1_enrich_vs_pooled78", "G1_lo", "chip_rank_of_79", "n_lines_met", "strong_phenotype", "function_supported"]].round(3).to_string())
print(miss[miss > 0].to_string())
