# Post-manifest check (added after manifest v7; v7 unchanged): R5 core top 1,000 vs the B3/B4 "raw_direct" ablation arm (R2 direct list,
# per_tf rank_k1 <= 1,000, as b3b4.py builds it), with the arm's knockdown effects from m4_validation/b3b4 (already-computed tables only).
# Writes checks/per_tf_core_vs_b3b4.tsv. usage: checks_core_vs_b3b4.py
import os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/regulatome/r5_core"; R2 = f"{X}/regulatome/r2_targets"; B = f"{X}/regulatome/m4_validation/b3b4"
S = pd.read_csv(f"{R}/tf_summary.tsv", sep="\t").set_index("tf"); n_cells = pd.read_csv(f"{R2}/r2_summary.tsv", sep="\t").set_index("tf").n_cells
Ab = pd.read_csv(f"{B}/ablation_per_tf.tsv", sep="\t"); Pt = pd.read_csv(f"{B}/per_tf_effects.tsv", sep="\t"); Pt = Pt[Pt.tier.str.startswith("direct-")]
ab = lambda arm, lad, t: Ab[(Ab.arm == arm) & (Ab.ladder == lad) & (Ab.tf == t)].effect.squeeze() if ((Ab.arm == arm) & (Ab.ladder == lad) & (Ab.tf == t)).any() else np.nan
pt = lambda arm, t, col: Pt[(Pt.arm == arm) & (Pt.tf == t)][col].squeeze() if ((Pt.arm == arm) & (Pt.tf == t)).any() else np.nan
rows = []; diff_top = 0
for t in S.index:
    L = pd.read_csv(f"{R}/lists/{t}.tsv.gz", sep="\t"); core = set(L.gene[L.top_1000])
    P = pd.read_csv(f"{R2}/per_tf/{t}.tsv.gz", sep="\t", usecols=["gene", "rank_k1"]); arm = set(P.gene[P.rank_k1 <= 1000])
    f = f"{R2}/top1000/{t}.tsv"; diff_top += int(os.path.exists(f) and set(pd.read_csv(f, sep="\t").gene) != arm)
    rows.append(dict(tf=t, genes_S_pos=S.genes_S_pos[t], promoter_share=S.median_promoter_share_top1000[t], loop_support_frac=S.frac_top1000_loop_supported[t],
                     spearman_S_atac=S.spearman_S_Satac[t], spearman_S_pool=S.spearman_S_Spool[t], splithalf_jaccard=S.jaccard_split_half_top1000[t],
                     jaccard_core_vs_rawABC_arm=len(core & arm) / len(core | arm),
                     arm1_rawABC_effect=ab("arm1", "raw_direct", t), arm2_rawABC_effect=ab("arm2", "raw_direct", t),
                     arm1_direct_effect_vs_atac_null=pt("arm1", t, "effect_atac"), arm2_direct_effect_vs_atac_null=pt("arm2", t, "effect_atac"),
                     arm1_promoter_only_effect=ab("arm1", "promoter_only", t), arm2_promoter_only_effect=ab("arm2", "promoter_only", t), n_cells=n_cells.get(t, np.nan)))
T = pd.DataFrame(rows); T.to_csv(f"{R}/checks/per_tf_core_vs_b3b4.tsv", sep="\t", index=False, float_format="%.4g", na_rep="NA")
j = T.jaccard_core_vs_rawABC_arm; print("jaccard core vs raw_direct arm: median", round(j.median(), 3), "range", round(j.min(), 3), round(j.max(), 3)); print(T.nsmallest(5, "jaccard_core_vs_rawABC_arm")[["tf", "jaccard_core_vs_rawABC_arm", "genes_S_pos", "n_cells"]].to_string(index=False))
print("TFs where r2_targets/top1000 differs from per_tf rank_k1 <= 1000:", diff_top)
