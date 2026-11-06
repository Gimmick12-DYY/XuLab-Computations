# R2 summary: per-TF list sizes and Gate 1 tier; overlap of the top lists with the ATAC-only list built the same way (ABC x log1p ATAC CPM).
import glob, numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/r2_targets"
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
Ti = pd.read_csv(f"{X}/regulatome/gate1/v2_1/tf_level_tier.tsv", sep="\t").set_index("tf")
def tier(t):
    if t in Ti.index: return "specific" if Ti.loc[t, "tier"] == "TF-level" else "accessibility-limited"
    return "undetermined"
TOP = 500; rows = []
for t in tfs:
    T = pd.read_csv(f"{O}/per_tf/{t}.tsv.gz", sep="\t"); n = len(T); ok = T[T.count_ok]
    a = set(T.nlargest(TOP, "S_atac").gene); r = dict(tf=t, tier=tier(t), n_cells=int(Ti.n_cells.get(t, -1)), genes=n, S_pos=int((T.S_k0 > 0).sum()), count_ok=len(ok),
         list_pos_countok=int((ok.S_k0 > 0).sum()), list_pos_countok_k1=int((ok.S_k1 > 0).sum()), k1_rank_changed=int((T.rank_k0 != T.rank_k1).sum()))
    for k in ("S_k0", "S_k1", "S_rawcpm"):
        r[f"top{TOP}_ovl_atac_{k}"] = len(set(T.nlargest(TOP, k).gene) & a)
    if len(ok) >= TOP:
        ao = set(ok.nlargest(TOP, "S_atac").gene); r[f"top{TOP}_ovl_atac_countok"] = len(set(ok.nlargest(TOP, "S_k0").gene) & ao)
    r["spearman_S_atac"] = round(T.S_k0.corr(T.S_atac, method="spearman"), 3); r["chance"] = round(TOP * TOP / n, 1); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{O}/r2_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)
print(R.tier.value_counts().to_string()); print(R.sort_values(["tier", "tf"]).to_string(index=False))
print(R.groupby("tier")[["list_pos_countok", f"top{TOP}_ovl_atac_S_k0", f"top{TOP}_ovl_atac_S_rawcpm", "spearman_S_atac"]].median().round(2).to_string())
