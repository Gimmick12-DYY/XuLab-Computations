# m54: generated from m52_compare.py (m52) by gen54.py.
# m48: workbook with top 5,000 (m45 / m47) beside top 10,000 (m48) for the summary tables.
import sys
import pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; A = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out5k"; B = "/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out"
rd = lambda d, n: pd.read_csv(f"{d}/{n}.tsv", sep="\t", dtype={"padding": str})
sheets = {}
for n in ("1_recall_pooled", "2_recall_vs_background", "3_dose_response", "4_own_list_vs_similar_size", "5_gene_shuffle", "6_shuffle_by_tf_set", "7_leave_one_tf_out_summary", "list_sizes"):
    try: a = rd(A, n).assign(**{"peak set": "top 5,000"})
    except FileNotFoundError: a = None
    b = rd(B, n).assign(**{"peak set": "top 10,000"}); d = pd.concat([x for x in (a, b) if x is not None], ignore_index=True); cols = ["peak set"] + [c for c in d.columns if c != "peak set"]; sheets[n] = d[cols]
for n in ("8_tf_label_shuffle",): sheets[n] = rd(B, n)
sheets["9_counts_per_cutoff"] = rd(B, "9_counts_per_cutoff")
P47 = pd.read_csv("/work/users/d/y/dyy12/XuLab/regulatome/m54_stage1_percentile_tables/out47/3_precision.tsv", sep="\t", dtype={"padding": str}); sheets["precision"] = P47[P47["peak set"].isin(["top 5,000", "top 10,000"])]
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
with pd.ExcelWriter(f"{B}/stage1_top10k_vs_top5k.xlsx") as w:
    for n, d in sheets.items(): d.to_excel(w, sheet_name=n[:31], index=False)
print("done", list(sheets))
