# Add pass/flag columns to the per-TF table
import sys, pandas as pd
t = pd.read_csv(sys.argv[1], sep="\t")
t["passes_kd_q05_2guides"] = (t.kd_q_bh < 0.05) & (t.n_guides_library >= 2) & (t.knockdown_pct > 0)
fl = []
for _, r in t.iterrows():
    f = []
    if r.n_cells < 50: f.append("low_cells")
    if r.ntc_frac_cells_detected < 0.2: f.append("low_expr_target")
    if r.name_mapping == "alias": f.append("non_HGNC_panel_name")
    fl.append(",".join(f) or ".")
t["flags"] = fl
t.to_csv(sys.argv[1], sep="\t", index=False, float_format="%.4g")
p = t.groupby("panel_tf").passes_kd_q05_2guides.any()
print("panel TFs passing:", p.sum(), "/", len(p)); print("fail:", list(p[~p].index))
print(t.loc[t["flags"] != ".", ["panel_tf","gene_symbol","n_cells","ntc_frac_cells_detected","knockdown_pct","kd_q_bh","flags"]].to_string())
