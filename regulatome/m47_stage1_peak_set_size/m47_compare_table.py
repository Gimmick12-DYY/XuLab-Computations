# m47: comparison table bulk ChIP (HEK293T / HEK293) vs our HITAG, per TF, without ZBTB7A (user, 2026-11-20). Reformats 4_bulk_vs_ours_per_tf.tsv; no new numbers.
import pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m47_stage1_peak_set_size/out"
T = pd.read_csv(f"{O}/4_bulk_vs_ours_per_tf.tsv", sep="\t", dtype={"padding": str}); T = T[T.TF != "ZBTB7A"]
LISTS = ["bulk ChIP top 5,000", "bulk ChIP top 10,000", "bulk ChIP full list", "our HITAG top 5,000", "our HITAG top 10,000"]
SHORT = {"bulk ChIP top 5,000": "Bulk top 5k", "bulk ChIP top 10,000": "Bulk top 10k", "bulk ChIP full list": "Bulk full", "our HITAG top 5,000": "Ours top 5k", "our HITAG top 10,000": "Ours top 10k"}
CELL = T[T.list.str.startswith("bulk")].groupby("TF")["reference / cell line"].first()
def cell(r):
    if r["n < 10"]: return "n < 10"
    return f"{r['recall (%)']:.1f}% | {r['observed / shuffle']:.2f}x | p {r['p (one-sided)']:.3f}" + (" *" if r["p (one-sided)"] < 0.05 else "")
sheets = {}
for pad in ("0", "10 kb", "50 kb"):
    rows = []
    for tf in ["CTCF", "ZNF282", "MAZ", "ZIC2", "ZNF777", "SETDB1"]:
        for c in ("|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"):
            g = T[(T.TF == tf) & (T.padding == pad) & (T.cutoff == c)].set_index("list")
            row = {"TF": tf, "bulk ChIP reference": CELL[tf], "cutoff": c.replace(">=", "≥"), "Perturb-seq genes": int(g["Perturb-seq genes"].iloc[0])}
            for l in LISTS: row[SHORT[l]] = cell(g.loc[l])
            rows.append(row)
    D = pd.DataFrame(rows)
    cnt = {"TF": "TF × cutoff cells above shuffle (p < 0.05, n ≥ 10)", "bulk ChIP reference": "", "cutoff": "", "Perturb-seq genes": ""}
    for l in LISTS: cnt[SHORT[l]] = int(((T.padding == pad) & (T.list == l) & (~T["n < 10"]) & (T["p (one-sided)"] < 0.05)).sum())
    sheets[f"padding {pad}"] = pd.concat([D, pd.DataFrame([cnt])], ignore_index=True)
note = pd.DataFrame({"note": ["Each cell: recall (share of the TF's Perturb-seq genes in the list) | precision observed / matched-shuffle precision | one-sided p (1,000 shuffles within NTC expression x promoter ATAC deciles). * = p < 0.05.",
                              "Same Stage 1 path for all lists: summit in a promoter or canonical Hi-C ABC-linked element, plus padding; 11,228 genes; top-25% sgRNA cells; own gene removed.",
                              "Peaks ranked by signalValue. ZNF282 (8,514 peaks) and ZNF777 (6,223) have fewer than 10,000 bulk peaks: their bulk top 10k = full file.",
                              "ZBTB7A removed (user, 2026-11-20). No K562 or HepG2 references. No pooled number. n < 10 = fewer than 10 Perturb-seq genes at that cutoff."]})
with pd.ExcelWriter(f"{O}/stage1_bulk_vs_ours_comparison_no_ZBTB7A.xlsx") as w:
    for k, d in sheets.items(): d.to_excel(w, sheet_name=k, index=False)
    note.to_excel(w, sheet_name="notes", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 40)
print(sheets["padding 10 kb"].to_string(index=False)); print("done")
