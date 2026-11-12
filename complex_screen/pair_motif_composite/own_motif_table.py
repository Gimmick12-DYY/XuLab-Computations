# Own-motif table for all 79 TFs (2026-11-12; user request: p-values and ranks to judge how serious the problem is).
# Sources (all existing runs):
#  A) lab MACS2 summits, top 1,000 in open chromatin (HOT removed), background matched on GC x CpG x promoter x ATAC level, HOMER known
#     (472 motifs), odd (disc) and even + X (val) chromosomes: recall/homer/raw_lab/<TF>/<part>/knownResults.txt
#  B) same sites, 2,528-motif DB (JASPAR + HOCOMOCO v12 + Codebook), FIMO presence z: motif_valid/scores_fisher_*.tsv -> two-sided p
#  C) earlier imputed-peak HOMER runs (motif_analysis/motif/<tf>/{homer = HOMER genome background, homer_ocr = open-chromatin background})
# Cross-library negative control for A: the same own motif's log p in the other 78 libraries -> percentile of the own library.
# Output motif_tables/own_motif_all_tfs.tsv
import os, re, glob
import numpy as np, pandas as pd
from scipy.stats import norm
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; T = f"{O}/motif_tables"; os.makedirs(T, exist_ok=True)
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
FAM = {"MLX": r"^(Usf[12]|Max|c-Myc|n-Myc|CLOCK|NPAS2|MNT|bHLHE40)\(", "TBX3": r"^Tbx[235]\(", "HES1": r"^(HEY[12]?|Hes1?|HEY)\(", "SREBF2": r"^SREBP?[12]?\(", "E2F3": r"^E2F[0-9]*\(",
       "HOXA10": r"^Hox[abcd](9|10|11|13)\(", "HOXB9": r"^Hox[abcd](9|10|11|13)\(", "HOXC9": r"^Hox[abcd](9|10|11|13)\(", "HOXD10": r"^Hox[abcd](9|10|11|13)\(", "HOXA5": r"^Hox[abcd][1-8]\(",
       "HOXB6": r"^Hox[abcd][1-8]\(", "HOXB8": r"^Hox[abcd][1-8]\(", "HOXC4": r"^Hox[abcd][1-8]\(", "MSX2": r"^Msx[12]?\(", "EPAS1": r"^(HIF2a|HIF-1a|HIF1b|NPAS)", "RELA": r"^(p65|NFkB-p65)",
       "DDIT3": r"^(CHOP|Chop|Atf4|CEBP:AP1)", "ETS2": r"^(ETS|Ets1|ETS1|GABPA|ELF|Fli1|ERG)\(", "MEF2A": r"^Mef2[abcd]\(", "MEF2D": r"^Mef2[abcd]\(", "FOXC1": r"^Fox[a-z][0-9]*\(", "FOXJ3": r"^Fox[a-z][0-9]*\(",
       "FOXK1": r"^FOXK[12]\(", "SOX4": r"^Sox[0-9]+\(", "SOX12": r"^Sox[0-9]+\(", "NEUROD4": r"^(NeuroD1|NeuroG2|Atoh1|Olig2|Ascl1)\(", "RBPJ": r"^RBPJ", "ZIC2": r"^Zic[0-9]?\(", "NR1D2": r"^(Rev-Erb|RORgt|RORa|Erra)",
       "ZBTB7A": r"^ZBTB7", "NFYA": r"^NFY\(", "NRF1": r"^NRF1?\(", "ZNF143": r"ZNF143", "MAZ": r"^Maz\(", "CTCF": r"^CTCF\(", "E4F1": r"^E4F1?\(", "ZNF768": r"^ZNF768", "ZNF282": r"^ZNF282"}
def homer(f, t):
    if not os.path.exists(f): return None
    k = pd.read_csv(f, sep="\t"); k.columns = ["name", "cons", "p", "logp", "q", "nt", "pt", "nb", "pb"]
    m = k.name.str.split("(").str[0].str.upper().str.split(r"[|:/\-]").apply(lambda L: t in L)
    if t in FAM: m |= k.name.str.contains(FAM[t], regex=True)
    if not m.any(): return dict(n=len(k))
    j = int(np.flatnonzero(m.values)[0]); b = k.iloc[j]; pb = float(str(b.pb).rstrip("%")); pt = float(str(b.pt).rstrip("%"))
    return dict(n=len(k), motif=b["name"].split("/")[0], rank=j + 1, log10p=round(b.logp / np.log(10), 2), q=b.q, pct_sites=pt, pct_bg=pb, fold=round(pt / max(pb, .01), 2), key=b["name"])
rows = []
for t in cells.index:
    r = dict(tf=t, cells=int(cells[t]))
    for pt in ["disc", "val"]:
        h = homer(f"{O}/recall/homer/raw_lab/{t}/{pt}/knownResults.txt", t)
        if h and "rank" in h:
            r["homer_motif"] = h["motif"]; r[f"A_{pt}_rank"] = h["rank"]; r[f"A_{pt}_log10p"] = h["log10p"]; r[f"A_{pt}_q"] = h["q"]; r[f"A_{pt}_fold"] = h["fold"]; r[f"_key"] = h["key"]
    for lab, sub in [("C_imputed_genomebg", "homer"), ("C_imputed_ocrbg", "homer_ocr")]:
        h = homer(f"{X}/motif_analysis/motif/{t.lower()}/{sub}/knownResults.txt", t)
        if h and "rank" in h: r[f"{lab}_rank"] = h["rank"]; r[f"{lab}_log10p"] = h["log10p"]; r[f"{lab}_fold"] = h["fold"]
    rows.append(r)
D = pd.DataFrame(rows)
# cross-library control for A (discovery + held-out averaged log p of the same motif in every library)
LP = {}
for f in glob.glob(f"{O}/recall/homer/raw_lab/*/*/knownResults.txt"):
    t, pt = f.split("/")[-3:-1]; k = pd.read_csv(f, sep="\t"); LP[(t, pt)] = dict(zip(k.iloc[:, 0], k.iloc[:, 3] / np.log(10)))
for i, r in D.iterrows():
    if not isinstance(r.get("_key"), str): continue
    for pt in ["disc", "val"]:
        own = LP.get((r.tf, pt), {}).get(r._key); oth = [LP[(u, pt)].get(r._key) for u in cells.index if u != r.tf and (u, pt) in LP and r._key in LP[(u, pt)]]
        if own is not None and oth: D.at[i, f"A_{pt}_own_vs_other_libs_pct"] = round(float(np.mean(np.array(oth) > own)), 3)  # log p more negative = stronger; pct of libraries weaker
# B: 2,528-motif DB
Fd = pd.read_csv(f"{O}/motif_valid/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{O}/motif_valid/scores_fisher_val.tsv", sep="\t", index_col=0); OW = pd.read_csv(f"{O}/motif_valid/own_gate.tsv", sep="\t"); OW = OW[OW.method == "fisher"].set_index("tf")
for i, r in D.iterrows():
    if r.tf in OW.index:
        m = OW.at[r.tf, "best_own"]; D.at[i, "B_motif"] = m; D.at[i, "B_disc_rank"] = OW.at[r.tf, "rank_disc"]; D.at[i, "B_val_rank"] = OW.at[r.tf, "rank_val"]
        for pt, F in [("disc", Fd), ("val", Fv)]:
            z = F.at[m, r.tf] if m in F.index and r.tf in F.columns else np.nan; D.at[i, f"B_{pt}_z"] = round(z, 2); D.at[i, f"B_{pt}_log10p"] = round(np.log10(max(2 * norm.sf(abs(z)), 1e-300)) * np.sign(z), 2) if np.isfinite(z) else np.nan
D["has_motif"] = D.homer_motif.notna() | D.B_motif.notna()
D["A_pass"] = (D.A_disc_rank <= 10) & (D.A_val_rank <= 10); D["A_sig_both_p1e5"] = (D.A_disc_log10p <= -5) & (D.A_val_log10p <= -5)
D["B_pass"] = (D.B_disc_rank <= 10) & (D.B_val_rank <= 10); D["B_sig_both_p1e5"] = (D.B_disc_log10p <= -5) & (D.B_val_log10p <= -5)
D = D.drop(columns=["_key"]).sort_values(["has_motif", "A_disc_rank"], ascending=[False, True]); D.to_csv(f"{T}/own_motif_all_tfs.tsv", sep="\t", index=False)
m = D[D.has_motif]
print(len(D), "TFs;", int(D.has_motif.sum()), "with an own motif;", int(D.homer_motif.notna().sum()), "in HOMER library;", int(D.B_motif.notna().sum()), "in 2,528 DB")
print("A (HOMER, matched OCR bg): rank<=10 both", int(m.A_pass.sum()), "| p<=1e-5 both", int(m.A_sig_both_p1e5.sum()), "| p<=0.01 both", int(((m.A_disc_log10p <= -2) & (m.A_val_log10p <= -2)).sum()), "| own library top 5% of libraries (both)", int(((m.A_disc_own_vs_other_libs_pct >= .95) & (m.A_val_own_vs_other_libs_pct >= .95)).sum()))
print("B (FIMO 2,528, matched OCR bg): rank<=10 both", int(m.B_pass.sum()), "| p<=1e-5 both", int(m.B_sig_both_p1e5.sum()), "| p<=0.01 both", int(((m.B_disc_log10p <= -2) & (m.B_val_log10p <= -2)).sum()))
c = m.dropna(subset=["C_imputed_genomebg_rank"]); print("C imputed peaks: genome bg rank<=10", int((c.C_imputed_genomebg_rank <= 10).sum()), "of", len(c), "| OCR bg rank<=10", int((c.C_imputed_ocrbg_rank <= 10).sum()))
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
print(D[["tf", "cells", "homer_motif", "A_disc_rank", "A_disc_log10p", "A_val_rank", "A_val_log10p", "A_disc_fold", "A_val_own_vs_other_libs_pct", "B_disc_rank", "B_val_rank", "B_disc_log10p", "B_val_log10p", "C_imputed_genomebg_rank", "C_imputed_ocrbg_rank"]].to_string(index=False))
