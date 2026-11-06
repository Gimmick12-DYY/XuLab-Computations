# Corrected (enrichment ratio + SpQN) vs uncorrected ChromHMM-element matrices: side-by-side percentile figure for the repressive set,
# Spearman between matrices, recovery table corrected vs uncorrected, and cell counts of the repressive-only group.
# run from scripts/complex_catalog; usage: python3 <this> <uncorrected dir> <corrected dir>
import sys, numpy as np, pandas as pd
U, C = sys.argv[1], sys.argv[2]
exec(open(__file__.replace("compare_corrected.py", "plot_repressive_chromhmm.py")).read().split("rec = {}; rows = []")[0].replace("D, OUT = sys.argv[1], sys.argv[2]", "D, OUT = C + '/data', C"))
exec("rec = {}; rows = []" + open(__file__.replace("compare_corrected.py", "plot_repressive_chromhmm.py")).read().split("rec = {}; rows = []")[1].split("T = pd.DataFrame(rows)")[0])
T = pd.DataFrame(rows)
exec(open(__file__.replace("compare_corrected.py", "plot_repressive_chromhmm.py")).read().split("def ov(")[0].split("T = pd.DataFrame(rows);")[0][-1:] if False else "")
src = open(__file__.replace("compare_corrected.py", "plot_repressive_chromhmm.py")).read()
exec("def ov(" + src.split("def ov(")[1].split("orders = {}")[0])
# uncorrected recovery flags for the left panel
recC = dict(rec); D0 = D
def getU(kind, s):
    P = pd.read_csv(f"{U}/data/{kind}_{s}.tsv", sep="\t", index_col=0); P.index = P.index.str.lower(); P.columns = P.columns.str.lower(); return P
TU = pd.read_csv(f"{U}/known_complex_recovery.tsv", sep="\t")
for s, _ in SETS:
    for k, name, mem in KNOWN: rec[(s, k)] = bool(TU[(TU.set == s) & (TU.complex == name)].recovered.iloc[0])
fig = plt.figure(figsize=(12.6, 6.3))
for i, (lab, d) in enumerate((("Uncorrected", f"{U}/data"), ("Enrichment-ratio corrected", f"{C}/data"))):
    D = d
    if i == 1: rec.update(recC)
    ax = fig.add_axes([0.05 + i * 0.47, 0.08, 0.40, 0.80]); im, h = ov(ax, "repressive", DECK, 3.9, f"Repressive elements — {lab} (deck order)")
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.452 + i * 0.47, 0.08), fontsize=6, frameon=False, handlelength=1.3, title="Recovered", title_fontsize=6.5)
cbar(fig, im, [0.925, 0.55, 0.009, 0.25], "Percentile (SpQN r)"); save(fig, "a_side_by_side_repressive_uncorrected_vs_corrected")
D = D0
TU = TU.rename(columns={"mean_pair_pct": "uncorr_pct", "depth_matched_p": "uncorr_p", "recovered": "uncorr_rec"})
TC = pd.read_csv(f"{C}/known_complex_recovery.tsv", sep="\t").rename(columns={"mean_pair_pct": "corr_pct", "depth_matched_p": "corr_p", "recovered": "corr_rec"})
TM = pd.read_csv(f"{C}/min20/known_complex_recovery.tsv", sep="\t").rename(columns={"mean_pair_pct": "corr_min20_pct", "depth_matched_p": "corr_min20_p", "recovered": "corr_min20_rec"})
J = TU.merge(TC, on=["set", "complex", "members"]).merge(TM, on=["set", "complex", "members"]); J.to_csv(f"{C}/recovery_corrected_vs_uncorrected.tsv", sep="\t", index=False)
iu = np.triu_indices(79, 1)
def sp(a, b): return round(pd.Series(a.values[iu]).corr(pd.Series(b.values[iu]), method="spearman"), 3)
for s in ("repressive", "polycomb", "het", "znf_rpts"):
    print(s, "Spearman corrected vs corrected enhancer:", sp(get("pct", s), get("pct", "enhancer_biv")),
          "| corrected vs uncorrected same set:", sp(get("pct", s), getU("pct", s)))
print("enhancer corrected vs uncorrected:", sp(get("pct", "enhancer_biv"), getU("pct", "enhancer_biv")))
low = "akap8l hes1 hoxa5 msx2 neurod4 nr1d2 rela setdb1 sox12 stag2 tbx3 znf503 znf581 znf606 znf768 znf79".split()
print("cell rank (1 = fewest) of the 16-TF repressive-only group:", sorted(int(rk[t]) for t in low))
