# Systematic own-motif recovery across the 79-TF panel (single-cell Tn5 data; 2026-11-12, user go).
# Part A (full depth, all methods already run): per TF, own-motif rank under each method (HOMER known on lab peaks [recall raw_lab, +/- artifact
#   drop], FIMO enrichment / panel / CentriMo on 2,528 motifs [motif_valid], motif-centred read profiles [motif_centred], k-mer model
#   [sequence_model]); recovered = rank <= 10 on both chromosome halves in >= 1 method; plus cells, reads, motif information content.
# Part B (depth): downsampling curves (downsample/res, HOMER known, own rank vs subsampled cells, 2-3 seeds) for every TF with a HOMER own motif;
#   per depth the fraction of TFs recovered (median rank <= 10 on both halves), and per TF the smallest depth with recovery.
# Output downsample/systematic_full_depth.tsv, downsample/systematic_depth_curve.tsv, downsample/systematic_{summary.png,svg}
import json, glob, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; D = f"{O}/downsample"; sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; Pc = rp.use()
cells = pd.read_csv(f"{X}/complex_screen/pair_motif_lookup/tf_cells.tsv", sep="\t").set_index("tf").n_cells
reads = {}
for f in glob.glob(f"{O}/reversed/depth/counts/*.info.tsv"): r = pd.read_csv(f, sep="\t").iloc[0]; reads[r.tf] = int(r.n_reads)
pwm, cur, rr = {}, None, []
for l in list(open(f"{O}/db/known.meme")) + ["MOTIF __end__"]:
    if l.startswith("MOTIF "):
        if cur and rr: pwm[cur] = np.array(rr)
        cur = l.split()[1]; rr = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.": rr.append([float(v) for v in l.split()])
def ic(m): p = pwm[m] / pwm[m].sum(1, keepdims=True); return float((2 + (p * np.log2(np.maximum(p, 1e-9))).sum(1)).sum())
rows = {t: dict(tf=t, cells=int(cells[t]), reads=reads.get(t, np.nan)) for t in cells.index}
def add(tab, method, tfcol, dcol, vcol, motifcol=None, filt=None):
    d = pd.read_csv(tab, sep="\t")
    if filt is not None: d = d[filt(d)]
    for r in d.itertuples():
        t = getattr(r, tfcol)
        if t not in rows: continue
        rows[t][f"{method}_rank"] = f"{int(getattr(r, dcol))}/{int(getattr(r, vcol))}" if pd.notna(getattr(r, dcol)) and pd.notna(getattr(r, vcol)) else ""
        rows[t][f"{method}_ok"] = bool(getattr(r, dcol) <= 10 and getattr(r, vcol) <= 10)
        if motifcol and getattr(r, motifcol) in pwm: rows[t]["own_motif_db"] = getattr(r, motifcol); rows[t]["own_motif_ic"] = round(ic(getattr(r, motifcol)), 1)
add(f"{O}/recall/gate_own_motif.tsv", "homer_peaks", "tf", "disc", "val", filt=lambda d: d.variant == "raw_lab")
add(f"{O}/recall/gate_own_motif_noart.tsv", "homer_peaks_noart", "tf", "disc", "val", filt=lambda d: d.variant == "raw_lab")
for m in ["fisher", "panel", "centrimo", "centrimo_panel"]: add(f"{O}/motif_valid/own_gate.tsv", f"mv_{m}", "tf", "rank_disc", "rank_val", "best_own", filt=lambda d, m=m: d.method == m)
for m in ["cf", "cf_panel", "occ", "occ_panel"]: add(f"{O}/motif_centred/own_gate.tsv", f"mc_{m}", "tf", "rank_disc", "rank_val", filt=lambda d, m=m: d.method == m)
add(f"{O}/sequence_model/own_gate.tsv", "kmer", "tf", "rank_disc_model", "rank_val_model")
A = pd.DataFrame(rows.values()); okc = [c for c in A.columns if c.endswith("_ok")]; A["has_own_motif"] = A[[c.replace("_ok", "_rank") for c in okc]].notna().any(axis=1) & (A[[c.replace("_ok", "_rank") for c in okc]].fillna("") != "").any(axis=1)
A["n_methods_recovered"] = A[okc].fillna(False).astype(bool).sum(1); A["recovered_any"] = A.n_methods_recovered > 0
A = A.sort_values(["recovered_any", "n_methods_recovered", "cells"], ascending=[False, False, False]); A.to_csv(f"{D}/systematic_full_depth.tsv", sep="\t", index=False)
m = A[A.has_own_motif]; print(f"TFs with an own motif: {len(m)} of {len(A)}; recovered (rank <= 10 both halves) by >= 1 method: {int(m.recovered_any.sum())}:", ",".join(m.tf[m.recovered_any]))
print("per method:", {c[:-3]: int(m[c].fillna(False).astype(bool).sum()) for c in okc})
from scipy.stats import mannwhitneyu
for col in ["cells", "own_motif_ic"]:
    a, b = m[m.recovered_any][col].dropna(), m[~m.recovered_any][col].dropna()
    if len(a) > 1 and len(b) > 1: print(col, "recovered median", a.median(), "not recovered median", b.median(), "MWU p", round(mannwhitneyu(a, b).pvalue, 4))
T = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{D}/res/*.json")])
if len(T):
    T["ok"] = (T.disc_own_rank <= 10) & (T.val_own_rank <= 10); T["rank_mean"] = (T.disc_own_rank + T.val_own_rank) / 2
    C = T.groupby(["tf", "n_cells"]).agg(rank_mean=("rank_mean", "median"), frac_ok=("ok", "mean"), seeds=("seed", "size")).reset_index(); C.to_csv(f"{D}/systematic_depth_curve.tsv", sep="\t", index=False)
    lv = [500, 1000, 1500, 3000, 6000, 12000, 25000]; cur = []
    for L in lv:
        s = C[C.n_cells == L]; cur.append(dict(cells=L, n_tf=len(s), frac_recovered=round((s.frac_ok >= .5).mean(), 3) if len(s) else np.nan))
    print(pd.DataFrame(cur).to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for t, g in C.groupby("tf"):
        col = Pc.evidence.hic_loop if (g.frac_ok >= .5).any() else Pc.neutral.light; ax[0].plot(g.n_cells, g.rank_mean, "-", lw=.8, color=col)
        if (g.frac_ok >= .5).any(): ax[0].text(g.n_cells.iloc[-1] * 1.05, g.rank_mean.iloc[-1], t, fontsize=4.5, va="center")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].invert_yaxis(); ax[0].axhline(10, color=Pc.neutral.ink, lw=.5, ls="--"); ax[0].set_xlabel("cells (subsampled)"); ax[0].set_ylabel("own-motif rank (HOMER, of 472)")
    cv = pd.DataFrame(cur).dropna(); ax[1].plot(cv.cells, cv.frac_recovered, "o-", color=Pc.neutral.ink, ms=3); ax[1].set_xscale("log"); ax[1].set_ylim(0, max(.3, cv.frac_recovered.max() * 1.2)); ax[1].set_xlabel("cells (subsampled)"); ax[1].set_ylabel("fraction of TFs recovering own motif")
    for x, ls in [(1508, ":"), (3331, "--"), (5837, ":")]:
        for a in ax[:2]: a.axvline(x, color=Pc.neutral.mid, lw=.5, ls=ls)
    mm = m.dropna(subset=["own_motif_ic"]); ax[2].scatter(mm.cells, mm.own_motif_ic, s=10, c=[Pc.evidence.hic_loop if r else Pc.neutral.light for r in mm.recovered_any], edgecolor=Pc.neutral.ink, lw=.3)
    for r in mm[mm.recovered_any].itertuples(): ax[2].text(r.cells * 1.05, r.own_motif_ic, r.tf, fontsize=4.5)
    ax[2].set_xscale("log"); ax[2].set_xlabel("cells per library (full depth)"); ax[2].set_ylabel("own motif information content (bits)")
    fig.suptitle("Own-motif recovery in single-cell Tn5 TF libraries (red = recovered, rank <= 10 on both chromosome halves); dashed = panel median depth", fontsize=6); fig.tight_layout()
    fig.savefig(f"{D}/systematic_summary.png", dpi=200); fig.savefig(f"{D}/systematic_summary.svg")
