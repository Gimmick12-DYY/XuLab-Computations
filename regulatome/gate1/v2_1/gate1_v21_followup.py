# Gate 1 v2.1 follow-ups (saved profiles only):
# 1) TF-level tier over the 40 deep TFs: formal v2.1 call (v1 residual) vs count-aware calls at expected >= 5 and >= 2.
#    TF-level = formal pass that is pass / borderline / unscoreable under count-aware; plus formal-borderline TFs that pass
#    count-aware at both thresholds (flagged borderline). A count-aware FAIL removes the TF.
# 2) PRC2 vs NuRD: (a) module specificity with shared members removed from both sides (PRC2 core = EZH2, MTF2, PCGF6 vs
#    NuRD core = MBD3; comparators = non-member deep TFs + other modules / module-minus-shared variants);
#    (b) direct split-half margins PRC2 core vs MBD3 (both directions) on all genes, CpG-island genes and non-CGI genes
#    (CGI promoter: TSS +/- 500 bp CpG o/e >= 0.6 and GC >= 0.5), and the profile difference by class.
# 3) leiden_1 minus E4F1: members, cells, library group.
import glob, json, os, subprocess
import numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu
G = "/work/users/d/y/dyy12/XuLab/regulatome/gate1"; V = f"{G}/v2_1"; X = "/work/users/d/y/dyy12/XuLab"
DEEP = 3400
cells = {l.strip(): json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{l.strip()}.json"))["cells"]
         for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()}
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
group = (F.all_elements > 0.22).map({True: "high", False: "low"})

# ---- 1) tier table --------------------------------------------------------------------------------------------------------
formal = pd.read_csv(f"{V}/gate1_v2_1_calls.tsv", sep="\t", index_col=0)
def ca(path):
    t = pd.read_csv(path, sep="\t"); t = t[(t.rep == "raw") & (t.arm == "abc")].set_index("unit")
    return t[["call", "genes_used", "margin", "splits_margin_pos", "n_splits"]]
e5, e2 = ca(f"{V}/gate1_v2_1_rerun_tf.tsv"), ca(f"{V}/gate1_v2_1_rerun_tf_e2.tsv")
deep = [t for t in formal.index if formal.loc[t, "n_cells"] >= DEEP]
rows = []
for t in deep:
    f_call = formal.loc[t, "call"]
    c5 = e5.loc[t, "call"] if t in e5.index else "unscoreable (< 50 genes with expected >= 5)"
    c2 = e2.loc[t, "call"] if t in e2.index else "unscoreable"
    g5 = float(e5.loc[t, "genes_used"]) if t in e5.index else 0.0
    if t in e5.index and g5 < 200: c5 = f"{c5} (low genes: {g5:.0f})"
    failed = any(str(c).startswith("fail") for c in (c5, c2))
    if f_call == "pass" and not failed: tier, flag = "TF-level", ("unscoreable at expected >= 5" if not str(c5).startswith(("pass", "borderline")) else ".")
    elif f_call == "borderline" and str(c5).startswith("pass") and str(c2).startswith("pass"): tier, flag = "TF-level", "borderline (formal), pass count-aware"
    elif failed: tier, flag = "out", "fails count-aware"
    else: tier, flag = "out", "."
    rows.append(dict(tf=t, n_cells=formal.loc[t, "n_cells"], group=formal.loc[t, "group"], formal_v2_1=f_call,
                     formal_margin=formal.loc[t, "margin"], formal_splits=formal.loc[t, "splits_margin_pos"],
                     countaware_e5=c5, genes_e5=g5, margin_e5=e5.loc[t, "margin"] if t in e5.index else np.nan,
                     splits_e5=f"{e5.loc[t, 'splits_margin_pos']:.0f}/{e5.loc[t, 'n_splits']:.0f}" if t in e5.index else ".",
                     countaware_e2=c2, margin_e2=e2.loc[t, "margin"] if t in e2.index else np.nan,
                     splits_e2=f"{e2.loc[t, 'splits_margin_pos']:.0f}/{e2.loc[t, 'n_splits']:.0f}" if t in e2.index else ".",
                     tier=tier, flag=flag))
TT = pd.DataFrame(rows).sort_values(["tier", "n_cells"], ascending=[False, False])
TT.to_csv(f"{V}/tf_level_tier.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 260)
print(TT[TT.tier == "TF-level"].to_string(index=False))
print("formal passes not in tier:", TT[(TT.formal_v2_1 == "pass") & (TT.tier != "TF-level")][["tf", "countaware_e5", "countaware_e2"]].to_string(index=False))
print("pass only at expected >= 2:", list(TT[(~TT.countaware_e5.str.startswith("pass")) & TT.countaware_e2.str.startswith("pass")].tf))

# ---- 2) PRC2 vs NuRD ------------------------------------------------------------------------------------------------------
prof = {}
for f in glob.glob(f"{V}/results/*.profiles.npz") + glob.glob(f"{V}/results_modules/*.profiles.npz"):
    z = np.load(f, allow_pickle=True); prof[f.split("/")[-1].replace(".profiles.npz", "")] = z["OE"]; genes = z["genes"]
NS = {u: v.shape[0] for u, v in prof.items()}
def half(u, k, h, emin):
    o, e = prof[u][k, 0 if h == "A" else 2], prof[u][k, 1 if h == "A" else 3]
    return np.log2((o + 0.5) / (e + 0.5)), e >= emin
def rho(x, y, m):
    return float(np.corrcoef(rankdata(x[m]), rankdata(y[m]))[0, 1]) if m.sum() >= 50 else np.nan
def margins(u, comps, emin, gmask=None):
    out = {}
    for v in comps:
        ms = []
        for k in range(min(NS[u], NS[v])):
            ra, ea = half(u, k, "A", emin); rb, eb = half(u, k, "B", emin); rv, ev = half(v, k, "B", emin)
            g = ea & eb & ev if gmask is None else ea & eb & ev & gmask
            o_, x_ = rho(ra, rb, g), rho(ra, rv, g)
            if np.isfinite(o_) and np.isfinite(x_): ms.append(o_ - x_)
        if ms: out[v] = (float(np.mean(ms)), int(np.sum(np.array(ms) > 0)), len(ms))
    return out

# CpG-island status of each gene's TSS
A = pd.read_csv(f"{X}/regulatome/m2_abc/results/Predictions_powerlaw/EnhancerPredictionsFull_thr0.02.tsv", sep="\t",
                usecols=["chr", "TargetGene", "TargetGeneTSS"]).drop_duplicates("TargetGene").set_index("TargetGene")
tss = A.reindex(genes)
bed = f"{V}/gene_tss500.bed"
pd.DataFrame({"c": tss.chr.values, "s": (tss.TargetGeneTSS - 500).clip(lower=0).astype(int).values,
              "e": (tss.TargetGeneTSS + 500).astype(int).values, "g": genes}).to_csv(bed, sep="\t", header=False, index=False)
fa = subprocess.run(f"bedtools getfasta -fi {X}/downstream/cache/hg38.fa -bed {bed} -nameOnly -tab", shell=True,
                    capture_output=True, text=True, check=True).stdout.strip().split("\n")
cg = {}
for l in fa:
    n, s = l.split("\t"); s = s.upper(); c, g_ = s.count("C"), s.count("G")
    cg[n.split("::")[0]] = ((c + g_) / len(s), s.count("CG") * len(s) / (c * g_) if c and g_ else 0)
cgi = np.array([cg.get(g, (0, 0))[0] >= 0.5 and cg.get(g, (0, 0))[1] >= 0.6 for g in genes])
print(f"\nCGI promoters: {cgi.sum():,} of {len(genes):,} genes")

P2, N2 = "PRC2__minus_RBBP4_RBBP7", "NuRD__minus_RBBP4_RBBP7"
full_mods = [u for u in prof if u in ("G9a_WIZ", "HES1_RBPJ", "HP1_ChAHP", "MAZ_ZNF143", "cohesin_CTCF", "PRC2", "NuRD", "MEF2") or
             (u.startswith("leiden_") and "__minus_" not in u)]
deep_tfs = [t for t in cells if cells[t] >= DEEP and t in prof]
out2 = []
for emin in (5, 2):
    for u, mem, swap in ((P2, ["EZH2", "MTF2", "PCGF6"], {"NuRD": N2}), (N2, ["MBD3"], {"PRC2": P2})):
        comps = [v for v in deep_tfs if v not in mem + ["RBBP4", "RBBP7"]] + [swap.get(m, m) for m in full_mods if m not in ("PRC2", "NuRD")] + list(swap.values())
        comps = [c for c in comps if c in prof and c != u]
        mg = margins(u, comps, emin)
        w = min(mg, key=lambda v: mg[v][0])
        out2.append(dict(test="(a) shared RBBP4/7 removed from both sides", unit=u, emin=emin, worst_comparator=w,
                         margin=mg[w][0], splits_pos=f"{mg[w][1]}/{mg[w][2]}",
                         vs_other_core=f"{mg[list(swap.values())[0]][0]:+.3f} ({mg[list(swap.values())[0]][1]}/{mg[list(swap.values())[0]][2]})"))
    for nm, gm in (("all genes", None), ("CGI genes", cgi), ("non-CGI genes", ~cgi)):
        for u, v in ((P2, N2), (N2, P2)):
            mg = margins(u, [v], emin, gm)
            if v in mg: out2.append(dict(test=f"(b) direct, {nm}", unit=u, emin=emin, worst_comparator=v, margin=mg[v][0], splits_pos=f"{mg[v][1]}/{mg[v][2]}"))
O2 = pd.DataFrame(out2); O2.to_csv(f"{V}/prc2_vs_nurd.tsv", sep="\t", index=False, float_format="%.4f")
print(O2.round(3).to_string(index=False))
# profile difference by gene class (pooled halves, mean over splits; genes with expected >= 5 in both units)
def pooled(u):
    o = prof[u][:, 0] + prof[u][:, 2]; e = prof[u][:, 1] + prof[u][:, 3]
    return np.log2((o.mean(0) + 0.5) / (e.mean(0) + 0.5)), e.mean(0)
rp, ep = pooled(P2); rn, en = pooled(N2); ok = (ep >= 10) & (en >= 10)
for nm, gm in (("CGI", cgi), ("non-CGI", ~cgi)):
    m = ok & gm
    print(f"{nm}: genes {m.sum():,}; mean log2 obs/exp PRC2core {rp[m].mean():+.3f}, MBD3 {rn[m].mean():+.3f}; "
          f"Spearman(PRC2core, MBD3) {np.corrcoef(rankdata(rp[m]), rankdata(rn[m]))[0, 1]:.3f}")
d_ = rp - rn
print(f"PRC2core - MBD3 difference, CGI vs non-CGI: {d_[ok & cgi].mean():+.3f} vs {d_[ok & ~cgi].mean():+.3f}, "
      f"MW p = {mannwhitneyu(d_[ok & cgi], d_[ok & ~cgi])[1]:.1e}")

# ---- 3) leiden_1 minus E4F1 composition --------------------------------------------------------------------------------------
mem = "DR1,ETS2,HOXA10,HOXB6,HOXC9,POGK,SREBF2,ZIC2,ZNF512,ZUFSP".split(",")
L = pd.DataFrame({"tf": mem, "cells": [cells[t] for t in mem], "group": [group[t] for t in mem]}).sort_values("cells", ascending=False)
L.to_csv(f"{V}/leiden_1_minus_E4F1_composition.tsv", sep="\t", index=False)
print("\nleiden_1 minus E4F1:", L.to_string(index=False), "\ntotal cells", L.cells.sum(), "| groups", L.group.value_counts().to_dict())
