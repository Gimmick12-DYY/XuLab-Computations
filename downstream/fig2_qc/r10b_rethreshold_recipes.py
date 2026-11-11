#!/usr/bin/env python
"""Imputed audit part 3: re-thresholded imputer (r10a_rethreshold.py, rules a/b) through the same 1 kb tests.
Tracks per TF: fraction of cells positive per 1 kb bin, unmasked (raw U predicted) or masked (predicted kept only in the
ATAC union). Features (as the continuous arm of part 2): (i) log2((centre + e)/(flank + e)), centre = summit bin, flank =
mean of bins b-2, b+2, group means over summits, e = 1e-5; (ii) (i) minus the same for the pool = mean of the other
re-thresholded tracks (5 panel + 20 others). Recipes: 6x6 (preferred 2x, top 5000; halves top 2500) and HEK293T 2x2
(top 5000), 30 groups per reference; exact permutation p (720; 120 without CTCF). Reference rows (raw at 1 kb, bulk
positive control) are copied from part2_summary.tsv. Diagnostic: own / mean of the other 5 panel TFs at the summit bin.
Outputs OUT/imputed_audit/part3_*.tsv."""
import itertools
import subprocess

import numpy as np
import pandas as pd

from fig2_common import D, INP, OUT, ROOT, TFS, bin_index

OA = OUT / "imputed_audit"
OC = OA / "continuous"
G = 30
E = 1e-5
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
atac = fr["atac_union"]
cstart = np.zeros(N, np.int64); cend = np.zeros(N, np.int64)
for c, (o, n) in off.items():
    cstart[o:o + n] = o
    cend[o:o + n] = o + n
bins_frame = np.load(OC / "bins_frame.npy")
tlist = [t for t in (OC / "tf_list.txt").read_text().split()
         if (ROOT / f"unified/work/{t.lower()}/predict/rethreshold_audit_bins.npz").exists()]
assert all(t in tlist for t in TFS)
TR = {}
cuts = []
for t in tlist:
    z = np.load(ROOT / f"unified/work/{t.lower()}/predict/rethreshold_audit_bins.npz")
    cuts.append(dict(tf=t, cut_a=float(z["cut_a"]), cut_b=float(z["cut_b"]), frac_b=float(z["frac_b"]), n_cells_used=int(z["n_cells_used"])))
    for s in ("a", "b"):
        un = np.full(N, np.nan, np.float32); un[bins_frame] = z[f"{s}_union"]
        ro = np.full(N, np.nan, np.float32); ro[bins_frame] = z["raw_only"]
        TR[(s, "unmasked", t)] = un
        TR[(s, "masked", t)] = np.where(atac, un, ro)
pd.DataFrame(cuts).to_csv(OA / "part3_cutoffs.tsv", sep="\t", index=False, float_format="%.5g")
print(pd.DataFrame(cuts).round(4).to_string(), flush=True)


def flank(b):
    return [np.where((b + d >= cstart[b]) & (b + d < cend[b]), b + d, b) for d in (-2, 2)]


def groups(S, tfs, mask, half, n):
    cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
    h = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    gr = []
    for t in tfs:
        ix = np.flatnonzero((S.ref.values == t) & mask & h)
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def perm(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def feats(s, mk, t, b, fl, gr):
    a = TR[(s, mk, t)]
    oth = [u for u in tlist if u != t]
    P = np.nanmean(np.array([TR[(s, mk, u)] [np.r_[b, fl[0], fl[1]]] for u in oth]), 0)
    n = len(b)
    pc, pf0, pf1 = P[:n], P[n:2 * n], P[2 * n:]
    i_, ii_ = np.zeros(len(gr)), np.zeros(len(gr))
    for k, g in enumerate(gr):
        c, f = np.nanmean(a[b[g]]), np.nanmean(np.r_[a[fl[0][g]], a[fl[1][g]]])
        i_[k] = np.log2((c + E) / (f + E))
        ii_[k] = i_[k] - np.log2((np.nanmean(pc[g]) + E) / (np.nanmean(np.r_[pf0[g], pf1[g]]) + E))
    return i_, ii_


rows = []


def emit(recipe, arm, feat, half, B, tfs):
    k = len(tfs)
    d = dict(recipe=recipe, arm=arm, feature=feat, half=half,
             own_max_rows=int(sum(np.argmax(B[i]) == i for i in range(k))),
             own_max_cols=int(sum(np.argmax(B[:, j]) == j for j in range(k))))
    if k == 6:
        d["diag_minus_off"], d["perm_p"] = perm(B)
        kk = list(range(1, 6)); d["noCTCF_diag_minus_off"], d["noCTCF_perm_p"] = perm(B[np.ix_(kk, kk)])
    else:
        d["diag_minus_off"] = float((B[0, 0] + B[1, 1]) / 2 - (B[0, 1] + B[1, 0]) / 2)
        d["r_CTCF_vs_ZNF282ChIP"], d["r_ZNF282_vs_CTCFChIP"] = B[0, 1], B[1, 0]
    for i, t in enumerate(tfs):
        d[f"r_own_{t}"] = B[i, i]
    rows.append(d)
    pd.DataFrame(B, tfs, tfs).to_csv(OA / f"part3_{recipe}_{arm}_{feat}_{half}.tsv", sep="\t", float_format="%.4f")


S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
Bk6 = S6[[f"bulk_{t}" for t in TFS]].values
S2 = pd.read_csv(OUT / "hek293t_2x2/summits.tsv", sep="\t", low_memory=False)
T2 = ["CTCF", "ZNF282"]
bb = OA / "w250_r9.bed"
BAM2 = [ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", D / "ZNF282_HEK293.bam"]
txt = subprocess.run(["bedtools", "multicov", "-bams", *map(str, BAM2), "-bed", str(bb)], check=True, capture_output=True, text=True).stdout
Bk2 = np.zeros((len(S2), 2))
for l in txt.splitlines():
    p_ = l.split("\t")
    Bk2[int(p_[3])] = [float(x) for x in p_[4:]]
for recipe, S, Bk, tfs, mask, halves in (
        ("6x6", S6, Bk6, TFS, S6.bulk_spec.values > 1, (("all", 5000), ("odd", 2500), ("even_X", 2500))),
        ("2x2", S2, Bk2, T2, S2.preferred.values.astype(bool), (("all", 5000), ("odd", 5000), ("even_X", 5000)))):
    b_all = np.array([off[c][0] for c in S.chrom.values]) + S.pos.values // 1000
    fl = flank(b_all)
    for half, n in halves:
        gr = groups(S, tfs, mask, half, n)
        gB = np.array([Bk[g].sum(0) for g in gr])
        bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
        for s in ("a", "b"):
            for mk in ("unmasked", "masked"):
                ff = [feats(s, mk, t, b_all, fl, gr) for t in tfs]
                for fi, fname in ((0, "i_centre_flank"), (1, "ii_vs_pool")):
                    X = np.column_stack([f[fi] for f in ff])
                    B = np.corrcoef(np.column_stack([bulk, X]).T)[len(tfs):, :len(tfs)]
                    emit(recipe, f"rethr_{s}_{mk}", fname, half, B, tfs)
p2 = pd.read_csv(OA / "part2_summary.tsv", sep="\t")
ref = p2[p2.arm.isin(["raw_1kb", "positive_control"])]
sm = pd.concat([pd.DataFrame(rows), ref], ignore_index=True)
sm.to_csv(OA / "part3_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
c = [x for x in ["recipe", "arm", "feature", "half", "own_max_rows", "own_max_cols", "diag_minus_off", "perm_p", "noCTCF_perm_p",
                 "r_own_CTCF", "r_own_MAZ", "r_own_ZBTB7A", "r_own_ZIC2", "r_own_ZNF282", "r_own_ZNF777", "r_CTCF_vs_ZNF282ChIP",
                 "r_ZNF282_vs_CTCFChIP"] if x in sm]
print(sm[c].round(3).to_string())

# diagnostic: own / mean of other 5 panel TFs at summit bins
mask6 = S6.bulk_spec.values > 1
b6 = np.array([off[c][0] for c in S6.chrom.values]) + S6.pos.values // 1000
dg = []
for refT in TFS:
    ix = np.flatnonzero((S6.ref.values == refT) & mask6)
    ix = ix[np.argsort(-S6.signal.values[ix], kind="stable")][:5000]
    for s in ("a", "b"):
        for mk in ("unmasked", "masked"):
            cen = {t: float(np.nanmean(TR[(s, mk, t)][b6[ix]])) for t in TFS}
            for t in TFS:
                dg.append(dict(summit_set=refT, scheme=f"{s}_{mk}", tf=t, centre=cen[t],
                               own_over_mean_other5=cen[t] / np.mean([cen[u] for u in TFS if u != t])))
dg = pd.DataFrame(dg)
dg.to_csv(OA / "part3_diagnostic.tsv", sep="\t", index=False, float_format="%.4g")
print(dg[dg.summit_set == "CTCF"].pivot_table(index="tf", columns="scheme", values="own_over_mean_other5").round(3).to_string())
print(dg[dg.tf == dg.summit_set].pivot_table(index="tf", columns="scheme", values="own_over_mean_other5").round(3).to_string())
