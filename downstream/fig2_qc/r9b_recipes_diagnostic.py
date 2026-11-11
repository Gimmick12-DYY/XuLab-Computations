#!/usr/bin/env python
"""Imputed audit, part 2. All arms go through the same 1 kb recipe; features declared up front.

Arms: continuous (model pseudobulk, pre-binarization, pre-mask; 6 panel TFs all cells + 20 other TFs 3000 cells, pool =
mean of the other tracks), unmasked (binarized raw U predicted before the ATAC mask; pool = sum of the other 77 tracks,
ZNF768 has no unmasked matrix), masked (final imputed; pool = other 78), raw_1kb (reads per 1 kb bin, ZNF777 depth, 3 seeds;
pool = other 78), positive_control (bulk ChIP reads in 1 kb bins; pool = other 5 bulk tracks; 6x6 all chromosomes only).
Features: (i) centre bin vs flank bins b-2 and b+2 within the track (no pool); (ii) same minus the same for the pool.
Group level: counts summed over the group's summits with +.5; continuous scores averaged over summits (+1e-6).
Recipes: 6x6 (preferred 2x; top 5000; halves top 2500) and HEK293T 2x2 (top 5000), 30 groups per reference.
Diagnostic D: per TF, at CTCF (and own) summits, at each step: raw, continuous, thresholded (fraction of cells above threshold,
pre-mask), unmasked binarized, masked final: centre value, centre/flank, own / mean of the other 5 panel TFs.
Outputs OUT/imputed_audit/: part2_summary.tsv, part2_<recipe>_<arm>_<feature>_<half>.tsv, diagnostic_steps.tsv, thresholds.tsv.
"""
import itertools
import json
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fig2_common import COUNTS, BAM, D, DM_DEPTH, DM_SEEDS, INP, OUT, ROOT, TFS, bin_index

OA = OUT / "imputed_audit"
OC = OA / "continuous"
G = 30
NT6 = 6
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = [str(t) for t in fr["tf79"]]
ncell = fr["ncells_imp79"].astype(float)
cstart = np.zeros(N, np.int64); cend = np.zeros(N, np.int64)
for c, (o, n) in off.items():
    cstart[o:o + n] = o
    cend[o:o + n] = o + n
reg = pd.read_csv(ROOT / "unified/work/ctcf/impute/regions.tsv", header=None)[0].str.extract(r"^(chr[^:]+):(\d+)-")
fb = np.array([off[c][0] for c in reg[0]]) + reg[1].astype(np.int64).values // 1000


def load_csr(p):
    try:
        return sp.load_npz(p).tocsr()
    except Exception:
        z = np.load(p)
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


# ---- tracks ----
tlist = [t for t in (OC / "tf_list.txt").read_text().split()
         if (ROOT / f"unified/work/{t.lower()}/predict/pseudobulk_audit_bins.npz").exists()]
assert all(t in tlist for t in TFS), "panel continuous scores missing"
print("continuous tracks available:", len(tlist), flush=True)
bins_frame = np.load(OC / "bins_frame.npy")
CONT, FRAC, NCONT = {}, {}, {}
for t in tlist:
    z = np.load(ROOT / f"unified/work/{t.lower()}/predict/pseudobulk_audit_bins.npz")
    a = np.full(N, np.nan, np.float32); f = a.copy()
    a[bins_frame] = z["mean_score"]; f[bins_frame] = z["frac_above_thr"]
    CONT[t], FRAC[t] = a, f
    NCONT[t] = (int(z["n_cells_used"]), int(z["n_cells_total"]), float(z["threshold"]))
print("continuous tracks", len(CONT), flush=True)
UNM = {}
for t in tf79:
    p = ROOT / f"unified/work/{t.lower()}/impute/matrix_csr.npz.pre_openmask"
    if not p.exists():
        continue
    m = load_csr(p)
    if m.shape[0] != len(fb):
        m = m.T.tocsr()
    v = np.zeros(N, np.float64)
    v[fb] = np.asarray((m > 0).sum(1)).ravel()
    UNM[t] = v
print("unmasked tracks", len(UNM), flush=True)
UPOOL = np.sum(list(UNM.values()), 0)
imp79 = sp.load_npz(INP / "imp79.npz").tocsr()
MASKED = {t: imp79[tf79.index(t)].toarray().ravel().astype(float) for t in TFS}
MPOOL = np.asarray(imp79.sum(0)).ravel().astype(float)
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
RAW = {t: raw[tf79.index(t)].toarray().ravel().astype(float) for t in TFS}
RPOOL = np.asarray(raw.sum(0)).ravel().astype(float)
rtot = np.asarray(raw.sum(1)).ravel().astype(float)
cells = sp.load_npz(COUNTS / "cells.npz").tocsr()


def flank_bins(b):
    out = []
    for d in (-2, 2):
        j = b + d
        j = np.where((j >= cstart[b]) & (j < cend[b]), j, -1)
        out.append(j)
    return out


def take(a, j):
    v = np.zeros(len(j))
    ok = j >= 0
    v[ok] = a[j[ok]]
    return v


def groups(S, tfs, mask, half, n):
    cn = S.chrom.str.replace("chr", "").replace("X", "23").astype(int).values
    h = {"all": np.ones(len(S), bool), "odd": (cn % 2 == 1) & (cn != 23), "even_X": (cn % 2 == 0) | (cn == 23)}[half]
    gr = []
    for t in tfs:
        ix = np.flatnonzero((S.ref.values == t) & mask & h)
        ix = ix[np.argsort(-S.signal.values[ix], kind="stable")][:n]
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def gbulk(Bk, gr):
    gB = np.array([Bk[g].sum(0) for g in gr])
    return np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)


def perm(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


def cf_counts(a, b, flb):
    return a[b], take(a, flb[0]) + take(a, flb[1])


def feat_count(track, pool_other, b, flb, gr):
    """track, pool_other: full-frame arrays. Returns (i, ii) group vectors."""
    c, f = cf_counts(track, b, flb)
    cp, fp = cf_counts(pool_other, b, flb)
    i_, ii_ = np.zeros(len(gr)), np.zeros(len(gr))
    for a, g in enumerate(gr):
        i_[a] = np.log2((c[g].sum() + .5) / (f[g].sum() + .5))
        ii_[a] = i_[a] - np.log2((cp[g].sum() + .5) / (fp[g].sum() + .5))
    return i_, ii_


def feat_cont(t, b, flb, gr):
    C = CONT[t]
    others = [u for u in tlist if u != t]
    Cb = C[b]; F0 = C[np.where(flb[0] >= 0, flb[0], b)]; F1 = C[np.where(flb[1] >= 0, flb[1], b)]
    pb = np.nanmean(np.array([CONT[u][b] for u in others]), 0)
    pf0 = np.nanmean(np.array([CONT[u][np.where(flb[0] >= 0, flb[0], b)] for u in others]), 0)
    pf1 = np.nanmean(np.array([CONT[u][np.where(flb[1] >= 0, flb[1], b)] for u in others]), 0)
    i_, ii_ = np.zeros(len(gr)), np.zeros(len(gr))
    e = 1e-6
    for a, g in enumerate(gr):
        cv, fv = np.nanmean(Cb[g]), np.nanmean(np.r_[F0[g], F1[g]])
        i_[a] = np.log2((cv + e) / (fv + e))
        pcv, pfv = np.nanmean(pb[g]), np.nanmean(np.r_[pf0[g], pf1[g]])
        ii_[a] = i_[a] - np.log2((pcv + e) / (pfv + e))
    return i_, ii_


def block_of(bulk, X):
    k = bulk.shape[1]
    return np.corrcoef(np.column_stack([bulk, X]).T)[k:, :k]


rows = []
S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
Bk6 = S6[[f"bulk_{t}" for t in TFS]].values
mask6 = S6.bulk_spec.values > 1
S2 = pd.read_csv(OUT / "hek293t_2x2/summits.tsv", sep="\t", low_memory=False)
T2 = ["CTCF", "ZNF282"]
BAM2 = [ROOT / "chip_gse103651/work/bam/gse103651.pooled.bam", D / "ZNF282_HEK293.bam"]
bb = OA / "w250_r9.bed"
o_ = np.lexsort((S2.pos.values, S2.chrom.values))
pd.DataFrame({0: S2.chrom.values[o_], 1: S2.pos.values[o_] - 250, 2: S2.pos.values[o_] + 250, 3: o_}).to_csv(
    bb, sep="\t", header=False, index=False)
out = subprocess.run(["bedtools", "multicov", "-bams", *map(str, BAM2), "-bed", str(bb)], check=True,
                     capture_output=True, text=True).stdout
Bk2 = np.zeros((len(S2), 2))
for l in out.splitlines():
    p_ = l.split("\t")
    Bk2[int(p_[3])] = [float(x) for x in p_[4:]]
mask2 = S2.preferred.values.astype(bool)


def emit(recipe, arm, feat, half, B, tfs):
    k = len(tfs)
    d = dict(recipe=recipe, arm=arm, feature=feat, half=half)
    d["own_max_rows"] = int(sum(np.argmax(B[i]) == i for i in range(k)))
    d["own_max_cols"] = int(sum(np.argmax(B[:, j]) == j for j in range(k)))
    if k == 6:
        o, p = perm(B); d["diag_minus_off"], d["perm_p"] = o, p
        keep = list(range(1, 6)); d["noCTCF_diag_minus_off"], d["noCTCF_perm_p"] = perm(B[np.ix_(keep, keep)])
    else:
        d["diag_minus_off"] = float((B[0, 0] + B[1, 1]) / 2 - (B[0, 1] + B[1, 0]) / 2)
        d["r_CTCF_vs_ZNF282ChIP"], d["r_ZNF282_vs_CTCFChIP"] = B[0, 1], B[1, 0]
    for i, t in enumerate(tfs):
        d[f"r_own_{t}"] = B[i, i]
    rows.append(d)
    pd.DataFrame(B, tfs, tfs).to_csv(OA / f"part2_{recipe}_{arm}_{feat}_{half}.tsv", sep="\t", float_format="%.4f")
    print(recipe, arm, feat, half, {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in d.items() if k_ not in ("recipe", "arm", "feature", "half")}, flush=True)


def run(recipe, S, Bk, tfs, mask, halves):
    b_all = np.array([off[c][0] for c in S.chrom.values]) + S.pos.values // 1000
    flb_all = flank_bins(b_all)
    for half, n in halves:
        gr = groups(S, tfs, mask, half, n)
        bulk = gbulk(Bk, gr)
        arms = {}
        arms["continuous"] = [np.column_stack(x) for x in zip(*[feat_cont(t, b_all, flb_all, gr) for t in tfs])]
        un = [feat_count(UNM[t], UPOOL - UNM[t], b_all, flb_all, gr) for t in tfs]
        arms["unmasked"] = [np.column_stack([u[f] for u in un]) for f in (0, 1)]
        ma = [feat_count(MASKED[t], MPOOL - MASKED[t], b_all, flb_all, gr) for t in tfs]
        arms["masked"] = [np.column_stack([u[f] for u in ma]) for f in (0, 1)]
        for arm, (fi, fii) in arms.items():
            emit(recipe, arm, "i_centre_flank", half, block_of(bulk, fi), tfs)
            emit(recipe, arm, "ii_vs_pool", half, block_of(bulk, fii), tfs)
        # raw 1 kb, ZNF777 depth
        bi_, bii_ = [], []
        for s in DM_SEEDS:
            ri = []; rii = []
            for t in tfs:
                k = tf79.index(t)
                p = min(1.0, DM_DEPTH / rtot[k])
                th = np.random.default_rng(s * 100 + k).binomial(RAW[t].astype(np.int64), p).astype(float)
                i_, _ = feat_count(th, RPOOL - RAW[t], b_all, flb_all, gr)
                cp, fp = cf_counts(RPOOL - RAW[t], b_all, flb_all)
                ii_ = i_ - np.array([np.log2((cp[g].sum() + .5) / (fp[g].sum() + .5)) for g in gr])
                ri.append(i_); rii.append(ii_)
            bi_.append(block_of(bulk, np.column_stack(ri))); bii_.append(block_of(bulk, np.column_stack(rii)))
        emit(recipe, "raw_1kb", "i_centre_flank", half, np.mean(bi_, 0), tfs)
        emit(recipe, "raw_1kb", "ii_vs_pool", half, np.mean(bii_, 0), tfs)


run("6x6", S6, Bk6, TFS, mask6, (("all", 5000), ("odd", 2500), ("even_X", 2500)))
run("2x2", S2, Bk2, T2, mask2, (("all", 5000), ("odd", 5000), ("even_X", 5000)))

# positive control: bulk ChIP binned the same way (6x6, all chromosomes)
gr = groups(S6, TFS, mask6, "all", 5000)
sel = np.unique(np.concatenate(gr))
b_all = np.array([off[c][0] for c in S6.chrom.values]) + S6.pos.values // 1000
flb = flank_bins(b_all)
needed = np.unique(np.concatenate([b_all[sel], flb[0][sel][flb[0][sel] >= 0], flb[1][sel][flb[1][sel] >= 0]]))
inv_chrom = np.searchsorted(bi.offset.values, needed, side="right") - 1
bed = OA / "posctrl_bins.bed"
pd.DataFrame({0: bi.chrom.values[inv_chrom], 1: (needed - bi.offset.values[inv_chrom]) * 1000,
              2: (needed - bi.offset.values[inv_chrom] + 1) * 1000}).sort_values([0, 1]).to_csv(bed, sep="\t", header=False, index=False)
txt = subprocess.run(["bedtools", "multicov", "-bams", *[str(BAM[t]) for t in TFS], "-bed", str(bed)], check=True,
                     capture_output=True, text=True).stdout
BB = np.zeros((6, N))
for l in txt.splitlines():
    p_ = l.split("\t")
    idx = off[p_[0]][0] + int(p_[1]) // 1000
    BB[:, idx] = [float(x) for x in p_[3:]]
PB = BB.sum(0)
bulk = gbulk(Bk6, gr)
fi, fii = zip(*[feat_count(BB[i], PB - BB[i], b_all, flb, gr) for i in range(6)])
emit("6x6", "positive_control", "i_centre_flank", "all", block_of(bulk, np.column_stack(fi)), TFS)
emit("6x6", "positive_control", "ii_vs_pool", "all", block_of(bulk, np.column_stack(fii)), TFS)
sm = pd.DataFrame(rows)
sm.to_csv(OA / "part2_summary.tsv", sep="\t", index=False, float_format="%.4f")

# ---- Diagnostic D ----
thr = []
for t in TFS:
    m = json.loads((ROOT / f"unified/work/{t.lower()}/impute/meta.json").read_text())
    n_b, n_c = m["shape"]
    thr.append(dict(tf=t, n_cells=n_c, threshold=m["threshold"], raw_nnz=m["raw_nnz"], predicted_nnz_pre_mask=m.get("predicted_nnz_pre_mask"),
                    predicted_nnz_masked=m["predicted_nnz"], output_nnz=m["output_nnz"],
                    frac_pos_raw=m["raw_nnz"] / (n_b * n_c), frac_pos_pred_pre_mask=(m.get("predicted_nnz_pre_mask") or np.nan) / (n_b * n_c),
                    frac_pos_output=m["output_nnz"] / (n_b * n_c)))
pd.DataFrame(thr).to_csv(OA / "thresholds.tsv", sep="\t", index=False, float_format="%.4g")
print(pd.DataFrame(thr).round(6).to_string(), flush=True)
b_all = np.array([off[c][0] for c in S6.chrom.values]) + S6.pos.values // 1000
flb = flank_bins(b_all)
ncell_t = {t: ncell[tf79.index(t)] for t in TFS}
diag = []
for ref in TFS:
    ix = np.flatnonzero((S6.ref.values == ref) & mask6)
    ix = ix[np.argsort(-S6.signal.values[ix], kind="stable")][:5000]
    sb = b_all[ix]; fl = [flb[0][ix], flb[1][ix]]
    def getv(a, j):
        return np.where(j >= 0, a[np.where(j >= 0, j, 0)], np.nan)
    step = {}
    for t in TFS:
        k = tf79.index(t)
        cnt = cells[k].toarray().ravel() / ncell_t[t]
        step[t] = {"raw_positive_cell_frac": cnt, "continuous_mean_score": CONT[t], "frac_cells_above_threshold_premask": FRAC[t],
                   "unmasked_binarized_frac": UNM[t] / ncell_t[t], "masked_final_frac": MASKED[t] / ncell_t[t]}
    for st in ("raw_positive_cell_frac", "continuous_mean_score", "frac_cells_above_threshold_premask", "unmasked_binarized_frac", "masked_final_frac"):
        centre = {t: float(np.nanmean(step[t][st][sb])) for t in TFS}
        flank = {t: float(np.nanmean(np.r_[getv(step[t][st], fl[0]), getv(step[t][st], fl[1])])) for t in TFS}
        for t in TFS:
            oth = np.mean([centre[u] for u in TFS if u != t])
            diag.append(dict(summit_set=ref, step=st, tf=t, centre=centre[t], flank=flank[t], centre_over_flank=centre[t] / max(flank[t], 1e-12),
                             own_over_mean_other5=centre[t] / max(oth, 1e-12)))
dg = pd.DataFrame(diag)
dg.to_csv(OA / "diagnostic_steps.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(dg[dg.summit_set == "CTCF"].pivot_table(index="tf", columns="step", values="own_over_mean_other5").round(3).to_string())
print(dg[(dg.summit_set == "CTCF")].pivot_table(index="tf", columns="step", values="centre_over_flank").round(2).to_string())
print(dg[(dg.tf == dg.summit_set)].pivot_table(index="tf", columns="step", values="own_over_mean_other5").round(3).to_string())
