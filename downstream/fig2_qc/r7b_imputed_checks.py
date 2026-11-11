#!/usr/bin/env python
"""Sanity checks on the imputed loaders and features used in r6_imputed_samerecipe.py, also applied to raw.
 1 coordinates: chr1 bin count of unified/work/ctcf/impute/regions.tsv vs hg38 (248,957) and hg19 (249,251); same map as
   downstream/imputation_auprc/imputation_auprc_bulk.py; imputed CTCF at five CTCF sites.
 2 reproduce imputed CTCF AUPRC/AUROC vs CTCF bulk (all 1 kb bins minus blacklist) with these loaders.
 3 aggregate profile: mean per-cell signal in bins -10..+10 around top-5000 preferred summits, own vs mean of the other 78; raw too.
 5 feature direction: plain group value (no pool) and pool ratio with a cell-normalised pseudocount.
 6 masking: fraction of summit bins in the ATAC mask; 6x6 restricted to summits inside the mask.
Outputs OUT/imputed_audit/checks/."""
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

from fig2_common import COUNTS, DM_DEPTH, DM_SEEDS, INP, OUT, ROOT, TFS, bin_index

OC = OUT / "imputed_audit" / "checks"
OC.mkdir(parents=True, exist_ok=True)
G = 30
bi, off, N = bin_index()
fr = np.load(INP / "frame.npz", allow_pickle=True)
tf79 = list(fr["tf79"])
atac, uni = fr["atac_union"], fr["universe"]
ncell = fr["ncells_imp79"].astype(float)
imp = sp.load_npz(INP / "imp79.npz").tocsr()
raw = sp.load_npz(COUNTS / "raw.npz").tocsr()
tot = np.asarray(raw.sum(1)).ravel().astype(float)
impT = {t: imp[tf79.index(t)].toarray().ravel().astype(float) for t in TFS}
rawT = {t: raw[tf79.index(t)].toarray().ravel().astype(float) for t in TFS}
impF = (sp.diags(1 / ncell) @ imp)
frac_sum = np.asarray(impF.sum(0)).ravel()
raw_cpm_sum = np.asarray((sp.diags(1e6 / tot) @ raw).sum(0)).ravel()

# 1 coordinates
reg = pd.read_csv(ROOT / "unified/work/ctcf/impute/regions.tsv", header=None)[0]
nchr1 = int(reg.str.startswith("chr1:").sum())
print(f"[1] regions.tsv: chr1 bins {nchr1} (hg38 248,957; hg19 249,251); total {len(reg)}; first {reg.iloc[0]} last {reg.iloc[-1]}", flush=True)
rs = reg.str.extract(r"^(chr[^:]+):(\d+)-(\d+)")
mapped = np.array([off[c][0] for c in rs[0]]) + rs[1].astype(np.int64).values // 1000
print(f"[1] regions mapped to unique frame bins: {len(np.unique(mapped))} of {len(mapped)}; bin width all 1000: {bool((rs[2].astype(int) - rs[1].astype(int) == 1000).all())}", flush=True)
S6 = pd.read_csv(OUT / "round2/summits.tsv", sep="\t", low_memory=False)
top = S6[(S6.ref == "CTCF")].sort_values("signal", ascending=False).head(3)
sites = [("chr17", 18960982), ("chr1", 225777658)] + list(zip(top.chrom, top.pos.astype(int)))
for c, x in sites:
    b = off[c][0] + x // 1000
    print(f"[1] {c}:{x}: imputed CTCF +-2 bins {impT['CTCF'][b-2:b+3].astype(int).tolist()}; raw CTCF {rawT['CTCF'][b-2:b+3].astype(int).tolist()}; "
          f"mask {atac[b-2:b+3].astype(int).tolist()}", flush=True)
b0 = off["chr11"][0]
print(f"[1] H19/IGF2 ICR region chr11:2,000-2,008 kb: imputed CTCF {impT['CTCF'][b0+2000:b0+2009].astype(int).tolist()}; raw {rawT['CTCF'][b0+2000:b0+2009].astype(int).tolist()}", flush=True)

# 2 reproduce AUPRC
ref = fr["ref"]
y = ref[0][uni]
s = impT["CTCF"][uni]
print(f"[2] imputed CTCF vs CTCF 2-of-3 bins: AUPRC {average_precision_score(y, s):.4f} AUROC {roc_auc_score(y, s):.4f} prevalence {y.mean():.4f}", flush=True)
sr = rawT["CTCF"][uni]
print(f"[2] raw CTCF: AUPRC {average_precision_score(y, sr):.4f} AUROC {roc_auc_score(y, sr):.4f}", flush=True)

# 3 profile
mask6 = S6.bulk_spec.values > 1
prof = []
for t in TFS:
    ix = np.flatnonzero((S6.ref.values == t) & mask6)
    ix = ix[np.argsort(-S6.signal.values[ix], kind="stable")][:5000]
    b = np.array([off[c][0] for c in S6.chrom.values[ix]]) + S6.pos.values[ix] // 1000
    k = tf79.index(t)
    for d in range(-10, 11):
        j = b + d
        j = j[(j >= 0) & (j < N)]
        oth_i = (frac_sum[j] - impT[t][j] / ncell[k]) / (len(tf79) - 1)
        oth_r = (raw_cpm_sum[j] - rawT[t][j] / tot[k] * 1e6) / (len(tf79) - 1)
        prof.append(dict(ref=t, offset_kb=d, imp_frac_own=(impT[t][j] / ncell[k]).mean(), imp_frac_others=oth_i.mean(),
                         raw_cpm_own=(rawT[t][j] / tot[k] * 1e6).mean(), raw_cpm_others=oth_r.mean()))
pf = pd.DataFrame(prof)
pf.to_csv(OC / "check3_profiles.tsv", sep="\t", index=False, float_format="%.6g")
summ = []
for t, g in pf.groupby("ref"):
    g = g.set_index("offset_kb")
    fl = g.loc[[-10, -9, -8, 8, 9, 10]]
    summ.append(dict(ref=t, peak_bin_imp_own=int(g.imp_frac_own.idxmax()), peak_bin_raw_own=int(g.raw_cpm_own.idxmax()),
                     imp_own_0=g.loc[0, "imp_frac_own"], imp_others_0=g.loc[0, "imp_frac_others"],
                     imp_ratio_0=g.loc[0, "imp_frac_own"] / g.loc[0, "imp_frac_others"],
                     imp_peak_over_flank_own=g.loc[0, "imp_frac_own"] / fl.imp_frac_own.mean(),
                     imp_peak_over_flank_others=g.loc[0, "imp_frac_others"] / fl.imp_frac_others.mean(),
                     raw_ratio_0=g.loc[0, "raw_cpm_own"] / g.loc[0, "raw_cpm_others"],
                     raw_peak_over_flank_own=g.loc[0, "raw_cpm_own"] / fl.raw_cpm_own.mean(),
                     raw_peak_over_flank_others=g.loc[0, "raw_cpm_others"] / fl.raw_cpm_others.mean()))
sd = pd.DataFrame(summ).set_index("ref").loc[TFS]
sd.to_csv(OC / "check3_summary.tsv", sep="\t", float_format="%.4g")
pd.set_option("display.width", 260)
print("[3]\n", sd.round(4).to_string(), flush=True)

# 5, 6 features
Bk6 = S6[[f"bulk_{t}" for t in TFS]].values
sb = np.array([off[c][0] for c in S6.chrom.values]) + S6.pos.values // 1000
inmask = atac[sb]
print("[6] masked/absent bins: stored as 0, not NaN; fraction of summit bins in ATAC mask per reference (preferred top 5000 used in R-B):", flush=True)


def groups(mask):
    gr = []
    for t in TFS:
        ix = np.flatnonzero((S6.ref.values == t) & mask)
        ix = ix[np.argsort(-S6.signal.values[ix], kind="stable")][:5000]
        print("   ", t, f"n {len(ix)}, in mask {inmask[ix].mean():.3f}", flush=True)
        gr += [g for g in np.array_split(ix, G) if len(g)]
    return gr


def perm(B):
    k = len(B)
    off_ = ~np.eye(k, dtype=bool)
    st = lambda M: M.diagonal().mean() - M[off_].mean()
    obs = st(B)
    null = np.array([st(B[list(q)]) for q in itertools.permutations(range(k))])
    return float(obs), float(np.mean(null >= obs - 1e-12))


rows = []
EPS = 1e-4
for mname, m in (("all summits", mask6), ("summit bin inside ATAC mask only", mask6 & inmask)):
    print(mname, flush=True)
    gr = groups(m)
    gB = np.array([Bk6[g].sum(0) for g in gr])
    bulk = np.log2(gB / gB.sum(0, keepdims=True) * 1e6 + 1)
    F = {}
    F["plain_imputed_log_fraction"] = np.array([[np.log2((impT[t][sb[g]] / ncell[tf79.index(t)]).mean() + EPS) for t in TFS] for g in gr])
    F["imputed_ratio_to_others_cellnorm_eps1e-4"] = np.array(
        [[np.log2(((impT[t][sb[g]] / ncell[tf79.index(t)]).mean() + EPS) /
                  (((frac_sum[sb[g]] - impT[t][sb[g]] / ncell[tf79.index(t)]) / (len(tf79) - 1)).mean() + EPS)) for t in TFS] for g in gr])
    F["plain_raw_log_cpm"] = np.array([[np.log2((rawT[t][sb[g]] / tot[tf79.index(t)] * 1e6).mean() + 1e-2) for t in TFS] for g in gr])
    for nm, X in F.items():
        B = np.corrcoef(np.column_stack([bulk, X]).T)[6:, :6]
        o, p = perm(B)
        rows.append(dict(summits=mname, feature=nm, diag_minus_off=o, perm_p=p, own_max_rows=int(sum(np.argmax(B[i]) == i for i in range(6))),
                         own_max_cols=int(sum(np.argmax(B[:, j]) == j for j in range(6))), **{f"r_own_{t}": B[i, i] for i, t in enumerate(TFS)}))
        pd.DataFrame(B, TFS, TFS).to_csv(OC / f"block_{nm}_{'mask' if 'mask' in mname else 'all'}.tsv", sep="\t", float_format="%.3f")
        print(nm, "\n", pd.DataFrame(B, TFS, TFS).round(2).to_string(), flush=True)
pd.DataFrame(rows).to_csv(OC / "check5_6_features.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(rows).round(3).to_string())
