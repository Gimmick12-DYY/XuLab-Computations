"""Shared paths and helpers for the Figure 2 QC (bulk ChIP agreement + specificity) scripts.
Panel = the 6 TFs with bulk ChIP references (TFS); our cells are HEK293T; see REF_SOURCE for each reference cell line.

Frame: 1 kb bins over chr1-22, X, Y in the order of
results_293T_analysis/ctcf_promoter_fp/counts/bin_index.tsv (same frame as raw.npz).
Universe for scoring: chr1-22 + X, minus the hg38 blacklist v2.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/work/users/d/y/dyy12/XuLab")
D = ROOT / "data"
OUT = ROOT / "results_293T_analysis/fig2_qc"
INP = OUT / "inputs"
COUNTS = ROOT / "results_293T_analysis/ctcf_promoter_fp/counts"
FRAG_DIR = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment")  # read-only
BIN = 1000

# NFYA dropped 2026-11-06 (user): its only reference is K562, not HEK293T.
TFS = ["CTCF", "MAZ", "ZBTB7A", "ZIC2", "ZNF282", "ZNF777"]
NT = len(TFS)
REF = {
    "CTCF": D / "CTCF_majority2of3.bed",
    "MAZ": D / "MAZ_HEK293_peaks.bed",
    "NFYA": D / "NFYA_K562_peaks.bed",
    "ZBTB7A": D / "ZBTB7A_HEK293_peaks.bed",
    "ZIC2": D / "ZIC2_HEK293_peaks.bed",
    "ZNF282": D / "ZNF282_HEK293_peaks.bed",
    "ZNF777": D / "ZNF777_HEK293_peaks.bed",
}
REF_SOURCE = {
    "CTCF": "majority 2/3 of ENCODE ENCFF314ZAL (HEK293 IDR) + GSE103651 HEK293T ChIP-nexus + 293T-con1 pval bigwig",
    "MAZ": "ENCODE ENCFF529XTF (HEK293)",
    "NFYA": "ENCODE ENCFF765GYT (K562)",
    "ZBTB7A": "ENCODE ENCFF114DGV (HEK293)",
    "ZIC2": "ENCODE ENCFF359GIY (HEK293)",
    "ZNF282": "GEO GSM2466515 HA-tag ChIP-exo (HEK293T), MACS2 re-call hg38",
    "ZNF777": "ENCODE ENCFF735CAY (HEK293)",
}
BAM = {
    "CTCF": D / "ENCFF139DCW.bam", "MAZ": D / "MAZ_HEK293.bam", "NFYA": D / "NFYA_K562.bam",
    "ZBTB7A": D / "ZBTB7A_HEK293.bam", "ZIC2": D / "ZIC2_HEK293.bam",
    "ZNF282": D / "ZNF282_HEK293.bam", "ZNF777": D / "ZNF777_HEK293.bam",
}
ATAC_UNION = D / "HEK293T_ATAC_union3.bed.gz"
ATAC_BW = ROOT / ("atac_omniatac/caper_work/atac/70bfc7e8-3760-496f-bd8e-5a4bc9e02b51/"
                  "call-macs2_signal_track_pooled/execution/rep.pooled.fc.signal.bigwig")
BLACKLIST = ROOT / "downstream/cache/hg38-blacklist.v2.bed.gz"
FASTA = ROOT / "downstream/cache/hg38.fa"
GTF = D / "gencode_v46/gencode.v46.basic.annotation.gtf.gz"
MEME_DB = ROOT / "downstream/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme"


def bin_index():
    bi = pd.read_csv(COUNTS / "bin_index.tsv", sep="\t")
    off = {r.chrom: (int(r.offset), int(r.nbins)) for r in bi.itertuples()}
    return bi, off, int(bi.nbins.sum())


def read_bed(path, ncol=3):
    op = __import__("gzip").open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            rows.append(p[:max(ncol, 3)])
    df = pd.DataFrame(rows)
    df[1] = df[1].astype(np.int64)
    df[2] = df[2].astype(np.int64)
    return df


def interval_mask(df, off, n):
    """Boolean over frame bins: bin overlaps any interval (cols 0,1,2)."""
    m = np.zeros(n, dtype=bool)
    for c, s, e in zip(df[0].values, df[1].values, df[2].values):
        if c not in off:
            continue
        o, nb = off[c]
        a, b = max(int(s) // BIN, 0), min((int(e) - 1) // BIN, nb - 1)
        if b >= a:
            m[o + a:o + b + 1] = True
    return m


def load_inputs():
    z = np.load(INP / "frame.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_raw7():
    return np.load(INP / "raw_panel.npy")            # NT x N int32, TFS order


def load_imp7():
    return np.load(INP / "imp_panel.npy")            # NT x N float32 (cells positive per bin)


def auc_pair(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


PERMS = np.array(list(itertools.permutations(range(NT))))   # exact null: all NT! row-label shuffles


def diag_stats(M, higher_better=True):
    """M: rows = sc tracks (TFS), cols = references (TFS).
    Returns diag-minus-offdiag, own-best counts, exact permutation p over all NT! row-label shuffles."""
    M = np.asarray(M, float)
    k = M.shape[0]
    off = ~np.eye(k, dtype=bool)

    def stat(A):
        return np.nanmean(np.diag(A)) - np.nanmean(A[off])
    obs = stat(M)
    null = np.array([stat(M[p]) for p in PERMS])
    sgn = 1 if higher_better else -1
    p = float(np.mean(sgn * null >= sgn * obs - 1e-12))
    own_col = int(sum((np.nanargmax(M[:, j]) if higher_better else np.nanargmin(M[:, j])) == j for j in range(k)))
    own_row = int(sum((np.nanargmax(M[i]) if higher_better else np.nanargmin(M[i])) == i for i in range(k)))
    z = float((obs - null.mean()) / null.std()) if null.std() > 0 else np.nan
    return {"diag_minus_offdiag": float(obs), "own_best_col": own_col, "own_best_row": own_row,
            "perm_p": p, "perm_z": z, "n_perm": len(PERMS)}


def write_mat(M, path, rows=TFS, cols=TFS, rlab="sc_track", clab="reference"):
    df = pd.DataFrame(np.asarray(M), index=pd.Index(rows, name=f"{rlab}\\{clab}"), columns=cols)
    df.to_csv(path, sep="\t", float_format="%.5g")
    return df


def append_summary(rows, part=None):
    """Write this script's diagonal-statistic rows to OUT/summary_parts/<part>.tsv
    (part defaults to the calling script's name); 99_summary.py merges the parts."""
    import inspect, os
    part = part or os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
    d = OUT / "summary_parts"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{part}.tsv", sep="\t", index=False, float_format="%.5g")


def score_matrix(S, pos, keep, neg=None):
    """AUROC/AUPRC tracks x refs. S: tracks x N scores; pos: refs x N bool; keep: N bool scoring universe.
    neg: optional refs x N bool negatives (default: all keep bins not positive for that reference)."""
    R = np.full((S.shape[0], pos.shape[0]), np.nan)
    P = R.copy()
    for j in range(pos.shape[0]):
        if neg is None:
            sel = keep
            y = pos[j][sel]
        else:
            sel = keep & (pos[j] | neg[j])
            y = pos[j][sel]
        if y.sum() == 0 or y.all():
            continue
        for i in range(S.shape[0]):
            R[i, j], P[i, j] = auc_pair(y, S[i][sel])
    return R, P


DM_DEPTH = 420339          # ZNF777 raw reads, shallowest of the panel
DM_SEEDS = (11, 22, 33)


def thin(counts, target, seed):
    """Binomial thinning of per-bin read counts to ~target total reads (read-level downsampling)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(counts)
    p = min(1.0, target / c.sum())
    return rng.binomial(c.astype(np.int64), p).astype(np.float32)


def raw7_dm(seed, raw7=None):
    """Panel TFs (TFS order) thinned to DM_DEPTH reads."""
    raw7 = load_raw7() if raw7 is None else raw7
    return np.vstack([thin(raw7[k], DM_DEPTH, seed * 100 + k) for k in range(NT)])


def raw_arms(raw7=None):
    """Yields (arm, list of 7xN score arrays): 'raw' (full depth, one array) and 'raw_dm' (3 seeds)."""
    raw7 = load_raw7() if raw7 is None else raw7
    yield "raw", [raw7.astype(np.float32)]
    yield "raw_dm", [raw7_dm(s, raw7) for s in DM_SEEDS]
