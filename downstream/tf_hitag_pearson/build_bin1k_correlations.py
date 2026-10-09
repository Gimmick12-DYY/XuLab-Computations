#!/usr/bin/env python3
"""Pearson correlations on the native 1 kb unified bins.

Per TF, tracks in this order, omitting any that do not exist:
  Raw        HiTAG aggregate (unweighted smooth bigWig)
  Imputed    log1p of the 5-bin smoothed cell-sum of the union-masked impute matrix
  Pred       Chromnitron SA bigWig (former pred_2)
  Bulk       bulk ChIP CPM bigWig

Also writes two cross-TF matrices (same TF order on both axes):
  raw_vs_imputed          all TFs
  imputed_vs_pred         only TFs that have a Chromnitron bigWig
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
from scipy import sparse

ROOT = Path("/work/users/d/y/dyy12/XuLab")
OUT = ROOT / "downstream" / "tf_hitag_pearson" / "bin1k"
COL = OUT / "columns"
TAG = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig")
SA = Path("/users/x/u/xujie/HEK293T_SA")
BULK_DIR = ROOT / "downstream" / "tracks" / "bulk_chipseq"
REGIONS = ROOT / "unified" / "work" / "ctcf" / "impute" / "regions.tsv"
SMOOTH = 5
N_WORKERS = int(os.environ.get("N_WORKERS", "8"))


def discover_tfs() -> list[str]:
    return sorted(p.parent.parent.name for p in (ROOT / "unified" / "work").glob("*/impute/matrix_csr.npz"))


def load_regions(path: Path):
    chroms, starts, ends = [], [], []
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            chrom, se = ln.split(":", 1)
            s, e = se.split("-", 1)
            chroms.append(chrom)
            starts.append(int(s))
            ends.append(int(e))
    return np.array(chroms), np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def chrom_groups(chroms: np.ndarray):
    groups = []
    a = 0
    for i in range(1, len(chroms)):
        if chroms[i] != chroms[a]:
            groups.append((str(chroms[a]), a, i))
            a = i
    groups.append((str(chroms[a]), a, len(chroms)))
    return groups


def score_bw(path: str, starts: np.ndarray, ends: np.ndarray, groups) -> np.ndarray:
    bw = pyBigWig.open(path)
    if bw is None:
        raise FileNotFoundError(path)
    out = np.full(starts.size, np.nan, dtype=np.float32)
    have = set(bw.chroms())
    for chrom, a, b in groups:
        if chrom not in have:
            continue
        i = a
        while i < b:
            j = i + 1
            width = int(ends[i] - starts[i])
            while j < b and int(starts[j]) == int(ends[j - 1]) and int(ends[j] - starts[j]) == width:
                j += 1
            n = j - i
            vals = bw.stats(chrom, int(starts[i]), int(ends[j - 1]), nBins=n, type="mean")
            if vals:
                for k, val in enumerate(vals):
                    if val is not None:
                        out[i + k] = np.float32(val)
            i = j
    bw.close()
    return out


def smooth_log1p(chroms, starts, scores: np.ndarray) -> np.ndarray:
    out = np.zeros(scores.size, dtype=np.float64)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    k = np.ones(SMOOTH, dtype=np.float64) / SMOOTH
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[np.argsort(starts[idx])]
        st = starts[idx]
        lo, hi = int(st[0]), int(st[-1])
        nslot = (hi - lo) // 1000 + 1
        grid = np.zeros(nslot, dtype=np.float64)
        grid[(st - lo) // 1000] = scores[idx]
        sm = np.convolve(grid, k, mode="same")
        out[idx] = sm[(st - lo) // 1000]
    return np.log1p(np.clip(out, 0, None)).astype(np.float32)


_STARTS = None
_ENDS = None
_GROUPS = None
_CHROMS = None


def _init(chroms, starts, ends, groups):
    global _CHROMS, _STARTS, _ENDS, _GROUPS
    _CHROMS, _STARTS, _ENDS, _GROUPS = chroms, starts, ends, groups


def _one(task: tuple[str, str]) -> str:
    kind, tf = task
    dest = COL / kind / f"{tf.upper()}.npy"
    if dest.is_file() and dest.stat().st_size > 0:
        return f"skip {kind} {tf.upper()}"
    up = tf.upper()
    if kind == "raw":
        path = TAG / "mtx2bw" / f"{up}_HiTAG_unweighted_smooth.bw"
        if not path.is_file():
            return f"MISSING raw {up}"
        vec = score_bw(str(path), _STARTS, _ENDS, _GROUPS)
    elif kind == "pred":
        path = SA / up / "processed" / "data.bigwig"
        if not path.is_file():
            return f"MISSING pred {up}"
        vec = score_bw(str(path), _STARTS, _ENDS, _GROUPS)
    elif kind == "bulk":
        path = BULK_DIR / f"{up}_bulk_CPM.bw"
        if not path.is_file():
            return f"MISSING bulk {up}"
        vec = score_bw(str(path), _STARTS, _ENDS, _GROUPS)
    elif kind == "imputed":
        imp = ROOT / "unified" / "work" / tf / "impute" / "matrix_csr.npz"
        scores = np.asarray(sparse.load_npz(imp).sum(axis=1)).ravel().astype(np.float64)
        if scores.size != _STARTS.size:
            return f"MISMATCH imputed {up} {scores.size} vs {_STARTS.size}"
        vec = smooth_log1p(_CHROMS, _STARTS, scores)
    else:
        return f"bad kind {kind}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, vec)
    return f"ok {kind} {up} finite={int(np.isfinite(vec).sum()):,} max={np.nanmax(vec):.4g}"


def load_matrix(kind: str, tfs: list[str]) -> tuple[list[str], np.ndarray]:
    keep, cols = [], []
    for tf in tfs:
        path = COL / kind / f"{tf.upper()}.npy"
        if not path.is_file():
            continue
        keep.append(tf.upper())
        cols.append(np.load(path))
    if not cols:
        raise SystemExit(f"no columns for {kind}")
    return keep, np.column_stack(cols)


def pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Pearson. Rows that are non-finite in either column are dropped."""
    p, q = A.shape[1], B.shape[1]
    R = np.empty((p, q), dtype=np.float64)
    bmask = [np.isfinite(B[:, j]) for j in range(q)]
    for i in range(p):
        a = A[:, i]
        ma = np.isfinite(a)
        for j in range(q):
            m = ma & bmask[j]
            n = int(m.sum())
            if n < 1000:
                R[i, j] = np.nan
                continue
            aa = a[m].astype(np.float64)
            bb = B[m, j].astype(np.float64)
            aa -= aa.mean()
            bb -= bb.mean()
            denom = np.sqrt((aa * aa).sum() * (bb * bb).sum())
            R[i, j] = (aa * bb).sum() / denom if denom > 0 else np.nan
    return R


def write_tsv(path: Path, rows: list[str], cols: list[str], R: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write("track\t" + "\t".join(cols) + "\n")
        for i, lab in enumerate(rows):
            fh.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(cols))) + "\n")
    print(f"wrote {path}", flush=True)


def plot_matrix(path: Path, rows: list[str], cols: list[str], R: np.ndarray, title: str, numbers: bool) -> None:
    n, m = R.shape
    # Large cells so names and values stay readable when the PNG is opened.
    cell = 1.15 if n <= 6 else 0.55
    fig_w = m * cell + 3.2
    fig_h = n * cell + 2.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="upper", aspect="equal", interpolation="nearest")
    lab_fs = 22 if n <= 6 else 16
    num_fs = 20 if n <= 6 else 11
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=lab_fs)
    ax.set_yticklabels(rows, fontsize=lab_fs)
    ax.tick_params(length=0, pad=6)
    ax.set_title(title, fontsize=lab_fs + 2, pad=12)
    if numbers:
        for i in range(n):
            for j in range(m):
                v = R[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=num_fs,
                        color=("white" if v >= 0.65 else "black"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", fontsize=lab_fs)
    cb.ax.tick_params(labelsize=lab_fs - 2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def summarize(name: str, labels: list[str], R: np.ndarray) -> None:
    diag = np.diag(R)
    off = R[~np.eye(len(labels), dtype=bool)]
    print(
        f"[{name}] n={len(labels)} diag_median={np.nanmedian(diag):+.3f} "
        f"diag_min={np.nanmin(diag):+.3f} off_median={np.nanmedian(off):+.3f}",
        flush=True,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tfs = discover_tfs()
    print(f"tfs {len(tfs)}", flush=True)
    print("loading 1 kb regions", flush=True)
    chroms, starts, ends = load_regions(REGIONS)
    print(f"bins {starts.size:,} width_mode={(ends-starts)[:1000].min()}-{(ends-starts)[:1000].max()}", flush=True)
    groups = chrom_groups(chroms)
    pred_tfs = [tf for tf in tfs if (SA / tf.upper() / "processed" / "data.bigwig").is_file()]
    bulk_tfs = [tf for tf in tfs if (BULK_DIR / f"{tf.upper()}_bulk_CPM.bw").is_file()]
    no_pred = [tf.upper() for tf in tfs if tf not in pred_tfs]
    (OUT / "tfs_without_pred.txt").write_text("\n".join(no_pred) + "\n")
    print(f"no Chromnitron ({len(no_pred)}): {', '.join(no_pred)}", flush=True)
    print(f"with bulk ({len(bulk_tfs)}): {', '.join(t.upper() for t in bulk_tfs)}", flush=True)

    tasks = [("raw", tf) for tf in tfs] + [("imputed", tf) for tf in tfs]
    tasks += [("pred", tf) for tf in pred_tfs] + [("bulk", tf) for tf in bulk_tfs]
    print(f"tasks {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init, initargs=(chroms, starts, ends, groups)) as pool:
        for fut in as_completed([pool.submit(_one, t) for t in tasks]):
            print(fut.result(), flush=True)

    raw_labs, raw = load_matrix("raw", tfs)
    imp_labs, imp = load_matrix("imputed", tfs)
    if raw_labs != imp_labs:
        raise SystemExit(f"raw/imputed labels differ {raw_labs} vs {imp_labs}")
    print("raw vs imputed", flush=True)
    R_ri = pairwise(raw, imp)
    write_tsv(OUT / "raw_vs_imputed_pearson.tsv", raw_labs, imp_labs, R_ri)
    plot_matrix(
        OUT / "raw_vs_imputed_pearson.png", raw_labs, imp_labs, R_ri,
        "Raw (HiTAG aggregate) vs Imputed aggregate\nPearson r, 1 kb bins",
        numbers=True,
    )
    summarize("raw_vs_imputed", raw_labs, R_ri)

    chrom_labs, chrom = load_matrix("pred", pred_tfs)
    imp_idx = {lab: i for i, lab in enumerate(imp_labs)}
    imp_sub = imp[:, [imp_idx[lab] for lab in chrom_labs]]
    print("imputed vs pred", flush=True)
    R_ip = pairwise(imp_sub, chrom)
    write_tsv(OUT / "imputed_vs_pred_pearson.tsv", chrom_labs, chrom_labs, R_ip)
    plot_matrix(
        OUT / "imputed_vs_pred_pearson.png", chrom_labs, chrom_labs, R_ip,
        "Imputed aggregate vs Pred (Chromnitron)\nPearson r, 1 kb bins",
        numbers=True,
    )
    summarize("imputed_vs_pred", chrom_labs, R_ip)

    pred_set = set(chrom_labs)
    bulk_set = {tf.upper() for tf in bulk_tfs if (COL / "bulk" / f"{tf.upper()}.npy").is_file()}
    col = {lab: i for i, lab in enumerate(raw_labs)}
    for tf in raw_labs:
        labels = ["Raw", "Imputed"]
        cols = [raw[:, col[tf]], imp[:, col[tf]]]
        if tf in pred_set:
            labels.append("Pred")
            cols.append(chrom[:, chrom_labs.index(tf)])
        if tf in bulk_set:
            labels.append("Bulk")
            cols.append(np.load(COL / "bulk" / f"{tf}.npy"))
        M = np.column_stack(cols)
        R = pairwise(M, M)
        write_tsv(OUT / f"{tf}_pearson.tsv", labels, labels, R)
        plot_matrix(OUT / f"{tf}_pearson.png", labels, labels, R, f"{tf}  Pearson r, 1 kb bins", numbers=True)
    print(f"[done] {OUT}", flush=True)
    print(f"TFs without Pred ({len(no_pred)}), left out of imputed-vs-pred: {', '.join(no_pred)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
