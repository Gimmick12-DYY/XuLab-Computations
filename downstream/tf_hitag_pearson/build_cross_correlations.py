#!/usr/bin/env python3
"""Two cross-TF Pearson matrices on the shared 10 kb bins.

  raw_vs_imputed:          rows = HiTAG aggregate, cols = Imputed aggregate
  imputed_vs_chromnitron:  rows = Imputed aggregate, cols = Chromnitron (SA / pred_2)

Same TF order on both axes, so the diagonal is the matched-TF correlation.
Chromnitron is included only for TFs that have the SA bigWig.
"""
from __future__ import annotations

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
OUT = ROOT / "downstream" / "tf_hitag_pearson" / "cross"
COL = OUT / "columns"
BINS = ROOT / "downstream" / "ctcf_hitag_pearson" / "bins_10kb.bed"
TAG = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig")
SA = Path("/users/x/u/xujie/HEK293T_SA")
SMOOTH_BINS = 5
N_WORKERS = 4


def discover_tfs() -> list[str]:
    return sorted(p.parent.parent.name for p in (ROOT / "unified" / "work").glob("*/impute/matrix_csr.npz"))


def load_bins(path: Path):
    chroms, starts, ends = [], [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            chroms.append(p[0])
            starts.append(int(p[1]))
            ends.append(int(p[2]))
    return np.array(chroms), np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def chrom_groups(chroms: np.ndarray):
    groups = []
    start = 0
    for i in range(1, len(chroms)):
        if chroms[i] != chroms[start]:
            groups.append((str(chroms[start]), start, i))
            start = i
    groups.append((str(chroms[start]), start, len(chroms)))
    return groups


def score_bw(path: str, chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray, groups) -> np.ndarray:
    bw = pyBigWig.open(path)
    if bw is None:
        raise FileNotFoundError(path)
    out = np.full(chroms.size, np.nan, dtype=np.float32)
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
            if n == 1:
                v = bw.stats(chrom, int(starts[i]), int(ends[i]), type="mean")
                if v and v[0] is not None:
                    out[i] = float(v[0])
            else:
                vals = bw.stats(chrom, int(starts[i]), int(ends[j - 1]), nBins=n, type="mean")
                if vals:
                    for k, val in enumerate(vals):
                        if val is not None:
                            out[i + k] = float(val)
            i = j
    bw.close()
    return out


def parse_region(tok: str):
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def load_regions(tf: str):
    path = ROOT / "unified" / "work" / tf / "impute" / "regions.tsv"
    chroms, starts, ends = [], [], []
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            c, s, e = parse_region(ln)
            chroms.append(c)
            starts.append(s)
            ends.append(e)
    return chroms, np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def smooth_1kb(chroms, starts, scores, w: int) -> np.ndarray:
    out = np.zeros_like(scores)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    k = np.ones(w, dtype=float) / w
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[np.argsort(starts[idx])]
        st = starts[idx]
        lo, hi = int(st[0]), int(st[-1])
        nslot = (hi - lo) // 1000 + 1
        grid = np.zeros(nslot, dtype=float)
        grid[(st - lo) // 1000] = scores[idx]
        sm = np.convolve(grid, k, mode="same")
        out[idx] = sm[(st - lo) // 1000]
    return out


def reduce_on_10kb(chroms, starts, ends, scores, q_chrom, q_start, q_end) -> np.ndarray:
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    grids = {}
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        st = starts[idx]
        lo = int(st.min())
        hi = int(ends[idx].max())
        nslot = max(1, (hi - lo) // 1000)
        grid = np.zeros(nslot, dtype=float)
        rel = np.clip((st - lo) // 1000, 0, nslot - 1)
        grid[rel] = scores[idx]
        grids[c] = (lo, grid)
    out = np.full(len(q_chrom), np.nan, dtype=np.float32)
    for i, c in enumerate(q_chrom):
        rec = grids.get(str(c))
        if rec is None:
            continue
        lo, grid = rec
        a = max(0, (int(q_start[i]) - lo) // 1000)
        b = max(a + 1, (int(q_end[i]) - lo + 999) // 1000)
        b = min(len(grid), b)
        if a >= len(grid):
            continue
        out[i] = float(grid[a:b].mean())
    return out


_REGIONS = None
_BINS = None
_GROUPS = None


def _init_worker(regions, bins, groups):
    global _REGIONS, _BINS, _GROUPS
    _REGIONS = regions
    _BINS = bins
    _GROUPS = groups


def _one(task: tuple[str, str]) -> str:
    kind, tf = task
    dest = COL / kind / f"{tf.upper()}.npy"
    if dest.is_file() and dest.stat().st_size > 0:
        return f"skip {kind} {tf}"
    chroms, starts, ends = _BINS
    if kind == "raw":
        path = TAG / "mtx2bw" / f"{tf.upper()}_HiTAG_unweighted_smooth.bw"
        if not path.is_file():
            return f"MISSING raw {tf}"
        vec = score_bw(str(path), chroms, starts, ends, _GROUPS)
    elif kind == "chromnitron":
        path = SA / tf.upper() / "processed" / "data.bigwig"
        if not path.is_file():
            return f"MISSING chromnitron {tf}"
        vec = score_bw(str(path), chroms, starts, ends, _GROUPS)
    elif kind == "imputed":
        r_chroms, r_starts, r_ends = _REGIONS
        imp = ROOT / "unified" / "work" / tf / "impute" / "matrix_csr.npz"
        scores = np.asarray(sparse.load_npz(imp).sum(axis=1)).ravel().astype(float)
        if scores.size != len(r_chroms):
            return f"MISMATCH imputed {tf} {scores.size} vs {len(r_chroms)} regions"
        smooth = smooth_1kb(r_chroms, r_starts, scores, SMOOTH_BINS)
        vec = np.log1p(np.clip(reduce_on_10kb(r_chroms, r_starts, r_ends, smooth, chroms, starts, ends), 0, None))
    else:
        return f"bad kind {kind}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, vec.astype(np.float32))
    finite = int(np.isfinite(vec).sum())
    return f"ok {kind} {tf.upper()} finite={finite:,} max={np.nanmax(vec):.4g}"


def cross_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A, B are bins x tracks. Pairwise-complete Pearson."""
    p, q = A.shape[1], B.shape[1]
    R = np.empty((p, q), dtype=np.float64)
    finite_b = [np.isfinite(B[:, j]) for j in range(q)]
    for i in range(p):
        a = A[:, i]
        ma = np.isfinite(a)
        for j in range(q):
            m = ma & finite_b[j]
            n = int(m.sum())
            if n < 1000:
                R[i, j] = np.nan
                continue
            aa = a[m]
            bb = B[m, j]
            aa = aa - aa.mean()
            bb = bb - bb.mean()
            denom = np.sqrt((aa * aa).sum() * (bb * bb).sum())
            R[i, j] = float((aa * bb).sum() / denom) if denom > 0 else np.nan
    return R


def write_tsv(path: Path, row_labels: list[str], col_labels: list[str], R: np.ndarray) -> None:
    with path.open("w") as fh:
        fh.write("track\t" + "\t".join(col_labels) + "\n")
        for i, lab in enumerate(row_labels):
            fh.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(col_labels))) + "\n")
    print(f"wrote {path}", flush=True)


def plot_cross(path: Path, row_labels: list[str], col_labels: list[str], R: np.ndarray, title: str, cbar: str) -> None:
    n, m = R.shape
    cell = 0.22
    fig_w = max(8.0, m * cell + 2.4)
    fig_h = max(8.0, n * cell + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    finite = R[np.isfinite(R)]
    vmin = float(np.floor(np.percentile(finite, 5) * 20) / 20)
    vmax = float(np.ceil(np.percentile(finite, 95) * 20) / 20)
    if vmax <= vmin:
        vmax = vmin + 0.05
    im = ax.imshow(R, cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="upper", aspect="equal", interpolation="nearest")
    fs = 6 if max(n, m) > 40 else 8
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(row_labels, fontsize=fs)
    ax.tick_params(length=0)
    ax.set_title(title)
    ax.set_xlabel(cbar.split(" vs ")[-1] if " vs " in cbar else "")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"Pearson r\n({vmin:.2f}–{vmax:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def load_matrix(kind: str, tfs: list[str]) -> tuple[list[str], np.ndarray]:
    keep, cols = [], []
    for tf in tfs:
        path = COL / kind / f"{tf.upper()}.npy"
        if not path.is_file():
            print(f"  missing column {kind} {tf}", flush=True)
            continue
        keep.append(tf.upper())
        cols.append(np.load(path))
    if not cols:
        raise SystemExit(f"no columns for {kind}")
    return keep, np.column_stack(cols)


def summarize(name: str, labels: list[str], R: np.ndarray) -> None:
    off = R[~np.eye(R.shape[0], dtype=bool)]
    diag = np.diag(R)
    print(
        f"[{name}] n={len(labels)} diag_median={np.nanmedian(diag):+.4f} "
        f"diag_min={np.nanmin(diag):+.4f} off_median={np.nanmedian(off):+.4f} "
        f"off_min={np.nanmin(off):+.4f} off_max={np.nanmax(off):+.4f}",
        flush=True,
    )
    order = np.argsort(diag)
    print(f"  lowest diagonal: " + ", ".join(f"{labels[i]}={diag[i]:.3f}" for i in order[:5]), flush=True)
    print(f"  highest diagonal: " + ", ".join(f"{labels[i]}={diag[i]:.3f}" for i in order[-5:]), flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tfs = discover_tfs()
    print(f"tfs {len(tfs)}", flush=True)
    chroms, starts, ends = load_bins(BINS)
    groups = chrom_groups(chroms)
    print("loading shared 1 kb regions from ctcf", flush=True)
    regions = load_regions("ctcf")
    # Confirm another TF uses the same region count.
    n_adnp = sum(1 for ln in open(ROOT / "unified/work/adnp/impute/regions.tsv") if ln.strip())
    if n_adnp != len(regions[0]):
        raise SystemExit(f"region count differs: ctcf {len(regions[0])} adnp {n_adnp}")

    tasks = [("raw", tf) for tf in tfs] + [("imputed", tf) for tf in tfs]
    tasks += [("chromnitron", tf) for tf in tfs if (SA / tf.upper() / "processed" / "data.bigwig").is_file()]
    print(f"tasks {len(tasks)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(regions, (chroms, starts, ends), groups),
    ) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for fut in as_completed(futs):
            print(fut.result(), flush=True)

    raw_labs, raw = load_matrix("raw", tfs)
    imp_labs, imp = load_matrix("imputed", tfs)
    if raw_labs != imp_labs:
        raise SystemExit(f"raw/imputed TF sets differ: {len(raw_labs)} vs {len(imp_labs)}")
    print("correlating raw vs imputed", flush=True)
    R_ri = cross_pearson(raw, imp)
    write_tsv(OUT / "raw_vs_imputed_pearson.tsv", raw_labs, imp_labs, R_ri)
    plot_cross(
        OUT / "raw_vs_imputed_pearson.png",
        raw_labs, imp_labs, R_ri,
        "HiTAG aggregate (rows) vs Imputed aggregate (cols)\nPearson r, 10 kb bins",
        "HiTAG aggregate vs Imputed aggregate",
    )
    summarize("raw_vs_imputed", raw_labs, R_ri)

    chrom_tfs = [tf for tf in tfs if (COL / "chromnitron" / f"{tf.upper()}.npy").is_file()]
    imp_idx = {lab: i for i, lab in enumerate(imp_labs)}
    chrom_labs, chrom = load_matrix("chromnitron", chrom_tfs)
    imp_sub = imp[:, [imp_idx[lab] for lab in chrom_labs]]
    print("correlating imputed vs chromnitron", flush=True)
    R_ic = cross_pearson(imp_sub, chrom)
    write_tsv(OUT / "imputed_vs_chromnitron_pearson.tsv", chrom_labs, chrom_labs, R_ic)
    plot_cross(
        OUT / "imputed_vs_chromnitron_pearson.png",
        chrom_labs, chrom_labs, R_ic,
        "Imputed aggregate (rows) vs Chromnitron (cols)\nPearson r, 10 kb bins",
        "Imputed aggregate vs Chromnitron",
    )
    summarize("imputed_vs_chromnitron", chrom_labs, R_ic)
    print(f"[done] {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
