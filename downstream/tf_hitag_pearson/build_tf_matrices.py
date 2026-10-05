#!/usr/bin/env python3
"""10 kb Pearson/Spearman matrices for the 6 non-CTCF TFs that have bulk ChIP."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
from scipy import sparse
from scipy.stats import rankdata

ROOT = Path("/work/users/d/y/dyy12/XuLab")
OUT = ROOT / "downstream" / "tf_hitag_pearson"
BINS = ROOT / "downstream" / "ctcf_hitag_pearson" / "bins_10kb.bed"
TAG = Path("/vast/som/xujie_lab/TAGATG-293T/bigwig")
SA = Path("/users/x/u/xujie/HEK293T_SA")
SMOOTH_BINS = 5

# CTCF already done. These six have bulk ChIP (ENCODE/GEO) in the panel.
TFS = ["MAZ", "ZBTB7A", "ZIC2", "ZNF777", "ZNF282", "NFYA"]
BULK_LABEL = {
    "MAZ": "ENCODE",
    "ZBTB7A": "ENCODE",
    "ZIC2": "ENCODE",
    "ZNF777": "ENCODE",
    "ZNF282": "GEO",
    "NFYA": "ENCODE_K562",
}


def load_bins(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chroms, starts, ends = [], [], []
    with path.open() as fh:
        for line in fh:
            p = line.split()
            chroms.append(p[0])
            starts.append(int(p[1]))
            ends.append(int(p[2]))
    return np.array(chroms), np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def score_bw(path: Path, chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    bw = pyBigWig.open(str(path))
    if bw is None:
        raise FileNotFoundError(path)
    out = np.full(chroms.size, np.nan)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[str(c)].append(i)
    for c, idx in by.items():
        if c not in bw.chroms():
            continue
        for i in idx:
            v = bw.stats(c, int(starts[i]), int(ends[i]), type="mean")
            if v is not None and v[0] is not None:
                out[i] = float(v[0])
    bw.close()
    return out


def score_bam_counts(bam: Path, bed: Path, n: int) -> np.ndarray:
    """Read counts in each BED interval (Pearson is scale-invariant)."""
    out = np.full(n, np.nan)
    cmd = ["bedtools", "coverage", "-a", str(bed), "-b", str(bam), "-counts"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for i, line in enumerate(proc.stdout.splitlines()):
        p = line.split()
        out[i] = float(p[-1])
    if i + 1 != n:
        raise RuntimeError(f"bedtools coverage rows {i+1} != bins {n}")
    return out


def parse_region(tok: str) -> tuple[str, int, int]:
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def smooth_1kb(chroms: list[str], starts: np.ndarray, scores: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(scores)
    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    k = np.ones(w, dtype=float) / w
    for c, idx in by.items():
        idx = np.asarray(idx, dtype=np.int64)
        order = np.argsort(starts[idx])
        idx = idx[order]
        st = starts[idx]
        if len(st) == 0:
            continue
        lo, hi = int(st[0]), int(st[-1])
        nslot = (hi - lo) // 1000 + 1
        grid = np.zeros(nslot, dtype=float)
        grid[(st - lo) // 1000] = scores[idx]
        sm = np.convolve(grid, k, mode="same")
        out[idx] = sm[(st - lo) // 1000]
    return out


def reduce_on_10kb(chroms, starts, ends, scores, q_chrom, q_start, q_end, how: str):
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
    out = np.full(len(q_chrom), np.nan)
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
        sl = grid[a:b]
        out[i] = float(sl.sum() if how == "sum" else sl.mean())
    return out


def imputed_columns(tf: str, q_chrom, q_start, q_end):
    imp_dir = ROOT / "unified" / "work" / tf.lower() / "impute"
    sm = sparse.load_npz(imp_dir / "matrix_csr.npz")
    scores_1kb = np.asarray(sm.sum(axis=1)).ravel().astype(float)
    regs = [parse_region(ln) for ln in (imp_dir / "regions.tsv").read_text().splitlines() if ln.strip()]
    chroms = [r[0] for r in regs]
    starts = np.array([r[1] for r in regs], dtype=np.int64)
    ends = np.array([r[2] for r in regs], dtype=np.int64)
    raw10 = reduce_on_10kb(chroms, starts, ends, scores_1kb, q_chrom, q_start, q_end, "sum")
    smooth = smooth_1kb(chroms, starts, scores_1kb, SMOOTH_BINS)
    agg10 = reduce_on_10kb(chroms, starts, ends, smooth, q_chrom, q_start, q_end, "mean")
    return np.log1p(np.clip(raw10, 0, None)), np.log1p(np.clip(agg10, 0, None))


def write_tsv(path: Path, labels: list[str], R: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("track\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{R[i, j]:.4f}" for j in range(len(labels))) + "\n")
    print(f"wrote {path}")


def plot_matrix(tsv: Path, png: Path, title: str, cbar_label: str) -> None:
    lines = tsv.read_text().splitlines()
    labels = lines[0].split("\t")[1:]
    R = np.array([[float(x) for x in ln.split("\t")[1:]] for ln in lines[1:]], dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.05 * n + 2.2, 1.05 * n + 1.4))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="upper")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = R[i, j]
            ink = "white" if v >= 0.65 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=ink, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(f"wrote {png}")


def tracks_for(tf: str) -> list[tuple[str, Path | None, str]]:
    """(label, path, kind) kind in {bw, bam}."""
    bulk_bw = ROOT / "downstream" / "tracks" / "bulk_chipseq" / f"{tf}_bulk_CPM.bw"
    bulk_bam = {
        "NFYA": ROOT / "data" / "NFYA_K562.bam",
        "ZNF282": ROOT / "data" / "ZNF282_HEK293.bam",
    }.get(tf)
    rows: list[tuple[str, Path | None, str]] = [
        # Chromnitron = SA replicate (former pred_2); eLife/pred_1 dropped.
        ("Chromnitron", SA / tf / "processed" / "data.bigwig", "bw"),
        (
            BULK_LABEL[tf],
            bulk_bw
            if bulk_bw.exists()
            else (OUT / "NFYA_bulk_bedcov.tsv" if (OUT / "NFYA_bulk_bedcov.tsv").exists() else bulk_bam),
            "bw" if bulk_bw.exists() else ("counts" if (OUT / "NFYA_bulk_bedcov.tsv").exists() else "bam"),
        ),
        ("HiTAG_weight", TAG / "mtx2bw" / f"{tf}_HiTAG_weighted_smooth.bw", "bw"),
        ("HiTAG_macs2", TAG / "bw" / f"TF.{tf}" / f"TF.{tf}_treat_pileup.srt.bw", "bw"),
        ("HiTAG_aggregate", TAG / "mtx2bw" / f"{tf}_HiTAG_unweighted_smooth.bw", "bw"),
    ]
    out = []
    for lab, p, kind in rows:
        if p is None or not Path(p).exists():
            print(f"  skip {tf} {lab}: missing {p}")
            continue
        out.append((lab, Path(p), kind))
    return out


def corr(X: np.ndarray, method: str) -> np.ndarray:
    if method == "pearson":
        return np.corrcoef(X, rowvar=False)
    ranks = np.column_stack([rankdata(X[:, j], method="average") for j in range(X.shape[1])])
    return np.corrcoef(ranks, rowvar=False)


def run_tf(tf: str, chroms, starts, ends) -> None:
    print(f"\n=== {tf} ===")
    cols = []
    labels = []
    for lab, path, kind in tracks_for(tf):
        print(f"  score {lab} {path.name}")
        if kind == "bam":
            v = score_bam_counts(path, BINS, chroms.size)
        elif kind == "counts":
            v = np.loadtxt(path, dtype=float)
            if v.size != chroms.size:
                raise RuntimeError(f"{path} has {v.size} rows, expected {chroms.size}")
        else:
            v = score_bw(path, chroms, starts, ends)
        print(f"    finite={np.isfinite(v).sum():,}  max={np.nanmax(v):.4g}")
        labels.append(lab)
        cols.append(v)
    print("  imputed CSR log1p")
    imp_log, agg_log = imputed_columns(tf, chroms, starts, ends)
    labels.extend(["Imputed_log1p", "Imputed_aggregate_log1p"])
    cols.extend([imp_log, agg_log])
    X = np.column_stack(cols)
    keep = np.isfinite(X).all(axis=1)
    Xk = X[keep]
    print(f"  complete bins {int(keep.sum()):,} / {len(keep):,}")
    np.savez_compressed(OUT / f"{tf}_signals_10kb.npz", chrom=chroms, start=starts, end=ends, **dict(zip(labels, cols)))
    for method, stem in (("pearson", f"{tf}_pearson_10kb"), ("spearman", f"{tf}_spearman_10kb")):
        R = corr(Xk, method)
        tsv = OUT / f"{stem}.tsv"
        write_tsv(tsv, labels, R)
        plot_matrix(tsv, OUT / f"{stem}.png", f"{tf}  {method.capitalize()} r, 10 kb bins", f"{method.capitalize()} r")
        if "Imputed_log1p" in labels:
            ii = labels.index("Imputed_log1p")
            for lab in ("HiTAG_macs2", "HiTAG_aggregate", BULK_LABEL[tf], "Chromnitron"):
                if lab in labels:
                    print(f"  {method} Imputed_log1p vs {lab}: {R[ii, labels.index(lab)]:.3f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chroms, starts, ends = load_bins(BINS)
    print(f"bins {chroms.size:,} from {BINS}")
    tfs = sys.argv[1:] if len(sys.argv) > 1 else TFS
    for tf in tfs:
        tf = tf.upper()
        run_tf(tf, chroms, starts, ends)


if __name__ == "__main__":
    main()
