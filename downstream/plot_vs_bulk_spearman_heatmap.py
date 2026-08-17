#!/usr/bin/env python3
"""Spearman concordance vs bulk ChIP on shared 1 kb bins.

Three matrices (same panel / feature space):
  1. raw pseudobulk (rows) × bulk bedcov (cols)
  2. imputed occupancy (rows) × bulk bedcov (cols)
  3. bulk bedcov (rows) × bulk bedcov (cols)  — reference similarity

Bulk = samtools bedcov over the TF ChIP BAM on the shared regions bed.
Pseudobulks match plot_raw_vs_imputed_spearman_heatmap.py:
  raw = sum over cells; imputed = mean over cells.

Outputs under downstream/plots/tf_bulk_pearson/:
  spearman_{features}_raw_vs_bulk{.tsv,.png,_annot.png}
  spearman_{features}_imputed_vs_bulk{.tsv,.png,_annot.png}
  spearman_{features}_bulk_vs_bulk{.tsv,.png,_annot.png}

Usage:
  python3 downstream/plot_vs_bulk_spearman_heatmap.py
  python3 downstream/plot_vs_bulk_spearman_heatmap.py --features union_peaks
  python3 downstream/plot_vs_bulk_spearman_heatmap.py --features union_peaks --open-chromatin
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "unified" / "scripts"))
from _regions import open_chromatin_bin_mask  # noqa: E402

DEFAULT_OPEN_BED = REPO / "data" / "HEK293T_OmniATAC_idr_optimal.narrowPeak.gz"

PANEL = [
    ("CTCF", "ctcf"),
    ("MAZ", "maz"),
    ("ZIC2", "zic2"),
    ("ZBTB7A", "zbtb7a"),
    ("ZNF777", "znf777"),
    ("NFYA", "nfya"),
    ("ZNF282", "znf282"),
]

# Same BAM resolution as downstream SLURM wrappers.
BULK_BAM = {
    "CTCF": REPO / "data" / "ENCFF139DCW.bam",
    "MAZ": REPO / "data" / "MAZ_HEK293.bam",
    "ZIC2": REPO / "data" / "ZIC2_HEK293.bam",
    "ZBTB7A": REPO / "data" / "ZBTB7A_HEK293.bam",
    "ZNF777": REPO / "data" / "ZNF777_HEK293.bam",
    "NFYA": REPO / "data" / "NFYA_K562.bam",
    "ZNF282": REPO / "data" / "ZNF282_HEK293.bam",
}

PEAK_BED = {
    "CTCF": REPO / "data" / "CTCF_HEK293_peaks.bed",
    "MAZ": REPO / "data" / "MAZ_HEK293_peaks.bed",
    "ZIC2": REPO / "data" / "ZIC2_HEK293_peaks.bed",
    "ZBTB7A": REPO / "data" / "ZBTB7A_HEK293_peaks.bed",
    "ZNF777": REPO / "data" / "ZNF777_HEK293_peaks.bed",
    "NFYA": REPO / "data" / "NFYA_K562_peaks.bed",
    "ZNF282": REPO / "data" / "ZNF282_HEK293_peaks.bed",
}

CCRE_CACHE = REPO / "downstream" / "bins_ctcf_peak_cutoff_q05" / "cache"
CCRE_BEDS = [
    CCRE_CACHE / "SCREEN-293-cCRE-ENCFF439DDQ_ENCFF885SUR_ENCFF128UTY.bed",
    CCRE_CACHE / "SCREEN-293T-cCRE-ENCFF529BOG.bed",
]

BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"
SAMTOOLS = "samtools"
RSCRIPT = "/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/Rscript"


def regions_bed(regions_tsv_gz: Path, out_bed: Path) -> int:
    import gzip

    n = 0
    with gzip.open(regions_tsv_gz, "rt") as fin, out_bed.open("w") as fout:
        for i, line in enumerate(fin):
            chrom, rest = line.strip().split(":", 1)
            start, end = rest.split("-", 1)
            fout.write(f"{chrom}\t{start}\t{end}\t{i}\n")
            n += 1
    return n


def build_feature_bed(kind: str, out_bed: Path) -> Path:
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        raw = td_p / "raw.bed"
        with raw.open("w") as fout:
            if kind == "ccre":
                sources = CCRE_BEDS
            elif kind == "union_peaks":
                sources = list(PEAK_BED.values())
            else:
                raise SystemExit(f"unknown --features {kind}")
            for bed in sources:
                if not bed.is_file():
                    raise SystemExit(f"missing feature bed: {bed}")
                with bed.open() as fin:
                    for line in fin:
                        p = line.split("\t")
                        fout.write(f"{p[0]}\t{p[1]}\t{p[2]}\n")
        sorted_bed = td_p / "sorted.bed"
        with sorted_bed.open("w") as fout:
            subprocess.run([BEDTOOLS, "sort", "-i", str(raw)], check=True, stdout=fout)
        with out_bed.open("w") as fout:
            subprocess.run(
                [BEDTOOLS, "merge", "-i", str(sorted_bed)],
                check=True,
                stdout=fout,
            )
    return out_bed


def feature_bin_indices(
    regions_bed_path: Path, feature_bed: Path, out_idx: Path
) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w") as tmp:
        hit = Path(tmp.name)
    try:
        with hit.open("w") as fout:
            subprocess.run(
                [
                    BEDTOOLS,
                    "intersect",
                    "-u",
                    "-a",
                    str(regions_bed_path),
                    "-b",
                    str(feature_bed),
                ],
                check=True,
                stdout=fout,
            )
        idx = [int(line.rstrip().split("\t")[3]) for line in hit.open()]
        arr = np.unique(np.asarray(idx, dtype=np.int64))
        np.save(out_idx, arr)
        return arr
    finally:
        hit.unlink(missing_ok=True)


def load_pseudobulk(work: Path, kind: str) -> np.ndarray:
    if kind == "raw":
        mat = sio.mmread(str(work / "mm" / "matrix.mtx.gz")).tocsr()
        return np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    mat = sp.load_npz(work / "impute" / "matrix_csr.npz").tocsr()
    return np.asarray(mat.mean(axis=1)).ravel().astype(np.float64)


def load_bulk_bedcov(
    regions_bed_path: Path,
    bam: Path,
    cache_npy: Path,
    *,
    n_regions: int,
) -> np.ndarray:
    """Per-bin read counts from samtools bedcov; order matches regions bed (col4 = index)."""
    if cache_npy.is_file():
        arr = np.load(cache_npy)
        if arr.size == n_regions:
            return arr
        print(
            f"  stale cache {cache_npy.name} (len={arr.size} != {n_regions}); recomputing",
            flush=True,
        )

    if not bam.is_file():
        raise SystemExit(f"missing bulk BAM: {bam}")
    bai = Path(str(bam) + ".bai")
    bai2 = bam.with_suffix(".bai")
    if not bai.is_file() and not bai2.is_file():
        raise SystemExit(f"missing BAM index for {bam}")

    cov = np.zeros(n_regions, dtype=np.float64)
    print(f"  samtools bedcov → {cache_npy.name} (n={n_regions})...", flush=True)
    proc = subprocess.Popen(
        [SAMTOOLS, "bedcov", "-Q", "0", str(regions_bed_path), str(bam)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for i, line in enumerate(proc.stdout):
        parts = line.rstrip("\n").split("\t")
        # last column = coverage; 4th column (0-based idx 3) is bin index if present
        if len(parts) >= 5 and parts[3].isdigit():
            bi = int(parts[3])
        else:
            bi = i
        if bi >= n_regions:
            raise SystemExit(f"bedcov bin index {bi} >= n_regions {n_regions}")
        cov[bi] = float(parts[-1])
        if (i + 1) % 500000 == 0:
            print(f"    ... {i + 1} regions", flush=True)
    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"samtools bedcov failed ({rc}): {stderr[:500]}")
    if i + 1 != n_regions:
        print(
            f"  warning: bedcov lines={i + 1} vs n_regions={n_regions}",
            flush=True,
        )
    np.save(cache_npy, cov)
    return cov


def corr_matrix(row_pb: np.ndarray, col_pb: np.ndarray, method: str = "spearman") -> np.ndarray:
    n = row_pb.shape[1]
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if method == "spearman":
                rho, _ = spearmanr(row_pb[:, i], col_pb[:, j])
                R[i, j] = float(rho)
            else:
                x, y = row_pb[:, i], col_pb[:, j]
                if x.std() == 0 or y.std() == 0:
                    R[i, j] = float("nan")
                else:
                    R[i, j] = float(np.corrcoef(x, y)[0, 1])
    return R


def write_tsv(path: Path, M: np.ndarray, labels: list[str], row_col: str) -> None:
    with path.open("w") as f:
        f.write(f"{row_col}\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(
                lab
                + "\t"
                + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels)))
                + "\n"
            )


def plot_complexheatmap(
    tsv: Path,
    png: Path,
    title: str,
    *,
    row_title: str,
    col_title: str,
    annot: bool = False,
    legend: str = "Spearman",
) -> None:
    if annot:
        cell_fun = textwrap.dedent(
            """\
            cell_fun = function(j, i, x, y, width, height, fill) {
              grid.text(
                sprintf("%.2f", mat[i, j]),
                x, y,
                gp = gpar(fontsize = 8)
              )
            }
            """
        )
    else:
        cell_fun = "cell_fun = NULL\n"

    r_code = textwrap.dedent(
        f"""\
        suppressPackageStartupMessages({{
          library(ComplexHeatmap)
          library(circlize)
          library(grid)
        }})
        tsv <- "{tsv}"
        png_path <- "{png}"
        title <- "{title}"
        legend_name <- "{legend}"
        row_title <- "{row_title}"
        col_title <- "{col_title}"

        tab <- read.delim(tsv, check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#313695", "#FFFFBF", "#A50026"))
        {cell_fun}
        png(png_path, width = 1700, height = 1550, res = 220)
        ht <- Heatmap(
          mat,
          name = legend_name,
          col = col_fun,
          cluster_rows = FALSE,
          cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = 0.5),
          cell_fun = cell_fun,
          column_title = title,
          column_title_gp = gpar(fontsize = 11),
          row_title = row_title,
          row_title_gp = gpar(fontsize = 11),
          row_names_side = "left",
          column_names_rot = 45,
          width = ncol(mat) * unit(7.5, "mm"),
          height = nrow(mat) * unit(7.5, "mm"),
          heatmap_legend_param = list(title = legend_name, at = c(-1, 0, 1))
        )
        draw(
          ht,
          column_title = col_title,
          column_title_side = "bottom",
          column_title_gp = gpar(fontsize = 11),
          padding = unit(c(4, 4, 4, 10), "mm")
        )
        dev.off()
        cat("Wrote", png_path, "\\n")
        """
    )
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as tmp:
        tmp.write(r_code)
        rpath = tmp.name
    try:
        subprocess.run([RSCRIPT, rpath], check=True)
    finally:
        Path(rpath).unlink(missing_ok=True)


def print_matrix(name: str, R: np.ndarray, labels: list[str]) -> None:
    print(f"{name}:")
    print("       " + "".join(f"{l:>8s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(len(labels))))
    print("diag:", {labels[i]: round(float(R[i, i]), 3) for i in range(len(labels))})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-root", type=Path, default=REPO / "unified" / "work")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "plots" / "tf_bulk_pearson"
    )
    ap.add_argument(
        "--features",
        choices=["ccre", "union_peaks", "all"],
        default="all",
        help="Shared genomic feature space (default: all).",
    )
    ap.add_argument(
        "--open-chromatin",
        nargs="?",
        const=str(DEFAULT_OPEN_BED),
        default=None,
        metavar="BED",
        help=(
            "Restrict correlation bins to OmniATAC open chromatin "
            f"(default BED if flag set with no path: {DEFAULT_OPEN_BED.name}). "
            "Applies to raw, imputed, and bulk tracks; writes *_open_* outputs."
        ),
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_cache")
    cache.mkdir(parents=True, exist_ok=True)

    labels = [p[0] for p in PANEL]
    n_tf = len(PANEL)

    regions_gz = args.work_root / "ctcf" / "mm" / "regions.tsv.gz"
    if not regions_gz.is_file():
        raise SystemExit(f"missing {regions_gz}")

    reg_bed = cache / "regions_1kb.bed"
    if not reg_bed.is_file():
        print(f"[features] writing regions bed from {regions_gz}", flush=True)
        print(f"  n_regions={regions_bed(regions_gz, reg_bed)}", flush=True)

    # Genome-wide region count (bedcov always runs on full regions bed, then subset).
    with gzip.open(regions_gz, "rt") as fin:
        regions = [ln.strip() for ln in fin]
    n_reg_full = len(regions)

    if args.features == "all":
        idx = np.arange(n_reg_full, dtype=np.int64)
        print(f"[features] all bins: n={idx.size}", flush=True)
    else:
        feat_bed = cache / f"features_{args.features}.bed"
        if not feat_bed.is_file():
            print(f"[features] building {args.features} bed...", flush=True)
            build_feature_bed(args.features, feat_bed)
            n_iv = sum(1 for _ in feat_bed.open())
            print(f"  wrote {feat_bed} ({n_iv} intervals)", flush=True)

        idx_path = cache / f"bin_idx_{args.features}.npy"
        if idx_path.is_file():
            idx = np.load(idx_path)
            print(f"[features] loaded {idx.size} bins from {idx_path}", flush=True)
        else:
            print(f"[features] intersecting regions ∩ {args.features}...", flush=True)
            idx = feature_bin_indices(reg_bed, feat_bed, idx_path)
            print(f"  n_feature_bins={idx.size}", flush=True)

    feat_tag = args.features
    if args.open_chromatin is not None:
        open_bed = Path(args.open_chromatin)
        open_idx_path = cache / "bin_idx_open_chromatin.npy"
        if open_idx_path.is_file():
            open_idx = np.load(open_idx_path)
            print(
                f"[features] loaded {open_idx.size} open bins from {open_idx_path}",
                flush=True,
            )
        else:
            print(f"[features] open chromatin mask from {open_bed}...", flush=True)
            open_mask = open_chromatin_bin_mask(regions, open_bed)
            if open_mask is None:
                raise SystemExit(f"open chromatin mask disabled/missing: {open_bed}")
            open_idx = np.flatnonzero(open_mask).astype(np.int64)
            np.save(open_idx_path, open_idx)
            print(f"  n_open_bins={open_idx.size} → {open_idx_path}", flush=True)
        n_before = idx.size
        idx = np.intersect1d(idx, open_idx, assume_unique=False)
        feat_tag = f"{args.features}_open"
        print(
            f"[features] ∩ open chromatin: {n_before} → {idx.size} bins ({feat_tag})",
            flush=True,
        )
        if idx.size < 100:
            raise SystemExit(f"too few bins after open filter: {idx.size}")

    raw_pb = np.zeros((idx.size, n_tf), dtype=np.float64)
    imp_pb = np.zeros((idx.size, n_tf), dtype=np.float64)
    bulk_pb = np.zeros((idx.size, n_tf), dtype=np.float64)

    for j, (name, key) in enumerate(PANEL):
        work = args.work_root / key
        print(f"[{name}] loading raw / imputed / bulk...", flush=True)
        if not (work / "mm" / "matrix.mtx.gz").is_file():
            raise SystemExit(f"missing raw mm for {name}")
        if not (work / "impute" / "matrix_csr.npz").is_file():
            raise SystemExit(f"missing impute for {name}")
        bam = BULK_BAM[name]
        if not bam.is_file():
            raise SystemExit(f"missing bulk BAM for {name}: {bam}")

        r = load_pseudobulk(work, "raw")
        u = load_pseudobulk(work, "imputed")
        if r.size != u.size:
            raise SystemExit(f"{name}: shape mismatch raw={r.size} imputed={u.size}")
        if r.size != n_reg_full:
            raise SystemExit(
                f"{name}: matrix bins={r.size} != regions={n_reg_full}"
            )
        raw_pb[:, j] = r[idx]
        imp_pb[:, j] = u[idx]

        bulk_cache = cache / f"bulk_bedcov_{key}.npy"
        b = load_bulk_bedcov(reg_bed, bam, bulk_cache, n_regions=n_reg_full)
        bulk_pb[:, j] = b[idx]
        print(
            f"  raw_nz={(raw_pb[:, j] > 0).mean():.3f}  "
            f"imp_nz={(imp_pb[:, j] > 0).mean():.3f}  "
            f"bulk_nz={(bulk_pb[:, j] > 0).mean():.3f}  "
            f"bulk_med={np.median(bulk_pb[:, j]):.1f}",
            flush=True,
        )

    pairs = [
        ("raw", raw_pb, "raw", "bulk", f"spearman_{feat_tag}_raw_vs_bulk"),
        (
            "imputed",
            imp_pb,
            "imputed",
            "bulk",
            f"spearman_{feat_tag}_imputed_vs_bulk",
        ),
        ("bulk", bulk_pb, "bulk", "bulk", f"spearman_{feat_tag}_bulk_vs_bulk"),
    ]
    for pretty_row, mat, row_lab, col_lab, tag in pairs:
        print(f"[corr] Spearman {pretty_row} × {col_lab}...", flush=True)
        R = corr_matrix(mat, bulk_pb, "spearman")
        tsv = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv, R, labels, f"{row_lab}/{col_lab}")
        print(f"Wrote {tsv}")
        print_matrix(f"Spearman {pretty_row} × {col_lab}", R, labels)

        title = f"Spearman  {pretty_row} × {col_lab}  ({feat_tag} bins)"
        png = args.out_dir / f"{tag}.png"
        png_annot = args.out_dir / f"{tag}_annot.png"
        print(f"[plot] {tag}...", flush=True)
        plot_complexheatmap(
            tsv, png, title, row_title=row_lab, col_title=col_lab, annot=False
        )
        plot_complexheatmap(
            tsv, png_annot, title, row_title=row_lab, col_title=col_lab, annot=True
        )
        print(f"Wrote {png}\nWrote {png_annot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
