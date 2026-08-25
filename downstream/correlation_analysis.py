#!/usr/bin/env python3
"""Unified correlation analysis for the imputed vs raw / bulk TF panel.

One entry point with subcommands, each replacing a former standalone script and
preserving its arguments, defaults, and output paths:

  raw-vs-imputed      raw pseudobulk (rows) × imputed (cols), Spearman/Pearson
                      heatmap over cCRE / union_peaks / all bins.
                      (was plot_raw_vs_imputed_spearman_heatmap.py)
  vs-bulk-spearman    raw / imputed / bulk × bulk Spearman on shared bins
                      (samtools bedcov); optional --open-chromatin restriction.
                      (was plot_vs_bulk_spearman_heatmap.py)
  vs-bulk-qn-pearson  raw / imputed × bulk after quantile normalization + Pearson.
                      (was plot_vs_bulk_qn_pearson_heatmap.py)
  sc-vs-bulk          sc vs bulk ChIP validation: deepTools/bedcov CPM Spearman +
                      diagonal delta + AUROC vs bulk-peak labels.
                      (was plot_sc_vs_bulk_validation.py)
  raw-vs-unified      per-TF Pearson raw vs unified-imputed metrics table (+JSON).
                      (was raw_vs_unified_pearson.py)

Usage:
  python3 downstream/correlation_analysis.py <subcommand> [options]
  python3 downstream/correlation_analysis.py vs-bulk-spearman --features union_peaks --open-chromatin
  python3 downstream/correlation_analysis.py raw-vs-imputed --tfs all --features ccre
  python3 downstream/correlation_analysis.py raw-vs-unified --tfs znf282 maz
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# --------------------------------------------------------------------------- #
# Shared panel / reference tables                                             #
# --------------------------------------------------------------------------- #
PANEL = [
    ("CTCF", "ctcf"),
    ("MAZ", "maz"),
    ("ZIC2", "zic2"),
    ("ZBTB7A", "zbtb7a"),
    ("ZNF777", "znf777"),
    ("NFYA", "nfya"),
    ("ZNF282", "znf282"),
]

# Panel default for raw-vs-unified (lowercase keys).
DEFAULT_TFS = ["ctcf", "maz", "nfya", "zbtb7a", "zic2", "znf777", "znf282"]

BULK_BAM = {
    "CTCF": REPO / "data" / "ENCFF139DCW.bam",   # HEK293, SE
    "MAZ": REPO / "data" / "MAZ_HEK293.bam",      # PE
    "ZIC2": REPO / "data" / "ZIC2_HEK293.bam",    # PE
    "ZBTB7A": REPO / "data" / "ZBTB7A_HEK293.bam",  # PE
    "ZNF777": REPO / "data" / "ZNF777_HEK293.bam",  # PE
    "NFYA": REPO / "data" / "NFYA_K562.bam",      # K562 SE — cell-line mismatch vs sc
    "ZNF282": REPO / "data" / "ZNF282_HEK293.bam",  # SE ChIP-exo
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
BLACKLIST_GZ = CCRE_CACHE / "hg38-blacklist.v2.bed.gz"
BLACKLIST = CCRE_CACHE / "hg38-blacklist.v2.bed"
if not BLACKLIST.is_file() and BLACKLIST_GZ.is_file():
    BLACKLIST = BLACKLIST_GZ  # deepTools accepts .gz

DEFAULT_OPEN_BED = REPO / "data" / "HEK293T_OmniATAC_idr_optimal.narrowPeak.gz"

# Tool paths (kept identical to the former scripts).
BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"
SAMTOOLS = "samtools"  # vs-bulk-spearman relies on PATH (its sbatch prepends it)
SAMTOOLS_ABS = "/nas/longleaf/rhel9/apps/samtools/1.24/bin/samtools"  # sc-vs-bulk
RSCRIPT = "/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/Rscript"
MULTIBAM_CANDIDATES = [
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/multiBamSummary",
    "/nas/longleaf/rhel9/apps/deeptools/3.5.4/miniconda3/envs/deeptools/bin/multiBamSummary",
    "/nas/longleaf/apps/deeptools/3.2.0/bin/multiBamSummary",
]
MULTIBAM = next((p for p in MULTIBAM_CANDIDATES if Path(p).is_file()), MULTIBAM_CANDIDATES[0])


# --------------------------------------------------------------------------- #
# Shared I/O + feature helpers                                                #
# --------------------------------------------------------------------------- #
def _read_lines(path: Path) -> list[str]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return [ln.strip() for ln in f if ln.strip()]


def regions_bed(regions_tsv_gz: Path, out_bed: Path) -> int:
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
            subprocess.run([BEDTOOLS, "merge", "-i", str(sorted_bed)], check=True, stdout=fout)
    return out_bed


def feature_bin_indices(regions_bed_path: Path, feature_bed: Path, out_idx: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w") as tmp:
        hit = Path(tmp.name)
    try:
        with hit.open("w") as fout:
            subprocess.run(
                [BEDTOOLS, "intersect", "-u", "-a", str(regions_bed_path), "-b", str(feature_bed)],
                check=True, stdout=fout,
            )
        idx = [int(line.rstrip().split("\t")[3]) for line in hit.open()]
        arr = np.unique(np.asarray(idx, dtype=np.int64))
        np.save(out_idx, arr)
        return arr
    finally:
        hit.unlink(missing_ok=True)


def write_subset_bed(regions_bed_path: Path, idx: np.ndarray, out_bed: Path) -> None:
    want = set(int(x) for x in idx)
    with regions_bed_path.open() as fin, out_bed.open("w") as fout:
        for line in fin:
            bi = int(line.split("\t")[3])
            if bi in want:
                fout.write(line)


def load_pseudobulk(work: Path, kind: str) -> np.ndarray:
    if kind == "raw":
        mat = sio.mmread(str(work / "mm" / "matrix.mtx.gz")).tocsr()
        return np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    mat = sp.load_npz(work / "impute" / "matrix_csr.npz").tocsr()
    return np.asarray(mat.mean(axis=1)).ravel().astype(np.float64)


def load_bulk_bedcov(regions_bed_path: Path, bam: Path, cache_npy: Path, *, n_regions: int) -> np.ndarray:
    """Per-bin read counts from samtools bedcov; order matches regions bed (col4 = index)."""
    if cache_npy.is_file():
        arr = np.load(cache_npy)
        if arr.size == n_regions:
            return arr
        print(f"  stale cache {cache_npy.name} (len={arr.size} != {n_regions}); recomputing", flush=True)
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
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for i, line in enumerate(proc.stdout):
        parts = line.rstrip("\n").split("\t")
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
        print(f"  warning: bedcov lines={i + 1} vs n_regions={n_regions}", flush=True)
    np.save(cache_npy, cov)
    return cov


# --------------------------------------------------------------------------- #
# Correlation helpers                                                         #
# --------------------------------------------------------------------------- #
def corr_matrix_pairwise(row_pb: np.ndarray, col_pb: np.ndarray, method: str = "spearman") -> np.ndarray:
    """Pairwise corr of row_pb columns (rows) vs col_pb columns (cols)."""
    n = row_pb.shape[1]
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if method == "spearman":
                rho, _ = spearmanr(row_pb[:, i], col_pb[:, j])
                R[i, j] = float(rho)
            else:
                x, y = row_pb[:, i], col_pb[:, j]
                R[i, j] = float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
    return R


def corr_cross_modality(raw_pb: np.ndarray, imp_pb: np.ndarray, method: str) -> np.ndarray:
    """Cross-modality corr: rows=raw TFs, cols=imputed TFs (vectorized)."""
    n = raw_pb.shape[1]
    if method == "spearman":
        raw_r = np.empty_like(raw_pb)
        imp_r = np.empty_like(imp_pb)
        for j in range(n):
            raw_r[:, j] = rankdata(raw_pb[:, j], method="average")
            imp_r[:, j] = rankdata(imp_pb[:, j], method="average")
        stacked = np.concatenate([raw_r.T, imp_r.T], axis=0)  # (2n, n_bins)
        C = np.corrcoef(stacked)
        return C[:n, n:].astype(np.float64)
    if method == "pearson":
        stacked = np.concatenate([raw_pb.T, imp_pb.T], axis=0)
        C = np.corrcoef(stacked)
        return C[:n, n:].astype(np.float64)
    raise SystemExit(f"unknown method {method}")


def quantile_normalize(X: np.ndarray) -> np.ndarray:
    """Classic QN on columns of X (n_bins × n_samples)."""
    n, k = X.shape
    order = np.argsort(X, axis=0, kind="mergesort")
    sorted_X = np.take_along_axis(X, order, axis=0)
    mean_sorted = sorted_X.mean(axis=1)
    out = np.empty_like(X, dtype=np.float64)
    ranks = np.empty_like(order)
    for j in range(k):
        ranks[order[:, j], j] = np.arange(n)
    for j in range(k):
        out[:, j] = mean_sorted[ranks[:, j]]
    return out


def apply_qn(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cols":
        return quantile_normalize(X)
    if mode == "rows":
        return quantile_normalize(X.T).T
    if mode == "both":
        return quantile_normalize(quantile_normalize(X).T).T
    raise SystemExit(f"unknown --qn mode {mode!r}")


def pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson corr of columns of A (rows) vs columns of B (cols)."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    As = np.sqrt((Ac * Ac).sum(axis=0))
    Bs = np.sqrt((Bc * Bc).sum(axis=0))
    As = np.where(As == 0, np.nan, As)
    Bs = np.where(Bs == 0, np.nan, Bs)
    return (Ac.T @ Bc) / np.outer(As, Bs)


def write_tsv(path: Path, M: np.ndarray, labels: list[str], corner: str) -> None:
    with path.open("w") as f:
        f.write(corner + "\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "\t" + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels))) + "\n")


def print_matrix(name: str, R: np.ndarray, labels: list[str], *, show_best: bool = False) -> None:
    print(f"{name}:")
    print("       " + "".join(f"{l:>8s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(len(labels))))
    print("diag:", {labels[i]: round(float(R[i, i]), 3) for i in range(len(labels))})
    if show_best:
        n_best = 0
        for i, rl in enumerate(labels):
            jmax = int(np.nanargmax(R[i, :]))
            ok = labels[jmax] == rl
            n_best += int(ok)
            print(f"  row {rl}: best={labels[jmax]} r={R[i, jmax]:.3f} diag={R[i, i]:.3f}{' OK' if ok else ''}")
        print(f"  diagonal is row-max in {n_best}/{len(labels)} rows")


# --------------------------------------------------------------------------- #
# Heatmap renderers (ComplexHeatmap via Rscript)                             #
# --------------------------------------------------------------------------- #
def _run_rscript(r_code: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as tmp:
        tmp.write(r_code)
        rpath = tmp.name
    try:
        subprocess.run([RSCRIPT, rpath], check=True)
    finally:
        Path(rpath).unlink(missing_ok=True)


def plot_heatmap_panel(tsv: Path, png: Path, title: str, *, row_title: str, col_title: str,
                       annot: bool = False, legend: str = "Spearman") -> None:
    """Fixed 7.5 mm blue-yellow-red panel heatmap (vs-bulk-spearman / qn-pearson)."""
    cell_fun = (
        textwrap.dedent(
            """\
            cell_fun = function(j, i, x, y, width, height, fill) {
              grid.text(sprintf("%.2f", mat[i, j]), x, y, gp = gpar(fontsize = 8))
            }
            """
        )
        if annot else "cell_fun = NULL\n"
    )
    r_code = textwrap.dedent(
        f"""\
        suppressPackageStartupMessages({{
          library(ComplexHeatmap); library(circlize); library(grid)
        }})
        tab <- read.delim("{tsv}", check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#313695", "#FFFFBF", "#A50026"))
        {cell_fun}
        png("{png}", width = 1700, height = 1550, res = 220)
        ht <- Heatmap(
          mat, name = "{legend}", col = col_fun,
          cluster_rows = FALSE, cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = 0.5),
          cell_fun = cell_fun,
          column_title = "{title}", column_title_gp = gpar(fontsize = 11),
          row_title = "{row_title}", row_title_gp = gpar(fontsize = 11),
          row_names_side = "left", column_names_rot = 45,
          width = ncol(mat) * unit(7.5, "mm"),
          height = nrow(mat) * unit(7.5, "mm"),
          heatmap_legend_param = list(title = "{legend}", at = c(-1, 0, 1))
        )
        draw(
          ht, column_title = "{col_title}", column_title_side = "bottom",
          column_title_gp = gpar(fontsize = 11),
          padding = unit(c(4, 4, 4, 10), "mm")
        )
        dev.off()
        cat("Wrote", "{png}", "\\n")
        """
    )
    _run_rscript(r_code)


def plot_heatmap_scaled(tsv: Path, png: Path, title: str, *, annot: bool = False,
                        legend: str = "Spearman", n_tf: int | None = None) -> None:
    """raw × imputed heatmap with canvas scaling for large TF panels."""
    n = n_tf
    if n is None:
        with tsv.open() as fh:
            n = len(fh.readline().rstrip().split("\t")) - 1
    if n <= 10:
        cell_mm, name_fs, annot_fs, border_lwd = 7.5, 10, 8, 0.5
        png_w, png_h, png_res = 1700, 1550, 220
    elif n <= 30:
        cell_mm, name_fs, annot_fs, border_lwd = 4.5, 7, 5, 0.3
        png_w, png_h, png_res = max(2200, int(n * 55 + 400)), max(2000, int(n * 55 + 350)), 180
    else:
        cell_mm, name_fs, annot_fs, border_lwd = 2.2, 5, 3, 0.15
        png_w, png_h, png_res = max(3200, int(n * 28 + 500)), max(3000, int(n * 28 + 450)), 160
    cell_fun = (
        textwrap.dedent(
            f"""\
            cell_fun = function(j, i, x, y, width, height, fill) {{
              grid.text(sprintf("%.2f", mat[i, j]), x, y, gp = gpar(fontsize = {annot_fs}))
            }}
            """
        )
        if annot else "cell_fun = NULL\n"
    )
    r_code = textwrap.dedent(
        f"""\
        suppressPackageStartupMessages({{
          library(ComplexHeatmap); library(circlize); library(grid)
        }})
        tab <- read.delim("{tsv}", check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#313695", "#FFFFBF", "#A50026"))
        {cell_fun}
        png("{png}", width = {png_w}, height = {png_h}, res = {png_res})
        ht <- Heatmap(
          mat, name = "{legend}", col = col_fun,
          cluster_rows = FALSE, cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = {border_lwd}),
          cell_fun = cell_fun,
          column_title = "{title}", column_title_gp = gpar(fontsize = 11),
          row_title = "raw", row_title_gp = gpar(fontsize = 11),
          row_names_side = "left",
          row_names_gp = gpar(fontsize = {name_fs}),
          column_names_gp = gpar(fontsize = {name_fs}),
          column_names_rot = 90,
          width = ncol(mat) * unit({cell_mm}, "mm"),
          height = nrow(mat) * unit({cell_mm}, "mm"),
          heatmap_legend_param = list(title = "{legend}", at = c(-1, 0, 1))
        )
        draw(
          ht, column_title = "imputed", column_title_side = "bottom",
          column_title_gp = gpar(fontsize = 11),
          padding = unit(c(4, 4, 4, 10), "mm")
        )
        dev.off()
        cat("Wrote", "{png}", "\\n")
        """
    )
    _run_rscript(r_code)


def plot_heatmap_rdylbu(tsv: Path, png: Path, title: str, *, row_title: str, col_title: str,
                        annot: bool, footnote: str = "") -> None:
    """RdYlBu heatmap with optional footnote (sc-vs-bulk validation)."""
    cell = (
        textwrap.dedent(
            """\
            cell_fun = function(j, i, x, y, width, height, fill) {
              grid.text(sprintf("%.2f", mat[i, j]), x, y, gp = gpar(fontsize = 8))
            }
            """
        )
        if annot else "cell_fun = NULL\n"
    )
    foot = ""
    if footnote:
        foot = textwrap.dedent(
            f"""\
            grid.text(
              "{footnote}",
              x = unit(0.5, "npc"), y = unit(2, "mm"),
              gp = gpar(fontsize = 7, col = "grey30")
            )
            """
        )
    r_code = textwrap.dedent(
        f"""\
        suppressPackageStartupMessages({{
          library(ComplexHeatmap); library(circlize); library(grid); library(RColorBrewer)
        }})
        tab <- read.delim("{tsv}", check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        cols <- rev(brewer.pal(3, "RdYlBu"))
        col_fun <- colorRamp2(c(-1, 0, 1), cols)
        {cell}
        png("{png}", width = 1750, height = 1650, res = 220)
        ht <- Heatmap(
          mat, name = "Spearman", col = col_fun,
          cluster_rows = FALSE, cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = 0.5),
          cell_fun = cell_fun,
          column_title = "{title}", column_title_gp = gpar(fontsize = 10),
          row_title = "{row_title}", row_title_gp = gpar(fontsize = 11),
          row_names_side = "left", column_names_rot = 45,
          width = ncol(mat) * unit(7.5, "mm"),
          height = nrow(mat) * unit(7.5, "mm"),
          heatmap_legend_param = list(title = "Spearman", at = c(-1, 0, 1))
        )
        draw(
          ht, column_title = "{col_title}", column_title_side = "bottom",
          column_title_gp = gpar(fontsize = 11),
          padding = unit(c(8, 4, 4, 10), "mm")
        )
        {foot}
        dev.off()
        cat("Wrote", "{png}", "\\n")
        """
    )
    _run_rscript(r_code)


# --------------------------------------------------------------------------- #
# sc-vs-bulk: bulk count backends                                            #
# --------------------------------------------------------------------------- #
def cpm_normalize(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=0)
    lib = np.where(lib == 0, 1.0, lib)
    return (counts.astype(np.float64) * 1e6) / lib


def _read_outrawcounts(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        header = f.readline()
        if not header.startswith("#"):
            parts = header.strip().split("\t")
            if parts[0] in ("#'chr'", "#chr", "chr"):
                pass
            else:
                rows.append([float(x) for x in parts[3:]])
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append([float(x) for x in parts[3:]])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != len(PANEL):
        raise SystemExit(f"outRawCounts has {arr.shape[1]} TF cols, expected {len(PANEL)}")
    return arr


def multiBamSummary_counts(bins_bed: Path, out_npz: Path, out_raw: Path, *, n_procs: int = 8) -> np.ndarray:
    if out_raw.is_file() and out_raw.stat().st_size > 0:
        print(f"  [cache] loading {out_raw}", flush=True)
        return _read_outrawcounts(out_raw)
    if not Path(MULTIBAM).is_file():
        raise SystemExit(f"multiBamSummary not found at {MULTIBAM}. Install deeptools in the cistopic env.")
    if not BLACKLIST.is_file():
        raise SystemExit(f"missing blacklist {BLACKLIST}")
    labels = [name for name, _ in PANEL]
    bams = [str(BULK_BAM[name]) for name, _ in PANEL]
    for b in bams:
        if not Path(b).is_file():
            raise SystemExit(f"missing BAM {b}")
        if not Path(b + ".bai").is_file() and not Path(b).with_suffix(".bai").is_file():
            print(f"  indexing {b}...", flush=True)
            subprocess.run([SAMTOOLS_ABS, "index", b], check=True)
    cmd = [
        MULTIBAM, "BED-file", "--BED", str(bins_bed), "--bamfiles", *bams, "--labels", *labels,
        "--outRawCounts", str(out_raw), "--outFileName", str(out_npz),
        "--minMappingQuality", "30", "--blackListFileName", str(BLACKLIST),
        "--extendReads", "200", "--numberOfProcessors", str(n_procs),
    ]
    print("  running:", " ".join(cmd[:8]), "...", flush=True)
    subprocess.run(cmd, check=True)
    return _read_outrawcounts(out_raw)


def _bedcov_one(args: tuple) -> tuple[int, np.ndarray]:
    j, name, bins_bed, bam, n_bins = args
    cov = np.zeros(n_bins, dtype=np.float64)
    proc = subprocess.Popen(
        [SAMTOOLS_ABS, "bedcov", "-Q", "30", str(bins_bed), str(bam)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for i, line in enumerate(proc.stdout):
        cov[i] = float(line.rstrip("\n").split("\t")[-1])
    err = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bedcov failed {name}: {err[:500]}")
    return j, cov


def bedcov_q30_matrix(bins_bed: Path, out_npy: Path, *, blacklist: Path, n_procs: int = 7) -> np.ndarray:
    if out_npy.is_file():
        print(f"  [cache] loading {out_npy}", flush=True)
        return np.load(out_npy)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_bins = sum(1 for _ in bins_bed.open())
    counts = np.zeros((n_bins, len(PANEL)), dtype=np.float64)
    jobs = []
    for j, (name, _) in enumerate(PANEL):
        bam = BULK_BAM[name]
        if not bam.is_file():
            raise SystemExit(f"missing BAM {bam}")
        jobs.append((j, name, str(bins_bed), str(bam), n_bins))
    print(f"  bedcov -Q 30 × {len(PANEL)} BAMs (parallel)...", flush=True)
    with ProcessPoolExecutor(max_workers=min(n_procs, len(PANEL))) as ex:
        futs = {ex.submit(_bedcov_one, job): job[1] for job in jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            j, cov = fut.result()
            counts[:, j] = cov
            print(f"    done {name} nz={(cov > 0).mean():.3f}", flush=True)
    if blacklist.is_file():
        print("  masking blacklist-overlapping bins...", flush=True)
        with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp:
            hit = Path(tmp.name)
        try:
            with hit.open("w") as fout:
                subprocess.run(
                    [BEDTOOLS, "intersect", "-u", "-a", str(bins_bed), "-b", str(blacklist)],
                    check=True, stdout=fout,
                )
            bad_idx = {int(l.split("\t")[3]) for l in hit.open()}
            bad_rows = []
            with bins_bed.open() as fbed:
                for i, line in enumerate(fbed):
                    if int(line.split("\t")[3]) in bad_idx:
                        bad_rows.append(i)
            if bad_rows:
                counts[np.asarray(bad_rows, dtype=np.int64), :] = 0.0
                print(f"  zeroed {len(bad_rows)} blacklist bins", flush=True)
        finally:
            hit.unlink(missing_ok=True)
    np.save(out_npy, counts)
    return counts


def peak_membership(regions_bed_path: Path, peak_bed: Path, idx: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp:
        sub = Path(tmp.name)
    try:
        write_subset_bed(regions_bed_path, idx, sub)
        with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp2:
            hit = Path(tmp2.name)
        try:
            with hit.open("w") as fout:
                subprocess.run(
                    [BEDTOOLS, "intersect", "-u", "-a", str(sub), "-b", str(peak_bed)],
                    check=True, stdout=fout,
                )
            hits = {int(l.split("\t")[3]) for l in hit.open()}
        finally:
            hit.unlink(missing_ok=True)
    finally:
        sub.unlink(missing_ok=True)
    return np.asarray([int(i) in hits for i in idx], dtype=bool)


# --------------------------------------------------------------------------- #
# raw-vs-unified helpers                                                      #
# --------------------------------------------------------------------------- #
def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_aligned(mm_dir: Path, impute_dir: Path) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    raw = sio.mmread(str(mm_dir / "matrix.mtx.gz")).tocsr()
    imp = sp.load_npz(impute_dir / "matrix_csr.npz").tocsr()
    if raw.shape != imp.shape:
        raise SystemExit(f"shape mismatch raw{raw.shape} vs impute{imp.shape} in {mm_dir}")
    n_raw = len(_read_lines(mm_dir / "barcodes.tsv.gz"))
    n_imp = len(_read_lines(impute_dir / "barcodes.tsv"))
    if n_raw != raw.shape[1] or n_imp != imp.shape[1]:
        raise SystemExit(f"barcode count mismatch: mm={n_raw}/{raw.shape[1]} impute={n_imp}/{imp.shape[1]}")
    return raw, imp


def _final_bin_idx(tf: str) -> np.ndarray | None:
    tsv = ROOT / f"bins_{tf}_final" / "pos_neg_bins.tsv"
    if not tsv.exists():
        return None
    idx = []
    with tsv.open() as f:
        header = f.readline()
        cols = header.rstrip("\n").split("\t")
        if "bin_idx_post_blacklist" in cols:
            j = cols.index("bin_idx_post_blacklist")
        elif "bin_idx_orig" in cols:
            j = cols.index("bin_idx_orig")
        else:
            j = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > j and parts[j]:
                idx.append(int(parts[j]))
    if not idx:
        return None
    return np.unique(np.asarray(idx, dtype=np.int64))


def _per_axis_pearson(a: sp.csr_matrix, b: sp.csr_matrix, axis: int, sample_n: int | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if axis == 1:
        a = a.tocsr(); b = b.tocsr()
        n = a.shape[0]
        out = np.full(n, np.nan, dtype=np.float64)
        take = np.arange(n)
        if sample_n is not None and 0 < sample_n < n:
            take = np.sort(rng.choice(n, size=sample_n, replace=False))
        for i in take:
            out[i] = _safe_pearson(a.getrow(i).toarray().ravel(), b.getrow(i).toarray().ravel())
        return out
    a = a.tocsc(); b = b.tocsc()
    n = a.shape[1]
    out = np.full(n, np.nan, dtype=np.float64)
    take = np.arange(n)
    if sample_n is not None and 0 < sample_n < n:
        take = np.sort(rng.choice(n, size=sample_n, replace=False))
    for j in take:
        out[j] = _safe_pearson(a.getcol(j).toarray().ravel(), b.getcol(j).toarray().ravel())
    return out


def _analyze_tf_pearson(tf: str, work_root: Path, seed: int, bin_sample: int) -> dict:
    mm = work_root / tf / "mm"
    impute = work_root / tf / "impute"
    if not (mm / "matrix.mtx.gz").exists():
        raise SystemExit(f"[{tf}] missing {mm / 'matrix.mtx.gz'}")
    if not (impute / "matrix_csr.npz").exists():
        raise SystemExit(f"[{tf}] missing {impute / 'matrix_csr.npz'}")
    print(f"[{tf}] loading matrices...", flush=True)
    raw, imp = _load_aligned(mm, impute)
    n_bins, n_cells = raw.shape
    raw_bin_tot = np.asarray(raw.sum(axis=1)).ravel()
    imp_bin_tot = np.asarray(imp.sum(axis=1)).ravel()
    raw_cell_tot = np.asarray(raw.sum(axis=0)).ravel()
    imp_cell_tot = np.asarray(imp.sum(axis=0)).ravel()
    out: dict = {
        "tf": tf, "n_bins": n_bins, "n_cells": n_cells,
        "raw_nnz": int(raw.nnz), "impute_nnz": int(imp.nnz),
        "r_bin_totals": _safe_pearson(raw_bin_tot, imp_bin_tot),
        "r_cell_totals": _safe_pearson(raw_cell_tot, imp_cell_tot),
    }
    bin_idx = _final_bin_idx(tf)
    scope = "final_posneg"
    if bin_idx is None:
        scope = f"random_{bin_sample}"
        rng = np.random.default_rng(seed)
        bin_idx = np.sort(rng.choice(n_bins, size=min(bin_sample, n_bins), replace=False))
    else:
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]
    out["profile_scope"] = scope
    out["n_profile_bins"] = int(bin_idx.size)
    print(f"[{tf}] profile on {bin_idx.size} bins ({scope}); per-cell + per-bin Pearson...", flush=True)
    raw_sub = (raw[bin_idx] > 0).astype(np.float32)
    imp_sub = (imp[bin_idx] > 0).astype(np.float32)
    per_cell = _per_axis_pearson(raw_sub, imp_sub, axis=0, sample_n=None, seed=seed)
    per_bin = _per_axis_pearson(raw_sub, imp_sub, axis=1, sample_n=None, seed=seed)

    def _summ(arr: np.ndarray, prefix: str) -> None:
        ok = arr[np.isfinite(arr)]
        out[f"{prefix}_n"] = int(ok.size)
        out[f"{prefix}_median"] = float(np.median(ok)) if ok.size else float("nan")
        out[f"{prefix}_mean"] = float(np.mean(ok)) if ok.size else float("nan")

    _summ(per_cell, "r_per_cell")
    _summ(per_bin, "r_per_bin")
    ok_c = per_cell[np.isfinite(per_cell)]
    out["frac_cells_r_gt_0.99"] = float(np.mean(ok_c > 0.99)) if ok_c.size else float("nan")
    out["r_bin_totals_profile"] = _safe_pearson(raw_bin_tot[bin_idx], imp_bin_tot[bin_idx])
    return out


# --------------------------------------------------------------------------- #
# Subcommand: raw-vs-imputed                                                  #
# --------------------------------------------------------------------------- #
def _discover_tfs(work_root: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for d in sorted(work_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "mm" / "matrix.mtx.gz").is_file():
            continue
        if not (d / "impute" / "matrix_csr.npz").is_file():
            continue
        found.append((d.name.upper(), d.name))
    return found


def _resolve_panel(work_root: Path, tfs_arg: str) -> list[tuple[str, str]]:
    raw = tfs_arg.strip().lower()
    if raw in {"panel", "default", "bulk7", "7"}:
        return list(PANEL)
    if raw == "all":
        panel = _discover_tfs(work_root)
        if not panel:
            raise SystemExit(f"no TF dirs with mm+impute under {work_root}")
        return panel
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise SystemExit("--tfs is empty")
    panel: list[tuple[str, str]] = []
    for key in keys:
        work = work_root / key
        if not (work / "mm" / "matrix.mtx.gz").is_file():
            raise SystemExit(f"missing raw mm for {key}: {work / 'mm'}")
        if not (work / "impute" / "matrix_csr.npz").is_file():
            raise SystemExit(f"missing impute for {key}: {work / 'impute'}")
        panel.append((key.upper(), key))
    return panel


def cmd_raw_vs_imputed(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (ROOT / "plots" / "tf_bulk_pearson" / "_cache")
    cache.mkdir(parents=True, exist_ok=True)

    panel = _resolve_panel(args.work_root, args.tfs)
    labels = [p[0] for p in panel]
    n_tf = len(panel)
    print(f"[panel] n_tf={n_tf}: {', '.join(labels)}", flush=True)

    regions_gz = args.work_root / "ctcf" / "mm" / "regions.tsv.gz"
    if not regions_gz.is_file():
        regions_gz = args.work_root / panel[0][1] / "mm" / "regions.tsv.gz"
    if not regions_gz.is_file():
        raise SystemExit(f"missing {regions_gz}")

    if args.features == "all":
        with gzip.open(regions_gz, "rt") as fin:
            n_reg = sum(1 for _ in fin)
        idx = np.arange(n_reg, dtype=np.int64)
        print(f"[features] all bins: n={idx.size}", flush=True)
    else:
        reg_bed = cache / "regions_1kb.bed"
        if not reg_bed.is_file():
            print(f"[features] writing regions bed from {regions_gz}", flush=True)
            print(f"  n_regions={regions_bed(regions_gz, reg_bed)}", flush=True)
        feat_bed = cache / f"features_{args.features}.bed"
        if not feat_bed.is_file():
            print(f"[features] building {args.features} bed...", flush=True)
            build_feature_bed(args.features, feat_bed)
            print(f"  wrote {feat_bed} ({sum(1 for _ in feat_bed.open())} intervals)", flush=True)
        idx_path = cache / f"bin_idx_{args.features}.npy"
        if idx_path.is_file():
            idx = np.load(idx_path)
            print(f"[features] loaded {idx.size} bins from {idx_path}", flush=True)
        else:
            print(f"[features] intersecting regions ∩ {args.features}...", flush=True)
            idx = feature_bin_indices(reg_bed, feat_bed, idx_path)
            print(f"  n_feature_bins={idx.size}", flush=True)

    raw_pb = np.zeros((idx.size, n_tf), dtype=np.float64)
    imp_pb = np.zeros((idx.size, n_tf), dtype=np.float64)
    for j, (name, key) in enumerate(panel):
        work = args.work_root / key
        print(f"[{j+1}/{n_tf} {name}] pseudobulk raw=sum, imputed=mean...", flush=True)
        r = load_pseudobulk(work, "raw")
        u = load_pseudobulk(work, "imputed")
        if r.size != u.size:
            raise SystemExit(f"{name}: shape mismatch raw={r.size} imputed={u.size}")
        raw_pb[:, j] = r[idx]
        imp_pb[:, j] = u[idx]
        print(f"  raw_nz={(raw_pb[:, j] > 0).mean():.3f}  imp_nz={(imp_pb[:, j] > 0).mean():.3f}", flush=True)

    do_annot = True if args.annot == "yes" else False if args.annot == "no" else n_tf <= 12
    methods = ["spearman", "pearson"] if args.method == "both" else [args.method]
    for method in methods:
        pretty = "Spearman" if method == "spearman" else "Pearson"
        print(f"[corr] {pretty} raw × imputed ({n_tf}×{n_tf})...", flush=True)
        R = corr_cross_modality(raw_pb, imp_pb, method)
        tag = f"{method}_{args.features}"
        tsv_raw = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv_raw, R, labels, "raw/imputed")
        print(f"Wrote {tsv_raw}")
        if n_tf <= 12:
            print(f"{pretty} r:")
            print("       " + "".join(f"{l:>8s}" for l in labels))
            for i, lab in enumerate(labels):
                print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(n_tf)))
        print("diag mean/min/max:", round(float(np.nanmean(np.diag(R))), 3),
              round(float(np.nanmin(np.diag(R))), 3), round(float(np.nanmax(np.diag(R))), 3))
        png_raw = args.out_dir / f"{tag}.png"
        title = f"{pretty}  raw × imputed  ({args.features}, n={n_tf})"
        print(f"[plot] {pretty} ComplexHeatmap (−1..1)...", flush=True)
        plot_heatmap_scaled(tsv_raw, png_raw, title, annot=False, legend=pretty, n_tf=n_tf)
        print(f"Wrote {png_raw}")
        if do_annot:
            png_annot = args.out_dir / f"{tag}_annot.png"
            print(f"[plot] {pretty} annotated...", flush=True)
            plot_heatmap_scaled(tsv_raw, png_annot, title, annot=True, legend=pretty, n_tf=n_tf)
            print(f"Wrote {png_annot}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: vs-bulk-spearman                                                #
# --------------------------------------------------------------------------- #
def cmd_vs_bulk_spearman(args: argparse.Namespace) -> int:
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
            print(f"  wrote {feat_bed} ({sum(1 for _ in feat_bed.open())} intervals)", flush=True)
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
        sys.path.insert(0, str(REPO / "unified" / "scripts"))
        from _regions import open_chromatin_bin_mask  # noqa: E402
        open_bed = Path(args.open_chromatin)
        open_idx_path = cache / "bin_idx_open_chromatin.npy"
        if open_idx_path.is_file():
            open_idx = np.load(open_idx_path)
            print(f"[features] loaded {open_idx.size} open bins from {open_idx_path}", flush=True)
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
        print(f"[features] ∩ open chromatin: {n_before} → {idx.size} bins ({feat_tag})", flush=True)
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
            raise SystemExit(f"{name}: matrix bins={r.size} != regions={n_reg_full}")
        raw_pb[:, j] = r[idx]
        imp_pb[:, j] = u[idx]
        bulk_cache = cache / f"bulk_bedcov_{key}.npy"
        b = load_bulk_bedcov(reg_bed, bam, bulk_cache, n_regions=n_reg_full)
        bulk_pb[:, j] = b[idx]
        print(f"  raw_nz={(raw_pb[:, j] > 0).mean():.3f}  imp_nz={(imp_pb[:, j] > 0).mean():.3f}  "
              f"bulk_nz={(bulk_pb[:, j] > 0).mean():.3f}  bulk_med={np.median(bulk_pb[:, j]):.1f}", flush=True)

    pairs = [
        ("raw", raw_pb, "raw", "bulk", f"spearman_{feat_tag}_raw_vs_bulk"),
        ("imputed", imp_pb, "imputed", "bulk", f"spearman_{feat_tag}_imputed_vs_bulk"),
        ("bulk", bulk_pb, "bulk", "bulk", f"spearman_{feat_tag}_bulk_vs_bulk"),
    ]
    for pretty_row, mat, row_lab, col_lab, tag in pairs:
        print(f"[corr] Spearman {pretty_row} × {col_lab}...", flush=True)
        R = corr_matrix_pairwise(mat, bulk_pb, "spearman")
        tsv = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv, R, labels, f"{row_lab}/{col_lab}")
        print(f"Wrote {tsv}")
        print_matrix(f"Spearman {pretty_row} × {col_lab}", R, labels)
        title = f"Spearman  {pretty_row} × {col_lab}  ({feat_tag} bins)"
        png = args.out_dir / f"{tag}.png"
        png_annot = args.out_dir / f"{tag}_annot.png"
        print(f"[plot] {tag}...", flush=True)
        plot_heatmap_panel(tsv, png, title, row_title=row_lab, col_title=col_lab, annot=False)
        plot_heatmap_panel(tsv, png_annot, title, row_title=row_lab, col_title=col_lab, annot=True)
        print(f"Wrote {png}\nWrote {png_annot}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: vs-bulk-qn-pearson                                              #
# --------------------------------------------------------------------------- #
def cmd_vs_bulk_qn_pearson(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    labels = [p[0] for p in PANEL]
    n_tf = len(PANEL)

    if args.features == "all":
        idx = None
        print("[features] all bins", flush=True)
    else:
        idx_path = cache / f"bin_idx_{args.features}.npy"
        if not idx_path.is_file():
            raise SystemExit(
                f"missing {idx_path}; run 'correlation_analysis.py vs-bulk-spearman "
                f"--features {args.features}' first to build the index"
            )
        idx = np.load(idx_path)
        print(f"[features] {args.features}: n={idx.size}", flush=True)

    raw_pb = imp_pb = bulk_pb = None
    n_use = None
    for j, (name, key) in enumerate(PANEL):
        print(f"[{name}] loading...", flush=True)
        work = args.work_root / key
        r = load_pseudobulk(work, "raw")
        u = load_pseudobulk(work, "imputed")
        b = np.load(cache / f"bulk_bedcov_{key}.npy")
        if r.size != u.size or r.size != b.size:
            raise SystemExit(f"{name}: length mismatch raw/imp/bulk")
        if idx is not None:
            r, u, b = r[idx], u[idx], b[idx]
        if n_use is None:
            n_use = r.size
            raw_pb = np.zeros((n_use, n_tf), dtype=np.float64)
            imp_pb = np.zeros((n_use, n_tf), dtype=np.float64)
            bulk_pb = np.zeros((n_use, n_tf), dtype=np.float64)
        raw_pb[:, j] = r
        imp_pb[:, j] = u
        bulk_pb[:, j] = b

    assert raw_pb is not None and imp_pb is not None and bulk_pb is not None
    qn_label = {"cols": "QN-cols", "rows": "QN-rows", "both": "QN-cols+rows"}[args.qn]
    print(f"[QN] {qn_label} on raw / imputed / bulk (shape={raw_pb.shape})...", flush=True)
    raw_qn = apply_qn(raw_pb, args.qn)
    imp_qn = apply_qn(imp_pb, args.qn)
    bulk_qn = apply_qn(bulk_pb, args.qn)
    print("  after QN bulk col means: "
          + ", ".join(f"{labels[j]}={bulk_qn[:, j].mean():.2f}" for j in range(n_tf)), flush=True)

    qn_suffix = "" if args.qn == "both" else f"_{args.qn}"
    pairs = [
        ("raw", raw_qn, "raw", "bulk", f"pearson_{args.features}_raw_vs_bulk_qn{qn_suffix}"),
        ("imputed", imp_qn, "imputed", "bulk", f"pearson_{args.features}_imputed_vs_bulk_qn{qn_suffix}"),
    ]
    for pretty, mat, row_lab, col_lab, tag in pairs:
        print(f"[corr] Pearson {pretty} × bulk ({qn_label})...", flush=True)
        R = pearson_matrix(mat, bulk_qn)
        tsv = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv, R, labels, f"{row_lab}/{col_lab} {qn_label}")
        print(f"Wrote {tsv}")
        print_matrix(f"Pearson {pretty} × bulk ({qn_label})", R, labels, show_best=True)
        title = f"Pearson  {pretty} × bulk  ({args.features}, {qn_label})"
        plot_heatmap_panel(tsv, args.out_dir / f"{tag}.png", title,
                           row_title=row_lab, col_title=col_lab, annot=False, legend="Pearson")
        plot_heatmap_panel(tsv, args.out_dir / f"{tag}_annot.png", title,
                           row_title=row_lab, col_title=col_lab, annot=True, legend="Pearson")
        print(f"Wrote {args.out_dir / f'{tag}.png'}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: sc-vs-bulk                                                      #
# --------------------------------------------------------------------------- #
def cmd_sc_vs_bulk(args: argparse.Namespace) -> int:
    from sklearn.metrics import roc_auc_score

    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    val_dir = args.out_dir / "sc_vs_bulk_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    labels = [p[0] for p in PANEL]
    n_tf = len(PANEL)

    regions_gz = args.work_root / "ctcf" / "mm" / "regions.tsv.gz"
    reg_bed = cache / "regions_1kb.bed"
    if not reg_bed.is_file():
        print(f"[regions] writing {reg_bed}", flush=True)
        print(f"  n={regions_bed(regions_gz, reg_bed)}", flush=True)
    with gzip.open(regions_gz, "rt") as fin:
        n_reg = sum(1 for _ in fin)
    print(f"[regions] n={n_reg} genome=hg38 (BAM chroms=chr*)", flush=True)

    print("[sc] loading raw / imputed pseudobulks...", flush=True)
    raw_full = np.zeros((n_reg, n_tf), dtype=np.float64)
    imp_full = np.zeros((n_reg, n_tf), dtype=np.float64)
    for j, (name, key) in enumerate(PANEL):
        work = args.work_root / key
        raw_full[:, j] = load_pseudobulk(work, "raw")
        imp_full[:, j] = load_pseudobulk(work, "imputed")
        print(f"  {name} raw_nz={(raw_full[:, j] > 0).mean():.3f} imp_nz={(imp_full[:, j] > 0).mean():.3f}", flush=True)

    delta_rows: list[dict] = []
    auroc_rows: list[dict] = []
    for feat in feats:
        print(f"\n======== feature space: {feat} ========", flush=True)
        if feat == "all":
            idx = np.arange(n_reg, dtype=np.int64)
            print("  NOTE: all-bins is a sensitivity check — co-absence inflates ρ.", flush=True)
        else:
            feat_bed = cache / f"features_{feat}.bed"
            if not feat_bed.is_file():
                print(f"  building {feat} intervals...", flush=True)
                build_feature_bed(feat, feat_bed)
            idx_path = cache / f"bin_idx_{feat}.npy"
            idx = np.load(idx_path) if idx_path.is_file() else feature_bin_indices(reg_bed, feat_bed, idx_path)
            print(f"  n_bins={idx.size}", flush=True)

        bins_bed = cache / f"bins_for_bulk_{feat}.bed"
        if not bins_bed.is_file() or bins_bed.stat().st_size == 0:
            print(f"  writing subset bed {bins_bed.name}...", flush=True)
            write_subset_bed(reg_bed, idx, bins_bed)

        raw_counts_path = cache / f"bulk_multibam_{feat}.tab"
        npz_path = cache / f"bulk_multibam_{feat}.npz"
        bedcov_npy = cache / f"bulk_bedcov_q30_{feat}.npy"
        use_deeptools = Path(MULTIBAM).is_file() and not args.force_bedtools and not args.force_bedcov
        if use_deeptools and not args.skip_deeptools:
            try:
                print(f"  [bulk] deepTools multiBamSummary ({feat})...", flush=True)
                counts = multiBamSummary_counts(bins_bed, npz_path, raw_counts_path, n_procs=args.n_procs)
            except Exception as e:
                print(f"  deepTools failed ({e}); using bedcov -Q 30", flush=True)
                counts = bedcov_q30_matrix(bins_bed, bedcov_npy, blacklist=BLACKLIST, n_procs=args.n_procs)
        else:
            print("  [bulk] samtools bedcov -Q 30 + blacklist mask...", flush=True)
            counts = bedcov_q30_matrix(bins_bed, bedcov_npy, blacklist=BLACKLIST, n_procs=args.n_procs)

        if counts.shape[0] != idx.size:
            raise SystemExit(f"bulk rows {counts.shape[0]} != idx {idx.size} for {feat}")
        bulk_pb = cpm_normalize(counts)
        print(f"  bulk CPM: col_sums={bulk_pb.sum(axis=0).round(1)} (should be ~1e6 each)", flush=True)

        raw_pb = raw_full[idx, :]
        imp_pb = imp_full[idx, :]
        print(f"  [corr] Spearman raw × bulk ({feat})...", flush=True)
        Rb = corr_matrix_pairwise(raw_pb, bulk_pb, "spearman")
        print(f"  [corr] Spearman imputed × bulk ({feat})...", flush=True)
        Ib = corr_matrix_pairwise(imp_pb, bulk_pb, "spearman")
        print_matrix(f"Rb raw×bulk ({feat})", Rb, labels)
        print_matrix(f"Ib imputed×bulk ({feat})", Ib, labels)

        print("\n  TF       raw_diag  imp_diag   delta   note")
        for i, lab in enumerate(labels):
            d = float(Ib[i, i] - Rb[i, i])
            note = "K562 bulk vs HEK293 sc — expect weak/neg delta" if lab == "NFYA" else ""
            print(f"  {lab:8s} {Rb[i, i]:8.4f}  {Ib[i, i]:8.4f}  {d:+7.4f}  {note}")
            delta_rows.append({"features": feat, "TF": lab, "raw_diag": Rb[i, i],
                               "imp_diag": Ib[i, i], "delta": d, "note": note})

        for tag, M, row_t in [
            (f"spearman_{feat}_raw_vs_bulk_cpm", Rb, "raw"),
            (f"spearman_{feat}_imputed_vs_bulk_cpm", Ib, "imputed"),
        ]:
            tsv = val_dir / f"{tag}.tsv"
            write_tsv(tsv, M, labels, f"{row_t}/bulk_CPM")
            title = f"Spearman  {row_t} × bulk CPM  ({feat} bins; MQ≥30, blacklist, extendReads)"
            foot = "NFYA: K562 bulk vs HEK293 sc (cell-line mismatch)"
            plot_heatmap_rdylbu(tsv, val_dir / f"{tag}.png", title, row_title=row_t,
                                col_title="bulk", annot=False, footnote=foot)
            plot_heatmap_rdylbu(tsv, val_dir / f"{tag}_annot.png", title, row_title=row_t,
                                col_title="bulk", annot=True, footnote=foot)

        print(f"  [AUROC] vs bulk-peak labels ({feat})...", flush=True)
        for i, (lab, key) in enumerate(PANEL):
            y = peak_membership(reg_bed, PEAK_BED[lab], idx)
            if y.sum() == 0 or y.sum() == y.size:
                raw_a = imp_a = float("nan")
            else:
                raw_a = float(roc_auc_score(y, raw_pb[:, i]))
                imp_a = float(roc_auc_score(y, imp_pb[:, i]))
            da = imp_a - raw_a if np.isfinite(raw_a) and np.isfinite(imp_a) else float("nan")
            note = ""
            if lab == "NFYA":
                note = "K562 bulk peaks vs HEK293 sc"
            if lab == "CTCF":
                note = "peak-label AUROC secondary; lead with continuous Spearman (legacy peak thresholding asymmetry)"
            print(f"    {lab:8s} raw={raw_a:.4f}  imp={imp_a:.4f}  delta={da:+.4f}  pos={int(y.sum())}/{y.size}")
            auroc_rows.append({"features": feat, "TF": lab, "raw_AUROC": raw_a, "imp_AUROC": imp_a,
                               "delta_AUROC": da, "n_pos": int(y.sum()), "n_bins": int(y.size), "note": note})

    delta_path = val_dir / "diagonal_delta.tsv"
    with delta_path.open("w") as f:
        f.write("features\tTF\traw_diag\timp_diag\tdelta\tnote\n")
        for r in delta_rows:
            f.write(f"{r['features']}\t{r['TF']}\t{r['raw_diag']:.4f}\t{r['imp_diag']:.4f}\t{r['delta']:+.4f}\t{r['note']}\n")
    auroc_path = val_dir / "auroc_vs_bulk_peaks.tsv"
    with auroc_path.open("w") as f:
        f.write("features\tTF\traw_AUROC\timp_AUROC\tdelta_AUROC\tn_pos\tn_bins\tnote\n")
        for r in auroc_rows:
            f.write(f"{r['features']}\t{r['TF']}\t{r['raw_AUROC']:.4f}\t{r['imp_AUROC']:.4f}\t"
                    f"{r['delta_AUROC']:+.4f}\t{r['n_pos']}\t{r['n_bins']}\t{r['note']}\n")
    readme = val_dir / "README.txt"
    readme.write_text(textwrap.dedent(
        """\
        scCUT&Tag vs bulk ChIP validation
        =================================
        Bulk: deepTools multiBamSummary BED-file (fallback: bedtools multicov)
          --minMappingQuality 30, hg38-blacklist.v2, --extendReads 200, then CPM.
        BAMs are hg38 (chr*). One BAM per TF (no replicates to pool).
        Treated as single-end with fixed fragment extend (ZNF282 is SE ChIP-exo).

        Primary feature spaces: cCRE, union_peaks.
        'all' (~3M bins) is a sensitivity check only — co-absence inflates ρ.

        Matrices: spearman_<feat>_raw_vs_bulk_cpm*, spearman_<feat>_imputed_vs_bulk_cpm*
        Tables:   diagonal_delta.tsv, auroc_vs_bulk_peaks.tsv

        NFYA: bulk is K562; sc is HEK293 — flag weak/negative delta as expected.
        CTCF: continuous Spearman is primary; peak-label AUROC is secondary.
        """
    ))
    print(f"\nWrote {delta_path}")
    print(f"Wrote {auroc_path}")
    print(f"Wrote plots under {val_dir}")
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: raw-vs-unified                                                  #
# --------------------------------------------------------------------------- #
def cmd_raw_vs_unified(args: argparse.Namespace) -> int:
    tfs = [t.lower() for t in (args.tfs or DEFAULT_TFS)]
    rows = []
    for tf in tfs:
        try:
            rows.append(_analyze_tf_pearson(tf, args.work_root, args.seed, args.bin_sample))
        except SystemExit as e:
            print(f"ERROR {e}", file=sys.stderr)
            return 1
    cols = [
        "tf", "n_bins", "n_cells", "raw_nnz", "impute_nnz",
        "r_bin_totals", "r_cell_totals", "r_bin_totals_profile",
        "profile_scope", "n_profile_bins",
        "r_per_cell_n", "r_per_cell_median", "r_per_cell_mean", "frac_cells_r_gt_0.99",
        "r_per_bin_n", "r_per_bin_median", "r_per_bin_mean",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols
            ) + "\n")
    print("\n=== Pearson raw vs unified ===")
    hdr = (f"{'TF':<8} {'r_binTot':>9} {'r_cellTot':>9} "
           f"{'r_cellMed':>9} {'r_binMed':>9} {'f_r>0.99':>8}  scope")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tf']:<8} {r['r_bin_totals']:9.4f} {r['r_cell_totals']:9.4f} "
              f"{r['r_per_cell_median']:9.4f} {r['r_per_bin_median']:9.4f} "
              f"{r['frac_cells_r_gt_0.99']:8.3f}  {r['profile_scope']} (n={r['n_profile_bins']})")
    print(f"\nWrote {args.out}")
    js = args.out.with_suffix(".json")
    js.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {js}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_out = ROOT / "plots" / "tf_bulk_pearson"
    default_work = REPO / "unified" / "work"

    p = sub.add_parser("raw-vs-imputed", help="raw × imputed Spearman/Pearson heatmap")
    p.add_argument("--work-root", type=Path, default=default_work)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--tfs", default="panel",
                   help="TF set: 'panel' (7), 'all', or comma-separated keys (e.g. ctcf,maz,zic2).")
    p.add_argument("--features", choices=["ccre", "union_peaks", "all"], default="ccre")
    p.add_argument("--method", choices=["spearman", "pearson", "both"], default="spearman")
    p.add_argument("--annot", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.set_defaults(func=cmd_raw_vs_imputed)

    p = sub.add_parser("vs-bulk-spearman", help="raw/imputed/bulk × bulk Spearman")
    p.add_argument("--work-root", type=Path, default=default_work)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--features", choices=["ccre", "union_peaks", "all"], default="all")
    p.add_argument("--open-chromatin", nargs="?", const=str(DEFAULT_OPEN_BED), default=None, metavar="BED",
                   help=f"Restrict bins to OmniATAC open chromatin (default {DEFAULT_OPEN_BED.name}); writes *_open_* outputs.")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.set_defaults(func=cmd_vs_bulk_spearman)

    p = sub.add_parser("vs-bulk-qn-pearson", help="raw/imputed × bulk QN + Pearson")
    p.add_argument("--work-root", type=Path, default=default_work)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--features", choices=["ccre", "union_peaks", "all"], default="union_peaks")
    p.add_argument("--qn", choices=["cols", "rows", "both"], default="both")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.set_defaults(func=cmd_vs_bulk_qn_pearson)

    p = sub.add_parser("sc-vs-bulk", help="sc vs bulk ChIP validation (CPM Spearman + AUROC)")
    p.add_argument("--work-root", type=Path, default=default_work)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--features", default="ccre,union_peaks",
                   help="Comma list: ccre,union_peaks,all (default: ccre,union_peaks)")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--n-procs", type=int, default=8)
    p.add_argument("--force-bedtools", action="store_true", help="Deprecated alias for --force-bedcov")
    p.add_argument("--force-bedcov", action="store_true", help="Skip deepTools; use samtools bedcov -Q 30")
    p.add_argument("--skip-deeptools", action="store_true", help="Same as --force-bedcov")
    p.set_defaults(func=cmd_sc_vs_bulk)

    p = sub.add_parser("raw-vs-unified", help="per-TF Pearson raw vs unified-imputed table")
    p.add_argument("--tfs", nargs="*", default=None, help="TF ids (lowercase). Default: panel.")
    p.add_argument("--work-root", type=Path, default=default_work)
    p.add_argument("--out", type=Path, default=ROOT / "raw_vs_unified_pearson.tsv")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bin-sample", type=int, default=20000)
    p.set_defaults(func=cmd_raw_vs_unified)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
