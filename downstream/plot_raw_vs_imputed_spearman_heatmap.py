#!/usr/bin/env python3
"""Cross-modality concordance: raw pseudobulk (rows) × imputed (cols).

Feature space: 1 kb bins overlapping the ENCODE SCREEN cCRE panel (293 + 293T),
not genome-wide empty tiles. Pseudobulks: raw = sum over cells; imputed = mean
over cells (occupancy fraction after sparsity-matched binarization). Correlation:
Spearman (rank-based; avoids count-vs-binary scale mismatch).

Outputs under downstream/plots/tf_bulk_pearson/:
  spearman_ccre.tsv / .png              Spearman ρ, color scale −1..1

Usage:
  python3 downstream/plot_raw_vs_imputed_spearman_heatmap.py
  python3 downstream/plot_raw_vs_imputed_spearman_heatmap.py --features union_peaks
  python3 downstream/plot_raw_vs_imputed_spearman_heatmap.py --tfs all --out-dir downstream/plots/tf_bulk_pearson/all_tfs
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

PANEL = [
    ("CTCF", "ctcf"),
    ("MAZ", "maz"),
    ("ZIC2", "zic2"),
    ("ZBTB7A", "zbtb7a"),
    ("ZNF777", "znf777"),
    ("NFYA", "nfya"),
    ("ZNF282", "znf282"),
]


def discover_tfs(work_root: Path) -> list[tuple[str, str]]:
    """All TF dirs with both raw mm and imputed CSR (label, key)."""
    found: list[tuple[str, str]] = []
    for d in sorted(work_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "mm" / "matrix.mtx.gz").is_file():
            continue
        if not (d / "impute" / "matrix_csr.npz").is_file():
            continue
        key = d.name
        found.append((key.upper(), key))
    return found


def resolve_panel(work_root: Path, tfs_arg: str) -> list[tuple[str, str]]:
    """Parse --tfs: 'panel' (default 7), 'all', or comma-separated keys."""
    raw = tfs_arg.strip().lower()
    if raw in {"panel", "default", "bulk7", "7"}:
        return list(PANEL)
    if raw == "all":
        panel = discover_tfs(work_root)
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


def corr_matrix(raw_pb: np.ndarray, imp_pb: np.ndarray, method: str) -> np.ndarray:
    """Cross-modality corr: rows=raw TFs, cols=imputed TFs.

    For large TF panels, Spearman is Pearson-of-ranks via one corrcoef call.
    """
    n = raw_pb.shape[1]
    if method == "spearman":
        # Rank each TF column once, then Pearson on ranks (= Spearman).
        raw_r = np.empty_like(raw_pb)
        imp_r = np.empty_like(imp_pb)
        for j in range(n):
            raw_r[:, j] = rankdata(raw_pb[:, j], method="average")
            imp_r[:, j] = rankdata(imp_pb[:, j], method="average")
        # corrcoef wants variables as rows when rowvar=True (default).
        stacked = np.concatenate([raw_r.T, imp_r.T], axis=0)  # (2n, n_bins)
        C = np.corrcoef(stacked)
        return C[:n, n:].astype(np.float64)
    if method == "pearson":
        stacked = np.concatenate([raw_pb.T, imp_pb.T], axis=0)
        C = np.corrcoef(stacked)
        return C[:n, n:].astype(np.float64)
    raise SystemExit(f"unknown method {method}")


def write_tsv(path: Path, M: np.ndarray, labels: list[str]) -> None:
    with path.open("w") as f:
        f.write("raw/imputed\t" + "\t".join(labels) + "\n")
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
    annot: bool = False,
    legend: str = "Spearman",
    n_tf: int | None = None,
) -> None:
    n = n_tf
    if n is None:
        # Infer from TSV width when caller omits it.
        with tsv.open() as fh:
            n = len(fh.readline().rstrip().split("\t")) - 1

    # Scale canvas / cell size for giant panels (labels stay readable).
    if n <= 10:
        cell_mm = 7.5
        name_fs = 10
        annot_fs = 8
        border_lwd = 0.5
        png_w, png_h, png_res = 1700, 1550, 220
    elif n <= 30:
        cell_mm = 4.5
        name_fs = 7
        annot_fs = 5
        border_lwd = 0.3
        png_w = max(2200, int(n * 55 + 400))
        png_h = max(2000, int(n * 55 + 350))
        png_res = 180
    else:
        cell_mm = 2.2
        name_fs = 5
        annot_fs = 3
        border_lwd = 0.15
        png_w = max(3200, int(n * 28 + 500))
        png_h = max(3000, int(n * 28 + 450))
        png_res = 160

    if annot:
        cell_fun = textwrap.dedent(
            f"""\
            cell_fun = function(j, i, x, y, width, height, fill) {{
              grid.text(
                sprintf("%.2f", mat[i, j]),
                x, y,
                gp = gpar(fontsize = {annot_fs})
              )
            }}
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

        tab <- read.delim(tsv, check.names = FALSE, row.names = 1)
        mat <- as.matrix(tab)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#313695", "#FFFFBF", "#A50026"))
        {cell_fun}
        png(png_path, width = {png_w}, height = {png_h}, res = {png_res})
        ht <- Heatmap(
          mat,
          name = legend_name,
          col = col_fun,
          cluster_rows = FALSE,
          cluster_columns = FALSE,
          rect_gp = gpar(col = "white", lwd = {border_lwd}),
          cell_fun = cell_fun,
          column_title = title,
          column_title_gp = gpar(fontsize = 11),
          row_title = "raw",
          row_title_gp = gpar(fontsize = 11),
          row_names_side = "left",
          row_names_gp = gpar(fontsize = {name_fs}),
          column_names_gp = gpar(fontsize = {name_fs}),
          column_names_rot = 90,
          width = ncol(mat) * unit({cell_mm}, "mm"),
          height = nrow(mat) * unit({cell_mm}, "mm"),
          heatmap_legend_param = list(title = legend_name, at = c(-1, 0, 1))
        )
        draw(
          ht,
          column_title = "imputed",
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
        subprocess.run(
            ["/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/Rscript", rpath],
            check=True,
        )
    finally:
        Path(rpath).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-root", type=Path, default=REPO / "unified" / "work")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "plots" / "tf_bulk_pearson"
    )
    ap.add_argument(
        "--tfs",
        default="panel",
        help="TF set: 'panel' (7 bulk-ref TFs), 'all' (every mm+impute under "
        "--work-root), or comma-separated keys (e.g. ctcf,maz,zic2).",
    )
    ap.add_argument(
        "--features",
        choices=["ccre", "union_peaks", "all"],
        default="ccre",
        help="Shared genomic feature space: ccre (default), union_peaks, or all (~3M bins).",
    )
    ap.add_argument(
        "--method",
        choices=["spearman", "pearson", "both"],
        default="spearman",
        help="Correlation method (default: spearman).",
    )
    ap.add_argument(
        "--annot",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Write annotated heatmap (_annot.png). auto=yes if n_tf<=12.",
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Share feature-index cache with the 7-TF plots so cCRE intersect is reused.
    cache = args.cache_dir or (ROOT / "plots" / "tf_bulk_pearson" / "_cache")
    cache.mkdir(parents=True, exist_ok=True)

    panel = resolve_panel(args.work_root, args.tfs)
    labels = [p[0] for p in panel]
    n_tf = len(panel)
    print(f"[panel] n_tf={n_tf}: {', '.join(labels)}", flush=True)

    regions_gz = args.work_root / "ctcf" / "mm" / "regions.tsv.gz"
    if not regions_gz.is_file():
        # Fall back to first TF in the panel for the shared bin universe.
        regions_gz = args.work_root / panel[0][1] / "mm" / "regions.tsv.gz"
    if not regions_gz.is_file():
        raise SystemExit(f"missing {regions_gz}")

    if args.features == "all":
        # Genome-wide: every 1 kb bin (includes vast empty space).
        import gzip

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

    raw_pb = np.zeros((idx.size, n_tf), dtype=np.float64)
    imp_pb = np.zeros((idx.size, n_tf), dtype=np.float64)

    for j, (name, key) in enumerate(panel):
        work = args.work_root / key
        print(f"[{j+1}/{n_tf} {name}] pseudobulk raw=sum, imputed=mean...", flush=True)
        if not (work / "mm" / "matrix.mtx.gz").is_file():
            raise SystemExit(f"missing raw mm for {name}")
        if not (work / "impute" / "matrix_csr.npz").is_file():
            raise SystemExit(f"missing impute for {name}")
        r = load_pseudobulk(work, "raw")
        u = load_pseudobulk(work, "imputed")
        if r.size != u.size:
            raise SystemExit(f"{name}: shape mismatch raw={r.size} imputed={u.size}")
        raw_pb[:, j] = r[idx]
        imp_pb[:, j] = u[idx]
        print(
            f"  raw_nz={(raw_pb[:, j] > 0).mean():.3f}  "
            f"imp_nz={(imp_pb[:, j] > 0).mean():.3f}",
            flush=True,
        )

    do_annot = (
        True
        if args.annot == "yes"
        else False
        if args.annot == "no"
        else n_tf <= 12
    )

    methods = ["spearman", "pearson"] if args.method == "both" else [args.method]
    for method in methods:
        pretty = "Spearman" if method == "spearman" else "Pearson"
        print(f"[corr] {pretty} raw × imputed ({n_tf}×{n_tf})...", flush=True)
        R = corr_matrix(raw_pb, imp_pb, method)

        tag = f"{method}_{args.features}"
        tsv_raw = args.out_dir / f"{tag}.tsv"
        write_tsv(tsv_raw, R, labels)
        print(f"Wrote {tsv_raw}")
        if n_tf <= 12:
            print(f"{pretty} r:")
            print("       " + "".join(f"{l:>8s}" for l in labels))
            for i, lab in enumerate(labels):
                print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(n_tf)))
        print(
            "diag mean/min/max:",
            round(float(np.nanmean(np.diag(R))), 3),
            round(float(np.nanmin(np.diag(R))), 3),
            round(float(np.nanmax(np.diag(R))), 3),
        )

        png_raw = args.out_dir / f"{tag}.png"
        title = f"{pretty}  raw × imputed  ({args.features}, n={n_tf})"
        print(f"[plot] {pretty} ComplexHeatmap (−1..1)...", flush=True)
        plot_complexheatmap(
            tsv_raw, png_raw, title, annot=False, legend=pretty, n_tf=n_tf
        )
        print(f"Wrote {png_raw}")
        if do_annot:
            png_annot = args.out_dir / f"{tag}_annot.png"
            print(f"[plot] {pretty} annotated...", flush=True)
            plot_complexheatmap(
                tsv_raw, png_annot, title, annot=True, legend=pretty, n_tf=n_tf
            )
            print(f"Wrote {png_annot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
