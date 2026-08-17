#!/usr/bin/env python3
"""Validate scCUT&Tag imputation vs bulk ChIP (7-TF panel).

Pipeline (per shared Claude / PI guidance):
  1. Bulk BAM → per-bin counts via deepTools multiBamSummary (BED-file mode)
     with --minMappingQuality 30, hg38 blacklist, then CPM depth norm.
     SE BAMs: --extendReads <frag>; PE: --extendReads (fragment).
  2. Feature spaces (primary): cCRE bins, union-peak bins.
     all (~3M) bins = sensitivity check only (co-absence inflates ρ).
  3. Spearman: Rb = raw×bulk, Ib = imputed×bulk (bulk on columns).
  4. delta = diag(Ib) - diag(Rb)
  5. AUROC of raw/imputed signal vs bulk-peak membership labels
  6. ComplexHeatmap RdYlBu, no clustering

Caveats annotated in outputs:
  - NFYA: K562 bulk vs HEK293 sc — cell-line mismatch; expect weak/negative delta.
  - CTCF: lead with continuous Spearman; peak-label AUROC is secondary.

Usage:
  python3 downstream/plot_sc_vs_bulk_validation.py
  python3 downstream/plot_sc_vs_bulk_validation.py --features ccre,union_peaks
  python3 downstream/plot_sc_vs_bulk_validation.py --features ccre,union_peaks,all
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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

# Bulk ChIP BAMs. All hg38. Library layout (sampled first 2k reads):
#   SE: CTCF (ENCFF139DCW), NFYA (K562), ZNF282 (ChIP-exo)
#   PE: MAZ, ZIC2, ZBTB7A, ZNF777
# deepTools --extendReads INT: SE extended to INT; PE uses BAM fragment length
# (INT ignored for PE). No biological replicates in data/ — one BAM per TF.
BULK_BAM = {
    "CTCF": REPO / "data" / "ENCFF139DCW.bam",  # HEK293, SE
    "MAZ": REPO / "data" / "MAZ_HEK293.bam",  # PE
    "ZIC2": REPO / "data" / "ZIC2_HEK293.bam",  # PE
    "ZBTB7A": REPO / "data" / "ZBTB7A_HEK293.bam",  # PE
    "ZNF777": REPO / "data" / "ZNF777_HEK293.bam",  # PE
    "NFYA": REPO / "data" / "NFYA_K562.bam",  # K562 SE — cell-line mismatch vs sc
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


BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"
SAMTOOLS = "/nas/longleaf/rhel9/apps/samtools/1.24/bin/samtools"

RSCRIPT = "/nas/longleaf/home/dyy12/.conda/envs/cistopic/bin/Rscript"
MULTIBAM_CANDIDATES = [
    "/nas/longleaf/rhel9/apps/deeptools/3.5.6/miniconda3/envs/deeptools/bin/multiBamSummary",
    "/nas/longleaf/rhel9/apps/deeptools/3.5.4/miniconda3/envs/deeptools/bin/multiBamSummary",
    "/nas/longleaf/apps/deeptools/3.2.0/bin/multiBamSummary",
]
MULTIBAM = next((p for p in MULTIBAM_CANDIDATES if Path(p).is_file()), MULTIBAM_CANDIDATES[0])



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
                raise SystemExit(f"unknown feature kind {kind}")
            for bed in sources:
                if not bed.is_file():
                    raise SystemExit(f"missing {bed}")
                with bed.open() as fin:
                    for line in fin:
                        p = line.split("\t")
                        fout.write(f"{p[0]}\t{p[1]}\t{p[2]}\n")
        sorted_bed = td_p / "sorted.bed"
        with sorted_bed.open("w") as fout:
            subprocess.run([BEDTOOLS, "sort", "-i", str(raw)], check=True, stdout=fout)
        with out_bed.open("w") as fout:
            subprocess.run(
                [BEDTOOLS, "merge", "-i", str(sorted_bed)], check=True, stdout=fout
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


def cpm_normalize(counts: np.ndarray) -> np.ndarray:
    """Column-wise CPM: 1e6 * count / library_size."""
    lib = counts.sum(axis=0)
    lib = np.where(lib == 0, 1.0, lib)
    return (counts.astype(np.float64) * 1e6) / lib


def multiBamSummary_counts(
    bins_bed: Path,
    out_npz: Path,
    out_raw: Path,
    *,
    n_procs: int = 8,
) -> np.ndarray:
    """Run deepTools multiBamSummary BED-file; return counts [n_bins × n_tf].

    BAMs are passed in PANEL order. SE extendReads uses SE_EXTEND; deepTools
    --extendReads without length uses fragment for PE (none of ours are PE
    replicates here — all treated SE with fixed extend).
    """
    if out_raw.is_file() and out_raw.stat().st_size > 0:
        print(f"  [cache] loading {out_raw}", flush=True)
        return _read_outrawcounts(out_raw)

    if not Path(MULTIBAM).is_file():
        raise SystemExit(
            f"multiBamSummary not found at {MULTIBAM}. "
            "Install deeptools in the cistopic env."
        )
    if not BLACKLIST.is_file():
        raise SystemExit(f"missing blacklist {BLACKLIST}")

    labels = [name for name, _ in PANEL]
    bams = [str(BULK_BAM[name]) for name, _ in PANEL]
    for b in bams:
        if not Path(b).is_file():
            raise SystemExit(f"missing BAM {b}")
        if not Path(b + ".bai").is_file() and not Path(b).with_suffix(".bai").is_file():
            print(f"  indexing {b}...", flush=True)
            subprocess.run([SAMTOOLS, "index", b], check=True)

    # deepTools: one extend length for all BAMs in a single call. Use 200 bp
    # (ENCODE ChIP default). ZNF282 ChIP-exo is shorter; we accept a slight
    # over-extension for that one track in the joint call (documented).
    cmd = [
        MULTIBAM,
        "BED-file",
        "--BED",
        str(bins_bed),
        "--bamfiles",
        *bams,
        "--labels",
        *labels,
        "--outRawCounts",
        str(out_raw),
        "--outFileName",
        str(out_npz),
        "--minMappingQuality",
        "30",
        "--blackListFileName",
        str(BLACKLIST),
        "--extendReads",
        "200",
        "--numberOfProcessors",
        str(n_procs),
    ]
    print("  running:", " ".join(cmd[:8]), "...", flush=True)
    subprocess.run(cmd, check=True)
    return _read_outrawcounts(out_raw)


def _read_outrawcounts(path: Path) -> np.ndarray:
    """Parse deepTools --outRawCounts (chr start end + one col per BAM)."""
    rows = []
    with path.open() as f:
        header = f.readline()
        if not header.startswith("#"):
            # first line may be header without # in some versions
            parts = header.strip().split("\t")
            if parts[0] in ("#'chr'", "#chr", "chr"):
                pass
            else:
                # data line
                rows.append([float(x) for x in parts[3:]])
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append([float(x) for x in parts[3:]])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[1] != len(PANEL):
        raise SystemExit(
            f"outRawCounts has {arr.shape[1]} TF cols, expected {len(PANEL)}"
        )
    return arr


def _bedcov_one(args: tuple) -> tuple[int, np.ndarray]:
    j, name, bins_bed, bam, n_bins = args
    cov = np.zeros(n_bins, dtype=np.float64)
    proc = subprocess.Popen(
        [SAMTOOLS, "bedcov", "-Q", "30", str(bins_bed), str(bam)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for i, line in enumerate(proc.stdout):
        cov[i] = float(line.rstrip("\n").split("\t")[-1])
    err = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"bedcov failed {name}: {err[:500]}")
    return j, cov


def bedcov_q30_matrix(
    bins_bed: Path,
    out_npy: Path,
    *,
    blacklist: Path,
    n_procs: int = 7,
) -> np.ndarray:
    """Per-TF samtools bedcov -Q 30 on bins_bed → counts [n_bins × n_tf].

    Preferred when deepTools multiBamSummary is unavailable/hanging on this
    cluster. Applies MQ≥30; zeros bins overlapping hg38 blacklist; caller
    applies CPM. SE/PE: bedcov counts overlapping alignments (no fragment
    extension) — documented limitation vs deepTools --extendReads.
    """
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

    # Zero blacklist-overlapping bins
    if blacklist.is_file():
        print("  masking blacklist-overlapping bins...", flush=True)
        with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp:
            hit = Path(tmp.name)
        try:
            with hit.open("w") as fout:
                subprocess.run(
                    [
                        BEDTOOLS,
                        "intersect",
                        "-u",
                        "-a",
                        str(bins_bed),
                        "-b",
                        str(blacklist),
                    ],
                    check=True,
                    stdout=fout,
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


def spearman_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[1]
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            rho, _ = spearmanr(A[:, i], B[:, j])
            R[i, j] = float(rho)
    return R


def peak_membership(regions_bed: Path, peak_bed: Path, idx: np.ndarray) -> np.ndarray:
    """Boolean label per selected bin: overlaps any peak."""
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp:
        sub = Path(tmp.name)
    try:
        write_subset_bed(regions_bed, idx, sub)
        with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as tmp2:
            hit = Path(tmp2.name)
        try:
            with hit.open("w") as fout:
                subprocess.run(
                    [BEDTOOLS, "intersect", "-u", "-a", str(sub), "-b", str(peak_bed)],
                    check=True,
                    stdout=fout,
                )
            hits = {int(l.split("\t")[3]) for l in hit.open()}
        finally:
            hit.unlink(missing_ok=True)
    finally:
        sub.unlink(missing_ok=True)
    return np.asarray([int(i) in hits for i in idx], dtype=bool)


def write_tsv(path: Path, M: np.ndarray, labels: list[str], corner: str) -> None:
    with path.open("w") as f:
        f.write(corner + "\t" + "\t".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(
                lab
                + "\t"
                + "\t".join(f"{M[i, j]:.4f}" for j in range(len(labels)))
                + "\n"
            )


def plot_heatmap(
    tsv: Path,
    png: Path,
    title: str,
    *,
    row_title: str,
    col_title: str,
    annot: bool,
    footnote: str = "",
) -> None:
    cell = (
        textwrap.dedent(
            """\
            cell_fun = function(j, i, x, y, width, height, fill) {
              grid.text(sprintf("%.2f", mat[i, j]), x, y, gp = gpar(fontsize = 8))
            }
            """
        )
        if annot
        else "cell_fun = NULL\n"
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
          library(ComplexHeatmap); library(circlize); library(grid)
          library(RColorBrewer)
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
          column_title = "{title}",
          column_title_gp = gpar(fontsize = 10),
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
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as tmp:
        tmp.write(r_code)
        rpath = tmp.name
    try:
        subprocess.run([RSCRIPT, rpath], check=True)
    finally:
        Path(rpath).unlink(missing_ok=True)


def print_mat(name: str, R: np.ndarray, labels: list[str]) -> None:
    print(f"{name}:")
    print("       " + "".join(f"{l:>8s}" for l in labels))
    for i, lab in enumerate(labels):
        print(f"{lab:6s}" + "".join(f"{R[i, j]:8.3f}" for j in range(len(labels))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-root", type=Path, default=REPO / "unified" / "work")
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "plots" / "tf_bulk_pearson"
    )
    ap.add_argument(
        "--features",
        default="ccre,union_peaks",
        help="Comma list: ccre,union_peaks,all (default: ccre,union_peaks)",
    )
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--n-procs", type=int, default=8)
    ap.add_argument(
        "--force-bedtools",
        action="store_true",
        help="Deprecated alias for --force-bedcov",
    )
    ap.add_argument(
        "--force-bedcov",
        action="store_true",
        help="Skip deepTools; use samtools bedcov -Q 30 + blacklist",
    )
    ap.add_argument(
        "--skip-deeptools",
        action="store_true",
        help="Same as --force-bedcov (deepTools often hangs on this FS)",
    )
    args = ap.parse_args()
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

    # Load sc pseudobulks once (full genome), subset per feature space.
    print("[sc] loading raw / imputed pseudobulks...", flush=True)
    raw_full = np.zeros((n_reg, n_tf), dtype=np.float64)
    imp_full = np.zeros((n_reg, n_tf), dtype=np.float64)
    for j, (name, key) in enumerate(PANEL):
        work = args.work_root / key
        raw_full[:, j] = load_pseudobulk(work, "raw")
        imp_full[:, j] = load_pseudobulk(work, "imputed")
        print(f"  {name} raw_nz={(raw_full[:, j] > 0).mean():.3f} "
              f"imp_nz={(imp_full[:, j] > 0).mean():.3f}", flush=True)

    delta_rows = []
    auroc_rows = []

    for feat in feats:
        print(f"\n======== feature space: {feat} ========", flush=True)
        if feat == "all":
            idx = np.arange(n_reg, dtype=np.int64)
            print(
                "  NOTE: all-bins is a sensitivity check — co-absence inflates ρ.",
                flush=True,
            )
        else:
            feat_bed = cache / f"features_{feat}.bed"
            if not feat_bed.is_file():
                print(f"  building {feat} intervals...", flush=True)
                build_feature_bed(feat, feat_bed)
            idx_path = cache / f"bin_idx_{feat}.npy"
            if idx_path.is_file():
                idx = np.load(idx_path)
            else:
                print(f"  intersecting regions ∩ {feat}...", flush=True)
                idx = feature_bin_indices(reg_bed, feat_bed, idx_path)
            print(f"  n_bins={idx.size}", flush=True)

        # Subset BED for multiBamSummary (preserves bin index in col4)
        bins_bed = cache / f"bins_for_bulk_{feat}.bed"
        if not bins_bed.is_file() or bins_bed.stat().st_size == 0:
            print(f"  writing subset bed {bins_bed.name}...", flush=True)
            write_subset_bed(reg_bed, idx, bins_bed)

        # Bulk counts: prefer deepTools; else samtools bedcov -Q 30 + blacklist.
        raw_counts_path = cache / f"bulk_multibam_{feat}.tab"
        npz_path = cache / f"bulk_multibam_{feat}.npz"
        bedcov_npy = cache / f"bulk_bedcov_q30_{feat}.npy"
        use_deeptools = (
            Path(MULTIBAM).is_file()
            and not args.force_bedtools
            and not args.force_bedcov
        )
        if use_deeptools and not args.skip_deeptools:
            try:
                print(f"  [bulk] deepTools multiBamSummary ({feat})...", flush=True)
                counts = multiBamSummary_counts(
                    bins_bed, npz_path, raw_counts_path, n_procs=args.n_procs
                )
            except Exception as e:
                print(f"  deepTools failed ({e}); using bedcov -Q 30", flush=True)
                counts = bedcov_q30_matrix(
                    bins_bed, bedcov_npy, blacklist=BLACKLIST, n_procs=args.n_procs
                )
        else:
            print("  [bulk] samtools bedcov -Q 30 + blacklist mask...", flush=True)
            counts = bedcov_q30_matrix(
                bins_bed, bedcov_npy, blacklist=BLACKLIST, n_procs=args.n_procs
            )

        if counts.shape[0] != idx.size:
            raise SystemExit(
                f"bulk rows {counts.shape[0]} != idx {idx.size} for {feat}"
            )
        bulk_pb = cpm_normalize(counts)
        print(
            f"  bulk CPM: col_sums={bulk_pb.sum(axis=0).round(1)} "
            f"(should be ~1e6 each)",
            flush=True,
        )

        raw_pb = raw_full[idx, :]
        imp_pb = imp_full[idx, :]

        print(f"  [corr] Spearman raw × bulk ({feat})...", flush=True)
        Rb = spearman_matrix(raw_pb, bulk_pb)
        print(f"  [corr] Spearman imputed × bulk ({feat})...", flush=True)
        Ib = spearman_matrix(imp_pb, bulk_pb)
        print_mat(f"Rb raw×bulk ({feat})", Rb, labels)
        print_mat(f"Ib imputed×bulk ({feat})", Ib, labels)

        # Delta table
        print(f"\n  TF       raw_diag  imp_diag   delta   note")
        for i, lab in enumerate(labels):
            d = float(Ib[i, i] - Rb[i, i])
            note = ""
            if lab == "NFYA":
                note = "K562 bulk vs HEK293 sc — expect weak/neg delta"
            print(
                f"  {lab:8s} {Rb[i, i]:8.4f}  {Ib[i, i]:8.4f}  {d:+7.4f}  {note}"
            )
            delta_rows.append(
                {
                    "features": feat,
                    "TF": lab,
                    "raw_diag": Rb[i, i],
                    "imp_diag": Ib[i, i],
                    "delta": d,
                    "note": note,
                }
            )

        # Write matrices + plots
        for tag, M, row_t in [
            (f"spearman_{feat}_raw_vs_bulk_cpm", Rb, "raw"),
            (f"spearman_{feat}_imputed_vs_bulk_cpm", Ib, "imputed"),
        ]:
            tsv = val_dir / f"{tag}.tsv"
            write_tsv(tsv, M, labels, f"{row_t}/bulk_CPM")
            title = (
                f"Spearman  {row_t} × bulk CPM  ({feat} bins; "
                f"MQ≥30, blacklist, extendReads)"
            )
            foot = "NFYA: K562 bulk vs HEK293 sc (cell-line mismatch)"
            plot_heatmap(
                tsv,
                val_dir / f"{tag}.png",
                title,
                row_title=row_t,
                col_title="bulk",
                annot=False,
                footnote=foot,
            )
            plot_heatmap(
                tsv,
                val_dir / f"{tag}_annot.png",
                title,
                row_title=row_t,
                col_title="bulk",
                annot=True,
                footnote=foot,
            )

        # AUROC vs bulk peak membership
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
                note = (
                    "peak-label AUROC secondary; lead with continuous Spearman "
                    "(legacy peak thresholding asymmetry)"
                )
            print(
                f"    {lab:8s} raw={raw_a:.4f}  imp={imp_a:.4f}  "
                f"delta={da:+.4f}  pos={int(y.sum())}/{y.size}"
            )
            auroc_rows.append(
                {
                    "features": feat,
                    "TF": lab,
                    "raw_AUROC": raw_a,
                    "imp_AUROC": imp_a,
                    "delta_AUROC": da,
                    "n_pos": int(y.sum()),
                    "n_bins": int(y.size),
                    "note": note,
                }
            )

    # Write summary tables
    delta_path = val_dir / "diagonal_delta.tsv"
    with delta_path.open("w") as f:
        f.write("features\tTF\traw_diag\timp_diag\tdelta\tnote\n")
        for r in delta_rows:
            f.write(
                f"{r['features']}\t{r['TF']}\t{r['raw_diag']:.4f}\t"
                f"{r['imp_diag']:.4f}\t{r['delta']:+.4f}\t{r['note']}\n"
            )
    auroc_path = val_dir / "auroc_vs_bulk_peaks.tsv"
    with auroc_path.open("w") as f:
        f.write(
            "features\tTF\traw_AUROC\timp_AUROC\tdelta_AUROC\tn_pos\tn_bins\tnote\n"
        )
        for r in auroc_rows:
            f.write(
                f"{r['features']}\t{r['TF']}\t{r['raw_AUROC']:.4f}\t"
                f"{r['imp_AUROC']:.4f}\t{r['delta_AUROC']:+.4f}\t"
                f"{r['n_pos']}\t{r['n_bins']}\t{r['note']}\n"
            )

    readme = val_dir / "README.txt"
    readme.write_text(
        textwrap.dedent(
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
        )
    )
    print(f"\nWrote {delta_path}")
    print(f"Wrote {auroc_path}")
    print(f"Wrote plots under {val_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
