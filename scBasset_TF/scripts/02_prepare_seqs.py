#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_seqs.py
#
# Same contract as scBasset/02_prepare_seqs.py but with two differences:
#   1. `seqs.seq_length` is configurable (default 768). The architecture in
#      03_train.py is length-agnostic via GlobalAvgPool, so any value works.
#   2. Regions are allowed to be variable-width (peak matrices) -- we always
#      center on (start + end) // 2 and extract a fixed seq_length window
#      around it, regardless of the region's own width.
#
# Per-cell metadata (paths.cell_metadata) is validated against the mm
# barcodes if provided, but is not consumed by the model. The input may be a
# CSV or TSV (delimiter is sniffed) and the `barcode` key column is located by
# name. It is normalised into <work>/seqs/cell_metadata.tsv for downstream
# evaluation only.
#
# Outputs (under <work>/seqs/):
#   seqs.h5
#     X            (n_bins_kept, seq_length, 4)  uint8   one-hot
#     mm_row_idx   (n_bins_kept,)                int64
#     regions      (n_bins_kept,)                S64
#     trainable    (n_bins_kept,)                bool
#     center       (n_bins_kept,)                int64
#     region_width (n_bins_kept,)                int32   end - start of source region
#     attrs: seq_length, n_cells, n_bins_full, genome_fa, input_kind
#   matrix_train.npz   sparse CSR (n_trainable, n_cells)  binary
#   barcodes.tsv       cell barcode order (matches training matrix columns)
#   cell_metadata.tsv  copy of paths.cell_metadata if provided, aligned to barcodes
#   meta.json          counts, filter outcomes, source paths, input_kind
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("02_seqs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")
_BLACKLIST_URL = (
    "https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/"
    "lists/hg38-blacklist.v2.bed.gz"
)

_BASE_LUT = np.full(256, -1, dtype=np.int8)
for ch, idx in zip(b"ACGT", range(4)):
    _BASE_LUT[ch] = idx
    _BASE_LUT[ch + 32] = idx


def read_lines_gz(p: Path) -> list[str]:
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh]


def parse_regions(names: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends   = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        m = _REGION_RE.match(str(name).strip())
        if not m:
            raise SystemExit(f"Bad region name row {i}: {name!r}")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i]   = int(m.group(3))
    return chroms, starts, ends


def default_blacklist_path(work_dir: Path) -> Path:
    wd = work_dir.resolve()
    for anc in [wd, *wd.parents]:
        cand = anc / "downstream" / "cache" / "hg38-blacklist.v2.bed.gz"
        if cand.is_file():
            return cand
    if len(wd.parents) >= 3:
        return wd.parents[2] / "downstream" / "cache" / "hg38-blacklist.v2.bed.gz"
    return wd / "hg38-blacklist.v2.bed.gz"


def ensure_blacklist(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading blacklist %s -> %s", _BLACKLIST_URL, path)
    req = urllib.request.Request(_BLACKLIST_URL, headers={"User-Agent": "scbasset_tf/02"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        path.write_bytes(resp.read())
    return path


def load_intervals(bed_gz: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opener = gzip.open if str(bed_gz).endswith(".gz") else open
    with opener(bed_gz, "rt") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            out[parts[0]].append((int(parts[1]), int(parts[2])))
    return out


def overlap_mask(
    chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    intervals: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    hit = np.zeros(len(chroms), dtype=bool)
    by_chr: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chroms):
        by_chr[str(c)].append(i)
    for chrom, idx_list in by_chr.items():
        iv = intervals.get(chrom) or intervals.get(chrom.replace("chr", "")) or []
        if not iv:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        s_sub, e_sub = starts[idx], ends[idx]
        local = np.zeros(len(idx), dtype=bool)
        for bs, be in iv:
            local |= np.maximum(s_sub, bs) < np.minimum(e_sub, be)
        hit[idx[local]] = True
    return hit


def one_hot_chunk(seq_bytes: np.ndarray) -> np.ndarray:
    """seq_bytes: (B, L) uint8 ASCII -> (B, L, 4) uint8 one-hot."""
    idx = _BASE_LUT[seq_bytes]
    B, L = idx.shape
    out = np.zeros((B, L, 4), dtype=np.uint8)
    mask = idx >= 0
    bb, ll = np.where(mask)
    out[bb, ll, idx[bb, ll]] = 1
    return out


def load_and_align_metadata(
    meta_path: Path, barcodes: np.ndarray
) -> tuple[list[str], list[list[str]]] | None:
    """Read the per-cell metadata table and reindex it to the trained matrix's
    barcode order.

    The delimiter is sniffed from the header (tab or comma), so both a
    barcode-first TSV and the AllTF `*.meta.csv` (columns
    DNA_id,cell,TF,barcode,subset) are accepted. The key column is whichever
    column is literally named "barcode"; if none exists, the first column is
    used. Every remaining column is carried through as a value column, so the
    barcode does not have to be first.

    Return (value_column_names, rows_in_barcode_order) or None."""
    with open(meta_path) as fh:
        first = fh.readline().rstrip("\r\n")
        if not first:
            log.warning("cell_metadata empty: %s", meta_path)
            return None
        sep = "\t" if "\t" in first else ("," if "," in first else "\t")
        header = first.split(sep)
        if len(header) < 2:
            raise SystemExit(
                f"cell_metadata must have >= 2 columns (barcode + ...). got {first!r}")
        bc_idx = header.index("barcode") if "barcode" in header else 0
        rest_cols = [c for i, c in enumerate(header) if i != bc_idx]
        log.info("cell_metadata sep=%r, barcode col=%r (idx %d), value cols=%r",
                 sep, header[bc_idx], bc_idx, rest_cols)
        rows: dict[str, list[str]] = {}
        for ln in fh:
            parts = ln.rstrip("\r\n").split(sep)
            if len(parts) <= bc_idx or not parts[bc_idx]:
                continue
            rows[parts[bc_idx]] = [
                parts[i] if i < len(parts) else "" for i in range(len(header)) if i != bc_idx
            ]
    aligned: list[list[str]] = []
    missing = 0
    for bc in barcodes.tolist():
        if bc in rows:
            aligned.append(rows[bc])
        else:
            aligned.append([""] * len(rest_cols))
            missing += 1
    log.info("cell_metadata aligned %d / %d barcodes (missing %d)",
             len(barcodes) - missing, len(barcodes), missing)
    return rest_cols, aligned


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--genome-fa", type=str, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    seqs_cfg = cfg.setdefault("seqs", {})
    seq_length = int(seqs_cfg.get("seq_length", 768))
    max_n_frac = float(seqs_cfg.get("max_n_frac", 0.10))
    chrom_whitelist = set(seqs_cfg.get("chrom_whitelist") or [])
    min_cells_train = int(seqs_cfg.get("min_cells_per_peak_train", 2))
    do_blacklist = bool(seqs_cfg.get("exclude_blacklist", True))
    bin_bed = seqs_cfg.get("bin_bed") or None
    input_kind = str(cfg.get("run", {}).get("input_kind", "bin"))
    if seq_length <= 0:
        log.error("seqs.seq_length must be > 0; got %d", seq_length)
        return 1

    genome_fa = args.genome_fa or cfg["paths"].get("genome_fa")
    if not genome_fa:
        log.error("paths.genome_fa is unset and --genome-fa not provided.")
        return 1
    genome_fa = Path(genome_fa).expanduser().resolve()
    fai = Path(str(genome_fa) + ".fai")
    if not genome_fa.exists():
        log.error("Genome FASTA missing: %s", genome_fa)
        return 1
    if not fai.exists():
        log.error("FASTA index missing (%s). Run: samtools faidx %s", fai, genome_fa)
        return 1

    try:
        from pyfaidx import Fasta
    except ImportError:
        log.error("pyfaidx not installed; pip install pyfaidx")
        return 1

    mm_dir = Path(cfg["paths"]["mm"])
    log.info("Loading mm matrix from %s (input_kind=%s, seq_length=%d)",
             mm_dir, input_kind, seq_length)
    mat = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr().astype(np.int32)
    regions  = np.asarray(read_lines_gz(mm_dir / "regions.tsv.gz"))
    barcodes = np.asarray(read_lines_gz(mm_dir / "barcodes.tsv.gz"))
    assert mat.shape == (len(regions), len(barcodes))
    log.info("Raw matrix: %d regions x %d cells, nnz=%d", *mat.shape, mat.nnz)

    chroms, starts, ends = parse_regions(regions)
    widths = (ends - starts).astype(np.int32)
    log.info("Region widths: min=%d  median=%d  max=%d (input_kind=%s)",
             int(widths.min()), int(np.median(widths)), int(widths.max()), input_kind)
    keep = np.ones(len(regions), dtype=bool)

    if chrom_whitelist:
        keep &= np.isin(chroms, list(chrom_whitelist))
        log.info("Chrom whitelist: %d / %d regions remain", int(keep.sum()), len(keep))

    if do_blacklist:
        bl_path = ensure_blacklist(default_blacklist_path(Path(cfg["paths"]["work_dir"])))
        bl_iv = load_intervals(bl_path)
        bl_hit = overlap_mask(chroms, starts, ends, bl_iv)
        keep &= ~bl_hit
        log.info("Blacklist: dropped %d more -> %d remain",
                 int(bl_hit.sum()), int(keep.sum()))

    if bin_bed:
        bed_iv = load_intervals(Path(bin_bed).expanduser().resolve())
        bed_hit = overlap_mask(chroms, starts, ends, bed_iv)
        keep &= bed_hit
        log.info("bin_bed restriction: %d remain", int(keep.sum()))

    fa = Fasta(str(genome_fa), as_raw=True, sequence_always_upper=False)
    chrom_lens = {name: len(fa[name]) for name in fa.keys()}
    centers = (starts + ends) // 2
    half = seq_length // 2
    win_start = centers - half
    win_end   = win_start + seq_length

    in_bounds = np.zeros(len(regions), dtype=bool)
    for i in np.where(keep)[0]:
        L = chrom_lens.get(str(chroms[i]), 0)
        if 0 <= win_start[i] and win_end[i] <= L:
            in_bounds[i] = True
    drop_oob = int(keep.sum() - in_bounds.sum())
    keep &= in_bounds
    log.info("In-bounds: dropped %d off-the-end -> %d remain",
             drop_oob, int(keep.sum()))

    kept_idx = np.where(keep)[0]
    n_kept = len(kept_idx)
    log.info("Extracting %d sequences of length %d", n_kept, seq_length)

    accept = np.zeros(n_kept, dtype=bool)
    max_n_count = int(max_n_frac * seq_length)

    CHUNK = 4096
    log.info("Pass 1/2: pre-scan for N content")
    for c0 in range(0, n_kept, CHUNK):
        sel = kept_idx[c0:c0 + CHUNK]
        for k, i in enumerate(sel):
            seq = fa.get_seq(str(chroms[i]), int(win_start[i]) + 1, int(win_end[i]))
            s = str(seq).upper()
            if s.count("N") <= max_n_count:
                accept[c0 + k] = True
    n_accept = int(accept.sum())
    log.info("Pass 1 done: %d / %d survive N filter (max_n_frac=%.2f)",
             n_accept, n_kept, max_n_frac)
    if n_accept == 0:
        log.error("No regions survived; aborting.")
        return 2

    final_idx        = kept_idx[accept]
    final_chroms     = chroms[final_idx]
    final_centers    = centers[final_idx]
    final_win_starts = win_start[final_idx]
    final_win_ends   = win_end[final_idx]
    final_widths     = widths[final_idx]

    seqs_dir = Path(cfg["paths"]["seqs"])
    h5_path = seqs_dir / "seqs.h5"
    log.info("Pass 2/2: extract + one-hot -> %s", h5_path)
    with h5py.File(h5_path, "w") as h5:
        X = h5.create_dataset(
            "X", shape=(n_accept, seq_length, 4),
            dtype="uint8", chunks=(min(256, n_accept), seq_length, 4),
            compression="gzip", compression_opts=4,
        )
        for c0 in range(0, n_accept, CHUNK):
            c1 = min(n_accept, c0 + CHUNK)
            buf = np.zeros((c1 - c0, seq_length), dtype=np.uint8)
            for k in range(c1 - c0):
                i_full = int(final_idx[c0 + k])
                seq = fa.get_seq(
                    str(chroms[i_full]),
                    int(win_start[i_full]) + 1,
                    int(win_end[i_full]),
                )
                buf[k] = np.frombuffer(str(seq).encode("ascii"), dtype=np.uint8)
            X[c0:c1] = one_hot_chunk(buf)
            if c0 % (CHUNK * 16) == 0:
                log.info("  ... %d / %d", c1, n_accept)

        h5.create_dataset("mm_row_idx",   data=final_idx.astype(np.int64))
        h5.create_dataset("regions",      data=regions[final_idx].astype("S64"))
        h5.create_dataset("center",       data=final_centers.astype(np.int64))
        h5.create_dataset("region_width", data=final_widths.astype(np.int32))
        per_bin_cells = np.asarray((mat[final_idx] > 0).sum(axis=1)).ravel()
        trainable = per_bin_cells >= min_cells_train
        h5.create_dataset("trainable", data=trainable)
        h5.attrs["seq_length"]    = seq_length
        h5.attrs["genome_fa"]     = str(genome_fa)
        h5.attrs["n_cells"]       = int(mat.shape[1])
        h5.attrs["n_bins_full"]   = int(mat.shape[0])
        h5.attrs["input_kind"]    = input_kind

    log.info("Trainable regions (cells >= %d): %d / %d",
             min_cells_train, int(trainable.sum()), n_accept)

    train_mat = (mat[final_idx[trainable]] > 0).astype(np.uint8).tocsr()
    sp.save_npz(seqs_dir / "matrix_train.npz", train_mat)
    (seqs_dir / "barcodes.tsv").write_text("\n".join(barcodes.tolist()) + "\n")
    np.save(seqs_dir / "trainable_idx_in_h5.npy", np.where(trainable)[0].astype(np.int64))

    # Sparsity stats for the loss-side sparsity attack in 03_train.py.
    n_entries = int(train_mat.shape[0]) * int(train_mat.shape[1])
    n_pos = int(train_mat.nnz)
    pos_freq = (n_pos / n_entries) if n_entries > 0 else 0.0
    if pos_freq > 0:
        pos_weight_auto = (1.0 - pos_freq) / pos_freq
    else:
        pos_weight_auto = None
    log.info("Sparsity stats on matrix_train: pos_freq=%.5f  (1 entry per ~%d)  "
             "pos_weight_auto=%s",
             pos_freq, int(1.0 / pos_freq) if pos_freq > 0 else -1,
             f"{pos_weight_auto:.2f}" if pos_weight_auto is not None else "n/a")

    # Optional per-cell metadata pass-through (eval only)
    metadata_path = cfg["paths"].get("cell_metadata")
    metadata_meta = None
    if metadata_path:
        mp = Path(metadata_path).expanduser().resolve()
        if not mp.is_file():
            log.error("cell_metadata not found: %s", mp)
            return 3
        aligned = load_and_align_metadata(mp, barcodes)
        if aligned is not None:
            cols, rows = aligned
            out_md = seqs_dir / "cell_metadata.tsv"
            with open(out_md, "w") as fh:
                fh.write("\t".join(["barcode", *cols]) + "\n")
                for bc, row in zip(barcodes.tolist(), rows):
                    fh.write("\t".join([bc, *row]) + "\n")
            metadata_meta = {"source": str(mp), "columns": cols, "out": str(out_md)}
            log.info("Wrote aligned cell metadata -> %s", out_md)

    meta = {
        "input_kind":         input_kind,
        "n_regions_in":       int(mat.shape[0]),
        "n_cells":            int(mat.shape[1]),
        "n_bins_kept_in_h5":  int(n_accept),
        "n_bins_trainable":   int(trainable.sum()),
        "seq_length":         seq_length,
        "max_n_frac":         max_n_frac,
        "exclude_blacklist":  do_blacklist,
        "bin_bed":            str(bin_bed) if bin_bed else None,
        "genome_fa":          str(genome_fa),
        "h5":                 str(h5_path),
        "matrix_train_npz":   str(seqs_dir / "matrix_train.npz"),
        "region_width_stats": {
            "min":    int(final_widths.min()),
            "median": int(np.median(final_widths)),
            "max":    int(final_widths.max()),
        },
        "pos_freq":           float(pos_freq),
        "pos_weight_auto":    float(pos_weight_auto) if pos_weight_auto is not None else None,
        "n_train_positives":  n_pos,
        "n_train_entries":    n_entries,
        "cell_metadata":      metadata_meta,
    }
    (seqs_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", {k: meta[k] for k in
                          ("input_kind", "n_bins_kept_in_h5", "n_bins_trainable",
                           "seq_length", "region_width_stats")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
