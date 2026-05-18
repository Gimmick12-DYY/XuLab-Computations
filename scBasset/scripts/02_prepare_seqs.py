#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_seqs.py
#
# For every bin in mm/regions.tsv.gz, extract a `seqs.seq_length`-bp centered
# sequence from the hg38 FASTA, one-hot encode it, and save everything that
# 03_train.py / 04_predict.py need into <work>/seqs/:
#
#   seqs.h5
#     X            (n_bins_kept, seq_length, 4)  uint8   one-hot
#     mm_row_idx   (n_bins_kept,)                int64   row indices into mm/regions
#     regions      (n_bins_kept,)                S64     bin names (chr:start-end)
#     trainable    (n_bins_kept,)                bool    bin has >= min_cells_per_peak_train
#     center       (n_bins_kept,)                int64   center base position (0-based)
#   matrix_train.npz   sparse CSR (n_trainable_bins, n_cells)  binary
#   barcodes.tsv       n_cells barcodes (same order as the trained matrix columns)
#   meta.json          counts, filter outcomes, source paths
#
# Bins are dropped when (a) chromosome is not in the whitelist, (b) the
# centered window runs off the end of the chromosome, (c) > max_n_frac of the
# bases are N, or (d) they overlap the ENCODE hg38 blacklist (if enabled).
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

# A->0 C->1 G->2 T->3, everything else -> -1 (becomes all-zero one-hot)
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for ch, idx in zip(b"ACGT", range(4)):
    _BASE_LUT[ch] = idx
    _BASE_LUT[ch + 32] = idx  # lowercase soft-mask


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
    req = urllib.request.Request(_BLACKLIST_URL, headers={"User-Agent": "scbasset/02"})
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
    """True at rows whose [start, end) overlaps any interval on the same chrom."""
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
    """seq_bytes: (B, L) uint8 ASCII -> (B, L, 4) uint8 one-hot. N/other -> all 0."""
    idx = _BASE_LUT[seq_bytes]              # (B, L) int8 in {-1, 0..3}
    B, L = idx.shape
    out = np.zeros((B, L, 4), dtype=np.uint8)
    mask = idx >= 0
    bb, ll = np.where(mask)
    out[bb, ll, idx[bb, ll]] = 1
    return out


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--genome-fa", type=str, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    seqs_cfg = cfg.setdefault("seqs", {})
    seq_length = int(seqs_cfg.get("seq_length", 1344))
    max_n_frac = float(seqs_cfg.get("max_n_frac", 0.10))
    chrom_whitelist = set(seqs_cfg.get("chrom_whitelist") or [])
    min_cells_train = int(seqs_cfg.get("min_cells_per_peak_train", 2))
    do_blacklist = bool(seqs_cfg.get("exclude_blacklist", True))
    bin_bed = seqs_cfg.get("bin_bed") or None

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
    log.info("Loading mm matrix from %s", mm_dir)
    mat = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr().astype(np.int32)
    regions  = np.asarray(read_lines_gz(mm_dir / "regions.tsv.gz"))
    barcodes = np.asarray(read_lines_gz(mm_dir / "barcodes.tsv.gz"))
    assert mat.shape == (len(regions), len(barcodes))
    log.info("Raw matrix: %d regions x %d cells, nnz=%d", *mat.shape, mat.nnz)

    chroms, starts, ends = parse_regions(regions)
    keep = np.ones(len(regions), dtype=bool)

    # 1) chromosome whitelist
    if chrom_whitelist:
        keep &= np.isin(chroms, list(chrom_whitelist))
        log.info("Chrom whitelist: %d / %d bins remain", int(keep.sum()), len(keep))

    # 2) blacklist
    if do_blacklist:
        bl_path = ensure_blacklist(default_blacklist_path(Path(cfg["paths"]["work_dir"])))
        bl_iv = load_intervals(bl_path)
        bl_hit = overlap_mask(chroms, starts, ends, bl_iv)
        keep &= ~bl_hit
        log.info("Blacklist: dropped %d more bins -> %d remain",
                 int(bl_hit.sum()), int(keep.sum()))

    # 3) optional bin_bed restriction
    if bin_bed:
        bed_iv = load_intervals(Path(bin_bed).expanduser().resolve())
        bed_hit = overlap_mask(chroms, starts, ends, bed_iv)
        keep &= bed_hit
        log.info("bin_bed restriction: %d bins remain", int(keep.sum()))

    # 4) window-fits-in-chromosome check
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
    log.info("In-bounds: dropped %d off-the-end bins -> %d remain",
             drop_oob, int(keep.sum()))

    # 5) extract sequences chunk by chunk; also drop high-N bins
    kept_idx = np.where(keep)[0]
    n_kept = len(kept_idx)
    log.info("Extracting %d sequences of length %d from %s", n_kept, seq_length, genome_fa)

    seqs_dir = Path(cfg["paths"]["seqs"])
    h5_path = seqs_dir / "seqs.h5"
    # Two-pass: first extract+filter on N, write to a temp list of accepted rows.
    accept = np.zeros(n_kept, dtype=bool)
    max_n_count = int(max_n_frac * seq_length)

    CHUNK = 4096
    # We need to know final n_accepted before allocating the h5 dataset cleanly.
    # Pass 1: count and remember per-bin reject reason.
    log.info("Pass 1/2: pre-scan for N content")
    for c0 in range(0, n_kept, CHUNK):
        sel = kept_idx[c0:c0 + CHUNK]
        for k, i in enumerate(sel):
            seq = fa.get_seq(str(chroms[i]), int(win_start[i]) + 1, int(win_end[i]))
            # pyfaidx 1-based inclusive coordinates; sequence_always_upper=False
            # keeps soft-mask but we collapse case below.
            s = str(seq).upper()
            n_count = s.count("N")
            if n_count <= max_n_count:
                accept[c0 + k] = True
    n_accept = int(accept.sum())
    log.info("Pass 1 done: %d / %d bins survive N filter (max_n_frac=%.2f)",
             n_accept, n_kept, max_n_frac)
    if n_accept == 0:
        log.error("No bins survived; aborting.")
        return 2

    final_idx = kept_idx[accept]
    final_chroms = chroms[final_idx]
    final_starts = starts[final_idx]
    final_ends   = ends[final_idx]
    final_centers = centers[final_idx]
    final_win_starts = win_start[final_idx]
    final_win_ends   = win_end[final_idx]

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

        h5.create_dataset("mm_row_idx", data=final_idx.astype(np.int64))
        h5.create_dataset("regions",    data=regions[final_idx].astype("S64"))
        h5.create_dataset("center",     data=final_centers.astype(np.int64))
        # trainable mask = enough cells observed at this bin (in the raw matrix)
        per_bin_cells = np.asarray((mat[final_idx] > 0).sum(axis=1)).ravel()
        trainable = per_bin_cells >= min_cells_train
        h5.create_dataset("trainable",  data=trainable)
        h5.attrs["seq_length"]    = seq_length
        h5.attrs["genome_fa"]     = str(genome_fa)
        h5.attrs["n_cells"]       = int(mat.shape[1])
        h5.attrs["n_bins_full"]   = int(mat.shape[0])

    log.info("Trainable bins (cells >= %d): %d / %d",
             min_cells_train, int(trainable.sum()), n_accept)

    # Save the training subset of the binary matrix.
    train_mat = (mat[final_idx[trainable]] > 0).astype(np.uint8).tocsr()
    sp.save_npz(seqs_dir / "matrix_train.npz", train_mat)
    (seqs_dir / "barcodes.tsv").write_text("\n".join(barcodes.tolist()) + "\n")
    np.save(seqs_dir / "trainable_idx_in_h5.npy", np.where(trainable)[0].astype(np.int64))

    meta = {
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
    }
    (seqs_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
