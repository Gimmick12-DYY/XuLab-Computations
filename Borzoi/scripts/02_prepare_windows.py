#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_windows.py
#
# Plan Borzoi inference windows for every mm bin.
#
# Borzoi has a fixed 524288 bp input and ~196608 bp central output. To cover
# the genome we tile each chromosome at stride = central_output_bp. For every
# kept mm bin we record which window covers it and the bin's offset (in 32 bp
# units) inside that window's central output. The actual sequence fetch +
# inference happens in 03_borzoi_predict.py.
#
# Filters (same vocabulary as scBasset/02_prepare_seqs.py):
#   - chrom whitelist
#   - ENCODE hg38 blacklist (overlap)
#   - in-bounds (with optional N-pad at chromosome ends)
#
# Outputs (under <work>/windows/):
#   windows.h5
#     /windows/chrom        (n_windows,) S16
#     /windows/input_start  (n_windows,) int64   genomic coord; may be < 0
#                                                or > chrom_len when N-padded
#     /windows/central_start (n_windows,) int64
#     /bins/mm_row_idx      (n_kept,)    int64   row in mm/regions
#     /bins/window_id       (n_kept,)    int32
#     /bins/out_pos_start   (n_kept,)    int32   0-based in 32-bp output units
#     /bins/out_pos_end     (n_kept,)    int32
#     attrs: input_length, central_output_bp, output_bin_bp, n_output_positions
#   meta.json   counts, filter outcomes, source paths
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

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger("02_windows")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REGION_RE = re.compile(r"^([^:]+):(\d+)-(\d+)$")
_BLACKLIST_URL = (
    "https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/"
    "lists/hg38-blacklist.v2.bed.gz"
)


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
    req = urllib.request.Request(_BLACKLIST_URL, headers={"User-Agent": "borzoi/02"})
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


def chrom_lengths(genome_fa: Path) -> dict[str, int]:
    fai = Path(str(genome_fa) + ".fai")
    if not fai.exists():
        raise SystemExit(f"FASTA index missing ({fai}). Run: samtools faidx {genome_fa}")
    out: dict[str, int] = {}
    with open(fai) as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--genome-fa", type=str, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    w_cfg = cfg.setdefault("windows", {})
    L_in    = int(w_cfg.get("input_length",      524288))
    L_out   = int(w_cfg.get("central_output_bp", 196608))
    bin_bp  = int(w_cfg.get("output_bin_bp",     32))
    chrom_whitelist = set(w_cfg.get("chrom_whitelist") or [])
    do_blacklist = bool(w_cfg.get("exclude_blacklist", True))
    npad_edges   = bool(w_cfg.get("npad_edges", True))

    if (L_in - L_out) % 2 != 0:
        raise SystemExit("input_length - central_output_bp must be even")
    if L_out % bin_bp != 0:
        raise SystemExit("central_output_bp must be divisible by output_bin_bp")
    n_out_positions = L_out // bin_bp
    flank = (L_in - L_out) // 2

    genome_fa = args.genome_fa or cfg["paths"].get("genome_fa")
    if not genome_fa:
        log.error("paths.genome_fa is unset and --genome-fa not provided.")
        return 1
    genome_fa = Path(genome_fa).expanduser().resolve()
    chrom_len = chrom_lengths(genome_fa)
    log.info("Loaded chrom lengths for %d sequences from %s.fai", len(chrom_len), genome_fa)

    mm_dir = Path(cfg["paths"]["mm"])
    regions = np.asarray(read_lines_gz(mm_dir / "regions.tsv.gz"))
    log.info("Loaded %d mm regions", len(regions))
    chroms, starts, ends = parse_regions(regions)
    keep = np.ones(len(regions), dtype=bool)

    # 1) chrom whitelist
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

    # 3) chrom present in FASTA
    has_chr = np.array([str(c) in chrom_len for c in chroms])
    keep &= has_chr
    log.info("FASTA chrom check: %d remain", int(keep.sum()))

    # 4) in-bounds (unless we pad edges with N at inference time)
    if not npad_edges:
        in_bounds = np.zeros(len(regions), dtype=bool)
        for i in np.where(keep)[0]:
            L_chr = chrom_len[str(chroms[i])]
            # central output position required to fully contain the bin
            mid = (starts[i] + ends[i]) // 2
            win_input_start = mid - mid % L_out - flank  # nearest tile start - flank
            win_input_end   = win_input_start + L_in
            if 0 <= win_input_start and win_input_end <= L_chr:
                in_bounds[i] = True
        keep &= in_bounds
        log.info("In-bounds: %d remain (npad_edges=false)", int(keep.sum()))

    kept_idx = np.where(keep)[0]
    n_kept = len(kept_idx)
    if n_kept == 0:
        log.error("No bins remain; aborting.")
        return 2

    # ---- assign each kept bin to its window ----------------------------------
    # Tile each chromosome with non-overlapping central outputs of length L_out.
    # Window i on chrom c has central span [i*L_out, (i+1)*L_out) and input
    # span [i*L_out - flank, i*L_out - flank + L_in).
    # A bin [bs, be) lives in window i = floor(bs / L_out) iff be <= (i+1)*L_out
    # i.e. iff the bin does not straddle a tile boundary. 1 kb bins on a 196 kb
    # grid almost always satisfy this; the rare straddlers are dropped.
    bin_starts = starts[kept_idx]
    bin_ends   = ends[kept_idx]
    bin_chroms = chroms[kept_idx]
    win_local_id = bin_starts // L_out
    next_boundary = (win_local_id + 1) * L_out
    straddle = bin_ends > next_boundary
    if straddle.any():
        log.info("Dropping %d bins that straddle a window boundary", int(straddle.sum()))
        keep_b = ~straddle
        kept_idx = kept_idx[keep_b]
        bin_starts = bin_starts[keep_b]
        bin_ends   = bin_ends[keep_b]
        bin_chroms = bin_chroms[keep_b]
        win_local_id = win_local_id[keep_b]
        n_kept = len(kept_idx)

    # Build the unique (chrom, win_local_id) -> global window_id map.
    chrom_wid = np.array(
        [f"{c}#{int(w)}" for c, w in zip(bin_chroms, win_local_id)],
        dtype=object,
    )
    uniq, inverse = np.unique(chrom_wid, return_inverse=True)
    n_windows = len(uniq)
    log.info("Unique inference windows: %d (covering %d bins)", n_windows, n_kept)

    win_chrom = np.empty(n_windows, dtype=object)
    win_input_start = np.zeros(n_windows, dtype=np.int64)
    win_central_start = np.zeros(n_windows, dtype=np.int64)
    for k, key in enumerate(uniq):
        chrom, wid_s = key.split("#")
        wid = int(wid_s)
        win_chrom[k] = chrom
        win_central_start[k] = wid * L_out
        win_input_start[k]   = wid * L_out - flank

    # Per-bin output position offsets (in 32 bp units) within window's central
    # output. central_start of window k = win_central_start[k].
    bin_central_start = win_central_start[inverse]
    out_pos_start = ((bin_starts - bin_central_start) // bin_bp).astype(np.int32)
    out_pos_end   = ((bin_ends   - bin_central_start + bin_bp - 1) // bin_bp).astype(np.int32)
    # Clip just in case of off-by-one at the high end.
    np.clip(out_pos_end, 0, n_out_positions, out=out_pos_end)
    np.clip(out_pos_start, 0, n_out_positions, out=out_pos_start)

    out_dir = Path(cfg["paths"]["windows"])
    h5_path = out_dir / "windows.h5"
    log.info("Writing %s", h5_path)
    with h5py.File(h5_path, "w") as h5:
        wg = h5.create_group("windows")
        wg.create_dataset("chrom",         data=np.asarray(win_chrom, dtype="S16"))
        wg.create_dataset("input_start",   data=win_input_start.astype(np.int64))
        wg.create_dataset("central_start", data=win_central_start.astype(np.int64))

        bg = h5.create_group("bins")
        bg.create_dataset("mm_row_idx",    data=kept_idx.astype(np.int64))
        bg.create_dataset("window_id",     data=inverse.astype(np.int32))
        bg.create_dataset("out_pos_start", data=out_pos_start)
        bg.create_dataset("out_pos_end",   data=out_pos_end)

        h5.attrs["input_length"]       = L_in
        h5.attrs["central_output_bp"]  = L_out
        h5.attrs["output_bin_bp"]      = bin_bp
        h5.attrs["n_output_positions"] = n_out_positions
        h5.attrs["flank"]              = flank
        h5.attrs["genome_fa"]          = str(genome_fa)
        h5.attrs["npad_edges"]         = npad_edges

    meta = {
        "n_regions_in":      int(len(regions)),
        "n_kept_bins":       int(n_kept),
        "n_windows":         int(n_windows),
        "input_length":      L_in,
        "central_output_bp": L_out,
        "output_bin_bp":     bin_bp,
        "n_output_positions": n_out_positions,
        "flank":             flank,
        "genome_fa":         str(genome_fa),
        "windows_h5":        str(h5_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
