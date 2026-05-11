#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 02_prepare_scopen_input.py
#
# Convert the Matrix Market output of 01_export_rds_to_mm.R into the 10X-format
# directory scOpen ingests:
#
#   <work>/scopen_input/
#     matrix.mtx          un-gzipped, integer-valued, regions x cells
#     barcodes.tsv        one barcode per line
#     peaks.bed           tab-separated chrom\tstart\tend
#     regions.tsv         original chr:start-end strings, in row order
#
# scOpen's `--input_format 10X` loader reads the mtx via scipy.io.mmread and
# rebuilds peak IDs as `<chrom>:<start>-<end>`, so we keep the BED columns in
# sync with our region naming convention.
#
# scOpen was designed for *peak* matrices where every row is a candidate
# open-chromatin site. Genome-wide 1 kb bins violate that assumption -- ~95 %
# of the genome is truly inaccessible and scOpen's NMF will happily smear
# small TF-IDF signals into those rows ("over-imputation"). Two mechanisms in
# this step mitigate the problem:
#
#   * `--candidate-bed PATH` (or `filter.candidate_regions_bed`): restrict the
#     input matrix to bins overlapping the user-supplied BED. Bins outside
#     the BED are dropped before scOpen sees them, so they never receive a
#     positive imputed value. Recommended for genome-wide bin data.
#   * `--candidate-invert` (or `filter.candidate_invert`): treat the BED as
#     a blacklist instead of a whitelist (drop bins that DO overlap).
#
# We also binarise (matching the scOpen training regime) and drop all-zero
# rows that cannot contribute to NMF.
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('02_prepare')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def _write_lines(lines: list[str], path: Path) -> None:
    path.write_text('\n'.join(lines) + '\n')


_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _region_to_bed_row(name: str) -> tuple[str, str, str]:
    m = _REGION_RE.match(name)
    if not m:
        raise SystemExit(f'Cannot parse region {name!r} as chr:start-end. '
                         f'Re-run 01_export_rds_to_mm.R or pass plain coords.')
    return m.group(1), m.group(2), m.group(3)


def _parse_bed(path: Path) -> list[tuple[str, int, int]]:
    """Minimal BED parser: skip comments / track / browser lines, take cols 1-3."""
    out: list[tuple[str, int, int]] = []
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as fh:
        for ln in fh:
            if not ln or ln.startswith(('#', 'track', 'browser')):
                continue
            parts = ln.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            try:
                out.append((parts[0], int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return out


def _bin_overlap_mask(region_names: np.ndarray,
                      bed: list[tuple[str, int, int]]) -> np.ndarray:
    """Return bool mask over `region_names` (chr:start-end strings) marking
    bins that overlap any BED interval.

    Per-chromosome bucket + sweep:
        for each bin's [bs, be), find BED intervals on the same chrom; the
        first interval whose end > bs is the earliest possible overlap; iterate
        forward until interval start >= be.
    Runtime is O(N + M + overlaps); fine at 3 M bins and 100 k BED entries.
    """
    by_chrom: dict[str, list[tuple[int, int]]] = {}
    for c, s, e in bed:
        by_chrom.setdefault(c, []).append((s, e))
    # Sort intervals within each chromosome by start.
    for c in by_chrom:
        by_chrom[c].sort()
    # Pre-extract starts as a numpy array per chromosome for bisect.
    chrom_starts: dict[str, np.ndarray] = {
        c: np.fromiter((s for s, _ in ivs), dtype=np.int64, count=len(ivs))
        for c, ivs in by_chrom.items()
    }

    mask = np.zeros(len(region_names), dtype=bool)
    for i, name in enumerate(region_names):
        try:
            chrom, span = name.split(':')
            bs_s, be_s = span.split('-')
            bs_i, be_i = int(bs_s), int(be_s)
        except ValueError:
            continue
        ivs = by_chrom.get(chrom)
        if not ivs:
            continue
        starts = chrom_starts[chrom]
        # First candidate: leftmost interval whose start < be_i.
        j = int(np.searchsorted(starts, be_i, side='left'))
        # Walk backward from j-1 while interval end > bs_i; an overlap means
        # interval_start < be_i AND interval_end > bs_i.
        k = j - 1
        while k >= 0:
            s, e = ivs[k]
            if e <= bs_i:
                break
            if s < be_i:
                mask[i] = True
                break
            k -= 1
    return mask


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--no-binarize', action='store_true')
    ap.add_argument('--drop-zero-rows', action='store_true', default=True,
                    help='Drop regions with zero counts across all cells. '
                         'These are uninformative and waste memory in scOpen NMF. '
                         'Default: True.')
    ap.add_argument('--no-drop-zero-rows', dest='drop_zero_rows',
                    action='store_false')
    ap.add_argument('--candidate-bed', type=str, default=None,
                    help='Path to a BED file of candidate (potentially-positive) '
                         'regions. Bins not overlapping any interval are dropped '
                         'before scOpen sees them. Overrides '
                         'filter.candidate_regions_bed.')
    ap.add_argument('--candidate-invert', action='store_true', default=False,
                    help='Treat the candidate BED as a *blacklist* (drop bins '
                         'that DO overlap any interval). Overrides '
                         'filter.candidate_invert.')
    ap.add_argument('--no-candidate-bed', action='store_true',
                    help='Force-disable the candidate-region filter even if '
                         'the config specifies one.')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    flt = cfg.setdefault('filter', {})

    if args.no_binarize:
        flt['binarize_input'] = False
    binarize = bool(flt.get('binarize_input', True))

    # CLI overrides for the candidate BED knobs.
    if args.candidate_bed is not None:
        flt['candidate_regions_bed'] = args.candidate_bed
    if args.candidate_invert:
        flt['candidate_invert'] = True
    if args.no_candidate_bed:
        flt['candidate_regions_bed'] = None
    candidate_bed = flt.get('candidate_regions_bed', None)
    candidate_invert = bool(flt.get('candidate_invert', False))

    mm_dir = Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error('Missing mm input: %s. Run 01_export_rds_to_mm.R first.', p)
            return 1

    log.info('Reading %s', mtx_path)
    mat = sio.mmread(mtx_path).tocsr()
    regions = np.asarray(_read_lines_gz(reg_path))
    barcodes = np.asarray(_read_lines_gz(bar_path))
    if mat.shape != (len(regions), len(barcodes)):
        log.error('Shape mismatch: matrix %s vs regions %d / barcodes %d',
                  mat.shape, len(regions), len(barcodes))
        return 1
    log.info('Loaded %d regions x %d cells, nnz=%d (density=%.3g)',
             *mat.shape, mat.nnz, mat.nnz / float(mat.shape[0] * mat.shape[1]))

    n_regions_input = int(mat.shape[0])

    # -- binarise -------------------------------------------------------------
    if binarize:
        log.info('Binarising matrix (count > 0 -> 1).')
        mat = mat.astype(bool).astype(np.int8)

    # -- candidate-region mask -----------------------------------------------
    candidate_meta: dict[str, object] = {
        'candidate_regions_bed': candidate_bed,
        'candidate_invert':      candidate_invert,
        'n_bed_intervals':       0,
        'n_regions_overlap':     0,
        'n_regions_kept':        n_regions_input,
    }
    if candidate_bed:
        bed_path = Path(candidate_bed)
        if not bed_path.exists():
            log.error('candidate_regions_bed not found: %s', bed_path)
            return 1
        log.info('Reading candidate BED %s', bed_path)
        bed = _parse_bed(bed_path)
        log.info('  %d BED intervals across %d chromosomes',
                 len(bed), len({c for c, _, _ in bed}))
        log.info('Computing bin overlap mask ...')
        overlap = _bin_overlap_mask(regions, bed)
        n_overlap = int(overlap.sum())
        log.info('  %d / %d bins overlap the BED (%.3f%%)',
                 n_overlap, len(regions),
                 100.0 * n_overlap / max(len(regions), 1))
        keep = (~overlap) if candidate_invert else overlap
        n_kept = int(keep.sum())
        action = 'blacklist (drop overlapping)' if candidate_invert else 'whitelist (keep overlapping)'
        log.info('Candidate filter (%s): keeping %d / %d bins (%.3f%%)',
                 action, n_kept, len(regions),
                 100.0 * n_kept / max(len(regions), 1))
        if n_kept == 0:
            log.error('Candidate filter removed all bins; check the BED file '
                      'or invert flag.')
            return 1
        mat = mat[keep]
        regions = regions[keep]
        candidate_meta.update({
            'n_bed_intervals':   int(len(bed)),
            'n_regions_overlap': n_overlap,
            'n_regions_kept':    n_kept,
        })

    # -- drop all-zero rows ---------------------------------------------------
    # NMF cannot do anything useful with all-zero rows, so we drop them. This
    # is NOT the cisTopic-style quality filter -- it just removes rows that
    # contribute zero information to the factorisation.
    if args.drop_zero_rows:
        row_nnz = np.diff(mat.indptr)
        keep_rows = row_nnz > 0
        n_drop = int((~keep_rows).sum())
        if n_drop > 0:
            log.info('Dropping %d / %d all-zero rows; keeping %d.',
                     n_drop, len(keep_rows), int(keep_rows.sum()))
            mat = mat[keep_rows]
            regions = regions[keep_rows]

    log.info('scOpen input matrix: %d regions x %d cells, nnz=%d',
             *mat.shape, mat.nnz)

    # -- write 10X-format input directory ------------------------------------
    out_dir = Path(cfg['paths']['scopen_input'])
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path  = out_dir / 'matrix.mtx'
    barcode_path = out_dir / 'barcodes.tsv'
    peak_path    = out_dir / 'peaks.bed'
    regions_path = out_dir / 'regions.tsv'

    log.info('Writing %s', matrix_path)
    sio.mmwrite(str(matrix_path), sp.coo_matrix(mat),
                field='integer', symmetry='general')

    log.info('Writing %s', barcode_path)
    _write_lines(list(barcodes), barcode_path)

    log.info('Writing %s', peak_path)
    with open(peak_path, 'w') as fh:
        for r in regions:
            chrom, s, e = _region_to_bed_row(str(r))
            fh.write(f'{chrom}\t{s}\t{e}\n')

    _write_lines(list(regions), regions_path)

    meta = {
        'shape':              [int(mat.shape[0]), int(mat.shape[1])],
        'n_regions_input':    n_regions_input,
        'n_regions':          int(mat.shape[0]),
        'n_cells':            int(mat.shape[1]),
        'binarize_input':     binarize,
        'drop_zero_rows':     bool(args.drop_zero_rows),
        'candidate_filter':   candidate_meta,
        'nnz':                int(mat.nnz),
        'matrix_path':        str(matrix_path),
        'barcodes_path':      str(barcode_path),
        'peaks_path':         str(peak_path),
        'regions_path':       str(regions_path),
    }
    (out_dir / 'prepare_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
