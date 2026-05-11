#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 04_run_scopen.py
#
# Subset the raw mm matrix to the refined-mask bins from step 03, write a
# 10X-format directory scOpen consumes, and invoke the scopen CLI with
# --no_impute (we reconstruct W @ H ourselves in step 05).
#
# Outputs (under <work>/scopen_input/):
#   matrix.mtx, barcodes.tsv, peaks.bed, regions.tsv, prepare_meta.json
# Outputs (under <work>/scopen_run/):
#   scOpen_peaks.txt   W matrix (refined-mask bins x k)
#   scOpen_barcodes.txt  H matrix (k x cells)
#   scOpen_error.{txt,pdf}  rank vs reconstruction error (if --estimate_rank)
#   scopen_cmd.txt   the exact scopen invocation
# -----------------------------------------------------------------------------
from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('04_scopen')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, 'rt') as fh:
        return [ln.rstrip('\n') for ln in fh]


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln]


def _write_lines(lines: list[str], path: Path) -> None:
    path.write_text('\n'.join(lines) + '\n')


def _bed_row(name: str) -> tuple[str, str, str]:
    m = _REGION_RE.match(name)
    if not m:
        raise SystemExit(f'Cannot parse region {name!r} as chr:start-end.')
    return m.group(1), m.group(2), m.group(3)


def _build_scopen_cmd(bin_path: str, input_dir: Path, run_dir: Path,
                      output_prefix: str, sc: dict, estimate_rank: bool
                      ) -> list[str]:
    cmd = [
        bin_path,
        '--input',         str(input_dir),
        '--input_format',  '10X',
        '--output_dir',    str(run_dir),
        '--output_prefix', output_prefix,
        '--output_format', 'dense',
        '--no_impute',
        '--alpha',         str(float(sc.get('alpha', 1.0))),
        '--max_iter',      str(int(sc.get('max_iter', 500))),
        '--init',          str(sc.get('init', 'nndsvd')),
        '--random_state',  str(int(sc.get('random_state', 42))),
        '--nc',            str(int(sc.get('nc', 1))),
        '--verbose',       str(int(sc.get('verbose', 0))),
    ]
    if estimate_rank:
        cmd += [
            '--estimate_rank',
            '--min_n_components',  str(int(sc.get('min_n_components', 5))),
            '--max_n_components',  str(int(sc.get('max_n_components', 30))),
            '--step_n_components', str(int(sc.get('step_n_components', 5))),
        ]
    else:
        cmd += ['--n_components', str(int(sc.get('n_components', 30)))]
    return cmd


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--scopen-bin', type=str, default=None,
                    help='Path to the scopen executable (default: search PATH).')
    ap.add_argument('--no-binarize', action='store_true',
                    help='Disable binarisation (scOpen still binarises internally).')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    sc = cfg.setdefault('scopen', {})
    bin_path = args.scopen_bin or shutil.which('scopen')
    if not bin_path:
        log.error('scopen binary not found in PATH. `pip install scopen` or pass --scopen-bin.')
        return 2

    mm_dir = Path(cfg['paths']['mm'])
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            log.error('Missing %s. Run 01_export_rds_to_mm.R first.', p)
            return 1

    refined_tsv = Path(cfg['paths']['pu']) / 'refined_mask.tsv'
    if not refined_tsv.exists():
        log.error('Missing %s. Run 03_pu_classifier.py first.', refined_tsv)
        return 1
    refined_names = _read_lines(refined_tsv)
    if not refined_names:
        log.error('Refined mask is empty; nothing to feed to scOpen.')
        return 1
    log.info('Refined mask: %d bins', len(refined_names))

    log.info('Reading raw matrix %s', mtx_path)
    mat = sio.mmread(mtx_path).tocsr()
    regions = _read_lines_gz(reg_path)
    barcodes = _read_lines_gz(bar_path)
    if mat.shape != (len(regions), len(barcodes)):
        log.error('Shape mismatch: matrix %s vs regions %d / barcodes %d',
                  mat.shape, len(regions), len(barcodes))
        return 1

    name_to_row = {r: i for i, r in enumerate(regions)}
    keep_idx_list: list[int] = []
    kept_names: list[str] = []
    missing = 0
    for nm in refined_names:
        ri = name_to_row.get(nm)
        if ri is None:
            missing += 1
            continue
        keep_idx_list.append(ri)
        kept_names.append(nm)
    if missing:
        log.warning('%d / %d refined-mask names not found in mm regions; skipping.',
                    missing, len(refined_names))
    keep_idx = np.asarray(keep_idx_list, dtype=np.int64)
    if keep_idx.size == 0:
        log.error('No refined-mask bins line up with the mm matrix; aborting.')
        return 1

    sub = mat[keep_idx]
    if not args.no_binarize:
        log.info('Binarising sub-matrix (count > 0 -> 1).')
        sub = sub.astype(bool).astype(np.int8)

    # Drop accidentally all-zero rows (e.g., a refined-mask bin where the raw
    # matrix has no observations for any cell).
    nz_per_row = np.diff(sub.indptr)
    nz_keep = nz_per_row > 0
    n_drop = int((~nz_keep).sum())
    if n_drop:
        log.info('Dropping %d / %d refined-mask bins with zero raw observations.',
                 n_drop, sub.shape[0])
        sub = sub[nz_keep]
        kept_names = [kept_names[i] for i in np.where(nz_keep)[0]]
    log.info('scOpen input: %d regions x %d cells, nnz=%d',
             *sub.shape, int(sub.nnz))

    # ------------------------------------------------------------------
    # Write 10X-format input directory
    # ------------------------------------------------------------------
    in_dir = Path(cfg['paths']['scopen_input'])
    in_dir.mkdir(parents=True, exist_ok=True)
    matrix_out = in_dir / 'matrix.mtx'
    barcodes_out = in_dir / 'barcodes.tsv'
    peaks_out = in_dir / 'peaks.bed'
    regions_out = in_dir / 'regions.tsv'

    log.info('Writing %s', matrix_out)
    sio.mmwrite(str(matrix_out), sp.coo_matrix(sub),
                field='integer', symmetry='general')
    _write_lines(list(barcodes), barcodes_out)
    with open(peaks_out, 'w') as fh:
        for nm in kept_names:
            chrom, s, e = _bed_row(nm)
            fh.write(f'{chrom}\t{s}\t{e}\n')
    _write_lines(kept_names, regions_out)
    (in_dir / 'prepare_meta.json').write_text(json.dumps({
        'shape':        [int(sub.shape[0]), int(sub.shape[1])],
        'n_regions':    int(sub.shape[0]),
        'n_cells':      int(sub.shape[1]),
        'nnz':          int(sub.nnz),
        'matrix_path':  str(matrix_out),
        'regions_path': str(regions_out),
        'barcodes_path': str(barcodes_out),
        'peaks_path':   str(peaks_out),
        'source_refined_mask': str(refined_tsv),
    }, indent=2) + '\n')

    # ------------------------------------------------------------------
    # Invoke scopen
    # ------------------------------------------------------------------
    run_dir = Path(cfg['paths']['scopen_run'])
    run_dir.mkdir(parents=True, exist_ok=True)
    use_estimate_rank = bool(sc.get('estimate_rank', True))
    cmd = _build_scopen_cmd(bin_path, in_dir, run_dir, args.output_prefix, sc,
                            use_estimate_rank)
    log.info('Running: %s', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=run_dir)
    if r.returncode != 0 and use_estimate_rank:
        fallback_k = int(sc.get('n_components', sc.get('max_n_components', 30)))
        log.warning('scopen rank-estimation failed (rc=%d); retrying with '
                    '--n-components %d.', r.returncode, fallback_k)
        sc_fb = dict(sc); sc_fb['n_components'] = fallback_k
        cmd = _build_scopen_cmd(bin_path, in_dir, run_dir, args.output_prefix, sc_fb, False)
        log.info('Running fallback: %s', ' '.join(cmd))
        r = subprocess.run(cmd, cwd=run_dir)

    if r.returncode != 0:
        log.error('scopen exited with status %d', r.returncode)
        return r.returncode

    (run_dir / 'scopen_cmd.txt').write_text(' '.join(cmd) + '\n')

    expected_w = run_dir / f'{args.output_prefix}_peaks.txt'
    expected_h = run_dir / f'{args.output_prefix}_barcodes.txt'
    if not expected_w.exists() or not expected_h.exists():
        log.error('scopen finished but W (%s) or H (%s) is missing.',
                  expected_w, expected_h)
        return 3
    log.info('scopen wrote %s and %s', expected_w, expected_h)
    return 0


if __name__ == '__main__':
    sys.exit(main())
