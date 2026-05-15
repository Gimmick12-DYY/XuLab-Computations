#!/usr/bin/env python
# -----------------------------------------------------------------------------
# cell_graph_kneighbour.py
#
# Complementary axis to the bin-axis genomic-window propagation that lives
# inside cisTopic 05_impute.py and PUscOpen 06_impute.py: this script does the
# same thing but along the CELL axis. For each cell c we compute its k nearest
# neighbours in some embedding space, then propagate raw signal at every
# (region, cell) coordinate from those neighbours -- i.e. if at least
# `min_neighbours` of cell c's k neighbours had a raw nonzero at bin r, c
# inherits a propagated entry there.
#
# The point: a bin that was never observed in cell c but observed in its kNN
# neighbours is a credible recovery target. This is the MAGIC trick applied
# at the *raw* level so it can introduce NEW (region, cell) entries -- bins
# that LDA / NMF / scOpen could never reach because they never saw signal at
# them in *cell c specifically*.
#
# Algorithm:
#   1. Load raw mm matrix (binarised) -> sparse CSR (R, C).
#   2. Load cell embeddings (cells x features) and align to mm barcode order.
#   3. Build a cell-cell kNN graph (sklearn NearestNeighbors).
#   4. Compute count[r, c] = #{c' in NN(c) : raw[r, c'] > 0} via a single
#      sparse @ sparse multiply: raw_bool @ A_i^T  where A_i[c, c'] = 1
#      iff c' is in c's neighbour list.
#   5. Keep entries with count >= min_neighbours; propagated value is
#      `weight * (count / k)` -- a fraction-of-supporting-neighbours score.
#   6. Optionally restrict to bins where raw[r, c] == 0 (no overwrite of raw).
#   7. Optionally union with raw to produce a full augmented matrix.
#
# Inputs: --mm-dir (required), --embeddings (required), --out-dir (required).
# Embeddings can be:
#   * a directory containing factors.npz with key 'H' (cisTopic / scOpen);
#     the script extracts H.T as the (cells, topics) embedding and reads
#     barcodes.tsv next to it for alignment.
#   * a .npz file with key 'H' (K, C) or 'embeddings'/'cell_features' (C, F).
#   * a .npy file shaped (C, F).
#
# If the embeddings come from a different barcode order than mm, pass
# --embeddings-barcodes to align them. Otherwise the script assumes
# embeddings rows correspond 1:1 to mm/barcodes.tsv.gz order.
#
# Outputs (under --out-dir):
#   matrix_csr.npz   full mm-aligned CSR (raw + propagated entries unless
#                    --no-keep-raw is passed)
#   regions.tsv      full mm region order
#   barcodes.tsv     full mm barcode order
#   <out_name>_meta.json   parameters + n_propagated, n_raw, n_output
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger('cell_kgraph')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _read_lines(path: Path) -> list[str]:
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def load_raw(mm_dir: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    mtx_path = mm_dir / 'matrix.mtx.gz'
    reg_path = mm_dir / 'regions.tsv.gz'
    bar_path = mm_dir / 'barcodes.tsv.gz'
    for p in (mtx_path, reg_path, bar_path):
        if not p.exists():
            raise SystemExit(f'Missing {p}')
    log.info('Reading raw mm matrix %s', mtx_path)
    mat = sio.mmread(str(mtx_path)).tocsr()
    bin_mat = (mat > 0).astype(np.float32)
    regs = _read_lines(reg_path)
    bars = _read_lines(bar_path)
    log.info('  raw: %d regions x %d cells, nnz=%d',
             *bin_mat.shape, bin_mat.nnz)
    return bin_mat, regs, bars


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Return (embeddings (C, F), barcodes or None).

    Auto-detect:
      directory  ->  factors.npz['H'].T  + barcodes.tsv next to it
      .npz       ->  key 'H' (K, C) -> H.T   OR   'embeddings' / 'cell_features' (C, F)
      .npy       ->  array as-is, assumed (C, F)
    """
    if path.is_dir():
        npz_path = path / 'factors.npz'
        if not npz_path.exists():
            raise SystemExit(
                f'No factors.npz in {path}; pass --embeddings <path/to/file.npz|.npy> '
                f'for non-factored pipelines (FITS / MAGIC / PUscOpen csr).'
            )
        log.info('Loading embeddings from %s (H matrix from factors.npz)', npz_path)
        with np.load(npz_path) as z:
            if 'H' not in z.files:
                raise SystemExit(f'{npz_path} has no H matrix')
            H = np.asarray(z['H'], dtype=np.float64)  # (K, C)
        emb = H.T  # (C, K)
        bars: list[str] | None = None
        for name in ('barcodes.tsv', 'barcodes.tsv.gz'):
            p = path / name
            if p.exists():
                bars = _read_lines(p)
                break
        return emb, bars

    suffix = path.suffix.lower()
    if suffix == '.npz':
        log.info('Loading embeddings from %s', path)
        with np.load(path) as z:
            if 'H' in z.files:
                H = np.asarray(z['H'], dtype=np.float64)
                return H.T, None
            for k_try in ('embeddings', 'cell_features', 'X'):
                if k_try in z.files:
                    return np.asarray(z[k_try], dtype=np.float64), None
            raise SystemExit(
                f'{path} has no recognised embedding key '
                f'(H / embeddings / cell_features / X)'
            )
    if suffix == '.npy':
        log.info('Loading embeddings from %s', path)
        return np.load(path).astype(np.float64), None
    raise SystemExit(f'Unsupported embeddings path: {path}')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ or '',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--mm-dir', type=Path, required=True,
                    help='Raw mm directory (matrix.mtx.gz + regions.tsv.gz + barcodes.tsv.gz)')
    ap.add_argument('--embeddings', type=Path, required=True,
                    help='Cell embeddings: directory with factors.npz, .npz with key '
                         'H/embeddings, or .npy shaped (n_cells, n_features).')
    ap.add_argument('--embeddings-barcodes', type=Path, default=None,
                    help='Barcodes aligned to embeddings rows. If omitted, '
                         'a sibling barcodes.tsv next to embeddings is used; '
                         'failing that, embeddings rows are assumed to match '
                         'mm/barcodes.tsv.gz exactly.')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--out-name', type=str, default='cell_kgraph')
    ap.add_argument('--k', type=int, default=20,
                    help='Number of neighbours per cell (default 20).')
    ap.add_argument('--min-neighbours', type=int, default=3,
                    help='Minimum neighbours with raw > 0 at a bin before '
                         'propagating signal there (default 3).')
    ap.add_argument('--weight', type=float, default=1.0,
                    help='Multiplier on the propagated value '
                         '(value = weight * count / k; default 1.0).')
    ap.add_argument('--metric', type=str, default='cosine',
                    choices=['cosine', 'euclidean', 'manhattan'],
                    help='kNN distance metric (default cosine).')
    ap.add_argument('--include-self', action='store_true',
                    help="Include each cell as its own neighbour. Default: drop self.")
    ap.add_argument('--no-only-zero-targets', dest='only_zero_targets',
                    action='store_false', default=True,
                    help='Allow propagation to overwrite raw-positive entries '
                         '(default: only fill bins where raw is zero in the cell).')
    ap.add_argument('--no-keep-raw', dest='keep_raw',
                    action='store_false', default=True,
                    help='Output ONLY the propagated entries, not the union with raw.')
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help='Parallel jobs for kNN search (-1 = all cores).')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Raw -----
    raw_bool, mm_regs, mm_bars = load_raw(args.mm_dir)
    R, C_mm = raw_bool.shape

    # ----- Embeddings -----
    embeddings, emb_bars = load_embeddings(args.embeddings)
    log.info('Embeddings: shape=%s', embeddings.shape)

    if args.embeddings_barcodes is not None:
        emb_bars = _read_lines(args.embeddings_barcodes)

    if emb_bars is None:
        if embeddings.shape[0] != C_mm:
            raise SystemExit(
                f'Embeddings rows ({embeddings.shape[0]}) != mm cells ({C_mm}); '
                f'pass --embeddings-barcodes to align by barcode name.'
            )
        log.info('No embeddings-barcodes given; assuming row order matches mm '
                 '(%d cells).', C_mm)
    else:
        if len(emb_bars) != embeddings.shape[0]:
            raise SystemExit(
                f'embeddings barcodes ({len(emb_bars)}) != embeddings rows '
                f'({embeddings.shape[0]})'
            )
        log.info('Aligning embeddings to mm barcode order ...')
        bar_to_emb_idx = {b: i for i, b in enumerate(emb_bars)}
        new_emb = np.zeros((C_mm, embeddings.shape[1]), dtype=embeddings.dtype)
        n_missing = 0
        for j, bc in enumerate(mm_bars):
            ei = bar_to_emb_idx.get(bc)
            if ei is None:
                n_missing += 1
            else:
                new_emb[j] = embeddings[ei]
        embeddings = new_emb
        if n_missing:
            log.warning('  %d / %d mm barcodes had no embedding; '
                        'using zero vector (those cells will get no propagation).',
                        n_missing, C_mm)

    # ----- kNN graph -----
    k = int(args.k)
    log.info('Building cell-cell kNN graph: k=%d, metric=%s, n_jobs=%s',
             k, args.metric, args.n_jobs)
    n_query = k if args.include_self else (k + 1)
    n_query = min(n_query, C_mm)
    nn = NearestNeighbors(
        n_neighbors=n_query, metric=args.metric, n_jobs=args.n_jobs,
    )
    nn.fit(embeddings)
    nbr_idx = nn.kneighbors(embeddings, return_distance=False)
    if not args.include_self:
        log.info('  dropping self from each cell\'s neighbour list ...')
        out = np.empty((C_mm, k), dtype=np.int64)
        for i in range(C_mm):
            row = nbr_idx[i]
            row = row[row != i]
            if row.size < k:
                pad = np.full(k - row.size, i, dtype=row.dtype)
                row = np.concatenate([row, pad])
            out[i] = row[:k]
        nbr_idx = out
    log.info('  built kNN: %d cells x %d neighbours', C_mm, k)

    # ----- Sparse cell-cell adjacency (indicator) -----
    A_rows = np.repeat(np.arange(C_mm, dtype=np.int64), k)
    A_cols = nbr_idx.ravel().astype(np.int64)
    A_data = np.ones(A_rows.size, dtype=np.float32)
    A_i = sparse.csr_matrix(
        (A_data, (A_rows, A_cols)), shape=(C_mm, C_mm), dtype=np.float32,
    )
    A_i_T = A_i.T.tocsr()

    # ----- count[r, c] = # nbrs of c with raw > 0 at bin r -----
    log.info('Computing propagation counts: count = raw_bool @ A_i.T ...')
    count_mat = (raw_bool @ A_i_T).tocoo()
    log.info('  count_mat: shape=%s, nnz=%d', count_mat.shape, count_mat.nnz)

    # ----- Filter by min_neighbours -----
    keep_mask = count_mat.data >= float(args.min_neighbours)
    n_pre = int(count_mat.nnz)
    n_post = int(keep_mask.sum())
    log.info('  applied min_neighbours=%d threshold: %d -> %d entries',
             args.min_neighbours, n_pre, n_post)

    if n_post == 0:
        log.warning('No entries passed the min_neighbours threshold; '
                    'try lowering --min-neighbours or raising --k.')
        propagated = sparse.csr_matrix((R, C_mm), dtype=np.float32)
    else:
        # value = weight * (count / k); proportional to fraction of supporting nbrs
        prop_vals = (count_mat.data[keep_mask] / float(k)) * float(args.weight)
        prop_rows = count_mat.row[keep_mask]
        prop_cols = count_mat.col[keep_mask]
        propagated = sparse.coo_matrix(
            (prop_vals, (prop_rows, prop_cols)),
            shape=(R, C_mm), dtype=np.float32,
        ).tocsr()
        propagated.sum_duplicates()

    n_pre_zero_filter = int(propagated.nnz)
    if args.only_zero_targets and propagated.nnz:
        # Subtract entries that overlap with raw-positive coords.
        raw_indicator = (raw_bool != 0).astype(np.float32)
        overlap = propagated.multiply(raw_indicator)
        propagated = (propagated - overlap)
        propagated.eliminate_zeros()
        log.info('  only-zero-targets: %d -> %d entries (removed %d overlap with raw)',
                 n_pre_zero_filter, propagated.nnz,
                 n_pre_zero_filter - propagated.nnz)

    # ----- Combine with raw -----
    raw_nnz = int(raw_bool.nnz)
    propagated_nnz = int(propagated.nnz)
    if args.keep_raw:
        # Disjoint pattern when only_zero_targets=True; safe addition either way.
        output = (raw_bool.astype(np.float32) + propagated).tocsr()
    else:
        output = propagated.tocsr()
    output.sum_duplicates()
    output_nnz = int(output.nnz)

    # ----- Persist -----
    csr_path     = args.out_dir / 'matrix_csr.npz'
    regions_out  = args.out_dir / 'regions.tsv'
    barcodes_out = args.out_dir / 'barcodes.tsv'
    meta_out     = args.out_dir / f'{args.out_name}_meta.json'

    log.info('Writing %s (shape=%s, nnz=%d)', csr_path, output.shape, output_nnz)
    sparse.save_npz(csr_path, output)
    regions_out.write_text('\n'.join(mm_regs) + '\n')
    barcodes_out.write_text('\n'.join(mm_bars) + '\n')

    summary = {
        'pipeline':           'cell_kgraph',
        'format':             'csr_full_mm',
        'matrix_path':        str(csr_path),
        'regions_path':       str(regions_out),
        'barcodes_path':      str(barcodes_out),
        'shape':              [int(R), int(C_mm)],
        'raw_nnz':            raw_nnz,
        'propagated_nnz':     propagated_nnz,
        'output_nnz':         output_nnz,
        'gain_vs_raw':        int(output_nnz - raw_nnz),
        'k':                  int(k),
        'min_neighbours':     int(args.min_neighbours),
        'weight':             float(args.weight),
        'metric':             args.metric,
        'include_self':       bool(args.include_self),
        'only_zero_targets':  bool(args.only_zero_targets),
        'keep_raw':           bool(args.keep_raw),
        'embeddings_path':    str(args.embeddings),
        'embeddings_shape':   [int(embeddings.shape[0]), int(embeddings.shape[1])],
        'mm_dir':             str(args.mm_dir),
    }
    meta_out.write_text(json.dumps(summary, indent=2) + '\n')
    log.info('Wrote %s', meta_out)
    log.info('Done. raw_nnz=%d, propagated_nnz=%d, output_nnz=%d (gain=%+d)',
             raw_nnz, propagated_nnz, output_nnz, output_nnz - raw_nnz)
    return 0


if __name__ == '__main__':
    sys.exit(main())
