#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_run_magic.py
#
# Run MAGIC (Markov Affinity-based Graph Imputation of Cells) on the sparse
# (regions x cells) matrix written by step 02.
#
# MAGIC expects the canonical scRNA-seq orientation -- (cells x features) --
# so we transpose before fit_transform and transpose the output back to
# (cells x features) for storage. Step 05 transposes a final time to the
# (regions x cells) convention shared by the other pipelines.
#
# Outputs (under <work>/magic_run/):
#   imputed_cells_x_regions.npy   float32 (n_cells, n_regions) raw MAGIC output
#   meta.json                     params used + auto-selected t (if applicable)
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('03_magic')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _coerce_t(value) -> int | str:
    if isinstance(value, str):
        return 'auto' if value.lower() == 'auto' else int(value)
    if value is None:
        return 'auto'
    return int(value)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--knn',     type=int,   default=None)
    ap.add_argument('--knn-max', type=int,   default=None)
    ap.add_argument('--decay',   type=int,   default=None)
    ap.add_argument('--t',       type=str,   default=None,
                    help='"auto" or an integer.')
    ap.add_argument('--n-pca',   type=int,   default=None)
    ap.add_argument('--solver',  type=str,   default=None,
                    choices=['exact', 'approximate'])
    ap.add_argument('--n-jobs',  type=int,   default=None)
    ap.add_argument('--random-state', type=int, default=None)
    ap.add_argument('--verbose', type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    mc = cfg.setdefault('magic', {})
    for k, v in (
        ('knn',          args.knn),
        ('knn_max',      args.knn_max),
        ('decay',        args.decay),
        ('t',            args.t),
        ('n_pca',        args.n_pca),
        ('solver',       args.solver),
        ('n_jobs',       args.n_jobs),
        ('random_state', args.random_state),
        ('verbose',      args.verbose),
    ):
        if v is not None:
            mc[k] = v

    in_dir = Path(cfg['paths']['magic_input'])
    matrix_path = in_dir / 'matrix.npz'
    regions_path = in_dir / 'regions.tsv'
    barcodes_path = in_dir / 'barcodes.tsv'
    for p in (matrix_path, regions_path, barcodes_path):
        if not p.exists():
            log.error('Missing %s. Run 02_prepare_magic_input.py first.', p)
            return 1

    log.info('Loading sparse input matrix from %s', matrix_path)
    mat = sp.load_npz(matrix_path).tocsr()  # regions x cells
    n_regions, n_cells = mat.shape
    log.info('  shape (regions x cells) = %s, nnz=%d', mat.shape, mat.nnz)

    # MAGIC wants cells x features; transpose once.
    log.info('Transposing to cells x regions for MAGIC...')
    X = mat.T.tocsr()  # cells x regions
    del mat
    log.info('  shape (cells x regions) = %s, nnz=%d', X.shape, X.nnz)

    try:
        import magic  # type: ignore
    except ImportError as e:
        log.error('magic-impute is not importable (%s). `pip install magic-impute`.', e)
        return 2

    knn = int(mc.get('knn', 5))
    knn_max = mc.get('knn_max', None)
    knn_max = int(knn_max) if knn_max not in (None, 'null') else None
    decay = mc.get('decay', 1)
    decay = int(decay) if decay not in (None, 'null') else None
    t = _coerce_t(mc.get('t', 'auto'))
    n_pca = mc.get('n_pca', 100)
    n_pca = int(n_pca) if n_pca not in (None, 'null', 0) else None
    solver = str(mc.get('solver', 'exact'))
    knn_dist = str(mc.get('knn_dist', 'euclidean'))
    n_jobs = int(mc.get('n_jobs', 1))
    random_state = int(mc.get('random_state', 42))
    verbose = int(mc.get('verbose', 1))

    log.info('Initialising MAGIC(knn=%s, knn_max=%s, decay=%s, t=%s, n_pca=%s, '
             'solver=%s, knn_dist=%s, n_jobs=%s, random_state=%s)',
             knn, knn_max, decay, t, n_pca, solver, knn_dist, n_jobs, random_state)
    op = magic.MAGIC(
        knn=knn,
        knn_max=knn_max,
        decay=decay,
        t=t,
        n_pca=n_pca,
        solver=solver,
        knn_dist=knn_dist,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    log.info('Running MAGIC.fit_transform on %d cells x %d regions ...',
             X.shape[0], X.shape[1])
    X_magic = op.fit_transform(X)
    if hasattr(X_magic, 'values'):  # pandas DataFrame
        X_magic = X_magic.values
    X_magic = np.asarray(X_magic, dtype=np.float32)
    log.info('MAGIC output shape %s, mean=%.4g, max=%.4g',
             X_magic.shape, float(X_magic.mean()), float(X_magic.max()))

    if X_magic.shape != (n_cells, n_regions):
        log.error('Unexpected MAGIC output shape %s (expected (%d, %d)).',
                  X_magic.shape, n_cells, n_regions)
        return 3

    out_dir = Path(cfg['paths']['magic_run'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'imputed_cells_x_regions.npy'
    log.info('Writing raw MAGIC output -> %s (float32)', out_path)
    np.save(out_path, X_magic)

    # Try to surface the auto-selected t.
    selected_t = None
    for attr in ('optimal_t', 't_opt', 't'):
        if hasattr(op, attr):
            try:
                v = getattr(op, attr)
                if v is not None:
                    selected_t = int(v) if not isinstance(v, str) else v
                    break
            except Exception:
                continue

    meta = {
        'pipeline':      'MAGIC',
        'output_path':   str(out_path),
        'output_shape':  [int(X_magic.shape[0]), int(X_magic.shape[1])],
        'n_cells':       int(n_cells),
        'n_regions':     int(n_regions),
        'params': {
            'knn':          knn,
            'knn_max':      knn_max,
            'decay':        decay,
            't_requested':  t,
            't_selected':   selected_t,
            'n_pca':        n_pca,
            'solver':       solver,
            'knn_dist':     knn_dist,
            'n_jobs':       n_jobs,
            'random_state': random_state,
        },
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    log.info('Done. Selected t=%s', selected_t)
    return 0


if __name__ == '__main__':
    sys.exit(main())
