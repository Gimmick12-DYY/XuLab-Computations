#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('05_compare_fits')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', dtype=np.float64)


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--imputed-file', type=str, default=None)
    ap.add_argument('--thresholds', type=str, default=None)
    ap.add_argument('--out-name', type=str, default='fits_compare')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    cmp_cfg = cfg.setdefault('compare', {})

    inp_csv = Path(cfg['paths']['fits_input']) / 'fits_input.csv'
    if not inp_csv.exists():
        log.error('Missing FITS input csv: %s', inp_csv)
        return 1

    if args.imputed_file:
        imp_path = Path(args.imputed_file)
    else:
        marker = Path(cfg['paths']['fits_run']) / 'fits_imputed_path.txt'
        if not marker.exists():
            log.error('Missing %s. Run 04_run_fits_phase2.py first.', marker)
            return 1
        imp_path = Path(marker.read_text().strip())

    if not imp_path.exists():
        log.error('Imputed file not found: %s', imp_path)
        return 1

    x = _load_csv(inp_csv)
    y = _load_csv(imp_path)
    if x.shape != y.shape:
        log.error('Shape mismatch input %s vs imputed %s', x.shape, y.shape)
        return 2

    x_bin = (x > 0)
    orig_nnz = int(x_bin.sum())
    total = int(x_bin.size)

    thresholds = args.thresholds
    if thresholds is None:
        thresholds = cmp_cfg.get('thresholds', [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    if isinstance(thresholds, str):
        tvals = [float(t.strip()) for t in thresholds.split(',') if t.strip()]
    else:
        tvals = [float(t) for t in thresholds]

    rows = []
    for t in sorted(tvals):
        pred = (y > t)
        p = int(pred.sum())
        tp = int((pred & x_bin).sum())
        fp = p - tp
        fn = orig_nnz - tp
        precision = tp / p if p > 0 else float('nan')
        recall = tp / orig_nnz if orig_nnz > 0 else float('nan')
        f1 = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0.0
        rows.append({
            'threshold': t,
            'pred_nnz': p,
            'pred_density': p / total,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'nnz_fold_vs_original': p / orig_nnz if orig_nnz > 0 else float('nan'),
        })

    out_dir = Path(cfg['paths']['eval'])
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv = out_dir / f'{args.out_name}.tsv'
    keys = list(rows[0].keys())
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(keys) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[k]) for k in keys) + '\n')

    summary = {
        'input_csv': str(inp_csv),
        'imputed_file': str(imp_path),
        'shape': [int(x.shape[0]), int(x.shape[1])],
        'original_nnz': orig_nnz,
        'original_density': orig_nnz / total,
        'rows': rows,
        'best_f1_row': max(rows, key=lambda r: r['f1']),
    }
    js = out_dir / f'{args.out_name}.json'
    js.write_text(json.dumps(summary, indent=2) + '\n')

    log.info('Wrote %s and %s', tsv, js)
    log.info('Best F1 threshold %.3g (F1=%.4f)', summary['best_f1_row']['threshold'], summary['best_f1_row']['f1'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
