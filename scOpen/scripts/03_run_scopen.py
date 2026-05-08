#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_run_scopen.py
#
# Invoke the scopen CLI on the 10X-format directory written by step 02.
# We pass --no_impute so scopen skips writing a huge dense imputed text file;
# step 05 reconstructs W @ H.T directly into HDF5 from the small W and H text
# outputs.
#
# Outputs (under <work>/scopen_run/):
#   scOpen_peaks.txt       W matrix: peaks x k (one row per filtered peak)
#   scOpen_barcodes.txt    H matrix: cells x k (one row per cell)
#   scOpen_error.txt       error per rank (only when estimate_rank: true)
#   scOpen_error.pdf       elbow curve
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('03_scopen')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def _build_cmd(bin_name: str, input_dir: Path, run_dir: Path, output_prefix: str, sc: dict,
               estimate_rank: bool) -> list[str]:
    cmd = [
        bin_name,
        '--input',         str(input_dir),
        '--input_format',  '10X',
        '--output_dir',    str(run_dir),
        '--output_prefix', output_prefix,
        '--output_format', 'dense',          # scopen still writes W/H regardless;
                                              # this only affects the imputed file
                                              # (skipped by --no_impute).
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
    ap.add_argument('--n-components', type=int, default=None)
    ap.add_argument('--alpha',        type=float, default=None)
    ap.add_argument('--max-iter',     type=int, default=None)
    ap.add_argument('--nc',           type=int, default=None)
    ap.add_argument('--no-estimate-rank', action='store_true',
                    help='Force fixed n_components instead of grid + kneedle.')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    ap.add_argument('--scopen-bin', type=str, default=None,
                    help='Path to the scopen executable (default: search PATH).')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    sc = cfg.setdefault('scopen', {})
    if args.n_components is not None: sc['n_components'] = args.n_components
    if args.alpha is not None:        sc['alpha']        = args.alpha
    if args.max_iter is not None:     sc['max_iter']     = args.max_iter
    if args.nc is not None:           sc['nc']           = args.nc
    if args.no_estimate_rank:         sc['estimate_rank'] = False

    bin_name = args.scopen_bin or shutil.which('scopen')
    if not bin_name:
        log.error('scopen binary not found in PATH. Install with `pip install scopen` '
                  'or pass --scopen-bin.')
        return 2

    input_dir = Path(cfg['paths']['scopen_input'])
    if not (input_dir / 'matrix.mtx').exists():
        log.error('Missing %s/matrix.mtx. Run 02_prepare_scopen_input.py first.', input_dir)
        return 1
    run_dir = Path(cfg['paths']['scopen_run'])
    run_dir.mkdir(parents=True, exist_ok=True)

    use_estimate_rank = bool(sc.get('estimate_rank', True))
    cmd = _build_cmd(bin_name, input_dir, run_dir, args.output_prefix, sc, use_estimate_rank)

    log.info('Running: %s', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=run_dir)
    if r.returncode != 0 and use_estimate_rank:
        # scopen occasionally fails in its internal plot_knee() when kneedle
        # finds no elbow (kl.knee is None). Retry once in fixed-rank mode.
        fallback_k = int(sc.get('n_components', sc.get('max_n_components', 30)))
        log.warning(
            'scopen rank-estimation run failed (rc=%d). Retrying once with '
            '--no-estimate-rank --n-components %d.',
            r.returncode, fallback_k
        )
        sc_fallback = dict(sc)
        sc_fallback['n_components'] = fallback_k
        cmd = _build_cmd(bin_name, input_dir, run_dir, args.output_prefix, sc_fallback, False)
        log.info('Running fallback: %s', ' '.join(cmd))
        r = subprocess.run(cmd, cwd=run_dir)

    if r.returncode != 0:
        log.error('scopen exited with status %d', r.returncode)
        return r.returncode

    # Persist the scopen invocation for reproducibility.
    (run_dir / 'scopen_cmd.txt').write_text(' '.join(cmd) + '\n')

    # Sanity check expected outputs.
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
