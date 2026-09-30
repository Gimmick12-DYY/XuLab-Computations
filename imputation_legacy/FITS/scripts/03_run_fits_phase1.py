#!/usr/bin/env python
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('03_fits_phase1')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--large-mode', type=str, default=None, choices=['auto', 'small', 'large'])
    ap.add_argument('--depth', type=int, default=None)
    ap.add_argument('--output-prefix', type=str, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    fits_cfg = cfg.setdefault('fits', {})

    repo = Path(cfg['paths']['fits_repo'])
    run_dir = Path(cfg['paths']['fits_run'])
    inp = Path(cfg['paths']['fits_input']) / 'fits_input.csv'
    if not repo.exists():
        log.error('FITS repo missing at %s. Clone from https://github.com/reggenlab/FITSpython first.', repo)
        return 2
    if not inp.exists():
        log.error('Missing FITS input CSV at %s. Run 02_prepare_fits_csv.py first.', inp)
        return 1

    mode = args.large_mode or fits_cfg.get('large_mode', 'auto')
    depth = int(args.depth or fits_cfg.get('phase1_depth', 4))
    out_prefix = args.output_prefix or fits_cfg.get('phase1_output_prefix', 'FITS_OUTPUT')

    # Auto mode: use L variant when many cells.
    n_cells = len((Path(cfg['paths']['fits_input']) / 'barcodes.tsv').read_text().splitlines())
    if mode == 'auto':
        mode = 'large' if n_cells > 3000 else 'small'

    script = 'main_FITS1.py' if mode == 'small' else 'FITSPhase1L.py'
    cmd = [sys.executable, str(repo / script), '-i', str(inp), '-o', out_prefix, '-l', str(depth)]

    run_dir.mkdir(parents=True, exist_ok=True)
    log.info('Running Phase1 (%s mode): %s', mode, ' '.join(cmd))
    r = subprocess.run(cmd, cwd=run_dir)
    if r.returncode != 0:
        return r.returncode
    log.info('Phase1 done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
