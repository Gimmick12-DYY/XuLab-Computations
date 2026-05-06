#!/usr/bin/env python
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('04_fits_phase2')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--large-mode', type=str, default=None, choices=['auto', 'small', 'large'])
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    fits_cfg = cfg.setdefault('fits', {})

    repo = Path(cfg['paths']['fits_repo'])
    run_dir = Path(cfg['paths']['fits_run'])
    inp = Path(cfg['paths']['fits_input']) / 'fits_input.csv'
    if not repo.exists():
        log.error('FITS repo missing at %s.', repo)
        return 2
    if not inp.exists():
        log.error('Missing FITS input CSV at %s.', inp)
        return 1

    mode = args.large_mode or fits_cfg.get('large_mode', 'auto')
    n_cells = len((Path(cfg['paths']['fits_input']) / 'barcodes.tsv').read_text().splitlines())
    if mode == 'auto':
        mode = 'large' if n_cells > 3000 else 'small'

    script = 'main_FITS2.py' if mode == 'small' else 'FITSPhase2L.py'

    before = {p.resolve() for p in run_dir.glob('*')}
    cmd = [sys.executable, str(repo / script), '-i', str(inp)]
    log.info('Running Phase2 (%s mode): %s', mode, ' '.join(cmd))
    r = subprocess.run(cmd, cwd=run_dir)
    if r.returncode != 0:
        return r.returncode

    after = sorted([p.resolve() for p in run_dir.glob('*') if p.resolve() not in before], key=lambda p: p.stat().st_mtime)
    candidates = [p for p in after if p.suffix.lower() in ('.csv', '.txt', '.tsv')]
    if not candidates:
        # fallback to newest csv-like file in run dir
        candidates = sorted([p.resolve() for p in run_dir.glob('*.csv')], key=lambda p: p.stat().st_mtime)

    if not candidates:
        log.error('Could not locate FITS output matrix in %s', run_dir)
        return 3

    out_path = candidates[-1]
    (run_dir / 'fits_imputed_path.txt').write_text(str(out_path) + '\n')
    log.info('Detected FITS imputed output: %s', out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
