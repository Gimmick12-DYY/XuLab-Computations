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

    # Phase2L writes ``<name2save>.txt`` (default ``FITS_OUTPUT.txt``) in cwd.
    # It does not accept ``-o`` from this wrapper, so the basename is fixed unless
    # FITSPhase2L.py is edited.
    out_prefix = str(fits_cfg.get('phase1_output_prefix', 'FITS_OUTPUT'))
    canonical_txt = (run_dir / f'{out_prefix}.txt').resolve()

    before_mtime: dict[Path, float] = {}
    for p in run_dir.glob('*'):
        if not p.is_file():
            continue
        try:
            before_mtime[p.resolve()] = p.stat().st_mtime
        except OSError:
            pass

    cmd = [sys.executable, str(repo / script), '-i', str(inp)]
    log.info('Running Phase2 (%s mode): %s', mode, ' '.join(cmd))
    r = subprocess.run(cmd, cwd=run_dir)
    if r.returncode != 0:
        return r.returncode

    if canonical_txt.is_file():
        out_path = canonical_txt
    else:
        # Prefer brand-new files; also accept existing paths whose mtime changed
        # (Phase2 overwrites FITS_OUTPUT.txt, so it is never "new" by path set).
        touched: list[Path] = []
        for p in run_dir.glob('*'):
            if not p.is_file():
                continue
            pr = p.resolve()
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            prev = before_mtime.get(pr)
            if prev is None or m > prev + 1e-6:
                touched.append(pr)
        candidates = [p for p in touched if p.suffix.lower() in ('.csv', '.txt', '.tsv')]
        if not candidates:
            candidates = sorted(
                [p.resolve() for p in run_dir.glob('*')
                 if p.is_file() and p.suffix.lower() in ('.csv', '.txt', '.tsv')],
                key=lambda p: p.stat().st_mtime,
            )
        if not candidates:
            log.error('Could not locate FITS output matrix in %s', run_dir)
            return 3
        out_path = candidates[-1]
    (run_dir / 'fits_imputed_path.txt').write_text(str(out_path) + '\n')
    log.info('Detected FITS imputed output: %s', out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
