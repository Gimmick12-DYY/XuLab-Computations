#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 04_select_model.py
#
# scOpen runs the rank grid + kneedle internally and only persists W/H for the
# chosen rank. This step:
#   * parses scOpen_error.txt (the rank-vs-error curve scopen wrote)
#   * (re-)runs kneedle to identify the elbow
#   * inspects the W matrix to determine the rank scopen actually used
#   * writes select/model_selection.png + selected_model.meta.yaml
#
# Mirrors cisTopic's 04_select_model.py so the workflow shape is consistent.
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _cfg import base_parser, load_config, resolve_paths

log = logging.getLogger('04_select')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main() -> int:
    ap = base_parser(__doc__ or '')
    ap.add_argument('--output-prefix', type=str, default='scOpen')
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    run_dir    = Path(cfg['paths']['scopen_run'])
    select_dir = Path(cfg['paths']['select'])

    err_path = run_dir / f'{args.output_prefix}_error.txt'
    w_path   = run_dir / f'{args.output_prefix}_peaks.txt'

    if not w_path.exists():
        log.error('Missing %s. Run 03_run_scopen.py first.', w_path)
        return 1

    # The number of columns in W (minus the index) is the rank scopen used.
    # Read just the header to avoid loading the whole 119k-row matrix here.
    with open(w_path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
    rank_used = max(0, len(header) - 1)
    log.info('Detected rank from %s: k=%d', w_path.name, rank_used)

    summary_lines = [f'rank: {rank_used}', f'source_w: {w_path}']

    if err_path.exists():
        df = pd.read_csv(err_path, sep='\t')
        if not {'ranks', 'error'} <= set(df.columns):
            log.warning('Unexpected columns in %s: %s', err_path, list(df.columns))
        else:
            ks   = df['ranks'].to_numpy()
            errs = df['error'].to_numpy()

            knee = None
            try:
                from kneed import KneeLocator
                kl = KneeLocator(ks, errs, S=1.0, curve='convex', direction='decreasing')
                knee = int(kl.knee) if kl.knee is not None else None
            except Exception as e:  # kneed missing or degenerate curve
                log.warning('Kneedle failed (%s); falling back to rank-from-W only.', e)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(ks, errs, 'o-', color='#377EB8', label='reconstruction error')
            ax.axvline(rank_used, color='red', ls='--', lw=1,
                       label=f'rank used = {rank_used}')
            if knee is not None and knee != rank_used:
                ax.axvline(knee, color='green', ls=':', lw=1,
                           label=f'kneedle (recomputed) = {knee}')
            ax.set_xlabel('rank (n_components)')
            ax.set_ylabel('NMF reconstruction error')
            ax.set_title('scOpen model selection')
            ax.legend(frameon=False)
            fig.tight_layout()
            png_path = select_dir / 'model_selection.png'
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            log.info('Wrote %s', png_path)

            (select_dir / 'model_selection.tsv').write_text(
                df.to_csv(sep='\t', index=False)
            )

            summary_lines.append(f'kneedle_recomputed: {knee}')
            summary_lines.append('grid:')
            for k, e in zip(ks, errs):
                summary_lines.append(f'  - rank: {int(k)}\n    error: {float(e):.6g}')
    else:
        log.warning('No %s — likely scopen ran without --estimate_rank.', err_path)

    (select_dir / 'selected_model.meta.yaml').write_text(
        '\n'.join(summary_lines) + '\n'
    )
    log.info('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
