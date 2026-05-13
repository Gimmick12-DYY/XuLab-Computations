#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# imputed_matrix_to_bigwig.py
#
# Collapse an imputed (or raw) regions x cells matrix to a **single-track**
# BigWig (one score per bin) for IGV / UCSC Genome Browser.
#
# This is the same *deliverable* as Bioconductor **rtracklayer::export(...,
# format = "BigWig")** on a base-pair GRanges track: we write IGV-compatible
# BigWig using the **pyBigWig** library (https://github.com/deeptools/pyBigWig).
#
# Supported inputs (same layout ideas as downstream/compare_pos_neg.py):
#   * directory with **matrix.npy** + **regions.tsv**
#   * directory with **matrix_csr.npz** + **regions.tsv** (e.g. PUscOpen full-mm)
#   * directory with **factors.npz** + **regions.tsv** (cisTopic / scOpen)
#   * directory with **matrix.mtx.gz** + **regions.tsv.gz** + **barcodes.tsv.gz**
#     (raw / exported mm; regions must match matrix rows)
#
# **Factored (cisTopic / scOpen) tracks:** the per-bin value is the same
# pseudobulk used in **compare_pos_neg.py** — (W @ column-sum(H)) * scalar
# per_cell_factor, or W @ (H @ per_cell_factor) when a vector is stored.
# cisTopic keeps only **LDA-filtered** regions (~10^5 bins), not the full
# mm row set. LDA imputation is **smooth** (topic mixtures), so a summed
# pseudobulk .bw often looks like a high genome-wide baseline next to
# sparse raw counts or peakier matrix imputations; that is expected unless
# you switch IGV to log scaling or use **--aggregate mean** for a per-cell
# average pseudobulk. ChIP fold-enrichment tracks are not on the same scale.
#
# On **your** cisTopic CTCF export we measured ~75% of regions within a few
# percent of the pseudobulk median while p99/max are much larger, so linear
# IGV autoscale often looks like a flat saturated block; use **--transform log1p**
# (display only) or IGV's own log scale — that is not a separate pipeline bug.
# For a **zero-ish baseline** on the same pseudobulk, use **--center median**
# (subtract median then clip at 0; display-only). **--aggregate max** (factored
# only) uses per-bin max over cells instead of sum — peakier, not comparable to
# compare_pos_neg pseudobulk.
#
# **Different pipelines use different units** (raw counts vs rescaled pseudobulk
# vs PUscOpen). IGV autoscales each track independently, so y-ranges look
# unrelated even when peak *shapes* match. Use **--normalize max** or
# **--normalize p99** (display-only, per track) so values are on a comparable
# scale before writing the .bw.
#
# Region names must look like **chr:start-end** with integer coordinates
# (0-based start, end exclusive), matching your mm export convention.
#
# Chromosome lengths:
#   Pass **--chrom-sizes** pointing to a UCSC *chrom.sizes* file, e.g.:
#     wget -O hg38.chrom.sizes \\
#       https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes
#
# Example:
#   python downstream/imputed_matrix_to_bigwig.py \\
#       --input /work/.../PUscOpen/work/ctcf/impute \\
#       --output /work/.../tracks/PUscOpen_ctcf_mean.bw \\
#       --chrom-sizes /work/.../ref/hg38.chrom.sizes \\
#       --aggregate mean
#
# SLURM:
#   sbatch downstream/slurm/imputed_to_bigwig.sbatch
#     single input → one .bw (sets conda env, pyBigWig, chrom.sizes).
#   sbatch downstream/slurm/igv_ctcf_cistopic_puscopen_raw.sbatch
#     three .bw with defaults: per-track --normalize p99 (comparable IGV scale),
#     cisTopic --transform log1p; override with NORMALIZE_* / TRANSFORM_* env.
#
# Dependencies:
#   pip install pyBigWig numpy scipy
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger('imputed2bw')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

_REGION_RE = re.compile(r'^([^:]+):(\d+)-(\d+)$')


def parse_regions(lines: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (chroms, starts0, ends0) as parallel lists/arrays (0-based half-open)."""
    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        m = _REGION_RE.match(s)
        if not m:
            raise SystemExit(f'Bad region name line {i + 1}: {s!r} (expected chr:start-end)')
        chroms.append(m.group(1))
        starts.append(int(m.group(2)))
        ends.append(int(m.group(3)))
        if ends[-1] <= starts[-1]:
            raise SystemExit(f'Empty or inverted interval line {i + 1}: {s}')
    if not chroms:
        raise SystemExit('No regions parsed.')
    return chroms, np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def read_lines_maybe_gz(path: Path) -> list[str]:
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as fh:
            return [ln.rstrip('\n') for ln in fh if ln.strip()]
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def load_chrom_sizes(path: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith('#'):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        out.append((parts[0], int(parts[1])))
    if not out:
        raise SystemExit(f'No chromosomes in {path}')
    return out


def detect_kind(impute_dir: Path) -> str:
    if (impute_dir / 'matrix_csr.npz').is_file() and (impute_dir / 'regions.tsv').is_file():
        return 'csr'
    if (impute_dir / 'matrix.npy').is_file() and (impute_dir / 'regions.tsv').is_file():
        return 'dense'
    if (impute_dir / 'factors.npz').is_file() and (impute_dir / 'regions.tsv').is_file():
        return 'factored'
    if ((impute_dir / 'matrix.mtx.gz').is_file()
            and (impute_dir / 'regions.tsv.gz').is_file()):
        return 'mm'
    raise SystemExit(
        f'Cannot infer input kind under {impute_dir}. '
        'Expected csr (matrix_csr.npz+regions.tsv), dense, factored, or mm.'
    )


def per_row_aggregate_factored(impute_dir: Path, aggregate: str,
                               chunk_rows: int) -> np.ndarray:
    zpath = impute_dir / 'factors.npz'
    with np.load(zpath) as z:
        W = np.asarray(z['W'], dtype=np.float64)
        H = np.asarray(z['H'], dtype=np.float64)
        n_cells = int(H.shape[1])
        if W.shape[1] != H.shape[0]:
            raise SystemExit(f'Rank mismatch W{W.shape} vs H{H.shape}')
        if aggregate == 'max':
            r = int(W.shape[0])
            out = np.zeros(r, dtype=np.float64)
            cr = max(500, int(chunk_rows))
            pcf_vec: np.ndarray | None = None
            pcf_scalar: float | None = None
            if 'per_cell_factor' in z.files:
                pcf_raw = z['per_cell_factor']
                if np.ndim(pcf_raw) > 0:
                    pcf_vec = np.asarray(pcf_raw, dtype=np.float64).ravel()
                    if pcf_vec.shape[0] != n_cells:
                        raise SystemExit(
                            f'per_cell_factor length {pcf_vec.shape[0]} != H columns {n_cells}'
                        )
                else:
                    pcf_scalar = float(pcf_raw)
            for s in range(0, r, cr):
                e = min(s + cr, r)
                blk = (W[s:e] @ H).astype(np.float64, copy=False)
                if pcf_vec is not None:
                    blk *= pcf_vec[np.newaxis, :]
                elif pcf_scalar is not None:
                    blk *= pcf_scalar
                out[s:e] = blk.max(axis=1)
            return out.astype(np.float64)

        if 'per_cell_factor' in z.files:
            pcf_raw = z['per_cell_factor']
            if np.ndim(pcf_raw) > 0:
                pcf = np.asarray(pcf_raw, dtype=np.float64).ravel()
                if pcf.shape[0] != n_cells:
                    raise SystemExit(
                        f'per_cell_factor length {pcf.shape[0]} != H columns {n_cells}'
                    )
                vec = H @ pcf
            else:
                vec = H.sum(axis=1) * float(pcf_raw)
        else:
            vec = H.sum(axis=1)
    scores = (W @ vec).ravel()
    if aggregate == 'mean' and n_cells > 0:
        scores = scores / float(n_cells)
    return scores.astype(np.float64)


def per_row_aggregate_dense(impute_dir: Path, aggregate: str,
                            chunk_cols: int) -> np.ndarray:
    arr = np.load(impute_dir / 'matrix.npy', mmap_mode='r')
    n, c = arr.shape
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, c, chunk_cols):
        e = min(s + chunk_cols, c)
        block = np.asarray(arr[:, s:e], dtype=np.float64)
        out += block.sum(axis=1)
    if aggregate == 'mean' and c > 0:
        out /= float(c)
    return out


def per_row_aggregate_csr(impute_dir: Path, aggregate: str,
                          chunk_rows: int) -> np.ndarray:
    from scipy import sparse

    sm = sparse.load_npz(impute_dir / 'matrix_csr.npz')
    n, c = sm.shape
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        blk = sm[s:e]
        out[s:e] = np.asarray(blk.sum(axis=1)).ravel()
    if aggregate == 'mean' and c > 0:
        out /= float(c)
    return out


def per_row_aggregate_mm(impute_dir: Path, aggregate: str,
                         chunk_cols: int) -> np.ndarray:
    import scipy.io as sio
    import scipy.sparse as sp

    mat = sio.mmread(str(impute_dir / 'matrix.mtx.gz')).tocsr()
    n, c = mat.shape
    out = np.zeros(n, dtype=np.float64)
    for s in range(0, c, chunk_cols):
        e = min(s + chunk_cols, c)
        out += np.asarray(mat[:, s:e].sum(axis=1)).ravel()
    if aggregate == 'mean' and c > 0:
        out /= float(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True,
                    help='Impute (or mm) directory')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output .bw path')
    ap.add_argument('--chrom-sizes', type=Path, required=True,
                    help='UCSC chrom.sizes (tab: chrom length)')
    ap.add_argument('--aggregate', choices=('sum', 'mean', 'max'), default='sum',
                    help='Per-bin score: sum or mean across cells; max = per-bin max '
                         'over cells (factored inputs only; peakier IGV). Default: sum.')
    ap.add_argument('--chunk-cols', type=int, default=500,
                    help='Matrix columns per chunk for dense/mm (memory)')
    ap.add_argument('--chunk-rows', type=int, default=10_000,
                    help='CSR rows per chunk')
    ap.add_argument('--transform', choices=('none', 'log1p'), default='none',
                    help='Display-only transform applied to scores before writing '
                         '(log1p: use for long-tailed factored pseudobulk in IGV). '
                         'Default: none.')
    ap.add_argument('--normalize', choices=('none', 'max', 'p99'), default='none',
                    help='Display-only scaling after chrom filter: divide all scores '
                         'by global max (max) or by the 99th percentile (p99, robust). '
                         'Use for multi-track IGV when pipelines use different units. '
                         'Default: none.')
    ap.add_argument('--center', choices=('none', 'median', 'p25'), default='none',
                    help='After chrom filter, subtract genome-wide median (or p25) '
                         'then clip at 0 (display-only; removes cisTopic "plateau" '
                         'floor in IGV). Applied before --normalize / --transform.')
    args = ap.parse_args()

    try:
        import pyBigWig  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit('Install pyBigWig: pip install pyBigWig') from exc

    impute_dir = args.input.expanduser().resolve()
    if not impute_dir.is_dir():
        raise SystemExit(f'--input must be a directory: {impute_dir}')

    kind = detect_kind(impute_dir)
    log.info('Detected input kind: %s', kind)

    if args.aggregate == 'max' and kind != 'factored':
        raise SystemExit('--aggregate max is only supported for factored (factors.npz) inputs.')

    if kind == 'factored':
        scores = per_row_aggregate_factored(impute_dir, args.aggregate, args.chunk_rows)
        reg_path = impute_dir / 'regions.tsv'
    elif kind == 'dense':
        scores = per_row_aggregate_dense(impute_dir, args.aggregate, args.chunk_cols)
        reg_path = impute_dir / 'regions.tsv'
    elif kind == 'csr':
        scores = per_row_aggregate_csr(impute_dir, args.aggregate, args.chunk_rows)
        reg_path = impute_dir / 'regions.tsv'
    else:
        scores = per_row_aggregate_mm(impute_dir, args.aggregate, args.chunk_cols)
        reg_path = impute_dir / 'regions.tsv.gz'

    chroms, starts, ends = parse_regions(read_lines_maybe_gz(reg_path))
    if len(chroms) != int(scores.shape[0]):
        raise SystemExit(
            f'regions ({len(chroms)}) vs scores ({scores.shape[0]}): length mismatch'
        )
    smin = float(np.min(scores))
    smax = float(np.max(scores))
    smed = float(np.median(scores))
    log.info('Per-interval scores (pre-filter): min=%.6g median=%.6g max=%.6g',
             smin, smed, smax)

    sizes = load_chrom_sizes(args.chrom_sizes.expanduser().resolve())
    size_map = dict(sizes)
    allowed = set(size_map)

    df = pd.DataFrame({
        'chrom': chroms,
        'start': starts,
        'end': ends,
        'score': scores.astype(np.float64),
    })
    bad = ~df['chrom'].isin(allowed)
    if bad.any():
        n_bad = int(bad.sum())
        log.warning('Dropping %d intervals on chromosomes not in chrom.sizes', n_bad)
        df = df.loc[~bad].copy()
    if df.empty:
        raise SystemExit('No intervals left after chrom.sizes filter.')

    if args.center != 'none':
        s = df['score'].to_numpy(dtype=np.float64, copy=False)
        ref = float(np.median(s)) if args.center == 'median' else float(np.percentile(s, 25.0))
        df['score'] = np.clip(s - ref, 0.0, None)
        log.info(
            'Applied --center %s (ref=%.6g): min=%.6g median=%.6g max=%.6g',
            args.center, ref,
            float(df['score'].min()),
            float(df['score'].median()),
            float(df['score'].max()),
        )

    if args.normalize != 'none':
        s = df['score'].to_numpy(dtype=np.float64, copy=False)
        if args.normalize == 'max':
            denom = float(np.max(s))
        else:
            denom = float(np.percentile(s, 99.0))
        eps = 1e-30
        if denom <= eps:
            log.warning('--normalize %s: denominator ~0; leaving scores unchanged',
                        args.normalize)
        else:
            df['score'] = s / denom
            log.info('Applied --normalize %s (denom=%.6g): min=%.6g median=%.6g max=%.6g',
                     args.normalize, denom,
                     float(df['score'].min()),
                     float(df['score'].median()),
                     float(df['score'].max()))

    if args.transform == 'log1p':
        df['score'] = np.log1p(np.maximum(df['score'].to_numpy(), 0.0))
        log.info(
            'Applied log1p: score min=%.6g median=%.6g max=%.6g',
            float(df['score'].min()),
            float(df['score'].median()),
            float(df['score'].max()),
        )

    chrom_order = [c for c, _ in sizes]
    df['chrom'] = pd.Categorical(df['chrom'], categories=chrom_order, ordered=True)
    df.sort_values(['chrom', 'start', 'end'], inplace=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    bw = pyBigWig.open(str(args.output), 'w')
    try:
        bw.addHeader(sizes)
        bw.addEntries(
            df['chrom'].astype(str).tolist(),
            df['start'].astype(int).tolist(),
            ends=df['end'].astype(int).tolist(),
            values=df['score'].astype(float).tolist(),
        )
    finally:
        bw.close()

    log.info('Wrote %s (%d intervals)', args.output, len(df))
    return 0


if __name__ == '__main__':
    sys.exit(main())
