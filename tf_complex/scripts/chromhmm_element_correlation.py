#!/usr/bin/env python3
# ChromHMM element correlation.
# Unit = one interval from the ChromHMM BED (not the 18 pooled states).
# Signal = plain pseudobulk reads in the 1 kb bin whose start falls in the
# interval, then divided by that TF's cell count. Pearson is unchanged by the
# per-TF cell-count scale. SpQN then orders TFs by the same cell counts.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_TFC = Path(__file__).resolve().parents[1]
_DOWN = _TFC.parent / "downstream"
sys.path.insert(0, str(_TFC / "scripts"))
sys.path.insert(0, str(_DOWN))

from compartment_rpkm_correlation import load_cells, spqn, write_matrix, _confound  # noqa: E402
from peak_coverage import load_per_bin_signal  # noqa: E402
from plot_correlation_matrix import linkage_from_pearson, plot_correlation_matrix  # noqa: E402


def load_elements(path: Path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            rows.append((p[0], int(p[1]), int(p[2]), p[3]))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def element_index(rows):
    by = {}
    for i, (c, s, e, _st) in enumerate(rows):
        by.setdefault(c, []).append((s, e, i))
    idx = {}
    for c, lst in by.items():
        idx[c] = (
            np.array([x[0] for x in lst], dtype=np.int64),
            np.array([x[1] for x in lst], dtype=np.int64),
            np.array([x[2] for x in lst], dtype=np.int64),
        )
    return idx


def parse_region(name: str):
    chrom, rest = name.split(":")
    return chrom, int(rest.split("-")[0])


def assign_bins(signal, regions, idx, n_elem):
    reads = np.zeros(n_elem, dtype=np.float64)
    for i, name in enumerate(regions):
        v = signal[i]
        if v == 0:
            continue
        chrom, start = parse_region(name)
        if chrom not in idx:
            continue
        starts, ends, gidx = idx[chrom]
        k = int(np.searchsorted(starts, start, side="right") - 1)
        if k >= 0 and start < ends[k]:
            reads[gidx[k]] += v
    return reads


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    root = _TFC.parent
    ap.add_argument("--bed", type=Path, default=root / "data" / "HEK293T_chromHMM18.bed.gz")
    ap.add_argument("--work-root", type=Path, default=root / "unified" / "work")
    ap.add_argument("--tfs", type=Path,
                    default=_TFC / "results_compartment_counts_spqn_domain_sm2" / "tf_similarity_genome.tsv")
    ap.add_argument("--cell-meta", type=Path, default=root / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--matrix-dir", type=Path, default=_TFC / "work" / "chromhmm_elements_counts")
    ap.add_argument("--out-dir", type=Path, default=_TFC / "results_chromhmm_elements_counts_spqn")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    header = args.tfs.read_text().splitlines()[0].split("\t")
    tfs = [t.strip() for t in header[1:] if t.strip()]
    cells = load_cells(args.cell_meta)
    n_cells = np.array([cells.get(t.lower(), 0) for t in tfs], dtype=np.float64)
    missing = [t for t, n in zip(tfs, n_cells) if n <= 0]
    if missing:
        raise SystemExit(f"no cell count for: {', '.join(missing)}")

    mat_path = args.matrix_dir / "element_matrix.npz"
    if args.rebuild or not mat_path.is_file():
        rows = load_elements(args.bed)
        idx = element_index(rows)
        print(f"[elements] {len(rows):,}", flush=True)
        cols = []
        for j, tf in enumerate(tfs):
            signal, regions, kind, _n = load_per_bin_signal(args.work_root / tf / "mm")
            reads = assign_bins(signal, regions, idx, len(rows))
            cols.append(reads / n_cells[j])
            print(f"[{j+1}/{len(tfs)}] {tf} reads={reads.sum():.0f} cells={int(n_cells[j])} kind={kind}",
                  flush=True)
        M = np.column_stack(cols).astype(np.float32)
        args.matrix_dir.mkdir(parents=True, exist_ok=True)
        sp.save_npz(mat_path, sp.csc_matrix(M))
        (args.matrix_dir / "tfs.txt").write_text("\n".join(tfs) + "\n")
        with (args.matrix_dir / "elements.tsv").open("w") as fh:
            fh.write("chrom\tstart\tend\tstate\tbp\n")
            for c, s, e, st in rows:
                fh.write(f"{c}\t{s}\t{e}\t{st}\t{e-s}\n")
        (args.matrix_dir / "norm.txt").write_text(
            "reads per cell: plain pseudobulk reads in the element / n_cells. No /kb.\n"
        )
        print(f"[matrix] {M.shape} -> {mat_path}", flush=True)
    else:
        M = np.asarray(sp.load_npz(mat_path).todense())
        print(f"[matrix] loaded {M.shape}", flush=True)

    keep = np.asarray(M.sum(axis=1)).ravel() > 0
    X = M[keep]
    print(f"[keep] {int(keep.sum()):,}/{M.shape[0]:,} elements with any read", flush=True)
    R = np.nan_to_num(np.corrcoef(X, rowvar=False))
    depth = n_cells
    Rn = spqn(R, depth)
    off = ~np.eye(len(tfs), dtype=bool)
    print(f"[pearson] elements={X.shape[0]:,} median={np.median(R[off]):+.4f} "
          f"confound={_confound(R, depth):+.3f}", flush=True)
    print(f"[spqn] median={np.median(Rn[off]):+.4f} confound={_confound(Rn, depth):+.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.out_dir / "tf_similarity_genome.tsv", R, tfs)
    write_matrix(args.out_dir / "spqn_similarity_genome.tsv", Rn, tfs)
    Z0 = linkage_from_pearson(R)
    Z1 = linkage_from_pearson(Rn)
    plot_correlation_matrix(
        R, tfs, args.out_dir / "chromhmm_elements_per_cell_genome.png",
        cells=cells, color="percentile", Z=Z0,
        title="ChromHMM elements\nreads / cell\nPearson r\npercentile\n(genome)",
    )
    plot_correlation_matrix(
        Rn, tfs, args.out_dir / "chromhmm_elements_spqn_genome.png",
        cells=cells, color="percentile", Z=Z1,
        title="ChromHMM elements\nreads / cell\n+ SpQN\npercentile\n(genome)",
    )
    plot_correlation_matrix(
        Rn, tfs, args.out_dir / "chromhmm_elements_spqn_genome_r.png",
        cells=cells, color="r", Z=Z1,
        title="ChromHMM elements\nreads / cell\n+ SpQN\npearson cor\n(genome)",
    )
    print(f"[done] {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
