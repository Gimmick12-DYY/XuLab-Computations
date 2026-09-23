#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# chromhmm_nsplit_sweep.py
#
# Build ChromHMM-18 × N E1-quantile RPKM matrices (N = 2, 3, 5, 10) and the
# TF×TF Pearson correlation figure used to recover cobinding complexes.
# Each TF's pseudobulk (raw mm or imputed CSR) is loaded once and scored
# against every split.
#
#   python tf_complex/scripts/chromhmm_nsplit_sweep.py
#   python tf_complex/scripts/chromhmm_nsplit_sweep.py --source imputed
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
TFC = ROOT / "tf_complex"
_DOWN = ROOT / "downstream"
if str(_DOWN) not in sys.path:
    sys.path.insert(0, str(_DOWN))
if str(TFC / "scripts") not in sys.path:
    sys.path.insert(0, str(TFC / "scripts"))

from build_chromhmm_matrix import (  # noqa: E402
    discover_tfs, load_chromhmm, load_per_bin_signal, parse_region,
)


def accumulate(signal, regions, seg_idx, sid, n_states):
    reads = np.zeros(n_states, dtype=np.float64)
    for i, r in enumerate(regions):
        v = signal[i]
        if v == 0:
            continue
        c, s = parse_region(r)
        if c not in seg_idx:
            continue
        starts, ends, labs = seg_idx[c]
        k = int(np.searchsorted(starts, s, side="right")) - 1
        if k >= 0 and s < ends[k]:
            reads[sid[labs[k]]] += v
    return reads


def write_matrix(out_dir: Path, states, state_bp, tfs, cols):
    out_dir.mkdir(parents=True, exist_ok=True)
    M = sp.hstack(cols).tocsc()
    sp.save_npz(out_dir / "chromhmm_matrix.npz", M)
    (out_dir / "tfs.txt").write_text("\n".join(tfs) + "\n")
    with (out_dir / "states.tsv").open("w") as f:
        for s in states:
            f.write(f"{s}\t{state_bp[s]}\n")
    print(f"[write] {M.shape[0]} states x {M.shape[1]} TFs -> {out_dir/'chromhmm_matrix.npz'}",
          flush=True)


def correlate(matrix_dir: Path, results: Path, cell_meta: Path, n_clusters: int,
              vmin: str = "0.65", tag: str = ""):
    results.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TFC / "scripts" / "chromhmm_correlation.py"),
        "--matrix-dir", str(matrix_dir),
        "--out-dir", str(results),
        "--transform", "rpkm",
        "--n-clusters", str(n_clusters),
        "--cell-meta", str(cell_meta),
        "--vmin", str(vmin),
    ]
    if tag:
        cmd.extend(["--tag", tag])
    print(f"[corr] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,3,5,10")
    ap.add_argument("--work-root", type=Path, default=ROOT / "unified" / "work")
    ap.add_argument("--chromhmm-bed", type=Path,
                    default=ROOT / "data" / "HEK293T_chromHMM18.bed.gz")
    ap.add_argument("--ab-bed", type=Path,
                    default=ROOT / "hic" / "work" / "compartments" / "compartments_25000.AB.bed")
    ap.add_argument("--cell-meta", type=Path,
                    default=ROOT / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--n-clusters", type=int, default=0,
                    help="0 = silhouette-chosen k")
    ap.add_argument("--source", choices=["raw", "imputed"], default="raw")
    ap.add_argument("--residualize", action="store_true",
                    help="imputed only: subtract consensus accessibility "
                         "(residual = max(0, signal - expected)) before RPKM")
    ap.add_argument("--consensus-scale", type=float, default=1.0)
    ap.add_argument("--force-matrix", action="store_true")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    if args.residualize and args.source != "imputed":
        raise SystemExit("--residualize requires --source imputed")
    tag = "" if args.source == "raw" else "_imputed"
    if args.residualize:
        tag = "_imputed_tfspec"
    sub = "mm" if args.source == "raw" else "impute"
    matfile = "matrix.mtx.gz" if args.source == "raw" else "matrix_csr.npz"

    schemes = {}
    need_build = []
    for n in ns:
        split_dir = TFC / "work" / f"chromhmm_x{n}"
        matrix_dir = TFC / "work" / f"chromhmm_x{n}{tag}"
        split_bed = split_dir / f"chromhmm18_x{n}.bed"
        if not split_bed.is_file():
            print(f"[split] N={n} {split_bed}", flush=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, str(TFC / "scripts" / "split_chromhmm_states.py"),
                 "--chromhmm-bed", str(args.chromhmm_bed),
                 "--ab-bed", str(args.ab_bed),
                 "--n-split", str(n),
                 "--out", str(split_bed)],
                check=True,
            )
        matrix_dir.mkdir(parents=True, exist_ok=True)
        npz = matrix_dir / "chromhmm_matrix.npz"
        if args.force_matrix or not npz.is_file():
            need_build.append(n)
        schemes[n] = {"matrix": matrix_dir, "bed": split_bed}
        print(f"[plan] N={n} source={args.source}  bed={split_bed.is_file()}  "
              f"matrix={npz.is_file()}  build={'yes' if n in need_build else 'reuse'}",
              flush=True)

    if need_build:
        loaded = {}
        for n in need_build:
            seg_idx, state_bp = load_chromhmm(schemes[n]["bed"])
            states = list(state_bp.keys())
            loaded[n] = {
                "seg_idx": seg_idx,
                "state_bp": state_bp,
                "states": states,
                "sid": {s: i for i, s in enumerate(states)},
                "bp": np.array([state_bp[s] for s in states], dtype=np.float64),
                "cols": [],
            }
            print(f"[chromhmm] N={n}  {len(states)} units, {int(loaded[n]['bp'].sum()):,} bp",
                  flush=True)

        tfs = discover_tfs(args.work_root, sub, matfile)
        tracks = []  # (tf, signal)
        regions = None
        for tf in tfs:
            d = args.work_root / tf / sub
            if not (d / matfile).is_file():
                print(f"[skip] {tf}: no {sub} {matfile}", file=sys.stderr)
                continue
            signal, regs, kind, ncells = load_per_bin_signal(d)
            if regions is None:
                regions = regs
            elif len(regs) != len(regions):
                print(f"[skip] {tf}: bin count {len(regs)} != {len(regions)}",
                      file=sys.stderr)
                continue
            tracks.append((tf, signal, kind, ncells))
            print(f"[{len(tracks)}] {tf}: {kind} ncells={ncells} "
                  f"sum={float(signal.sum()):.4g}", flush=True)

        if not tracks:
            raise SystemExit("no TF matrices loaded")

        if args.residualize:
            acc = np.zeros(len(regions), dtype=np.float64)
            n_ok = 0
            for tf, signal, kind, ncells in tracks:
                tot = float(signal.sum())
                if tot <= 0:
                    continue
                acc += signal / tot
                n_ok += 1
            cons = acc / float(n_ok)
            cons_dir = TFC / "work" / "chromhmm_imputed_tfspec"
            cons_dir.mkdir(parents=True, exist_ok=True)
            np.save(cons_dir / "consensus_accessibility.npy", cons)
            print(f"[consensus] {n_ok} TFs, {len(regions)} bins, "
                  f"nonzero={int(np.count_nonzero(cons))}", flush=True)
            residual_tracks = []
            for tf, signal, kind, ncells in tracks:
                tot = float(signal.sum())
                residual = np.clip(signal - args.consensus_scale * cons * tot, 0.0, None)
                kept = float(residual.sum()) / tot if tot > 0 else 0.0
                n_pos = int(np.count_nonzero(residual))
                print(f"[resid] {tf}: {n_pos:,} bins  kept {100 * kept:.1f}% of counts",
                      flush=True)
                if residual.sum() <= 0:
                    print(f"[skip] {tf}: residual all zero", file=sys.stderr)
                    continue
                residual_tracks.append((tf, residual, kind, ncells))
            tracks = residual_tracks
            del cons, acc

        used = []
        for tf, signal, kind, ncells in tracks:
            used.append(tf)
            for n in need_build:
                L = loaded[n]
                reads = accumulate(signal, regions, L["seg_idx"], L["sid"], len(L["states"]))
                libM = reads.sum() / 1e6
                rpkm = reads / (L["bp"] / 1e3) / (libM if libM > 0 else 1.0)
                L["cols"].append(sp.csc_matrix(rpkm.reshape(-1, 1)))
            print(f"[rpkm] {tf}", flush=True)

        if not used:
            raise SystemExit("no TF matrices loaded")
        for n in need_build:
            L = loaded[n]
            write_matrix(schemes[n]["matrix"], L["states"], L["state_bp"], used, L["cols"])

    vmin = "auto" if args.residualize else "0.65"
    for n in ns:
        ntag = f"chromHMM18×{n} tf-spec" if args.residualize else ""
        correlate(
            schemes[n]["matrix"],
            TFC / f"results_chromhmm_x{n}{tag}",
            args.cell_meta,
            args.n_clusters,
            vmin=vmin,
            tag=ntag,
        )
    print("[done] sweep", ns, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
