#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# compartment_imputed_run.py
#
# A/B compartment RPKM Pearson on imputed data (domain, SMOOTH=2), plus the
# consensus-subtracted (tf-spec) version. One load pass per TF fills both
# matrices, then compartment_rpkm_correlation.py plots genome / A / B.
#
#   python tf_complex/scripts/compartment_imputed_run.py
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
if str(TFC / "scripts") not in sys.path:
    sys.path.insert(0, str(TFC / "scripts"))
from build_compartment_matrix import (  # noqa: E402
    discover_tfs, domain_index, load_domains, load_per_bin_signal, parse_region,
)


def accumulate(signal, regions, didx, nD):
    acc = np.zeros(nD, dtype=np.float64)
    for i, r in enumerate(regions):
        v = signal[i]
        if v == 0:
            continue
        c, s = parse_region(r)
        if c not in didx:
            continue
        starts, ends, gidx = didx[c]
        k = int(np.searchsorted(starts, s, side="right")) - 1
        if k >= 0 and s < ends[k]:
            acc[gidx[k]] += v
    return acc


def to_rpkm(acc, bp):
    libM = float(acc.sum()) / 1e6
    return acc / (bp / 1e3) / (libM if libM > 0 else 1.0)


def write_matrix(out_dir: Path, cols, used, domains):
    out_dir.mkdir(parents=True, exist_ok=True)
    M = sp.hstack(cols).tocsc()
    sp.save_npz(out_dir / "compartment_matrix.npz", M)
    (out_dir / "tfs.txt").write_text("\n".join(used) + "\n")
    with (out_dir / "domains.tsv").open("w") as f:
        for c, s, e, lab in domains:
            f.write(f"{c}\t{s}\t{e}\t{lab}\t{e - s}\n")
    print(f"[write] {M.shape[0]:,} domains x {M.shape[1]} TFs -> "
          f"{out_dir / 'compartment_matrix.npz'}", flush=True)


def correlate(matrix_dir: Path, results: Path, cell_meta: Path, n_clusters: int):
    results.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TFC / "scripts" / "compartment_rpkm_correlation.py"),
        "--matrix-dir", str(matrix_dir),
        "--out-dir", str(results),
        "--transform", "rpkm",
        "--n-clusters", str(n_clusters),
        "--cell-meta", str(cell_meta),
    ]
    print(f"[corr] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=ROOT / "unified" / "work")
    ap.add_argument(
        "--domains-bed", type=Path,
        default=TFC / "work" / "compartment_rpkm_domain_sm2" /
        "compartments_smoothed.domains.bed",
    )
    ap.add_argument(
        "--consensus-npy", type=Path,
        default=TFC / "work" / "chromhmm_imputed_tfspec" / "consensus_accessibility.npy",
    )
    ap.add_argument("--consensus-scale", type=float, default=1.0)
    ap.add_argument("--n-clusters", type=int, default=3)
    ap.add_argument("--cell-meta", type=Path,
                    default=ROOT / "data" / "TF1000cells.meta.csv")
    ap.add_argument("--skip-plain", action="store_true")
    ap.add_argument("--skip-tfspec", action="store_true")
    args = ap.parse_args()
    if not args.domains_bed.is_file():
        raise SystemExit(f"missing smoothed domains: {args.domains_bed}")

    domains = load_domains(args.domains_bed, "domain")
    didx = domain_index(domains)
    nD = len(domains)
    bp = np.array([e - s for _, s, e, _ in domains], dtype=np.float64)
    bp[bp <= 0] = 1.0
    print(f"[domains] {nD:,}  A={sum(d[3]=='A' for d in domains):,}  "
          f"B={sum(d[3]=='B' for d in domains):,}", flush=True)

    cons = None
    if not args.skip_tfspec:
        if not args.consensus_npy.is_file():
            raise SystemExit(f"missing consensus: {args.consensus_npy}")
        cons = np.load(args.consensus_npy)
        print(f"[consensus] {cons.shape[0]} bins", flush=True)

    tfs = discover_tfs(args.work_root, "impute", "matrix_csr.npz")
    cols_imp, cols_res, used = [], [], []
    for tf in tfs:
        d = args.work_root / tf / "impute"
        if not (d / "matrix_csr.npz").is_file():
            continue
        signal, regions, kind, ncells = load_per_bin_signal(d)
        tot = float(signal.sum())
        if not args.skip_plain:
            cols_imp.append(sp.csc_matrix(to_rpkm(accumulate(signal, regions, didx, nD), bp).reshape(-1, 1)))
        if cons is not None:
            if cons.shape[0] != signal.shape[0]:
                raise SystemExit(f"{tf}: bins {signal.shape[0]} != consensus {cons.shape[0]}")
            residual = np.clip(signal - args.consensus_scale * cons * tot, 0.0, None)
            kept = float(residual.sum()) / tot if tot > 0 else 0.0
            cols_res.append(sp.csc_matrix(
                to_rpkm(accumulate(residual, regions, didx, nD), bp).reshape(-1, 1)))
            print(f"[{len(used)+1}] {tf}: residual kept {100*kept:.1f}%", flush=True)
        else:
            print(f"[{len(used)+1}] {tf}: {kind} ncells={ncells}", flush=True)
        used.append(tf)

    if not used:
        raise SystemExit("no imputed TFs")
    out_imp = TFC / "work" / "compartment_rpkm_domain_sm2_imputed"
    out_res = TFC / "work" / "compartment_rpkm_domain_sm2_imputed_tfspec"
    if not args.skip_plain:
        write_matrix(out_imp, cols_imp, used, domains)
        correlate(out_imp, TFC / "results_compartment_rpkm_domain_sm2_imputed",
                  args.cell_meta, args.n_clusters)
    if cols_res:
        write_matrix(out_res, cols_res, used, domains)
        correlate(out_res, TFC / "results_compartment_rpkm_domain_sm2_imputed_tfspec",
                  args.cell_meta, args.n_clusters)
    print("[done] compartment imputed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
