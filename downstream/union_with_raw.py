#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# union_with_raw.py
#
# Build a full-mm CSR matrix
#     union[r, c] = max(raw[r, c], imputed_soft[r, c] * 1[imputed > t])
# so observed fragments are never removed; thresholded imputation only adds
# on zeros. Use this script for non–cisTopic pipelines or to post-process a
# saved imputed CSR; cisTopic step 05 writes thresholded imputation only
# (no max with raw).
#
# Required:
#   --mm-dir       raw MatrixMarket + regions.tsv.gz + barcodes.tsv.gz
#   --impute-dir   directory with one of: matrix_csr.npz, factors.npz, matrix.npy
#   --out-dir      writes <out-dir>/<label>/matrix_csr.npz, regions.tsv,
#                  barcodes.tsv, meta.json
#
# Threshold (imputed soft values → sparse calls before max with raw):
#   --from-meta    use impute-dir/meta.json "threshold" if present
#   --threshold T  fixed cutoff (overrides --from-meta)
#   --threshold-mode sparsity_match[:K] | quantile:Q | fixed:T
#                  default: sparsity_match with same K semantics as 05_impute
#                  when union is applied *here*: #(imp>t) target = (K-1)*raw_nnz
#                  on the modeled subspace (addition mode). If union is disabled
#                  below, target = K*raw_nnz (replacement).
#
#   --no-union     only threshold to CSR (no max with raw); for debugging.
#   --addition-mode When using sparsity_match, use (K-1)*raw_nnz as #(imp>t)
#                  target before union (default true). False → K*raw_nnz then
#                  still take max(raw, imp) unless --no-union.
#
# factored/ inputs: region_topic_phi.parquet + cell_topic_theta.parquet (cisTopic),
# or factors.npz + regions.tsv(.gz) + barcodes.tsv(.gz) with row/col order matching
# W/H (e.g. scOpen 05_impute).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

log = logging.getLogger("union_with_raw")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _read_lines_either(dir_path: Path, base: str) -> list[str] | None:
    for name in (f"{base}.tsv", f"{base}.tsv.gz"):
        p = dir_path / name
        if p.exists():
            return _read_lines(p) if p.suffix != ".gz" else _read_lines_gz(p)
    return None


def load_raw_mm(mm_dir: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    mtx = mm_dir / "matrix.mtx.gz"
    reg = mm_dir / "regions.tsv.gz"
    bar = mm_dir / "barcodes.tsv.gz"
    for p in (mtx, reg, bar):
        if not p.exists():
            raise SystemExit(f"Missing {p}")
    mat = sio.mmread(str(mtx)).tocsr().astype(np.float32)
    regs = _read_lines_gz(reg)
    bars = _read_lines_gz(bar)
    if mat.shape != (len(regs), len(bars)):
        raise SystemExit(f"raw shape {mat.shape} vs {len(regs)}/{len(bars)} labels")
    return mat, regs, bars


def detect_kind(impute_dir: Path) -> str:
    if (impute_dir / "matrix_csr.npz").exists():
        return "csr"
    if (impute_dir / "factors.npz").exists():
        return "factored"
    if (impute_dir / "matrix.npy").exists():
        return "dense"
    raise SystemExit(f"Cannot detect impute kind in {impute_dir}")


def factored_modeled_labels(
    impute_dir: Path,
    imp_regs: list[str] | None,
    imp_bars: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Row/cell names for the modeled (W @ H) subspace.

    Prefer cisTopic parquets when present; otherwise require regions.tsv +
    barcodes.tsv whose lengths match factors.npz W and H.
    """
    rp = impute_dir / "region_topic_phi.parquet"
    cp = impute_dir / "cell_topic_theta.parquet"
    if rp.exists() and cp.exists():
        modeled_regs = pd.read_parquet(rp).index.astype(str).tolist()
        modeled_cells = pd.read_parquet(cp).columns.astype(str).tolist()
        return modeled_regs, modeled_cells

    if imp_regs is None or imp_bars is None:
        raise SystemExit(
            f"factored layout in {impute_dir}: add region_topic_phi.parquet and "
            "cell_topic_theta.parquet, or provide regions.tsv(.gz) and "
            "barcodes.tsv(.gz) aligned to factors.npz (W rows, H cols)."
        )
    with np.load(impute_dir / "factors.npz") as z:
        if "W" not in z.files or "H" not in z.files:
            raise SystemExit(
                f"factors.npz in {impute_dir} must contain W and H arrays."
            )
        w = np.asarray(z["W"])
        h = np.asarray(z["H"])
        nr_w, nc_h = int(w.shape[0]), int(h.shape[1])
    if len(imp_regs) != nr_w or len(imp_bars) != nc_h:
        raise SystemExit(
            f"factors.npz W/H modeled shape ({nr_w}, {nc_h}) vs regions/barcodes "
            f"({len(imp_regs)}, {len(imp_bars)}) mismatch in {impute_dir}."
        )
    return imp_regs, imp_bars


def parse_threshold_mode(s: str) -> tuple[str, float]:
    s = (s or "sparsity_match:2").strip()
    if s == "sparsity_match":
        return "sparsity_match", 1.0
    if s.startswith("sparsity_match:"):
        return "sparsity_match", float(s.split(":", 1)[1])
    if s.startswith("quantile:"):
        q = float(s.split(":", 1)[1])
        if not 0.0 <= q <= 1.0:
            raise SystemExit("quantile must be in [0,1]")
        return "quantile", q
    if s.startswith("fixed:"):
        return "fixed", float(s.split(":", 1)[1])
    raise SystemExit(f"Unknown threshold_mode {s!r}")


def sample_imp(phi: np.ndarray, theta: np.ndarray, scale: float, n: int,
               rng: np.random.Generator) -> np.ndarray:
    R, _ = phi.shape
    _, C = theta.shape
    n = int(min(n, R * C))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rr = rng.integers(0, R, size=n, dtype=np.int64)
    cc = rng.integers(0, C, size=n, dtype=np.int64)
    return (np.einsum("ik,ki->i", phi[rr].astype(np.float64),
                      theta[:, cc].astype(np.float64)) * float(scale))


def sample_dense(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    R, C = arr.shape
    n = int(min(n, R * C))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rr = rng.integers(0, R, size=n, dtype=np.int64)
    cc = rng.integers(0, C, size=n, dtype=np.int64)
    return np.asarray(arr[rr, cc], dtype=np.float64)


def pick_threshold(phi: np.ndarray, theta: np.ndarray, scale: float,
                   mode: str, param: float, sparsity_target: int,
                   sample_size: int, rng: np.random.Generator) -> tuple[float, str]:
    if mode == "fixed":
        return float(param), f"fixed:{param}"
    samp = sample_imp(phi, theta, scale, sample_size, rng)
    if samp.size == 0:
        return 0.0, "empty_sample"
    R, C = phi.shape[0], theta.shape[1]
    tot = int(R) * int(C)
    if mode == "sparsity_match":
        K = float(param)
        tgt = int(sparsity_target)
        if tgt <= 0:
            return float(np.max(samp) * (1.0 + 1e-9)), f"sparsity_match:{K} target=0"
        tgt = max(1, min(tgt, tot))
        q = max(0.0, min(1.0, 1.0 - tgt / tot))
        t = float(np.quantile(samp, q))
        return t, f"sparsity_match:{K} target_calls={tgt} q={q:.6g}"
    if mode == "quantile":
        q = float(param)
        return float(np.quantile(samp, q)), f"quantile:{q}"
    raise SystemExit(f"Unknown mode {mode}")


def pick_threshold_dense(arr: np.ndarray, mode: str, param: float,
                         sparsity_target: int, sample_size: int,
                         rng: np.random.Generator) -> tuple[float, str]:
    if mode == "fixed":
        return float(param), f"fixed:{param}"
    samp = sample_dense(arr, sample_size, rng)
    if samp.size == 0:
        return 0.0, "empty_sample"
    R, C = arr.shape[0], arr.shape[1]
    tot = int(R) * int(C)
    if mode == "sparsity_match":
        K = float(param)
        tgt = int(sparsity_target)
        if tgt <= 0:
            return float(np.max(samp) * (1.0 + 1e-9)), f"sparsity_match:{K} target=0"
        tgt = max(1, min(tgt, tot))
        q = max(0.0, min(1.0, 1.0 - tgt / tot))
        t = float(np.quantile(samp, q))
        return t, f"sparsity_match:{K} (dense sample) target_calls={tgt} q={q:.6g}"
    if mode == "quantile":
        q = float(param)
        return float(np.quantile(samp, q)), f"quantile:{q}"
    raise SystemExit(f"Unknown mode {mode}")


def csr_thresholded_from_csr(imp: sparse.csr_matrix, t: float) -> sparse.csr_matrix:
    imp = imp.tocsr().astype(np.float32, copy=True)
    mask = imp.data > t
    imp.data *= mask.astype(np.float32)
    imp.eliminate_zeros()
    return imp


def build_imp_csr_factored(
    impute_dir: Path,
    full_regs: list[str],
    full_bars: list[str],
    threshold: float,
    chunk_rows: int,
) -> sparse.csr_matrix:
    imp_regs = _read_lines_either(impute_dir, "regions")
    imp_bars = _read_lines_either(impute_dir, "barcodes")
    modeled_regs, modeled_cells = factored_modeled_labels(
        impute_dir, imp_regs, imp_bars,
    )

    z = np.load(impute_dir / "factors.npz")
    phi = np.asarray(z["W"], dtype=np.float32)
    theta = np.asarray(z["H"], dtype=np.float32)
    if "per_cell_factor" in z.files:
        pcf = z["per_cell_factor"]
        scale = float(pcf) if pcf.ndim == 0 else float(np.mean(pcf))
    else:
        scale = 1.0
    z.close()

    if phi.shape[0] != len(modeled_regs) or theta.shape[1] != len(modeled_cells):
        raise SystemExit("factors.npz shape vs modeled region/cell labels mismatch")

    nr, nc = len(full_regs), len(full_bars)
    rmap = {r: i for i, r in enumerate(full_regs)}
    cmap = {b: i for i, b in enumerate(full_bars)}
    try:
        mrow = np.array([rmap[r] for r in modeled_regs], dtype=np.int64)
        mcol = np.array([cmap[b] for b in modeled_cells], dtype=np.int64)
    except KeyError as e:
        raise SystemExit(f"Modeled name missing from mm: {e}") from e

    Rm = phi.shape[0]
    chunk = max(1, chunk_rows)
    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    vals_acc: list[np.ndarray] = []
    for s in range(0, Rm, chunk):
        e = min(s + chunk, Rm)
        blk = (phi[s:e] @ theta).astype(np.float32) * np.float32(scale)
        mask = blk > threshold
        if not mask.any():
            continue
        lr, lc = np.where(mask)
        rows_acc.append(mrow[s + lr])
        cols_acc.append(mcol[lc])
        vals_acc.append(blk[lr, lc].astype(np.float32, copy=False))
    if not rows_acc:
        return sparse.csr_matrix((nr, nc), dtype=np.float32)
    coo = sparse.coo_matrix(
        (
            np.concatenate(vals_acc),
            (np.concatenate(rows_acc), np.concatenate(cols_acc))),
        shape=(nr, nc),
        dtype=np.float32,
    )
    out = coo.tocsr()
    out.sum_duplicates()
    return out


def build_imp_csr_dense(
    impute_dir: Path,
    full_regs: list[str],
    full_bars: list[str],
    threshold: float,
    chunk_rows: int,
) -> sparse.csr_matrix:
    arr = np.load(impute_dir / "matrix.npy", mmap_mode="r")
    imp_regs = _read_lines_either(impute_dir, "regions")
    imp_bars = _read_lines_either(impute_dir, "barcodes")
    if imp_regs is None or imp_bars is None:
        raise SystemExit("dense impute needs regions.tsv + barcodes.tsv")
    if arr.shape != (len(imp_regs), len(imp_bars)):
        raise SystemExit("matrix.npy shape vs tsv mismatch")
    nr, nc = len(full_regs), len(full_bars)
    rmap = {r: i for i, r in enumerate(full_regs)}
    cmap = {b: i for i, b in enumerate(full_bars)}
    mrow = np.array([rmap[r] for r in imp_regs], dtype=np.int64)
    mcol = np.array([cmap[b] for b in imp_bars], dtype=np.int64)

    rows_acc: list[np.ndarray] = []
    cols_acc: list[np.ndarray] = []
    vals_acc: list[np.ndarray] = []
    for s in range(0, arr.shape[0], chunk_rows):
        e = min(s + chunk_rows, arr.shape[0])
        blk = np.asarray(arr[s:e], dtype=np.float32)
        mask = blk > threshold
        if not mask.any():
            continue
        lr, lc = np.where(mask)
        rows_acc.append(mrow[s + lr])
        cols_acc.append(mcol[lc])
        vals_acc.append(blk[lr, lc])
    if not rows_acc:
        return sparse.csr_matrix((nr, nc), dtype=np.float32)
    coo = sparse.coo_matrix(
        (
            np.concatenate(vals_acc),
            (np.concatenate(rows_acc), np.concatenate(cols_acc))),
        shape=(nr, nc),
        dtype=np.float32,
    )
    out = coo.tocsr()
    out.sum_duplicates()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mm-dir", type=Path, required=True)
    ap.add_argument("--impute-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", type=str, default="union")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--from-meta", action="store_true",
                    help="Use threshold from impute-dir/meta.json")
    ap.add_argument("--threshold-mode", type=str, default="sparsity_match:2")
    ap.add_argument("--addition-mode", action="store_true", default=True,
                    help="For sparsity_match, target #(imp>t)=(K-1)*raw_nnz "
                         "(default; union semantics).")
    ap.add_argument("--replacement-mode", action="store_true",
                    help="For sparsity_match, target K*raw_nnz before optional union.")
    ap.add_argument("--no-union", action="store_true",
                    help="Write thresholded imputed CSR only (no max with raw).")
    ap.add_argument("--sample-size", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-rows", type=int, default=2000)
    args = ap.parse_args()

    mm_dir = args.mm_dir.expanduser().resolve()
    imp_dir = args.impute_dir.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    if args.replacement_mode:
        args.addition_mode = False

    raw, full_regs, full_bars = load_raw_mm(mm_dir)
    kind = detect_kind(imp_dir)
    log.info("impute kind=%s raw_shape=%s", kind, raw.shape)

    meta_in = {}
    meta_path = imp_dir / "meta.json"
    if args.from_meta and meta_path.exists():
        meta_in = json.loads(meta_path.read_text())
        log.info("Loaded meta.json keys: %s", list(meta_in.keys()))

    rng = np.random.default_rng(int(args.seed))

    t_explicit: float | None = args.threshold
    if t_explicit is None and args.from_meta and "threshold" in meta_in:
        t_explicit = float(meta_in["threshold"])
        log.info("Using threshold from meta.json: %g", t_explicit)

    mode, param = parse_threshold_mode(
        str(meta_in.get("threshold_mode", args.threshold_mode))
        if args.from_meta and "threshold_mode" in meta_in
        else args.threshold_mode
    )

    imp_thr: sparse.csr_matrix
    t_used: float
    threshold_origin: str

    if kind == "csr":
        imp_csr = sparse.load_npz(imp_dir / "matrix_csr.npz").tocsr().astype(np.float32)
        if imp_csr.shape != raw.shape:
            raise SystemExit(f"CSR shape {imp_csr.shape} != raw {raw.shape}")
        if t_explicit is None:
            log.info(
                "No --threshold / meta: treating CSR as pre-thresholded; union only."
            )
            imp_thr = imp_csr
            t_used = float("nan")
            threshold_origin = "pre_thresholded_csr"
        else:
            imp_thr = csr_thresholded_from_csr(imp_csr, t_explicit)
            t_used = float(t_explicit)
            threshold_origin = "CLI_or_meta_rethreshold"
    else:
        imp_regs = _read_lines_either(imp_dir, "regions")
        imp_bars = _read_lines_either(imp_dir, "barcodes")
        if imp_regs is None or imp_bars is None:
            raise SystemExit("impute-dir needs regions.tsv(.gz) and barcodes.tsv(.gz)")
        rmap = {r: i for i, r in enumerate(full_regs)}
        cmap = {b: i for i, b in enumerate(full_bars)}

        if t_explicit is not None:
            t_used = float(t_explicit)
            threshold_origin = "CLI_or_meta.json"
        else:
            if kind == "factored":
                modeled_regs, modeled_cells = factored_modeled_labels(
                    imp_dir, imp_regs, imp_bars,
                )
                try:
                    mrow = np.array([rmap[r] for r in modeled_regs], dtype=np.int64)
                    mcol = np.array([cmap[b] for b in modeled_cells], dtype=np.int64)
                except KeyError as e:
                    raise SystemExit(f"alignment {e}") from e
            else:
                mrow = np.array([rmap[r] for r in imp_regs], dtype=np.int64)
                mcol = np.array([cmap[b] for b in imp_bars], dtype=np.int64)

            raw_bin = (raw > 0).astype(np.int8)
            sub = raw_bin[mrow][:, mcol]
            raw_nnz = int(sub.nnz)

            if mode == "fixed":
                t_used = float(param)
                threshold_origin = f"fixed:{param}"
            elif mode == "sparsity_match":
                K = float(param)
                if args.addition_mode and not args.no_union:
                    st = max(0, int(round((K - 1.0) * raw_nnz)))
                else:
                    st = max(1, int(round(K * raw_nnz)))
                if kind == "factored":
                    z = np.load(imp_dir / "factors.npz")
                    phi = np.asarray(z["W"], dtype=np.float32)
                    theta = np.asarray(z["H"], dtype=np.float32)
                    if "per_cell_factor" in z.files:
                        pcf = z["per_cell_factor"]
                        sc = float(pcf) if pcf.ndim == 0 else float(np.mean(pcf))
                    else:
                        sc = 1.0
                    z.close()
                    t_used, threshold_origin = pick_threshold(
                        phi.astype(np.float64), theta.astype(np.float64), sc,
                        mode, param, st, args.sample_size, rng,
                    )
                else:
                    arr = np.load(imp_dir / "matrix.npy", mmap_mode="r")
                    t_used, threshold_origin = pick_threshold_dense(
                        arr, mode, param, st, args.sample_size, rng,
                    )
            elif mode == "quantile":
                if kind == "factored":
                    z = np.load(imp_dir / "factors.npz")
                    phi = np.asarray(z["W"], dtype=np.float32)
                    theta = np.asarray(z["H"], dtype=np.float32)
                    sc = 1.0
                    if "per_cell_factor" in z.files:
                        pcf = z["per_cell_factor"]
                        sc = float(pcf) if pcf.ndim == 0 else float(np.mean(pcf))
                    z.close()
                    t_used, threshold_origin = pick_threshold(
                        phi.astype(np.float64), theta.astype(np.float64), sc,
                        mode, param, 0, args.sample_size, rng,
                    )
                else:
                    arr = np.load(imp_dir / "matrix.npy", mmap_mode="r")
                    t_used, threshold_origin = pick_threshold_dense(
                        arr, mode, param, 0, args.sample_size, rng,
                    )
            else:
                raise SystemExit(f"Unsupported threshold mode {mode!r} without --threshold")

            log.info("Estimated threshold %g (%s)", t_used, threshold_origin)

        if kind == "factored":
            imp_thr = build_imp_csr_factored(
                imp_dir, full_regs, full_bars, float(t_used), args.chunk_rows
            )
        else:
            imp_thr = build_imp_csr_dense(
                imp_dir, full_regs, full_bars, float(t_used), args.chunk_rows
            )

    nnz_imp = int(imp_thr.nnz)
    if args.no_union:
        out = imp_thr
        union_applied = False
    else:
        # scipy.sparse.maximum removed in recent SciPy; CSR.maximum is supported.
        out = raw.maximum(imp_thr.tocsr())
        union_applied = True
    nnz_out = int(out.nnz)

    label = args.label.strip() or "union"
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "matrix_csr.npz", out.astype(np.float32))
    (out_dir / "regions.tsv").write_text("\n".join(full_regs) + "\n")
    (out_dir / "barcodes.tsv").write_text("\n".join(full_bars) + "\n")

    meta = {
        "pipeline": "union_with_raw.py",
        "mm_dir": str(mm_dir),
        "source_impute_dir": str(imp_dir),
        "input_kind": kind,
        "threshold": float(t_used) if math.isfinite(t_used) else None,
        "threshold_origin": threshold_origin,
        "threshold_mode_resolved": f"{mode}:{param}",
        "union_applied": union_applied,
        "addition_mode": bool(args.addition_mode) and union_applied,
        "nnz_imp_gt_t": nnz_imp,
        "nnz_output": nnz_out,
        "shape": [int(out.shape[0]), int(out.shape[1])],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Wrote %s (nnz=%d)", out_dir / "matrix_csr.npz", nnz_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
