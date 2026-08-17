#!/usr/bin/env python3
"""Held-out imputation scatter (tutorial style).

For each TF, sample known matrix entries (and matched zeros), score them with
the trained unified model (continuous yhat, *without* keep_raw), then plot:

    X = Original value
    Y = Imputed value
    diagonal y = x
    annotate Pearson r

This matches the usual masking check: do imputed values recover the originals
and sit on the diagonal?

Modes:
  --score   run model, write pairs npz under --out-dir
  --plot    make tutorial-style scatters from saved pairs
  (default) score + plot

Usage:
  # GPU node (borzoi env has torch):
  conda activate borzoi
  python downstream/imputation_heldout_scatter.py --score --tfs znf282

  # any node with matplotlib:
  python downstream/imputation_heldout_scatter.py --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
UNIFIED_SCRIPTS = REPO / "unified" / "scripts"

DEFAULT_TFS = [
    "ctcf", "maz", "nfya", "zbtb7a", "zic2", "znf777", "znf282",
]


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or x.size != y.size:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score_tf(
    tf: str,
    work_root: Path,
    out_dir: Path,
    n_pos: int,
    n_neg: int,
    seed: int,
    chunk_bins: int,
    device_str: str,
) -> Path:
    import h5py
    import torch

    sys.path.insert(0, str(UNIFIED_SCRIPTS))
    from _model import GatedUnifiedModel  # noqa: E402

    work = work_root / tf
    mm_dir = work / "mm"
    seqs_dir = work / "seqs"
    ckpt_path = work / "model" / "ckpt.pt"
    for p in (mm_dir / "matrix.mtx.gz", seqs_dir / "seqs.h5", seqs_dir / "chrom_ranges.json", ckpt_path):
        if not p.is_file():
            raise SystemExit(f"[{tf}] missing {p}")

    device = torch.device(
        device_str if device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[{tf}] device={device}", flush=True)

    raw = sio.mmread(mm_dir / "matrix.mtx.gz").tocsr()
    n_bins_full, n_cells = raw.shape
    h5 = h5py.File(seqs_dir / "seqs.h5", "r")
    X_ds = h5["X"]
    mm_row_idx = np.asarray(h5["mm_row_idx"], dtype=np.int64)  # kept -> full
    n_kept = int(X_ds.shape[0])
    # full mm row -> kept index (-1 if dropped)
    full_to_kept = np.full(n_bins_full, -1, dtype=np.int64)
    full_to_kept[mm_row_idx] = np.arange(n_kept, dtype=np.int64)

    rng = np.random.default_rng(seed)
    # Sample observed + unobserved entries that fall on kept bins only
    raw_coo = raw.tocoo()
    kept_mask = full_to_kept[raw_coo.row] >= 0
    pos_rows = raw_coo.row[kept_mask]
    pos_cols = raw_coo.col[kept_mask]
    pos_vals = raw_coo.data[kept_mask].astype(np.float64)
    if pos_rows.size == 0:
        raise SystemExit(f"[{tf}] no observed entries on kept bins")

    n_take_pos = min(n_pos, pos_rows.size)
    pi = rng.choice(pos_rows.size, size=n_take_pos, replace=False)
    s_rows = pos_rows[pi]
    s_cols = pos_cols[pi]
    s_orig = pos_vals[pi]

    # Negatives: random (kept-bin, cell) with raw==0
    neg_rows_l: list[int] = []
    neg_cols_l: list[int] = []
    target_neg = n_neg
    tries = 0
    while len(neg_rows_l) < target_neg and tries < 80:
        tries += 1
        need = target_neg - len(neg_rows_l)
        br = rng.integers(0, n_kept, size=max(need * 4, 4096))
        bc = rng.integers(0, n_cells, size=br.size)
        fr = mm_row_idx[br]
        # Vectorized: build a temporary indicator via lil is slow; use
        # csr fancy indexing in batches.
        batch = 8192
        for a in range(0, fr.size, batch):
            b = min(fr.size, a + batch)
            # raw[fr[i], bc[i]] via advanced indexing on CSR is not elementwise;
            # use a small python loop over this batch only.
            for r, c in zip(fr[a:b], bc[a:b]):
                if raw[int(r), int(c)] == 0:
                    neg_rows_l.append(int(r))
                    neg_cols_l.append(int(c))
                    if len(neg_rows_l) >= target_neg:
                        break
            if len(neg_rows_l) >= target_neg:
                break
    n_take_neg = len(neg_rows_l)
    if n_take_neg:
        s_rows = np.concatenate([s_rows, np.asarray(neg_rows_l, dtype=np.int64)])
        s_cols = np.concatenate([s_cols, np.asarray(neg_cols_l, dtype=np.int64)])
        s_orig = np.concatenate([s_orig, np.zeros(n_take_neg, dtype=np.float64)])

    print(
        f"[{tf}] sampled {n_take_pos} observed + {n_take_neg} zero entries "
        f"(of {n_bins_full}x{n_cells})",
        flush=True,
    )

    # Load model (non-sparsity-aware checkpoints)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    m_cfg = cfg.get("model") or {}
    seq_length = int((cfg.get("seqs") or {}).get("seq_length", 768))
    D = int(m_cfg.get("latent_dim", 64))
    sparsity_aware = bool(m_cfg.get("sparsity_aware", False))
    if sparsity_aware:
        raise SystemExit(
            f"[{tf}] sparsity_aware ckpt: held-out scoring must zero obs_count; "
            "not implemented in this script yet."
        )

    model = GatedUnifiedModel(
        n_cells=n_cells,
        seq_length=seq_length,
        latent_dim=D,
        stem_filters=int(m_cfg.get("stem_filters", 288)),
        stem_kernel=int(m_cfg.get("stem_kernel", 17)),
        stem_pool=int(m_cfg.get("stem_pool", 3)),
        tower_channels=tuple(m_cfg.get("tower_channels", [288, 323, 363, 407])),
        tower_kernel=int(m_cfg.get("tower_kernel", 5)),
        head_filters=int(m_cfg.get("head_filters", 256)),
        dropout=float(m_cfg.get("dropout", 0.20)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 4, 16])),
        context_kernel=int(m_cfg.get("context_kernel", 3)),
        alpha_init=float(m_cfg.get("alpha_init", 0.1)),
        gate_hidden=tuple(m_cfg.get("gate_hidden", [128, 64, 32])),
        gate_use_concat=bool(m_cfg.get("gate_use_concat", True)),
        sparsity_aware=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Group samples by kept-bin index for chunked predict
    kept_idx = full_to_kept[s_rows]
    assert (kept_idx >= 0).all()
    order = np.argsort(kept_idx)
    s_rows, s_cols, s_orig, kept_idx = (
        s_rows[order], s_cols[order], s_orig[order], kept_idx[order]
    )
    yhat = np.empty(s_rows.size, dtype=np.float64)

    with torch.no_grad():
        i0 = 0
        while i0 < kept_idx.size:
            k0 = int(kept_idx[i0])
            # take a contiguous kept-bin window covering upcoming samples
            i1 = i0
            while i1 < kept_idx.size and kept_idx[i1] < k0 + chunk_bins:
                i1 += 1
            k1 = int(kept_idx[i1 - 1]) + 1
            # encode [k0, k1)
            seq_np = X_ds[k0:k1].astype(np.float32)
            e = model.encode_bins(torch.from_numpy(seq_np).to(device))
            pred = model.predict_chunk(e).detach().cpu().numpy()  # (k1-k0, n_cells)
            for j in range(i0, i1):
                yhat[j] = float(pred[kept_idx[j] - k0, s_cols[j]])
            i0 = i1
            if i0 % 5000 < chunk_bins:
                print(f"[{tf}] scored {i0}/{kept_idx.size}", flush=True)

    h5.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tf}_heldout_pairs.npz"
    np.savez_compressed(
        out,
        original=s_orig.astype(np.float32),
        imputed=yhat.astype(np.float32),
        rows=s_rows.astype(np.int64),
        cols=s_cols.astype(np.int64),
        n_pos=np.int64(n_take_pos),
        n_neg=np.int64(n_take_neg),
        r_pearson=np.float64(_safe_pearson(s_orig, yhat)),
        r_binary=np.float64(_safe_pearson((s_orig > 0).astype(np.float64), yhat)),
    )
    meta = {
        "tf": tf,
        "n_pos": int(n_take_pos),
        "n_neg": int(n_take_neg),
        "r_pearson_count_vs_score": float(_safe_pearson(s_orig, yhat)),
        "r_pearson_binary_vs_score": float(
            _safe_pearson((s_orig > 0).astype(np.float64), yhat)
        ),
        "note": (
            "imputed = continuous model yhat at sampled entries "
            "(no keep_raw / no binarize); original = raw count"
        ),
    }
    (out_dir / f"{tf}_heldout_pairs.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{tf}] wrote {out}  r_bin={meta['r_pearson_binary_vs_score']:.4f} "
          f"r_cnt={meta['r_pearson_count_vs_score']:.4f}", flush=True)
    return out


def plot_tf(pairs_path: Path, out_png: Path, max_points: int, seed: int) -> float:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pairs_path)
    orig = np.asarray(z["original"], dtype=np.float64)
    imp = np.asarray(z["imputed"], dtype=np.float64)
    # Tutorial axes: both on [0, 1] — original as observed(0/1), imputed as score
    x = (orig > 0).astype(np.float64)
    y = imp
    r = _safe_pearson(x, y)

    rng = np.random.default_rng(seed)
    if x.size > max_points:
        take = rng.choice(x.size, size=max_points, replace=False)
        xs, ys = x[take], y[take]
    else:
        xs, ys = x, y

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    # jitter x slightly so 0/1 stacks are visible
    jitter = (rng.random(xs.size) - 0.5) * 0.06
    ax.scatter(
        xs + jitter,
        ys,
        s=10,
        alpha=0.25,
        c="#222222",
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, 1], [0, 1], color="#b85c38", lw=1.6, ls="--", label="y = x")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Original value", fontsize=12)
    ax.set_ylabel("Imputed value", fontsize=12)
    tf = pairs_path.name.split("_")[0].upper()
    ax.set_title(
        f"{tf}\nPearson correlation between the original and\n"
        f"imputed values was r = {r:.2f}",
        fontsize=11,
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Continuous companion: log1p(count) vs score (observed+zeros)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    xc = np.log1p(orig)
    # scale score to log1p count range for a meaningful diagonal overlay
    # (display only): map y to [0, max(xc)] by rank? Better: show raw scales + fit
    if xc.size > max_points:
        take = rng.choice(xc.size, size=max_points, replace=False)
        x2, y2 = xc[take], y[take]
    else:
        x2, y2 = xc, y
    ax.scatter(x2, y2, s=10, alpha=0.25, c="#222222", edgecolors="none", rasterized=True)
    r2 = _safe_pearson(xc, y)
    # identity only if we plot score vs binary already; here draw linear fit
    if np.isfinite(r2) and x2.std() > 0:
        coef = np.polyfit(x2, y2, 1)
        xx = np.linspace(float(x2.min()), float(x2.max()), 50)
        ax.plot(xx, coef[0] * xx + coef[1], color="#b85c38", lw=1.4, ls="--", label="linear fit")
    ax.set_xlabel("Original value (log1p count)", fontsize=12)
    ax.set_ylabel("Imputed value (model score)", fontsize=12)
    ax.set_title(f"{tf}  r = {r2:.2f}", fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    alt = out_png.with_name(out_png.stem + "_logcount.png")
    fig.savefig(alt, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png} and {alt} (r_binary={r:.4f})", flush=True)
    return r


def plot_multipanel(out_dir: Path, tfs: list[str], max_points: int, seed: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [out_dir / f"{tf}_heldout_pairs.npz" for tf in tfs]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit("No heldout pair files to plot")

    n = len(paths)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.1 * nrows), squeeze=False)
    fig.suptitle(
        "Held-out check: original vs imputed (model score)\n"
        "Points should concentrate around y = x if imputation recovers known values",
        fontsize=13,
        y=1.02,
    )
    rng = np.random.default_rng(seed)
    for i, p in enumerate(paths):
        ax = axes[i // ncols][i % ncols]
        z = np.load(p)
        orig = np.asarray(z["original"], dtype=np.float64)
        imp = np.asarray(z["imputed"], dtype=np.float64)
        x = (orig > 0).astype(np.float64)
        y = imp
        r = _safe_pearson(x, y)
        if x.size > max_points:
            take = rng.choice(x.size, size=max_points, replace=False)
            xs, ys = x[take], y[take]
        else:
            xs, ys = x, y
        jitter = (rng.random(xs.size) - 0.5) * 0.06
        ax.scatter(xs + jitter, ys, s=6, alpha=0.2, c="#222", edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#b85c38", lw=1.2, ls="--")
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        tf = p.name.split("_")[0]
        ax.set_title(f"{tf.upper()}\nr = {r:.2f}", fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel("Imputed value")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Original value")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    out = out_dir / "heldout_scatter_all.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tfs", nargs="*", default=None)
    ap.add_argument("--work-root", type=Path, default=REPO / "unified" / "work")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "plots" / "imputation_heldout")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--n-pos", type=int, default=25000)
    ap.add_argument("--n-neg", type=int, default=25000)
    ap.add_argument("--chunk-bins", type=int, default=256)
    ap.add_argument("--max-points", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    tfs = [t.lower() for t in (args.tfs or DEFAULT_TFS)]
    do_score = args.score or not args.plot
    do_plot = args.plot or not args.score
    # if neither flag: do both; if only one: that one
    if args.score and not args.plot:
        do_plot = False
    if args.plot and not args.score:
        do_score = False

    if do_score:
        for tf in tfs:
            score_tf(
                tf, args.work_root, args.out_dir,
                args.n_pos, args.n_neg, args.seed, args.chunk_bins, args.device,
            )
    if do_plot:
        for tf in tfs:
            pairs = args.out_dir / f"{tf}_heldout_pairs.npz"
            if not pairs.is_file():
                print(f"[skip plot] missing {pairs}", flush=True)
                continue
            plot_tf(pairs, args.out_dir / f"{tf}_heldout_scatter.png",
                    args.max_points, args.seed)
        plot_multipanel(args.out_dir, tfs, args.max_points, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
