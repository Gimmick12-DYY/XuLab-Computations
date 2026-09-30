#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_borzoi_predict.py
#
# Frozen Borzoi inference over the windows planned by 02_prepare_windows.py.
#
# Strategy:
#   - Load borzoi-pytorch model at the requested fold(s); apply the
#     track_prefilter regex to Borzoi's targets_human.txt so we keep only the
#     output channels we care about (CTCF/FOXA/etc. by default).
#   - Stream windows in order of chromosome. For each window:
#       1. Fetch the 524288 bp input slice from hg38 FASTA. N-pad off-the-end
#          flanks if needed.
#       2. One-hot encode -> tensor (1, 4, 524288), to GPU at requested
#          precision.
#       3. Forward pass through Borzoi -> (1, n_tracks_kept, n_pos).
#       4. Slice the central output (n_output_positions positions).
#       5. For every mm bin assigned to this window, mean-pool the bin's
#          output_position range -> one scalar per (bin, track).
#   - Write per-chrom h5 shards under <work>/predict/shards/. The final
#     bin_track.h5 is assembled at the end of step 03.
#   - The shards make 03 restartable: existing shards are skipped.
#
# borzoi-pytorch API note: this script targets the lucidrains port. If the API
# changes, only `_borzoi_load`, `_borzoi_targets`, and `_borzoi_forward` need
# adjustment.
#
# Outputs (under <work>/predict/):
#   shards/<chrom>.h5            per-chrom (n_bins_in_chrom, n_tracks_kept) f16
#   bin_track.h5                 merged across chroms
#     /X                         (n_kept_bins, n_tracks_kept) float16
#     /mm_row_idx                (n_kept_bins,)               int64
#     /track_names               (n_tracks_kept,)             S256
#     attrs: fold, input_length, central_output_bp, output_bin_bp
#   meta.json                    timing + counts + git/hash info
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from _cfg import base_parser, load_config, resolve_paths

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

log = logging.getLogger("03_borzoi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# A->0 C->1 G->2 T->3, everything else -> -1 (becomes all-zero one-hot)
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for ch, idx in zip(b"ACGT", range(4)):
    _BASE_LUT[ch] = idx
    _BASE_LUT[ch + 32] = idx


def one_hot(seq_bytes: np.ndarray) -> np.ndarray:
    """(L,) uint8 ASCII -> (4, L) uint8 one-hot. N/other -> zero column."""
    idx = _BASE_LUT[seq_bytes]                 # (L,) int8 in {-1, 0..3}
    L = idx.shape[0]
    out = np.zeros((4, L), dtype=np.uint8)
    mask = idx >= 0
    pos = np.where(mask)[0]
    out[idx[pos], pos] = 1
    return out


def fetch_window_seq(fa, chrom: str, input_start: int, L_in: int,
                     chrom_len: int) -> np.ndarray:
    """Return uint8 ASCII array of length L_in, N-padded if out of bounds."""
    a = max(0, int(input_start))
    b = min(chrom_len, int(input_start) + L_in)
    if b <= a:
        return np.full(L_in, ord("N"), dtype=np.uint8)
    s = str(fa.get_seq(chrom, a + 1, b)).upper()
    buf = np.full(L_in, ord("N"), dtype=np.uint8)
    off = a - int(input_start)
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    buf[off:off + arr.size] = arr
    return buf


# ----------------------------- borzoi-pytorch backend ------------------------

def _borzoi_load(weights_dir: Path, fold: int, precision: str, device):
    """Load Borzoi for one fold. Returns (model, n_tracks)."""
    try:
        from borzoi_pytorch import Borzoi
    except ImportError as e:
        raise SystemExit(
            "borzoi-pytorch not installed. pip install borzoi-pytorch"
        ) from e

    # The lucidrains port stores folds as separate HF Hub repos. Local
    # weights live under <weights_dir>/fold<fold>/.
    local = weights_dir / f"fold{fold}"
    if local.is_dir():
        log.info("Loading Borzoi fold %d from %s", fold, local)
        model = Borzoi.from_pretrained(str(local))
    else:
        repo_id = f"johahi/borzoi-replicate-{fold}"
        log.info("Loading Borzoi fold %d from HF Hub: %s", fold, repo_id)
        model = Borzoi.from_pretrained(repo_id)

    # Keep weights in fp32: borzoi_pytorch runs the final head with autocast
    # disabled and x.float(), so model.half() causes Input type (float) vs
    # bias type (Half) errors in human_head.
    return model.eval().to(device)


def _borzoi_targets(weights_dir: Path) -> list[str]:
    """Load Borzoi target track names. Expected at <weights_dir>/targets_human.txt."""
    for cand in (weights_dir / "targets_human.txt",
                 weights_dir / "targets.txt"):
        if cand.is_file():
            log.info("Loading target names from %s", cand)
            names: list[str] = []
            with open(cand) as fh:
                header = fh.readline()
                # The published file is tab-separated with a header.
                # The track name is conventionally column 5 (description) or 1.
                # We try a couple of common layouts.
                fields = header.rstrip("\n").split("\t")
                desc_col = None
                for c in ("description", "identifier", "name"):
                    if c in fields:
                        desc_col = fields.index(c)
                        break
                if desc_col is None:
                    desc_col = min(5, len(fields) - 1)
                for ln in fh:
                    parts = ln.rstrip("\n").split("\t")
                    if len(parts) <= desc_col:
                        continue
                    names.append(parts[desc_col])
            return names
    raise SystemExit(
        f"Could not find targets_human.txt under {weights_dir}. "
        "Run fetch_borzoi_weights.sh to populate it."
    )


def _borzoi_forward(model, x_onehot_4xL, device, precision: str):
    """Run one forward pass. Returns numpy (n_tracks, n_pos) float32."""
    import torch
    x = torch.from_numpy(x_onehot_4xL).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if precision == "fp16":
            with torch.amp.autocast(device.type, dtype=torch.float16):
                y = model(x)
        elif precision == "bf16":
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                y = model(x)
        else:
            y = model(x)
    # lucidrains' Borzoi returns (B, n_tracks, n_pos). Coerce to that shape.
    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
    elif y.ndim == 2:
        pass
    else:
        raise SystemExit(f"Unexpected Borzoi output shape: {tuple(y.shape)}")
    return y.float().cpu().numpy()


# ----------------------------- main ------------------------------------------

def _select_track_indices(track_names: list[str], pattern: str | None) -> np.ndarray:
    if not pattern:
        return np.arange(len(track_names), dtype=np.int64)
    rx = re.compile(pattern)
    keep = [i for i, n in enumerate(track_names) if rx.search(n)]
    if not keep:
        raise SystemExit(f"track_prefilter {pattern!r} matched 0 of {len(track_names)} tracks")
    log.info("track_prefilter %r: keeping %d / %d Borzoi tracks",
             pattern, len(keep), len(track_names))
    return np.asarray(keep, dtype=np.int64)


def _shard_path(out_dir: Path, chrom: str) -> Path:
    return out_dir / "shards" / f"{chrom}.h5"


def _load_windows_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as h5:
        w_chrom = np.asarray([s.decode() if isinstance(s, bytes) else s
                              for s in h5["windows/chrom"][:]])
        w_input_start   = np.asarray(h5["windows/input_start"])
        w_central_start = np.asarray(h5["windows/central_start"])
        b_mm_row    = np.asarray(h5["bins/mm_row_idx"])
        b_window_id = np.asarray(h5["bins/window_id"])
        b_pos_start = np.asarray(h5["bins/out_pos_start"])
        b_pos_end   = np.asarray(h5["bins/out_pos_end"])
        L_in        = int(h5.attrs["input_length"])
        L_out       = int(h5.attrs["central_output_bp"])
        bin_bp      = int(h5.attrs["output_bin_bp"])
        n_pos       = int(h5.attrs["n_output_positions"])
        flank       = int(h5.attrs["flank"])
        genome_fa   = h5.attrs.get("genome_fa")
    return dict(
        w_chrom=w_chrom, w_input_start=w_input_start, w_central_start=w_central_start,
        b_mm_row=b_mm_row, b_window_id=b_window_id,
        b_pos_start=b_pos_start, b_pos_end=b_pos_end,
        L_in=L_in, L_out=L_out, bin_bp=bin_bp, n_pos=n_pos, flank=flank,
        genome_fa=str(genome_fa),
    )


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    b_cfg = cfg.setdefault("borzoi", {})
    fold       = int(args.fold if args.fold is not None else b_cfg.get("fold", 0))
    precision  = str(b_cfg.get("precision", "fp16"))
    prefilter  = b_cfg.get("track_prefilter", None)
    do_chk     = bool(b_cfg.get("checkpoint_per_chrom", True))

    windows_h5 = Path(cfg["paths"]["windows"]) / "windows.h5"
    if not windows_h5.exists():
        log.error("Missing %s -- run 02_prepare_windows.py first", windows_h5)
        return 1
    win = _load_windows_h5(windows_h5)

    weights_dir = Path(cfg["paths"]["borzoi_weights"]).expanduser().resolve()
    if not weights_dir.is_dir():
        log.error("borzoi_weights missing: %s", weights_dir)
        log.error("Run scripts/fetch_borzoi_weights.sh.")
        return 1

    genome_fa = Path(win["genome_fa"]).expanduser().resolve()
    if not genome_fa.exists():
        log.error("Genome FASTA missing: %s", genome_fa)
        return 1
    try:
        from pyfaidx import Fasta
    except ImportError:
        log.error("pyfaidx not installed; pip install pyfaidx")
        return 1
    fa = Fasta(str(genome_fa), as_raw=True, sequence_always_upper=False)
    chrom_lens = {name: len(fa[name]) for name in fa.keys()}

    log.info("Importing torch ...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Torch %s, device=%s", torch.__version__, device)
    if device.type == "cpu":
        log.warning("No GPU detected. Borzoi inference on CPU is impractical at genome scale.")

    track_names_all = _borzoi_targets(weights_dir)
    track_keep_idx = _select_track_indices(track_names_all, prefilter)
    track_names_kept = [track_names_all[i] for i in track_keep_idx.tolist()]
    n_tracks_kept = len(track_keep_idx)

    model = _borzoi_load(weights_dir, fold, precision, device)

    out_dir = Path(cfg["paths"]["predict"])
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)

    # Group bins by chromosome, then by window_id within chrom.
    chrom_arr = win["w_chrom"][win["b_window_id"]]
    unique_chroms = sorted(set(chrom_arr.tolist()))
    log.info("Inference over %d chromosomes, %d windows, %d bins, %d tracks",
             len(unique_chroms), len(win["w_chrom"]), len(win["b_mm_row"]), n_tracks_kept)

    t_total = time.time()
    for ci, chrom in enumerate(unique_chroms):
        shard = _shard_path(out_dir, chrom)
        if do_chk and shard.is_file():
            log.info("[%d/%d] %s: shard exists, skipping", ci + 1, len(unique_chroms), chrom)
            continue
        chrom_mask = chrom_arr == chrom
        chrom_bins = np.where(chrom_mask)[0]
        chrom_window_ids = np.unique(win["b_window_id"][chrom_bins])
        log.info("[%d/%d] %s: %d windows, %d bins",
                 ci + 1, len(unique_chroms), chrom, len(chrom_window_ids), len(chrom_bins))

        per_bin_scores = np.zeros((len(chrom_bins), n_tracks_kept), dtype=np.float16)
        per_bin_mm_row = win["b_mm_row"][chrom_bins].astype(np.int64)
        per_bin_local  = {int(bi): k for k, bi in enumerate(chrom_bins)}

        # Precompute bin lists per window
        bins_by_window: dict[int, list[int]] = {}
        for bi in chrom_bins:
            wid = int(win["b_window_id"][bi])
            bins_by_window.setdefault(wid, []).append(int(bi))

        t_chrom = time.time()
        for wj, wid in enumerate(chrom_window_ids):
            wid_int = int(wid)
            input_start = int(win["w_input_start"][wid_int])
            seq_bytes = fetch_window_seq(
                fa, chrom, input_start, win["L_in"], chrom_lens[chrom]
            )
            x = one_hot(seq_bytes)                  # (4, L_in)
            preds = _borzoi_forward(model, x, device, precision)
            # preds: (n_tracks, n_pos_total). The central output is the last
            # n_pos positions if Borzoi already crops; else we crop manually.
            if preds.shape[-1] != win["n_pos"]:
                # Manual central crop in 32-bp units.
                total = preds.shape[-1]
                if total < win["n_pos"]:
                    raise SystemExit(
                        f"Borzoi returned {total} positions but expected >= {win['n_pos']}"
                    )
                cut = (total - win["n_pos"]) // 2
                preds = preds[:, cut:cut + win["n_pos"]]
            preds_kept = preds[track_keep_idx, :]   # (n_tracks_kept, n_pos)

            for bi in bins_by_window[wid_int]:
                ps = int(win["b_pos_start"][bi])
                pe = max(ps + 1, int(win["b_pos_end"][bi]))
                # Mean-pool over the bin's output-position range
                v = preds_kept[:, ps:pe].mean(axis=1)
                per_bin_scores[per_bin_local[bi]] = v.astype(np.float16)

            if (wj + 1) % 25 == 0:
                dt = time.time() - t_chrom
                rate = (wj + 1) / max(dt, 1e-6)
                eta = (len(chrom_window_ids) - wj - 1) / max(rate, 1e-9)
                log.info("    %s: %d/%d windows (%.2f w/s, eta %.0f s)",
                         chrom, wj + 1, len(chrom_window_ids), rate, eta)

        # Write shard
        with h5py.File(shard, "w") as h5:
            h5.create_dataset("X",          data=per_bin_scores)
            h5.create_dataset("mm_row_idx", data=per_bin_mm_row)
            h5.attrs["chrom"]      = chrom
            h5.attrs["fold"]       = fold
            h5.attrs["n_tracks"]   = n_tracks_kept
        log.info("    %s: wrote %s (%.1f s)", chrom, shard, time.time() - t_chrom)

    # ---- merge shards into bin_track.h5 -------------------------------------
    log.info("Merging shards into bin_track.h5")
    rows: list[np.ndarray] = []
    mm_rows: list[np.ndarray] = []
    for chrom in unique_chroms:
        shard = _shard_path(out_dir, chrom)
        with h5py.File(shard, "r") as h5:
            rows.append(np.asarray(h5["X"]))
            mm_rows.append(np.asarray(h5["mm_row_idx"]))
    X      = np.concatenate(rows, axis=0).astype(np.float16)
    mm_row = np.concatenate(mm_rows, axis=0).astype(np.int64)

    merged = out_dir / "bin_track.h5"
    with h5py.File(merged, "w") as h5:
        h5.create_dataset("X",           data=X, chunks=(min(4096, X.shape[0]), n_tracks_kept),
                          compression="gzip", compression_opts=4)
        h5.create_dataset("mm_row_idx",  data=mm_row)
        h5.create_dataset("track_names", data=np.asarray(track_names_kept, dtype="S256"))
        h5.attrs["fold"]              = fold
        h5.attrs["input_length"]      = win["L_in"]
        h5.attrs["central_output_bp"] = win["L_out"]
        h5.attrs["output_bin_bp"]     = win["bin_bp"]
        h5.attrs["n_output_positions"] = win["n_pos"]
        h5.attrs["track_prefilter"]   = str(prefilter) if prefilter else ""
    log.info("Wrote %s shape=%s", merged, X.shape)

    meta = {
        "fold":             fold,
        "precision":        precision,
        "track_prefilter":  str(prefilter) if prefilter else None,
        "n_tracks_kept":    n_tracks_kept,
        "n_bins":           int(X.shape[0]),
        "n_windows":        int(len(win["w_chrom"])),
        "elapsed_seconds":  time.time() - t_total,
        "bin_track_h5":     str(merged),
    }
    (out_dir / "predict_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
