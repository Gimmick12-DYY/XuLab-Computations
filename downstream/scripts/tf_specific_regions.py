#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# tf_specific_regions.py
#
# Define TF-specific foreground and a matched background from the panel matrix,
# for motif discovery that is not swamped by open-chromatin grammar.
#
# The problem this solves: peaks called on the imputed calls are essentially the
# HEK293T open-chromatin landscape (imputation is masked to the ATAC union, and
# 75-91% of peaks are shared between any two TFs). Enriching those against the
# genome, or against generic OCR, asks "is this region open?" -- to which the
# answer is a promoter motif (Maz/SP/NFY/CTCF/E2F) for every TF in the panel.
#
# The panel gives a better contrast. With 78 TFs on one 1 kb grid in one cell
# type, ask instead: which bins are high for THIS TF and low for the rest, and
# compare them to bins that are equally bound *by the panel as a whole* but not
# by this TF. Foreground and background are then both open, both promoter-heavy,
# and matched on general "boundness" -- so shared accessibility cancels and what
# is left is the part that can carry the TF's own motif.
#
# Scoring (on the rank-normalized columns, so per-TF depth/breadth cannot leak):
#   self      = ranknorm[:, tf]
#   others_hi = quantile(ranknorm[:, other TFs], --others-quantile, axis=1)
#   spec      = self - others_hi
#   foreground: self >= --fg-self-min and spec >= --fg-spec-min, top --n-fg by spec
#   background: self <= --bg-self-max, sampled to match the foreground's
#               others_hi distribution (--match-bins strata), --bg-ratio x n_fg
#
# --exclude-others matters for complexes: RBBP4/RBBP7, EZH2/MTF2/PCGF6 etc. share
# targets, so a partner sitting in "others" cancels the very signal being sought.
# Drop known partners from the contrast rather than from the panel.
#
# Outputs (--out-dir):
#   <tf>.specific_fg.bed   TF-specific bins
#   <tf>.specific_bg.bed   matched, panel-bound-but-not-this-TF bins
#   <tf>.specificity.tsv   per-bin self/others_hi/spec for the foreground
#   <tf>.summary.json      counts + the thresholds actually used
# Exit 3 if fewer than --min-fg foreground bins exist (no TF-specific signal).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_RE_REGION = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def parse_regions(regions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chrom = np.empty(regions.size, dtype=object)
    start = np.zeros(regions.size, dtype=np.int64)
    end = np.zeros(regions.size, dtype=np.int64)
    for i, r in enumerate(regions):
        m = _RE_REGION.match(str(r))
        if m is None:
            raise SystemExit(f"unparsable region {r!r}")
        chrom[i] = m.group("chrom")
        start[i] = int(m.group("start"))
        end[i] = int(m.group("end"))
    return chrom, start, end


def write_bed(path: Path, chrom, start, end, idx, name_prefix: str, score) -> None:
    with open(path, "w") as fh:
        for n, i in enumerate(idx):
            fh.write(f"{chrom[i]}\t{start[i]}\t{end[i]}\t"
                     f"{name_prefix}{n}\t{score[i]:.5f}\t.\n")


def match_background(
    cand: np.ndarray, fg: np.ndarray, others_hi: np.ndarray,
    n_want: int, n_strata: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample from `cand` so its others_hi distribution matches the foreground's.

    Without this the background drifts to weakly-bound bins and the contrast
    turns back into "bound vs unbound" -- i.e. accessibility again.
    """
    edges = np.quantile(others_hi[fg], np.linspace(0, 1, n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    fg_counts, _ = np.histogram(others_hi[fg], bins=edges)
    per_stratum = np.maximum(1, np.round(fg_counts / max(fg.size, 1) * n_want).astype(int))

    cand_stratum = np.digitize(others_hi[cand], edges[1:-1])
    picked: list[np.ndarray] = []
    for s in range(n_strata):
        pool = cand[cand_stratum == s]
        if pool.size == 0:
            continue
        take = min(per_stratum[s], pool.size)
        picked.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-npz", type=Path, required=True)
    ap.add_argument("--tf", required=True, help="lowercase TF key")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exclude-others", nargs="*", default=[],
                    help="TFs to drop from the contrast (complex partners/paralogs)")
    ap.add_argument("--others-quantile", type=float, default=0.90,
                    help="quantile across the other TFs (0.9 = 'almost nobody else')")
    ap.add_argument("--fg-self-min", type=float, default=0.90,
                    help="min rank in the TF's own track (0.90 = its top 10% of bins)")
    ap.add_argument("--fg-spec-min", type=float, default=0.20,
                    help="min self - others_hi")
    ap.add_argument("--bg-self-max", type=float, default=0.50)
    ap.add_argument("--n-fg", type=int, default=20000)
    ap.add_argument("--bg-ratio", type=float, default=2.0)
    ap.add_argument("--match-bins", type=int, default=20)
    ap.add_argument("--min-fg", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with np.load(args.panel_npz, allow_pickle=False) as z:
        ranknorm = z["ranknorm"]
        regions = z["regions"]
        tfs = [str(x) for x in z["tfs"]]

    tf = args.tf.lower()
    if tf not in tfs:
        raise SystemExit(f"{tf} not in panel ({len(tfs)} TFs)")
    t = tfs.index(tf)

    drop = {x.lower() for x in args.exclude_others} | {tf}
    other_cols = [j for j, name in enumerate(tfs) if name not in drop]
    if len(other_cols) < 5:
        raise SystemExit(f"only {len(other_cols)} contrast TFs left after --exclude-others")

    self_r = ranknorm[:, t].astype(np.float32)
    others_hi = np.quantile(ranknorm[:, other_cols], args.others_quantile,
                            axis=1).astype(np.float32)
    spec = self_r - others_hi

    fg_mask = (self_r >= args.fg_self_min) & (spec >= args.fg_spec_min)
    fg_all = np.flatnonzero(fg_mask)
    if fg_all.size > args.n_fg:
        fg = fg_all[np.argsort(spec[fg_all])[::-1][:args.n_fg]]
    else:
        fg = fg_all
    print(f"[{tf}] contrast against {len(other_cols)} TFs | "
          f"candidate fg bins={fg_all.size} -> using {fg.size}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tf": tf,
        "n_contrast_tfs": len(other_cols),
        "excluded": sorted(drop - {tf}),
        "n_fg_candidates": int(fg_all.size),
        "n_fg": int(fg.size),
        "others_quantile": args.others_quantile,
        "fg_self_min": args.fg_self_min,
        "fg_spec_min": args.fg_spec_min,
    }

    if fg.size < args.min_fg:
        summary["status"] = "insufficient_specific_bins"
        (args.out_dir / f"{tf}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[{tf}] only {fg.size} TF-specific bins (< --min-fg {args.min_fg}); "
              "no TF-specific signal to discover a motif from", flush=True)
        return 3

    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero((self_r <= args.bg_self_max) & (others_hi > 0))
    bg = match_background(cand, fg, others_hi, int(args.bg_ratio * fg.size),
                          args.match_bins, rng)
    print(f"[{tf}] background: {bg.size} bins from {cand.size} candidates", flush=True)

    chrom, start, end = parse_regions(regions)
    write_bed(args.out_dir / f"{tf}.specific_fg.bed", chrom, start, end, fg, f"{tf}_fg", spec)
    write_bed(args.out_dir / f"{tf}.specific_bg.bed", chrom, start, end, bg, f"{tf}_bg", spec)

    with open(args.out_dir / f"{tf}.specificity.tsv", "w") as fh:
        fh.write("region\tself_rank\tothers_hi\tspecificity\n")
        for i in fg[np.argsort(spec[fg])[::-1]]:
            fh.write(f"{regions[i]}\t{self_r[i]:.4f}\t{others_hi[i]:.4f}\t{spec[i]:.4f}\n")

    summary.update({
        "status": "ok",
        "n_bg": int(bg.size),
        "fg_spec_median": float(np.median(spec[fg])),
        "fg_others_hi_median": float(np.median(others_hi[fg])),
        "bg_others_hi_median": float(np.median(others_hi[bg])) if bg.size else None,
    })
    (args.out_dir / f"{tf}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{tf}] -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
