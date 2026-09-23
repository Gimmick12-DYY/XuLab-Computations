#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# chromhmm18_split_composition.py
#
# Cobinding distal ChromHMM composition, swept over the same 18-state
# segmentation split into N E1-quantile subclasses (N = 2, 3, 5, 10).
#
# For each TF with a .pdc (lab fitConns or cobinding/work/<TF>.generated.pdc):
#   * 18-state stacked bar  (same layout as distal_chromhmm_composition.png)
#   * for each N, an 18-row panel of q1..qN composition within each state
#
# Also writes pooled (all-TF) versions, plus one original-style bar per
# state x N so every cell of the sweep has "the plot".
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from distal_chromhmm_composition import (  # noqa: E402
    BROAD, STATE2BROAD, COLOR as BROAD_COLOR, _COORD, assign, load_segments,
)

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {
    "CTCF_gse103651", "CTCF_gse103651_union3",
    "CTCF_majority2of3", "CTCF_majority2of3_union3", "RBBP4_repro",
}
EPI_ORDER = [
    "TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv",
    "EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk", "EnhBiv",
    "Tx", "TxWk", "ReprPC", "ReprPCWk", "Het", "ZNF/Rpts", "Quies",
]
STATE_COLOR = {
    "TssA": "#ff0000", "TssFlnk": "#ff4500", "TssFlnkU": "#ff6a4d",
    "TssFlnkD": "#ff9a4d", "TssBiv": "#cd5c5c",
    "EnhA1": "#ffa700", "EnhA2": "#ffc34d", "EnhG1": "#54c254",
    "EnhG2": "#6e8b3d", "EnhWk": "#e6d325", "EnhBiv": "#c2c254",
    "Tx": "#008000", "TxWk": "#3f9e4d",
    "ReprPC": "#7b68c4", "ReprPCWk": "#a89dc8",
    "Het": "#3a4750", "ZNF/Rpts": "#66cdaa", "Quies": "#d9dce1",
    "Unassigned": "#ffffff",
}


def parent_state(lab: str) -> str:
    if lab == "Unassigned" or lab is None:
        return "Unassigned"
    return lab.rsplit(".q", 1)[0]


def quantile_of(lab: str) -> int | None:
    if lab is None or ".q" not in lab:
        return None
    try:
        return int(lab.rsplit(".q", 1)[1])
    except ValueError:
        return None


def slug(state: str) -> str:
    return state.replace("/", "_")


def load_pairs(path: Path, qval: float | None):
    opener = gzip.open if str(path).endswith(".gz") else open
    pairs, rows = {}, 0
    with opener(path, "rt") as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 11:
                continue
            rows += 1
            prox, dist, q = p[1], p[5], float(p[10])
            if qval is not None and q > qval:
                continue
            key = (prox, dist)
            if key not in pairs or q < pairs[key]:
                pairs[key] = q
    return pairs, rows


def find_pdc(tf: str, data_dir: Path, work_dir: Path) -> Path | None:
    for cand in (
        data_dir / f"TF.{tf}.fitConns.res.pdc",
        data_dir / f"TF.{tf.lower()}.fitConns.res.pdc",
        work_dir / f"{tf}.generated.pdc",
    ):
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def q_cmap(n: int):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("coolwarm")
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_state_bar(counts, order, title, out_png, colors, min_label=7.0):
    """One horizontal stacked bar — same layout as distal_chromhmm_composition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = sum(counts.get(s, 0) for s in order)
    if n <= 0:
        return
    fig, ax = plt.subplots(figsize=(11, 2.1))
    x = 0.0
    for s in order:
        v = counts.get(s, 0)
        if not v:
            continue
        w = 100 * v / n
        col = colors[s] if isinstance(colors, dict) else colors[order.index(s)]
        ax.barh(0, w, left=x, height=1.0, color=col, edgecolor="white", linewidth=1.4)
        if w >= min_label:
            dark = s in ("Quies", "Quiescent", "Unassigned") or (
                isinstance(s, str) and s.endswith(".q1")
            )
            ax.text(x + w / 2, 0.17, s, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#222" if dark else "white")
            ax.text(x + w / 2, -0.20, f"{w:.1f}%", ha="center", va="center",
                    fontsize=9, color="#222" if dark else "white")
        x += w
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.75, 0.75)
    ax.axis("off")
    ax.set_title(title, fontsize=13, loc="left", pad=12)
    handles = []
    for i, s in enumerate(order):
        v = counts.get(s, 0)
        if not v:
            continue
        col = colors[s] if isinstance(colors, dict) else colors[i]
        handles.append(Patch(facecolor=col, edgecolor="#bbb",
                             label=f"{s} {100 * v / n:.1f}%  (n={v:,})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.25),
              ncol=4, frameon=False, fontsize=8.5)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quantile_panel(state_q_counts, n_split, title, out_png, n_pairs_total):
    """18 rows: stacked q1..qN composition of distal partners in each state."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = q_cmap(n_split)
    states = [s for s in EPI_ORDER if s in state_q_counts] + [
        s for s in state_q_counts if s not in EPI_ORDER and s != "Unassigned"
    ]
    if not states:
        return
    fig, axes = plt.subplots(
        len(states), 1, figsize=(11, max(4.5, 0.42 * len(states))),
        sharex=True,
    )
    if len(states) == 1:
        axes = [axes]
    for ax, st in zip(axes, states):
        cq = state_q_counts[st]
        n = sum(cq.get(q, 0) for q in range(1, n_split + 1))
        x = 0.0
        for q in range(1, n_split + 1):
            v = cq.get(q, 0)
            w = (100 * v / n) if n else 0.0
            ax.barh(0, w, left=x, height=0.72, color=colors[q - 1],
                    edgecolor="white", linewidth=0.8)
            if w >= (100 / n_split) * 0.55 and w >= 8:
                ax.text(x + w / 2, 0, f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color="#222" if q <= n_split / 2 else "white")
            x += w
        ax.axvline(100 / n_split, color="#555", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_ylabel(st, rotation=0, ha="right", va="center", fontsize=8.5,
                      labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
        nlab = f"n={n:,}"
        ax.text(101, 0, nlab, ha="left", va="center", fontsize=7.5, color="#555")
    axes[-1].set_xlabel("% of distal cobinding partners in this ChromHMM state")
    fig.suptitle(title, fontsize=13, x=0.02, ha="left")
    handles = [
        Patch(facecolor=colors[q - 1], edgecolor="#bbb",
              label=f"q{q}" + ("  B-like" if q == 1 else "  A-like" if q == n_split else ""))
        for q in range(1, n_split + 1)
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8,
               title=f"E1 quantile (n={n_split})")
    fig.text(
        0.02, 0.01,
        "q1 = lowest Hi-C E1 (B-like), qN = highest E1 (A-like).  "
        "Equal-bp splits within each ChromHMM-18 state (dotted line = 1/N).  "
        f"{n_pairs_total:,} distal–proximal pairs.",
        fontsize=8, color="#666",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.86, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(header)
        for row in rows:
            fh.write(row)


def classify(pairs, seg_idx, states):
    uniq = sorted({d for _, d in pairs})
    st_of = {d: assign(d, seg_idx, len(states)) for d in uniq}
    labels = []
    for _, d in pairs:
        i = st_of[d]
        labels.append("Unassigned" if i is None else states[i])
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "cobinding" / "results")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "cobinding" / "work")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--chromhmm", type=Path,
                    default=ROOT / "data" / "HEK293T_chromHMM18.bed.gz")
    ap.add_argument("--split-root", type=Path,
                    default=ROOT / "tf_complex" / "work")
    ap.add_argument("--ns", default="2,3,5,10")
    ap.add_argument("--qval", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "cobinding" / "results" / "chromhmm18_split")
    ap.add_argument("--tfs", default="",
                    help="comma-separated TF list (default: all with a .pdc)")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    if args.tfs.strip():
        tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    else:
        tfs = []
        seen = set()
        for p in sorted(args.results_dir.glob("*/nodes.tsv")):
            name = p.parent.name
            if name in SKIP_DIRS or name in seen:
                continue
            if find_pdc(name, args.data_dir, args.work_dir):
                tfs.append(name)
                seen.add(name)
        for p in sorted(args.work_dir.glob("*.generated.pdc")):
            name = p.name.replace(".generated.pdc", "")
            if name in SKIP_DIRS or name in seen:
                continue
            tfs.append(name)
            seen.add(name)

    print(f"[tfs] {len(tfs)}", flush=True)
    chrom18_idx, chrom18_states = load_segments(args.chromhmm)
    print(f"[chromhmm18] {len(chrom18_states)} states", flush=True)

    split = {}
    for n in ns:
        bed = args.split_root / f"chromhmm_x{n}" / f"chromhmm18_x{n}.bed"
        if not bed.is_file():
            raise SystemExit(f"missing split bed {bed}")
        split[n] = load_segments(bed)
        print(f"[x{n}] {len(split[n][1])} classes", flush=True)

    pooled18 = Counter()
    pooled_q = {n: defaultdict(Counter) for n in ns}  # n -> state -> Counter(q)
    pooled_n_pairs = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tf in tfs:
        pdc = find_pdc(tf, args.data_dir, args.work_dir)
        if pdc is None:
            print(f"[skip] {tf} no pdc", flush=True)
            continue
        pairs, nrows = load_pairs(pdc, args.qval)
        if not pairs:
            print(f"[skip] {tf} 0 pairs", flush=True)
            continue
        print(f"[{tf}] {nrows:,} rows -> {len(pairs):,} pairs  ({pdc.name})",
              flush=True)
        pooled_n_pairs += len(pairs)

        labs18 = classify(pairs, chrom18_idx, chrom18_states)
        c18 = Counter(labs18)
        pooled18.update(c18)
        tf_dir = args.results_dir / tf
        plot_state_bar(
            c18, EPI_ORDER + ["Unassigned"],
            title=(
                f"ChromHMM-18 composition of distal regions\n"
                f"{tf} | {len(pairs):,} unique distal-proximal pairs | "
                f"classified by the distal endpoint"
            ),
            out_png=tf_dir / "distal_chromhmm18_composition.png",
            colors=STATE_COLOR,
            min_label=6.0,
        )
        with (tf_dir / "distal_chromhmm18_composition.tsv").open("w") as fh:
            fh.write(f"# {tf}\t{len(pairs)} pairs\tchromhmm18\n")
            fh.write("state\tn_pairs\tpct_pairs\n")
            n = sum(c18.values())
            for s in EPI_ORDER + ["Unassigned"]:
                fh.write(f"{s}\t{c18.get(s,0)}\t{100*c18.get(s,0)/n:.2f}\n")

        for n in ns:
            idx, states = split[n]
            labs = classify(pairs, idx, states)
            sq = defaultdict(Counter)
            for lab in labs:
                par = parent_state(lab)
                q = quantile_of(lab)
                if q is None:
                    continue
                sq[par][q] += 1
                pooled_q[n][par][q] += 1
            out_n = tf_dir / f"chromhmm18_x{n}"
            plot_quantile_panel(
                sq, n,
                title=(
                    f"ChromHMM-18 × {n}  distal cobinding  ({tf}, "
                    f"{len(pairs):,} pairs)"
                ),
                out_png=out_n / "distal_chromhmm_composition.png",
                n_pairs_total=len(pairs),
            )
            with (out_n / "distal_chromhmm_composition.tsv").open("w") as fh:
                fh.write(f"# {tf}\t{len(pairs)} pairs\tx{n}\n")
                fh.write("state\tq\tn_pairs\tpct_in_state\n")
                for st in EPI_ORDER:
                    tot = sum(sq[st].values())
                    for q in range(1, n + 1):
                        v = sq[st].get(q, 0)
                        pct = 100 * v / tot if tot else 0.0
                        fh.write(f"{st}\t{q}\t{v}\t{pct:.2f}\n")

    # pooled 18-state bar
    plot_state_bar(
        pooled18, EPI_ORDER + ["Unassigned"],
        title=(
            f"ChromHMM-18 composition of distal regions\n"
            f"all cobinding TFs | {pooled_n_pairs:,} unique distal-proximal pairs | "
            f"classified by the distal endpoint"
        ),
        out_png=args.out_dir / "pooled_chromhmm18_composition.png",
        colors=STATE_COLOR,
        min_label=6.0,
    )
    n18 = sum(pooled18.values())
    with (args.out_dir / "pooled_chromhmm18_composition.tsv").open("w") as fh:
        fh.write(f"# pooled\t{pooled_n_pairs} pairs\t{len(tfs)} TFs\n")
        fh.write("state\tn_pairs\tpct_pairs\n")
        for s in EPI_ORDER + ["Unassigned"]:
            fh.write(f"{s}\t{pooled18.get(s,0)}\t{100*pooled18.get(s,0)/max(n18,1):.2f}\n")

    for n in ns:
        plot_quantile_panel(
            pooled_q[n], n,
            title=(
                f"ChromHMM-18 × {n}  distal cobinding  "
                f"(all TFs, {pooled_n_pairs:,} pairs)"
            ),
            out_png=args.out_dir / f"x{n}" / "pooled_quantiles.png",
            n_pairs_total=pooled_n_pairs,
        )
        colors = q_cmap(n)
        color_map = {f"q{q}": colors[q - 1] for q in range(1, n + 1)}
        for st in EPI_ORDER:
            cq = pooled_q[n][st]
            counts = {f"q{q}": cq.get(q, 0) for q in range(1, n + 1)}
            tot = sum(counts.values())
            if tot < 2:
                continue
            plot_state_bar(
                counts, [f"q{q}" for q in range(1, n + 1)],
                title=(
                    f"ChromHMM composition of distal regions  ({st} × {n})\n"
                    f"all cobinding TFs | {tot:,} pairs in {st} | "
                    f"E1 quantiles q1 (B-like) → q{n} (A-like)"
                ),
                out_png=args.out_dir / f"x{n}" / f"{slug(st)}_composition.png",
                colors=color_map,
                min_label=8.0,
            )
        with (args.out_dir / f"x{n}" / "pooled_quantiles.tsv").open("w") as fh:
            fh.write(f"# pooled\tx{n}\t{pooled_n_pairs} pairs\n")
            fh.write("state\tq\tn_pairs\tpct_in_state\n")
            for st in EPI_ORDER:
                tot = sum(pooled_q[n][st].values())
                for q in range(1, n + 1):
                    v = pooled_q[n][st].get(q, 0)
                    pct = 100 * v / tot if tot else 0.0
                    fh.write(f"{st}\t{q}\t{v}\t{pct:.2f}\n")

    print(f"[done] pooled pairs {pooled_n_pairs:,}  -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
