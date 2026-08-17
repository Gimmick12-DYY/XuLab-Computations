#!/usr/bin/env python3
"""Pie chart: unified imputation progress across TF1000cells.meta.csv TFs.

Done = impute/matrix_csr.npz exists, OR currently running (count as done).
Waiting = remaining metadata TFs.

Usage:
  python3 downstream/plot_imputation_progress_pie.py
  python3 downstream/plot_imputation_progress_pie.py --running-as-done sox4,znf512
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
META = REPO / "data" / "TF1000cells.meta.csv"
WORK = REPO / "unified" / "work"
OUT_DIR = ROOT / "plots" / "imputation_progress"


def discover_running_tfs() -> set[str]:
    """Parse squeue job names for TF keys (best-effort)."""
    running: set[str] = set()
    try:
        out = subprocess.check_output(
            ["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip(),
             "-h", "-o", "%j %T"],
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return running
    # Job names often like unified_sox4, tf_sox4, sox4_impute, 99_full_pipeline (with TF in export)
    for line in out.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        name = parts[0].lower()
        state = parts[1] if len(parts) > 1 else ""
        if state not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            continue
        for token in name.replace("-", "_").split("_"):
            # heuristic: match work dir names later
            if token and token.isalpha() and len(token) >= 3:
                running.add(token)
    return running


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--running-as-done",
        default="",
        help="Comma-separated TF keys to count as done (overrides/adds to squeue).",
    )
    ap.add_argument(
        "--n-running",
        type=int,
        default=2,
        help="If squeue TF names unclear, assume this many running jobs count as done "
        "(default 2 per user request).",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META)
    tfs = sorted({str(t).strip() for t in meta["TF"].dropna().unique()})
    # work dirs are lowercase
    tf_keys = [(tf, tf.lower()) for tf in tfs]

    done_files: set[str] = set()
    for tf, key in tf_keys:
        if (WORK / key / "impute" / "matrix_csr.npz").is_file():
            done_files.add(key)

    # Running: explicit list, else squeue heuristic, else n_running placeholder
    explicit = {
        x.strip().lower()
        for x in args.running_as_done.split(",")
        if x.strip()
    }
    sq = discover_running_tfs()
    # only keep keys that are real metadata TFs and not already done
    meta_keys = {k for _, k in tf_keys}
    running = (explicit | (sq & meta_keys)) - done_files

    # If user said "two running" and we couldn't name them, still add 2 to done count
    unnamed_running = 0
    if not running and args.n_running > 0:
        # pick waiting TFs that have work dirs / configs as likely runners
        waiting_keys = [k for _, k in tf_keys if k not in done_files]
        # prefer those with a work dir already (pipeline started)
        started = [k for k in waiting_keys if (WORK / k).is_dir()]
        pool = started if started else waiting_keys
        for k in pool[: args.n_running]:
            running.add(k)
        unnamed_running = max(0, args.n_running - len(running))
        # if still short, just inflate done count
        if len(running) < args.n_running:
            unnamed_running = args.n_running - len(running)

    done = done_files | running
    n_total = len(tf_keys)
    n_done_files = len(done_files)
    n_running = len(running) + unnamed_running
    n_done = n_done_files + n_running  # running counted as completed
    # avoid double-count if running already in done_files
    n_done = len(done_files | running) + unnamed_running
    n_waiting = n_total - n_done
    if n_waiting < 0:
        n_waiting = 0
        n_done = n_total

    print(f"Metadata TFs: {n_total}")
    print(f"Impute on disk: {n_done_files}")
    print(f"Running (count as done): {n_running} -> {sorted(running) or '(unnamed)'}")
    print(f"Done (incl. running): {n_done}")
    print(f"Waiting: {n_waiting}")

    # Pie: Done vs Waiting (running folded into Done)
    sizes = [n_done, n_waiting]
    labels = [f"Done\n({n_done})", f"Waiting\n({n_waiting})"]
    colors = ["#2A9D8F", "#E9C46A"]
    explode = (0.02, 0.02)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        wedgeprops=dict(width=1.0, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=12),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title(
        f"Unified imputation progress\n"
        f"{n_done} / {n_total} TFs done "
        f"(incl. {n_running} running counted as done)",
        fontsize=13,
        pad=16,
    )
    # subtitle note
    fig.text(
        0.5,
        0.02,
        f"Source: TF1000cells.meta.csv ({n_total} unique TFs); "
        f"done = impute/matrix_csr.npz present",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    png = args.out_dir / "imputation_progress_pie.png"
    pdf = args.out_dir / "imputation_progress_pie.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

    # also write a small TSV summary
    tsv = args.out_dir / "imputation_progress_summary.tsv"
    with tsv.open("w") as f:
        f.write("category\tn\n")
        f.write(f"done_on_disk\t{n_done_files}\n")
        f.write(f"running_as_done\t{n_running}\n")
        f.write(f"done_total\t{n_done}\n")
        f.write(f"waiting\t{n_waiting}\n")
        f.write(f"total_meta_tfs\t{n_total}\n")
        f.write(f"running_tfs\t{','.join(sorted(running)) if running else ''}\n")
    print(f"Wrote {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
