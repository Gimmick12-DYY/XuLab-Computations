#!/usr/bin/env python3
"""Presentable TF table:

  TF | Pipeline | Cell Count | Mean UMIs per Cell
    | % Pos Peak coverage | % Neg Peak coverage | AUROC

Uses the same panel / modes as slide_top12k_q05:
  CTCF/MAZ/ZIC2 = final; others = peak_cutoff_q05.

Mean UMIs/cell = mean of per-cell column sums, computed separately for
each pipeline (raw mm vs unified imputed matrix).
Peak coverages from peak_coverage_*_table.txt / .tsv.
AUROC from compare_pos_neg_*.json.

Writes:
  downstream/slide_tables/presentable_tf_table.{txt,csv}
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SLIDE = ROOT / "slide_tables"
SLIDE.mkdir(exist_ok=True)
WORK = REPO / "unified" / "work"

PANEL = [
    ("ctcf", "final"),
    ("maz", "final"),
    ("zic2", "final"),
    ("zbtb7a", "peak_cutoff_q05"),
    ("znf777", "peak_cutoff_q05"),
    ("nfya", "peak_cutoff_q05"),
    ("znf282", "peak_cutoff_q05"),
]

TF_DISPLAY = {
    "ctcf": "CTCF",
    "maz": "MAZ",
    "zic2": "ZIC2",
    "zbtb7a": "ZBTB7A",
    "znf777": "ZNF777",
    "nfya": "NFYA",
    "znf282": "ZNF282",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = (
        path.read_text()
        .replace(": TRUE", ": true")
        .replace(": FALSE", ": false")
        .replace(": NA", ": null")
        .replace(": NaN", ": null")
    )
    text = re.sub(r":\s*NaN\b", ": null", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def pct_to_frac(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def parse_peak_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines()[2:]:
        if not line.strip() or line.startswith("-"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 2:
            continue
        label = parts[0].strip()
        rows[label] = {
            "pos_in_peaks_pct": parts[5] if len(parts) > 5 else "",
            "neg_in_peaks_pct": parts[6] if len(parts) > 6 else "",
            "auroc": parts[-1] if parts else "",
        }
    return rows


def mean_umis(tf: str, pipe: str) -> tuple[int, float]:
    """n_cells, mean UMIs/cell from the pipeline matrix (column sums)."""
    work = WORK / tf
    if pipe == "raw":
        mat = sio.mmread(str(work / "mm" / "matrix.mtx.gz")).tocsr()
    elif pipe == "unified":
        mat = sp.load_npz(work / "impute" / "matrix_csr.npz").tocsr()
    else:
        raise SystemExit(f"unknown pipeline {pipe}")
    depth = np.asarray(mat.sum(axis=0)).ravel().astype(np.float64)
    return int(depth.size), float(depth.mean())


def gather(tf: str, mode: str) -> dict:
    tag = f"{tf}_{mode}"
    cmp_ = load_json(ROOT / f"compare_pos_neg_{tag}" / f"compare_pos_neg_{tag}.json")
    peak = parse_peak_table(
        ROOT / f"peak_coverage_{tag}" / f"peak_coverage_{tag}_table.txt"
    )
    peak_tsv = ROOT / f"peak_coverage_{tag}" / f"peak_coverage_{tag}.tsv"

    auroc: dict[str, float] = {}
    for row in cmp_.get("inputs", []):
        if not isinstance(row, dict) or not row.get("label"):
            continue
        val = row.get("auroc_full", row.get("auroc"))
        if val is not None:
            auroc[row["label"]] = float(val)
    if not auroc:
        for pipe, st in peak.items():
            if st.get("auroc"):
                auroc[pipe] = float(st["auroc"])

    peak_pos: dict[str, float | None] = {
        pipe: pct_to_frac(st.get("pos_in_peaks_pct", "")) for pipe, st in peak.items()
    }
    peak_neg: dict[str, float | None] = {
        pipe: pct_to_frac(st.get("neg_in_peaks_pct", "")) for pipe, st in peak.items()
    }
    if peak_tsv.is_file():
        with peak_tsv.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lab = row.get("run") or row.get("label") or ""
                try:
                    if peak_pos.get(lab) is None:
                        peak_pos[lab] = float(row["positive_peak_coverage_pct"]) / 100.0
                    if peak_neg.get(lab) is None:
                        peak_neg[lab] = float(row["negative_peak_coverage_pct"]) / 100.0
                except (KeyError, ValueError, TypeError):
                    pass

    pipes = {}
    n_cells_ref = None
    for pipe in ("unified", "raw"):
        n_cells, mean_umi = mean_umis(tf, pipe)
        if n_cells_ref is None:
            n_cells_ref = n_cells
        elif n_cells != n_cells_ref:
            raise SystemExit(
                f"{tf}: cell count mismatch {pipe}={n_cells} vs {n_cells_ref}"
            )
        pipes[pipe] = {
            "auroc": auroc.get(pipe),
            "peak_pos": peak_pos.get(pipe),
            "peak_neg": peak_neg.get(pipe),
            "mean_umis": mean_umi,
        }
    return {
        "tf": TF_DISPLAY.get(tf, tf.upper()),
        "mode": mode,
        "n_cells": n_cells_ref,
        "pipes": pipes,
    }


def fmt_pct(frac: float | None, nd: int = 1) -> str:
    if frac is None:
        return ""
    v = 100.0 * float(frac)
    s = f"{v:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def neg_pct_nd(frac: float | None) -> int:
    if frac is None:
        return 1
    if frac < 0.01:
        return 3
    if frac < 0.1:
        return 2
    return 1


def fmt_umi(mean: float) -> str:
    if mean >= 100:
        return f"{mean:.0f}"
    if mean >= 10:
        return f"{mean:.1f}"
    return f"{mean:.2f}"


def main() -> int:
    rows = []
    missing = []
    for tf, mode in PANEL:
        print(f"[{tf}] loading metrics + per-pipeline mean UMIs...", flush=True)
        g = gather(tf, mode)
        if g["pipes"]["unified"]["auroc"] is None:
            missing.append(f"{tf}/{mode}")
        rows.append(g)
        print(
            f"  raw meanUMI={g['pipes']['raw']['mean_umis']:.2f}  "
            f"unified meanUMI={g['pipes']['unified']['mean_umis']:.2f}",
            flush=True,
        )

    if missing:
        print("MISSING: " + ", ".join(missing))
        return 1

    lines = [
        "Presentable table (CTCF/MAZ/ZIC2 = final top~12.8k; "
        "others = peak_cutoff q<0.05)",
        "Mean UMIs/cell = mean of per-cell column sums "
        "(raw mm vs unified imputed; pipeline-specific).",
        "",
    ]
    hdr = (
        f"{'TF':<8} {'Pipeline':<10} {'Cell Count':>10} "
        f"{'Mean UMIs/Cell':>15} {'Pos Peak %':>11} {'Neg Peak %':>11} "
        f"{'AUROC':>8}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    csv_rows = [
        [
            "TF",
            "Pipeline",
            "Cell Count",
            "Mean UMIs per Cell",
            "% Pos Peak coverage",
            "% Neg Peak coverage",
            "AUROC",
        ]
    ]

    for g in rows:
        for pipe in ("unified", "raw"):
            p = g["pipes"][pipe]
            mean = p["mean_umis"]
            line = (
                f"{g['tf']:<8} {pipe:<10} {g['n_cells']:>10} "
                f"{fmt_umi(mean):>15} {fmt_pct(p['peak_pos'], 1):>11} "
                f"{fmt_pct(p['peak_neg'], neg_pct_nd(p['peak_neg'])):>11} "
                f"{p['auroc']:>8.4f}"
            )
            lines.append(line)
            csv_rows.append(
                [
                    g["tf"],
                    pipe,
                    g["n_cells"],
                    f"{mean:.4f}",
                    fmt_pct(p["peak_pos"], 1),
                    fmt_pct(p["peak_neg"], 3),
                    f"{p['auroc']:.4f}",
                ]
            )

    txt = SLIDE / "presentable_tf_table.txt"
    csvp = SLIDE / "presentable_tf_table.csv"
    txt.write_text("\n".join(lines) + "\n")
    with csvp.open("w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(txt.read_text())
    print(f"Wrote {txt}")
    print(f"Wrote {csvp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
