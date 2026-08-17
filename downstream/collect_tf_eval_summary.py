#!/usr/bin/env python3
"""Collect TF benchmark metrics into a tidy long-form TSV.

Kept metrics (coverage as fractions in [0, 1]):
  n_pos / n_neg
      Reference bin counts (pipeline=ref).
  auroc
      Per-bin signal AUROC over the FULL pos/neg universe (zeros included).
      Matches historical peak_coverage / slide-table definition. Taken from
      compare_pos_neg auroc_full when available.
  bin_pos_cov / bin_neg_cov
      Fraction of reference bins with any nonzero signal (Pos/Neg.cov %).
      Prefer bin_coverage_*.json when present (supports sparsity-matched mode);
      else fall back to peak_coverage table valued %.
  peak_pos_cov / peak_neg_cov
      Fraction of reference bins overlapping a MACS3 peak (Pos/Neg in peaks %).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TFS = ["ctcf", "zbtb7a", "zic2", "maz", "znf777", "nfya", "znf282"]
# `final` = CTCF/MAZ/ZIC2 top ~12.8k; others all_bound.
# `slide_q05` assembled separately (top12k + peak_cutoff_q05).
MODES = ["final", "fixed_n", "all_bound", "peak_cutoff", "peak_cutoff_q05"]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text()
    text = (
        text.replace(": TRUE", ": true")
        .replace(": FALSE", ": false")
        .replace(": NA", ": null")
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _pct_to_frac(s: str) -> str:
    if s is None or s == "":
        return ""
    try:
        return f"{float(s) / 100.0:.6g}"
    except ValueError:
        return s


def parse_peak_table(path: Path) -> dict[str, dict[str, str]]:
    """Parse peak_coverage_*_table.txt -> per-pipeline stats."""
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
            "pos_valued_pct": parts[3] if len(parts) > 3 else "",
            "neg_valued_pct": parts[4] if len(parts) > 4 else "",
            "pos_in_peaks_pct": parts[5] if len(parts) > 5 else "",
            "neg_in_peaks_pct": parts[6] if len(parts) > 6 else "",
            "auroc": parts[-1] if parts else "",
        }
    return rows


def parse_bin_coverage(path: Path) -> dict[str, dict[str, float]]:
    """Parse bin_coverage_*.json -> per-pipeline pos/neg coverage fractions."""
    data = load_json(path)
    out: dict[str, dict[str, float]] = {}
    for row in data.get("inputs", []):
        if not isinstance(row, dict) or not row.get("label"):
            continue
        pos = row.get("pos") or {}
        neg = row.get("neg") or {}
        out[row["label"]] = {
            "bin_pos_cov": float(pos.get("coverage", float("nan"))),
            "bin_neg_cov": float(neg.get("coverage", float("nan"))),
        }
    return out


def bins_dir_for(tf: str, mode: str) -> Path:
    return ROOT / f"bins_{tf}_{mode}"


def emit_run(
    tf: str,
    mode: str,
    bins_meta: dict,
    cmp: dict,
    peak_stats: dict[str, dict],
    bin_stats: dict[str, dict[str, float]],
) -> None:
    if bins_meta.get("n_pos") is not None:
        print(f"{tf}\t{mode}\tn_pos\tref\t{bins_meta['n_pos']}")
    if bins_meta.get("n_neg") is not None:
        print(f"{tf}\t{mode}\tn_neg\tref\t{bins_meta['n_neg']}")

    auroc_by_pipe: dict[str, str] = {}
    for row in cmp.get("inputs", []):
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if not label:
            continue
        val = row.get("auroc_full", row.get("auroc"))
        if val is not None:
            auroc_by_pipe[label] = f"{float(val):.4f}"
    if not auroc_by_pipe:
        for pipe, stats in peak_stats.items():
            if stats.get("auroc"):
                auroc_by_pipe[pipe] = stats["auroc"]

    pipes = list(
        dict.fromkeys(
            [*auroc_by_pipe.keys(), *peak_stats.keys(), *bin_stats.keys()]
        )
    )
    pipes = [p for p in pipes if p != "raw"] + [p for p in pipes if p == "raw"]

    for pipe in pipes:
        if pipe in auroc_by_pipe:
            print(f"{tf}\t{mode}\tauroc\t{pipe}\t{auroc_by_pipe[pipe]}")

        if pipe in bin_stats:
            bp = bin_stats[pipe].get("bin_pos_cov")
            bn = bin_stats[pipe].get("bin_neg_cov")
            if bp == bp:  # not NaN
                print(f"{tf}\t{mode}\tbin_pos_cov\t{pipe}\t{bp:.6g}")
            if bn == bn:
                print(f"{tf}\t{mode}\tbin_neg_cov\t{pipe}\t{bn:.6g}")
        else:
            stats = peak_stats.get(pipe)
            if stats:
                print(
                    f"{tf}\t{mode}\tbin_pos_cov\t{pipe}\t"
                    f"{_pct_to_frac(stats.get('pos_valued_pct', ''))}"
                )
                print(
                    f"{tf}\t{mode}\tbin_neg_cov\t{pipe}\t"
                    f"{_pct_to_frac(stats.get('neg_valued_pct', ''))}"
                )

        stats = peak_stats.get(pipe)
        if stats:
            print(
                f"{tf}\t{mode}\tpeak_pos_cov\t{pipe}\t"
                f"{_pct_to_frac(stats.get('pos_in_peaks_pct', ''))}"
            )
            print(
                f"{tf}\t{mode}\tpeak_neg_cov\t{pipe}\t"
                f"{_pct_to_frac(stats.get('neg_in_peaks_pct', ''))}"
            )


def main() -> None:
    print("tf\tmode\tmetric\tpipeline\tvalue")
    for tf in TFS:
        for mode in MODES:
            tag = f"{tf}_{mode}"
            emit_run(
                tf,
                mode,
                load_json(bins_dir_for(tf, mode) / "pos_neg_bins.meta.json"),
                load_json(
                    ROOT / f"compare_pos_neg_{tag}" / f"compare_pos_neg_{tag}.json"
                ),
                parse_peak_table(
                    ROOT / f"peak_coverage_{tag}" / f"peak_coverage_{tag}_table.txt"
                ),
                parse_bin_coverage(
                    ROOT / f"bin_coverage_{tag}" / f"bin_coverage_{tag}.json"
                ),
            )


if __name__ == "__main__":
    main()
