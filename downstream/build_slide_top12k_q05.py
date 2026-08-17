#!/usr/bin/env python3
"""Build slide table: CTCF/MAZ/ZIC2 = final top-12805; others = peak_cutoff_q05.

Excludes ZNF768. Writes:
  downstream/slide_tables/slide_top12k_q05.{txt,csv}
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDE = ROOT / "slide_tables"
SLIDE.mkdir(exist_ok=True)

# (tf, mode)
PANEL = [
    ("ctcf", "final"),
    ("maz", "final"),
    ("zic2", "final"),
    ("zbtb7a", "peak_cutoff_q05"),
    ("znf777", "peak_cutoff_q05"),
    ("nfya", "peak_cutoff_q05"),
    ("znf282", "peak_cutoff_q05"),
]


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
            "pos_valued_pct": parts[3] if len(parts) > 3 else "",
            "neg_valued_pct": parts[4] if len(parts) > 4 else "",
            "pos_in_peaks_pct": parts[5] if len(parts) > 5 else "",
            "neg_in_peaks_pct": parts[6] if len(parts) > 6 else "",
            "auroc": parts[-1] if parts else "",
        }
    return rows


def fmt_pct(frac: float | None, nd: int = 1) -> str:
    if frac is None:
        return ""
    v = 100.0 * float(frac)
    s = f"{v:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def gather(tf: str, mode: str) -> dict:
    tag = f"{tf}_{mode}"
    bins = load_json(ROOT / f"bins_{tf}_{mode}" / "pos_neg_bins.meta.json")
    cmp_ = load_json(ROOT / f"compare_pos_neg_{tag}" / f"compare_pos_neg_{tag}.json")
    peak = parse_peak_table(
        ROOT / f"peak_coverage_{tag}" / f"peak_coverage_{tag}_table.txt"
    )
    binj = load_json(ROOT / f"bin_coverage_{tag}" / f"bin_coverage_{tag}.json")

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

    bin_cov: dict[str, dict[str, float]] = {}
    for row in binj.get("inputs", []):
        if not isinstance(row, dict) or not row.get("label"):
            continue
        bin_cov[row["label"]] = {
            "pos": float((row.get("pos") or {}).get("coverage", float("nan"))),
            "neg": float((row.get("neg") or {}).get("coverage", float("nan"))),
        }

    out = {"tf": tf, "mode": mode, "n_pos": bins.get("n_pos"), "pipes": {}}
    for pipe in ("unified", "raw"):
        bp = bin_cov.get(pipe, {}).get("pos")
        bn = bin_cov.get(pipe, {}).get("neg")
        if bp is None or (isinstance(bp, float) and bp != bp):
            bp = pct_to_frac(peak.get(pipe, {}).get("pos_valued_pct", ""))
        if bn is None or (isinstance(bn, float) and bn != bn):
            bn = pct_to_frac(peak.get(pipe, {}).get("neg_valued_pct", ""))
        out["pipes"][pipe] = {
            "auroc": auroc.get(pipe),
            "bin_pos": bp,
            "bin_neg": bn,
            "peak_pos": pct_to_frac(peak.get(pipe, {}).get("pos_in_peaks_pct", "")),
            "peak_neg": pct_to_frac(peak.get(pipe, {}).get("neg_in_peaks_pct", "")),
        }
    return out


def main() -> int:
    rows = []
    missing = []
    for tf, mode in PANEL:
        g = gather(tf, mode)
        if g["n_pos"] is None or g["pipes"]["unified"]["auroc"] is None:
            missing.append(f"{tf}/{mode}")
        rows.append(g)

    if missing:
        print("MISSING (not written): " + ", ".join(missing))
        return 1

    lines = []
    lines.append(
        "Slide (CTCF/MAZ/ZIC2 top ~12.8k; others peak_cutoff q<0.05; ZNF768 dropped)"
    )
    lines.append("")
    hdr = (
        f"{'TF':<8} {'Pipeline':<10} {'n_pos':>7} {'AUROC':>8} "
        f"{'Bin.pos %':>10} {'Bin.neg %':>10} {'Peak.pos %':>11} "
        f"{'Peak.neg %':>11} {'ΔAUROC':>9}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    csv_rows = [
        [
            "TF",
            "Pipeline",
            "n_pos",
            "AUROC",
            "Bin.pos %",
            "Bin.neg %",
            "Peak.pos %",
            "Peak.neg %",
            "ΔAUROC",
            "pos_mode",
        ]
    ]

    for g in rows:
        au_u = g["pipes"]["unified"]["auroc"]
        au_r = g["pipes"]["raw"]["auroc"]
        delta = f"{au_u - au_r:+.4f}" if au_u is not None and au_r is not None else ""
        for pipe in ("unified", "raw"):
            p = g["pipes"][pipe]
            dlt = delta if pipe == "unified" else ""
            line = (
                f"{g['tf']:<8} {pipe:<10} {g['n_pos']:>7} {p['auroc']:>8.4f} "
                f"{fmt_pct(p['bin_pos'],1):>10} {fmt_pct(p['bin_neg'],1):>10} "
                f"{fmt_pct(p['peak_pos'],2):>11} {fmt_pct(p['peak_neg'],3):>11} "
                f"{dlt:>9}"
            )
            lines.append(line)
            csv_rows.append(
                [
                    g["tf"],
                    pipe,
                    g["n_pos"],
                    f"{p['auroc']:.4f}",
                    fmt_pct(p["bin_pos"], 1),
                    fmt_pct(p["bin_neg"], 1),
                    fmt_pct(p["peak_pos"], 2),
                    fmt_pct(p["peak_neg"], 3),
                    dlt,
                    g["mode"],
                ]
            )

    txt = SLIDE / "slide_top12k_q05.txt"
    csvp = SLIDE / "slide_top12k_q05.csv"
    txt.write_text("\n".join(lines) + "\n")
    with csvp.open("w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(txt.read_text())
    print(f"Wrote {txt}")
    print(f"Wrote {csvp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
