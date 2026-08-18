#!/usr/bin/env python3
"""Replace CTCF rows in all_universe / openpos slide tables; keep other TFs."""
from __future__ import annotations

import csv
from pathlib import Path

import build_slide_all_universe as univ
import build_slide_all_universe_openpos as openpos

ROOT = Path(__file__).resolve().parent
SLIDE = ROOT / "slide_tables"


def fmt_pct(frac, nd: int = 1) -> str:
    if frac is None:
        return ""
    v = 100.0 * float(frac)
    s = f"{v:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def rows_from_gather(g: dict) -> list[dict]:
    au_u = g["pipes"]["unified"]["auroc"]
    au_r = g["pipes"]["raw"]["auroc"]
    delta = f"{au_u - au_r:+.4f}" if au_u is not None and au_r is not None else ""
    out = []
    for pipe in ("unified", "raw"):
        p = g["pipes"][pipe]
        out.append(
            {
                "TF": g["tf"],
                "Pipeline": pipe,
                "n_pos": g["n_pos"],
                "n_neg": g["n_neg"],
                "AUROC": f"{p['auroc']:.4f}" if p["auroc"] is not None else "",
                "Bin.pos %": fmt_pct(p["bin_pos"], 1),
                "Bin.neg %": fmt_pct(p["bin_neg"], 1),
                "Peak.pos %": fmt_pct(p["peak_pos"], 2),
                "Peak.neg %": fmt_pct(p["peak_neg"], 3),
                "ΔAUROC": delta if pipe == "unified" else "",
                "pos_mode": g.get("mode", ""),
                "neg_mode": "all_candidates",
            }
        )
    return out


def patch_csv(path: Path, ctcf_rows: list[dict]) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open() as fh:
        r = csv.DictReader(fh)
        fields = list(r.fieldnames or [])
        others = [row for row in r if row.get("TF", "").lower() != "ctcf"]
    # normalize fieldnames
    for row in ctcf_rows:
        for k in fields:
            row.setdefault(k, "")
    merged = ctcf_rows + others
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    return merged


def write_txt(path: Path, title: str, merged: list[dict]) -> None:
    # group by TF order: ctcf first then others as in csv
    hdr = (
        f"{'TF':<8} {'Pipeline':<10} {'n_pos':>7} {'n_neg':>7} {'AUROC':>8} "
        f"{'Bin.pos %':>10} {'Bin.neg %':>10} {'Peak.pos %':>11} "
        f"{'Peak.neg %':>11} {'ΔAUROC':>9}"
    )
    lines = [title, "", hdr, "-" * len(hdr)]
    for row in merged:
        auroc = row.get("AUROC", "")
        try:
            auroc_s = f"{float(auroc):>8.4f}"
        except (TypeError, ValueError):
            auroc_s = f"{auroc:>8}"
        lines.append(
            f"{row.get('TF',''):<8} {row.get('Pipeline',''):<10} "
            f"{str(row.get('n_pos','')):>7} {str(row.get('n_neg','')):>7} "
            f"{auroc_s} "
            f"{str(row.get('Bin.pos %','')):>10} {str(row.get('Bin.neg %','')):>10} "
            f"{str(row.get('Peak.pos %','')):>11} {str(row.get('Peak.neg %','')):>11} "
            f"{str(row.get('ΔAUROC','')):>9}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    g_u = univ.gather("ctcf")
    if g_u["n_pos"] is None or g_u["pipes"]["unified"]["auroc"] is None:
        raise SystemExit(f"CTCF all_universe gather incomplete: {g_u}")
    merged_u = patch_csv(SLIDE / "slide_all_universe.csv", rows_from_gather(g_u))
    write_txt(
        SLIDE / "slide_all_universe.txt",
        "All-universe (positives=all TF-bound cCRE bins; "
        "negatives=all zero-bulk-read non-TF-bound candidates)",
        merged_u,
    )
    print(f"updated {SLIDE / 'slide_all_universe.txt'} n_pos={g_u['n_pos']}")

    g_o = openpos.gather("ctcf")
    if g_o["n_pos"] is None or g_o["pipes"]["unified"]["auroc"] is None:
        raise SystemExit(f"CTCF openpos gather incomplete: {g_o}")
    merged_o = patch_csv(SLIDE / "slide_all_universe_openpos.csv", rows_from_gather(g_o))
    write_txt(
        SLIDE / "slide_all_universe_openpos.txt",
        "Open-pos universe (positives=all_bound ∩ OmniATAC open; "
        "negatives=all_candidates unchanged)",
        merged_o,
    )
    print(f"updated {SLIDE / 'slide_all_universe_openpos.txt'} n_pos={g_o['n_pos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
