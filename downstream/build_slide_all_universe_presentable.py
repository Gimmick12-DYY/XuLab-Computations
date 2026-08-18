#!/usr/bin/env python3
"""Build the presentation all_universe table (post open-chromatin filter).

Format matches the slide "After applying the filter" layout, with TF order:
  nfya, ctcf, zbtb7a, zic2, maz, znf777, znf282
plus a Mean frags/cell column (pipeline-specific column-sum mean).

Reads metrics from slide_all_universe.csv and mean frags from
mean_frags_per_cell_raw_vs_unified.csv (falls back to recomputing).

Writes:
  downstream/slide_tables/slide_all_universe_presentable.{txt,csv}
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDE = ROOT / "slide_tables"

# Exact display order from the existing presentation table.
TF_ORDER = ["nfya", "ctcf", "zbtb7a", "zic2", "maz", "znf777", "znf282"]
TF_DISP = {
    "nfya": "nfya",
    "ctcf": "ctcf",
    "zbtb7a": "zbtb7a",
    "zic2": "zic2",
    "maz": "maz",
    "znf777": "znf777",
    "znf282": "znf282",
}


def load_slide_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            tf = row["TF"].strip().lower()
            pipe = row["Pipeline"].strip().lower()
            rows[(tf, pipe)] = row
    return rows


def load_mean_frags(path: Path) -> dict[tuple[str, str], str]:
    """Return (tf, pipe) -> formatted mean frags/cell."""
    out: dict[tuple[str, str], str] = {}
    if not path.is_file():
        return out
    with path.open() as fh:
        r = csv.DictReader(fh)
        # expect TF, n_cells, mean_raw, mean_unified, ...
        for row in r:
            tf = row.get("TF") or row.get("tf") or ""
            tf = tf.strip().lower()
            if tf not in TF_ORDER:
                continue
            raw = row.get("mean_raw") or row.get("Mean raw") or ""
            uni = row.get("mean_unified") or row.get("Mean unified") or ""
            if raw != "":
                out[(tf, "raw")] = _fmt_frag(raw)
            if uni != "":
                out[(tf, "unified")] = _fmt_frag(uni)
    return out


def _fmt_frag(v: str) -> str:
    try:
        x = float(v)
    except ValueError:
        return v
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".") if "." in f"{x:.1f}" else f"{x:.1f}"


def fmt_num(s: str) -> str:
    return str(s).strip()


def main() -> int:
    slide = load_slide_rows(SLIDE / "slide_all_universe.csv")
    frags = load_mean_frags(SLIDE / "mean_frags_per_cell_raw_vs_unified.csv")
    if not frags:
        # fallback: presentable table naming
        p = SLIDE / "presentable_tf_table.csv"
        if p.is_file():
            with p.open() as fh:
                for row in csv.DictReader(fh):
                    tf = row["TF"].strip().lower()
                    pipe = row["Pipeline"].strip().lower()
                    key = "Mean UMIs per Cell" if "Mean UMIs per Cell" in row else "Mean UMIs/Cell"
                    if key in row and tf in TF_ORDER:
                        frags[(tf, pipe)] = _fmt_frag(row[key])

    missing = [tf for tf in TF_ORDER if (tf, "unified") not in slide]
    if missing:
        raise SystemExit(f"missing TFs in slide_all_universe.csv: {missing}")

    title = (
        "After applying the filter (all_universe; open-chromatin-masked impute)\n"
        "Mean frags/cell = mean of per-cell column sums (raw mm vs unified imputed)."
    )
    hdr = (
        f"{'TF':<8} {'Pipeline':<10} {'n_pos':>7} {'n_neg':>7} {'AUROC':>8} "
        f"{'Bin.pos %':>10} {'Bin.neg %':>10} {'Peak.pos %':>11} {'Peak.neg %':>11} "
        f"{'Mean frags/cell':>15}"
    )
    lines = [title, "", hdr, "-" * len(hdr)]
    csv_rows = [
        [
            "TF",
            "Pipeline",
            "n_pos",
            "n_neg",
            "AUROC",
            "Bin.pos %",
            "Bin.neg %",
            "Peak.pos %",
            "Peak.neg %",
            "Mean frags/cell",
        ]
    ]

    for tf in TF_ORDER:
        for pipe in ("raw", "unified"):
            # Image order is raw then unified per TF
            r = slide[(tf, pipe)]
            frag = frags.get((tf, pipe), "")
            line = (
                f"{TF_DISP[tf]:<8} {pipe:<10} "
                f"{fmt_num(r['n_pos']):>7} {fmt_num(r['n_neg']):>7} "
                f"{float(r['AUROC']):>8.4f} "
                f"{fmt_num(r['Bin.pos %']):>10} {fmt_num(r['Bin.neg %']):>10} "
                f"{fmt_num(r['Peak.pos %']):>11} {fmt_num(r['Peak.neg %']):>11} "
                f"{frag:>15}"
            )
            lines.append(line)
            csv_rows.append(
                [
                    TF_DISP[tf],
                    pipe,
                    r["n_pos"],
                    r["n_neg"],
                    f"{float(r['AUROC']):.4f}",
                    r["Bin.pos %"],
                    r["Bin.neg %"],
                    r["Peak.pos %"],
                    r["Peak.neg %"],
                    frag,
                ]
            )

    out_txt = SLIDE / "slide_all_universe_presentable.txt"
    out_csv = SLIDE / "slide_all_universe_presentable.csv"
    out_txt.write_text("\n".join(lines) + "\n")
    with out_csv.open("w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    print(f"Wrote {out_txt}")
    print(f"Wrote {out_csv}")
    print(out_txt.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
