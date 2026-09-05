#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# summarize_motif_table.py
#
# Fill the per-TF motif-recovery table: for each TF, the p-value of its OWN motif
# under each condition -- Raw / Imputed(whole-genome bg) / Imputed(OCR bg), each x
# {peaks, bins} x {AME, HOMER}. Extracted from the motif_enrichment.sbatch output
# trees (homer_{genome,ocr}/knownResults.txt, ame_{genome,ocr}/<tf>/ame.tsv).
#
#   HOMER value = P-value of the TF's own motif in knownResults.txt (col 3)
#   AME   value = adj_p-value of the TF's own motif in ame.tsv (col adj_p-value)
#   NA = TF's own motif absent (not in that DB / not called); blank = condition not run.
#
# Each condition = a root dir whose <tf>/ has homer_<bg>/ + ame_<bg>/. Point --*-root at
# the peaks tree (downstream/motif) and/or a bins tree; genome vs OCR come from _<bg>.
#
#   python downstream/summarize_motif_table.py \
#     --imp-peaks-root downstream/motif --imp-bins-root downstream/motif_bins \
#     --raw-peaks-root downstream/motif_raw \
#     --manifest downstream/motif_reference_manifest.tsv \
#     --cell-meta data/TF1000cells.meta.csv --out downstream/motif_recovery_table.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def tok(name):
    """gene token: 'NRF1(NRF)/..' -> nrf1 ; 'NRF1_HUMAN.H11..' -> nrf1 ; 'Sox4' -> sox4."""
    return re.split(r"[(/_.]", name.strip())[0].lower()


def homer_own(cdir, tf):
    """P-value (col 3) of the TF's own motif in knownResults.txt, or 'NA'."""
    f = cdir / "knownResults.txt"
    if not f.is_file():
        return ""
    best = None
    with open(f) as fh:
        next(fh, None)
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            if tok(p[0]) == tf.lower():
                try:
                    v = float(p[2])
                except ValueError:
                    continue
                if best is None or v < best:
                    best = v
    return f"{best:.2E}" if best is not None else "NA"


def ame_own(cdir, tf):
    """adj_p-value of the TF's own motif in ame.tsv, or 'NA' (AME lists only significant)."""
    f = cdir / "ame.tsv"
    if not f.is_file():
        f = cdir / tf / "ame.tsv"           # ame_<bg>/<tf>/ame.tsv
    if not f.is_file():
        return ""
    best = None
    with open(f) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not (r.get("rank", "").strip().isdigit()):
                continue
            mid = r.get("motif_ID", "") or r.get("motif_alt_ID", "")
            if tok(mid) == tf.lower():
                try:
                    v = float(r.get("adj_p-value", "nan"))
                except (ValueError, TypeError):
                    continue
                if best is None or v < best:
                    best = v
    return f"{best:.2E}" if best is not None else "NA"


def cond_pvals(root, tf, bg):
    """(AME, HOMER) own-motif p for one condition = root/<tf>/{ame_<bg>, homer_<bg>}."""
    if root is None:
        return "", ""
    base = Path(root) / tf
    ame = ame_own(base / f"ame_{bg}", tf) if (base / f"ame_{bg}").exists() else \
        ame_own(base / "ame", tf)
    homer = homer_own(base / f"homer_{bg}", tf) if (base / f"homer_{bg}").exists() else \
        homer_own(base / "homer", tf)
    return ame, homer


def load_cells(path):
    cc = {}
    if path and Path(path).is_file():
        for r in csv.reader(open(path)):
            if len(r) >= 2 and r[1].strip().isdigit():
                cc[r[0].strip().lower()] = int(r[1])
    return cc


def load_manifest(path):
    """tf -> (homer_db 'yes'/'no', hocomoco_full grade)."""
    man = {}
    if path and Path(path).is_file():
        for ln in open(path):
            if ln.startswith("#") or ln.startswith("tf\t"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 4:
                man[p[0].lower()] = (p[1], p[3])
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imp-peaks-root", default=None, help="imputed motif tree (peaks): downstream/motif")
    ap.add_argument("--imp-bins-root", default=None, help="imputed motif tree (bins)")
    ap.add_argument("--raw-peaks-root", default=None)
    ap.add_argument("--raw-bins-root", default=None)
    ap.add_argument("--cell-meta", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tfs", nargs="*", default=None, help="TF order; default = discovered, by cell count desc")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = load_cells(args.cell_meta)
    man = load_manifest(args.manifest)

    # discover TFs
    tfs = args.tfs
    if not tfs:
        seen = set()
        for root in (args.imp_peaks_root, args.imp_bins_root, args.raw_peaks_root, args.raw_bins_root):
            if root and Path(root).is_dir():
                seen |= {d.name for d in Path(root).iterdir() if d.is_dir()}
        seen |= set(cells) | set(man)
        tfs = sorted(seen, key=lambda t: cells.get(t.lower(), 0), reverse=True)

    hdr = ["TF", "CellCount",
           "Raw_AME_peaks", "Raw_HOMER_peaks", "Raw_AME_bins", "Raw_HOMER_bins",
           "ImpGenome_AME_peaks", "ImpGenome_HOMER_peaks", "ImpGenome_AME_bins", "ImpGenome_HOMER_bins",
           "ImpOCR_AME_peaks", "ImpOCR_HOMER_peaks", "ImpOCR_AME_bins", "ImpOCR_HOMER_bins",
           "HOMER_db", "HOCOMOCOv11_db"]
    rows = [hdr]
    for tf in tfs:
        t = tf.lower()
        rap, rhp = cond_pvals(args.raw_peaks_root, tf, "genome")   # raw has one bg (genome-style)
        rab, rhb = cond_pvals(args.raw_bins_root, tf, "genome")
        gap, ghp = cond_pvals(args.imp_peaks_root, tf, "genome")
        gab, ghb = cond_pvals(args.imp_bins_root, tf, "genome")
        oap, ohp = cond_pvals(args.imp_peaks_root, tf, "ocr")
        oab, ohb = cond_pvals(args.imp_bins_root, tf, "ocr")
        hdb = "yes" if man.get(t, ("no",))[0].lower() == "yes" else ""
        hoco = man.get(t, ("", ""))[1] if t in man else ""
        rows.append([tf.upper(), str(cells.get(t, "")),
                     rap, rhp, rab, rhb, gap, ghp, gab, ghb, oap, ohp, oab, ohb, hdb, hoco])

    text = "\n".join("\t".join(r) for r in rows) + "\n"
    if args.out:
        args.out.write_text(text); print(f"[done] {len(rows)-1} TFs -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
