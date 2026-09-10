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
#   F  = target motif not found in results (run completed, TF is in that library)
#   NA = TF not present in that library (HOMER / HOCOMOCO)
#   blank = condition not run
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


# HOCOMOCO / HOMER sometimes ship the TF under an alias (CTCFL = BORIS is the
# CTCF-like paralog; we still accept it as a hit for CTCF when no CTCF_HUMAN row
# is present). Key = query TF (lowercase) -> acceptable motif-token set.
_ALIASES = {
    "ctcf": {"ctcf", "ctcfl"},
}


def _tokens_for(tf: str) -> set[str]:
    t = tf.lower()
    return _ALIASES.get(t, {t})


def _p_key(raw: str):
    """Sort key for p-value strings; survives float64 underflow (e.g. HOMER 1e-394)."""
    s = (raw or "").strip().lower()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v > 0:
        return v
    # Underflowed to 0.0 — compare by exponent (more negative = smaller p).
    if "e" in s:
        try:
            return (0.0, float(s.split("e")[1]))   # tuple sorts after any positive float? No.
        except ValueError:
            return None
    return 0.0


def _fmt_p(raw: str) -> str | None:
    """Format without losing exponents below ~1e-308."""
    s = (raw or "").strip()
    if not s:
        return None
    sl = s.lower()
    try:
        v = float(s)
    except ValueError:
        return None
    if v > 0:
        return f"{v:.2E}"
    if "e" in sl:                              # underflowed, e.g. 1e-394
        mant, _, exp = sl.partition("e")
        try:
            return f"{float(mant):.2f}E{int(exp)}"
        except ValueError:
            return s.upper()
    return "0.00E+00"


def _better(raw_a, raw_b) -> bool:
    """True if raw_a is a smaller p than raw_b (or b is missing)."""
    if raw_b is None:
        return True
    sa, sb = raw_a.strip().lower(), raw_b.strip().lower()
    fa, fb = float(sa), float(sb)
    if fa > 0 and fb > 0:
        return fa < fb
    # At least one underflowed: compare exponents (smaller/more-neg wins).
    def exp(s, v):
        if v > 0:
            return __import__("math").log10(v)
        if "e" in s:
            return float(s.split("e")[1])
        return float("-inf")
    return exp(sa, fa) < exp(sb, fb)


def homer_own(cdir, tf):
    """P-value (col 3) of the TF's own motif in knownResults.txt, or 'NA'."""
    f = cdir / "knownResults.txt"
    if not f.is_file():
        return ""
    want = _tokens_for(tf)
    best = None
    with open(f) as fh:
        next(fh, None)
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or tok(p[0]) not in want:
                continue
            if _p_key(p[2]) is None:
                continue
            if _better(p[2], best):
                best = p[2]
    return _fmt_p(best) if best is not None else "NA"


def ame_own(cdir, tf):
    """adj_p-value of the TF's own motif in ame.tsv, or 'NA' (AME lists only significant)."""
    f = cdir / "ame.tsv"
    if not f.is_file():
        f = cdir / tf / "ame.tsv"           # ame_<bg>/<tf>/ame.tsv
    if not f.is_file():
        return ""
    want = _tokens_for(tf)
    best = None
    with open(f) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not (r.get("rank", "").strip().isdigit()):
                continue
            mid = r.get("motif_ID", "") or r.get("motif_alt_ID", "")
            if tok(mid) not in want:
                continue
            raw = (r.get("adj_p-value") or "").strip()
            if _p_key(raw) is None:
                continue
            if _better(raw, best):
                best = raw
    return _fmt_p(best) if best is not None else "NA"


def _cond_dirs(base: Path, kind: str, bg: str):
    """Prefer kind_bg (e.g. ame_genome). Legacy kind/ only for genome bg, never OCR."""
    tagged = base / f"{kind}_{bg}"
    if tagged.exists():
        return tagged
    if bg == "genome":
        legacy = base / kind
        if legacy.exists():
            return legacy
    return None


def cond_pvals(root, tf, bg, in_ame: bool, in_homer: bool):
    """(AME, HOMER) coded cells for one condition."""
    if root is None:
        return "", ""
    base = Path(root) / tf
    ad = _cond_dirs(base, "ame", bg)
    hd = _cond_dirs(base, "homer", bg)
    ame = _code(ame_own(ad, tf) if ad else "", in_ame, ad is not None and (
        (ad / "ame.tsv").is_file() or (ad / tf / "ame.tsv").is_file()))
    homer = _code(homer_own(hd, tf) if hd else "", in_homer,
                  hd is not None and (hd / "knownResults.txt").is_file())
    return ame, homer


def _code(raw: str, in_lib: bool, ran: bool) -> str:
    if not ran:
        return ""
    if not in_lib:
        return "NA"
    if raw in ("", "NA"):
        return "F"
    return raw


def load_cells(path):
    """Accept TF,count or per-cell TF1000cells.meta.csv (count rows per TF)."""
    cc = {}
    if not path or not Path(path).is_file():
        return cc
    with open(path) as fh:
        rd = csv.reader(fh)
        rows = list(rd)
    if not rows:
        return cc
    hdr = [h.strip() for h in rows[0]]
    if "TF" in hdr:
        i = hdr.index("TF")
        for r in rows[1:]:
            if len(r) <= i:
                continue
            tf = r[i].strip().lower()
            if tf:
                cc[tf] = cc.get(tf, 0) + 1
        return cc
    for r in rows:
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
        hm, hoco_raw = man.get(t, ("no", "Missing"))
        in_homer = hm.lower() == "yes"
        in_ame = hoco_raw.lower() not in ("", "missing")
        rap, rhp = cond_pvals(args.raw_peaks_root, tf, "genome", in_ame, in_homer)
        rab, rhb = cond_pvals(args.raw_bins_root, tf, "genome", in_ame, in_homer)
        gap, ghp = cond_pvals(args.imp_peaks_root, tf, "genome", in_ame, in_homer)
        gab, ghb = cond_pvals(args.imp_bins_root, tf, "genome", in_ame, in_homer)
        oap, ohp = cond_pvals(args.imp_peaks_root, tf, "ocr", in_ame, in_homer)
        oab, ohb = cond_pvals(args.imp_bins_root, tf, "ocr", in_ame, in_homer)
        hdb = tf.upper() if in_homer else "#N/A"
        hoco = hoco_raw if in_ame else "NA"
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
