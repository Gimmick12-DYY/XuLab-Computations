#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# flag_motif_artifacts.py   (Novel Motif Finding, Step 2)
#
# Annotate de-novo motifs with artifact flags so Step-3 rankings can be filtered:
#   1. CONTAMINANT: best TOMTOM match (vs HOCOMOCO v12 + Codebook) is an open-
#      chromatin contaminant family (CTCF, NFY, YY1, SP/KLF, ETS) that shows up
#      regardless of the target TF. Families are configurable (--families) so the
#      Codebook contaminant set can be plugged in verbatim.
#   2. TE/REPEAT (optional): if --repeat-bed + --instances-bed are given, flag
#      motifs whose instances pile up in repeats/transposable elements.
#
# Joins onto the Step-3 ranked table (if --ranked) and writes a `pass` column
# (= not contaminant and not TE-clustered), so the shortlist is
#   high AUROC  AND  pass == True.
#
#   python flag_motif_artifacts.py --tomtom tomtom.tsv --ranked ranked.tsv \
#       --out annotated.tsv
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import re
from pathlib import Path

# contaminant families (open-chromatin, target-independent). Matched against the
# TOMTOM target id/name with the merge tag (h12|, cdbk|) stripped.
DEFAULT_FAMILIES = {
    "CTCF": r"^(CTCF|CTCFL|BORIS)\b",
    "NFY": r"^(NFYA|NFYB|NFYC|NFY)\b",
    "YY1": r"^(YY1|YY2)\b",
    "SP/KLF": r"^(SP\d+|KLF\d+)\b",
    "ETS": r"^(ETS\d*|ELK\d+|ELF\d+|ETV\d+|GABPA|FLI1|ERG|EHF|SPI[B1]?|FEV|ERF)\b",
}


def strip_tag(tid: str) -> str:
    """'h12|CTCF.H12CORE.0.A' -> 'CTCF.H12CORE.0.A'; take gene-ish leading token."""
    t = tid.split("|", 1)[-1]
    return t


def family_of(target: str, pats: dict) -> str | None:
    t = strip_tag(target).upper()
    for fam, rx in pats.items():
        if re.search(rx, t):
            return fam
    return None


def best_tomtom(path: Path):
    """Query_ID -> (target_id, q_value) best (smallest q) match from tomtom.tsv."""
    best: dict[str, tuple[str, float]] = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        col = {h.strip(): i for i, h in enumerate(hdr)}
        qi = col.get("Query_ID", 0)
        ti = col.get("Target_ID", 1)
        qq = col.get("q-value", col.get("q_value", 5))
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) <= max(qi, ti, qq) or not p[qi] or p[qi].startswith("#"):
                continue
            try:
                q = float(p[qq])
            except ValueError:
                continue
            cur = best.get(p[qi])
            if cur is None or q < cur[1]:
                best[p[qi]] = (p[ti], q)
    return best


def read_ranked(path: Path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for ln in fh:
            rows.append(ln.rstrip("\n").split("\t"))
    return hdr, rows


def repeat_overlap_frac(instances_bed: Path, repeat_bed: Path) -> dict[str, float]:
    """Per-motif fraction of instances overlapping repeats (needs bedtools)."""
    import shutil
    import subprocess
    if not shutil.which("bedtools"):
        print("[warn] bedtools not found; skipping TE check", flush=True)
        return {}
    # instances_bed: chrom start end motif_id ...
    tot: dict[str, int] = {}
    hit: dict[str, int] = {}
    with open(instances_bed) as fh:
        for ln in fh:
            p = ln.split("\t")
            if len(p) >= 4:
                tot[p[3]] = tot.get(p[3], 0) + 1
    res = subprocess.run(["bedtools", "intersect", "-u", "-a", str(instances_bed),
                          "-b", str(repeat_bed)], capture_output=True, text=True)
    for ln in res.stdout.splitlines():
        p = ln.split("\t")
        if len(p) >= 4:
            hit[p[3]] = hit.get(p[3], 0) + 1
    return {m: hit.get(m, 0) / n for m, n in tot.items() if n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tomtom", type=Path, required=True, help="tomtom.tsv: de-novo vs v12+Codebook")
    ap.add_argument("--artifact-tomtom", type=Path, default=None,
                    help="tomtom.tsv: de-novo vs Codebook artifact set (codebook_artifacts.meme). "
                         "Any match q<=max-q flags the motif (their empirical artifact set).")
    ap.add_argument("--ranked", type=Path, default=None, help="Step-3 ranked_motifs.tsv to annotate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-q", type=float, default=0.05,
                    help="only call contaminant if best TOMTOM q <= this")
    ap.add_argument("--families", type=Path, default=None,
                    help="optional TSV: family<TAB>regex (replaces defaults; e.g. Codebook set)")
    ap.add_argument("--repeat-bed", type=Path, default=None)
    ap.add_argument("--instances-bed", type=Path, default=None)
    ap.add_argument("--te-frac", type=float, default=0.5,
                    help="flag TE-clustered if this fraction of instances hit repeats")
    args = ap.parse_args()

    pats = DEFAULT_FAMILIES
    if args.families and args.families.is_file():
        pats = {}
        for ln in args.families.read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                fam, _, rx = ln.partition("\t")
                if rx:
                    pats[fam.strip()] = rx.strip()

    best = best_tomtom(args.tomtom)
    art_best = best_tomtom(args.artifact_tomtom) if args.artifact_tomtom and args.artifact_tomtom.is_file() else {}
    te = (repeat_overlap_frac(args.instances_bed, args.repeat_bed)
          if args.repeat_bed and args.instances_bed else {})

    def annotate(mid):
        tgt, q = best.get(mid, ("", float("nan")))
        fam = family_of(tgt, pats) if tgt and q == q and q <= args.max_q else None
        te_f = te.get(mid, float("nan"))
        te_flag = (te_f == te_f) and te_f >= args.te_frac
        contam = fam is not None
        at, aq = art_best.get(mid, ("", float("nan")))
        cb_art = bool(at) and aq == aq and aq <= args.max_q      # matches Codebook artifact set
        ok = not contam and not te_flag and not cb_art
        return tgt, q, fam or "", contam, cb_art, at, te_f, te_flag, ok

    extra = ["tomtom_best", "tomtom_q", "family", "is_contaminant", "codebook_artifact",
             "artifact_match", "te_frac", "te_clustered", "pass"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_pass = n_contam = 0
    if args.ranked and args.ranked.is_file():
        hdr, rows = read_ranked(args.ranked)
        mi = hdr.index("motif_id") if "motif_id" in hdr else 0
        with args.out.open("w") as f:
            f.write("\t".join(hdr + extra) + "\n")
            for r in rows:
                tgt, q, fam, contam, cb_art, at, te_f, te_flag, ok = annotate(r[mi])
                n_pass += ok; n_contam += (contam or cb_art)
                f.write("\t".join(r + [tgt, f"{q:.3g}", fam, str(contam), str(cb_art),
                                       at, f"{te_f:.3g}", str(te_flag), str(ok)]) + "\n")
    else:
        with args.out.open("w") as f:
            f.write("motif_id\t" + "\t".join(extra) + "\n")
            for mid in sorted(set(best) | set(art_best)):
                tgt, q, fam, contam, cb_art, at, te_f, te_flag, ok = annotate(mid)
                n_pass += ok; n_contam += (contam or cb_art)
                f.write(f"{mid}\t{tgt}\t{q:.3g}\t{fam}\t{contam}\t{cb_art}\t{at}\t"
                        f"{te_f:.3g}\t{te_flag}\t{ok}\n")
    print(f"[flag] contaminant/artifact={n_contam} pass={n_pass} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
