#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# summarize_complex_motifs.py
#
# Collate the per-complex HOMER + AME results (03_complex_motifs.sbatch) into one
# table answering checklist step 4: does each complex share / use the same motif?
#
# For each complex we report the top known motifs (HOMER knownResults.txt and AME)
# and flag SHARED_MEMBER when a top motif's TF matches a complex member (the complex
# co-binds that member's motif -- evidence the member is the DNA-binding anchor),
# and SHARED_COMMON when one motif dominates regardless of membership.
#
# Output: tsv to stdout (or --out).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_manifest(path):
    out = []
    for i, ln in enumerate(open(path)):
        if i == 0:
            continue
        t = ln.rstrip("\n").split("\t")
        out.append((t[0], t[5].split(",") if len(t) > 5 else []))
    return out


def homer_top(cid_dir, k=3):
    f = cid_dir / "homer" / "knownResults.txt"
    if not f.is_file():
        return []
    rows = []
    with open(f) as fh:
        next(fh, None)
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            name = p[0]
            try:
                q = float(p[4])
            except ValueError:
                q = float("nan")
            rows.append((name, q))
    return rows[:k]


def ame_top(cid_dir, k=3):
    # ame writes ame.tsv (TSV with header incl. motif_ID, motif_alt_ID, adj_p-value)
    f = cid_dir / "ame" / "ame.tsv"
    if not f.is_file():
        return []
    rows = []
    with open(f) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            mid = r.get("motif_ID") or r.get("motif_alt_ID") or "?"
            try:
                q = float(r.get("adj_p-value", r.get("p-value", "nan")))
            except (ValueError, TypeError):
                q = float("nan")
            rows.append((mid, q))
    return rows[:k]


def tf_token(motif_name):
    """Extract a gene-ish token: 'CTCF(Zf)/...' -> ctcf ; 'CTCF_HUMAN.H11MO..' -> ctcf."""
    s = re.split(r"[(/_.]", motif_name.strip())[0]
    return s.lower()


def motif_class(motif_name):
    """HOMER family class in the first parens: 'Prop1(Homeobox)/...' -> homeobox."""
    m = re.search(r"\(([^)]+)\)", motif_name)
    return m.group(1).split(",")[0].strip().lower() if m else ""


# member gene -> DNA-binding family, so a family-level motif hit counts even when the
# specific factor differs (e.g. a Homeobox motif for a HOX member). Only SPECIFIC
# families -- NOT the C2H2-Zf superfamily (it lumps SP/KLF/KRAB-ZNF/MAZ, which bind
# unrelated sequences, so a "Zf" match is meaningless).
FAMILY = {
    "homeobox": ("hox", "prop", "phox", "isl", "msx", "dlx", "nkx", "pitx", "lhx", "cdx"),
    "bhlh": ("mlx", "myc", "tcf", "hes", "neurod", "ascl"),
    "ets": ("ets", "elf", "elk", "erg", "fli"),
    "forkhead": ("foxc", "foxk", "foxj", "foxa", "foxo", "foxp"),
    "bzip": ("ddit3", "atf", "cebp", "jun", "fos", "e4f"),
}


def member_matches_motif(members_low, name):
    """True if a member matches this motif by name (substring, either dir, >=3) or by family."""
    tok = tf_token(name)
    for m in members_low:
        if len(tok) >= 3 and (tok in m or m in tok):
            return m
    cls = motif_class(name)
    for m in members_low:
        for fam, pref in FAMILY.items():
            if cls == fam and m.startswith(pref):
                return m
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bed-dir", type=Path, required=True, help="dir with complex_manifest.tsv")
    ap.add_argument("--motif-dir", type=Path, required=True, help="complex_motifs/ output root")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    manifest = read_manifest(args.bed_dir / "complex_manifest.tsv")
    lines = ["complex_id\tmembers\tshared_call\ttop_class\thomer_top\tame_top"]
    for cid, members in manifest:
        cdir = args.motif_dir / cid
        homer = homer_top(cdir, args.top)
        ame = ame_top(cdir, args.top)
        # scan deeper for a member motif (the #1 hit is often a generic promoter motif;
        # a real member's motif can sit a few ranks down) but only display the top-N.
        homer_deep = homer_top(cdir, max(args.top, 15))
        ame_deep = ame_top(cdir, max(args.top, 15))
        mem_l = [m.lower() for m in members]
        member_hit = sorted({member_matches_motif(mem_l, n) for n, _ in homer_deep + ame_deep
                             if member_matches_motif(mem_l, n)})
        # dominant HOMER family class among the top hits (context even w/o a member match)
        classes = [motif_class(n) for n, _ in homer if motif_class(n)]
        top_class = max(set(classes), key=classes.count) if classes else "-"
        # "common": same token is #1 in both HOMER and AME
        common = (homer and ame and tf_token(homer[0][0]) == tf_token(ame[0][0]))
        if member_hit:
            call = "SHARED_MEMBER:" + ",".join(member_hit)
        elif common:
            call = "SHARED_COMMON:" + tf_token(homer[0][0])
        else:
            call = "none/weak"
        h = "; ".join(f"{n}(q={q:.1e})" for n, q in homer) or "-"
        a = "; ".join(f"{n}(q={q:.1e})" for n, q in ame) or "-"
        lines.append(f"{cid}\t{','.join(members)}\t{call}\t{top_class}\t{h}\t{a}")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"[done] -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
