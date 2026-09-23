#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# cicero_conns_to_edges.py
#
# Cicero run_cicero() -> cobinding peak_edges.tsv (+ generated.pdc).
# Dedups Peak1/Peak2 as an unordered pair (Cicero emits both orientations);
# keeps max coaccess; drops trans / > --max-dist / coaccess < --coaccess-min.
# qval is left at 0 for pairs that already passed the Cicero score cutoff
# (Cicero does not emit BH p-values; RBBP4's q column came with the lab
# fitConns files).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from build_pairs_from_matrix import _PROMOTER, chromhmm_state, load_chromhmm  # noqa: E402

_US = re.compile(r"^(chr[0-9A-Za-z]+)_(\d+)_(\d+)$")
_COLON = re.compile(r"^(chr[0-9A-Za-z]+):(\d+)-(\d+)$")


def to_colon(tok: str) -> str | None:
    t = tok.strip()
    m = _COLON.match(t)
    if m:
        return f"{m.group(1)}:{m.group(2)}-{m.group(3)}"
    m = _US.match(t)
    if m:
        return f"{m.group(1)}:{m.group(2)}-{m.group(3)}"
    return None


def dist_bp(a: str, b: str) -> int | None:
    m1, m2 = _COLON.match(a), _COLON.match(b)
    if not m1 or not m2 or m1.group(1) != m2.group(1):
        return None
    mid1 = (int(m1.group(2)) + int(m1.group(3))) // 2
    mid2 = (int(m2.group(2)) + int(m2.group(3))) // 2
    return abs(mid1 - mid2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--conns", type=Path, required=True, help="cicero/coaccess.tsv.gz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chromhmm", type=Path, default=None)
    ap.add_argument("--max-dist", type=int, default=1_000_000)
    ap.add_argument("--coaccess-min", type=float, default=0.05,
                    help="Pliner et al. recommended Cicero cutoff")
    args = ap.parse_args()

    opener = gzip.open if str(args.conns).endswith(".gz") else open
    edges: dict[frozenset, float] = {}
    n_in = n_na = n_self = n_far = n_low = 0
    with opener(args.conns, "rt") as fh:
        header = fh.readline()
        for ln in fh:
            n_in += 1
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            a, b = to_colon(p[0]), to_colon(p[1])
            try:
                ca = float(p[2])
            except ValueError:
                n_na += 1
                continue
            if a is None or b is None or a == b:
                n_self += 1
                continue
            if ca != ca:  # NaN
                n_na += 1
                continue
            if ca < args.coaccess_min:
                n_low += 1
                continue
            d = dist_bp(a, b)
            if d is None or (args.max_dist and d > args.max_dist):
                n_far += 1
                continue
            key = frozenset((a, b))
            prev = edges.get(key)
            if prev is None or ca > prev:
                edges[key] = ca
    print(f"[{args.tf}] cicero rows={n_in:,} kept_unique={len(edges):,}  "
          f"na={n_na:,} self={n_self:,} low={n_low:,} far/trans={n_far:,}", flush=True)
    if not edges:
        raise SystemExit("no Cicero pairs survived filters")

    hmm = load_chromhmm(args.chromhmm) if args.chromhmm and args.chromhmm.is_file() else {}
    types: dict[str, set] = defaultdict(set)
    states: dict[str, set] = defaultdict(set)
    n_edges: dict[str, int] = defaultdict(int)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tf = args.tf
    edges_path = args.out_dir / f"{tf}.peak_edges.tsv"
    pdc_path = args.out_dir / f"{tf}.generated.pdc"
    n_pdc = 0
    with edges_path.open("w") as eh, pdc_path.open("w") as ph:
        eh.write("peak1\tpeak2\tcoaccess\tpval\tqval\ttype1\ttype2\tn_links\n")
        for key, ca in edges.items():
            a, b = sorted(key)
            sa = chromhmm_state(a, hmm) if hmm else None
            sb = chromhmm_state(b, hmm) if hmm else None
            ta = "proximal" if sa in _PROMOTER else "distal"
            tb = "proximal" if sb in _PROMOTER else "distal"
            if sa:
                states[a].add(sa)
            if sb:
                states[b].add(sb)
            types[a].add(ta)
            types[b].add(tb)
            n_edges[a] += 1
            n_edges[b] += 1
            # qval=0: pair already passed Cicero coaccess cutoff (no BH from run_cicero)
            eh.write(f"{a}\t{b}\t{ca:.6g}\t0\t0\t{ta}\t{tb}\t1\n")
            if ta != tb:
                prox, dist = (a, b) if ta == "proximal" else (b, a)
                ph.write(f"{prox.replace(':','-')}\t{prox}\tproximal\t.\t"
                         f"{dist.replace(':','-')}\t{dist}\tdistal\tnan\t"
                         f"{ca:.6g}\t0\t0\n")
                n_pdc += 1
    annot_path = args.out_dir / f"{tf}.peak_annot.tsv"
    with annot_path.open("w") as out:
        out.write("peak\ttypes\tgenes\tstates\tn_edges\n")
        for pk in sorted(n_edges):
            out.write(f"{pk}\t{','.join(sorted(types[pk])) or '.'}\t.\t"
                      f"{','.join(sorted(states[pk])) or '.'}\t{n_edges[pk]}\n")
    print(f"[done] edges={len(edges):,} pdc={n_pdc:,} peaks={len(n_edges):,}")
    print(f"[done] wrote {edges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
