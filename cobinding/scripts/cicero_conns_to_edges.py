#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# cicero_conns_to_edges.py
#
# Cicero run_cicero() -> cobinding peak_edges.tsv (+ generated.pdc).
# Dedups Peak1/Peak2 as an unordered pair (Cicero emits both orientations);
# keeps max coaccess; drops trans / > --max-dist / coaccess < --coaccess-min.
#
# Significance (reproduces the lab fitConns pval/qval that run_cicero does NOT
# emit): treat each pair's coaccess as a correlation r and test r != 0 with a
# two-sided t-test, df = n_eff - 2, where n_eff = the Cicero metacell count
# (from the sibling cicero_info.tsv; --effective-n overrides). BH-adjust the
# p-values across all kept pairs -> qval. This matches the reference fitConns:
# calibrating n from the p=0.05 pair reproduces its pvals (RBBP4 n_eff ~8.5k).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

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


def read_info(conns_path: Path) -> dict:
    """Parse the sibling cicero_info.tsv (key\\tvalue) written by 03_run_cicero.R."""
    info: dict[str, str] = {}
    p = conns_path.parent / "cicero_info.tsv"
    if p.is_file():
        for ln in p.read_text().splitlines():
            k, _, v = ln.partition("\t")
            info[k.strip()] = v.strip()
    return info


def resolve_n(effective_n: str, info: dict) -> int:
    """Effective sample size for the coaccess correlation test: an explicit int,
    or 'metacell'/'cells' resolved from cicero_info.tsv (falls back metacell->cells)."""
    s = str(effective_n).strip().lower()
    if s.isdigit():
        return int(s)
    prefer = "n_cells" if s in ("cell", "cells") else "n_metacell"
    for key in (prefer, "n_metacell", "n_cells"):
        v = info.get(key, "")
        if v.isdigit():
            return int(v)
    return 0


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, monotone-enforced."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def coaccess_significance(coaccess: np.ndarray, n_eff: int):
    """Two-sided t-test that each coaccess (as a correlation r) differs from 0,
    df = n_eff - 2, then BH -> (pvals, qvals)."""
    r = np.clip(np.asarray(coaccess, dtype=float), -0.999999, 0.999999)
    if not n_eff or n_eff <= 2:
        return np.full(r.size, np.nan), np.zeros(r.size)
    df = n_eff - 2
    tstat = r * np.sqrt(df / (1.0 - r ** 2))
    pv = 2.0 * t_dist.sf(np.abs(tstat), df)
    return pv, bh_qvalues(pv)


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
    ap.add_argument("--effective-n", default="metacell",
                    help="sample size for the coaccess correlation test: an integer, "
                         "or 'metacell'/'cells' (read from sibling cicero_info.tsv)")
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

    # ---- significance: coaccess-as-correlation t-test + BH (reproduce fitConns) --
    n_eff = resolve_n(args.effective_n, read_info(args.conns))
    ekeys = list(edges.keys())
    pv, qv = coaccess_significance(np.array([edges[k] for k in ekeys]), n_eff)
    sig = {k: (float(pv[i]), float(qv[i])) for i, k in enumerate(ekeys)}
    if n_eff > 2:
        print(f"[{args.tf}] significance: n_eff={n_eff:,} (df={n_eff - 2:,}); "
              f"FDR<=0.05: {int((qv <= 0.05).sum()):,}/{len(ekeys):,} pairs", flush=True)
    else:
        print(f"[{args.tf}] WARNING: no effective n (cicero_info.tsv lacks n_metacell/"
              f"n_cells) -> pval=nan, qval=0. Pass --effective-n <int>.", flush=True)

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
            p_, q_ = sig[key]
            eh.write(f"{a}\t{b}\t{ca:.6g}\t{p_:.6g}\t{q_:.6g}\t{ta}\t{tb}\t1\n")
            if ta != tb:
                prox, dist = (a, b) if ta == "proximal" else (b, a)
                ph.write(f"{prox.replace(':','-')}\t{prox}\tproximal\t.\t"
                         f"{dist.replace(':','-')}\t{dist}\tdistal\tnan\t"
                         f"{ca:.6g}\t{p_:.6g}\t{q_:.6g}\n")
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
