#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# cicero_conns_to_edges.py
#
# Cicero run_cicero() -> cobinding peak_edges.tsv (+ generated.pdc).
# Dedups Peak1/Peak2 as an unordered pair (Cicero emits both orientations);
# keeps max coaccess; drops trans / > --max-dist / coaccess < --coaccess-min.
#
# Significance (--pval-method):
#   ttest      (default) two-sided correlation t-test, df = n_eff - 2, then BH.
#              n_eff = Cicero metacell count (cicero_info.tsv / --effective-n).
#   shuffle    one-sided Gaussian upper tail using a shuffle-null mean/sd
#              (lab RBBP4.perm.fitConns.para.txt: p = 1-Phi((coaccess-mu)/sd)),
#              then BH across all unique pairs kept after distance filters.
#   precomputed use p / FDR columns already in the input table (lab fitConns.res).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import erfc
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


def parse_shuffle_para(path: Path) -> tuple[float, float]:
    """Read lab-style shuffle para (columns meanShuf, stdShuf)."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise SystemExit(f"shuffle para needs header + row: {path}")
    hdr = lines[0].split("\t")
    row = lines[1].split("\t")
    d = {h.strip(): v for h, v in zip(hdr, row)}
    if "meanShuf" not in d or "stdShuf" not in d:
        raise SystemExit(f"{path} missing meanShuf/stdShuf (got {list(d)})")
    return float(d["meanShuf"]), float(d["stdShuf"])


def gaussian_upper_p(coaccess: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """One-sided Gaussian upper tail: 1-Phi((x-mu)/sd). Matches lab perm p."""
    z = (np.asarray(coaccess, dtype=float) - mu) / max(float(sd), 1e-18)
    return 0.5 * erfc(z / np.sqrt(2.0))


def col_idx(header: list[str], *names: str) -> int | None:
    lower = {h.strip().lower(): i for i, h in enumerate(header)}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--conns", type=Path, required=True,
                    help="cicero/coaccess.tsv.gz or lab fitConns.res.txt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chromhmm", type=Path, default=None)
    ap.add_argument("--max-dist", type=int, default=1_000_000)
    ap.add_argument("--coaccess-min", type=float, default=0.05,
                    help="Pliner et al. recommended Cicero cutoff")
    ap.add_argument("--max-p", type=float, default=1.0,
                    help="drop pairs with pval > this after significance (1 = keep all)")
    ap.add_argument("--pval-method", choices=("ttest", "shuffle", "precomputed"),
                    default="ttest")
    ap.add_argument("--shuffle-para", type=Path, default=None,
                    help="lab-style para.txt (meanShuf, stdShuf) for --pval-method shuffle")
    ap.add_argument("--shuffle-mu", type=float, default=None)
    ap.add_argument("--shuffle-sd", type=float, default=None)
    ap.add_argument("--effective-n", default="metacell",
                    help="sample size for the coaccess correlation test: an integer, "
                         "or 'metacell'/'cells' (read from sibling cicero_info.tsv)")
    ap.add_argument("--sel", type=Path, default=None,
                    help="optional lab .sel file: report unique-pair overlap")
    ap.add_argument("--scored-all-out", type=Path, default=None,
                    help="optional Wang-style fitConns table for every unique pair "
                         "before --max-p filtering (Peak1 Peak2 coaccess nlog10p FDR p)")
    args = ap.parse_args()

    opener = gzip.open if str(args.conns).endswith(".gz") else open
    # key -> (coaccess, p_pre, q_pre); precomputed p/q kept from the max-coaccess row
    edges: dict[frozenset, tuple[float, float | None, float | None]] = {}
    n_in = n_na = n_self = n_far = n_low = 0
    with opener(args.conns, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_ca = col_idx(header, "coaccess") or 2
        i_p = col_idx(header, "p", "pval", "pvalue")
        i_q = col_idx(header, "fdr", "qval", "q")
        for ln in fh:
            n_in += 1
            p = ln.rstrip("\n").split("\t")
            if len(p) <= max(i_ca, 1):
                continue
            a, b = to_colon(p[0]), to_colon(p[1])
            try:
                ca = float(p[i_ca])
            except ValueError:
                n_na += 1
                continue
            if a is None or b is None or a == b:
                n_self += 1
                continue
            if ca != ca:  # NaN
                n_na += 1
                continue
            if args.coaccess_min and ca < args.coaccess_min:
                n_low += 1
                continue
            d = dist_bp(a, b)
            if d is None or (args.max_dist and d > args.max_dist):
                n_far += 1
                continue
            pre_p = pre_q = None
            if i_p is not None and i_p < len(p):
                try:
                    pre_p = float(p[i_p])
                except ValueError:
                    pre_p = None
            if i_q is not None and i_q < len(p):
                try:
                    pre_q = float(p[i_q])
                except ValueError:
                    pre_q = None
            key = frozenset((a, b))
            prev = edges.get(key)
            if prev is None or ca > prev[0]:
                edges[key] = (ca, pre_p, pre_q)
    print(f"[{args.tf}] cicero rows={n_in:,} kept_unique={len(edges):,}  "
          f"na={n_na:,} self={n_self:,} low={n_low:,} far/trans={n_far:,}", flush=True)
    if not edges:
        raise SystemExit("no Cicero pairs survived filters")

    ekeys = list(edges.keys())
    ca_arr = np.array([edges[k][0] for k in ekeys], dtype=float)
    method = args.pval_method
    if method == "precomputed":
        pv = np.array([np.nan if edges[k][1] is None else edges[k][1] for k in ekeys])
        qv = np.array([np.nan if edges[k][2] is None else edges[k][2] for k in ekeys])
        if np.isnan(pv).all():
            raise SystemExit("--pval-method precomputed but no p/FDR columns in --conns")
        if np.isnan(qv).all():
            qv = bh_qvalues(np.nan_to_num(pv, nan=1.0))
        print(f"[{args.tf}] significance: precomputed p/FDR; "
              f"FDR<=0.05: {int((qv <= 0.05).sum()):,}/{len(ekeys):,} pairs", flush=True)
    elif method == "shuffle":
        if args.shuffle_para is not None:
            mu, sd = parse_shuffle_para(args.shuffle_para)
        elif args.shuffle_mu is not None and args.shuffle_sd is not None:
            mu, sd = float(args.shuffle_mu), float(args.shuffle_sd)
        else:
            raise SystemExit("shuffle p-values need --shuffle-para or --shuffle-mu/--shuffle-sd")
        pv = gaussian_upper_p(ca_arr, mu, sd)
        qv = bh_qvalues(pv)
        print(f"[{args.tf}] significance: shuffle Gaussian mu={mu:.6g} sd={sd:.6g}; "
              f"FDR<=0.05: {int((qv <= 0.05).sum()):,}/{len(ekeys):,} pairs", flush=True)
    else:
        n_eff = resolve_n(args.effective_n, read_info(args.conns))
        pv, qv = coaccess_significance(ca_arr, n_eff)
        if n_eff > 2:
            print(f"[{args.tf}] significance: n_eff={n_eff:,} (df={n_eff - 2:,}); "
                  f"FDR<=0.05: {int((qv <= 0.05).sum()):,}/{len(ekeys):,} pairs", flush=True)
        else:
            print(f"[{args.tf}] WARNING: no effective n (cicero_info.tsv lacks n_metacell/"
                  f"n_cells) -> pval=nan, qval=0. Pass --effective-n <int>.", flush=True)

    if args.scored_all_out is not None:
        args.scored_all_out.parent.mkdir(parents=True, exist_ok=True)
        opener_out = gzip.open if str(args.scored_all_out).endswith(".gz") else open
        with opener_out(args.scored_all_out, "wt") as out:
            out.write("Peak1\tPeak2\tcoaccess\tnlog10p\tFDR\tp\n")
            for key, ca, p_, q_ in zip(ekeys, ca_arr, pv, qv):
                a, b = sorted(key)
                a_us = a.replace(":", "_").replace("-", "_")
                b_us = b.replace(":", "_").replace("-", "_")
                nlog10p = np.inf if float(p_) == 0.0 else -np.log10(float(p_))
                out.write(f"{a_us}\t{b_us}\t{ca:.15g}\t{nlog10p:.15g}\t"
                          f"{float(q_):.15g}\t{float(p_):.15g}\n")
        print(f"[{args.tf}] wrote all scored fitConns -> {args.scored_all_out}", flush=True)

    if args.max_p < 1.0:
        keep = pv <= args.max_p
        n_drop_p = int((~keep).sum())
        ekeys = [k for k, ok in zip(ekeys, keep) if ok]
        pv, qv = pv[keep], qv[keep]
        print(f"[{args.tf}] max-p={args.max_p}: dropped {n_drop_p:,}, kept {len(ekeys):,}",
              flush=True)
        if not ekeys:
            raise SystemExit("no pairs survived --max-p")

    sig = {k: (float(pv[i]), float(qv[i])) for i, k in enumerate(ekeys)}
    ca_map = {k: edges[k][0] for k in ekeys}

    if args.sel is not None and args.sel.is_file():
        sel_pairs: set[frozenset] = set()
        with args.sel.open() as sf:
            for ln in sf:
                toks = ln.rstrip("\n").split("\t")
                coords = [to_colon(t) for t in toks]
                coords = [c for c in coords if c]
                uniq = []
                for c in coords:
                    if c not in uniq:
                        uniq.append(c)
                if len(uniq) >= 2:
                    sel_pairs.add(frozenset(uniq[:2]))
        ours = set(ekeys)
        n_int = len(ours & sel_pairs)
        print(f"[{args.tf}] vs .sel unique pairs: sel={len(sel_pairs):,} ours={len(ours):,} "
              f"intersection={n_int:,} recovered_of_sel={n_int / len(sel_pairs) if sel_pairs else 0:.4f}",
              flush=True)

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
        for key in ekeys:
            ca = ca_map[key]
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
    print(f"[done] edges={len(ekeys):,} pdc={n_pdc:,} peaks={len(n_edges):,}")
    print(f"[done] wrote {edges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
