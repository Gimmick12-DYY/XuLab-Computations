#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# build_pairs_from_matrix.py
#
# For a TF that has no Cicero fitConns file.
#
# NOT the default cobinding path. Pearson of imputed cell vectors inside 1 Mb
# is not Cicero: with ~10k cells, BH FDR keeps almost every positive r and
# produces giant cliques (e.g. TRAFD1 405k edges vs RBBP4 Cicero 11k).
# Run cobinding/slurm/run_cicero_cobinding.sbatch instead (real run_cicero,
# 1 Mb window, then the same plots).
# pairs whose midpoints are <= --max-dist (default 1 Mb — the Cicero window),
# score each pair by Pearson correlation of the TF's per-cell signal at the
# overlapping matrix bins, and BH-adjust. Writes the same peak_edges /
# peak_annot tables as build_peak_graph.py, plus a .pdc-like file so
# distal_chromhmm_composition.py can run unchanged.
#
# Proximal vs distal: ChromHMM-18 promoter states (Tss*) = proximal, else distal.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from math import erfc, sqrt

_COORD = re.compile(r"^(chr[0-9A-Za-z]+):(\d+)-(\d+)$")
_PROMOTER = {"TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "TssBiv"}


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return [ln.rstrip("\n") for ln in fh if ln.strip()]


def load_peaks(bed: Path, max_peaks: int) -> list[tuple[str, int, int, float, str]]:
    rows = []
    with bed.open() as fh:
        for i, ln in enumerate(fh):
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3 or p[0].startswith(("#", "track", "browser")):
                continue
            try:
                score = float(p[4]) if len(p) > 4 and p[4] not in {".", ""} else 0.0
            except ValueError:
                score = 0.0
            chrom, s, e = p[0], int(p[1]), int(p[2])
            name = f"{chrom}:{s}-{e}"
            rows.append((chrom, s, e, score, name))
    rows.sort(key=lambda r: -r[3])
    if max_peaks and len(rows) > max_peaks:
        rows = rows[:max_peaks]
    return rows


def load_regions(mat_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    path = None
    for name in ("regions.tsv", "regions.tsv.gz"):
        cand = mat_dir / name
        if cand.is_file():
            path = cand
            break
    if path is None:
        raise SystemExit(f"no regions.tsv in {mat_dir}")
    names = read_lines(path)
    n = len(names)
    chroms = np.empty(n, dtype=object)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    for i, nm in enumerate(names):
        m = _COORD.match(nm.strip())
        if not m:
            raise SystemExit(f"bad region {nm!r}")
        chroms[i] = m.group(1)
        starts[i] = int(m.group(2))
        ends[i] = int(m.group(3))
    return names, chroms, starts, ends


def load_matrix(mat_dir: Path):
    csr = mat_dir / "matrix_csr.npz"
    if csr.is_file():
        return sp.load_npz(csr).tocsr()
    mtx = mat_dir / "matrix.mtx.gz"
    if mtx.is_file():
        return sio.mmread(mtx).tocsr()
    raise SystemExit(f"no matrix in {mat_dir}")


def chrom_index(chroms, starts, ends):
    by = defaultdict(list)
    for i, c in enumerate(chroms):
        by[c].append(i)
    idx = {}
    for c, ids in by.items():
        ids = np.asarray(ids, dtype=np.int64)
        order = np.argsort(starts[ids])
        ids = ids[order]
        idx[c] = (ids, starts[ids], ends[ids])
    return idx


def best_bin(chrom, s, e, idx) -> int | None:
    if chrom not in idx:
        return None
    ids, st, en = idx[chrom]
    j = int(np.searchsorted(st, e, side="left"))
    best_i, best_ov = None, 0
    k = j - 1
    while k >= 0 and en[k] > s:
        ov = min(e, int(en[k])) - max(s, int(st[k]))
        if ov > best_ov:
            best_ov, best_i = ov, int(ids[k])
        k -= 1
    k = j
    while k < len(st) and st[k] < e:
        ov = min(e, int(en[k])) - max(s, int(st[k]))
        if ov > best_ov:
            best_ov, best_i = ov, int(ids[k])
        k += 1
    return best_i


def load_chromhmm(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    by = defaultdict(list)
    with opener(path, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            st = re.sub(r"^\d+_", "", p[3].strip())
            by[p[0]].append((int(p[1]), int(p[2]), st))
    out = {}
    for c, segs in by.items():
        segs.sort()
        out[c] = (np.fromiter((x[0] for x in segs), int, len(segs)),
                  np.fromiter((x[1] for x in segs), int, len(segs)),
                  [x[2] for x in segs])
    return out


def chromhmm_state(peak: str, hmm) -> str | None:
    m = _COORD.match(peak)
    if not m or m.group(1) not in hmm:
        return None
    s, e = int(m.group(2)), int(m.group(3))
    st, en, lab = hmm[m.group(1)]
    j = int(np.searchsorted(st, e, side="left"))
    best, ov = None, 0
    k = max(0, j - 1)
    while k < len(st) and st[k] < e:
        o = min(e, int(en[k])) - max(s, int(st[k]))
        if o > ov:
            ov, best = o, lab[k]
        k += 1
    return best


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, dtype=np.float64)
    q[order] = np.minimum(1.0, p[order] * n / np.arange(1, n + 1))
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--peaks", type=Path, required=True)
    ap.add_argument("--mat-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chromhmm", type=Path, default=None)
    ap.add_argument("--max-dist", type=int, default=1_000_000)
    ap.add_argument("--max-peaks", type=int, default=20_000,
                    help="keep the strongest N peaks (score col 5); 0 = all")
    ap.add_argument("--qval", type=float, default=0.05)
    args = ap.parse_args()

    peaks = load_peaks(args.peaks, args.max_peaks)
    print(f"[{args.tf}] peaks={len(peaks)} from {args.peaks}", flush=True)
    names, chroms, starts, ends = load_regions(args.mat_dir)
    X = load_matrix(args.mat_dir)
    if X.shape[0] != len(names):
        raise SystemExit(f"matrix rows {X.shape[0]} != regions {len(names)}")
    idx = chrom_index(chroms, starts, ends)

    mapped: list[tuple[str, int, int, int]] = []  # name, chrom_sort, mid, bin
    for chrom, s, e, _sc, name in peaks:
        bi = best_bin(chrom, s, e, idx)
        if bi is None:
            continue
        mapped.append((name, chrom, (s + e) // 2, bi))
    print(f"[{args.tf}] mapped to bins: {len(mapped)} / {len(peaks)}", flush=True)
    if len(mapped) < 3:
        raise SystemExit(f"too few mapped peaks ({len(mapped)})")

    by_chrom: dict[str, list] = defaultdict(list)
    for rec in mapped:
        by_chrom[rec[1]].append(rec)
    for c in by_chrom:
        by_chrom[c].sort(key=lambda r: r[2])

    bin_ids = sorted({r[3] for r in mapped})
    bin_pos = {b: i for i, b in enumerate(bin_ids)}
    sub = X[bin_ids, :].astype(np.float32)
    if sp.issparse(sub):
        dense = sub.toarray()
    else:
        dense = np.asarray(sub, dtype=np.float32)
    # center + L2 for Pearson = cosine of centered rows
    dense = dense - dense.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(dense, axis=1)
    ok = nrm > 1e-8
    nrm[~ok] = 1.0
    dense = dense / nrm[:, None]
    n_cells = dense.shape[1]
    df_t = max(n_cells - 2, 1)

    pairs_p1, pairs_p2, pairs_r, pairs_p = [], [], [], []
    n_cand = 0
    for chrom, recs in by_chrom.items():
        mids = [r[2] for r in recs]
        for i, (n1, _c, m1, b1) in enumerate(recs):
            if not ok[bin_pos[b1]]:
                continue
            k = i + 1
            while k < len(recs) and mids[k] - m1 <= args.max_dist:
                n2, _c2, _m2, b2 = recs[k]
                k += 1
                if n1 == n2 or b1 == b2 or not ok[bin_pos[b2]]:
                    continue
                n_cand += 1
                r = float(dense[bin_pos[b1]] @ dense[bin_pos[b2]])
                r = float(np.clip(r, -0.999999, 0.999999))
                if r <= 0:
                    continue
                t = r * np.sqrt(df_t / (1.0 - r * r))
                p = erfc(abs(t) / sqrt(2.0))
                pairs_p1.append(n1 if n1 < n2 else n2)
                pairs_p2.append(n2 if n1 < n2 else n1)
                pairs_r.append(r)
                pairs_p.append(max(p, 1e-300))
    print(f"[{args.tf}] cis candidates ≤{args.max_dist}bp: {n_cand:,}; "
          f"positive-r: {len(pairs_r):,}", flush=True)
    if not pairs_r:
        raise SystemExit("no positive-correlation cis pairs")

    p = np.asarray(pairs_p, dtype=np.float64)
    q = bh_qvalues(p)
    keep = q <= args.qval
    print(f"[{args.tf}] FDR<={args.qval}: {int(keep.sum()):,} / {keep.size:,}", flush=True)
    if not np.any(keep):
        raise SystemExit("no pairs pass FDR")

    hmm = load_chromhmm(args.chromhmm) if args.chromhmm and args.chromhmm.is_file() else {}
    types: dict[str, set] = defaultdict(set)
    states: dict[str, set] = defaultdict(set)
    n_edges: dict[str, int] = defaultdict(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tf = args.tf
    edges_path = args.out_dir / f"{tf}.peak_edges.tsv"
    pdc_path = args.out_dir / f"{tf}.generated.pdc"
    n_write = n_pdc = 0
    with edges_path.open("w") as eh, pdc_path.open("w") as ph:
        eh.write("peak1\tpeak2\tcoaccess\tpval\tqval\ttype1\ttype2\tn_links\n")
        for i, use in enumerate(keep):
            if not use:
                continue
            a, b, r, pv, qv = pairs_p1[i], pairs_p2[i], pairs_r[i], pairs_p[i], float(q[i])
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
            eh.write(f"{a}\t{b}\t{r:.6g}\t{pv:.6g}\t{qv:.6g}\t{ta}\t{tb}\t1\n")
            n_write += 1
            if ta != tb:
                prox, dist = (a, b) if ta == "proximal" else (b, a)
                # 11-col layout expected by distal_chromhmm_composition.py
                ph.write(f"{prox.replace(':','-')}\t{prox}\tproximal\t.\t"
                         f"{dist.replace(':','-')}\t{dist}\tdistal\tnan\t"
                         f"{r:.6g}\t{pv:.6g}\t{qv:.6g}\n")
                n_pdc += 1

    annot_path = args.out_dir / f"{tf}.peak_annot.tsv"
    with annot_path.open("w") as out:
        out.write("peak\ttypes\tgenes\tstates\tn_edges\n")
        for pk in sorted(n_edges):
            out.write(f"{pk}\t{','.join(sorted(types[pk])) or '.'}\t.\t"
                      f"{','.join(sorted(states[pk])) or '.'}\t{n_edges[pk]}\n")
    print(f"[done] edges={n_write:,} pdc_pairs={n_pdc:,} peaks={len(n_edges):,}")
    print(f"[done] wrote {edges_path}\n[done] wrote {annot_path}\n[done] wrote {pdc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
