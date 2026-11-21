"""Shared genomic helpers for the Figure 3 panels: region parsing, Peakachu loops, the Hi-C loop-support rule, module edges.

Loop-support rule (used for every "on a Hi-C loop" call in Figure 3): a pair of regions on one chromosome is on a loop when
each region, extended by +-30 kb (SLOP), overlaps a different anchor of the same Peakachu score >= 0.7 loop. 5 kb loops are
tested first; a 10 kb loop is used only when no 5 kb loop matches. Pairs/edges are counted when their two region midpoints
are > 25 kb apart (MIN_SPAN).
"""
import gzip
from collections import defaultdict
from itertools import combinations

SLOP = 30_000
MIN_SPAN = 25_000
HG38 = {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
        "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
        "chr19": 58617616, "chr20": 64444167, "chr21": 46709983, "chr22": 50818468, "chrX": 156040895}


def parse_peak(tok):
    chrom, se = tok.split(":", 1)
    s, e = se.split("-", 1)
    return chrom, int(s), int(e)


def mid(p):
    return (p[1] + p[2]) // 2


def span_bp(pr):
    return abs(mid(pr["p1"]) - mid(pr["p2"]))


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_pairs(path):
    """Cicero pairs (peak1, peak2) on one chromosome, ordered by position."""
    rows = []
    with _open(path) as f:
        idx = {h: i for i, h in enumerate(f.readline().rstrip("\n").split("\t"))}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            a, b = parse_peak(p[idx["peak1"]]), parse_peak(p[idx["peak2"]])
            if a[0] != b[0]:
                continue
            if (a[0], a[1]) > (b[0], b[1]):
                a, b = b, a
            rows.append({"p1": a, "p2": b})
    return rows


def load_module_edges(path):
    """Every pair of regions inside one module (cliques.tsv: clique_id, regions = 'chr:s-e;chr:s-e;...')."""
    rows = []
    with _open(path) as f:
        idx = {h: i for i, h in enumerate(f.readline().rstrip("\n").split("\t"))}
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            peaks = sorted(parse_peak(t) for t in p[idx["regions"]].split(";") if t)
            for a, b in combinations(peaks, 2):
                if a[0] != b[0]:
                    continue
                if (a[0], a[1]) > (b[0], b[1]):
                    a, b = b, a
                rows.append({"clique_id": p[idx["clique_id"]], "n": len(peaks), "p1": a, "p2": b})
    return rows


def load_bedpe(path, res):
    out = []
    with _open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            x = ln.split()
            a, b = (x[0], int(x[1]), int(x[2])), (x[3], int(x[4]), int(x[5]))
            if (a[0], a[1]) > (b[0], b[1]):
                a, b = b, a
            out.append({"a": a, "b": b, "score": float(x[6]) if len(x) > 6 else 0.0, "res": res})
    return out


class Loops:
    """Peakachu 5 kb + 10 kb loops with an anchor index for the loop-support rule."""

    def __init__(self, bedpe5, bedpe10):
        self.sets = []
        for path, res in ((bedpe5, 5000), (bedpe10, 10000)):
            loops = load_bedpe(path, str(res))
            ix = defaultdict(list)
            for i, lp in enumerate(loops):
                for end in ("a", "b"):
                    iv = lp[end]
                    for b in range(iv[1] // res, (iv[2] - 1) // res + 1):
                        ix[(iv[0], b)].append((i, end))
            self.sets.append((loops, ix, res))
        self.n_total = sum(len(s[0]) for s in self.sets)

    @staticmethod
    def _on_peak(ix, loops, chrom, s, e, res):
        hit = defaultdict(set)
        lo, hi = max(0, s - SLOP), e + SLOP
        for b in range(lo // res, max(lo // res, (hi - 1) // res) + 1):
            for i, end in ix.get((chrom, b), ()):
                iv = loops[i][end]
                if min(iv[2], hi) > max(iv[1], lo):
                    hit[i].add(end)
        return hit

    def assign(self, pr):
        """The loop supporting this pair (dict with a, b, score, res), or None."""
        c = pr["p1"][0]
        for loops, ix, res in self.sets:
            h1 = self._on_peak(ix, loops, c, pr["p1"][1], pr["p1"][2], res)
            h2 = self._on_peak(ix, loops, c, pr["p2"][1], pr["p2"][2], res)
            best = None
            for i, e1 in h1.items():
                e2 = h2.get(i)
                if e2 and (("a" in e1 and "b" in e2) or ("b" in e1 and "a" in e2)) and (best is None or i < best):
                    best = i
            if best is not None:
                return loops[best]
        return None


class LoopArrays:
    """Same loop-support rule in array form (used by the module-coverage tables, cobinding/hub_figures/descriptive/clique_coverage.py):
    with p the left region and q the right one (by midpoint), the pair is on a loop when p +- 30 kb overlaps the left anchor and
    q +- 30 kb the right anchor of one same-chromosome loop (5 kb and 10 kb loops pooled)."""

    def __init__(self, bedpe5, bedpe10):
        import numpy as np
        L = defaultdict(list)
        for path in (bedpe5, bedpe10):
            with _open(path) as f:
                for ln in f:
                    x = ln.split()
                    if x[0] == x[3]:
                        a, b = (int(x[1]), int(x[2])), (int(x[4]), int(x[5]))
                        if a > b:
                            a, b = b, a
                        L[x[0]].append((*a, *b))
        self.A = {c: np.array(v) for c, v in L.items()}

    def on_loop(self, p, q, slop=SLOP):
        if mid(p) > mid(q):
            p, q = q, p
        A = self.A.get(p[0])
        return A is not None and bool(((p[1] <= A[:, 1] + slop) & (p[2] >= A[:, 0] - slop) &
                                       (q[1] <= A[:, 3] + slop) & (q[2] >= A[:, 2] - slop)).any())


class ElementClass:
    """Element class of a region: promoter = a RefSeq Select TSS within +-1 kb; otherwise the HEK293T 18-state ChromHMM state at
    the region midpoint: enhancer (EnhA1/A2/G1/G2/Wk/Biv), repressive (ReprPC/ReprPCWk/Het/ZNF-Rpts), other (everything else)."""
    GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
             "ReprPC": "repressive", "ReprPCWk": "repressive", "Het": "repressive", "ZNF/Rpts": "repressive"}

    def __init__(self, tss_tsv, chromhmm_bed):
        import numpy as np
        import pandas as pd
        T = pd.read_csv(tss_tsv, sep="\t")
        self.tss = {c: np.sort(d.tss.values) for c, d in T.groupby("chrom")}
        H = defaultdict(list)
        with _open(chromhmm_bed) as f:
            for ln in f:
                x = ln.split("\t")
                H[x[0]].append((int(x[1]), int(x[2]), x[3]))
        self.hmm = {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)])
                    for c, v in H.items()}

    def __call__(self, c, s, e):
        import numpy as np
        if c in self.tss:
            p = self.tss[c]
            i = np.searchsorted(p, s - 1000)
            if i < len(p) and p[i] <= e + 1000:
                return "promoter"
        if c not in self.hmm:
            return "other"
        st, en, lab = self.hmm[c]
        m = (s + e) // 2
        i = np.searchsorted(st, m, side="right") - 1
        return self.GROUP.get(lab[i], "other") if i >= 0 and en[i] > m else "other"
