# Tile elements > 2 kb into ~1 kb pieces centred on ATAC peak midpoints; keep parent_id; re-type tiles by TSS +/- 1 kb
import sys
import numpy as np, pandas as pd

elements, atac, tss1k, out, readme = sys.argv[1:6]
MAXW, TILE, MINW = 2000, 1000, 300
E = pd.read_csv(elements, sep="\t", header=None, names=["chrom", "start", "end", "element_id", "type"])
A = pd.read_csv(atac, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
A["mid"] = (A.start + A.end) // 2
T = pd.read_csv(tss1k, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
amid = {c: np.sort(d.mid.values) for c, d in A.groupby("chrom")}
tss = {c: (d.sort_values("start").start.values, d.sort_values("start").end.values) for c, d in T.groupby("chrom")}

def split_even(s, e):
    """Split [s, e) into ~TILE pieces; no piece shorter than MINW unless the gap itself is."""
    L = e - s
    if L <= 0: return []
    n = max(1, int(round(L / TILE)))
    b = np.linspace(s, e, n + 1).round().astype(int)
    return [(b[i], b[i + 1]) for i in range(n)]

def tile(s, e, mids):
    # 1 kb tiles centred on ATAC midpoints (clipped, overlapping merged); gaps cut into ~1 kb pieces
    t = []
    for m in mids:
        a, b = max(s, m - TILE // 2), min(e, m + TILE // 2)
        if t and a <= t[-1][1]: t[-1] = (t[-1][0], max(t[-1][1], b))
        else: t.append((a, b))
    pieces, cur = [], s
    for a, b in t:
        pieces += split_even(cur, a); pieces.append((a, b)); cur = b
    pieces += split_even(cur, e)
    # merge pieces shorter than MINW into the previous piece (or the next one if first)
    res = []
    for a, b in pieces:
        if res and b - a < MINW: res[-1] = (res[-1][0], b)
        elif res and res[-1][1] - res[-1][0] < MINW: res[-1] = (res[-1][0], b)
        else: res.append((a, b))
    return res

def overlaps_tss(c, a, b):
    # TSS windows are <= 2001 bp, so only windows starting in (a - 2001, b) can overlap [a, b)
    if c not in tss: return False
    st, en = tss[c]
    i, j = np.searchsorted(st, a - 2001), np.searchsorted(st, b)
    return bool((en[i:j] > a).any())

rows = []
for r in E.itertuples(index=False):
    if r.end - r.start <= MAXW:
        rows.append((r.chrom, r.start, r.end, r.type, r.element_id, r.type)); continue
    m = amid.get(r.chrom, np.array([], int))
    mids = m[(m >= r.start) & (m < r.end)]
    for a, b in tile(r.start, r.end, mids):
        typ = "promoter" if overlaps_tss(r.chrom, a, b) else "distal"
        rows.append((r.chrom, a, b, typ, r.element_id, r.type))
F = pd.DataFrame(rows, columns=["chrom", "start", "end", "type", "parent_id", "parent_type"])
F = F.sort_values(["chrom", "start"]).reset_index(drop=True)
assert (F.end > F.start).all()
for c, d in F.groupby("chrom"):
    assert (d.start.values[1:] >= d.end.values[:-1]).all(), c
F.insert(3, "element_id", [f"F{i:06d}" for i in range(1, len(F) + 1)])
F.to_csv(out, sep="\t", header=False, index=False)
w = F.end - F.start
msg = (f"elements_final.bed: {len(F)} elements from {len(E)} parents ({(E.end - E.start > MAXW).sum()} parents > {MAXW} bp tiled); "
       f"promoter {(F.type == 'promoter').sum()}, distal {(F.type == 'distal').sum()}; median width {int(w.median())}, "
       f"max {w.max()}, min {w.min()}; widths > 2 kb: {(w > MAXW).sum()}")
print(msg)
with open(readme, "a") as f:
    f.write(f"""
## elements_final.bed (use this one)
Source: elements_with_bivalent.bed (active states + TssBiv + EnhBiv; ReprPC not used). Parents > 2 kb were tiled by
tile_elements.py: 1 kb tiles centred on union3 ATAC peak midpoints (overlapping tiles merged), remaining stretches cut into
~1 kb pieces, pieces < 300 bp merged into a neighbour. Tiles are re-typed: promoter if the tile overlaps TSS +/- 1 kb, else distal.
Columns: chrom, start, end, element_id (F......), type, parent_id (E...... in elements_with_bivalent.bed), parent_type.
{msg}
""")
