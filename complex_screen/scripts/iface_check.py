# Interface residues (UniProt numbering; "L" = GS linker) of screen models MTF2-RBBP4 and RBBP4-RBBP7
import gemmi, pandas as pd
from scipy.spatial import cKDTree
C = "/work/users/d/y/dyy12/XuLab/complex_screen"
ch = pd.read_csv(f"{C}/chains/chains.tsv", sep="\t").set_index("chain_id")
def idx2u(t):
    segs = [tuple(map(int, x.split("-"))) for x in ch.loc[t, "segments"].split(",")]; m, k = {}, 0
    for i, (a, b) in enumerate(segs):
        for u in range(a, b + 1): k += 1; m[k] = u
        k += 10 if i < len(segs) - 1 else 0
    return m
def runs(v):
    v = sorted(v); out, s = [], v[0]
    for a, b in zip(v, v[1:] + [None]):
        if b is None or b != a + 1: out.append(f"{s}-{a}" if s != a else str(s)); s = b
    return ",".join(out)
for pid, (ta, tb) in [("MTF2__RBBP4", ("MTF2", "RBBP4")), ("RBBP4__RBBP7", ("RBBP4", "RBBP7"))]:
    m = gemmi.read_structure(f"{C}/screen/structures/{pid}.cif")[0]
    A = [(r.seqid.num, a.pos.tolist()) for r in m["A"] for a in r]; B = [(r.seqid.num, a.pos.tolist()) for r in m["B"] for a in r]
    tA, tB = cKDTree([p for _, p in A]), cKDTree([p for _, p in B])
    ra = {A[i][0] for i, nb in enumerate(tB.query_ball_point([p for _, p in A], 5)) if nb}
    rb = {B[i][0] for i, nb in enumerate(tA.query_ball_point([p for _, p in B], 5)) if nb}
    ma, mb = idx2u(ta), idx2u(tb)
    ua = [ma[x] for x in ra if x in ma]; ub = [mb[x] for x in rb if x in mb]
    print(f"{pid}: {ta} {len(ra)} interface residues ({len(ra) - len(ua)} in GS linkers), UniProt {runs(ua)}")
    print(f"   {tb} {len(rb)} interface residues ({len(rb) - len(ub)} in linkers), UniProt {runs(ub)}")
