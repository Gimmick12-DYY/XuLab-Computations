# M0 v2: elements_final.bed (active + bivalent, unchanged ids) plus repressive ChromHMM states
# (ReprPC + ReprPCWk = polycomb, ZNF/Rpts = znf_rpts, Het = het), blacklist removed, no exon mask, tiled to ~1 kb.
# Adds column 8 'class' = promoter / distal_active / bivalent / polycomb / znf_rpts / het
import io, subprocess, sys
import numpy as np, pandas as pd

final, chromhmm, blacklist, out, readme = sys.argv[1:6]
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
REPR = {"polycomb": ["ReprPC", "ReprPCWk"], "znf_rpts": ["ZNF/Rpts"], "het": ["Het"]}
BIV = ["TssBiv", "EnhBiv"]
TILE, MAXW, MINW, MINLEN = 1000, 2000, 300, 50

def bt(*args, inp=None):
    return subprocess.run(["bedtools", *args], input=inp, capture_output=True, text=True, check=True).stdout
def read_bed(text, n=3):
    if not text.strip(): return pd.DataFrame(columns=list(range(n)))
    return pd.read_csv(io.StringIO(text), sep="\t", header=None).iloc[:, :n]
def to_text(df):
    return df.to_csv(sep="\t", header=False, index=False)

CH = pd.read_csv(chromhmm, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "state"])
CH = CH[CH.chrom.isin(CHROMS)]
F = pd.read_csv(final, sep="\t", header=None, names=["chrom", "start", "end", "element_id", "type", "parent_id", "parent_type"])

# ---- active part: classify each tile by ChromHMM bp overlap (bivalent if Biv bp > active-state bp) -------------
ch_txt = to_text(CH.sort_values(["chrom", "start"]))
f_txt = to_text(F[["chrom", "start", "end", "element_id"]])
import tempfile, os
tmp = tempfile.mkdtemp(dir=out)
open(f"{tmp}/ch.bed", "w").write(ch_txt); open(f"{tmp}/f.bed", "w").write(f_txt)
o = pd.read_csv(io.StringIO(bt("intersect", "-wo", "-a", f"{tmp}/f.bed", "-b", f"{tmp}/ch.bed")), sep="\t", header=None)
o = o[[3, 7, 8]]; o.columns = ["element_id", "state", "bp"]
biv_bp = o[o.state.isin(BIV)].groupby("element_id").bp.sum()
oth_bp = o[~o.state.isin(BIV)].groupby("element_id").bp.sum()
F["biv_bp"] = F.element_id.map(biv_bp).fillna(0); F["oth_bp"] = F.element_id.map(oth_bp).fillna(0)
F["class"] = np.where(F.biv_bp > F.oth_bp, "bivalent", np.where(F.type == "promoter", "promoter", "distal_active"))
F = F.drop(columns=["biv_bp", "oth_bp"])

# ---- repressive part ---------------------------------------------------------------------------------------------
def split_even(s, e):
    L = e - s; n = max(1, int(round(L / TILE)))
    b = np.linspace(s, e, n + 1).round().astype(int)
    return [(b[i], b[i + 1]) for i in range(n)]
rows, pid = [], 0
open(f"{tmp}/active.bed", "w").write(f_txt)
for cls, states in REPR.items():
    seg = CH[CH.state.isin(states)].sort_values(["chrom", "start"])
    merged = bt("merge", "-i", "stdin", inp=to_text(seg[["chrom", "start", "end"]]))
    nobl = bt("intersect", "-v", "-a", "stdin", "-b", blacklist, inp=merged)
    # remove anything already in the active/bivalent universe (keeps elements non-overlapping); partial overlaps trimmed
    rest = bt("subtract", "-a", "stdin", "-b", f"{tmp}/active.bed", inp=nobl)
    R = read_bed(rest); R.columns = ["chrom", "start", "end"]
    R = R[R.end - R.start >= MINLEN]
    for c, s, e in R.itertuples(index=False):
        pid += 1; p = f"Q{pid:06d}"
        pieces = [(s, e)] if e - s <= MAXW else split_even(s, e)
        for a, b in pieces:
            rows.append((c, a, b, "repressive", p, "+".join(states), cls))
    print(cls, "segments", len(seg), "merged", len(merged.splitlines()), "after blacklist", len(nobl.splitlines()),
          "after removing active", len(R), flush=True)
Rp = pd.DataFrame(rows, columns=["chrom", "start", "end", "type", "parent_id", "parent_type", "class"])
Rp = Rp.sort_values(["chrom", "start"]).reset_index(drop=True)
Rp.insert(3, "element_id", [f"R{i:06d}" for i in range(1, len(Rp) + 1)])

V = pd.concat([F, Rp]).sort_values(["chrom", "start"]).reset_index(drop=True)
assert (V.end > V.start).all()
for c, d in V.groupby("chrom"):
    assert (d.start.values[1:] >= d.end.values[:-1]).all(), f"overlap on {c}"
V.to_csv(f"{out}/elements_v2.bed", sep="\t", header=False, index=False)
for f in os.listdir(tmp): os.remove(f"{tmp}/{f}")
os.rmdir(tmp)

w = V.end - V.start
S = V.assign(w=w).groupby("class").agg(elements=("w", "size"), Mb=("w", lambda x: x.sum() / 1e6), median_bp=("w", "median"))
S = S.loc[["promoter", "distal_active", "bivalent", "polycomb", "znf_rpts", "het"]]
S.loc["total"] = [len(V), w.sum() / 1e6, w.median()]
print(S.round(1).to_string())
with open(readme, "a") as fh:
    fh.write(f"""
## elements_v2.bed (2026-10-28): active/bivalent + repressive states
Built by build_m0_v2.py. elements_final.bed is kept unchanged (same F...... ids) and each tile gets a class: bivalent if
TssBiv+EnhBiv bp > other-state bp, else promoter (TSS +/- 1 kb) or distal_active. Added ChromHMM ReprPC+ReprPCWk (polycomb),
ZNF/Rpts (znf_rpts) and Het (het): merged per class, blacklist removed, anything overlapping elements_final subtracted,
pieces < 50 bp dropped, NO exon mask (KRAB-ZNF 3' exons are real targets), parents > 2 kb cut into ~1 kb tiles (R...... ids,
parent_id Q......). Columns: chrom, start, end, element_id, type (promoter/distal/repressive), parent_id, parent_type, class.
{S.round(1).to_string()}
""")
