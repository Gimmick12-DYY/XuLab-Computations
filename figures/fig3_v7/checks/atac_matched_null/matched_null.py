#!/usr/bin/env python3
"""Fig 3 v7 accessibility control, step 2: per-edge shuffle null for module edges >25 kb with BOTH anchors drawn from matched regions.
Same edge set, loops (Peakachu 5 kb + 10 kb, 0.7), +-30 kb slop and scoring as the panel a null (fill_slide_loop_table.score_pairs).
Each anchor is assigned to its universe region (build_universe.py) and stratum:
  --variant atac          (promoter yes/no) x ATAC decile (0 = not in the ATAC union, 1-10)
  --variant atac_density  ... x decile of this TF's network-peak density (its peak_annot peaks within +-500 kb; decile cut points from
                          this TF's observed anchors)
One shuffle of an edge: draw anchor 1 uniformly from anchor 1's stratum on the same chromosome; place anchor 2 at the region of anchor
2's stratum whose midpoint is nearest to mid1 +- spacing (random side; other side if closer); the shuffled peaks keep the observed
peak widths, centred on the drawn regions. Spacing error is recorded (median relative error per shuffle).
  task:    matched_null.py --tf TF --variant V --edges E --cliques C --annot A --fine F --coarse K --n-perm 1000 --chunk k --out x.npz"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/hic/scripts")
import fill_slide_loop_table as F  # noqa: E402
ap = argparse.ArgumentParser()
for k in ("--tf", "--variant", "--edges", "--cliques", "--annot", "--fine", "--coarse", "--out"): ap.add_argument(k, required=True)
ap.add_argument("--n-perm", type=int, default=1000); ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args(); HERE = Path(__file__).resolve().parent
fine = F.U.load_bedpe(Path(a.fine), "5000"); coarse = F.U.load_bedpe(Path(a.coarse), "10000")
ix5 = F.U.index_anchors(fine, 5000); ix10 = F.U.index_anchors(coarse, 10000)
edges = [e for e in F.C.load_clique_edges(Path(a.cliques)) if F.span_bp(e) > F.MIN_SPAN]
obs = F.score_pairs(edges, fine, coarse, ix5, ix10)[0] / len(edges)
U = pd.read_csv(HERE / "inputs/universe.tsv.gz", sep="\t"); U["m"] = (U.s + U.e) // 2
if a.variant == "atac_density":
    pk = pd.read_csv(a.annot, sep="\t", usecols=["peak"]); pc = pk.peak.str.split(":").str[0]; pm = pk.peak.str.split(":").str[1].str.split("-").apply(lambda v: (int(v[0]) + int(v[1])) // 2)
    dens = np.zeros(len(U), dtype=int)
    for c, g in U.groupby("c"):
        q = np.sort(pm[pc == c].values)
        if len(q): dens[g.index] = np.searchsorted(q, g.m.values + 500_000, side="right") - np.searchsorted(q, g.m.values - 500_000)
    U["dens"] = dens
UC = {c: g.sort_values("s") for c, g in U.groupby("c")}
def uidx(c, s, e):
    g = UC[c]; i = np.searchsorted(g.s.values, (s + e) // 2, side="right") - 1; return g.index.values[i]
A1 = np.array([uidx(*e["p1"]) for e in edges]); A2 = np.array([uidx(*e["p2"]) for e in edges])
if a.variant == "atac_density":
    cuts = np.unique(np.percentile(U.dens.values[np.r_[A1, A2]], np.arange(10, 100, 10)))
    U["dd"] = np.searchsorted(cuts, U.dens.values, side="right")
else: U["dd"] = 0
U["st"] = U.prom.astype(str) + "_" + U.atac_dec.astype(str) + "_" + U.dd.astype(str)
pool = {(c, st): (g.m.values, g.index.values) for (c, st), g in U.sort_values("m").groupby(["c", "st"])}
chrom = np.array([e["p1"][0] for e in edges]); st1 = U.st.values[A1]; st2 = U.st.values[A2]
w1 = np.array([e["p1"][2] - e["p1"][1] for e in edges]); w2 = np.array([e["p2"][2] - e["p2"][1] for e in edges])
m1o = np.array([(e["p1"][1] + e["p1"][2]) // 2 for e in edges]); m2o = np.array([(e["p2"][1] + e["p2"][2]) // 2 for e in edges]); sp = np.abs(m2o - m1o)
groups = pd.DataFrame({"c": chrom, "a": st1, "b": st2}).groupby(["c", "a", "b"]).indices
rng = np.random.default_rng(30_000 * (a.chunk + 1) + (11 if a.variant == "atac" else 13)); out = np.empty(a.n_perm); err = np.empty(a.n_perm)
for it in range(a.n_perm):
    M1 = np.empty(len(edges), dtype=np.int64); M2 = np.empty(len(edges), dtype=np.int64)
    for (c, s1_, s2_), ix in groups.items():
        p1m, _ = pool[(c, s1_)]; p2m, _ = pool[(c, s2_)]
        r1 = p1m[rng.integers(0, len(p1m), len(ix))]; sgn = rng.choice([-1, 1], len(ix)); best = None
        for sg in (sgn, -sgn):
            t = r1 + sg * sp[ix]; j = np.clip(np.searchsorted(p2m, t), 1, len(p2m) - 1) if len(p2m) > 1 else np.zeros(len(ix), dtype=int)
            cand = np.where(np.abs(p2m[j - 1] - t) <= np.abs(p2m[j] - t), p2m[j - 1], p2m[j]) if len(p2m) > 1 else p2m[j]
            best = cand if best is None else np.where(np.abs(cand - (r1 - sgn * sp[ix])) < np.abs(best - (r1 + sgn * sp[ix])), cand, best)
        M1[ix] = r1; M2[ix] = best
    sh = [{"p1": (c, int(x - w // 2), int(x + w - w // 2)), "p2": (c, int(y - v // 2), int(y + v - v // 2))} for c, x, y, w, v in zip(chrom, M1, M2, w1, w2)]
    out[it] = F.score_pairs(sh, fine, coarse, ix5, ix10)[0] / len(edges); err[it] = np.median(np.abs(np.abs(M2 - M1) - sp) / sp)
    if (it + 1) % 100 == 0: print(a.tf, a.variant, a.chunk, it + 1, flush=True)
np.savez(a.out, null=out, obs=np.array(obs), spacing_err=err, n=np.array(len(edges)))
print("done", a.tf, a.variant, a.chunk, obs, np.median(out), np.median(err), flush=True)
