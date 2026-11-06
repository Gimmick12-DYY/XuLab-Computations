"""Wider screen, same dot test as 01_screen_dots.py: every Cicero clique anchor (any TF) inside one of the 342 SE candidates, distal, pooled by 10 kb bin as a core; partners = regions in the same clique, > 25 kb, on a Peakachu 0.7 loop (+-30 kb), pooled by 10 kb bin. No minimum number of TFs at the core; candidates need >= 3 partners."""
import glob, itertools, sys
from collections import defaultdict
from pathlib import Path
src = open(Path(__file__).resolve().parent / "01_screen_dots.py").read(); exec(src[:src.index("rows = []\nfor h in HN.itertuples():")])
L = defaultdict(list); PAD = 30000
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == x[3]:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L[x[0]].append((*a, *b))
L = {c: np.array(v) for c, v in L.items()}
def on_loop(c, a, b):
    l = L.get(c); return l is not None and bool(((l[:, 0] - PAD < a[1]) & (l[:, 1] + PAD > a[0]) & (l[:, 2] - PAD < b[1]) & (l[:, 3] + PAD > b[0])).any())
sebyc = {c: d for c, d in SE.groupby("chrom")}; edge = defaultdict(lambda: dict(tfs=set(), core=None, partner=None)); nreg = 0
for f_ in sorted(glob.glob(f"{IN}/cl/*/cliques.tsv")):
    tf = Path(f_).parent.name; d = pd.read_csv(f_, sep="\t", usecols=["n_regions", "chromosome", "regions"])
    for x in d.itertuples():
        c = x.chromosome; se = sebyc.get(c)
        if se is None: continue
        rg = x.regions.split(";")
        for u in rg:
            _, s, e = pr(u); h = se[(se.end > s) & (se.start < e)]
            if h.empty: continue
            for v in rg:
                if v == u: continue
                _, s2, e2 = pr(v); mu, mv = (s + e) // 2, (s2 + e2) // 2
                if abs(mu - mv) <= 25000 or not on_loop(c, *(((s, e), (s2, e2)) if mu < mv else ((s2, e2), (s, e)))): continue
                k = (c, mu // r, mv // r); t = edge[k]; t["tfs"].add(tf); t["core"] = t["core"] or u; t["partner"] = t["partner"] or v; t["se"] = h.iloc[0].se_id
print("SE-anchored loop-edges", len(edge))
E = pd.DataFrame([dict(chrom=k[0], bc=k[1], bp=k[2], core=t["core"], partner=t["partner"], se_id=t["se"], n_tfs=len(t["tfs"]), tfs=",".join(sorted(t["tfs"]))) for k, t in edge.items()]); E["core_bin"] = E.chrom + ":" + E.bc.astype(str); tc = {}
for cb, d in E.groupby("core_bin"): tc[cb] = set(",".join(d.tfs).split(","))
E["core_tss"] = [",".join(tss_g(x)) for x in E.core]; E = E[E.core_tss == ""]; n_ = E.groupby("core_bin").size(); E = E[E.core_bin.isin(n_[n_ >= 3].index)].reset_index(drop=True); print("distal cores with >= 3 partners", E.core_bin.nunique(), "SEs", E.se_id.nunique(), "edges", len(E))
res = []
for c, sub in E.groupby("chrom"):
    Raw, Ice, valid, exp, n = load(c); nulc = {}; I_, J_ = np.minimum(sub.bc.values, sub.bp.values), np.maximum(sub.bc.values, sub.bp.values); be, oe, rd, sa, sb, ex = score(c, I_, J_); ps = []
    for b_, d_ in zip(be, J_ - I_):
        if not np.isfinite(b_): ps.append(np.nan); continue
        if d_ not in nulc:
            a = rng.integers(10, n - d_ - 10, NN * 2); a = a[valid[a] & valid[a + d_]][:NN]; nn_ = score(c, a, a + d_)[0]; nulc[d_] = nn_[np.isfinite(nn_)]
        nn = nulc[d_]; ps.append(float((1 + (nn >= b_).sum()) / (len(nn) + 1)))
    res.append(sub.assign(i=I_, j=J_, best_don=be, best_oe=oe, best_reads=rd, si=sa, sj=sb, exact_don=ex, p=ps)); print(c, len(sub), len(nulc), flush=True)
P = pd.concat(res).reset_index(drop=True); t_ = P.p.notna(); m = int(t_.sum()); o = P.p[t_].sort_values(); q = (o * m / np.arange(1, m + 1))[::-1].cummin()[::-1]; P["q"] = np.nan; P.loc[o.index, "q"] = q; P["clear"] = (P.best_don >= 1.5) & (P.q < 0.05); P["clear_raw"] = (P.best_don >= 1.5) & (P.p < 0.05); P["bi"] = P.i + P.si; P["bj"] = P.j + P.sj; P["p_tss"] = [",".join(tss_g(x)) for x in P.partner]; P["p_art"] = [artifact(x) for x in P.partner]; P["p_exon"] = [exonic(x) for x in P.partner]; P.to_csv(R0 / "partner_dots_wide.tsv", sep="\t", index=False); print("tested", m, "of", len(P), "clear (q)", int(P.clear.sum()), "clear (raw p)", int(P.clear_raw.sum()), "min p", P.p.min(), "min q", P.q.min())
def distinct(d):
    pts = d[["bi", "bj"]].values; lab = list(range(len(pts)))
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if max(abs(pts[a] - pts[b])) <= 1:
                la, lb = lab[a], lab[b]; lab = [la if x == lb else x for x in lab]
    return len(set(lab))
out = []
for cb, d in P.groupby("core_bin"):
    se = SE[SE.se_id == d.se_id.iloc[0]].iloc[0]; t = T2[(T2.chrom == se.chrom) & (T2.end > d.bc.iloc[0] * r) & (T2.start < d.bc.iloc[0] * r + r)]; cl, clr = d[d.clear], d[d.clear_raw]; out.append(dict(se_id=se.se_id, core_bin=cb, core=d.core.iloc[0], se_rank=se["rank"], hek293t_rank=int(t["rank"].iloc[0]) if len(t) else -1, tfs_at_core=len(tc[cb]), tf_list=",".join(sorted(tc[cb]))[:60], n_partners=len(d), n_tested=int(d.p.notna().sum()), clear=len(cl), distinct_clear=distinct(cl) if len(cl) else 0, distinct_raw=distinct(clr) if len(clr) else 0, frac_raw=len(clr) / len(d), mean_don_raw=float(clr.best_don.mean()) if len(clr) else np.nan, genes=",".join(sorted({g_ for x in d.p_tss for g_ in x.split(",") if g_}))[:60], artifact=artifact(d.core.iloc[0]) or bool(d.p_art.mean() > 0.3), frac_exon=round(float(d.p_exon.mean()), 2), ctcf_only=tc[cb] == {"CTCF"}))
Q = pd.DataFrame(out).sort_values(["distinct_clear", "distinct_raw", "frac_raw", "mean_don_raw"], ascending=False); Q.to_csv(R0 / "se_candidates_ranked_wide.tsv", sep="\t", index=False); pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30); c_ = "se_id core se_rank hek293t_rank tfs_at_core n_partners n_tested clear distinct_clear distinct_raw frac_raw mean_don_raw artifact frac_exon ctcf_only genes".split(); print("candidates", len(Q), "SEs", Q.se_id.nunique()); print(Q.distinct_clear.value_counts().sort_index().to_dict(), Q.distinct_raw.value_counts().sort_index().to_dict()); print(Q[c_].head(14).round(2).to_string(index=False)); print(Q.reset_index(drop=True).assign(rk=lambda x: x.index + 1)[lambda x: x.se_id == "HEK293_SE_0027"][["rk"] + c_].round(2).to_string(index=False))
