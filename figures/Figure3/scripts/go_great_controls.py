"""Controls for the GREAT-style region enrichment (scripts/go_great_regions.py, user 2026-11-18):
(1) pooled lists count each element once: all TFs' peaks merged by overlap into one region set; foreground = merged regions overlapping a
    module region of the class (1-2 TFs = TF-specific, >= 3 TFs = conserved), background = all merged regions;
(2) composition control: distal foreground vs distal background only, and promoter (<= 2 kb from a protein-coding TSS) vs promoter only,
    per TF (TF's own peaks) and pooled;
(3) a term must be hit through >= 3 distinct genes (genes whose GREAT domain holds a foreground region); the top contributing genes and
    the share of foreground regions from the single top gene are reported;
(4) all of this for TF-specific and conserved elements. Domains, GO sets, region-level hypergeometric test, BH, >= 5 foreground regions,
    paralog-cluster flag: as go_great_regions.py.
Writes per_TF_controls.tsv.gz (all q < 0.2 terms), pooled_controls.tsv.gz, surviving_per_TF.tsv.
  python scripts/go_great_controls.py <outdir> <hub_regions.tsv>   (Longleaf, hic env)"""
import glob, gzip, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import hypergeom
X = Path("/work/users/d/y/dyy12/XuLab"); D = X / "data/go"; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); HUB = Path(sys.argv[2])
# ---- genes, TSS, GREAT domains ----
G = []
for ln in gzip.open(X / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene" or 'gene_type "protein_coding"' not in x[8]: continue
    G.append((x[0], int(x[3]) if x[6] == "+" else int(x[4]), x[6], re.search('gene_name "([^"]+)"', x[8]).group(1)))
G = pd.DataFrame(G, columns=["chrom", "tss", "strand", "gene"]).drop_duplicates("gene").sort_values(["chrom", "tss"]).reset_index(drop=True)
G["bs"] = np.where(G.strand == "+", G.tss - 5000, G.tss - 1000); G["be"] = np.where(G.strand == "+", G.tss + 1000, G.tss + 5000)
dom = {}
for c, d in G.groupby("chrom"):
    bs, be, tss = d.bs.values, d.be.values, d.tss.values; n = len(d)
    lo = np.array([min(bs[i], max(tss[i] - 1_000_000, be[i - 1] if i > 0 else 0)) for i in range(n)])
    hi = np.array([max(be[i], min(tss[i] + 1_000_000, bs[i + 1] if i < n - 1 else 10 ** 10)) for i in range(n)])
    dom[c] = (tss, lo, hi, d.gene.values)
def genes_at(c, m):
    if c not in dom: return ()
    tss, lo, hi, g = dom[c]; i = np.searchsorted(tss, m); out = []
    for j in range(max(0, i - 4), min(len(tss), i + 4)):
        if lo[j] <= m <= hi[j]: out.append(g[j])
    return tuple(sorted(out))
def dist_tss(c, m):
    if c not in dom: return 10 ** 9
    tss = dom[c][0]; i = np.searchsorted(tss, m); return min(abs(tss[j] - m) for j in (i - 1, i) if 0 <= j < len(tss))
# ---- GO ----
par, name, cur = defaultdict(set), {}, None
for ln in open(D / "go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("name: "): name[cur] = ln[6:]
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
anc = {}; sys.setrecursionlimit(10000)
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
GS = set(G.gene); g2t = defaultdict(set)
for ln in gzip.open(D / "goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P" or x[2] not in GS: continue
    g2t[x[2]] |= up(x[4])
t2g = defaultdict(set)
for g, ts in g2t.items():
    for t in ts: t2g[t].add(g)
TERMS = [t for t, s in t2g.items() if 10 <= len(s) <= 500]; ti = {t: i for i, t in enumerate(TERMS)}
gti = {g: np.array(sorted(ti[t] for t in ts if t in ti), dtype=np.int32) for g, ts in g2t.items()}
cache = {}
def tidx(gs):
    if gs not in cache: cache[gs] = np.unique(np.concatenate([gti.get(g, np.zeros(0, np.int32)) for g in gs])) if gs else np.zeros(0, np.int32)
    return cache[gs]
def counts(gsets):
    a = [tidx(g) for g in gsets]; return np.bincount(np.concatenate(a), minlength=len(TERMS)) if a else np.zeros(len(TERMS), int)
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1); q = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out
CL = [("OR", r"^OR\d"), ("PCDH", r"^PCDH"), ("KRTAP", r"^KRTAP"), ("ZNF", r"^ZNF"), ("histone", r"^(HIST|H2A|H2B|H3C|H4C|H1-)"), ("UGT", r"^UGT"), ("TAS2R", r"^TAS2R")]
def test(fg, bg, label, extra=None):
    n, N = len(fg), len(bg)
    if n < 20: return None
    k, K = counts(fg), counts(bg); p = hypergeom.sf(k - 1, N, K, n); p = np.where(k >= 5, p, 1.0)
    gcount = defaultdict(int)
    for s in fg:
        for g in s: gcount[g] += 1
    Gf = np.array([g for g in gcount if g in gti]); dg = np.bincount(np.concatenate([gti[g] for g in Gf]), minlength=len(TERMS)) if len(Gf) else np.zeros(len(TERMS), int)
    R = pd.DataFrame({"list": label, "term": TERMS, "name": [name.get(t, "") for t in TERMS], "fg_regions": k, "n_fg": n, "bg_regions": K, "n_bg": N,
                      "fold": np.where(K > 0, (k / n) / np.maximum(K / N, 1e-12), 0), "distinct_genes": dg, "p": p}); R["q"] = bh(R.p)
    R = R[(R.fg_regions >= 5) & (R.q < 0.2)].sort_values("p").copy()
    tops, shares, cls, cfr = [], [], [], []
    for t, kk in zip(R.term, R.fg_regions):
        gs = sorted(((gcount[g], g) for g in t2g[t] if g in gcount), reverse=True)
        tops.append(",".join(f"{g}({c})" for c, g in gs[:5])); shares.append(gs[0][0] / kk if gs else 0)
        names_ = [g for _, g in gs]; cc = {c: sum(bool(re.match(pt, g)) for g in names_) for c, pt in CL}; top = max(cc, key=cc.get)
        cls.append(top if cc[top] else ""); cfr.append(cc[top] / max(len(names_), 1))
    R["top_genes(regions)"] = tops; R["top_gene_share"] = shares; R["top_cluster"] = cls; R["cluster_frac"] = cfr
    R["passes"] = (R.q < 0.05) & (R.distinct_genes >= 3) & (np.array(cfr) <= 0.5)
    return R if extra is None else R.assign(**extra)
# ---- module regions and per-TF peaks ----
Hh = pd.read_csv(HUB, sep="\t", usecols=["hub_region", "n_tfs", "tfs"])
def iv(r):
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)
HR = defaultdict(list)
for r, n, tfs in zip(Hh.hub_region, Hh.n_tfs, Hh.tfs):
    c, s, e = iv(r); HR[c].append((s, e, n, set(tfs.split(","))))
HRs = {c: (np.array([a for a, *_ in sorted(v)]), sorted(v)) for c, v in HR.items()}
def hub_of(c, s, e, tf):
    if c not in HRs: return None
    st, v = HRs[c]; i = np.searchsorted(st, e)
    for j in range(max(0, i - 3), i):
        a, b, n, tfs = v[j]
        if a < e and b > s and tf in tfs: return n
    return None
files = {Path(f).parent.name: f for f in glob.glob(str(X / "cobinding/work/ourrun_k50/*/*.peak_annot.tsv"))}
files["RBBP4"] = str(X / "cobinding/work/RBBP4_sweep_k50_i10000_s555/RBBP4.peak_annot.tsv")
res = []; allpk = defaultdict(list)
for tf in sorted(files):
    pk = pd.read_csv(files[tf], sep="\t", usecols=["peak"]).peak; rows = []
    for r in pk:
        c, s, e = iv(r); m = (s + e) // 2; h = hub_of(c, s, e, tf); rows.append((genes_at(c, m), h, dist_tss(c, m) > 2000)); allpk[c].append((s, e))
    for grp, sel in (("specific", lambda h: h is not None and h <= 2), ("conserved", lambda h: h is not None and h >= 3)):
        for comp, dsel in (("all", lambda d: True), ("distal_vs_distal", lambda d: d), ("promoter_vs_promoter", lambda d: not d)):
            fg = [g for g, h, d in rows if sel(h) and dsel(d)]; bg = [g for g, h, d in rows if dsel(d)]
            R = test(fg, bg, f"{tf}|{grp}|{comp}", dict(tf=tf, group=grp, comparison=comp))
            if R is not None: res.append(R)
    print(tf, len(rows), flush=True)
A = pd.concat([r for r in res if len(r)]); A.to_csv(OUT / "per_TF_controls.tsv.gz", sep="\t", index=False)
# pooled, each element once: merge all TFs' peaks
U = []
for c, v in allpk.items():
    v.sort(); cs, ce = v[0]
    for s, e in v[1:]:
        if s <= ce: ce = max(ce, e)
        else: U.append((c, cs, ce)); cs, ce = s, e
    U.append((c, cs, ce))
def hub_any(c, s, e):
    if c not in HRs: return None
    st, v = HRs[c]; i = np.searchsorted(st, e); best = None
    for j in range(max(0, i - 6), i):
        a, b, n, _ = v[j]
        if a < e and b > s: best = max(best or 0, n)
    return best
UR = [(genes_at(c, (s + e) // 2), hub_any(c, s, e), dist_tss(c, (s + e) // 2) > 2000) for c, s, e in U]
PR = []
for grp, sel in (("specific", lambda h: h is not None and h <= 2), ("conserved", lambda h: h is not None and h >= 3)):
    for comp, dsel in (("all", lambda d: True), ("distal_vs_distal", lambda d: d), ("promoter_vs_promoter", lambda d: not d)):
        fg = [g for g, h, d in UR if sel(h) and dsel(d)]; bg = [g for g, h, d in UR if dsel(d)]
        R = test(fg, bg, f"pooled|{grp}|{comp}", dict(tf="pooled", group=grp, comparison=comp))
        if R is not None: PR.append(R)
P = pd.concat(PR); P.to_csv(OUT / "pooled_controls.tsv.gz", sep="\t", index=False)
print("merged regions", len(U), "; pooled fg sizes:", {f"{r.group.iloc[0]}|{r.comparison.iloc[0]}": int(r.n_fg.iloc[0]) for r in PR if len(r)})
S = A[A.passes & (A.comparison == "distal_vs_distal")]
S.to_csv(OUT / "surviving_per_TF.tsv", sep="\t", index=False)
for grp in ("specific", "conserved"):
    s = S[S.group == grp]; t_all = A[(A.group == grp) & (A.comparison == "all") & A.passes].tf.nunique()
    print(f"== {grp}: TFs with passing terms (all elements) {t_all}; after distal-vs-distal {s.tf.nunique()}")
    for tf, d in s.groupby("tf"): print("  ", tf, len(d), "|", "; ".join(f"{n} (q={q:.2g}, {g} genes; {tg})" for n, q, g, tg in zip(d.name[:2], d.q[:2], d.distinct_genes[:2], d["top_genes(regions)"][:2])))
for grp in ("specific", "conserved"):
    for comp in ("all", "distal_vs_distal", "promoter_vs_promoter"):
        d = P[(P.group == grp) & (P.comparison == comp) & P.passes]
        print(f"== pooled {grp} {comp}: passing terms {len(d)}"); print(d.head(8)[["name", "fg_regions", "fold", "distinct_genes", "q", "top_gene_share"]].round(4).to_string(index=False))
