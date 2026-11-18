"""Region-based GO enrichment (GREAT-style, McLean et al. 2010; implemented locally) for TF-specific vs conserved module elements
(user 2026-11-18). Unit = module element (promoter or distal), not promoter genes.
Elements: merged module regions (ourrun_k50 clique hubs, blacklist removed; hub_regions.tsv: region, n_tfs, tfs). Per TF, a peak of that TF
(work/ourrun_k50/<TF>/<TF>.peak_annot.tsv; RBBP4 from its sweep run) is a TF-specific element if it overlaps a module region of that TF with
n_tfs 1-2, conserved if n_tfs >= 3.
Gene assignment: GREAT basal+extension regulatory domains on GENCODE v46 protein-coding genes (one TSS per gene): basal = TSS -5 kb/+1 kb
(strand-aware), extended both ways to the nearest neighbouring basal domain, at most 1 Mb. A region (midpoint) is linked to every gene whose
domain contains it. GO: goa_human.gaf + go-basic.obo, BP, ancestors, NOT dropped; terms with 10-500 genes among domain genes.
Test per TF: foreground = its TF-specific (or conserved) peaks; background = all its peaks; region-level one-sided hypergeometric
(foreground regions hitting the term vs background regions hitting it); BH within the list; a term needs >= 5 foreground regions.
Pooled: all TFs' foreground peaks vs all TFs' peaks (stacked). Lists with < 20 foreground regions: n/a.
Artifact flag: > 50% of the hit genes from one paralog cluster (OR, PCDH, KRTAP, ZNF on chr19, histone, UGT, TAS2R).
Distal = element midpoint > 2 kb from any protein-coding TSS.
  python scripts/go_great_regions.py <outdir> <hub_regions.tsv>   (Longleaf, hic env)"""
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
def test(fg, bg, label):
    n, N = len(fg), len(bg)
    if n < 20: return None
    k, K = counts(fg), counts(bg); p = hypergeom.sf(k - 1, N, K, n); ok = k >= 5; p = np.where(ok, p, 1.0)
    R = pd.DataFrame({"list": label, "term": TERMS, "name": [name.get(t, "") for t in TERMS], "fg_regions": k, "n_fg": n, "bg_regions": K, "n_bg": N,
                      "fold": np.where(K > 0, (k / n) / np.maximum(K / N, 1e-12), 0), "p": p}); R["q"] = bh(R.p)
    R = R[R.fg_regions >= 5].sort_values("p")
    hit = []
    for t in R.term.head(15):
        gs = sorted({g for s in fg for g in s if g in t2g[t]}); cc = {c: sum(bool(re.match(pt, g)) for g in gs) for c, pt in CL}; top = max(cc, key=cc.get)
        hit.append((t, ",".join(gs[:30]), top if cc[top] else "", cc[top] / max(len(gs), 1)))
    H = pd.DataFrame(hit, columns=["term", "hit_genes", "top_cluster", "cluster_frac"]); R = R.merge(H, on="term", how="left")
    R["cluster_artifact"] = R.cluster_frac.fillna(0) > 0.5; return R
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
res, pooled, dist = [], {"specific": ([], []), "conserved": ([], [])}, []
allbg = []
for tf in sorted(files):
    pk = pd.read_csv(files[tf], sep="\t", usecols=["peak"]).peak
    rows = []
    for r in pk:
        c, s, e = iv(r); m = (s + e) // 2; rows.append((genes_at(c, m), hub_of(c, s, e, tf), dist_tss(c, m) > 2000))
    bg = [g for g, _, _ in rows]; spec = [g for g, h, _ in rows if h is not None and h <= 2]; cons = [g for g, h, _ in rows if h is not None and h >= 3]
    dist.append(dict(tf=tf, peaks=len(rows), specific_elements=len(spec), conserved_elements=len(cons),
                     specific_distal_frac=np.mean([d for _, h, d in rows if h is not None and h <= 2]) if spec else np.nan,
                     conserved_distal_frac=np.mean([d for _, h, d in rows if h is not None and h >= 3]) if cons else np.nan,
                     all_peaks_distal_frac=np.mean([d for _, _, d in rows])))
    allbg += bg; pooled["specific"][0].extend(spec); pooled["conserved"][0].extend(cons)
    for grp, fg in (("specific", spec), ("conserved", cons)):
        R = test(fg, bg, f"{tf}|{grp}")
        if R is not None: res.append(R.assign(tf=tf, group=grp))
    print(tf, len(rows), len(spec), len(cons), flush=True)
DS = pd.DataFrame(dist); DS.to_csv(OUT / "distal_fractions.tsv", sep="\t", index=False)
P = {grp: test(pooled[grp][0], allbg, f"pooled|{grp}") for grp in ("specific", "conserved")}
for grp, R in P.items(): R.to_csv(OUT / f"pooled_{grp}.tsv", sep="\t", index=False)
A = pd.concat(res); A.to_csv(OUT / "per_TF_all_tested_terms.tsv.gz", sep="\t", index=False)
S = A[A.q < 0.05]; S.to_csv(OUT / "per_TF_q005.tsv", sep="\t", index=False)
tested = A.groupby(["group", "tf"]).size().reset_index()[["group", "tf"]]
summ = []
for grp in ("specific", "conserved"):
    tt = tested[tested.group == grp].tf; s = S[S.group == grp]
    summ.append(dict(group=grp, TFs_tested=len(tt), TFs_with_terms=s.tf.nunique(), pooled_fg=int(P[grp].n_fg.iloc[0]), pooled_terms_q005=int((P[grp].q < 0.05).sum())))
pd.DataFrame(summ).to_csv(OUT / "summary.tsv", sep="\t", index=False)
print(pd.DataFrame(summ).to_string(index=False))
tot = lambda c: DS[c].sum()
print("distal fraction (element-weighted): specific", round((DS.specific_distal_frac * DS.specific_elements).sum() / tot("specific_elements"), 3),
      "conserved", round((DS.conserved_distal_frac * DS.conserved_elements).sum() / tot("conserved_elements"), 3), "all peaks", round((DS.all_peaks_distal_frac * DS.peaks).sum() / tot("peaks"), 3))
for grp in P: print("== pooled", grp); print(P[grp].head(12)[["name", "fg_regions", "fold", "q", "top_cluster", "cluster_frac"]].round(4).to_string(index=False))
for grp in ("specific", "conserved"):
    s = S[S.group == grp]; print(f"== per TF {grp}: TFs with terms", s.tf.nunique())
    for tf, d in s.groupby("tf"): print(" ", tf, len(d), "|", "; ".join(f"{n} (q={q:.2g}{', ARTIFACT ' + c if a else ''})" for n, q, c, a in zip(d.name[:3], d.q[:3], d.top_cluster[:3], d.cluster_artifact[:3])))
