"""Decisive controls for the GREAT-style region enrichment (user 2026-11-18, worried about background noise):
(1) Empirical null per TF: 100 random subsets of the TF's own peaks, matched to its foreground (TF-specific or conserved elements) in size
    and in the mix of distal/promoter x ChromHMM group (promoter / enhancer / repressive / transcribed / other; HEK293T 18-state, midpoint),
    sampled stratum by stratum; each draw run through the same test (background = all the TF's peaks, region-level hypergeometric, BH,
    >= 5 regions, >= 3 distinct genes, no paralog-cluster artifact). Reported per TF: passing terms real vs null (median, 95th percentile),
    empirical p = (1 + draws with >= as many passing terms) / 101, and per real passing term the fraction of draws where it also passes.
(2) Accessible only: foreground and background restricted to peaks overlapping HEK293T ATAC union peaks (data/HEK293T_ATAC_union3.bed.gz).
Writes null_calibration.tsv, null_term_fraction.tsv, atac_only.tsv.gz, atac_summary.tsv.
  python scripts/go_great_null.py <outdir> <hub_regions.tsv> [draws, default 100]   (Longleaf, hic env)"""
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
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 100
def load_bed(p, col=None):
    d = defaultdict(list)
    for ln in gzip.open(p, "rt"):
        x = ln.split("\t"); d[x[0]].append((int(x[1]), int(x[2]), x[3].strip() if col else ""))
    return {c: (np.array([a for a, _, _ in sorted(v)]), np.array([b for _, b, _ in sorted(v)]), [s for _, _, s in sorted(v)]) for c, v in d.items()}
ATAC = load_bed(X / "data/HEK293T_ATAC_union3.bed.gz"); CH = load_bed(X / "data/HEK293T_chromHMM18.bed.gz", col=3)
CHG = {"TssA": "promoter", "TssFlnk": "promoter", "TssFlnkU": "promoter", "TssFlnkD": "promoter", "TssBiv": "promoter", "EnhA1": "enhancer", "EnhA2": "enhancer",
       "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer", "ReprPC": "repressive", "ReprPCWk": "repressive", "Het": "repressive",
       "ZNF/Rpts": "repressive", "Tx": "transcribed", "TxWk": "transcribed"}
def ov(B, c, s, e):
    if c not in B: return False
    st, en, _ = B[c]; i = np.searchsorted(st, e); return i > 0 and en[i - 1] > s or (i > 1 and en[i - 2] > s)
def chs(c, m):
    if c not in CH: return "other"
    st, en, lab = CH[c]; i = np.searchsorted(st, m, side="right") - 1
    return CHG.get(lab[i], "other") if i >= 0 and en[i] > m else "other"
def passing(fg, bg, Kbg=None):
    n, N = len(fg), len(bg)
    if n < 20: return set(), None
    k = counts(fg); K = counts(bg) if Kbg is None else Kbg; p = hypergeom.sf(k - 1, N, K, n); p = np.where(k >= 5, p, 1.0); q = bh(p)
    cand = np.where(q < 0.05)[0]
    if not len(cand): return set(), K
    Gf = {g for s in fg for g in s if g in gti}; out = set()
    for i in cand:
        gs = t2g[TERMS[i]] & Gf
        if len(gs) < 3: continue
        cc = max(sum(bool(re.match(pt, g)) for g in gs) for _, pt in CL)
        if cc / len(gs) <= 0.5: out.add(TERMS[i])
    return out, K
rng = np.random.default_rng(7); NC, NT, AR = [], [], []
for tf in sorted(files):
    pk = pd.read_csv(files[tf], sep="\t", usecols=["peak"]).peak; rows = []
    for r in pk:
        c, s, e = iv(r); m = (s + e) // 2; d = dist_tss(c, m) > 2000
        rows.append((genes_at(c, m), hub_of(c, s, e, tf), f"{'distal' if d else 'proximal'}|{chs(c, m)}", ov(ATAC, c, s, e)))
    bg = [g for g, *_ in rows]; strata = defaultdict(list)
    for i, (_, _, st, _) in enumerate(rows): strata[st].append(i)
    for grp, sel in (("specific", lambda h: h is not None and h <= 2), ("conserved", lambda h: h is not None and h >= 3)):
        fi = [i for i, r in enumerate(rows) if sel(r[1])]
        real, K = passing([rows[i][0] for i in fi], bg)
        if K is None: continue
        need = pd.Series([rows[i][2] for i in fi]).value_counts(); nul = []; hitc = defaultdict(int)
        for _ in range(NDRAW):
            idx = np.concatenate([rng.choice(strata[s], min(n, len(strata[s])), replace=False) for s, n in need.items()])
            got, _ = passing([rows[i][0] for i in idx], bg, K); nul.append(len(got))
            for t in real & got: hitc[t] += 1
        nul = np.array(nul)
        NC.append(dict(tf=tf, group=grp, fg_elements=len(fi), real_passing_terms=len(real), null_median=float(np.median(nul)), null_p95=float(np.percentile(nul, 95)),
                       empirical_p=(1 + int((nul >= len(real)).sum())) / (NDRAW + 1)))
        for t in real: NT.append(dict(tf=tf, group=grp, term=t, name=name.get(t, ""), null_fraction=hitc[t] / NDRAW))
        # accessible only
        fa = [rows[i][0] for i in fi if rows[i][3]]; ba = [r[0] for r in rows if r[3]]
        ra, _ = passing(fa, ba)
        AR.append(dict(tf=tf, group=grp, fg_elements=len(fi), fg_in_atac=len(fa), frac_fg_in_atac=len(fa) / max(len(fi), 1), bg_in_atac=len(ba),
                       frac_bg_in_atac=len(ba) / len(rows), atac_passing_terms=len(ra), atac_terms=";".join(sorted(name.get(t, t) for t in ra)[:20]),
                       both_real_and_atac=";".join(sorted(name.get(t, t) for t in (ra & real))[:20])))
    print(tf, flush=True)
C = pd.DataFrame(NC); T = pd.DataFrame(NT); Aa = pd.DataFrame(AR)
C.to_csv(OUT / "null_calibration.tsv", sep="\t", index=False); T.to_csv(OUT / "null_term_fraction.tsv", sep="\t", index=False); Aa.to_csv(OUT / "atac_summary.tsv", sep="\t", index=False)
for grp in ("specific", "conserved"):
    c = C[C.group == grp]; a = Aa[Aa.group == grp]
    print(f"== {grp}: TFs tested {len(c)}; with passing terms {(c.real_passing_terms > 0).sum()}; beat null at p<0.05 {(c.empirical_p < 0.05).sum()}; "
          f"ATAC-only TFs with terms {(a.atac_passing_terms > 0).sum()}; median frac fg in ATAC {a.frac_fg_in_atac.median():.2f} vs bg {a.frac_bg_in_atac.median():.2f}")
    print(c.sort_values("empirical_p").head(15).to_string(index=False))
