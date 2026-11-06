"""Screen of the exact SEdb HEK293T super-enhancers (SE_02_1951, 342 regions, hg38; ROSE). Anchors: per-TF Cicero clique regions (Our Run k50, 1st step) overlapping the SE with a 5 kb pad. Partners: other regions in the same clique, > 25 kb, outside the SE, whose pair sits on a WT Peakachu 0.7 loop (+-30 kb), pooled by 10 kb bin.
Partner class: promoter = RefSeq Select hg38 TSS within 2.5 kb; enhancer = HEK293T ChromHMM Enh state or inside another SEdb SE; else other. Excludes artifact loci and CTCF-only edge sets (flagged)."""
import glob, gzip, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, re
IN = sys.argv[1]; R0 = Path(__file__).resolve().parents[1]; I = R0 / "inputs"; PAD = 30000
SE = pd.read_csv(R0 / "HEK293T_WT_SEdb_SE_02_1951_hg38.bed", sep="\t", header=None, names=["chrom", "start", "end", "se_id"]); by = {c: d for c, d in SE.groupby("chrom")}
L = defaultdict(list)
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == x[3]:
            a, b = sorted(((int(x[1]), int(x[2])), (int(x[4]), int(x[5])))); L[x[0]].append((*a, *b))
L = {c: np.array(v) for c, v in L.items()}
def on_loop(c, a, b):
    l = L.get(c); return l is not None and bool(((l[:, 0] - PAD < a[1]) & (l[:, 1] + PAD > a[0]) & (l[:, 2] - PAD < b[1]) & (l[:, 3] + PAD > b[0])).any())
def pr(x): c, a = x.split(":"); s, e = map(int, a.split("-")); return c, s, e
R = pd.read_csv(gzip.open(I / "ncbiRefSeqSelect.txt.gz", "rt"), sep="\t", header=None, usecols=[2, 3, 4, 5, 12], names=["c", "st", "s", "e", "g"]); R = R[R.c.str.match(r"^chr[0-9XY]+$")]; R["tss"] = np.where(R.st == "+", R.s, R.e); bt = {c: d for c, d in R.groupby("c")}
Cm = pd.read_csv(gzip.open(I / "HEK293T_chromHMM18.bed.gz", "rt"), sep="\t", header=None, usecols=[0, 1, 2, 3], names=["c", "s", "e", "st"]); Enh = Cm[Cm.st.str.contains("Enh")]; be = {c: d for c, d in Enh.groupby("c")}
TF = pd.read_csv(gzip.open(I / "tf_summits.tsv.gz", "rt"), sep="\t", header=None, names=["c", "p", "tf"]); bf = {c: d for c, d in TF.groupby("c")}
def tfs_in(c, s, e, pad=0):
    d = bf.get(c); return set(d.tf[(d.p >= s - pad) & (d.p < e + pad)]) if d is not None else set()
def cls(x, se_id):
    c, s, e = pr(x); d = bt.get(c); g = d[(d.tss >= s - 2500) & (d.tss <= e + 2500)] if d is not None else []
    if len(g): return "promoter", "/".join(sorted(set(g.g)))
    h = by.get(c); o = h[(h.end > s) & (h.start < e) & (h.se_id != se_id)] if h is not None else []
    k = be.get(c); en = k is not None and bool(((k.e > s) & (k.s < e)).any()); n = d.loc[(d.tss - (s + e) // 2).abs().idxmin()] if d is not None else None; lab = f"{n.g} {'+' if n.tss < s else '-'}{abs(n.tss - (s + e) // 2) // 1000} kb" if n is not None else ""
    return ("enhancer" if en or len(o) else "other"), lab
ART = {"SERPINH1", "DNAJB1", "BRD4", "RPS3", "TECR"}
def artifact(c, s, e):
    d = bt.get(c); gn = set(d.g[(d.e >= s - 5000) & (d.s <= e + 5000)]) if d is not None else set(); hist = any(re.match(r"^(H1-|H2A|H2B|H3C|H4C|H3-|H4-)", q) for q in gn)
    return bool(gn & ART) or hist or (c == "chr11" and s < 75700000 and e > 75300000) or (c == "chr6" and s < 28000000 and e > 25500000)
edge = defaultdict(lambda: dict(tfs=set(), partner=None, anchor=None))
for f_ in sorted(glob.glob(f"{IN}/cl/*/cliques.tsv")):
    tf = Path(f_).parent.name; d = pd.read_csv(f_, sep="\t", usecols=["chromosome", "regions"])
    for x in d.itertuples():
        c = x.chromosome; h = by.get(c)
        if h is None: continue
        rg = x.regions.split(";")
        for u in rg:
            _, s, e = pr(u); hit = h[(h.end + 5000 > s) & (h.start - 5000 < e)]
            for se in hit.itertuples():
                for v in rg:
                    if v == u: continue
                    _, s2, e2 = pr(v); mu, mv = (s + e) // 2, (s2 + e2) // 2
                    if abs(mu - mv) <= 25000 or (s2 < se.end and e2 > se.start): continue
                    if not on_loop(c, *(((s, e), (s2, e2)) if mu < mv else ((s2, e2), (s, e)))): continue
                    t = edge[(se.se_id, mv // 10000)]; t["tfs"].add(tf); t["partner"] = t["partner"] or v; t["anchor"] = t["anchor"] or u
E = pd.DataFrame([dict(se_id=k[0], bin=k[1], partner=t["partner"], anchor=t["anchor"], n_tfs=len(t["tfs"]), tfs=",".join(sorted(t["tfs"]))) for k, t in edge.items()]); q_ = [cls(p, s) for p, s in zip(E.partner, E.se_id)]; E["cls"] = [a for a, _ in q_]; E["label"] = [b for _, b in q_]; E.to_csv(R0 / "se_loop_edges.tsv", sep="\t", index=False); print("SEs with a loop-edge", E.se_id.nunique(), "edges", len(E), E.cls.value_counts().to_dict())
rows = []
for se in SE.itertuples():
    d = E[E.se_id == se.se_id]; tf = tfs_in(se.chrom, se.start, se.end); allt = set(",".join(d.tfs).split(",")) if len(d) else set()
    rows.append(dict(se_id=se.se_id, locus=f"{se.chrom}:{se.start}-{se.end}", kb=round((se.end - se.start) / 1000, 1), n_partners=len(d), promoters=int((d.cls == "promoter").sum()), enhancers=int((d.cls == "enhancer").sum()), other=int((d.cls == "other").sum()), shared2=int((d.n_tfs >= 2).sum()), tfs_in_se=len(tf), frac_pe=round(float((d.cls != "other").mean()), 2) if len(d) else 0, artifact=artifact(se.chrom, se.start, se.end), ctcf_only=allt == {"CTCF"}, genes=",".join(sorted({x for x in d.label[d.cls == "promoter"] for x in x.split("/")}))[:70]))
Q = pd.DataFrame(rows).sort_values(["n_partners", "tfs_in_se", "frac_pe"], ascending=False); Q["rank"] = np.arange(1, len(Q) + 1); K = Q[~Q.artifact & ~Q.ctcf_only].copy(); K["rank_clean"] = np.arange(1, len(K) + 1); K.to_csv(R0 / "se_ranked.tsv", sep="\t", index=False); print("clean", len(K), "excluded", len(Q) - len(K)); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); c_ = "rank_clean se_id locus kb n_partners promoters enhancers other shared2 tfs_in_se frac_pe genes".split(); print(K[c_].head(10).to_string(index=False)); print(K[K.se_id.isin(["SE_02_195100173", "SE_02_195100324", "SE_02_195100223"])][c_].to_string(index=False)); ch = K[(K.locus.str.startswith("chr16:")) & K.locus.str.contains(r"chr16:(?:29804210|30056475|30541264|30649707)-")]; print(ch[c_].to_string(index=False))
print(E[E.se_id == "SE_02_195100173"].sort_values("bin")[["partner", "anchor", "n_tfs", "tfs", "cls", "label"]].to_string(index=False))
