"""SE-hub step 1: call HEK293 super-enhancers (ROSE-style) and find TF hubs with an SE anchor on a loop-edge.

Super-enhancers: no HEK293/HEK293T sample in dbSUPER; SEdb per-sample files are not directly downloadable. So ROSE-style
calling (Whyte 2013; Loven 2013) from ENCODE HEK293 H3K27ac (ENCSR000FCH) replicated peaks (ENCFF451UZW, GRCh38):
  peaks fully inside TSS +-2.5 kb are excluded; remaining peaks stitched within 12.5 kb; region score = sum of
  signalValue x width of its peaks (peak-signal proxy instead of read density, since no BAM/bigWig was downloaded);
  SE = regions above the ROSE inflection point (rank where the slope of the normalised curve reaches 1).
Cross-checks: overlap with ChromHMM-18 enhancer states (HEK293T, hg38) and union ATAC.
Candidates: Cicero shared clique edges (top_shared_clique_edges.tsv) that are loop-edges (on_loop_0.7 == 1, >25 kb) with one
end in an SE and the other end a promoter (TSS +-2.5 kb); grouped per SE. Score: n SE loop-edges, TFs with a summit in the SE,
distinct linked promoter genes. Excluded genes: SERPINH1, DNAJB1, BRD4, RPS3 (expressed-gene/exon artifacts).
"""
import argparse, glob, gzip, os, re
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--peaks", required=True); ap.add_argument("--gtf", required=True); ap.add_argument("--chromhmm", required=True)
ap.add_argument("--atac", required=True); ap.add_argument("--edges", required=True); ap.add_argument("--tf-peaks", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
EXCL = {"SERPINH1", "DNAJB1", "BRD4", "RPS3"}
tss = []
with gzip.open(a.gtf, "rt") as g:
    for ln in g:
        if ln.startswith("#"): continue
        x = ln.split("\t", 9)
        if x[2] != "gene" or not re.search(r'gene_type "(protein_coding|lncRNA)"', x[8]): continue
        nm = re.search(r'gene_name "([^"]+)"', x[8]).group(1)
        tss.append((x[0], int(x[3]) if x[6] == "+" else int(x[4]), nm))
TSS = pd.DataFrame(tss, columns=["chrom", "pos", "gene"])
tss_by = {c: d.sort_values("pos") for c, d in TSS.groupby("chrom")}
def near_tss(c, s, e, pad=2500):
    d = tss_by.get(c)
    if d is None: return []
    m = (d.pos >= s - pad) & (d.pos <= e + pad); return sorted(set(d.gene[m]))
P = pd.read_csv(a.peaks, sep="\t", header=None, usecols=[0, 1, 2, 6]); P.columns = ["chrom", "s", "e", "sig"]
P = P[P.chrom.str.match(r"^chr[0-9XY]+$")]
def inside_tss(r):
    d = tss_by.get(r.chrom)
    if d is None: return False
    m = (d.pos - 2500 <= r.s) & (d.pos + 2500 >= r.e); return bool(m.any())
P["prom"] = [inside_tss(r) for r in P.itertuples()]
Q = P[~P.prom].sort_values(["chrom", "s"]); rows = []
for c, d in Q.groupby("chrom"):
    cs, ce, sc = None, None, 0.0
    for r in d.itertuples():
        if cs is not None and r.s - ce <= 12500: ce = max(ce, r.e); sc += r.sig * (r.e - r.s)
        else:
            if cs is not None: rows.append((c, cs, ce, sc))
            cs, ce, sc = r.s, r.e, r.sig * (r.e - r.s)
    rows.append((c, cs, ce, sc))
S = pd.DataFrame(rows, columns=["chrom", "start", "end", "score"]).sort_values("score").reset_index(drop=True)
y = S.score.values / S.score.max() * len(S); x = np.arange(len(S))
cut = int(np.argmin(np.abs(np.gradient(y) - 1)[: len(S)] + (np.gradient(y) < 1) * 1e9)) if (np.gradient(y) >= 1).any() else len(S) - 1
SE = S.iloc[cut:].copy().sort_values("score", ascending=False).reset_index(drop=True); SE["rank"] = np.arange(1, len(SE) + 1)
SE["se_id"] = [f"HEK293_SE_{i:04d}" for i in SE["rank"]]
# cross-checks
def load_bed(f, gz=True, cols=3):
    op = gzip.open if str(f).endswith(".gz") else open
    with op(f, "rt") as h:
        return pd.read_csv(h, sep="\t", header=None, comment="#", usecols=range(cols))
CH = load_bed(a.chromhmm, cols=4); CH.columns = ["chrom", "s", "e", "state"]
enh = CH[CH.state.astype(str).str.contains("Enh", case=False)]
AT = load_bed(a.atac); AT.columns = ["chrom", "s", "e"]
def ov_bp(B, c, s, e):
    d = B[(B.chrom == c) & (B.e > s) & (B.s < e)]; return int((np.minimum(d.e, e) - np.maximum(d.s, s)).sum())
SE["enh_frac"] = [ov_bp(enh, r.chrom, r.start, r.end) / (r.end - r.start) for r in SE.itertuples()]
SE["atac_bp"] = [ov_bp(AT, r.chrom, r.start, r.end) for r in SE.itertuples()]
SE.to_csv(O / "HEK293_SE_rose_hg38.tsv", sep="\t", index=False)
print(f"H3K27ac peaks {len(P)} ({P.prom.sum()} promoter-contained excluded); stitched regions {len(S)}; SEs {len(SE)}; "
      f"median SE {int((SE.end - SE.start).median())} bp; ChromHMM enhancer fraction median {SE.enh_frac.median():.2f}")
# TF summits per SE
tfs_in = {i: set() for i in SE.index}
se_by = {c: d for c, d in SE.groupby("chrom")}
for f in sorted(glob.glob(os.path.join(a.tf_peaks, "TF.*_peaks.q005.narrowPeak"))):
    tf = re.sub(r"^TF\.|_peaks\.q005\.narrowPeak$", "", os.path.basename(f))
    for ln in open(f):
        x = ln.split("\t"); d = se_by.get(x[0])
        if d is None: continue
        p = int(x[1]) + int(x[9]); m = d.index[(d.start <= p) & (d.end > p)]
        for i in m: tfs_in[i].add(tf)
SE["n_tfs"] = [len(tfs_in[i]) for i in SE.index]; SE["tfs"] = [",".join(sorted(tfs_in[i])) for i in SE.index]
SE.to_csv(O / "HEK293_SE_rose_hg38.tsv", sep="\t", index=False)
se_by = {c: d for c, d in SE.groupby("chrom")}
# loop-edges with an SE end and a promoter end
E = pd.read_csv(a.edges, sep="\t"); E = E[(E["on_loop_0.7"] == 1) & (E.distance_kb > 25)]
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return c, s, e
out = []
for r in E.itertuples():
    for se_end, pr_end in ((r.region1, r.region2), (r.region2, r.region1)):
        c, s, e = parse(se_end); d = se_by.get(c)
        if d is None: continue
        hit = d[(d.end > s) & (d.start < e)]
        if not len(hit): continue
        pc, ps, pe = parse(pr_end); genes = near_tss(pc, ps, pe)
        if not genes or set(genes) & EXCL: continue
        if set(near_tss(c, s, e)) & EXCL: continue
        h = hit.iloc[0]
        out.append(dict(se_id=h.se_id, se=f"{c}:{h.start}-{h.end}", se_rank=h["rank"], se_n_tfs=h.n_tfs, se_enh_frac=h.enh_frac,
                        se_end=se_end, promoter_end=pr_end, promoter_genes=",".join(genes), kb=r.distance_kb, edge_n_tfs=r.n_tfs, edge_tfs=r.tfs,
                        se_end_promoter=",".join(near_tss(c, s, e))))
C = pd.DataFrame(out).drop_duplicates(["se_id", "se_end", "promoter_end"])
C.to_csv(O / "se_loop_edges.tsv", sep="\t", index=False)
G = C.groupby(["se_id", "se", "se_rank", "se_n_tfs", "se_enh_frac"]).agg(n_loop_edges=("promoter_end", "size"),
        genes=("promoter_genes", lambda v: ",".join(sorted({g for x in v for g in x.split(",")}))),
        n_genes=("promoter_genes", lambda v: len({g for x in v for g in x.split(",")})), max_edge_tfs=("edge_n_tfs", "max"),
        se_end_promoter=("se_end_promoter", lambda v: ",".join(sorted({g for x in v for g in x.split(",") if g})))).reset_index()
G = G.sort_values(["n_loop_edges", "se_n_tfs", "n_genes"], ascending=False)
G.to_csv(O / "se_hub_candidates.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(G.head(12).to_string(index=False))
