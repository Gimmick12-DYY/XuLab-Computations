"""SE-hub step 1b: loop-supported anchor hubs (ourrun_k50_anchor_hubs_loop_edges: core shared by >=5 TFs via Cicero edges on a
WT Peakachu 0.7 loop, +-30 kb) whose core or partner overlaps a HEK293 ROSE super-enhancer (01_se_candidates.py) at a DISTAL
position (no protein-coding/lncRNA TSS within 2.5 kb). Each hub-SE combination is scored by: loop-edges between the SE and
promoters of the hub, TFs with a summit in the SE, distinct linked promoter genes, SE ChromHMM enhancer fraction.
Excluded: SERPINH1, DNAJB1, BRD4, RPS3."""
import argparse, gzip, re
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--hubdir", required=True); ap.add_argument("--se", required=True); ap.add_argument("--gtf", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args(); O = Path(a.out)
EXCL = {"SERPINH1", "DNAJB1", "BRD4", "RPS3"}
tss = []
with gzip.open(a.gtf, "rt") as g:
    for ln in g:
        if ln.startswith("#"): continue
        x = ln.split("\t", 9)
        if x[2] != "gene" or not re.search(r'gene_type "(protein_coding|lncRNA)"', x[8]): continue
        tss.append((x[0], int(x[3]) if x[6] == "+" else int(x[4]), re.search(r'gene_name "([^"]+)"', x[8]).group(1)))
TSS = pd.DataFrame(tss, columns=["chrom", "pos", "gene"]); tb = {c: d for c, d in TSS.groupby("chrom")}
def genes_at(c, s, e, pad=2500):
    d = tb.get(c); return [] if d is None else sorted(set(d.gene[(d.pos >= s - pad) & (d.pos <= e + pad)]))
def parse(r): c, s = r.split(":"); s, e = map(int, s.split("-")); return c, s, e
SE = pd.read_csv(a.se, sep="\t"); sb = {c: d for c, d in SE.groupby("chrom")}
def se_hit(r):
    c, s, e = parse(r); d = sb.get(c)
    if d is None: return None
    h = d[(d.end > s) & (d.start < e)]; return None if not len(h) else h.iloc[0]
HP = pd.read_csv(Path(a.hubdir) / "anchor_hub_partners.tsv", sep="\t"); HN = pd.read_csv(Path(a.hubdir) / "anchor_hubs_named.tsv", sep="\t")
rows = []
for core, g in HP.groupby("core"):
    regs = [core] + list(g.partner)
    for r in regs:
        h = se_hit(r)
        if h is None: continue
        c, s, e = parse(r)
        if genes_at(c, s, e): continue  # SE anchor must be distal
        # loop-edges from this SE anchor: core<->partner pairs that include it
        if r == core: links = g; prom = [genes_at(*parse(p)) for p in links.partner]
        else: links = g[g.partner == r]; prom = [genes_at(*parse(core))]
        genes = sorted({x for p in prom for x in p})
        if not genes or set(genes) & EXCL: continue
        hn = HN[HN.core == core].iloc[0]
        rows.append(dict(core=core, core_gene=hn.gene_symbol if isinstance(hn.gene_symbol, str) else "", core_n_tfs=hn.n_tfs, n_hub_partners=hn.n_partners,
                         se_anchor=r, anchor_is_core=r == core, se_id=h.se_id, se=f"{h.chrom}:{h.start}-{h.end}", se_rank=h["rank"], se_kb=(h.end - h.start) / 1000,
                         se_n_tfs=h.n_tfs, se_enh_frac=round(h.enh_frac, 2), n_se_loop_edges=int(sum(bool(p) for p in prom)), linked_genes=",".join(genes),
                         max_tfs_linking=int(links.n_tfs_linking.max())))
C = pd.DataFrame(rows).drop_duplicates(["core", "se_anchor"]).sort_values(["n_se_loop_edges", "se_n_tfs", "max_tfs_linking"], ascending=False)
C.to_csv(O / "se_hub_candidates_anchorhubs.tsv", sep="\t", index=False)
pd.set_option("display.width", 260); print(len(C)); print(C.head(15).to_string(index=False))
