# M0 shared universe: genes/TSS, merged protein-coding exons, regulatory elements (hg38)
import gzip, io, re, subprocess, sys
import pandas as pd

gtf, atac, chromhmm, blacklist, out = sys.argv[1:6]
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
ACTIVE = ["TssA", "TssFlnk", "TssFlnkU", "TssFlnkD", "EnhA1", "EnhA2", "EnhG1", "EnhG2", "EnhWk"]
BIV = ["TssBiv", "EnhBiv"]

def attr(s, k):
    m = re.search(k + r' "([^"]+)"', s)
    return m.group(1) if m else None

def bt(*args, inp=None):
    return subprocess.run(["bedtools", *args], input=inp, capture_output=True, text=True, check=True).stdout

def read_bed(text):
    return pd.read_csv(io.StringIO(text), sep="\t", header=None) if text.strip() else pd.DataFrame(columns=[0, 1, 2])

# ---- genes and exons ---------------------------------------------------------
tx, exons = [], []
with gzip.open(gtf, "rt") as f:
    for line in f:
        if line[0] == "#":
            continue
        c = line.rstrip("\n").split("\t")
        if c[0] not in CHROMS:
            continue
        a = c[8]
        gtype = attr(a, "gene_type")
        if gtype not in ("protein_coding", "lncRNA"):
            continue
        if c[2] == "transcript":
            tx.append((attr(a, "gene_id"), attr(a, "gene_name"), gtype, c[0], int(c[3]), int(c[4]), c[6], "Ensembl_canonical" in a))
        elif c[2] == "exon" and gtype == "protein_coding":
            exons.append((c[0], int(c[3]) - 1, int(c[4])))
tx = pd.DataFrame(tx, columns=["gene_id", "symbol", "type", "chrom", "start", "end", "strand", "canonical"])
tx["tss"] = tx.start.where(tx.strand == "+", tx.end)
rows = []
for gid, d in tx.groupby("gene_id", sort=False):
    can = d[d.canonical]
    if len(can):
        r, src = can.iloc[0], "Ensembl_canonical"
    else:
        r = d.loc[d.tss.idxmin()] if d.strand.iloc[0] == "+" else d.loc[d.tss.idxmax()]
        src = "most_upstream_basic"
    rows.append((gid, r.symbol, r.type, r.chrom, int(r.tss), r.strand, src))
genes = pd.DataFrame(rows, columns=["gene_id", "symbol", "type", "chrom", "tss", "strand", "tss_source"])
genes.to_csv(f"{out}/genes.tsv", sep="\t", index=False)
ex = pd.DataFrame(exons).drop_duplicates().sort_values([0, 1]).to_csv(sep="\t", header=False, index=False)
open(f"{out}/exons.bed", "w").write(bt("merge", "-i", "stdin", inp=ex))
t = genes.assign(s=(genes.tss - 1 - 1000).clip(lower=0), e=genes.tss + 1000)[["chrom", "s", "e", "symbol"]]
open(f"{out}/tss_1kb.bed", "w").write(t.sort_values(["chrom", "s"]).to_csv(sep="\t", header=False, index=False))

# ---- elements -----------------------------------------------------------------
def build(states, name):
    ch = pd.read_csv(chromhmm, sep="\t", header=None, usecols=[0, 1, 2, 3])
    ch = ch[ch[3].isin(states) & ch[0].isin(CHROMS)][[0, 1, 2]]
    at = pd.read_csv(atac, sep="\t", header=None, usecols=[0, 1, 2])
    at = at[at[0].isin(CHROMS)]
    u = pd.concat([at, ch]).sort_values([0, 1]).to_csv(sep="\t", header=False, index=False)
    merged = bt("merge", "-d", "200", "-i", "stdin", inp=u)
    nobl = bt("intersect", "-v", "-a", "stdin", "-b", blacklist, inp=merged)
    prom = bt("intersect", "-u", "-a", "stdin", "-b", f"{out}/tss_1kb.bed", inp=nobl)
    dist = bt("intersect", "-v", "-a", "stdin", "-b", f"{out}/tss_1kb.bed", inp=nobl)
    dist_masked = bt("subtract", "-a", "stdin", "-b", f"{out}/exons.bed", inp=dist)
    P = read_bed(prom).iloc[:, :3].assign(type="promoter")
    D = read_bed(dist_masked).iloc[:, :3].assign(type="distal")
    D = D[D[2] - D[1] >= 50]
    E = pd.concat([P, D]).sort_values([0, 1]).reset_index(drop=True)
    E.columns = ["chrom", "start", "end", "type"]
    E.insert(3, "element_id", [f"E{i:06d}" for i in range(1, len(E) + 1)])
    E.to_csv(f"{out}/{name}.bed", sep="\t", header=False, index=False)
    w = E.end - E.start
    isP = E.type == "promoter"
    print(f"{name}: merged {len(merged.splitlines())}, after blacklist {len(nobl.splitlines())}, distal before exon mask {len(dist.splitlines())}; "
          f"elements {len(E)} (promoter {isP.sum()}, distal {(~isP).sum()}); median width {int(w.median())} "
          f"(promoter {int(w[isP].median())}, distal {int(w[~isP].median())}); total {w.sum()/1e6:.1f} Mb; max width {w.max()}")

print("genes:", genes.type.value_counts().to_dict(), "tss_source:", genes.tss_source.value_counts().to_dict())
print("merged protein-coding exon intervals:", len(open(f"{out}/exons.bed").read().splitlines()))
build(ACTIVE, "elements")
build(ACTIVE + BIV, "elements_with_bivalent")
