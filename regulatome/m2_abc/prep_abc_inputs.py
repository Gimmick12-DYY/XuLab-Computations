# ABC inputs: protein-coding gene bounds (GENCODE v46, ABC CollapsedGeneBounds format), active-class candidate elements
# from elements_v2.bed, and the chrom-sizes bed; all sorted in the ABC chrom-sizes order
import gzip, re, sys
import pandas as pd
gtf, elements, chrom_sizes, out = sys.argv[1:5]
cs = pd.read_csv(chrom_sizes, sep="\t", header=None, names=["chr", "size"])
keep = [c for c in cs.chr if re.fullmatch(r"chr([0-9]+|X|Y)", c)]
order = {c: i for i, c in enumerate(keep)}
cs[cs.chr.isin(keep)].assign(s=0)[["chr", "s", "size"]].to_csv(f"{out}/chrom_sizes.bed", sep="\t", header=False, index=False)
cs[cs.chr.isin(keep)].to_csv(f"{out}/chrom_sizes.tsv", sep="\t", header=False, index=False)
rows = []
for line in gzip.open(gtf, "rt"):
    if line[0] == "#": continue
    c = line.split("\t")
    if c[2] != "gene" or c[0] not in order: continue
    if 'gene_type "protein_coding"' not in c[8]: continue
    name = re.search(r'gene_name "([^"]+)"', c[8]).group(1); gid = re.search(r'gene_id "([^".]+)', c[8]).group(1)
    rows.append((c[0], int(c[3]) - 1, int(c[4]), name, 0, c[6], gid, "protein_coding"))
G = pd.DataFrame(rows, columns=["chr", "start", "end", "name", "score", "strand", "Ensembl_ID", "gene_type"])
G = G.drop_duplicates("name", keep="first")  # ABC keys genes by name; drop the few PAR/duplicate symbols
G = G.sort_values(["chr", "start"], key=lambda s: s.map(order) if s.name == "chr" else s)
G.to_csv(f"{out}/genes_pc_v46.bed", sep="\t", header=False, index=False)
E = pd.read_csv(elements, sep="\t", header=None, names=["chr", "start", "end", "name", "type", "parent_id", "parent_type", "class"])
E = E[E["class"].isin(["promoter", "distal_active", "bivalent"]) & E.chr.isin(order)]
E = E.sort_values(["chr", "start"], key=lambda s: s.map(order) if s.name == "chr" else s)
E[["chr", "start", "end", "name"]].to_csv(f"{out}/candidates_active.bed", sep="\t", header=False, index=False)
E[["name", "class"]].to_csv(f"{out}/candidate_class.tsv", sep="\t", index=False)
print("genes", len(G), "candidates", len(E), E["class"].value_counts().to_dict())
