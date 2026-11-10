# Sanity check of the realigned H3K27ac: (1) Spearman of our counts per candidate element (ABC Neighborhoods) vs the ChIP-Atlas
# bigWig mean signal for SRX18002015/16; (2) fraction of our reads in GSE246314 untreated H3K27ac peaks (union of 3 reps) and at TSS +-1 kb
import sys, gzip, subprocess
import numpy as np, pandas as pd, pyBigWig
from scipy.stats import spearmanr
B, bw_dir = sys.argv[1:3]
X = "/work/users/d/y/dyy12/XuLab"
E = pd.read_csv(f"{B}/results/Neighborhoods/EnhancerList.txt", sep="\t")
kc = [c for c in E.columns if c.startswith("H3K27ac.") and c.endswith(".readCount")][0]
ac = [c for c in E.columns if c.startswith("ATAC.") and c.endswith(".readCount")][0]
out = []
for s in ["SRX18002015", "SRX18002016"]:
    bw = pyBigWig.open(f"{bw_dir}/{s}.bw")
    v = np.array([(bw.stats(r.chr, int(r.start), int(r.end), type="mean")[0] or 0.0) if r.chr in bw.chroms() else np.nan
                  for r in E.itertuples()])
    out.append((f"ours vs ChIP-Atlas {s}", spearmanr(E[kc], v, nan_policy="omit")[0]))
    out.append((f"ATAC vs ChIP-Atlas {s}", spearmanr(E[ac], v, nan_policy="omit")[0]))
out.append(("ours H3K27ac vs ATAC counts", spearmanr(E[kc], E[ac])[0]))
for c in ["isPromoterElement"]:
    out.append(("share of element H3K27ac reads in promoter elements", E.loc[E[c], kc].sum() / E[kc].sum()))
    out.append(("share of element ATAC reads in promoter elements", E.loc[E[c], ac].sum() / E[ac].sum()))
tag = f"{B}/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz"
pk = f"{B}/tmp/gse246314_union.bed"
subprocess.run(f"zcat {X}/results_3d_model/se_hub_293T_h3k27ac/raw/GSM78666*.narrowPeak.gz | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge > {pk}", shell=True, check=True)
tot = int(subprocess.run(f"zcat {tag} | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
inp = int(subprocess.run(f"bedtools intersect -u -a {tag} -b {pk} | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
tss = f"{B}/results/Neighborhoods/GeneList.TSS1kb.bed"
intss = int(subprocess.run(f"bedtools intersect -u -a {tag} -b {tss} | wc -l", shell=True, capture_output=True, text=True, check=True).stdout)
pk_bp = sum(int(l.split()[2]) - int(l.split()[1]) for l in open(pk))
out += [("tagAlign reads", tot), ("FRiP in GSE246314 union peaks", inp / tot), ("GSE246314 union peak bp", pk_bp),
        ("fraction of reads in protein-coding TSS +-1 kb", intss / tot)]
pd.DataFrame(out, columns=["metric", "value"]).to_csv(f"{B}/results/k27ac_sanity.tsv", sep="\t", index=False)
print(pd.DataFrame(out, columns=["metric", "value"]).to_string(index=False))
