# m17_fig5_panels: plain-text dump of the gene-example numbers for a redraw (HSPB1, PGAM1). Illustration on an already-read benchmark.
# usage: m17_gene_text.py GENE ... -> stdout and out/gene_example/text_<gene>.tsv
import os, subprocess, sys
import numpy as np, pandas as pd
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"; D_ = f"{RG_}/m17_fig5_panels/out/gene_example"; GTF = "/work/users/d/y/dyy12/XuLab/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
GENES = sys.argv[1:]; sys.argv = ["m15.py", "dc"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); _keep = D_; exec(src[:src.index("# ---- matched-gene strata")]); D_ = _keep
GT = pd.read_csv(f"{D_}/p4_gene_tf.tsv", sep="\t"); E = pd.read_csv(f"{D_}/p4_elements.tsv", sep="\t").rename(columns={"ABC.Score": "abc"}); L = pd.read_csv(f"{D_}/p4_links.tsv", sep="\t").rename(columns={"ABC.Score": "abc"})
EL = pd.read_csv(f"{RG_}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); ek = {f"{c}:{s}-{e}": i for i, (c, s, e) in enumerate(zip(EL.chr, EL.start, EL.end))}
SITES = {}
for f in os.listdir(f"{RG_}/m10_promoter_regulome/work/g"):
    if f.startswith("sites_") and f.endswith(".tsv"): SITES[f[6:-4]] = set(pd.read_csv(f"{RG_}/m10_promoter_regulome/work/g/{f}", sep="\t", header=None)[0].astype(int))
ugi = {g: k for k, g in enumerate(UG)}; ti = {t: k for k, t in enumerate(TF)}
for g in GENES:
    out = []
    rows = subprocess.run(f"zcat {GTF} | grep -w 'gene_name \"{g}\";'", shell=True, capture_output=True, text=True).stdout.strip().splitlines(); rows = [r.split("\t") for r in rows]
    Lg = L[L.TargetGene == g]; tss = int(Lg.TargetGeneTSS.iloc[0]); T = [r for r in rows if r[2] == "transcript"]; tid = lambda r: r[8].split('transcript_id "')[1].split('"')[0]
    b = min(T, key=lambda r: abs((int(r[3]) if r[6] == "+" else int(r[4])) - tss)); ex = sorted((int(r[3]), int(r[4])) for r in rows if r[2] == "exon" and tid(r) == tid(b))
    out.append(f"## {g} locus"); out.append("chr\tstrand\tTSS_ABC\tgene_start\tgene_end\ttranscript\texons"); out.append(f"{b[0]}\t{b[6]}\t{tss}\t{b[3]}\t{b[4]}\t{tid(b)}\t" + ";".join(f"{a}-{c}" for a, c in ex))
    out.append(f"## {g} ABC-linked elements (canonical, ABC >= 0.02)"); out.append("start\tend\ttype\tABC")
    for r in Lg.sort_values("start").itertuples(): out.append(f"{r.start}\t{r.end}\t{'promoter' if r.self else 'distal'}\t{r.abc:.4f}")
    out.append(f"## {g} TFs with a linked q005 peak"); out.append("TF\telement\ttype\tpeak_signalValue\tABC\tpct_raw_promoter_reads\tpct_peak_score\tz_all\tz_half1\tz_half2\tconfirmed_K10000\tconfirmed_K2000\tstrong_phenotype\town_motif_at_element\thas_knockdown")
    Eg = E[E.gene == g].sort_values("link_score", ascending=False).drop_duplicates("tf"); H = GT[GT.gene == g].set_index("tf")
    for r in Eg.sort_values("peak_signalValue", ascending=False).itertuples():
        h = H.loc[r.tf]; el = f"{r.chr}:{r.element_start}-{r.element_end}"; e = ek.get(el); mot = ("yes" if e in SITES[r.tf] else "no") if r.tf in SITES else "no motif"
        pb = PCT["b_raw_promoter_reads"][ti[r.tf], ugi[g]] if (r.tf in ti and g in ugi) else np.nan
        f = lambda v: "NA" if not np.isfinite(v) else f"{v:.2f}"
        out.append(f"{r.tf}\t{r.element_start}-{r.element_end}\t{'promoter' if r.self else 'distal'}\t{r.peak_signalValue:.2f}\t{r.abc:.4f}\t{f(pb)}\t{h.binding_pct:.2f}\t{f(h.z_all)}\t{f(h.z_half1)}\t{f(h.z_half2)}\t{'yes' if h.confirmed_K10000 else 'no'}\t{'yes' if h.confirmed_K2000 else 'no'}\t{'yes' if h.strong_phenotype else 'no'}\t{mot}\t{'yes' if h.has_knockdown else 'no'}")
    out.append(f"## {g} z under every knockdown (m15 double-centred, Arm 2, all cells), ranked by |z|"); out.append("TF\tz_all\tz_half1\tz_half2\tq005_peak_linked\tconfirmed_K10000")
    j = ugi[g]; o = np.argsort(-np.nan_to_num(np.abs(ZA[:, j]), nan=-1))
    for i in o:
        if np.isfinite(ZA[i, j]): out.append(f"{TF[i]}\t{ZA[i, j]:.2f}\t{Z1[i, j]:.2f}\t{Z2[i, j]:.2f}\t{'yes' if TF[i] in set(Eg.tf) else 'no'}\t{'yes' if bool(H.confirmed_K10000.get(TF[i], False)) else 'no'}")
    txt = "\n".join(out); open(f"{D_}/text_{g}.tsv", "w").write(txt + "\n"); print(txt)
