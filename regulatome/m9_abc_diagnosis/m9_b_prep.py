# m9_abc_diagnosis section B preparation (diagnostic, benchmark already read). No HITAG binding is used in section B.
# usage: m9_b_prep.py main        -> work/promoters.{bed,fa}, work/motifs/<TF>.meme, B_motif_workbook.tsv, work/scores_bulk.npz, B_sources_bulk.tsv
#        m9_b_prep.py chromnitron -> work/scores_chromnitron.npz, B_sources_chromnitron.tsv
# promoters = TSS +/- 500 bp (canonical ABC GeneList tss) for the genes of the Arm 1 or Arm 2 universe.
# Each testable TF's own HOCOMOCO v11 full motif (best grade, then lowest index) plus 20 column-permuted copies: a PWM has no dinucleotide order
# to preserve, so the shuffle permutes motif columns (keeps each column's base composition and the motif's information content).
# Bulk ChIP (downstream/tracks/bulk_chipseq) and Chromnitron (/users/x/u/xujie/HEK293T_SA, read only): mean signal over each promoter.
import os, subprocess, sys
import numpy as np, pandas as pd, pyBigWig
MODE = sys.argv[1]
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m9_abc_diagnosis"; W = f"{O}/work"; os.makedirs(f"{W}/motifs", exist_ok=True)
LAB = "diagnostic, benchmark already read"; BEDTOOLS = "/nas/longleaf/rhel9/apps/bedtools/2.31.1/bedtools2/bin/bedtools"; FA = f"{X}/downstream/cache/hg38.fa"
GLf = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/GeneList.txt", sep="\t", usecols=["chr", "name", "strand", "tss"]).drop_duplicates("name", keep="last").set_index("name")
U = sorted(set(np.load(f"{W}/responses_arm2.npz", allow_pickle=True)["genes"]) | set(np.load(f"{W}/responses_arm1.npz", allow_pickle=True)["genes"]))
P = GLf.reindex(U); P = P[P.tss.notna()].copy(); P["s"] = (P.tss.astype(int) - 500).clip(lower=0); P["e"] = P.tss.astype(int) + 500
def bw_means(path):
    b = pyBigWig.open(path); ch = b.chroms(); out = np.full(len(P), np.nan)
    for i, (c, s, e) in enumerate(zip(P.chr.values, P.s.values, P.e.values)):
        if c in ch: v = b.stats(c, int(s), int(min(e, ch[c])), type="mean")[0]; out[i] = 0.0 if v is None else v
    b.close(); return out
if MODE == "main":
    P[["chr", "s", "e"]].assign(name=P.index, score=0, strand=P.strand)[["chr", "s", "e", "name", "score", "strand"]].to_csv(f"{W}/promoters.bed", sep="\t", header=False, index=False)
    subprocess.run(f"{BEDTOOLS} getfasta -fi {FA} -bed {W}/promoters.bed -nameOnly -fo {W}/promoters.fa", shell=True, check=True); print("promoters", len(P), flush=True)
    M = open(f"{X}/downstream/cache/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme").read(); head = M.split("MOTIF")[0]; blocks = {b.split()[0]: "MOTIF" + b for b in M.split("MOTIF")[1:]}
    UNIPROT = {"ZNF143": "ZN143", "ZNF282": "ZN282", "ZNF768": "ZN768", "ZBTB7A": "ZBT7A", "HOXB6": "HXB6", "HOXB8": "HXB8", "HOXA5": "HXA5", "HOXA10": "HXA10", "HOXC9": "HXC9", "RBPJ": "SUH", "RELA": "TF65", "SREBF2": "SRBP2"}  # HOCOMOCO v11 uses UniProt entry names
    F = pd.read_csv(f"{RG}/m7_step2/splits/tf_folds.tsv", sep="\t"); rows = []; rng = np.random.default_rng(20261111)
    for t in F.TF:
        cand = [m for m in blocks if m.startswith(f"{UNIPROT.get(t, t)}_HUMAN.H11MO.")]
        if not cand: rows.append(dict(tf=t, hocomoco_v11_motif="none", n_motifs_for_tf=0)); continue
        m = sorted(cand, key=lambda x: (x.split(".")[-1], int(x.split(".")[-2])))[0]; lines = [l for l in blocks[m].strip().split("\n") if l.strip() and not l.startswith("URL")]
        mat = np.array([[float(v) for v in l.split()] for l in lines[2:]]); out = [head.strip(), "", "\n".join(lines), ""]
        for k in range(20):
            pm = mat[rng.permutation(len(mat))]; out += [f"MOTIF {m}_shuf{k:02d}", lines[1], *["\t".join(f"{v:.6f}" for v in r) for r in pm], ""]
        open(f"{W}/motifs/{t}.meme", "w").write("\n".join(out) + "\n"); rows.append(dict(tf=t, hocomoco_v11_motif=m, n_motifs_for_tf=len(cand), width=len(mat)))
    MS = pd.DataFrame(rows); WB = pd.read_excel(f"{X}/motif_analysis/TF1000cells_motifs.xlsx", sheet_name="TF1000cells_motifs").set_index("TF")
    MS = MS.join(WB, on="tf"); MS.insert(0, "label", LAB); MS.to_csv(f"{O}/B_motif_workbook.tsv", sep="\t", index=False); print(MS[["tf", "hocomoco_v11_motif"]].to_string(index=False), flush=True)
    BK = pd.read_csv(f"{X}/downstream/tracks/bulk_chipseq/manifest.tsv", sep="\t"); bulk = {}; meta = []
    for r in BK.itertuples():
        if os.path.exists(r.bigwig): bulk[r.tf] = bw_means(r.bigwig); src, fl = "mean CPM bigWig over promoter", r.bigwig
        else:
            pk = pd.read_csv(r.peaks, sep="\t", header=None, usecols=[0, 1, 2, 6]); pk.columns = ["c", "s", "e", "sig"]; v = np.zeros(len(P))
            for c, H in pk.groupby("c"):
                for i in np.flatnonzero(P.chr.values == c):
                    m = (H.s.values < P.e.values[i]) & (H.e.values > P.s.values[i])
                    if m.any(): v[i] = H.sig.values[m].max()
            bulk[r.tf] = v; src, fl = "max peak signalValue over promoter (no bigWig)", r.peaks
        meta.append(dict(label=LAB, source="bulk_chip", tf=r.tf, file=fl, cell_line="K562" if "K562" in str(r.bam) else "HEK293", score=src))
    c293 = f"{X}/downstream/tracks/bulk_chipseq/CTCF_293Tcon1_pval.signal.bigwig"
    if os.path.exists(c293): bulk["CTCF_293T"] = bw_means(c293); meta.append(dict(label=LAB, source="bulk_chip", tf="CTCF_293T", file=c293, cell_line="HEK293T", score="mean p-value signal bigWig over promoter"))
    np.savez(f"{W}/scores_bulk.npz", genes=np.array(P.index), **bulk); pd.DataFrame(meta).to_csv(f"{O}/B_sources_bulk.tsv", sep="\t", index=False); print("bulk", list(bulk), flush=True)
else:
    CN = "/users/x/u/xujie/HEK293T_SA"; chrom = {}; meta = []
    for t in sorted(os.listdir(CN)):
        f = f"{CN}/{t}/processed/data.bigwig"
        if not os.path.exists(f): continue
        try: chrom[t] = bw_means(f); meta.append(dict(label=LAB, source="chromnitron", tf=t, file=f, cell_line="HEK293T (prediction)", score="mean bigWig over promoter"))
        except Exception as ex: meta.append(dict(label=LAB, source="chromnitron", tf=t, file=f, cell_line="", score=f"unreadable: {ex}"))
        print("chromnitron", t, flush=True)
    np.savez(f"{W}/scores_chromnitron.npz", genes=np.array(P.index), **chrom); pd.DataFrame(meta).to_csv(f"{O}/B_sources_chromnitron.tsv", sep="\t", index=False)
