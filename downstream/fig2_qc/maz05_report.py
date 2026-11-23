"""MAZ reference audit, step 5: one readable workbook (maz_ref_audit.xlsx) from the maz02 and maz04 tables."""
from pathlib import Path
import pandas as pd

OUT = Path("/work/users/d/y/dyy12/XuLab/results_293T_analysis/maz_ref_audit")
M = "MAZ_HUMAN.H11MO.0.A"
S = pd.read_csv(OUT / "maz_crossval_summary.tsv", sep="\t")
cols = {"dataset": "dataset", "cell_line": "cell line", "assay": "assay / construct", "n_peaks": "peaks (after blacklist)",
        "ref_top1000_recall_200bp": "recall of HEK293 ref top-1000 (200 bp)", "frac_top_overlapping_ref": "own top-1000 hitting ref peaks",
        "jaccard_bp_vs_ref": "bp Jaccard vs ref", "CpG_promoter_frac_top": "CpG-rich promoter frac (top 1000)",
        f"{M}:central_frac": "MAZ motif within 50 bp (top 500)", f"{M}:central_frac_shuf": "same, dinuc-shuffled",
        f"{M}:enrich_vs_shuf": "MAZ motif enrichment vs shuffled", f"{M}:centrality": "MAZ motif centrality (±50 vs 150-250 bp)",
        "distal:" + f"{M}:enrich_vs_shuf": "MAZ motif enrichment, distal peaks only",
        "SP1_HUMAN.H11MO.0.A:enrich_vs_shuf": "SP1 (GC-box control) enrichment", "CTCF_HUMAN.H11MO.0.A:enrich_vs_shuf": "CTCF motif enrichment",
        "HAP1_log2_parental_over_MAZKO_median": "HAP1 MAZ ChIP log2 parental/MAZ-KO (median, top 1000)",
        "HAP1_frac_log2ratio_gt1": "frac of top 1000 losing >2x in MAZ-KO", "own_centre_flank": "own signal centre/flank"}
T = S[list(cols)].rename(columns=cols).round(3)
with pd.ExcelWriter(OUT / "maz_ref_audit.xlsx") as xw:
    pd.read_csv(OUT / "encode_maz_qc_summary.tsv", sep="\t").to_excel(xw, "ENCODE_MAZ_QC", index=False)
    T.to_excel(xw, "cross_validation", index=False)
    pd.read_csv(OUT / "ref_split_by_sharing.tsv", sep="\t").round(3).to_excel(xw, "HEK293_ref_by_sharing", index=False)
    pd.read_csv(OUT / "maz_pairwise_jaccard.tsv", sep="\t", index_col=0).round(3).to_excel(xw, "pairwise_jaccard")
    pd.read_csv(OUT / "maz_union_signal_spearman.tsv", sep="\t", index_col=0).round(3).to_excel(xw, "signal_spearman")
    pd.DataFrame({"note": [
        "Reference under test: ENCODE ENCSR290SSQ (HEK293, exogenous eGFP-MAZ, Flp-In, CMV), IDR peaks ENCFF529XTF. Our cells are HEK293T.",
        "Non-HEK sets (K562, GM12878, HepG2, MCF-7, IMR-90, A549, HeLa-S3, HAP1) are used only to judge the HEK293 reference, not as references.",
        "Motif: HOCOMOCO v11 MAZ_HUMAN.H11MO.0.A, FIMO p<1e-4, summit ±250 bp, background = same sequences dinucleotide-shuffled x3.",
        "MAZ-KO test: GSE215081 HAP1 anti-MAZ ChIP (ChIP-Atlas RPM bigWigs), parental rep1+2 vs MAZ-KO rep1+2, mean ±250 bp at each set's top-1000 summits. "
        "Backgrounds: 1000 CpG-rich promoters with no MAZ peak in any set (median 0.53, 21% >2x) and 1000 CTCF sites (0.56, 6%).",
        "CpG-rich promoter = within 1 kb of a GENCODE v46 basic TSS, ±500 bp GC>0.5 and CpG obs/exp>0.6 (same rule as r16).",
        "Scripts: XuLab downstream/fig2_qc/maz01-maz05."]}).to_excel(xw, "methods", index=False)
print("ok")
