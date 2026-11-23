"""MAZ reference audit, step 2: one QC row per ENCODE human MAZ TF ChIP-seq experiment (default ENCODE4 GRCh38 analysis).
Reads maz01 outputs; writes encode_maz_qc_summary.tsv.
Usable fragments = filtered mapped reads, halved for paired-end runs (ENCODE convention). ENCODE TF standard: >=10M usable
fragments per replicate (20M recommended), NSC>1.05, RSC>0.8, NRF>=0.8, PBC1>=0.8, rescue and self-consistency ratio <2.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/work/users/d/y/dyy12/XuLab/results_293T_analysis/maz_ref_audit")
e = pd.read_csv(OUT / "encode_maz_experiments.tsv", sep="\t")
f = pd.read_csv(OUT / "encode_maz_files.tsv", sep="\t")
ab = pd.read_csv(OUT / "encode_maz_antibodies.tsv", sep="\t").set_index("antibody")
au = pd.read_csv(OUT / "encode_maz_audits.tsv", sep="\t")


def join(v, fmt="{:.2f}"):
    return ";".join(fmt.format(x) for x in v if pd.notna(x))


rows = []
for r in e.itertuples():
    ff = f[f.exp == r.exp]
    aln = ff[(ff.output_type == "alignments")].sort_values("bio_rep")
    pe = 2 if "paired" in str(r.run_type) else 1
    idr = ff[(ff.output_type == "IDR thresholded peaks") & (ff.file_type == "bed narrowPeak") & (ff.bio_rep.astype(str).str.contains(","))]
    cons = ff[(ff.output_type == "conservative IDR thresholded peaks") & (ff.file_type == "bed narrowPeak")]
    idr1 = idr.iloc[0] if len(idr) else pd.Series(dtype=object)
    reps = ff[(ff.output_type == "IDR thresholded peaks") & (ff.file_type == "bed narrowPeak") & ~(ff.bio_rep.astype(str).str.contains(","))].sort_values("bio_rep")
    a = au[(au.exp == r.exp) & (au.level != "INTERNAL_ACTION")]
    mod = str(r.modification)
    construct = ("exogenous eGFP-MAZ cDNA, CMV promoter, Flp-In FRT single-copy integration" if "Flp-In" in mod
                 else "endogenous MAZ, CRISPR C-terminal LAP (eGFP) knock-in" if "LAP" in mod
                 else "endogenous MAZ, CRISPR 3xFLAG knock-in" if "HAIB" in r.lab
                 else "untagged endogenous MAZ (anti-MAZ antibody)")
    rows.append(dict(
        experiment=r.exp, cell_line=r.cell, lab=r.lab, released=r.released, run_type=r.run_type, construct=construct,
        antibody=f"{r.antibody} {ab.loc[r.antibody, 'source']} {ab.loc[r.antibody, 'product']} (target {ab.loc[r.antibody, 'targets']})",
        antibody_status=ab.loc[r.antibody, "status"], n_bioreps=r.n_bioreps,
        read_length=join(aln.read_length, "{:.0f}"),
        usable_fragments_M=join(aln["ChipAlignmentQualityMetric:mapped_reads"] / pe / 1e6, "{:.1f}"),
        NSC=join(aln["ChipAlignmentEnrichmentQualityMetric:NSC"]), RSC=join(aln["ChipAlignmentEnrichmentQualityMetric:RSC"]),
        NRF=join(aln["ChipLibraryQualityMetric:NRF"]), PBC1=join(aln["ChipLibraryQualityMetric:PBC1"]),
        pct_dup=join(aln["ChipLibraryQualityMetric:pct_duplicate_reads"], "{:.1f}"),
        FRiP_pooled_IDR=idr1.get("ChipPeakEnrichmentQualityMetric:frip", np.nan),
        FRiP_per_rep=join(reps["ChipPeakEnrichmentQualityMetric:frip"], "{:.3f}"),
        rescue_ratio=idr1.get("ChipReplicationQualityMetric:rescue_ratio", np.nan),
        self_consistency_ratio=idr1.get("ChipReplicationQualityMetric:self_consistency_ratio", np.nan),
        reproducibility=idr1.get("ChipReplicationQualityMetric:reproducibility", np.nan),
        audit_ERROR=int((a.level == "ERROR").sum()), audit_NOT_COMPLIANT=int((a.level == "NOT_COMPLIANT").sum()),
        audit_WARNING="; ".join(sorted(set(a[a.level == "WARNING"].category))),
        idr_peak_file=idr1.get("file", ""), idr_peak_url=idr1.get("href", ""),
        conservative_peak_file=";".join(cons.file),
        pooled_pval_bigwig=";".join(ff[(ff.output_type == "signal p-value") & ff.bio_rep.astype(str).str.contains(",")].file),
        pooled_fc_bigwig=";".join(ff[(ff.output_type == "fold change over control") & ff.bio_rep.astype(str).str.contains(",")].file),
        rep_bams=";".join(aln.file),
    ))
s = pd.DataFrame(rows)
s.to_csv(OUT / "encode_maz_qc_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
print(s.drop(columns=["lab", "released", "antibody", "audit_WARNING", "idr_peak_url", "rep_bams", "construct"]).to_string())
