# Source A (SPEC_SIGNED.md): one predicted sign per panel TF from its identity / complex membership (literature-level knowledge, written by hand,
# no data used). "unknown" wherever the direction is not established; "dual" where the factor is well documented as both. Writes
# signed_v1/tf_sign_annotation.tsv for all 79 panel TFs (HP1 = CBX1/3/5; ZUFSP = ZUP1). usage: build_sign_annotation.py
import os
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/regulatome/m6_improve/signed_v1"; os.makedirs(O, exist_ok=True)
R, A, D, U = "repressing", "activating", "dual", "unknown"
ANN = {  # TF: (sign, complex_or_class, evidence, confidence)
 "EZH2": (R, "PRC2", "PRC2 catalytic subunit; H3K27me3 writer", "high"),
 "MTF2": (R, "PRC2", "PRC2.1 accessory (PCL2); recruits PRC2 to CpG islands", "high"),
 "MBD3": (R, "NuRD", "NuRD core subunit (deacetylase/remodeller complex)", "high"),
 "RBBP4": (R, "NuRD/PRC2/SIN3", "histone chaperone in NuRD, PRC2 and SIN3 corepressors; no DNA binding", "high"),
 "RBBP7": (R, "NuRD/PRC2/SIN3", "histone chaperone in NuRD, PRC2 and SIN3 corepressors; no DNA binding", "high"),
 "HP1": (R, "HP1 heterochromatin", "CBX1/3/5 bind H3K9me3; heterochromatin silencing (CBX3 also in gene bodies)", "high"),
 "SETDB1": (R, "KAP1/SETDB1", "H3K9 methyltransferase; KRAB-ZNF/TRIM28 silencing", "high"),
 "DNMT1": (R, "DNA methylation", "maintenance DNA methyltransferase", "high"),
 "PCGF6": (R, "PRC1.6", "PRC1.6 subunit; H2AK119ub silencing of germline genes", "high"),
 "HES1": (R, "bHLH repressor", "Notch target; WRPW/TLE-dependent transcriptional repressor", "high"),
 "NR1D2": (R, "nuclear receptor (Rev-erb)", "Rev-erb beta; ligand-dependent repressor recruiting NCOR/HDAC3", "high"),
 "WIZ": (R, "G9a/GLP", "core of the G9a/GLP H3K9me2 complex", "med"),
 "ZNF644": (R, "G9a/GLP", "G9a/GLP-associated zinc finger", "med"),
 "ADNP": (R, "ChAHP", "ChAHP complex with CHD4 and HP1; represses targets", "med"),
 "CHAMP1": (R, "HP1/POGZ", "associates with POGZ and HP1", "low"),
 "TBX3": (R, "T-box repressor", "T-box transcriptional repressor", "med"),
 "MSX2": (R, "homeodomain repressor", "MSX homeoproteins act mainly as repressors", "med"),
 "ZNF703": (R, "corepressor", "Zeppo1/NLZ1; transcriptional corepressor (Groucho/TLE-interacting)", "med"),
 "ZNF746": (R, "KRAB-ZNF", "PARIS; KRAB zinc finger repressing PGC-1alpha", "med"),
 "ZNF282": (R, "KRAB-ZNF", "HUB1; KRAB-containing zinc finger reported as a repressor", "low"),
 "DR1": (R, "NC2", "NC2-beta (negative cofactor 2); inhibits TBP-dependent initiation", "med"),
 "SAFB": (R, "SAF-B", "reported corepressor (e.g. of ER signalling); nuclear matrix protein", "low"),
 "NRF1": (A, "promoter activator", "GC-rich promoter activator (mitochondrial/housekeeping genes)", "high"),
 "MAZ": (A, "promoter activator", "GC-box zinc finger activator at promoters", "high"),
 "NFYA": (A, "NF-Y", "CCAAT-box pioneer activator (NF-Y)", "high"),
 "E2F3": (A, "E2F", "activating E2F (E2F1-3 class); S-phase genes", "high"),
 "ZNF143": (A, "promoter activator", "STAF; activator of snRNA/Pol II promoters", "high"),
 "EPAS1": (A, "bHLH-PAS", "HIF-2alpha; hypoxia-response activator", "high"),
 "RELA": (A, "NF-kB", "NF-kB p65 transactivator", "high"),
 "MEF2A": (A, "MADS-box", "MEF2 activator", "high"),
 "MEF2D": (A, "MADS-box", "MEF2 activator", "high"),
 "ETS2": (A, "ETS", "ETS-family transactivator", "med"),
 "FOXC1": (A, "forkhead", "forkhead transactivator", "med"),
 "SOX4": (A, "SOX (SoxC)", "SoxC transactivator", "med"),
 "SOX12": (A, "SOX (SoxC)", "SoxC transactivator", "low"),
 "NEUROD4": (A, "bHLH activator", "proneural bHLH activator", "med"),
 "SRCAP": (A, "SRCAP", "H2A.Z deposition; CBP coactivator", "med"),
 "GLYR1": (A, "elongation", "NDF; facilitates Pol II elongation through nucleosomes", "low"),
 "NME2": (A, "PuF", "PuF; reported activator of MYC (NHE III1)", "low"),
 "RBPJ": (D, "Notch", "represses with corepressors; activates with Notch ICD", "high"),
 "ZBTB7A": (D, "BTB-ZF", "LRF/Pokemon; repressor and activator in different contexts", "med"),
 "E4F1": (D, "E4F", "both activator and repressor reported", "med"),
 "MLX": (D, "MLX network", "partners MondoA/ChREBP (activating) and MAD/MNT (repressing)", "med"),
 "SREBF2": (A, "bHLH-LZ", "SREBP2; sterol-regulated activator", "high"),
 "DDIT3": (D, "C/EBP family", "CHOP; inhibits C/EBP dimers and activates stress targets", "med"),
 "FOXK1": (D, "forkhead", "activator and SIN3-linked repressor reported", "med"),
 "ZIC2": (D, "ZIC", "activator; also recruits NuRD at enhancers", "med"),
 "CTCF": (U, "architectural", "insulator / loop anchor; direction context-dependent", "low"),
 "STAG2": (U, "cohesin", "cohesin subunit; architectural", "low"),
}
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; assert len(tfs) == 79
rows = [dict(TF=t, predicted_sign=ANN.get(t, (U,))[0], complex_or_class=ANN.get(t, (U, ".", ".", "."))[1], evidence=ANN.get(t, (U, ".", "no established direction", "."))[2], confidence=ANN.get(t, (U, ".", ".", "low"))[3]) for t in tfs]
for r in rows:
    if r["TF"] not in ANN: r.update(complex_or_class="." , evidence="no established direction (written as unknown rather than guessed)", confidence=".")
T = pd.DataFrame(rows); T.to_csv(f"{O}/tf_sign_annotation.tsv", sep="\t", index=False); print(T.predicted_sign.value_counts().to_dict())
