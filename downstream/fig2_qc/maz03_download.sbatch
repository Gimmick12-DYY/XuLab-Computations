#!/bin/bash
#SBATCH -J maz03_dl
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/results_293T_analysis/maz_ref_audit/logs/maz03_%j.out
# MAZ reference audit, step 3: download candidate MAZ bulk datasets (peaks + signal bigWigs, hg38).
# Cell lines are labelled in the file names. Non-HEK sets are for judging the ENCODE HEK293 set only, not as references.
set -euo pipefail
D=/work/users/d/y/dyy12/XuLab/data/external/maz_refs
mkdir -p $D/peaks $D/bw
cd $D
ca() { # name SRX
  [ -s bw/$1.bw ] || curl -sSfL --retry 4 -o bw/$1.bw https://chip-atlas.dbcls.jp/data/hg38/eachData/bw/$2.bw
  [ -s peaks/$1.chipatlas_q1e-5.bed ] || curl -sSfL --retry 4 -o peaks/$1.chipatlas_q1e-5.bed https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/$2.05.bed; }

# --- ENCODE (default ENCODE4 GRCh38 analysis): IDR thresholded peaks + pooled fold-change bigWig
while read name pk fc; do
  [ -s peaks/$name.encode_idr.bed.gz ] || curl -sSfL --retry 4 -o peaks/$name.encode_idr.bed.gz https://www.encodeproject.org/files/$pk/@@download/$pk.bed.gz
  if [ "$fc" != "-" ]; then [ -s bw/$name.bw ] || curl -sSfL --retry 4 -o bw/$name.bw https://www.encodeproject.org/files/$fc/@@download/$fc.bigWig; fi
done <<LIST
HEK293_eGFP-MAZ_ENCSR290SSQ ENCFF529XTF ENCFF214VQU
K562_LAP-MAZ_ENCSR643JRH ENCFF962GWS ENCFF459WXZ
GM12878_LAP-MAZ_ENCSR903MVU ENCFF618NGB ENCFF705KIU
HepG2_FLAG-MAZ_ENCSR700PNE ENCFF341OLL ENCFF873NBY
K562_FLAG-MAZ_ENCSR163IUV ENCFF704UVS ENCFF454IIJ
GM12878_abMAZ_ENCSR000DZA ENCFF467VSF ENCFF420YPZ
MCF-7_abMAZ_ENCSR288IJC ENCFF290CNC ENCFF116ZNK
HepG2_abMAZ_ENCSR000EDN ENCFF086ZTC -
K562_abMAZ_ENCSR000EFX ENCFF837NNR -
IMR-90_abMAZ_ENCSR000EFF ENCFF566MPI -
A549_abMAZ_ENCSR636YLV ENCFF652DIB -
HeLa-S3_abMAZ_ENCSR000ECL ENCFF424JHD -
LIST

# --- GEO GSE76494 (Schmitges 2016, Hughes lab): HEK293 Flp-In GFP-MAZ ChIP-seq, rep1 only; ChIP-Atlas hg38 reprocessing
ca HEK293_GFP-MAZ_GSE76494 SRX1514969
# --- GEO GSE249245 (2024): HEK293T V5-MAZ CUT&RUN (transient), with V5 mock IP and V5-CTCF from the same batch
for x in "HEK293T_V5-MAZ_CnR_rep1 SRX22725678 GSM7931993 MAZ_1" "HEK293T_V5-MAZ_CnR_rep2 SRX22725679 GSM7931994 MAZ_2" \
         "HEK293T_V5-CTCF_CnR_rep1 SRX22725662 GSM7931977 CTCF_1" "HEK293T_V5-CTCF_CnR_rep2 SRX22725663 GSM7931978 CTCF_2" \
         "HEK293T_V5-mock_CnR_rep1 SRX22725717 - -" "HEK293T_V5-mock_CnR_rep2 SRX22725718 - -"; do
  set -- $x; ca $1 $2
  # GEO per-sample links are partly broken (GSM7931994 returns 403; https on the NCBI ftp host also 403s, ftp:// works), so take the peaks from the series RAW tar (2.5 MB)
  if [ "$3" != "-" ] && [ ! -s peaks/$1.geo_macs2_q0.05.narrowPeak.gz ]; then
    mkdir -p geo_raw; [ -s geo_raw/GSE249245_RAW.tar ] || curl -sSfL --retry 4 -o geo_raw/GSE249245_RAW.tar ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE249nnn/GSE249245/suppl/GSE249245_RAW.tar
    tar -xOf geo_raw/GSE249245_RAW.tar ${3}_$4_peaks.narrowPeak.gz > peaks/$1.geo_macs2_q0.05.narrowPeak.gz
  fi
done
# --- GEO GSE215081 (HAP1): anti-MAZ ChIP-seq in parental vs MAZ-KO HAP1 (antibody specificity control)
ca HAP1_abMAZ_parental_rep1 SRX23678713
ca HAP1_abMAZ_parental_rep2 SRX23678714
ca HAP1_abMAZ_MAZKO_rep1 SRX23678717
ca HAP1_abMAZ_MAZKO_rep2 SRX23678718
ls -la peaks bw
md5sum peaks/* bw/* > MD5SUMS.txt
