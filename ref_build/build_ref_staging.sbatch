#!/bin/bash
#SBATCH -J ref_build
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/ref_build/logs/ref_build-%j.out
#SBATCH -e /work/users/d/y/dyy12/XuLab/ref_build/logs/ref_build-%j.err
# Build the shared lab reference tree (hg38, mm10, HEK293T) in a staging
# directory. Once the lab Ref/ folder is writable, copy with:
#   rsync -a --chmod=Dg+rx,Fg+r,o+rX $STAGE/ /vast/som/xujie_lab/Ref/
set -euo pipefail
module load samtools bedtools 2>/dev/null || true

STAGE=${STAGE:-/work/users/d/y/dyy12/Ref_staging}
XL=/work/users/d/y/dyy12/XuLab
HS=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50
MM=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25

fetch () {  # url outdir
  local f; f=$(basename "$1")
  if [ ! -s "$2/$f" ]; then
    wget -q --tries=4 --waitretry=30 -O "$2/$f.part" "$1" || { echo "download failed: $1" >&2; exit 1; }
    mv "$2/$f.part" "$2/$f"
  fi
}

genome () {  # base_url outdir fasta gtf gff3
  local u=$1 d=$2 fa=$3 gtf=$4 gff3=$5
  mkdir -p "$d"
  wget -q -O "$d/MD5SUMS.gencode" "$u/MD5SUMS"
  for f in "$fa" "$gtf" "$gff3"; do fetch "$u/$f" "$d"; done
  ( cd "$d" && grep -F -e "$fa" -e "$gtf" -e "$gff3" MD5SUMS.gencode | md5sum -c - )
  # FASTA: uncompressed + .fai for direct use by most tools
  zcat "$d/$fa" > "$d/${fa%.gz}"; rm "$d/$fa"
  samtools faidx "$d/${fa%.gz}"
  cut -f1,2 "$d/${fa%.gz}.fai" > "$d/chrom.sizes"
  # GFF: GENCODE ships GFF3 only; .gff is the same file under the .gff name
  cp "$d/$gff3" "$d/${gff3%.gff3.gz}.gff.gz"
}

genome "$HS" "$STAGE/hg38" GRCh38.primary_assembly.genome.fa.gz \
  gencode.v50.primary_assembly.annotation.gtf.gz gencode.v50.primary_assembly.annotation.gff3.gz
genome "$MM" "$STAGE/mm10" GRCm38.primary_assembly.genome.fa.gz \
  gencode.vM25.primary_assembly.annotation.gtf.gz gencode.vM25.primary_assembly.annotation.gff3.gz

# HEK293T project data (copies, not links)
H=$STAGE/HEK293T
mkdir -p "$H/ATAC/source_peaks" "$H/HiC_GSE216399_WT/loops_peakachu_0.7"
cp -p $XL/data/HEK293T_ATAC_union3.bed.gz "$H/ATAC/"
cp -p $XL/data/HEK293T_OmniATAC_idr_optimal.narrowPeak.gz \
      $XL/data/HEK293T_ATAC_gse152177_wt_MACS2.narrowPeak.gz \
      $XL/data/HEK293T_ATAC_gse283384_wt_MACS2.narrowPeak.gz "$H/ATAC/source_peaks/"
L=$XL/hic/work_wt293t/loops
cp -p $XL/hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool "$H/HiC_GSE216399_WT/"
for r in 5000 10000; do
  cp -p $L/peakachu_${r}.loops.0.7.bedpe $L/peakachu_${r}.loops.0.7.anchors.bed \
        $L/peakachu_${r}.loops.0.7.summary.tsv "$H/HiC_GSE216399_WT/loops_peakachu_0.7/"
done
( cd "$H" && find . -type f ! -name MD5SUMS -print0 | sort -z | xargs -0 md5sum > MD5SUMS )

find "$STAGE" -mindepth 1 -user "$USER" -exec chmod g+rX,o+rX {} +
echo done; du -sh "$STAGE"/*
