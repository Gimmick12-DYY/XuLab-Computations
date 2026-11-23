#!/bin/bash
# Copy final per-TF data into the lab share /vast/som/xujie_lab/293T_analysis/data (copies only; never overwrites).
# Usage: copy_to_293T_analysis.sh raw|atac_mask|accessibility_mask
set -euo pipefail
umask 002
X=/work/users/d/y/dyy12/XuLab; U=$X/unified/work
T=/vast/som/xujie_lab/293T_analysis/data
TAG=/vast/som/xujie_lab/TAGATG-293T/bigwig
CP="cp -n --preserve=timestamps"
up() { echo "$1" | tr a-z A-Z; }
case "$1" in
raw)
  for f in /vast/som/xujie_lab/data_share/293T_TF/fragments/TF.*.bed.gz; do $CP "$f" $T/raw/fragments/; done
  for f in $X/data/*_bin1000_mtx.rds; do $CP "$f" $T/raw/bin/; done
  for f in $TAG/peaks/TF.*_peaks.q005.narrowPeak; do $CP "$f" $T/raw/peak/; done
  ;;
atac_mask|accessibility_mask)
  if [ "$1" = atac_mask ]; then D=$T/imputed/ATAC_mask; SUB=impute; else D=$T/imputed/accessibility_mask; SUB=impute_atac_cpm_drop5; fi
  [ -e $D/regions.tsv.gz ] || gzip -c $U/ctcf/impute/regions.tsv > $D/regions.tsv.gz
  for w in $U/*/; do
    tf=$(basename $w); s=$w$SUB
    [ -f $s/matrix_csr.npz ] || { echo "skip $tf (no $SUB)"; continue; }
    cmp -s $s/regions.tsv $U/ctcf/impute/regions.tsv || { echo "ERROR regions differ $tf"; exit 1; }
    o=$D/$(up $tf); mkdir -p $o
    for n in matrix_csr.npz barcodes.tsv meta.json; do $CP $s/$n $o/; done
  done
  ;;
esac
chmod -R g+rwX,o+rX $T
echo done $1
