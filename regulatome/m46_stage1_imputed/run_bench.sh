#!/bin/bash
# m46 Amendment 4: submit the existing all_universe benchmark (AUROC, bin coverage, peak coverage) on raw + remasked inputs; outputs under m46 work/bench.
set -euo pipefail
ROOT=/work/users/d/y/dyy12/XuLab; DS=$ROOT/downstream/slurm; B=/work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/work/bench; I=/work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/work/bench_inputs; DEP=${1:-}
mkdir -p $B
for spec in ctcf:all_universe ctcf:maj_zeroneg maz:all_universe zic2:all_universe zbtb7a:all_universe znf777:all_universe nfya:all_universe znf282:all_universe; do
  tf=${spec%%:*}; mode=${spec##*:}; tag=${tf}_${mode}
  export BINS_TSV=$ROOT/downstream/eval/bins/bins_${tf}_${mode}/pos_neg_bins.tsv
  export INPUTS="raw=$ROOT/unified/work/$tf/mm Mpeak=$I/$tf/Mpeak M0=$I/$tf/M0 Mdrop5=$I/$tf/Mdrop5 Mzero=$I/$tf/Mzero M5=$I/$tf/M5 M50=$I/$tf/M50"
  export ROOT_DIR=$ROOT
  d=${DEP:+--dependency=afterok:$DEP}
  export OUT_DIR=$B/compare_pos_neg_$tag OUT_NAME=compare_pos_neg_$tag
  a=$(cd $ROOT && sbatch --parsable $d --mem=128G --time=08:00:00 -J m46b_auc_$tag -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/auc_${tag}_%j.log -e /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/auc_${tag}_%j.err --export=ALL,BINS_TSV,OUT_DIR,OUT_NAME,INPUTS $DS/compare_pos_neg.sbatch)
  export OUT_DIR=$B/bin_coverage_$tag OUT_NAME=bin_coverage_$tag
  b=$(cd $ROOT && sbatch --parsable $d --mem=128G --time=06:00:00 -J m46b_bin_$tag -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/bin_${tag}_%j.log -e /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/bin_${tag}_%j.err --export=ALL,BINS_TSV,OUT_DIR,OUT_NAME,INPUTS,ROOT_DIR $DS/bin_sensitivity_specificity.sbatch)
  export PEAK_COV_OUT_DIR=$B/peak_coverage_$tag OUT_NAME=peak_coverage_$tag; unset REUSE_PEAKS_FROM || true
  c=$(cd $ROOT && sbatch --parsable $d --mem=96G --time=24:00:00 -J m46b_peak_$tag -o /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/peak_${tag}_%j.log -e /work/users/d/y/dyy12/XuLab/regulatome/m46_stage1_imputed/out/bench/peak_${tag}_%j.err --export=ALL,BINS_TSV,PEAK_COV_OUT_DIR,OUT_NAME,INPUTS,ROOT_DIR $DS/peak_coverage.sbatch)
  echo "$tag auc=$a bin=$b peak=$c"
done
