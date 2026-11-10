#!/bin/bash
#SBATCH -J abc_k27_fu
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/followups/logs/abc_%x_%j.out
# Follow-up ABC variants (dev-set tuning). Usage: sbatch abc_variant.sbatch MODE
#   k27_hic   : m2b neighborhoods (ATAC x H3K27ac) + WT Hi-C 5 kb contact, settings of m2_abc Predictions_hic5kb
#               (the ATAC-only Hi-C run already exists: m2_abc/results/Predictions_hic5kb)
#   k27only   : activity = H3K27ac alone (H3K27ac tagAlign passed as the accessibility feature), power-law as m2_abc
#   thr       : re-filter the existing AllPutative predictions of old (m2_abc) and new (m2b) power-law runs at ABC >= 0.01 and 0.05
set -euo pipefail
MODE=$1
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
F=$B/followups
S=$A/abc_v1.1.2/workflow/scripts
R=$A/abc_v1.1.2/reference
I=$A/inputs
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
K27=$B/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz
PL="--hic_gamma 1.024238616787792 --hic_scale 5.9594510043736655 --hic_pseudocount_distance 5000 --scale_hic_using_powerlaw"
filt() {  # $1 = prediction dir, $2 = threshold
  python $S/filter_predictions.py --output_tsv_file $1/EnhancerPredictionsFull_thr$2.tsv \
    --output_slim_tsv_file $1/EnhancerPredictions_thr$2.tsv --output_bed_file $1/EnhancerPredictionsFull_thr$2.bedpe \
    --output_gene_stats_file $1/GenePredictionStats_thr$2.tsv --pred_file $1/EnhancerPredictionsAllPutative.tsv.gz \
    --pred_nonexpressed_file $1/EnhancerPredictionsAllPutativeNonExpressedGenes.tsv.gz --score_column ABC.Score \
    --threshold $2 --include_self_promoter True --only_expressed_genes False; }
case $MODE in
k27_hic)
  O=$F/k27_hic5kb
  python $S/predict.py --enhancers $B/results/Neighborhoods/EnhancerList.txt --genes $B/results/Neighborhoods/GeneList.txt --outdir $O --score_column ABC.Score \
    --chrom_sizes $I/chrom_sizes.tsv --accessibility_feature ATAC --cellType HEK293T \
    --hic_file $I/hic5kb --hic_type juicebox --hic_resolution 5000 $PL
  filt $O 0.02 ;;
k27only)
  N=$F/k27only/Neighborhoods; O=$F/k27only/Predictions_powerlaw
  python $S/run.neighborhoods.py --candidate_enhancer_regions $I/candidates_active.bed --ATAC $K27 \
    --default_accessibility_feature ATAC --chrom_sizes $I/chrom_sizes.tsv --chrom_sizes_bed $I/chrom_sizes.bed \
    --outdir $N --genes $I/genes_pc_v46.bed --ubiquitously_expressed_genes $R/UbiquitouslyExpressedGenes.txt \
    --qnorm $R/EnhancersQNormRef.K562.txt
  python $S/predict.py --enhancers $N/EnhancerList.txt --genes $N/GeneList.txt --outdir $O --score_column powerlaw.Score \
    --chrom_sizes $I/chrom_sizes.tsv --accessibility_feature ATAC --cellType HEK293T $PL
  filt $O 0.02 ;;
thr)
  for d in $A/results/Predictions_powerlaw $B/results/Predictions_powerlaw; do for t in 0.01 0.05; do filt $d $t; done; done ;;
esac
