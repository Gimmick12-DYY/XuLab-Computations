#!/bin/bash
#SBATCH -J abc_k27
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac/logs/abc_%j.out
# ABC v1.1.2 rebuilt exactly as m2_abc/abc_run.sbatch (power-law contact, same candidates, genes, K562 qnorm ref, ABC >= 0.02, self-promoter in,
# all genes) except --H3K27ac is added, so activity = sqrt(qnorm ATAC x qnorm H3K27ac) (Fulco 2019)
set -euo pipefail
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc
B=/work/users/d/y/dyy12/XuLab/regulatome/m2b_abc_k27ac
S=$A/abc_v1.1.2/workflow/scripts
R=$A/abc_v1.1.2/reference
I=$A/inputs
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
ATAC=$I/omniatac_merged.tn5.tagAlign.gz
K27=$B/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz
N=$B/results/Neighborhoods
O=$B/results/Predictions_powerlaw
PL="--hic_gamma 1.024238616787792 --hic_scale 5.9594510043736655 --hic_pseudocount_distance 5000 --scale_hic_using_powerlaw"
python $S/run.neighborhoods.py --candidate_enhancer_regions $I/candidates_active.bed --ATAC $ATAC --H3K27ac $K27 \
  --default_accessibility_feature ATAC --chrom_sizes $I/chrom_sizes.tsv --chrom_sizes_bed $I/chrom_sizes.bed \
  --outdir $N --genes $I/genes_pc_v46.bed --ubiquitously_expressed_genes $R/UbiquitouslyExpressedGenes.txt \
  --qnorm $R/EnhancersQNormRef.K562.txt
python $S/predict.py --enhancers $N/EnhancerList.txt --genes $N/GeneList.txt --outdir $O --score_column powerlaw.Score \
  --chrom_sizes $I/chrom_sizes.tsv --accessibility_feature ATAC --cellType HEK293T $PL
python $S/filter_predictions.py --output_tsv_file $O/EnhancerPredictionsFull_thr0.02.tsv \
  --output_slim_tsv_file $O/EnhancerPredictions_thr0.02.tsv --output_bed_file $O/EnhancerPredictionsFull_thr0.02.bedpe \
  --output_gene_stats_file $O/GenePredictionStats_thr0.02.tsv --pred_file $O/EnhancerPredictionsAllPutative.tsv.gz \
  --pred_nonexpressed_file $O/EnhancerPredictionsAllPutativeNonExpressedGenes.tsv.gz --score_column ABC.Score \
  --threshold 0.02 --include_self_promoter True --only_expressed_genes False
wc -l $O/EnhancerPredictionsFull_thr0.02.tsv
