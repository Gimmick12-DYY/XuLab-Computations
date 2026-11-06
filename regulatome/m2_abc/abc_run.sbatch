#!/bin/bash
#SBATCH -J abc_run
#SBATCH -p general
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m2_abc/logs/abc_%x_%j.out
# ABC v1.1.2 on our active candidate elements, ATAC only. Usage: sbatch abc_run.sbatch neighborhoods|powerlaw|hic
set -euo pipefail
MODE=$1
A=/work/users/d/y/dyy12/XuLab/regulatome/m2_abc
S=$A/abc_v1.1.2/workflow/scripts
R=$A/abc_v1.1.2/reference
I=$A/inputs
TA=/work/users/d/y/dyy12/XuLab/atac_omniatac/caper_work/atac/70bfc7e8-3760-496f-bd8e-5a4bc9e02b51/call-bam2ta
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/d/y/dyy12/envs/abc
# ABC tabix-indexes its accessibility input, so use one merged, coordinate-sorted, bgzipped copy of the 4 replicate tagAligns
ATAC=$I/omniatac_merged.tn5.tagAlign.gz
if [ ! -s $ATAC.tbi ]; then
  # sorted in the chrom_sizes.tsv order (ABC runs bedtools with -sorted -g chrom_sizes)
  zcat $TA/shard-*/execution/*.tn5.tagAlign.gz | awk -v OFS="\t" "NR==FNR{r[\$1]=NR; next} (\$1 in r){print r[\$1], \$0}" $I/chrom_sizes.tsv - | \
    LC_ALL=C sort -k1,1n -k3,3n -S 8G -T $A/tmp | cut -f2- | bgzip -@ 1 > $ATAC
  tabix -p bed $ATAC
fi
N=$A/results/Neighborhoods
PL="--hic_gamma 1.024238616787792 --hic_scale 5.9594510043736655 --hic_pseudocount_distance 5000 --scale_hic_using_powerlaw"
case $MODE in
neighborhoods)
  python $S/run.neighborhoods.py --candidate_enhancer_regions $I/candidates_active.bed --ATAC $ATAC \
    --default_accessibility_feature ATAC --chrom_sizes $I/chrom_sizes.tsv --chrom_sizes_bed $I/chrom_sizes.bed \
    --outdir $N --genes $I/genes_pc_v46.bed --ubiquitously_expressed_genes $R/UbiquitouslyExpressedGenes.txt \
    --qnorm $R/EnhancersQNormRef.K562.txt ;;
powerlaw)
  O=$A/results/Predictions_powerlaw
  python $S/predict.py --enhancers $N/EnhancerList.txt --genes $N/GeneList.txt --outdir $O --score_column powerlaw.Score \
    --chrom_sizes $I/chrom_sizes.tsv --accessibility_feature ATAC --cellType HEK293T $PL ;;
hic)
  O=$A/results/Predictions_hic5kb
  python $S/predict.py --enhancers $N/EnhancerList.txt --genes $N/GeneList.txt --outdir $O --score_column ABC.Score \
    --chrom_sizes $I/chrom_sizes.tsv --accessibility_feature ATAC --cellType HEK293T \
    --hic_file $I/hic5kb --hic_type juicebox --hic_resolution 5000 $PL ;;
esac
if [ $MODE != neighborhoods ]; then
  python $S/filter_predictions.py --output_tsv_file $O/EnhancerPredictionsFull_thr0.02.tsv \
    --output_slim_tsv_file $O/EnhancerPredictions_thr0.02.tsv --output_bed_file $O/EnhancerPredictionsFull_thr0.02.bedpe \
    --output_gene_stats_file $O/GenePredictionStats_thr0.02.tsv --pred_file $O/EnhancerPredictionsAllPutative.tsv.gz \
    --pred_nonexpressed_file $O/EnhancerPredictionsAllPutativeNonExpressedGenes.tsv.gz --score_column ABC.Score \
    --threshold 0.02 --include_self_promoter True --only_expressed_genes False
fi
