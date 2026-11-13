#!/bin/bash
# Composite + gapped motif study, one pair (line of composite_gapped/tasks.tsv).
#  Composite: HOMER de novo (len 8, 10, 12, 16; -noknown) on co-bound vs matched A-only + B-only background, discovery chromosomes; HOMER known
#             on the same contrast (discovery + held-out) for reference.
#  Gapped:    SpaMo on co-bound sites (summit +/- 150 bp) and, as control, on A-only sites; primary = A's best own DB motif (if any, else B's),
#             secondary = the 2,528-motif DB; spacing / orientation preferences between the primary and secondary motifs.
#SBATCH -J cg_pair
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 4:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/composite_gapped/logs/%x_%A_%a.out
set -euo pipefail
O=/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite; G=$O/composite_gapped
PAIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $G/tasks.tsv); R=$G/results/$PAIR; mkdir -p $R
module load homer/5.1 >/dev/null 2>&1
for pt in disc val; do findMotifsGenome.pl $G/pools/$PAIR/$pt/cob.bed hg38 $R/known_$pt -bg $G/pools/$PAIR/$pt/bg.bed -size given -nomotif -p 2 -preparsedDir $O/preparsed > /dev/null 2>&1; done
for L in 8 10 12 16; do findMotifsGenome.pl $G/pools/$PAIR/disc/cob.bed hg38 $R/dn_L$L -bg $G/pools/$PAIR/disc/bg.bed -size given -noknown -len $L -p 2 -preparsedDir $O/preparsed > /dev/null 2>&1; rm -rf $R/dn_L$L/homerResults/*.svg $R/dn_L$L/homerResults; done
cat $R/dn_L*/homerMotifs.all.motifs > $R/denovo_all.motifs 2>/dev/null || true
if [ -s $R/denovo_all.motifs ]; then findMotifsGenome.pl $G/pools/$PAIR/val/cob.bed hg38 $R/denovo_val -bg $G/pools/$PAIR/val/bg.bed -size given -nomotif -mknown $R/denovo_all.motifs -p 2 -preparsedDir $O/preparsed > /dev/null 2>&1; fi
python3 $O/cg_spamo.py $PAIR
echo done $PAIR
