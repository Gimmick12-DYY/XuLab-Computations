#!/bin/bash
#SBATCH -J f4_artm
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/funnel_20261120/art_match_%j.log
# Codebook artifact matching for the funnel motif sets, as complex_screen/pair_motif_composite/codebook_full.py: Tomtom -dist pearson -min-overlap 5 -thresh 0.05
# of every DB motif (db/known.meme) and every HOMER known motif (HOMER 5.1 vertebrate library) against the 37 Codebook artifact PWMs.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; O=$X/complex_screen/funnel_20261120; ART=$X/motif_analysis/cache/motifdb/codebook_artifacts.meme
module load meme/5.5.7 >/dev/null 2>&1 || true
python3 $O/homer2meme.py /nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs $O/homer_known.meme
for q in $O/homer_known.meme $X/complex_screen/pair_motif_composite/db/known.meme; do
  tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.05 $q $ART > $O/tomtom_$(basename $q .meme).tsv
done
wc -l $O/tomtom_*.tsv
