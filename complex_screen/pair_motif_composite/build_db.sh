#!/bin/bash
# Known motif DB for SEA / CentriMo / Tomtom / SpaMo: JASPAR 2024 CORE vertebrates non-redundant + HOCOMOCO v12 CORE + Codebook top-1
# + CAP-SELEX composites available here (MLX_FOXA1, MLX_FOXD2 from composite_motif/). Merged with stock meme2meme.
set -euo pipefail
X=/work/users/d/y/dyy12/XuLab; C=$X/motif_analysis/cache/motifdb; O=$X/complex_screen/pair_motif_composite/db; mkdir -p $O
module load meme/5.5.7 >/dev/null 2>&1
cp $C/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt $O/jaspar.meme
( head -9 $C/codebook_top1.meme | grep -v "^MOTIF"; cat $X/complex_screen/composite_motif/capselex_mlx_fox.meme.part ) > $O/capselex.meme
meme2meme $O/jaspar.meme $C/H12CORE_meme_format.meme $C/codebook_top1.meme $O/capselex.meme > $O/known.meme
grep -c "^MOTIF" $O/jaspar.meme $C/H12CORE_meme_format.meme $C/codebook_top1.meme $O/capselex.meme $O/known.meme
