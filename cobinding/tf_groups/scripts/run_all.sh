#!/bin/bash
# Full pipeline: incidence (all pairs, clique edges) -> curveball nulls (plain and library-stratified; all/odd/even chromosomes).
cd /work/users/d/y/dyy12/XuLab/cobinding/tf_groups; PY=~/.conda/envs/cistopic/bin/python
b1=$(sbatch --parsable -p general -c1 --mem=16g -t 1:00:00 -o logs/build_%j.log --wrap "$PY scripts/build_incidence.py results")
b2=$(sbatch --parsable -p general -c1 --mem=16g -t 1:00:00 -o logs/build_cliques_%j.log --wrap "$PY scripts/build_incidence.py results_cliques cliques")
i=0
for D in results:$b1 results_cliques:$b2; do d=${D%%:*}; dep=${D##*:}
  for L in pair region; do for S in all odd even; do i=$((i+1))
    sbatch -p general -c1 --mem=8g -t 4:00:00 -d afterok:$dep -o logs/${d}_nullstrat_${L}_${S}_%j.log --wrap "$PY scripts/null_z.py $d $L 1000 $S $i $d/library_strata.tsv"
  done
  i=$((i+1)); sbatch -p general -c1 --mem=8g -t 4:00:00 -d afterok:$dep -o logs/${d}_null_${L}_all_%j.log --wrap "$PY scripts/null_z.py $d $L 1000 all $i"
  done
done
i=$((i+1)); sbatch -p general -c1 --mem=8g -t 4:00:00 -d afterok:$b1 -o logs/results_nullstrat_pair_top1000_%j.log --wrap "$PY scripts/null_z.py results pair 1000 top:1000 $i results/library_strata.tsv"
