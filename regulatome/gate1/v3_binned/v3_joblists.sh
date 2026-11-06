#!/bin/bash
# job lists: per-TF x 4 unit types; modules x 4 unit types; 2-TF test
V3=/work/users/d/y/dyy12/XuLab/regulatome/gate1/v3_binned
T=/work/users/d/y/dyy12/XuLab/regulatome/m1_binding/v2/tfs.txt
MODS="PRC2core NuRDcore lowdepth lowdepth_noNEUROD4 lowdepth_noZNF777 prc2adj prc2adj_noEZH2MTF2 prc2adj_noMTF2 prc2adj_noEZH2 prc2adj_noDR1 prc2adj_noTHAP6 prc2adj_noCHAMP1 prc2adj_noWIZ prc2adj_noZNF644 prc2adj_noSRCAP $(for k in $(seq -w 1 20); do printf 'randshallow%s ' $k; done)"
: > $V3/jobs_tf.txt; : > $V3/jobs_mod.txt; : > $V3/jobs_test.txt
for u in bin10k bin50k reprpc genebody; do
  while read t; do [ -n "$t" ] && echo "$u tf $t" >> $V3/jobs_tf.txt; done < $T
  for m in $MODS; do echo "$u module $m" >> $V3/jobs_mod.txt; done
  for t in EZH2 NFYA; do echo "$u tf $t" >> $V3/jobs_test.txt; done
done
wc -l $V3/jobs_*.txt
