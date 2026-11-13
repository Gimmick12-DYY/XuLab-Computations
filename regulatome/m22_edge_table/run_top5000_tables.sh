#!/bin/bash
# m22 Amendment 4: downstream tables on the top-5,000 edge table (no plots).
set -e
P=/work/users/d/y/dyy12/conda/envs/cobinding/bin/python; cd /work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table
$P m22_analyse.py
$P share_depth.py
$P simple_correlation_point.py
$P simple_correlation_pertf.py
$P highconf_pairs.py
for t in NRF1 E4F1 CTCF NFYA MLX; do $P tf2genes.py $t; done
for g in HSPB1 PGAM1 MYC TXNIP; do $P gene2tfs.py $g; done
/work/users/d/y/dyy12/conda/envs/cobinding/bin/python candidates.py
