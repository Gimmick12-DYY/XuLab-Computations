#!/bin/bash
# m7_step2 stage driver (submitted by itself with sbatch dependencies). usage: chain.sh <kind> <stage> <retry>
# Stages: expr_tune -> expr_final -> kd_tune -> kd_final -> (raw only) imp features -> imp expr_final -> imp kd_final -> ready.
# Each stage first checks the previous array's output; missing tasks are resubmitted (max 2 times), then the chain stops with logs/CHAIN_STALLED.
set -u
LINK=${M7_LINK:-powerlaw}; export M7_LINK=$LINK; JP=""; [ "$LINK" != powerlaw ] && JP="${LINK}_"
kind=$1; stage=$2; retry=${3:-0}
S=/work/users/d/y/dyy12/XuLab/regulatome/m7_step2; PY=/work/users/d/y/dyy12/envs/boltz/bin/python; cd $S; export PYTHONPATH=/work/users/d/y/dyy12/envs/m7_pydeps
sub_array() { sbatch --parsable --array=$2%$3 --mem=$4 -J m7_$1 run_array.sbatch jobs/$5; }
next() { sbatch --parsable -J m7chain -p general -c 1 --mem=8G -t 2:00:00 --dependency=afterany:$1 -o logs/chain_%j.log chain.sh $2 $3 ${4:-0}; }
missing() { $PY - "$1" <<'PY'
import sys, os, pandas as pd
sys.argv=["m7", sys.argv[1]]
import m7
J=pd.read_csv(sys.argv[1],sep="\t",dtype=str,keep_default_na=False); miss=[i for i,j in J.iterrows() if not os.path.exists(os.path.join(m7.out_dir(j),"val.json"))]
print(",".join(map(str,miss)))
PY
}
checkpoint() { # args: jobsfile mem conc this_stage: returns 0 if all done
  m=$(missing $1); if [ -z "$m" ]; then return 0; fi
  n=$(echo $m | tr ',' '\n' | wc -l); if [ $retry -ge 2 ]; then echo "STALLED $stage $kind missing $n: $m" | tee -a logs/CHAIN_STALLED; exit 1; fi
  echo "resubmit $n missing tasks of $1"; A=$(sbatch --parsable --array=$m%$3 --mem=$2 -J m7_retry run_array.sbatch $1); next $A $kind $stage $((retry+1)); exit 0
}
case $stage in
 expr_final_submit) checkpoint jobs/${JP}${kind}_expr_tune.tsv 16G 40; $PY select_params.py raw expr; $PY make_jobs.py $kind expr_final; A=$(sub_array exprF 0-$(($(wc -l < jobs/${JP}${kind}_expr_final.tsv)-2)) 40 16G ${JP}${kind}_expr_final.tsv); next $A $kind kd_tune_submit;;
 kd_tune_submit) checkpoint jobs/${JP}${kind}_expr_final.tsv 16G 40; A=$(sub_array kdT 0-$(($(wc -l < jobs/${JP}${kind}_kd_tune.tsv)-2)) 40 64G ${JP}${kind}_kd_tune.tsv); next $A $kind kd_final_submit;;
 kd_final_submit) checkpoint jobs/${JP}${kind}_kd_tune.tsv 64G 40; $PY select_params.py raw kd; $PY make_jobs.py $kind kd_final; A=$(sub_array kdF 0-$(($(wc -l < jobs/${JP}${kind}_kd_final.tsv)-2)) 40 64G ${JP}${kind}_kd_final.tsv); next $A $kind imp_start;;
 imp_start) checkpoint jobs/${JP}${kind}_kd_final.tsv 64G 40; touch logs/RAW_ARM_DONE; A=$(sbatch --parsable run_features.sbatch imp $LINK); next $A imp imp_expr;;
 imp_expr) [ -f features/imp_${LINK}/feature_qc.json ] || { echo "imp features missing" | tee -a logs/CHAIN_STALLED; exit 1; }; $PY make_jobs.py imp expr_final; A=$(sub_array exprF 0-$(($(wc -l < jobs/${JP}imp_expr_final.tsv)-2)) 40 16G ${JP}imp_expr_final.tsv); next $A imp imp_kd;;
 imp_kd) checkpoint jobs/${JP}imp_expr_final.tsv 16G 40; $PY make_jobs.py imp kd_final; A=$(sub_array kdF 0-$(($(wc -l < jobs/${JP}imp_kd_final.tsv)-2)) 40 64G ${JP}imp_kd_final.tsv); next $A imp ready;;
 ready) checkpoint jobs/${JP}imp_kd_final.tsv 64G 40; touch logs/ALL_FITS_DONE;;
esac
