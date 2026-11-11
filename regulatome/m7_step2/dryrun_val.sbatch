#!/bin/bash
# m7_step2: exercise score.py and summarise.py before the freeze, on validation chromosomes and by-gene fits only.
# Touches no test-chromosome gene and no held-out-TF fold; writes only under logs/dryrun. usage: sbatch --export=ALL,M7_LINK=hicfixed dryrun_val.sbatch
#SBATCH -J m7dry
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/dryrun_%j.log
set -e
O=/work/users/d/y/dyy12/XuLab/regulatome/m7_step2; D=$O/logs/dryrun; PY=/work/users/d/y/dyy12/envs/boltz/bin/python
export PYTHONPATH=$O:/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=4
mkdir -p $D; ln -sfn $O/splits $D/splits; cd $D
sed -e 's/set_mask("test"/set_mask("val"/g' \
    -e "s#res = f\"{O}/results/{kind}_{LINK}\"#res = f\"$D/results/{kind}_{LINK}\"#" \
    -e 's/^assert os.path.exists(f"{O}\/FROZEN.txt")$//' \
    -e 's/_kd_final.tsv", sep="\\t", dtype=str, keep_default_na=False)$/_kd_final.tsv", sep="\\t", dtype=str, keep_default_na=False); jobs = jobs[jobs.split == "gene"].reset_index(drop=True)/' \
    -e 's/for split in \["gene"\] + \[f"fold{k}" for k in range(5)\]:/for split in ["gene"]:/' \
    $O/score.py > $D/score_val.py
! grep -q 'set_mask("test"' $D/score_val.py
grep -q 'jobs\[jobs.split == "gene"\]' $D/score_val.py
! grep -q 'range(5)\]:' $D/score_val.py
sed -e "s#^O = \"[^\"]*\"#O = \"$D\"#" $O/summarise.py > $D/summarise_val.py
$PY score_val.py raw
$PY summarise_val.py > summarise_val.out
$PY - <<PYEOF
import glob, json, os, numpy as np, pandas as pd
L = os.environ.get("M7_LINK", "powerlaw"); O = "$O"
d = pd.read_csv(f"$D/results/raw_{L}/per_tf_long.tsv", sep="\t", keep_default_na=False)
d = d[(d.split == "gene") & ~d.model.str.startswith("ref_")]
rows = []
for k, g in d.groupby(["arm", "centring", "model", "mode", "cond"]):
    f = f"{O}/fits/raw_{L}/final/kd/gene/{k[0]}_{k[1]}/{k[2]}_{k[3]}_{k[4]}/val.json"
    rows.append(dict(zip(("arm", "centring", "model", "mode", "cond"), k), rescored=g.spearman.mean(), n_tf=g.tf.nunique(), at_fit=json.load(open(f))["val_spearman_mean"]))
R = pd.DataFrame(rows); R["diff"] = (R.rescored - R.at_fit).abs(); print(len(R), "fits; max |rescored - at_fit| =", R["diff"].max()); print(R.sort_values("diff", ascending=False).head(8).to_string(index=False))
PYEOF
echo DRYRUN_OK
