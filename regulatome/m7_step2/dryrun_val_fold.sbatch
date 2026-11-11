#!/bin/bash
# m7_step2: exercise the by-TF branch of score.py before the freeze: fold fits scored on each fold's TRAINING TFs at validation genes. No held-out TF and no test-chromosome gene is used. Writes only under logs/dryrun_fold.
#SBATCH -J m7dry
#SBATCH -p general
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/regulatome/m7_step2/logs/dryrunfold_%j.log
set -e
O=/work/users/d/y/dyy12/XuLab/regulatome/m7_step2; D=$O/logs/dryrun_fold; PY=/work/users/d/y/dyy12/envs/boltz/bin/python
export PYTHONPATH=$O:/work/users/d/y/dyy12/envs/m7_pydeps OMP_NUM_THREADS=4
mkdir -p $D; cd $D
sed -e 's/set_mask("test"/set_mask("val"/g' \
    -e "s#res = f\"{O}/results/{kind}_{LINK}\"#res = f\"$D/results/{kind}_{LINK}\"#" \
    -e 's/^assert os.path.exists(f"{O}\/FROZEN.txt")$//' \
    -e 's/_kd_final.tsv", sep="\\t", dtype=str, keep_default_na=False)$/_kd_final.tsv", sep="\\t", dtype=str, keep_default_na=False); jobs = jobs[jobs.split == "fold0"].reset_index(drop=True)/' \
    -e 's/test_tfs = tf_lists(job.split, arm_tfs)\[1\]/test_tfs = tf_lists(job.split, arm_tfs)[0]/' \
    -e 's/for split in \["gene"\] + \[f"fold{k}" for k in range(5)\]:/for split in []:/' \
    $O/score.py > $D/score_val.py
! grep -q 'set_mask("test"' $D/score_val.py
! grep -q 'arm_tfs)\[1\]; tt, gg = kd_rows(test_tfs, set_mask("val", arm), Zp); p = kd_predict' $D/score_val.py
grep -q 'for split in \[\]:' $D/score_val.py
$PY score_val.py raw
$PY - <<PYEOF
import json, os, pandas as pd
L = os.environ.get("M7_LINK", "powerlaw"); O = "$O"
d = pd.read_csv(f"$D/results/raw_{L}/per_tf_long.tsv", sep="\t", keep_default_na=False); rows = []
for k, g in d.groupby(["arm", "centring", "model", "mode", "cond"]):
    v = json.load(open(f"{O}/fits/raw_{L}/final/kd/fold0/{k[0]}_{k[1]}/{k[2]}_{k[3]}_{k[4]}/val.json"))
    rows.append(dict(zip(("arm", "centring", "model", "mode", "cond"), k), rescored=g.spearman.mean(), n_tf=g.tf.nunique(), at_fit=v["val_spearman_mean"]))
R = pd.DataFrame(rows); R["diff"] = (R.rescored - R.at_fit).abs(); print(len(R), "fits; max diff", R["diff"].max()); print(R.groupby("model")["diff"].max()); print(R.sort_values("diff", ascending=False).head(6).to_string(index=False))
PYEOF
echo DRYRUN_OK
