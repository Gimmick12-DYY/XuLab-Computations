#!/bin/bash
#SBATCH -J motdiag
#SBATCH -p general
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH -t 00:45:00
#SBATCH -o /work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103/logs/diag_%j.out
O=/work/users/d/y/dyy12/XuLab/complex_screen/motif_shared_20261103; PY=/work/users/d/y/dyy12/conda/envs/snapatac2/bin/python; cd $O; $PY -I diag_fg.py || exit 1; module load meme/5.5.7
for d in diag/*/*/; do sea --p $d/fg.fa --n $d/bg.fa --m db/db.meme --thresh 1000 --noseqs --oc $d/sea >/dev/null 2>&1; done
$PY -I - <<'P'
import glob, json, os, pandas as pd
own = json.load(open("db/own.json"))["own"]; rows = []
for f in sorted(glob.glob("diag/*/*/sea/sea.tsv")):
    p = f.split("/"); d = pd.read_csv(f, sep="\t", comment="#"); d = d[d.ID.notna()]; t = p[2]; s = d[d.ID.isin(own.get(t.upper(), []))]; rows.append(dict(arm=p[1], tf=t, n_sig=int((d.QVALUE < .05).sum()), own_best_q=s.QVALUE.min(), own_rank=int(s.RANK.min()) if len(s) else -1, enr=round(s.sort_values("QVALUE").ENR_RATIO.iloc[0], 2) if len(s) else None, top3=", ".join(d.ALT_ID.fillna(d.ID).head(3))))
    if t in ("mlx", "srebf2"): rows[-1]["ebox_sre"] = "; ".join(f"{r.ID} q={r.QVALUE:.2g} r={r.ENR_RATIO}" for r in d[d.ID.isin(["MA0663.1", "MA0828.3", "MA0596.1"])].itertuples())
T = pd.DataFrame(rows); pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 80); print(T.to_string(index=False)); T.to_csv("diag/diag_summary.tsv", sep="\t", index=False)
P
