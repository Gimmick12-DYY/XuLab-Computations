# m7_step2: per-TF weights of the adopted Regulome model (section 8), from the frozen by-gene Arm 2 train-centred fit on real binding. Reads decision.tsv; uses no test-chromosome quantity.
# usage: weights.py  (after summarise.py) -> adopted_weights.tsv
# elastic net: per TF, the effective coefficient of its promoter and distal binding columns (shared + the TF's own copy), and of each pair column the TF belongs to.
# a LightGBM or network adoption stops here: its attribution is declared by a further amendment first.
import os, json, numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2"
D = pd.read_csv(f"{O}/decision.tsv", sep="\t"); rows = []
for _, a in D[D.adopted].iterrows():
    kind, link = a.kind.rsplit("_", 1); os.environ["M7_LINK"] = link
    import importlib, m7; importlib.reload(m7)
    d = f"{O}/fits/{kind}_{link}/final/kd/gene/arm2_train/{a.model}_{a['mode']}_real"
    if a.model == "enet":
        c = np.load(f"{d}/model.npz")["coef"]; nb = m7.Feat(kind).base.shape[1]; ns = nb + 2 + m7.NP + (1 if a["mode"] == "both" else 0); k = nb + 2
        for t, tf in enumerate(m7.TESTABLE):
            h = dict(kind=a.kind, model=a.model, mode=a["mode"], tf=tf)
            rows += [dict(h, feature="promoter_binding", coef=c[nb] + c[ns + t * k + nb]), dict(h, feature="distal_binding", coef=c[nb + 1] + c[ns + t * k + nb + 1])]
            rows += [dict(h, feature=f"pair_{m7.PAIRS.a[p]}_{m7.PAIRS.b[p]}", coef=c[nb + 2 + p]) for p in m7.PAIR_OF[tf]]
    else:
        raise SystemExit(f"{a.kind}: a complex model was adopted; its attribution method is declared by a further amendment before it is run")
pd.DataFrame(rows).to_csv(f"{O}/adopted_weights.tsv", sep="\t", index=False, float_format="%.6f"); print(len(rows), "rows")
