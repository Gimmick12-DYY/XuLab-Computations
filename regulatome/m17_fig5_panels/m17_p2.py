# m17_fig5_panels P2 (SPEC.md): ML half of panel b from saved m7_step2 per_tf_test.tsv only (post hoc subset of an already scored test). No refit, no new scoring.
# usage: m17_p2.py -> out/p2_per_tf.tsv, out/p2_pooled.tsv
import numpy as np, pandas as pd
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; OUT = f"{RG}/m17_fig5_panels/out"; LAB = "diagnostic, benchmark already read; post hoc subset of an already scored test"; rng = np.random.default_rng(20261111)
STRONG = ["CTCF", "DNMT1", "E4F1", "MEF2A", "NRF1", "PA2G4", "RBBP7", "RBPJ", "SAFB", "SETDB1"]
T = pd.read_csv(f"{RG}/m7_step2/per_tf_test.tsv", sep="\t"); T = T[T.kind.isin(["raw_hicfixed", "imp_hicfixed"]) & (T.arm == "arm2") & T.tf.isin(STRONG)].copy()
T["gain"] = T.real - T.base
P = T[["kind", "split", "centring", "model", "mode", "tf", "real", "base", "gain", "ref_win100k", "ref_v8", "ceiling", "cells", "fold", "n_test_genes"]].sort_values(["kind", "split", "centring", "tf", "model", "mode"]); P.insert(0, "label", LAB)
P.to_csv(f"{OUT}/p2_per_tf.tsv", sep="\t", index=False, float_format="%.4f")
def boot(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1) if len(v) else np.array([np.nan]); return float(v.mean()) if len(v) else np.nan, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
PO = []
for k, G in T.groupby(["kind", "split", "centring", "model", "mode"]):
    g, glo, ghi = boot(G.gain); a, alo, ahi = boot(G.real - G.ref_win100k); v8, vlo, vhi = boot(G.real - G.ref_v8)
    PO.append(dict(label=LAB, kind=k[0], split=k[1], centring=k[2], model=k[3], mode=k[4], n_tfs=G.tf.nunique(), tfs=",".join(sorted(G.tf)), mean_real=G.real.mean(), mean_base=G.base.mean(), mean_gain=g, gain_lo=glo, gain_hi=ghi,
                   mean_ref_win100k=G.ref_win100k.mean(), real_minus_win100k=a, w_lo=alo, w_hi=ahi, mean_ref_v8=G.ref_v8.mean(), real_minus_v8=v8, v_lo=vlo, v_hi=vhi))
PO = pd.DataFrame(PO); PO.to_csv(f"{OUT}/p2_pooled.tsv", sep="\t", index=False, float_format="%.4f")
print(PO[(PO.split == "gene")].drop(columns=["label", "tfs"]).round(3).to_string(index=False))
W = T[(T.kind == "raw_hicfixed") & (T.split == "gene") & (T.centring == "train")].pivot_table(index="tf", columns=["model", "mode"], values="real")
W["ref_win100k"] = T[(T.kind == "raw_hicfixed") & (T.split == "gene")].groupby("tf").ref_win100k.first(); W["ref_v8"] = T[(T.kind == "raw_hicfixed") & (T.split == "gene")].groupby("tf").ref_v8.first()
W.columns = ["_".join(c) if isinstance(c, tuple) else c for c in W.columns]; W.to_csv(f"{OUT}/p2_side_by_side_raw_bygene_train.tsv", sep="\t", float_format="%.4f"); print(W.round(3).to_string())
# Amendment 1: reliability column. ceiling = knockdown split-half reliability (m7); sqrt(reliability) = largest correlation any predictor could reach.
S = T[(T.kind == "raw_hicfixed") & (T.split == "gene") & (T.centring == "train")]; W = S.pivot_table(index="tf", columns=["model", "mode"], values="real"); W.columns = ["_".join(c) for c in W.columns]
c = S.groupby("tf").ceiling.first(); W.insert(0, "reliability_ceiling", c); W.insert(1, "sqrt_reliability_max_r", np.sqrt(c.clip(lower=0)))
W["best_model_real"] = W[[x for x in W.columns if x.split("_")[0] in ("enet", "lgb", "mlp")]].max(1); W["best_over_max_r"] = (W.best_model_real / W.sqrt_reliability_max_r).where(W.sqrt_reliability_max_r > 0)
W.insert(0, "label", LAB); W.reset_index().to_csv(f"{OUT}/p2_reliability.tsv", sep="\t", index=False, float_format="%.4f"); print(W.drop(columns="label").round(3).to_string())
