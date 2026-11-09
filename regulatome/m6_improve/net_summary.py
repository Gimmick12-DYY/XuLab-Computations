# SPEC_NETWORK summary of net1b/per_tf.tsv (dev TFs only). Per arm, with and without NRF1, per base (win100k, abc) and alpha (0.15/0.3/0.5/0.7/rel):
# ceiling-weighted specificity (b3b4 weights, bootstrap over TFs 10,000, seed 20261116) for graph_knn5 (primary), graph_dense, uniform, the 5 kNN5
# swap draws (min/median/max) and 2 dense permutations; weighted paired deltas of graph_knn5 vs alpha0, uniform and each swap draw (median, and
# fraction of draws with delta CI > 0); three-way contrast (own / kNN5 partner / non-partner z, partner - nonpartner with CI) per variant; gain over
# alpha0 by ceiling tertile. Writes net1b/net_summary.tsv, net1b/contrast.tsv, net1b/tertile_gain.tsv. usage: net_summary.py
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/regulatome/m6_improve"; P = pd.read_csv(f"{O}/net1b/per_tf.tsv", sep="\t"); C = pd.read_csv(f"{O}/../m4_validation/b3b4/reliability_ceilings.tsv", sep="\t"); Sp = pd.read_csv(f"{O}/split_v1.tsv", sep="\t").set_index("tf")
AL = ["a0.15", "a0.3", "a0.5", "a0.7", "a_rel"]; S, CT, TG = [], [], []
for excl in (False, True):
    rng = np.random.default_rng(20261116)
    for a in ("arm2", "arm1"):
        G = P[(P.arm == a) & ~(excl & (P.tf == "NRF1"))]; w = C[C.arm == a].groupby("panel_tf").weight.mean()
        piv = lambda col: G.pivot(index="tf", columns="variant", values=col)
        Wsp = piv("specificity").dropna(); idx = Wsp.index; wt = w.reindex(idx).fillna(0).values; ix = rng.integers(0, len(idx), (10000, len(idx)))
        f = lambda e: (e[ix] * wt[ix]).sum(1) / np.maximum(wt[ix].sum(1), 1e-12)
        def wd(e):
            b = f(e); return round(np.average(e, weights=wt), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)
        Zo, Zp, Zn = piv("z_own").loc[idx], piv("z_partner").loc[idx], piv("z_nonpartner").loc[idx]
        tag = dict(nrf1="excluded" if excl else "included", arm=a, n_tfs=len(idx), n_weighted=int((wt > 0).sum()))
        for b in ("win100k", "abc"):
            base = Wsp[f"{b}|alpha0"].values; m0 = wd(base); S.append(dict(**tag, base=b, alpha="0", variant="alpha0", wspec=m0[0], ci=f"[{m0[1]}, {m0[2]}]"))
            for al in AL:
                g = Wsp[f"{b}|graph_knn5|{al}"].values; row = dict(**tag, base=b, alpha=al)
                for nm in ("graph_knn5", "graph_dense", "uniform", "shufdense0", "shufdense1"):
                    m = wd(Wsp[f"{b}|{nm}|{al}"].values); row[nm] = m[0]; row[nm + "_ci"] = f"[{m[1]}, {m[2]}]"
                sh = [wd(Wsp[f"{b}|shufknn{k}|{al}"].values)[0] for k in range(5)]; row.update(shufknn_min=min(sh), shufknn_med=float(np.median(sh)), shufknn_max=max(sh))
                for nm, ref in (("d_vs_alpha0", base), ("d_vs_uniform", Wsp[f"{b}|uniform|{al}"].values)):
                    m = wd(g - ref); row[nm] = m[0]; row[nm + "_ci"] = f"[{m[1]}, {m[2]}]"
                dd = [wd(g - Wsp[f"{b}|shufknn{k}|{al}"].values) for k in range(5)]; row["d_vs_shufknn_med"] = float(np.median([d[0] for d in dd])); row["d_vs_shufknn_range"] = f"{min(d[0] for d in dd)}..{max(d[0] for d in dd)}"; row["n_draws_d_ci_gt0"] = sum(d[1] > 0 for d in dd)
                md = wd(Wsp[f"{b}|graph_dense|{al}"].values - Wsp[f"{b}|uniform|{al}"].values); row["dense_d_vs_uniform"] = md[0]; row["dense_d_vs_uniform_ci"] = f"[{md[1]}, {md[2]}]"
                S.append(row)
            for v in [f"{b}|alpha0"] + [f"{b}|{g_}|{al}" for g_ in ("graph_knn5", "uniform", "shufknn0", "graph_dense") for al in AL]:
                if v not in Zo: continue
                pn = (Zp[v] - Zn[v]).values; ok = ~np.isnan(pn); m = wd(np.where(ok, pn, 0)) if ok.all() else (np.nan, np.nan, np.nan)
                CT.append(dict(**tag, variant=v, z_own=round(np.average(Zo[v], weights=wt), 3), z_partner=round(np.nanmean(Zp[v]), 3), z_nonpartner=round(np.average(Zn[v], weights=wt), 3), partner_minus_nonpartner=m[0], pmn_ci=f"[{m[1]}, {m[2]}]", own_minus_nonpartner=round(np.average(Zo[v] - Zn[v], weights=wt), 3)))
            for al in AL:
                d = Wsp[f"{b}|graph_knn5|{al}"] - Wsp[f"{b}|alpha0"]; du = Wsp[f"{b}|graph_knn5|{al}"] - Wsp[f"{b}|uniform|{al}"]
                for st, D_ in pd.DataFrame(dict(d=d, du=du, st=Sp.stratum.reindex(idx))).groupby("st"): TG.append(dict(**tag, base=b, alpha=al, stratum=st, n=len(D_), gain_vs_alpha0=round(D_.d.mean(), 3), gain_vs_uniform=round(D_.du.mean(), 3)))
S = pd.DataFrame(S); S.to_csv(f"{O}/net1b/net_summary.tsv", sep="\t", index=False); pd.DataFrame(CT).to_csv(f"{O}/net1b/contrast.tsv", sep="\t", index=False); pd.DataFrame(TG).to_csv(f"{O}/net1b/tertile_gain.tsv", sep="\t", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
print(S.to_string(index=False)); print(pd.DataFrame(CT).to_string(index=False)); print(pd.DataFrame(TG).to_string(index=False))
