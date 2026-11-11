# (1) MLX-SREBF2 and E-box motifs: panel z per TF, generic status. (2) p-value distribution of the pair shared-motif statistic:
# stat = max over motifs of min(panel z_A, panel z_B) on discovery among motifs with min held-out z > 0; p = fraction of all 3,081 panel pairs with
# stat >= observed (empirical, pair-level). Computed with generic motifs excluded (as the suggestive list) and included. Output motif_valid/pair_pvalues.tsv, pair_pvalue_hist.png
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
X = "/work/users/d/y/dyy12/XuLab"; G = f"{X}/complex_screen/pair_motif_composite/motif_valid"; sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; Pc = rp.use()
Zd = pd.read_csv(f"{G}/scores_panel_disc.tsv", sep="\t", index_col=0); Zv = pd.read_csv(f"{G}/scores_panel_val.tsv", sep="\t", index_col=0); Fd = pd.read_csv(f"{G}/scores_fisher_disc.tsv", sep="\t", index_col=0)
alt = {}
for l in open(f"{X}/complex_screen/pair_motif_composite/db/known.meme"):
    if l.startswith("MOTIF "): x = l.split(); alt[x[1]] = x[2] if len(x) > 2 else x[1]
ntf = (Fd >= 3).sum(1); gen = ntf > 10
eb = [m for m in Zd.index if any(k in (m + "|" + alt[m]).upper() for k in ["MLX", "SREBF", "USF1", "USF2", "MAX.", "|MAX", "MNT", "MLXIP", "TFE3", "TFEB"])]
print("E-box family motifs:", len(eb), "generic:", int(gen[eb].sum()))
E = pd.DataFrame({"name": [alt[m] for m in eb], "n_tf_fisher_z3": ntf[eb], "generic": gen[eb], "MLX_fisher": Fd.loc[eb, "MLX"].round(2), "SREBF2_fisher": Fd.loc[eb, "SREBF2"].round(2),
                  "MLX_panel_d": Zd.loc[eb, "MLX"].round(2), "SREBF2_panel_d": Zd.loc[eb, "SREBF2"].round(2), "MLX_panel_v": Zv.loc[eb, "MLX"].round(2), "SREBF2_panel_v": Zv.loc[eb, "SREBF2"].round(2)}).sort_values("MLX_fisher", ascending=False)
pd.set_option("display.width", 250); print(E.head(15).to_string())
tfs = list(Zd.columns); allp = [(a, b) for i, a in enumerate(tfs) for b in tfs[i + 1:]]
def stat(a, b, keep):
    md = np.fmin(Zd[a].values, Zd[b].values); mv = np.fmin(Zv[a].values, Zv[b].values); s = keep & (mv > 0) & np.isfinite(md)
    return float(md[s].max()) if s.any() else -np.inf
A = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_all3081.tsv", sep="\t"); key = {tuple(sorted((a, b))): r for r, a, b in zip(A.itertuples(), A.tf_a, A.tf_b)}
rows = []
for a, b in allp:
    r = key.get(tuple(sorted((a, b)))); grp = "other"
    if r is not None:
        if str(r.candidate) == "True": grp = "candidate"
        elif str(r.known) == "True" and r.known_reason not in ("NEUROD4", "STAG2_class"): grp = "known"
    rows.append(dict(pair=f"{a}-{b}", group=grp, stat_nongeneric=stat(a, b, ~gen.values), stat_all=stat(a, b, np.ones(len(gen), bool)), best_pct=getattr(r, "best_pct", np.nan) if r is not None else np.nan,
                     af3=(r.af3_5seed_tier if r.af3_5seed_tier != "not_run" else r.af3_1seed_tier) if r is not None else ""))
T = pd.DataFrame(rows)
for c in ["stat_nongeneric", "stat_all"]: v = T[c].values; T[c.replace("stat", "p")] = [(np.sum(v >= x)) / len(v) for x in v]
T.to_csv(f"{G}/pair_pvalues.tsv", sep="\t", index=False)
for c in ["p_nongeneric", "p_all"]:
    print(c)
    for g, s in T.groupby("group"): print(f"  {g:10s} n={len(s):5d} frac p<0.05 {np.mean(s[c] < .05):.3f}  p<0.10 {np.mean(s[c] < .10):.3f}  median {s[c].median():.2f}")
    s = T[(T.group == "candidate") & T.af3.isin(["strong", "weak"])]; print(f"  cand AF3 s/w n={len(s)} frac p<0.05 {np.mean(s[c] < .05):.3f} median {s[c].median():.2f}")
print("MLX-SREBF2:", T[T.pair.isin(["MLX-SREBF2", "SREBF2-MLX"])].to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
for k, c in enumerate(["p_nongeneric", "p_all"]):
    for g, col in [("candidate", Pc.neutral.ink), ("known", Pc.evidence.hic_loop), ("other", Pc.neutral.light)]:
        s = T[T.group == g][c]; ax[k].hist(s, bins=np.linspace(0, 1, 21), density=True, histtype="step", color=col, lw=1, label=f"{g} (n={len(s)})")
    ax[k].axhline(1, color=Pc.neutral.mid, lw=.5, ls="--"); ax[k].set_xlabel("pair p-value (vs all 3,081 panel pairs)"); ax[k].set_ylabel("density")
    ax[k].set_title("generic motifs excluded" if k == 0 else "all motifs (incl. generic)")
ax[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(f"{G}/pair_pvalue_hist.png", dpi=200); fig.savefig(f"{G}/pair_pvalue_hist.svg")
