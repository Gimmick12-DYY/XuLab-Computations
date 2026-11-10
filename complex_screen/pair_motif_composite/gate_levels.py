# Strict and relaxed shared-level calls from a finished run (BASE folder with gate_summary.tsv and results/).
# strict : own motif fold >= 1.5 vs partner-only, HOMER p <= 1e-5 (as in summarize.py)
# relaxed: fold >= 1.3 vs partner-only, HOMER p <= 1e-3, permutation p <= 0.05 (only where permutations were run; pairs without
#          permutations are judged without this term, which makes their pass rate an upper bound), validation log2FC > 0 and P < 0.05
#          (own motif, pair vs singles on even chromosomes + X). A pair passes if either partner's own motif passes.
import os, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summarize as S
B = os.environ["BASE"]; G = pd.read_csv(f"{B}/gate_summary.tsv", sep="\t"); rows = []
for _, g in G.iterrows():
    k = S.known(f"{B}/results/{g.pair}/val_homer/knownResults.txt"); r = dict(pair=g.pair, category=g.category, n_pair=g.n_pair)
    for side, other in [("A", "B"), ("B", "A")]:
        m = g[f"own_{side}"]; f = g[f"{side}_fold_vs_partner_only"]; lp = g[f"{side}_log10p_vs_partner_only"]
        vf = vp = np.nan
        if k is not None and isinstance(m, str) and m != "none":
            h = [n for n in k.index if S.short(n) == m]
            if h: vf, vp = math.log2(k.loc[h[0]].fold), k.loc[h[0]].log10p
        pp = g.shared_perm_p if not pd.isna(g.shared_perm_p) else np.nan
        r.update({f"own_{side}": m, f"{side}_fold": f, f"{side}_log10p": lp, f"{side}_fold_own_only": g[f"{side}_fold_vs_own_only"], f"{side}_fold_input": g[f"{side}_fold_vs_input"],
                  f"{side}_val_log2fc": round(vf, 2) if not np.isnan(vf) else np.nan, f"{side}_val_log10p": round(vp, 2) if not np.isnan(vp) else np.nan,
                  f"{side}_strict": bool(f >= 1.5 and lp <= -5),
                  f"{side}_relaxed": bool(f >= 1.3 and lp <= -3 and (np.isnan(pp) or pp <= 0.05) and vf > 0 and vp < math.log10(0.05))})
    r.update(shared_perm_p=g.shared_perm_p, strict_pass=r["A_strict"] or r["B_strict"], relaxed_pass=r["A_relaxed"] or r["B_relaxed"],
             composite_n=g.composite_n, composite_n_validated=g.get("composite_n_validated", np.nan), streme_n_validated=g.get("streme_n_validated", np.nan), composite_best=g.composite_best)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{B}/gate_levels.tsv", sep="\t", index=False)
rnd = T[T.category.str.startswith("random")]
print(f"random pairs: strict {int(rnd.strict_pass.sum())}/{len(rnd)}, relaxed {int(rnd.relaxed_pass.sum())}/{len(rnd)}, composite>=1 validated {int((rnd.composite_n_validated > 0).sum())}/{len(rnd)}")
pd.set_option("display.width", 300); print(T.to_string())
