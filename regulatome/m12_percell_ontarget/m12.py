# m12_percell_ontarget (SPEC.md). Diagnostic, no Perturb-seq. usage: m12.py <TF> -> out/<TF>_cells.tsv, out/<TF>_summary.json, out/<TF>_heldout.tsv, out/<TF>_library.tsv
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m12_percell_ontarget"; OUT = f"{O}/out"; os.makedirs(OUT, exist_ok=True); V = "/vast/som/xujie_lab/TAGATG-293T"
LAB = "diagnostic, no Perturb-seq"; SEED = 20261111; rng = np.random.default_rng(SEED); t = sys.argv[1]; SW = {"NRF1": "ZNF143", "ZNF143": "NRF1", "CTCF": "NFYA", "NFYA": "CTCF"}[t]
EVEN = {f"chr{i}" for i in range(2, 23, 2)} | {"chrX"}
def merged(c, s, e):
    out = {}
    for ch in np.unique(c):
        m = c == ch; iv = np.column_stack([s[m], e[m]]); iv = iv[np.argsort(iv[:, 0])]; ms = []
        for a, b in iv:
            if ms and a <= ms[-1][1]: ms[-1][1] = max(ms[-1][1], b)
            else: ms.append([a, b])
        out[ch] = np.array(ms)
    return out
def hit(IV, c, p):
    r = np.zeros(len(p), bool)
    for ch in np.unique(c):
        if ch not in IV: continue
        m = c == ch; iv = IV[ch]; k = np.searchsorted(iv[:, 0], p[m], side="right") - 1; ok = k >= 0; ok[ok] = p[m][ok] < iv[k[ok], 1]; r[m] = ok
    return r
Z = np.load(f"{RG}/m10_promoter_regulome/work/g/lists.npz", allow_pickle=True)
def site_sets(nm):
    c, p = Z[f"{nm}|chr"].astype(str), Z[f"{nm}|pos"]; bc, bp = Z[f"{nm}__bg|chr"].astype(str), Z[f"{nm}__bg|pos"]; ev, bev = np.isin(c, list(EVEN)), np.isin(bc, list(EVEN)); S = {}
    for half, m, bm in (("all", np.ones(len(p), bool), np.ones(len(bp), bool)), ("odd", ~ev, ~bev), ("even", ev, bev)):
        S[half] = (merged(c[m], p[m] - 100, p[m] + 100), int(m.sum()), merged(bc[bm], bp[bm] - 100, bp[bm] + 100), int(bm.sum()))
    return S
LISTS = {"motif": site_sets(f"motif_{t}")}
if t == "CTCF": LISTS["chip_HEK293"] = site_sets("chip_CTCF_HEK293")
UA = pd.read_csv(f"{X}/data/HEK293T_ATAC_union3.bed.gz", sep="\t", header=None, usecols=[0, 1, 2]); UAIV = merged(UA[0].values.astype(str), UA[1].values, UA[2].values)
HB = pd.read_csv(f"{RG}/m11_peak_regulome/recall/hot_bins.bed", sep="\t", header=None); HIV = merged(HB[0].values.astype(str), HB[1].values, HB[2].values)
def qpeaks(tf):
    P = pd.read_csv(f"{V}/bigwig/macs2/TF.{tf}/TF.{tf}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 2, 8]); P = P[P[8] >= -np.log10(0.05)]; return merged(P[0].values.astype(str), P[1].values, P[2].values)
CELLS = pd.read_csv(f"{V}/combined/Cell1000/Final_DNA_cells_TF1000cells_qc20_ForSubset.csv")
def per_cell(tf, sites_from_tf_lists, peaks):
    F = pd.read_csv(f"{V}/bigwig/fragment/TF.{tf}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 3, 5], names=["c", "s", "e", "bc", "st"], dtype={"c": str, "s": np.int64, "e": np.int64, "bc": str, "st": str})
    c = F.c.values; p = np.where(F.st.values == "+", F.s.values, F.e.values - 1); bc = F.bc.values; ua = hit(UAIV, c, p); ev = np.isin(c, list(EVEN)); D = pd.DataFrame(dict(bc=bc, total=1, open=ua.astype(int), open_odd=(ua & ~ev).astype(int), open_even=(ua & ev).astype(int), hot=hit(HIV, c, p).astype(int), in_q005=hit(peaks, c, p).astype(int)))
    for ln, S in sites_from_tf_lists.items():
        for half in ("all", "odd", "even"):
            iv, n, biv, bn = S[half]; D[f"{ln}_own_{half}"] = hit(iv, c, p).astype(int); D[f"{ln}_bg_{half}"] = hit(biv, c, p).astype(int)
    G = D.groupby("bc").sum(); G["library"] = [b.rsplit(":", 3)[0] if b.count(":") >= 3 else b.split(":")[0] for b in G.index]; return G
G = per_cell(t, LISTS, qpeaks(t)); Gs = per_cell(SW, LISTS, qpeaks(SW))
ncell_table = int((CELLS.TF == t).sum()); summ = dict(label=LAB, tf=t, swap_tf=SW, n_cells_fragments=len(G), n_cells_in_table=ncell_table, n_cells_swap=len(Gs), fragments=int(G.total.sum()), frac_open=float(G.open.sum() / G.total.sum()), frac_hot=float(G.hot.sum() / G.total.sum()), frac_in_q005=float(G.in_q005.sum() / G.total.sum()))
HO = []; LIBR = []
for ln, S in LISTS.items():
    nS, nB = S["all"][1], S["all"][3]
    def enr(H, half="all"): n_, b_ = S[half][1], S[half][3]; return (H[f"{ln}_own_{half}"].sum() / n_) / max(H[f"{ln}_bg_{half}"].sum() / b_, 1e-12)
    for nm, H in (("own", G), ("swap", Gs)):
        o = H[f"{ln}_own_all"].values.astype(float); n = H.open.values.astype(float); prate = o.sum() / n.sum(); e = n * prate; m = e > 0
        phi = float(((o[m] - e[m]) ** 2 / e[m]).sum() / (m.sum() - 1)); sims = [float(((s - e[m]) ** 2 / e[m]).sum() / (m.sum() - 1)) for s in (rng.binomial(n[m].astype(int), prate) for _ in range(1000))]
        summ[f"{ln}_{nm}_pooled_enrichment"] = float(enr(H)); summ[f"{ln}_{nm}_frac_fragments_on_sites"] = float(o.sum() / H.total.sum()); summ[f"{ln}_{nm}_dispersion_phi"] = phi; summ[f"{ln}_{nm}_phi_null_mean"] = float(np.mean(sims)); summ[f"{ln}_{nm}_phi_null_p97_5"] = float(np.percentile(sims, 97.5)); summ[f"{ln}_{nm}_phi_p"] = float((1 + sum(x >= phi for x in sims)) / 1001)
        # held-out ranking: select on odd, measure on even
        sel = (H[f"{ln}_own_odd"] / H.open_odd.clip(lower=1)).values; order = np.lexsort((rng.random(len(H)), -sel))
        for f in (.10, .25, .50):
            k = max(1, int(round(len(H) * f))); top = np.zeros(len(H), bool); top[order[:k]] = True
            def ev_(ix): Hh = H.iloc[ix]; return enr(Hh, "even")
            bt = [ev_(np.flatnonzero(top)[rng.integers(0, top.sum(), top.sum())]) for _ in range(1000)]; br = [ev_(np.flatnonzero(~top)[rng.integers(0, (~top).sum(), (~top).sum())]) for _ in range(1000)]
            HO.append(dict(label=LAB, tf=t, sites=ln, cells=nm if nm == "own" else f"swap_{SW}", top_frac=f, n_top=int(top.sum()), fragments_top=int(H.total.values[top].sum()), heldout_enrichment_top=float(enr(H.iloc[np.flatnonzero(top)], "even")), top_lo=float(np.percentile(bt, 2.5)), top_hi=float(np.percentile(bt, 97.5)),
                           heldout_enrichment_rest=float(enr(H.iloc[np.flatnonzero(~top)], "even")), rest_lo=float(np.percentile(br, 2.5)), rest_hi=float(np.percentile(br, 97.5))))
    # per-cell enrichment proxy vs depth; library structure
    H = G; pe = ((H[f"{ln}_own_all"] / nS) / ((H[f"{ln}_bg_all"] + 0.5) / nB)).values; summ[f"{ln}_spearman_cellenrichment_vs_fragments"] = float(spearmanr(pe, H.total.values)[0])
    for lib, Hl in H.groupby("library"):
        if len(Hl) < 5: continue
        b = [enr(Hl.iloc[rng.integers(0, len(Hl), len(Hl))]) for _ in range(500)]
        LIBR.append(dict(label=LAB, tf=t, sites=ln, library=lib, n_cells=len(Hl), fragments=int(Hl.total.sum()), enrichment=float(enr(Hl)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), frac_hot=float(Hl.hot.sum() / Hl.total.sum()), frac_in_q005=float(Hl.in_q005.sum() / Hl.total.sum())))
G.to_csv(f"{OUT}/{t}_cells.tsv", sep="\t"); json.dump(summ, open(f"{OUT}/{t}_summary.json", "w"), indent=1); pd.DataFrame(HO).to_csv(f"{OUT}/{t}_heldout.tsv", sep="\t", index=False, float_format="%.4f"); pd.DataFrame(LIBR).to_csv(f"{OUT}/{t}_library.tsv", sep="\t", index=False, float_format="%.4f")
print(json.dumps(summ, indent=1))
