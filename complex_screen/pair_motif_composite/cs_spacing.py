# CAP-SELEX-style spacing / orientation test (2026-11-16, user go). For pairs where both TFs have an own DB motif (motif_valid fisher best_own) and
# >= 30 discovery / >= 15 held-out co-bound sites: scan co-bound sites (A summit +/- 150 bp) and the single-bound pool (A-only + B-only, same
# window) with FIMO (p <= 1e-4) for motif A and motif B; every A-hit x B-hit pair gives a gap (bp between the two motif edges, 0-20; overlaps
# excluded) and a relative orientation (same strand / opposite). Per (gap, orientation) bin: share of sites with >= 1 such pair, co-bound vs
# single-bound, one-sided Fisher; global chi-square over bins. Call = a bin with fold > 1 and p <= 0.05 on BOTH halves. Within-pair nulls
# (__null1/2 pseudo co-bound sets) run identically. Output composite_gapped/capselex_spacing_{bins,pairs}.tsv
import os, subprocess, numpy as np, pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/composite_gapped"; W = f"{G}/spacing_tmp"; os.makedirs(W, exist_ok=True)
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-800:]); return r.stdout
P = pd.read_csv(f"{G}/pairs.tsv", sep="\t"); OW = pd.read_csv(f"{O}/motif_valid/own_gate.tsv", sep="\t"); own = OW[OW.method == "fisher"].set_index("tf").best_own
base = P[(P.set != "null_within") & P.tf_a.isin(own.index) & P.tf_b.isin(own.index) & (P.disc_cob >= 30) & (P.val_cob >= 15)]
todo = list(base.pair) + [f"{p}__null{k}" for p in base.pair for k in (1, 2) if os.path.isdir(f"{G}/pools/{p}__null{k}")]
def seqs(bed, w, tag):
    d = pd.read_csv(bed, sep="\t", header=None, usecols=[0, 1, 2]); m = (d[1] + d[2]) // 2; d = pd.DataFrame({"c": d[0], "s": m - w, "e": m + w}); d["n"] = [f"{tag}{i}" for i in range(len(d))]
    f = f"{W}/{tag}.bed"; d.to_csv(f, sep="\t", header=False, index=False); sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -fi {X}/downstream/cache/hg38.fa -bed {f} -fo {W}/{tag}.fa"); return list(d.n)
bins, prs = [], []
for pair in todo:
    a, b = pair.split("__")[0].split("-"); ma, mb = own[a], own[b]
    sh(f"module load meme/5.5.7 >/dev/null 2>&1; meme-get-motif -id {ma} -id {mb} {O}/db/known.meme > {W}/ab.meme")
    row = dict(pair=pair, set=P.set_index("pair").at[pair, "set"], tf_a=a, tf_b=b, motif_a=ma, motif_b=mb)
    for pt in ["disc", "val"]:
        cob = seqs(f"{G}/pools/{pair}/{pt}/cob.bed", 150, f"c_{pt}_")
        sb = seqs(f"{G}/pools/{pair}/{pt}/bg.bed", 150, f"s_{pt}_")
        sh(f"cat {W}/c_{pt}_.fa {W}/s_{pt}_.fa > {W}/all.fa")
        H = pd.read_csv(pd.io.common.StringIO(sh(f"module load meme/5.5.7 >/dev/null 2>&1; fimo --thresh 1e-4 --text --skip-matched-sequence --verbosity 1 {W}/ab.meme {W}/all.fa")), sep="\t")
        H.columns = [c.strip("#").strip() for c in H.columns]; H = H[H.motif_id.isin([ma, mb])]
        cnt = {"c": {}, "s": {}}; n = {"c": len(cob), "s": len(sb)}
        for sid, g in H.groupby("sequence_name"):
            A_, B_ = g[g.motif_id == ma], g[g.motif_id == mb]; hit = set()
            for x in A_.itertuples():
                for y in B_.itertuples():
                    if x.start == y.start and ma == mb: continue
                    lo, hi = (x, y) if x.start < y.start else (y, x); gap = hi.start - lo.stop - 1
                    if 0 <= gap <= 20: hit.add((gap, "same" if x.strand == y.strand else "opposite"))
            grp = "c" if sid.startswith("c_") else "s"
            for h in hit: cnt[grp][h] = cnt[grp].get(h, 0) + 1
        keys = sorted(set(cnt["c"]) | set(cnt["s"])); tab = []
        for k in keys:
            c1, s1 = cnt["c"].get(k, 0), cnt["s"].get(k, 0); fo = ((c1 + .5) / n["c"]) / ((s1 + .5) / max(n["s"], 1)); p = fisher_exact([[c1, n["c"] - c1], [s1, n["s"] - s1]], alternative="greater")[1]
            bins.append(dict(pair=pair, part=pt, gap=k[0], orientation=k[1], cob_sites=c1, cob_n=n["c"], single_sites=s1, single_n=n["s"], fold=round(fo, 2), p=p)); tab.append([c1, s1])
        if len(tab) >= 2 and np.array(tab).sum(0).min() > 0:
            chi = chi2_contingency(np.array(tab) + .5); row[f"{pt}_chi2_p"] = chi[1]
        row[f"{pt}_cob"] = n["c"]; row[f"{pt}_single"] = n["s"]
    prs.append(row); print(row, flush=True)
Bn = pd.DataFrame(bins); Bn.to_csv(f"{G}/capselex_spacing_bins.tsv", sep="\t", index=False)
ok = Bn[(Bn.fold > 1) & (Bn.p <= .05)]; both = ok.groupby(["pair", "gap", "orientation"]).part.nunique().reset_index(); both = both[both.part == 2]
Pr = pd.DataFrame(prs); Pr["n_bins_both_halves"] = Pr.pair.map(both.groupby("pair").size()).fillna(0).astype(int)
Pr["bins"] = Pr.pair.map(both.groupby("pair").apply(lambda g: "; ".join(f"gap {r.gap} {r.orientation}" for r in g.itertuples()))).fillna("")
Pr.to_csv(f"{G}/capselex_spacing_pairs.tsv", sep="\t", index=False); pd.set_option("display.width", 250); print(Pr.to_string(index=False))
