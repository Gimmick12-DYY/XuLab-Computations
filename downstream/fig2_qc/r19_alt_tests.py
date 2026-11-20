#!/usr/bin/env python
"""Round 7 alternative own-TF agreement tests (spec: round7_alt/SPEC_alt.md, commit 3fcc271). Raw reads only.
1 pairwise discrimination AUROC; 2 motif-centred profiles (z vs pseudo-TFs); 3 FRiP-style log2 enrichment.
Outputs OUT/round7_alt/: summary.tsv, round7_alt_results.xlsx, m_*.tsv, figs/*.{png,pdf,svg}."""
import itertools
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

from fig2_common import DM_DEPTH, DM_SEEDS, FASTA, INP, MEME_DB, OUT, REF, ROOT, TFS, read_bed

O7 = OUT / "round7_alt"; (O7 / "figs").mkdir(parents=True, exist_ok=True); TMP = O7 / "tmp"; TMP.mkdir(exist_ok=True)
CH = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PEAKS = dict(REF); PEAKS["CTCF"] = ROOT / "downstream/tracks/ctcf_bulk_sources/1_ENCODE_HEK293_ENCFF314ZAL_peaks.bed"
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
MOTIF = {"CTCF": "CTCF_HUMAN.H11MO.0.A", "MAZ": "MAZ_HUMAN.H11MO.0.A", "ZBTB7A": "ZBT7A_HUMAN.H11MO.0.A",
         "ZIC2": "ZIC2_HUMAN.H11MO.0.D", "ZNF282": "ZN282_HUMAN.H11MO.0.D", "ZNF777": None}
tf79 = [str(t) for t in np.load(INP / "frame.npz", allow_pickle=True)["tf79"]]


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def load5(t):
    z = np.load(INP / "five" / f"{t}.npz")
    return {c: z[c] for c in z.files if c in CH}


five = {t: load5(t) for t in TFS}
pool = {}
for t in tf79:
    for c, v in load5(t).items():
        pool.setdefault(c, []).append(v)
pool = {c: np.sort(np.concatenate(v)) for c, v in pool.items()}
ptot = sum(len(v) for v in pool.values())
tot = {t: sum(len(v) for v in five[t].values()) for t in TFS}


def thin(d, n_keep, seed):
    rng = np.random.default_rng(seed); tot_ = sum(len(v) for v in d.values()); p = min(1.0, n_keep / tot_)
    return {c: v[rng.random(len(v)) < p] for c, v in d.items()}


thinned = {t: [thin(five[t], DM_DEPTH, s) for s in DM_SEEDS] for t in TFS}


def count(d, chrom, lo, hi):
    out = np.zeros(len(chrom))
    for c in np.unique(chrom):
        m = np.flatnonzero(chrom == c); p = d.get(c)
        if p is None:
            continue
        out[m] = np.searchsorted(p, hi[m]) - np.searchsorted(p, lo[m])
    return out


PERM5 = np.array(list(itertools.permutations(range(5))))
PERM6 = np.array(list(itertools.permutations(range(6))))


def stats(M):
    off = ~np.eye(6, dtype=bool); st = lambda A: np.nanmean(np.diag(A)) - np.nanmean(A[off])
    obs = st(M); null = np.array([st(M[p]) for p in PERM6])
    M5 = M[1:, 1:]; off5 = ~np.eye(5, dtype=bool); st5 = lambda A: np.nanmean(np.diag(A)) - np.nanmean(A[off5])
    null5 = np.array([st5(M5[p]) for p in PERM5])
    return dict(diag_minus_off=float(obs), perm_p=float(np.mean(null >= obs - 1e-12)),
                noCTCF_perm_p=float(np.mean(null5 >= st5(M5) - 1e-12)),
                own_best_rows=int(sum(np.nanargmax(M[i]) == i for i in range(6))),
                own_best_cols=int(sum(np.nanargmax(M[:, j]) == j for j in range(6))))


def chrnum(c):
    return np.array([23 if x == "chrX" else int(x[3:]) for x in c])


HALF = {"all": lambda c: np.ones(len(c), bool), "odd": lambda c: (chrnum(c) % 2 == 1) & (chrnum(c) != 23),
        "even_X": lambda c: (chrnum(c) % 2 == 0) | (chrnum(c) == 23)}
rows, mats = [], {}

# ---------------- 1. pairwise AUROC ----------------
pk = {}
for t in TFS:
    P = pd.read_csv(PEAKS[t], sep="\t", header=None, usecols=range(10)); P = P[P[0].isin(CH)]
    P = P.assign(summit=P[1] + P[9].clip(lower=0)).sort_values(6, ascending=False).reset_index(drop=True)
    pk[t] = P


def far_from(P, Q, d=500):
    """summits of P more than d bp from every interval of Q"""
    ok = np.ones(len(P), bool)
    for c in P[0].unique():
        m = np.flatnonzero(P[0].values == c); q = Q[Q[0] == c]
        if not len(q):
            continue
        o = np.argsort(q[1].values); s_, e_ = q[1].values[o], np.maximum.accumulate(q[2].values[o])
        x = P.summit.values[m]
        j = np.searchsorted(s_, x + d, side="right") - 1
        ok[m] = ~((j >= 0) & (e_[np.clip(j, 0, None)] >= x - d))
    return ok


def score_tables(P):
    c, x = P[0].values, P.summit.values
    lo, hi = x - 250, x + 251
    pl = count(pool, c, lo, hi)
    full = {t: count(five[t], c, lo, hi) for t in TFS}
    thn = {t: [count(d, c, lo, hi) for d in thinned[t]] for t in TFS}
    sc_full = {t: np.log2((full[t] + 0.5) / (pl - full[t] + 0.5)) for t in TFS}
    sc_thn = {t: [np.log2((v + 0.5) / (pl - full[t] + 0.5)) for v in thn[t]] for t in TFS}
    return sc_full, sc_thn


pair = {}
for A, B in itertools.permutations(TFS, 2):
    a = pk[A][far_from(pk[A], pk[B])].head(2000); b = pk[B][far_from(pk[B], pk[A])].head(2000)
    pair[(A, B)] = (a, b, score_tables(a), score_tables(b))
    print("pair", A, B, len(a), len(b), flush=True)
for depth in ("full", "znf777"):
    for half in HALF:
        M = np.zeros((6, 6))
        for i, X in enumerate(TFS):
            for j, A in enumerate(TFS):
                au = []
                for B in TFS:
                    if B == A:
                        continue
                    a, b, (fa, ta), (fb, tb) = pair[(A, B)]
                    ma, mb = HALF[half](a[0].values), HALF[half](b[0].values)
                    y = np.r_[np.ones(ma.sum()), np.zeros(mb.sum())]
                    if depth == "full":
                        au.append(roc_auc_score(y, np.r_[fa[X][ma], fb[X][mb]]))
                    else:
                        au.append(np.mean([roc_auc_score(y, np.r_[ta[X][k][ma], tb[X][k][mb]]) for k in range(len(DM_SEEDS))]))
                M[i, j] = np.mean(au)
        mats[("AUROC", depth, half)] = M
        rows.append(dict(test="1 pairwise AUROC", depth=depth, half=half, **stats(M), **{f"own_{t}": M[k, k] for k, t in enumerate(TFS)}))
        print(rows[-1], flush=True)

# ---------------- 3. FRiP-style ----------------
# reference-unique peaks recomputed from the peak files used here (inputs/unique_<TF>.bed used the CTCF majority set,
# not ENCFF314ZAL): A peaks overlapping no peak of the other 5 references (bedtools intersect -v). Spec note below.
uq = {}
for t in TFS:
    pk[t][[0, 1, 2]].sort_values([0, 1]).to_csv(TMP / f"pk_{t}.bed", sep="\t", header=False, index=False)
for t in TFS:
    out = sh("bedtools", "intersect", "-v", "-a", str(TMP / f"pk_{t}.bed"), "-b", *[str(TMP / f"pk_{o}.bed") for o in TFS if o != t])
    uq[t] = pd.read_csv(pd.io.common.StringIO(out), sep="\t", header=None)
    print("unique peaks", t, len(uq[t]), "of", len(pk[t]), flush=True)
for half in HALF:
    M = np.zeros((6, 6))
    for j, A in enumerate(TFS):
        u = uq[A]; u = u[u[0].isin(CH)]; u = u[HALF[half](u[0].values)]
        c, s, e = u[0].values, u[1].values.astype(int), u[2].values.astype(int)
        pin = count(pool, c, s, e).sum()
        for i, X in enumerate(TFS):
            xin = count(five[X], c, s, e).sum()
            M[i, j] = np.log2((xin / tot[X]) / ((pin - xin) / (ptot - tot[X])))
    mats[("FRiP", "full", half)] = M
    rows.append(dict(test="3 FRiP-style log2 enrichment", depth="full", half=half, **stats(M), **{f"own_{t}": M[k, k] for k, t in enumerate(TFS)}))
    print(rows[-1], flush=True)

# ---------------- 2. motif-centred profiles ----------------
W2, ST2 = 500, 10
e2 = np.arange(-W2, W2 + 1, ST2); x2 = (e2[:-1] + e2[1:]) / 2
CT2 = np.abs(x2) <= 50; FL2 = (np.abs(x2) >= 300)


def unique_summits(t):
    u = uq[t]; key = set(zip(u[0], u[1].astype(int), u[2].astype(int)))
    P = pk[t]; m = np.array([k in key for k in zip(P[0], P[1].astype(int), P[2].astype(int))])
    return P[m].reset_index(drop=True)


def motif_file(t):
    if MOTIF[t] is not None:
        return str(MEME_DB), MOTIF[t]
    S = unique_summits(t).head(500)
    bed = TMP / f"streme_{t}.bed"
    pd.DataFrame({0: S[0], 1: S.summit - 50, 2: S.summit + 50}).to_csv(bed, sep="\t", header=False, index=False)
    fa = TMP / f"streme_{t}.fa"; fa.write_text(sh("bedtools", "getfasta", "-fi", str(FASTA), "-bed", str(bed)))
    od = TMP / f"streme_{t}"
    sh("streme", "--p", str(fa), "--dna", "--nmotifs", "3", "--oc", str(od))
    first = [l for l in open(od / "streme.txt") if l.startswith("MOTIF")][0].split()[1]
    print(t, "STREME motif 1:", first, flush=True)
    return str(od / "streme.txt"), first


def anchors(t):
    S = unique_summits(t)
    reg = pd.DataFrame({0: S[0].values, 1: (S.summit - 100).values, 2: (S.summit + 100).values, 3: [f"r{k}" for k in range(len(S))]})
    bed = TMP / f"motif_{t}.bed"; reg.to_csv(bed, sep="\t", header=False, index=False)
    fa = TMP / f"motif_{t}.fa"
    txt = sh("bedtools", "getfasta", "-fi", str(FASTA), "-bed", str(bed), "-name")
    fa.write_text("".join((l.split("::")[0] if l.startswith(">") else l) + "\n" for l in txt.splitlines()))
    db, mid = motif_file(t)
    out = sh("fimo", "--text", "--thresh", "1e-5", "--motif", mid, db, str(fa))
    h = pd.read_csv(pd.io.common.StringIO(out), sep="\t")
    h = h.sort_values("p-value").drop_duplicates("sequence_name")
    k = h.sequence_name.astype(str).str[1:].astype(int).values
    cen = reg[1].values[k] + (h.start.values + h.stop.values) // 2 - 1
    return pd.DataFrame({"chrom": reg[0].values[k], "pos": cen, "strand": h.strand.values}), len(S), mid


def profile(d, A):
    h = np.zeros(len(x2))
    for c, g in A.groupby("chrom"):
        p = d.get(c)
        if p is None:
            continue
        lo = np.searchsorted(p, g.pos.values - W2); hi = np.searchsorted(p, g.pos.values + W2)
        for a, s, l, u in zip(g.pos.values, g.strand.values, lo, hi):
            rel = p[l:u] - a
            h += np.histogram(-rel if s == "-" else rel, e2)[0]
    return h


def lcf(h):
    return np.log2((h[CT2].mean() + 1e-9) / (h[FL2].mean() + 1e-9))


rng = np.random.default_rng(7)
pool_flat = {c: v for c, v in pool.items()}
pseudo = []
for k in range(50):
    pseudo.append({c: v[rng.random(len(v)) < DM_DEPTH / ptot] for c, v in pool_flat.items()})
Mz = np.zeros((6, 6)); Mraw = np.zeros((6, 6)); motif_info = []
for j, A in enumerate(TFS):
    anc, nuniq, mid = anchors(A)
    motif_info.append(dict(reference=A, motif=mid, unique_peaks=nuniq, motif_sites=len(anc)))
    print(motif_info[-1], flush=True)
    null = np.array([lcf(profile(p, anc)) for p in pseudo])
    for i, X in enumerate(TFS):
        v = np.mean([lcf(profile(d, anc)) for d in thinned[X]])
        Mraw[i, j] = v; Mz[i, j] = (v - null.mean()) / (null.std() + 1e-12)
mats[("motif_z", "znf777", "all")] = Mz; mats[("motif_log2cf", "znf777", "all")] = Mraw
rows.append(dict(test="2 motif-centred profile z vs pseudo-TFs", depth="znf777", half="all", **stats(Mz), **{f"own_{t}": Mz[k, k] for k, t in enumerate(TFS)}))
print(rows[-1], flush=True)

# ---------------- save ----------------
S = pd.DataFrame(rows); S.to_csv(O7 / "summary.tsv", sep="\t", index=False, float_format="%.4f")
with pd.ExcelWriter(O7 / "round7_alt_results.xlsx") as xw:
    S.to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame(motif_info).to_excel(xw, sheet_name="motif_sites", index=False)
    for (tst, dp, hf), M in mats.items():
        df = pd.DataFrame(M, [f"{t} sc" for t in TFS], [f"{t} ChIP ({CELL[t]})" for t in TFS])
        df.to_csv(O7 / f"m_{tst}_{dp}_{hf}.tsv", sep="\t", float_format="%.4f")
        df.to_excel(xw, sheet_name=f"{tst}_{dp}_{hf}"[:31])
pd.set_option("display.width", 250); print(S.round(3).to_string())

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402
P = rp.use()
plt.rcParams.update({"font.sans-serif": ["Arial"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42})
MM = 1 / 25.4
PLOTS = [(("AUROC", "znf777", "all"), "Pairwise discrimination AUROC\n(ZNF777 depth)", 0.5, "AUROC"),
         (("AUROC", "full", "all"), "Pairwise discrimination AUROC\n(full depth)", 0.5, "AUROC"),
         (("motif_z", "znf777", "all"), "Motif-centred profile\n(z vs pseudo-TFs, ZNF777 depth)", 0.0, "z"),
         (("FRiP", "full", "all"), "Reads in reference-unique peaks\n(log2 vs pool)", 0.0, "log2 enrichment")]
for key, ttl, centre, unit in PLOTS:
    M = mats[key]; st = [r for r in rows if r["test"].split()[0] == {"AUROC": "1", "motif_z": "2", "FRiP": "3"}[key[0]]
                         and r["depth"] == key[1] and r["half"] == key[2]][0]
    lim = float(np.nanmax(np.abs(M - centre))) or 1
    fig, ax = plt.subplots(figsize=(70 * MM, 66 * MM), layout="constrained")
    im = ax.imshow(M, cmap=P.div_cmap(), vmin=centre - lim, vmax=centre + lim)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j] - centre) > 0.65 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=6)
    ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=6); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{ttl}\nown best {st['own_best_rows']}/6, p = {st['perm_p']:.3f} (no CTCF {st['noCTCF_perm_p']:.3f})", fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(unit, fontsize=6); cb.ax.tick_params(labelsize=6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(O7 / "figs" / f"{key[0]}_{key[1]}.{ext}", dpi=600)
    plt.close(fig)
print("done")
