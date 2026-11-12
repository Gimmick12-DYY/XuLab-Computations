# m17_fig5_panels Fig 5d version B (SPEC Amendment 3): locus view + knockdown effect bars + bound-TF strip. Illustration on an already-read benchmark.
# usage: make_gene_tf_locus.py [GENE ...] -> out/gene_example_v2/fig5d_<gene>_locus.svg/.png and coverage_<gene>.tsv
import os, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch
import matplotlib.colors as mcolors
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4, "xtick.major.width": .4, "ytick.major.width": .4, "xtick.major.size": 1.5, "ytick.major.size": 1.5})
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; D4 = f"{RG}/m17_fig5_panels/out/gene_example"; OUT = f"{RG}/m17_fig5_panels/out/gene_example_v2"; os.makedirs(OUT, exist_ok=True)
V = "/vast/som/xujie_lab/TAGATG-293T"; TABIX = "/nas/longleaf/rhel9/apps/samtools/1.25/htslib/bin/tabix"; GTF = f"{X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
ATAC = f"{RG}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"; K27 = f"{RG}/m2b_abc_k27ac/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz"; QMIN = -np.log10(0.05); BIN = 50
DOWN, UP = "#009E73", "#CC79A7"; PROM_BAND, ENH_BAND = "#DCE8F5", "#FBE6CF"; MM = 1 / 25.4
GENES = sys.argv[1:] or ["PGAM1", "HSPB1"]
GT = pd.read_csv(f"{D4}/p4_gene_tf.tsv", sep="\t"); E = pd.read_csv(f"{D4}/p4_elements.tsv", sep="\t"); L = pd.read_csv(f"{D4}/p4_links.tsv", sep="\t").rename(columns={"ABC.Score": "abc"})
tfs79 = [l.strip() for l in open(f"{RG}/m1_binding/v2/tfs.txt") if l.strip()]
EL = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"])
def gene_model(g, tss):
    rows = [r.split("\t") for r in subprocess.run(f"zcat {GTF} | grep -w 'gene_name \"{g}\";'", shell=True, capture_output=True, text=True).stdout.strip().splitlines()]
    tid = lambda r: r[8].split('transcript_id "')[1].split('"')[0]; T = [r for r in rows if r[2] == "transcript"]
    b = min(T, key=lambda r: abs((int(r[3]) if r[6] == "+" else int(r[4])) - tss))
    return dict(chr=b[0], start=int(b[3]), end=int(b[4]), strand=b[6], exons=sorted((int(r[3]), int(r[4])) for r in rows if r[2] == "exon" and tid(r) == tid(b)), cds=sorted((int(r[3]), int(r[4])) for r in rows if r[2] == "CDS" and tid(r) == tid(b)))
# windows
WIN = {}
for g in GENES:
    Lg = L[L.TargetGene == g]; tss = int(Lg.TargetGeneTSS.iloc[0]); gm = gene_model(g, tss); far = Lg[(np.abs((Lg.start + Lg.end) / 2 - tss) > 200000)]
    near = Lg.drop(far.index); lo = min(near.start.min(), gm["start"]); hi = max(near.end.max(), gm["end"])
    if len(far): lo, hi = min(tss - 20000, gm["start"]), max(tss + 20000, gm["end"])
    pad = (hi - lo) * .1; WIN[g] = dict(chr=gm["chr"], lo=int(lo - pad), hi=int(hi + pad), tss=tss, gm=gm, links=Lg, far=len(far))
def bins(g): w = WIN[g]; return np.arange(w["lo"], w["hi"] + BIN, BIN)
# HITAG coverage: read 5' ends from raw per-TF fragment files, one pass over all 79 files
COV = {g: {} for g in GENES}; TOT = {}
for t in tfs79:
    f = f"{V}/bigwig/fragment/TF.{t}.bed.gz"
    if not os.path.exists(f): continue
    R = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); TOT[t] = len(R); p5 = np.where(R.st.values == "+", R.s.values, R.e.values - 1)
    for g in GENES:
        w = WIN[g]; m = (R.c.values == w["chr"]) & (p5 >= w["lo"]) & (p5 < w["hi"]); COV[g][t] = np.histogram(p5[m], bins=bins(g))[0].astype(float)
print("HITAG files read", len(TOT), flush=True)
def tagcov(path, g):
    w = WIN[g]; out = subprocess.run([TABIX, path, f"{w['chr']}:{w['lo']}-{w['hi']}"], capture_output=True, text=True).stdout.strip().splitlines()
    if not out: return np.zeros(len(bins(g)) - 1)
    a = np.array([l.split("\t") for l in out]); s, e, st = a[:, 1].astype(int), a[:, 2].astype(int), a[:, 5]; p5 = np.where(st == "+", s, e - 1); return np.histogram(p5, bins=bins(g))[0].astype(float)
def smooth(v, k=3): return np.convolve(v, np.ones(k) / k, mode="same")
def peaks(t, g):
    w = WIN[g]; f = f"{V}/bigwig/macs2/TF.{t}/TF.{t}_peaks.narrowPeak"
    P = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 8, 9]); P.columns = ["c", "s", "q", "off"]; P = P[(P.q >= QMIN) & (P.c == w["chr"])]; sm = P.s + P.off; return sm[(sm >= w["lo"]) & (sm < w["hi"])].values
def motifs(t, g):
    w = WIN[g]; out = []; f = f"{RG}/m10_promoter_regulome/work/g/sites_{t}.tsv"
    if not os.path.exists(f): return None
    S = pd.read_csv(f, sep="\t", header=None, names=["e", "a", "b"]); S = S[S.e.isin(np.flatnonzero((EL.chr.values == w["chr"]) & (EL.end.values > w["lo"]) & (EL.start.values < w["hi"])))]
    out += list(EL.start.values[S.e.values] + (S.a.values + S.b.values) // 2)
    P = pd.read_csv(f"{RG}/m10_promoter_regulome/work/g/psites_{t}.tsv", sep="\t", header=None, names=["gene", "a", "b"]); P = P[P.gene == g]; out += list(w["tss"] - 500 + (P.a.values + P.b.values) // 2)
    return np.unique(np.array(out, int))
for g in GENES:
    w = WIN[g]; H = GT[GT.gene == g].set_index("tf"); Eg = E[E.gene == g].sort_values("link_score", ascending=False).drop_duplicates("tf"); bound = set(Eg.tf)
    conf = [t for t in H.index if H.loc[t, "confirmed_K10000"]]; rows_tf = sorted([t for t in conf if t in bound], key=lambda t: H.loc[t, "z_all"]) + sorted([t for t in conf if t not in bound], key=lambda t: H.loc[t, "z_all"]); rows_tf = rows_tf[:6]
    x = bins(g)[:-1] + BIN / 2; atac = tagcov(ATAC, g); k27 = tagcov(K27, g)
    cpm = {t: smooth(COV[g][t] / TOT[t] * 1e6) for t in COV[g]}; others = [t for t in COV[g] if t not in rows_tf]
    pooled = smooth(np.sum([COV[g][t] for t in others], 0) / sum(TOT[t] for t in others) * 1e6)
    pd.DataFrame(dict(bin_start=bins(g)[:-1], atac=atac, h3k27ac=k27, pooled_other_cpm=pooled, **{f"cpm_{t}": cpm[t] for t in rows_tf})).to_csv(f"{OUT}/coverage_{g}.tsv", sep="\t", index=False, float_format="%.4g")
    # ---- layout (mm)
    nT = len(rows_tf); hts = [6, 5, 5] + [6] * nT + [4, 5]; gap = .8; top = 3; Wf = 120; trackW = 76; barW = 22; left = 15; BOT = 27
    Hf = top + sum(hts) + gap * len(hts) + BOT
    fig = plt.figure(figsize=(Wf * MM, Hf * MM)); yy = Hf - top; axs = []
    for h_ in hts:
        yy -= h_; axs.append(fig.add_axes([left / Wf, yy / Hf, trackW / Wf, h_ / Hf])); yy -= gap
    for a in axs:
        a.set_xlim(w["lo"], w["hi"]); a.set_xticks([]); [a.spines[s].set_visible(False) for s in ("top", "right", "bottom")]
        for r in w["links"].itertuples():
            if r.end > w["lo"] and r.start < w["hi"]: a.axvspan(r.start, r.end, color=PROM_BAND if r.self else ENH_BAND, lw=0, zorder=0)
    # arcs
    a = axs[0]; a.set_ylim(0, 1); a.set_yticks([]); a.spines["left"].set_visible(False); cmap = plt.get_cmap("Greys")
    for r in w["links"].itertuples():
        if r.self or not (r.end > w["lo"] and r.start < w["hi"]): continue
        xm = (r.start + r.end) / 2; c = cmap(.35 + .65 * min(r.abc / .12, 1)); a.add_patch(Arc(((xm + w["tss"]) / 2, 0), abs(xm - w["tss"]), 1.8, theta1=0, theta2=180, lw=.6, color=c))
    a.text(-.01, .5, "ABC links", transform=a.transAxes, ha="right", va="center")
    if w["far"]: a.text(1.0, .95, f"+{w['far']} linked elements > 200 kb not shown", transform=a.transAxes, ha="right", va="top", fontsize=5)
    def cov(a, v, lab, col, ymax=None):
        a.fill_between(x, 0, v, color=col, lw=0, step="mid"); ym = ymax or max(v.max(), 1e-9); a.set_ylim(0, ym); a.set_yticks([ym]); a.set_yticklabels([f"{ym:.0f}" if ym >= 10 else f"{ym:.1f}"], fontsize=5); a.text(-.01, .5, lab, transform=a.transAxes, ha="right", va="center", fontsize=6)
    cov(axs[1], atac, "ATAC", "#8C8C8C"); cov(axs[2], k27, "H3K27ac", "#8C8C8C")
    ymx = max([cpm[t].max() for t in rows_tf] + [1e-9]); ymx = float(f"{ymx:.2g}")
    for k, t in enumerate(rows_tf):
        a = axs[3 + k]; h = H.loc[t]; col = DOWN if h.z_all < 0 else UP; cov(a, cpm[t], t, "#4D4D4D", ymx)
        pk = peaks(t, g); mt = motifs(t, g); yl = a.get_ylim()[1]
        a.plot(pk, np.full(len(pk), -.12 * yl), "|", color="k", ms=3, mew=.6, clip_on=False)
        if mt is not None and len(mt): a.plot(mt, np.full(len(mt), -.3 * yl), "|", color="#B8860B", ms=2.5, mew=.6, clip_on=False)
        if t not in bound: a.text(1.0, .8, "no linked q005 peak", transform=a.transAxes, ha="right", fontsize=5, color="#555555")
        # effect bar
        b = fig.add_axes([(left + trackW + 4) / Wf, a.get_position().y0, barW / Wf, a.get_position().height]); zmax = max(abs(H.loc[rows_tf, "z_all"]).max(), 3) * 1.1
        b.barh(0, h.z_all, height=.55, color=col); b.plot([h.z_half1, h.z_half2], [0, 0], "o", ms=1.8, color="k", mfc="white", mew=.4); b.axvline(0, color="k", lw=.4)
        b.set_xlim(-zmax, zmax); b.set_ylim(-.6, .6); b.set_yticks([]); [b.spines[s].set_visible(False) for s in ("top", "right", "left")]
        if k < nT - 1: b.set_xticks([]); b.spines["bottom"].set_visible(False)
        else: b.set_xlabel("change on knockdown (z)", fontsize=6, labelpad=1)
        b.text(h.z_all + (.4 if h.z_all >= 0 else -.4), 0, f"{h.z_all:+.1f}", va="center", ha="left" if h.z_all >= 0 else "right", fontsize=5)
    if nT: axs[3].text(.005, 1.0, "HITAG, CPM (same scale for all TF rows)", transform=axs[3].transAxes, ha="left", va="top", fontsize=5, color="#555555")
    cov(axs[3 + nT], pooled, "other TFs", "#BDBDBD")
    # gene model
    a = axs[-1]; a.set_ylim(-1, 1); a.set_yticks([]); a.spines["left"].set_visible(False); gm = w["gm"]
    a.plot([gm["start"], gm["end"]], [0, 0], color="k", lw=.6)
    for s, e in gm["exons"]: a.add_patch(Rectangle((s, -.35), e - s, .7, color="k", lw=0))
    for s, e in gm["cds"]: a.add_patch(Rectangle((s, -.6), e - s, 1.2, color="k", lw=0))
    xt = gm["start"] if gm["strand"] == "+" else gm["end"]; dx = (w["hi"] - w["lo"]) * .03 * (1 if gm["strand"] == "+" else -1)
    a.add_patch(FancyArrowPatch((xt, .6), (xt + dx, .6), arrowstyle="-|>", mutation_scale=4, lw=.5, color="k")); a.plot([xt, xt], [0, .6], color="k", lw=.5)
    a.text(gm["end"] + (w["hi"] - w["lo"]) * .01, 0, g, fontstyle="italic", va="center", fontsize=6)
    span = w["hi"] - w["lo"]; sb = 10 ** np.floor(np.log10(span / 5)); a.plot([w["lo"], w["lo"] + sb], [-1, -1], color="k", lw=.6, clip_on=False); a.text(w["lo"] + sb / 2, -1.25, f"{sb / 1000:g} kb", ha="center", va="top", fontsize=5)
    a.text(1.0, -1.25, f"{w['chr']}:{w['lo']:,}-{w['hi']:,}", transform=a.get_xaxis_transform(), ha="right", va="top", fontsize=5)
    # strip: every TF with a linked q005 peak, ordered by z
    S = H.loc[sorted(bound & set(H.index[H.has_knockdown]), key=lambda t: H.loc[t, "z_all"])]; nb = len(S)
    s_ax = fig.add_axes([left / Wf, 11 / Hf, (trackW + barW + 4) / Wf, 5 / Hf]); norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-max(abs(S.z_all).max(), 3), vmax=max(abs(S.z_all).max(), 3)); cm = plt.get_cmap("PiYG_r")
    cm = mcolors.LinearSegmentedColormap.from_list("gp", [DOWN, "#F2F2F2", UP])
    xs = np.arange(nb); s_ax.scatter(xs, np.zeros(nb), s=4 + 30 * S.binding_pct.values ** 2, c=S.z_all.values, cmap=cm, norm=norm, edgecolors=["k" if c else "none" for c in S.confirmed_K10000], linewidths=.5, zorder=3)
    lab = set(S.index[S.confirmed_K10000]) | set(S[~S.confirmed_K10000].z_all.abs().sort_values(ascending=False).index[:3])
    for i, t in enumerate(S.index):
        if t in lab: s_ax.text(i, -.75, t, rotation=90, ha="center", va="top", fontsize=5, fontweight="bold" if S.loc[t, "confirmed_K10000"] else "normal")
    s_ax.axhline(0, color="#DDDDDD", lw=.4, zorder=1); s_ax.set_xlim(-.8, nb - .2); s_ax.set_ylim(-1, 1); s_ax.axis("off")
    s_ax.text(0, 1.1, f"All {nb} TFs with a linked q005 peak and knockdown data, ordered by z (left: gene down on knockdown)\ndot area = binding percentile; colour = z; outlined = confirmed in both halves of the cells",
              transform=s_ax.transAxes, fontsize=5, va="bottom")
    for ext in ("svg", "png"): fig.savefig(f"{OUT}/fig5d_{g}_locus.{ext}", dpi=400)
    plt.close(fig); print(g, "rows", rows_tf, "window", w["chr"], w["lo"], w["hi"], "far", w["far"], "max TF CPM", ymx, "ATAC max", atac.max(), "K27 max", k27.max(), flush=True)
