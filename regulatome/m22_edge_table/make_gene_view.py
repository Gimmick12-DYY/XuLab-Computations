# m22 gene view (SPEC 10): locus view from our data + per-TF bars of predicted_impact share (ours) and observed knockdown z (external, X-Atlas/Orion HEK293T) + strip of all bound TFs.
# usage: make_gene_view.py GENE [GENE ...] -> out/views/gene_<GENE>.svg/.png
import os, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch
import matplotlib.colors as mcolors
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4, "xtick.major.width": .4, "ytick.major.width": .4, "xtick.major.size": 1.5, "ytick.major.size": 1.5})
X = "/work/users/d/y/dyy12/XuLab"; RG = f"{X}/regulatome"; O = f"{RG}/m22_edge_table"; OUT = f"{O}/out/views"; os.makedirs(OUT, exist_ok=True)
V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; TABIX = "/nas/longleaf/rhel9/apps/samtools/1.25/htslib/bin/tabix"; GTF = f"{X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
ATAC = f"{RG}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"; K27 = f"{RG}/m2b_abc_k27ac/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz"; QMIN = -np.log10(0.05); BIN = 50
PROM_BAND, ENH_BAND, DOWN, UP, OURS = "#DCE8F5", "#FBE6CF", "#009E73", "#CC79A7", "#555555"; MM = 1 / 25.4; NROW = 6
ZC = "observed_kd_z [external: X-Atlas/Orion HEK293T]"; H1c, H2c = "observed_kd_z_half1 [external]", "observed_kd_z_half2 [external]"
E = pd.read_csv(f"{O}/out/regulome_edges.tsv.gz", sep="\t", low_memory=False)
L = pd.read_csv(f"{RG}/m2c_abc_canonical/results/HEK293T_hicfixed/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"]).rename(columns={"ABC.Score": "abc"})
L["self"] = L.isSelfPromoter.astype(str).str.lower().eq("true")
def gene_model(g, tss):
    rows = [r.split("\t") for r in subprocess.run(f"zcat {GTF} | grep -w 'gene_name \"{g}\";'", shell=True, capture_output=True, text=True).stdout.strip().splitlines()]
    tid = lambda r: r[8].split('transcript_id "')[1].split('"')[0]; T = [r for r in rows if r[2] == "transcript"]; b = min(T, key=lambda r: abs((int(r[3]) if r[6] == "+" else int(r[4])) - tss))
    return dict(chr=b[0], start=int(b[3]), end=int(b[4]), strand=b[6], exons=[(int(r[3]), int(r[4])) for r in rows if r[2] == "exon" and tid(r) == tid(b)], cds=[(int(r[3]), int(r[4])) for r in rows if r[2] == "CDS" and tid(r) == tid(b)])
def tagcov(path, c, lo, hi):
    out = subprocess.run([TABIX, path, f"{c}:{lo}-{hi}"], capture_output=True, text=True).stdout.strip().splitlines(); bins = np.arange(lo, hi + BIN, BIN)
    if not out: return np.zeros(len(bins) - 1)
    a = np.array([l.split("\t") for l in out]); p5 = np.where(a[:, 5] == "+", a[:, 1].astype(int), a[:, 2].astype(int) - 1); return np.histogram(p5, bins=bins)[0].astype(float)
def tfcov(t, c, lo, hi):
    F = pd.read_csv(f"{V}/fragment/TF.{t}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2, 5], names=["c", "s", "e", "st"]); p5 = np.where(F.st.values == "+", F.s.values, F.e.values - 1); m = (F.c.values == c) & (p5 >= lo) & (p5 < hi)
    return np.convolve(np.histogram(p5[m], bins=np.arange(lo, hi + BIN, BIN))[0] / len(F) * 1e6, np.ones(3) / 3, mode="same")
def peaks(t, c, lo, hi):
    P = pd.read_csv(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 8, 9], names=["c", "s", "q", "off"]); P = P[(P.q >= QMIN) & (P.c == c)]; sm = P.s + P.off; return sm[(sm >= lo) & (sm < hi)].values
for g in sys.argv[1:]:
    D = E[E.gene == g].sort_values("predicted_impact_share_at_gene", ascending=False).reset_index(drop=True)
    if not len(D): print(g, "no bound TF in the edge table"); continue
    Lg = L[L.TargetGene == g]; tss = int(Lg.TargetGeneTSS.iloc[0]); gm = gene_model(g, tss); far = Lg[np.abs((Lg.start + Lg.end) / 2 - tss) > 200000]; near = Lg.drop(far.index)
    lo, hi = min(near.start.min(), gm["start"]), max(near.end.max(), gm["end"])
    if len(far): lo, hi = min(tss - 20000, gm["start"]), max(tss + 20000, gm["end"])
    pad = (hi - lo) * .1; lo, hi = int(lo - pad), int(hi + pad); c = gm["chr"]; x = np.arange(lo, hi, BIN) + BIN / 2
    rows = list(D.tf.head(NROW)); nT = len(rows)
    hts = [6, 5, 5] + [6] * nT + [5]; gap = .8; top = 3; Wf = 120; trackW = 66; left = 15; BOT = 27; Hf = top + sum(hts) + gap * len(hts) + BOT
    fig = plt.figure(figsize=(Wf * MM, Hf * MM)); yy = Hf - top; axs = []
    for h_ in hts: yy -= h_; axs.append(fig.add_axes([left / Wf, yy / Hf, trackW / Wf, h_ / Hf])); yy -= gap
    for a in axs:
        a.set_xlim(lo, hi); a.set_xticks([]); [a.spines[s].set_visible(False) for s in ("top", "right", "bottom")]
        for r in Lg.itertuples():
            if r.end > lo and r.start < hi: a.axvspan(r.start, r.end, color=PROM_BAND if r.self else ENH_BAND, lw=0, zorder=0)
    a = axs[0]; a.set_ylim(0, 1); a.set_yticks([]); a.spines["left"].set_visible(False); cm = plt.get_cmap("Greys")
    for r in Lg.itertuples():
        if r.self or not (r.end > lo and r.start < hi): continue
        xm = (r.start + r.end) / 2; a.add_patch(Arc(((xm + tss) / 2, 0), abs(xm - tss), 1.8, theta1=0, theta2=180, lw=.6, color=cm(.35 + .65 * min(r.abc / .12, 1))))
    a.text(-.01, .5, "ABC links", transform=a.transAxes, ha="right", va="center")
    if len(far): a.text(1, .95, f"+{len(far)} linked elements > 200 kb not shown", transform=a.transAxes, ha="right", va="top", fontsize=5)
    def cov(a, v, lab, col, ym=None):
        a.fill_between(x[:len(v)], 0, v, color=col, lw=0, step="mid"); ym = ym or max(v.max(), 1e-9); a.set_ylim(0, ym); a.set_yticks([ym]); a.set_yticklabels([f"{ym:.0f}" if ym >= 10 else f"{ym:.1f}"], fontsize=5); a.text(-.01, .5, lab, transform=a.transAxes, ha="right", va="center")
    cov(axs[1], tagcov(ATAC, c, lo, hi), "ATAC", "#8C8C8C"); cov(axs[2], tagcov(K27, c, lo, hi), "H3K27ac", "#8C8C8C")
    CV = {t: tfcov(t, c, lo, hi) for t in rows}; ym = float(f"{max([v.max() for v in CV.values()] + [1e-9]):.2g}")
    zmax = max(np.nanmax(np.abs(D[ZC].head(NROW).values)) if D[ZC].head(NROW).notna().any() else 3, 3) * 1.1; smax = D.predicted_impact_share_at_gene.head(NROW).max() * 1.15
    for k, t in enumerate(rows):
        a = axs[3 + k]; cov(a, CV[t], t, "#4D4D4D", ym); pk = peaks(t, c, lo, hi); a.plot(pk, np.full(len(pk), -.12 * ym), "|", color="k", ms=3, mew=.6, clip_on=False)
        r = D[D.tf == t].iloc[0]; pos = a.get_position()
        b1 = fig.add_axes([(left + trackW + 3) / Wf, pos.y0, 15 / Wf, pos.height]); b1.barh(0, r.predicted_impact_share_at_gene, height=.55, color=OURS); b1.set_xlim(0, smax); b1.set_ylim(-.6, .6); b1.set_yticks([]); [b1.spines[s].set_visible(False) for s in ("top", "right", "left")]
        b1.text(r.predicted_impact_share_at_gene, 0, f" {r.predicted_impact_share_at_gene:.2f}", va="center", fontsize=5)
        b2 = fig.add_axes([(left + trackW + 21) / Wf, pos.y0, 15 / Wf, pos.height]); z = r[ZC]; [b2.spines[s].set_visible(False) for s in ("top", "right", "left")]; b2.set_xlim(-zmax, zmax); b2.set_ylim(-.6, .6); b2.set_yticks([]); b2.axvline(0, color="k", lw=.4)
        if np.isfinite(z):
            b2.barh(0, z, height=.55, color=DOWN if z < 0 else UP, alpha=.85); b2.plot([r[H1c], r[H2c]], [0, 0], "o", ms=1.6, color="k", mfc="white", mew=.4)
        else: b2.text(0, 0, "no knockdown", ha="center", va="center", fontsize=5, color="#777777")
        if k < nT - 1: b1.set_xticks([]); b2.set_xticks([]); b1.spines["bottom"].set_visible(False); b2.spines["bottom"].set_visible(False)
        else: b1.set_xlabel("predicted impact\nshare (ours)", fontsize=5.5, labelpad=1); b2.set_xlabel("knockdown z\n(external)", fontsize=5.5, labelpad=1)
    if nT: axs[3].text(.005, 1.0, f"HITAG, CPM (same scale); top {nT} TFs at {g} by predicted impact", transform=axs[3].transAxes, ha="left", va="top", fontsize=5, color="#555555")
    a = axs[-1]; a.set_ylim(-1, 1); a.set_yticks([]); a.spines["left"].set_visible(False); a.plot([gm["start"], gm["end"]], [0, 0], color="k", lw=.6)
    for s_, e_ in gm["exons"]: a.add_patch(Rectangle((s_, -.35), e_ - s_, .7, color="k", lw=0))
    for s_, e_ in gm["cds"]: a.add_patch(Rectangle((s_, -.6), e_ - s_, 1.2, color="k", lw=0))
    xt = gm["start"] if gm["strand"] == "+" else gm["end"]; dx = (hi - lo) * .03 * (1 if gm["strand"] == "+" else -1); a.add_patch(FancyArrowPatch((xt, .6), (xt + dx, .6), arrowstyle="-|>", mutation_scale=4, lw=.5, color="k")); a.plot([xt, xt], [0, .6], color="k", lw=.5)
    a.text(gm["end"] + (hi - lo) * .01 if gm["end"] < hi - (hi - lo) * .1 else gm["start"] - (hi - lo) * .01, 0, g, fontstyle="italic", va="center", ha="left" if gm["end"] < hi - (hi - lo) * .1 else "right", fontsize=6)
    span = hi - lo; sb = 10 ** np.floor(np.log10(span / 5)); a.plot([lo, lo + sb], [-1, -1], color="k", lw=.6, clip_on=False); a.text(lo + sb / 2, -1.25, f"{sb / 1000:g} kb", ha="center", va="top", fontsize=5); a.text(1, -1.25, f"{c}:{lo:,}-{hi:,}", transform=a.get_xaxis_transform(), ha="right", va="top", fontsize=5)
    S = D.sort_values("predicted_impact_share_at_gene", ascending=False).reset_index(drop=True); n = len(S); sax = fig.add_axes([left / Wf, 8 / Hf, (trackW + 36) / Wf, 4.5 / Hf])
    zz = S[ZC].values; lim = max(np.nanmax(np.abs(zz)) if np.isfinite(zz).any() else 3, 3); cmap = mcolors.LinearSegmentedColormap.from_list("gp", [DOWN, "#F2F2F2", UP]); norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    has = np.isfinite(zz); xs = np.arange(n); sz = 4 + 40 * S.predicted_impact_share_at_gene.values / max(S.predicted_impact_share_at_gene.max(), 1e-9)
    sax.scatter(xs[has], np.zeros(has.sum()), s=sz[has], c=zz[has], cmap=cmap, norm=norm, edgecolors="#444444", linewidths=.3, zorder=3); sax.scatter(xs[~has], np.zeros((~has).sum()), s=sz[~has], facecolors="white", edgecolors="#9E9E9E", linewidths=.4, zorder=3)
    for i in range(min(n, 8)): sax.text(i, -.8, S.tf[i], rotation=90, ha="center", va="top", fontsize=5)
    sax.axhline(0, color="#DDDDDD", lw=.4, zorder=1); sax.set_xlim(-.8, max(n, 10) - .2); sax.set_ylim(-1, 1); sax.axis("off")
    sax.text(0, 1.1, f"All {n} TFs bound at {g} (our data), ordered by predicted impact share; dot area = share.\nColour = knockdown z (external: X-Atlas/Orion HEK293T; green = gene down, pink = up); hollow = no knockdown data.", transform=sax.transAxes, fontsize=5, va="bottom")
    for ext in ("svg", "png"): fig.savefig(f"{OUT}/gene_{g}.{ext}", dpi=300)
    plt.close(fig); print(g, "bound TFs", n, "rows", rows, flush=True)
