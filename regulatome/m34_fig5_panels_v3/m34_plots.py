# m34: Figure 5 panels, paper (7 pt) and slide (12.9 x 5.8 in, >= 12 pt) versions. usage: m34_plots.py <panel letters, e.g. abcdefg>
import sys, os, subprocess, gzip, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
P = rp.use()
R34 = "/work/users/d/y/dyy12/XuLab/regulatome"; D = f"{R34}/m34_fig5_panels_v3"; O = f"{D}/out"; PL = f"{D}/plots"; os.makedirs(f"{PL}/paper", exist_ok=True); os.makedirs(f"{PL}/slide", exist_ok=True)
NB = f"{R34}/m2c_abc_canonical/results/HEK293T_hicfixed"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"; TABIX = "/nas/longleaf/rhel9/apps/samtools/1.25/htslib/bin/tabix"
ATAC = f"{R34}/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"; K27 = f"{R34}/m2b_abc_k27ac/inputs/h3k27ac_hek293t_srx18002015_16.tagAlign.gz"; GTF = "/work/users/d/y/dyy12/XuLab/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
OURS, BULK, ATACC, K27C = P.arms.raw, P.neutral.ink, P.arms.atac, P.neutral.mid
ACT, REP, ACT1, REP1 = P.div[2], P.div[6], P.div[0], P.div[8]   # activates (down on knockdown) light/dark (Tier 1), represses light/dark
INK, DARK, MID, LIGHT, PALE = P.neutral.ink, P.neutral.dark, P.neutral.mid, P.neutral.light, P.neutral.pale
SLIDE = (12.9, 5.8)
def style(mode):
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titlesize": 14, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9, "lines.linewidth": 1.5})
    else: rp.use()
def save(fig, name, mode):
    if mode == "slide": fig.savefig(f"{PL}/slide/{name}.png", dpi=300); fig.savefig(f"{PL}/slide/{name}.svg")
    else: fig.savefig(f"{PL}/paper/{name}.png", dpi=600); fig.savefig(f"{PL}/paper/{name}.svg"); fig.savefig(f"{PL}/paper/{name}.pdf")
    plt.close(fig); print("saved", mode, name, flush=True)
R = pd.read_csv(f"{O}/regulome_v1.tsv.gz", sep="\t", low_memory=False)
# ---------------- a: schematic
def panel_a(mode):
    style(mode); fs = 14 if mode == "slide" else 7; fig = plt.figure(figsize=SLIDE if mode == "slide" else (7.0, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis("off")
    def box(x, y, w, h, txt, fc, ec, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2 if mode == "slide" else 0.6))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK, linespacing=1.3)
    def arrow(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18 if mode == "slide" else 8, lw=1.4 if mode == "slide" else 0.7, color=DARK))
    ax.text(2, 41.5, "Binding layer (our data)", fontsize=fs, fontweight="bold", color=OURS, va="center")
    box(2, 30, 24, 8, "79 TFs\nHITAG peaks\n(top 5,000 per TF)", "#FBEBD3", OURS); box(38, 30, 24, 8, "Promoter or\nABC-linked element\n(Hi-C ABC)", "#F7F7F7", DARK); box(74, 30, 24, 8, "Target gene", "#F7F7F7", DARK)
    arrow(26.8, 34, 37.2, 34); arrow(62.8, 34, 73.2, 34)
    ax.text(2, 25, "Perturb-seq layer (external, HEK293T knockdowns)", fontsize=fs, fontweight="bold", color=ACT1, va="center")
    box(2, 14, 34, 8, "Knockdown of the same TF\nchanges the gene (|z| ≥ 2)?", "#E8F1F8", ACT1); box(46, 14, 52, 8, "Support:  Tier 1 reproducible in both cell halves · Tier 2 · Tier 3 bound only\nDirection:  down on knockdown = activates · up = represses", "#E8F1F8", ACT1)
    arrow(36.8, 18, 45.2, 18); arrow(86, 29.2, 86, 22.8)
    ax.text(2, 9, "Combination layer", fontsize=fs, fontweight="bold", color=INK, va="center")
    box(2, 1, 96, 5.5, "Regulome v1:  150,403 TF–gene edges · 79 TFs · 11,046 genes · 9,480 supported by Perturb-seq (796 Tier 1), each with a direction", "#FFFFFF", INK, bold=False)
    save(fig, "Fig5a_layers", mode)
# ---------------- b: at a glance
def panel_b(mode):
    style(mode); s = mode == "slide"; tb = pd.read_csv(f"{O}/panel_b_targets_per_tf.tsv", sep="\t"); tg = pd.read_csv(f"{O}/panel_b_tfs_per_gene.tsv", sep="\t")
    fig = plt.figure(figsize=SLIDE if s else (7.0, 2.3)); gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.4, 0.75], wspace=0.45, left=0.07, right=0.98, bottom=0.14 if s else 0.2, top=0.9)
    ax = fig.add_subplot(gs[0]); x = np.arange(len(tb)); ax.bar(x, tb.promoter + tb.both, color=OURS, width=0.85, label="promoter peak"); ax.bar(x, tb.distal, bottom=tb.promoter + tb.both, color="#F2C48A", width=0.85, label="distal only")
    ax.set_xlim(-1, len(tb)); ax.set_xticks([]); ax.set_xlabel("79 TFs, ordered by number of target genes"); ax.set_ylabel("target genes per TF"); ax.legend(loc="upper right")
    for t in ("RBBP4", "NFYA", "CTCF", "E4F1", "NRF1", "PCGF6"):
        k = int(np.flatnonzero(tb.tf.values == t)[0]); v = tb.n_targets.values[k]; ax.annotate(t, (k, v), xytext=(k + (3 if k < 60 else -3), v + 0.12 * tb.n_targets.max()), fontsize=12 if s else 6, ha="left" if k < 60 else "right", arrowprops=dict(arrowstyle="-", lw=0.6, color=MID))
    ax2 = fig.add_subplot(gs[1]); ax2.bar(tg.n_tfs_bound, tg.n_genes, color=DARK, width=0.9); med = 12; ax2.axvline(med, color=OURS, lw=1.5 if s else 0.8, ls="--"); ax2.text(med + 1.5, tg.n_genes.max() * 0.93, f"median {med} TFs", color=OURS, fontsize=13 if s else 6.5)
    ax2.set_xlabel("TFs bound per gene"); ax2.set_ylabel("genes")
    ax3 = fig.add_subplot(gs[2]); sh = R.peak_location.value_counts(normalize=True) * 100; vals = [sh.get("promoter", 0), sh.get("promoter and distal", 0), sh.get("distal", 0)]; cols = [OURS, "#E9A75A", "#F2C48A"]; bot = 0
    for v, c, l in zip(vals, cols, ["promoter", "both", "distal only"]):
        ax3.bar(0, v, bottom=bot, color=c, width=0.6); ax3.text(0.42, bot + v / 2, f"{l}\n{v:.0f}%", va="center", fontsize=12 if s else 6); bot += v
    ax3.set_xlim(-0.5, 1.6); ax3.set_ylim(0, 100); ax3.set_xticks([]); ax3.set_ylabel("% of edges"); ax3.spines["bottom"].set_visible(False)
    save(fig, "Fig5b_at_a_glance", mode)
# ---------------- c: ours vs bulk
def panel_c(mode):
    style(mode); s = mode == "slide"; C = pd.read_csv(f"{O}/panel_c_ours_vs_bulk.tsv", sep="\t"); order = ["NRF1", "NFYA", "E4F1", "CTCF", "MAZ", "ZNF777", "ZIC2", "ZBTB7A", "ZNF282", "SETDB1"]; order = [t for t in order if t in set(C.TF)]
    C["ratio200"] = C.top200_hits_resp / C.top200_expected_resp
    fig, axs = plt.subplots(1, 2, figsize=SLIDE if s else (7.0, 2.8), sharey=True, gridspec_kw=dict(wspace=0.08, left=0.1, right=0.83 if s else 0.86, bottom=0.17, top=0.86))
    for ax, col, xl in ((axs[0], "fold_resp", "overlap with responding genes\n(fold over chance)"), (axs[1], "ratio200", "responders among the top 200\n(fold over chance)")):
        for yi, t in enumerate(order):
            g = C[C.TF == t]; o = g[g.source.str.startswith("our")]; b = g[g.source.str.startswith("bulk")]
            vals = list(o[col]) + list(b[col]); ax.plot([min(vals), max(vals)], [yi, yi], color=PALE, lw=3 if s else 1.5, zorder=1)
            ax.scatter(b[col], [yi] * len(b), s=90 if s else 14, facecolor="white", edgecolor=BULK, lw=1.4 if s else 0.7, zorder=3)
            ax.scatter(o[col], [yi] * len(o), s=90 if s else 14, color=OURS, zorder=4)
            if ax is axs[1]:
                for v, cl in zip(b[col], b.reference_cell_line):
                    pass
        ax.axvline(1, color=MID, lw=1 if s else 0.5, ls="--", zorder=0); ax.set_xlabel(xl)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
    lab = []
    for t in order:
        b = C[(C.TF == t) & C.source.str.startswith("bulk")]; lab.append(" / ".join(b.reference_cell_line.str.replace(" ENCFF259YUE", "").str.replace(" ENCFF348EDE", "").str.replace(" ENCFF683WRK", "").str.replace(" GSE103651", "").str.replace(" ENCODE", "")))
    for yi, l in enumerate(lab): axs[1].text(1.01, yi, l, transform=axs[1].get_yaxis_transform(), va="center", fontsize=12 if s else 6, color=DARK)
    axs[1].text(1.01, -0.75, "bulk ChIP cell line", transform=axs[1].get_yaxis_transform(), va="center", fontsize=12 if s else 6, color=DARK, fontweight="bold")
    h1 = axs[0].scatter([], [], s=90 if s else 14, color=OURS, label="our HITAG (HEK293T)"); h2 = axs[0].scatter([], [], s=90 if s else 14, facecolor="white", edgecolor=BULK, label="bulk ChIP, same path")
    fig.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.1, 0.995), ncol=2, frameon=False)
    save(fig, "Fig5c_ours_vs_bulk", mode)
# ---------------- locus helpers
def tabix(f, c, a, b):
    out = subprocess.run([TABIX, f, f"{c}:{a}-{b}"], capture_output=True, text=True).stdout.strip().split("\n"); pos = []
    for l in out:
        if not l: continue
        x = l.split("\t"); pos.append(int(x[1]) if x[5] == "+" else int(x[2]) - 1)
    return np.array(pos)
def frag(t, c, a, b):
    out = subprocess.run(f"zcat {V}/fragment/TF.{t}.bed.gz | awk -F'\\t' '$1==\"{c}\" && $2>={a-1000} && $2<{b}'", shell=True, capture_output=True, text=True).stdout.strip().split("\n"); pos = []
    for l in out:
        if not l: continue
        x = l.split("\t"); p = int(x[1]) if x[5].startswith("+") else int(x[2]) - 1
        if a <= p < b: pos.append(p)
    return np.array(pos)
TOT = {}
def total(t):
    if t not in TOT: TOT[t] = float(np.load(f"{R34}/m10_promoter_regulome/work/counts/{t}.npz", allow_pickle=True)["total"])
    return TOT[t]
def genes_in(c, a, b):
    rows = subprocess.run(f"zcat {GTF} | awk -F'\\t' '$1==\"{c}\" && $4<{b} && $5>{a} && ($3==\"transcript\"||$3==\"exon\")'", shell=True, capture_output=True, text=True).stdout.strip().split("\n"); tx = {}
    for l in rows:
        x = l.split("\t")
        if len(x) < 9 or "Ensembl_canonical" not in x[8] or 'gene_type "protein_coding"' not in x[8]: continue
        gn = re.search(r'gene_name "([^"]+)"', x[8]).group(1); d = tx.setdefault(gn, dict(strand=x[6], ex=[]))
        if x[2] == "transcript": d["s"], d["e"] = int(x[3]), int(x[4])
        else: d["ex"].append((int(x[3]), int(x[4])))
    return {g: v for g, v in tx.items() if "s" in v}
def links_of(gene):
    L = pd.read_csv(f"{NB}/Predictions/EnhancerPredictionsFull_threshold0.02_self_promoter.tsv", sep="\t", usecols=["chr", "start", "end", "TargetGene", "TargetGeneTSS", "ABC.Score", "isSelfPromoter"]); return L[L.TargetGene == gene]
def draw_genes(ax, G, a, b, focus, fs):
    lanes = []
    for gn, v in sorted(G.items(), key=lambda kv: kv[1]["s"]):
        s0, e0 = max(v["s"], a), min(v["e"], b); lane = next((k for k, e in enumerate(lanes) if s0 > e + 0.02 * (b - a)), len(lanes))
        if lane == len(lanes): lanes.append(0)
        lanes[lane] = e0 + 0.06 * (b - a); y = -lane; col = P.chromhmm.promoter if gn == focus else DARK
        ax.plot([s0, e0], [y, y], color=col, lw=1.0)
        for xs, xe in v["ex"]: ax.add_patch(Rectangle((xs, y - 0.22), max(xe - xs, 0.001 * (b - a)), 0.44, color=col, lw=0))
        ax.text((s0 + e0) / 2, y - 0.33, gn, ha="center", va="top", fontsize=fs, style="italic", color=col, fontweight="bold" if gn == focus else "normal")
    ax.set_ylim(-len(lanes) - 0.4, 0.7); ax.axis("off")
def cov_track(ax, pos, a, b, nb, norm, color, label, fs, ymax=None):
    e = np.linspace(a, b, nb + 1); h, _ = np.histogram(pos, bins=e); h = h / norm * 1e6; ax.fill_between(e[:-1], h, step="post", color=color, lw=0)
    ax.set_xlim(a, b); ax.set_ylim(0, ymax if ymax else max(h.max() * 1.1, 1e-9)); ax.set_yticks([]); ax.set_xticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.text(-0.005, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=fs, color=color if color not in (PALE, LIGHT) else DARK)
    ax.text(0.002, 0.95, f"{ax.get_ylim()[1]:.1f}", transform=ax.transAxes, ha="left", va="top", fontsize=fs * 0.75, color=MID)
def arcs(ax, Lg, tss, a, b, hl, fs):
    for r in Lg.rename(columns={"ABC.Score": "abc"}).itertuples():
        m = (r.start + r.end) / 2
        if r.isSelfPromoter in (True, "True"): continue
        x1, x2 = sorted([m, tss]); t = np.linspace(0, np.pi, 80); xs = (x1 + x2) / 2 - (x2 - x1) / 2 * np.cos(t); hgt = (x2 - x1) / (b - a)
        ax.plot(xs, hgt * np.sin(t), color=OURS if (r.chr, r.start, r.end) in hl else LIGHT, lw=0.6 + 4 * min(r.abc, 1), zorder=3 if (r.chr, r.start, r.end) in hl else 1)
    ax.set_xlim(a, b); ax.set_ylim(0, 1.05 * max(0.05, ((Lg.start + Lg.end) / 2 - tss).abs().max() / (b - a))); ax.axis("off")
    ax.text(-0.005, 0.5, "ABC links", transform=ax.transAxes, ha="right", va="center", fontsize=fs, color=DARK)
def ruler(ax, c, a, b, fs):
    ax.set_xlim(a, b); ax.set_yticks([]); [ax.spines[k].set_visible(False) for k in ("left", "right", "top")]; tk = np.linspace(a, b, 5); ax.set_xticks(tk); ax.set_xticklabels([f"{v/1e6:.3f}" for v in tk], fontsize=fs * 0.85); ax.set_xlabel(f"{c} (Mb)", fontsize=fs)
# ---------------- d: NRF1
def panel_d(mode):
    style(mode); s = mode == "slide"; fs = 13 if s else 6.5; F = pd.read_csv(f"{O}/panel_d_nrf1_funnel.tsv", sep="\t").iloc[0]; gene = "PGAM1"
    Lg = links_of(gene); tss = int(Lg.TargetGeneTSS.iloc[0]); c = Lg.chr.iloc[0]; lo = int(min(Lg.start.min(), tss)); hi = int(max(Lg.end.max(), tss)); pad = int(0.08 * (hi - lo)) + 2000; a, b = lo - pad, hi + pad
    nrf = frag("NRF1", c, a, b); atac = tabix(ATAC, c, a, b); k27 = tabix(K27, c, a, b); G = genes_in(c, a, b)
    Pk = pd.read_csv(f"{V}/macs2/TF.NRF1/TF.NRF1_peaks.narrowPeak", sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); Pk = Pk[Pk.sig.rank(ascending=False, method="first") <= 5000]; Pk = Pk[(Pk.c == c) & (Pk.s + Pk.off >= a) & (Pk.s + Pk.off < b)]
    El = pd.read_csv(f"{NB}/Neighborhoods/EnhancerList.txt", sep="\t", usecols=["chr", "start", "end"]); El = El[El.chr == c]; sm = (Pk.s + Pk.off).values
    hl = {(c, r.start, r.end) for r in Lg.itertuples() if ((sm >= r.start) & (sm < r.end)).any()}
    z = R[(R.tf == "NRF1") & (R.gene == gene)].knockdown_z.iloc[0]
    fig = plt.figure(figsize=SLIDE if s else (7.0, 2.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.42, left=0.17, right=0.98, bottom=0.14, top=0.93)
    ax = fig.add_subplot(gs[0]); labs = ["bound genes", "supported by\nPerturb-seq", "down on knockdown\n(activates)", "up on knockdown\n(represses)", "Tier 1\n(both halves)"]; vals = [F.bound, F.supported, F.down_activates, F.up_represses, F.tier1]
    cols = [OURS, DARK, ACT, REP, ACT1]; y = np.arange(len(vals))[::-1]; ax.barh(y, vals, color=cols, height=0.7)
    for yy, v in zip(y, vals): ax.text(v + 25, yy, f"{v:,}", va="center", fontsize=fs)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_xlabel("NRF1 target genes"); ax.set_xlim(0, F.bound * 1.25)
    sub = gs[1].subgridspec(6, 1, height_ratios=[1.0, 0.8, 0.8, 1.0, 0.35, 1.1], hspace=0.12)
    a0 = fig.add_subplot(sub[0]); arcs(a0, Lg, tss, a, b, hl, fs); a0.set_title(f"{gene}: NRF1 knockdown z = {z:.1f} (Tier 1, activates)", fontsize=fs, loc="left")
    ax1 = fig.add_subplot(sub[1]); e = np.linspace(a, b, 301); h, _ = np.histogram(atac, bins=e); ax1.fill_between(e[:-1], h, step="post", color=ATACC, lw=0); ax1.set_xlim(a, b); ax1.axis("off"); ax1.text(-0.005, 0.5, "ATAC", transform=ax1.transAxes, ha="right", va="center", fontsize=fs, color=ATACC)
    ax2 = fig.add_subplot(sub[2]); h, _ = np.histogram(k27, bins=e); ax2.fill_between(e[:-1], h, step="post", color=K27C, lw=0); ax2.set_xlim(a, b); ax2.axis("off"); ax2.text(-0.005, 0.5, "H3K27ac", transform=ax2.transAxes, ha="right", va="center", fontsize=fs, color=K27C)
    ax3 = fig.add_subplot(sub[3]); h, _ = np.histogram(nrf, bins=e); ax3.fill_between(e[:-1], h / total("NRF1") * 1e6, step="post", color=OURS, lw=0); ax3.set_xlim(a, b); ax3.axis("off"); ax3.text(-0.005, 0.5, "NRF1\nHITAG", transform=ax3.transAxes, ha="right", va="center", fontsize=fs, color=OURS)
    ax4 = fig.add_subplot(sub[4]); [ax4.add_patch(Rectangle((r.s, 0), max(2 * r.off, 0.002 * (b - a)), 1, color=OURS, lw=0)) for r in Pk.itertuples()]; ax4.set_xlim(a, b); ax4.set_ylim(0, 1); ax4.axis("off"); ax4.text(-0.005, 0.5, "peaks", transform=ax4.transAxes, ha="right", va="center", fontsize=fs, color=OURS)
    ax5 = fig.add_subplot(sub[5]); draw_genes(ax5, G, a, b, gene, fs); ax5.set_xlim(a, b)
    ax5.text(0, -0.08, f"{c}:{a:,}-{b:,}", transform=ax5.transAxes, fontsize=fs * 0.9, color=MID, va="top")
    save(fig, "Fig5d_NRF1", mode)
# ---------------- e: HSPB1
def panel_e(mode, gene="HSPB1"):
    style(mode); s = mode == "slide"; fs = 12 if s else 6; T = pd.read_csv(f"{O}/panel_e_gene_{gene}.tsv", sep="\t"); ch = T[T.tier <= 2].sort_values("knockdown_z"); rest = [t for t in T.tf if t not in set(ch.tf)]
    Lg = links_of(gene); tss = int(Lg.TargetGeneTSS.iloc[0]); c = Lg.chr.iloc[0]; lo = int(min(Lg.start.min(), tss)); hi = int(max(Lg.end.max(), tss)); pad = int(0.06 * (hi - lo)) + 2000; a, b = lo - pad, hi + pad
    atac = tabix(ATAC, c, a, b); k27 = tabix(K27, c, a, b); G = genes_in(c, a, b); cov = {t: frag(t, c, a, b) for t in T.tf}
    bound_el = set()
    for t in ch.tf:
        el = T[T.tf == t].best_element.iloc[0]
        if isinstance(el, str) and ":" in el: cc, se = el.split(":"); x, y = se.split("-"); bound_el.add((cc, int(x), int(y)))
    rows = list(ch.tf) + [f"other {len(rest)} TFs"]; n = len(rows)
    fig = plt.figure(figsize=SLIDE if s else (7.0, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.04, left=0.14, right=0.985, bottom=0.08, top=0.96)
    hr = [1.1, 0.7, 0.7] + [0.55] * n + [1.0]; sub = gs[0].subgridspec(len(hr), 1, height_ratios=hr, hspace=0.08); subz = gs[1].subgridspec(len(hr), 1, height_ratios=hr, hspace=0.08)
    a0 = fig.add_subplot(sub[0]); arcs(a0, Lg, tss, a, b, bound_el, fs)
    e = np.linspace(a, b, 401)
    for k, (pos, col, lab) in enumerate(((atac, ATACC, "ATAC"), (k27, K27C, "H3K27ac"))):
        ax = fig.add_subplot(sub[1 + k]); h, _ = np.histogram(pos, bins=e); ax.fill_between(e[:-1], h, step="post", color=col, lw=0); ax.set_xlim(a, b); ax.axis("off"); ax.text(-0.005, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=fs, color=col)
    ymax = 0; H = []
    for r in rows:
        ts = rest if r.startswith("other") else [r]; pos = np.concatenate([cov[t] for t in ts]) if ts else np.array([]); nn = sum(total(t) for t in ts); h, _ = np.histogram(pos, bins=e); H.append(h / nn * 1e6 * (1 if not r.startswith("other") else 1)); 
    ymax = max(np.percentile(np.concatenate(H), 99.9), 1e-9)
    zmax = max(2.5, np.abs(ch.knockdown_z).max() * 1.08)
    for k, (r, h) in enumerate(zip(rows, H)):
        ax = fig.add_subplot(sub[3 + k]); col = OURS if not r.startswith("other") else DARK; ax.fill_between(e[:-1], np.minimum(h, ymax), step="post", color=col, lw=0); ax.set_xlim(a, b); ax.set_ylim(0, ymax); ax.axis("off")
        ax.text(-0.005, 0.5, r, transform=ax.transAxes, ha="right", va="center", fontsize=fs, color=col)
        az = fig.add_subplot(subz[3 + k]); az.set_xlim(-zmax, zmax); az.axvline(0, color=MID, lw=0.8 if s else 0.4); az.set_yticks([]); [az.spines[q].set_visible(False) for q in ("left", "right", "top")]
        if not r.startswith("other"):
            zz = float(ch[ch.tf == r].knockdown_z.iloc[0]); tr = int(ch[ch.tf == r].tier.iloc[0]); az.barh(0, zz, color=(ACT1 if tr == 1 else ACT) if zz < 0 else (REP1 if tr == 1 else REP), height=0.7)
            az.text(zz + (-0.3 if zz < 0 else 0.3) * zmax / 6, 0, f"{zz:.1f}", va="center", ha="right" if zz < 0 else "left", fontsize=fs * 0.9)
        else: az.text(0, 0, "not changed", ha="center", va="center", fontsize=fs * 0.9, color=MID)
        az.set_ylim(-0.6, 0.6)
        if k < n - 1: az.set_xticks([]); az.spines["bottom"].set_visible(False)
        else: az.set_xlabel("knockdown z\n(down = activates)", fontsize=fs)
    ag = fig.add_subplot(sub[-1]); draw_genes(ag, G, a, b, gene, fs); ag.set_xlim(a, b); ag.text(0, -0.1, f"{c}:{a:,}-{b:,} · HITAG tracks share one scale (CPM)", transform=ag.transAxes, fontsize=fs * 0.9, color=MID, va="top")
    save(fig, f"Fig5e_{gene}", mode)
# ---------------- f: activator / repressor
def panel_f(mode):
    style(mode); s = mode == "slide"; fs = 12 if s else 6; Fd = pd.read_csv(f"{O}/panel_f_activator_repressor.tsv", sep="\t"); Fd = Fd[Fd.supported >= 20].sort_values("pct_activates", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=SLIDE if s else (7.0, 2.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28, left=0.07, right=0.98, bottom=0.2, top=0.93)
    ax = fig.add_subplot(gs[0]); x = np.arange(len(Fd)); ax.bar(x, Fd.pct_activates, color=[ACT1 if v >= 50 else ACT for v in Fd.pct_activates], width=0.85); ax.axhline(50, color=MID, lw=1 if s else 0.5, ls="--")
    ax.set_xlim(-1, len(Fd)); ax.set_ylim(0, 100); ax.set_xticks([]); ax.set_ylabel("% of supported edges that go down\non knockdown (TF activates)"); ax.set_xlabel(f"{len(Fd)} TFs with ≥ 20 supported edges, ordered")
    named = [t for t in ("CTCF", "NRF1", "WIZ", "NFYA", "E4F1", "DNMT1", "NME2", "MBD3", "SRCAP") if t in set(Fd.tf)]
    for j, t in enumerate(named):
        k = int(np.flatnonzero(Fd.tf.values == t)[0]); v = Fd.pct_activates.values[k]; OFF = dict(CTCF=(4, 94), NRF1=(9, 86), NFYA=(0, 78), WIZ=(-5, 70), E4F1=(4, 64))
        dx, yt = OFF.get(t, (0, min(97, v + 12 + (j % 2) * 9)))
        ax.annotate(t, (k, v), xytext=(k + dx, yt), ha="center", va="bottom", fontsize=fs, arrowprops=dict(arrowstyle="-", lw=0.7 if s else 0.4, color=MID))
    big = Fd[Fd.supported >= 130].sort_values("supported"); ax2 = fig.add_subplot(gs[1]); y = np.arange(len(big))
    ax2.barh(y, -big.activates, color=ACT, height=0.7); ax2.barh(y, -big.tier1_activates, color=ACT1, height=0.7); ax2.barh(y, big.represses, color=REP, height=0.7); ax2.barh(y, big.tier1_represses, color=REP1, height=0.7)
    ax2.set_yticks(y); ax2.set_yticklabels(big.tf); ax2.axvline(0, color=INK, lw=0.8 if s else 0.4); m = max(big.activates.max(), big.represses.max()) * 1.05; ax2.set_xlim(-m, m)
    tk = ax2.get_xticks(); ax2.set_xticks(tk); ax2.set_xticklabels([f"{abs(int(v))}" for v in tk]); ax2.set_xlim(-m, m); ax2.set_xlabel("supported edges (darker = Tier 1)")
    ax2.text(0.02, 1.0, "activates", transform=ax2.transAxes, color=ACT1, fontsize=fs, va="bottom"); ax2.text(0.98, 1.0, "represses", transform=ax2.transAxes, color=REP1, fontsize=fs, va="bottom", ha="right")
    save(fig, "Fig5f_activator_repressor", mode)
# ---------------- g: Figure 4 pairs
def panel_g(mode):
    style(mode); s = mode == "slide"; fs = 12 if s else 6; Gt = pd.read_csv(f"{O}/panel_g_pairs_shared_targets.tsv", sep="\t"); Gt = Gt[Gt.figure4_pair]
    fig = plt.figure(figsize=SLIDE if s else (7.0, 2.4)); gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], wspace=0.32, left=0.13, right=0.98, bottom=0.2, top=0.9)
    ax = fig.add_subplot(gs[0]); y = np.arange(len(Gt))[::-1]; ax.barh(y + 0.2, Gt.n_bound_a, height=0.38, color=LIGHT); ax.barh(y - 0.2, Gt.n_bound_b, height=0.38, color=LIGHT); ax.barh(y, Gt.n_shared_targets, height=0.25, color=OURS)
    ax.set_yticks(y); ax.set_yticklabels(Gt.pair); ax.set_xlabel("target genes"); 
    xm = max(Gt.n_bound_a.max(), Gt.n_bound_b.max())
    for yy, v in zip(y, Gt.n_shared_targets): ax.text(xm * 1.04, yy, f"{v} shared", va="center", fontsize=fs, color=OURS)
    ax.set_xlim(0, xm * 1.45); from matplotlib.patches import Patch as _P; fig.legend(handles=[_P(color=OURS, label="shared targets (bound by both)"), _P(color=LIGHT, label="left: all targets of each member · right: that member's other targets")], loc="upper left", bbox_to_anchor=(0.12, 0.995), ncol=2, fontsize=fs, frameon=False)
    ax2 = fig.add_subplot(gs[1]); xs, labs, k = [], [], 0
    for r in Gt.itertuples():
        a_, b_ = r.pair.split("-")
        for side, t in (("a", a_), ("b", b_)):
            sh, oo = getattr(r, f"pct_shared_responding_{side}", np.nan), getattr(r, f"pct_own_only_targets_responding_{side}", np.nan)
            if isinstance(getattr(r, f"kd_{side}"), str) and "too few" in getattr(r, f"kd_{side}"):
                ax2.text(k, 0.5, f"{t} knockdown:\n18 cells, not used", ha="center", va="bottom", fontsize=fs * 0.85, color=MID, rotation=90)
            else:
                ax2.bar(k - 0.18, sh, width=0.34, color=OURS); ax2.bar(k + 0.18, oo, width=0.34, color=LIGHT)
            labs.append(t); xs.append(k); k += 1
        k += 0.6
    ax2.set_xticks(xs); ax2.set_xticklabels(labs, fontsize=fs * 0.95); ax2.set_ylabel("% of targets that respond\nto the knockdown (|z| ≥ 2)"); ax2.set_ylim(0, 8.5); ax2.set_xlabel("knockdown of", labelpad=36 if s else 18)
    pass
    for j in range(0, len(xs), 2): ax2.text((xs[j] + xs[j + 1]) / 2, -0.11, Gt.pair.values[j // 2], transform=ax2.get_xaxis_transform(), ha="center", va="top", fontsize=fs, fontweight="bold")
    save(fig, "Fig5g_pairs", mode)
FUN = dict(a=panel_a, b=panel_b, c=panel_c, d=panel_d, e=panel_e, f=panel_f, g=panel_g)
for p in sys.argv[1]:
    for mode in ("paper", "slide"): FUN[p](mode)
print("done")
