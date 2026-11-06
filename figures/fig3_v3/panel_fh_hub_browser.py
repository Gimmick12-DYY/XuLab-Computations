"""Fig 3f (112 x 72 mm) and 3h (179 x 46 mm): chr16p11.2 multi-TF clique hub browser and zoom-ins on two hub windows.
Same inputs and track logic as figures_paletteC/plot_hub_browser_paletteC.py (job 7246482), re-laid out at the final panel
sizes: Hi-C triangle (WT, 10 kb, balanced, log10) -> Peakachu 0.7 loops -> shared clique edges (red = on a 0.7 loop,
blue = Cicero only; width = no. of TFs) -> raw per-TF coverage (read 5' ends, CPM; top 10 TFs + the rest pooled, one scale
per view) -> ChromHMM (grouped) -> RefSeq genes (red = hub promoter). Hub elements are shaded through the tracks.
Step 1 (--cache, Slurm): reads the TF fragment beds (xujie_lab, read-only), the mcool, loops and ChromHMM once and writes
inputs/hub_chr16_cache.npz. Step 2 (default): draws both panels from the cache."""
import argparse, gzip, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
from matplotlib.patches import Rectangle
from fig3common import P, INP, XULAB, fig_mm, mm_axes, save, plt

HB = XULAB / "cobinding/results/hub_browser"
LOOPDIR = XULAB / "hic/work_wt293t/loops"
ap = argparse.ArgumentParser()
ap.add_argument("--cache", action="store_true")
ap.add_argument("--edges", default=str(HB / "hub_chr16_edges.tsv"))
ap.add_argument("--genes", default=str(HB / "genes_chr16_29.7-31.6.tsv"))
ap.add_argument("--mcool", default=str(XULAB / "hic/work_wt293t/coolers-hg38/HEK293TWT-MboI-R1-filtered.mcool"))
ap.add_argument("--loops", nargs="+", default=[str(LOOPDIR / "peakachu_5000.loops.0.7.bedpe"), str(LOOPDIR / "peakachu_10000.loops.0.7.bedpe")])
ap.add_argument("--frag-dir", default="/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment")
ap.add_argument("--chromhmm", default=str(XULAB / "data/HEK293T_chromHMM18.bed.gz"))
ap.add_argument("--zoom", nargs=2, default=["chr16:30008000-30036000", "chr16:30640000-30710000"])
ap.add_argument("--top", type=int, default=10)
a = ap.parse_args()
CACHE = INP / "hub_chr16_cache.npz"


def parse(r):
    c, se = r.split(":"); s, e = se.split("-"); return c, int(s), int(e)


E = pd.read_csv(a.edges, sep="\t").rename(columns={"on_loop_0.7": "loop"}); E["tfl"] = E.tfs.str.split(",")
elems = sorted({r for c in ("region1", "region2") for r in E[c]}, key=lambda r: parse(r)[1])
chrom = parse(elems[0])[0]; s0, e0 = parse(elems[0])[1], parse(elems[-1])[2]
pad = int(0.05 * (e0 - s0)); lo, hi = s0 - pad, e0 + pad
tf_n = pd.Series([t for l in E.tfl for t in l]).value_counts(); top = list(tf_n.index[:a.top]); rest = list(tf_n.index[a.top:])

if a.cache:
    import cooler
    arrs, meta = {}, {"tot": {}}
    for tf in tf_n.index:
        pos, n = [], 0
        with gzip.open(Path(a.frag_dir) / f"TF.{tf}.bed.gz", "rt") as fh:
            for ln in fh:
                n += 1; f = ln.split("\t", 6)
                if f[0] != chrom: continue
                p = int(f[1]) if f[5].startswith("+") else int(f[2]) - 1
                if lo <= p < hi: pos.append(p)
        arrs[f"pos_{tf}"] = np.array(pos, dtype=np.int64); meta["tot"][tf] = n; print(tf, len(pos), n, flush=True)
    clr = cooler.Cooler(f"{a.mcool}::/resolutions/10000")
    arrs["hic"] = clr.matrix(balance=True).fetch(f"{chrom}:{lo}-{hi}"); meta["binsize"] = int(clr.binsize)
    lp = []
    for f in a.loops:
        for ln in open(f):
            p = ln.split()
            if p[0] == chrom:
                m1, m2 = (int(p[1]) + int(p[2])) // 2, (int(p[4]) + int(p[5])) // 2
                if lo <= m1 and m2 <= hi: lp.append((m1, m2))
    arrs["loops"] = np.array(lp)
    st = []
    with gzip.open(a.chromhmm, "rt") as fh:
        for ln in fh:
            f = ln.split("\t")
            if f[0] == chrom and int(f[2]) > lo and int(f[1]) < hi: st.append((int(f[1]), int(f[2]), f[3]))
    arrs["st_s"] = np.array([x[0] for x in st]); arrs["st_e"] = np.array([x[1] for x in st]); arrs["st_l"] = np.array([x[2] for x in st])
    np.savez_compressed(CACHE, meta=json.dumps(meta), **arrs); print("cached", CACHE); raise SystemExit

C = np.load(CACHE, allow_pickle=False); meta = json.loads(str(C["meta"]))
cov = {tf: C[f"pos_{tf}"] for tf in tf_n.index}; tot = meta["tot"]
loops = [tuple(x) for x in C["loops"]]
G = pd.read_csv(a.genes, sep="\t")
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
         "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank",
         "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}
GCOL = {"enhancer": P.chromhmm.active_enh, "TSS/flank": P.chromhmm.tss_flank, "transcribed": P.chromhmm.transcribed, "repressed": P.chromhmm.repressed}
states = [(int(s), int(e), GROUP.get(str(l), "quiescent")) for s, e, l in zip(C["st_s"], C["st_e"], C["st_l"])]
EDGE, LOOPC, RAW, OTHER, SHADE = P.evidence.cicero_only, P.evidence.hic_loop, P.arms.raw, P.neutral.dark, P.highlight
el_mid = [(parse(r)[1] + parse(r)[2]) / 2 for r in elems]
G["tss"] = np.where(G.strand == "+", G.txStart, G.txEnd)
G["hub"] = [any(abs(t - m) <= 1500 for m in el_mid) for t in G.tss]
FS = 6


def arc(ax, x1, x2, h, color, lw, down=False, alpha=0.9):
    t = np.linspace(0, np.pi, 80); xs = (x1 + x2) / 2 - (x2 - x1) / 2 * np.cos(t)
    ax.plot(xs, (-1 if down else 1) * h * np.sin(t), color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def ew(n):   # edge line width by number of TFs (pt)
    return 0.35 + 0.2 * n


def coverage(x0, x1, binsz):
    rows = top + ([f"other {len(rest)} TFs"] if rest else [])
    edges = np.arange(x0, x1 + binsz, binsz); hs = []
    for name in rows:
        ts = rest if name.startswith("other") else [name]
        p = np.concatenate([cov[t] for t in ts]); n = sum(tot[t] for t in ts)
        h, _ = np.histogram(p, bins=edges); hs.append(h / n * 1e6)
    ctr = (edges[:-1] + edges[1:]) / 2; onhub = np.zeros(len(ctr), bool)
    for r in elems:
        _, es, ee = parse(r); onhub |= (ctr >= es - 500) & (ctr <= ee + 500)
    ymax = max(max(h[onhub].max() if onhub.any() else 0, np.percentile(h, 99.5), 1e-9) for h in hs) * 1.05
    return rows, hs, edges, ymax


def bare(ax, bottom=True):
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_visible(bottom); ax.spines["bottom"].set_color(P.neutral.light); ax.spines["bottom"].set_linewidth(0.3)


def label(ax, text, **kw):
    ax.text(-0.012, 0.5, text, transform=ax.transAxes, ha="right", va="center", fontsize=FS, **kw)


def shade(axs, x0, x1):
    for r in elems:
        _, s, e = parse(r)
        if e > x0 and s < x1:
            w = max(e - s, 0.0025 * (x1 - x0)); c = s + (e - s) / 2
            for ax in axs: ax.axvspan(c - w / 2, c + w / 2, color=SHADE, zorder=0, lw=0)


def draw_states(ax, x0, x1):
    for s, e, g in states:
        if e > x0 and s < x1 and g != "quiescent":
            ax.add_patch(Rectangle((max(s, x0), 0.1), min(e, x1) - max(s, x0), 0.8, color=GCOL[g], lw=0))
    ax.set_ylim(0, 1); bare(ax, bottom=False)


def gene_body(ax, r, y, x0, x1, col, h=0.32, lw=0.5):
    ax.plot([max(r.txStart, x0), min(r.txEnd, x1)], [y, y], color=col, lw=lw, solid_capstyle="butt")
    for s, e in zip(map(int, str(r.exonStarts).strip(",").split(",")), map(int, str(r.exonEnds).strip(",").split(","))):
        if e > x0 and s < x1:
            ax.add_patch(Rectangle((max(s, x0), y - h / 2), max(min(e, x1) - max(s, x0), 0.0008 * (x1 - x0)), h, color=col, lw=0))


def pack(genes, x0, x1, width_mm, fs=FS, gap_mm=1.0):
    """Lane per gene so neither the body nor its label (fs pt, italic) overlaps another."""
    bp_mm = (x1 - x0) / width_mm; lanes, out = [], []
    for r in genes.sort_values("txStart").itertuples():
        s, e = max(r.txStart, x0), min(r.txEnd, x1); lw = len(r.name) * fs * 0.56 * 0.3528 * bp_mm; xm = (s + e) / 2
        L, R = min(s, xm - lw / 2), max(e, xm + lw / 2)
        k = next((i for i, end in enumerate(lanes) if L > end + gap_mm * bp_mm), len(lanes))
        if k == len(lanes): lanes.append(0)
        lanes[k] = R; out.append((r, k, xm))
    return out, max(len(lanes), 1)


def mm_ticks(ax, x0, x1, step, fmt):
    t = np.arange(np.ceil(x0 / step) * step, x1 + 1, step); ax.set_xticks(t, [fmt(v) for v in t])
    ax.tick_params(axis="x", length=1.5, pad=1)

# ======================= panel f: full hub, 112 x 72 mm =======================
W, H = 112, 72; L0, AW = 17.5, 93.0
fig = fig_mm(W, H)
rows_f = [("hic", 12.3), ("loops", 3.3), ("edges", 5.4)] + [("cov", 2.4)] * (len(top) + (1 if rest else 0)) + [("hmm", 1.4), ("genes", 2.4), ("glab", 12.2)]
y = H - 1.0; ax_of = []
for kind, h in rows_f:
    y -= h; ax_of.append((kind, mm_axes(fig, W, H, L0, y, AW, h)))
    if kind in ("hic", "loops", "edges", "hmm"): y -= 0.25
for _, ax in ax_of: ax.set_xlim(lo, hi)
axs = dict()
for kind, ax in ax_of: axs.setdefault(kind, []).append(ax)
# Hi-C triangle
M = np.log10(np.nan_to_num(C["hic"]) + 1e-5); n = M.shape[0]; b = meta["binsize"]; x0b = (lo // b) * b
X, Y = np.meshgrid(np.arange(n + 1), np.arange(n + 1)); xs = x0b + (X + Y) / 2 * b; ys = (Y - X) / 2 * b
Mu = M.copy(); Mu[np.tril_indices(n, -1)] = np.nan
axh = axs["hic"][0]
axh.pcolormesh(xs, ys, np.ma.masked_invalid(Mu.T), cmap=P.seq_cmap(), vmin=np.nanpercentile(M, 50), vmax=np.nanpercentile(M, 99.5),
               shading="flat", rasterized=True)
axh.set_ylim(0, (hi - lo) / 2.0); axh.axis("off"); label(axh, "Hi-C (10 kb)")
# legends in the empty corners of the triangle
lg = axh.legend(handles=[plt.Line2D([], [], color=LOOPC, lw=1.2), plt.Line2D([], [], color=EDGE, lw=1.2), Rectangle((0, 0), 1, 1, color=SHADE)],
                labels=["Shared edge on a Hi-C loop", "Shared edge, Cicero only", "Hub element"], loc="upper left", bbox_to_anchor=(0, 1.04),
                fontsize=FS, handlelength=1.2, handletextpad=0.4, labelspacing=0.25, borderaxespad=0)
axh.add_artist(lg)
axh.legend(handles=[Rectangle((0, 0), 1, 1, color=GCOL[g]) for g in GCOL], labels=["Enhancer", "TSS/flank", "Transcribed", "Repressed"],
           loc="upper right", bbox_to_anchor=(1, 1.04), fontsize=FS, handlelength=0.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0, title=None)
axh.text(1.0, 1.04, "", transform=axh.transAxes)
axl = axs["loops"][0]
for m1, m2 in loops: arc(axl, m1, m2, (m2 - m1) / (hi - lo), P.neutral.light, 0.3, down=True, alpha=1.0)
axl.set_ylim(-0.75, 0.02); axl.axis("off"); label(axl, f"Loops (n = {len(loops)})")
axe = axs["edges"][0]
for r in E.sort_values("n_tfs").itertuples():
    m1, m2 = sorted(((parse(r.region1)[1] + parse(r.region1)[2]) / 2, (parse(r.region2)[1] + parse(r.region2)[2]) / 2))
    arc(axe, m1, m2, (m2 - m1) / (hi - lo), LOOPC if r.loop else EDGE, ew(r.n_tfs))
axe.set_ylim(0, 0.62); axe.axis("off"); label(axe, "Shared edges")
rows, hs, bins, ymax = coverage(lo, hi, 2000)
for k, (name, hh) in enumerate(zip(rows, hs)):
    ax = axs["cov"][k]; c = OTHER if name.startswith("other") else RAW
    ax.fill_between(bins[:-1], 0, np.minimum(hh, ymax), step="post", color=c, lw=0); ax.set_ylim(0, ymax); bare(ax)
    label(ax, name.replace("other", "Other") if name.startswith("other") else f"{name} ({tf_n[name]})")
cov_note = f"Coverage: CPM per 2 kb, 0–{ymax:.2g}"
draw_states(axs["hmm"][0], lo, hi); label(axs["hmm"][0], "ChromHMM")
# genes: all bodies in one lane (grey; hub promoters red, drawn on top); hub-promoter names below, vertical, spread apart
axg = axs["genes"][0]
for hub in (False, True):
    for r in G[(G.hub == hub) & (G.txEnd > lo) & (G.txStart < hi)].itertuples():
        gene_body(axg, r, 0, lo, hi, P.chromhmm.promoter if hub else P.neutral.mid, h=0.5, lw=0.4)
axg.set_ylim(-0.6, 0.6); axg.axis("off"); label(axg, "Genes")
axl2 = axs["glab"][0]; axl2.set_ylim(0, 1); bare(axl2)
hubg = G[G.hub & ~G.name.str.startswith("LOC")].drop_duplicates("name").sort_values("tss")
bp_mm = (hi - lo) / AW; sp = 2.35 * bp_mm; xs_ = list(hubg.tss.values.astype(float))
for _ in range(200):   # 1-D repel with minimum spacing, kept inside the axis
    for i in range(1, len(xs_)):
        d = xs_[i] - xs_[i - 1]
        if d < sp: xs_[i - 1] -= (sp - d) / 2; xs_[i] += (sp - d) / 2
    xs_ = list(np.clip(xs_, lo + 1.2 * bp_mm, hi - 1.2 * bp_mm))
for r, xl in zip(hubg.itertuples(), xs_):
    axl2.plot([r.tss, r.tss, xl], [1.0, 0.95, 0.89], color=P.chromhmm.promoter, lw=0.3, clip_on=False)
    axl2.text(xl, 0.87, r.name, rotation=90, ha="center", va="top", fontsize=FS, style="italic", color=P.chromhmm.promoter)
axl2.spines["bottom"].set_visible(True); axl2.spines["bottom"].set_color(P.neutral.ink); axl2.spines["bottom"].set_linewidth(0.5)
mm_ticks(axl2, lo, hi, 250000, lambda v: f"{v / 1e6:.2f}")
axl2.set_xlabel(f"{chrom} (Mb, hg38)", fontsize=FS, labelpad=1)
fig.canvas.draw(); bb = axl2.xaxis.label.get_window_extent()
fig.text(L0 / W, (bb.y0 + bb.y1) / 2 / fig.bbox.height, cov_note, ha="left", va="center", fontsize=FS, color=P.neutral.dark)
label(axl2, "Hub promoters", color=P.chromhmm.promoter)
shade(axs["loops"] + axs["edges"] + axs["cov"] + axs["hmm"] + axs["genes"], lo, hi)
save(fig, "fig3f_hub_chr16_browser")
print("hub promoter labels:", hubg.name.tolist())

# ======================= panel h: two zoom windows, 179 x 46 mm =======================
W, H = 179, 46; L0, GAP = 14.0, 7.0; CW = (W - L0 - GAP - 4.0) / 2
fig = fig_mm(W, H)
zs_ = [parse(z) for z in a.zoom]
gp = [pack(G[(G.txEnd > s) & (G.txStart < e)], s, e, CW) for _, s, e in zs_]; nl = max(n for _, n in gp)
rows_h = [("edges", 4.6)] + [("cov", 2.2)] * len(rows) + [("hmm", 1.6), ("genes", 3.3 * (nl - 1) + 3.9)]
for j, (_, zs, ze) in enumerate(zs_):
    x = L0 + j * (CW + GAP); y = H - 3.0; col = []
    for kind, h in rows_h:
        y -= h; ax = mm_axes(fig, W, H, x, y, CW, h); ax.set_xlim(zs, ze); col.append((kind, ax))
        if kind == "edges": y -= 0.25
        if kind == "cov" and len([1 for k, _ in col if k == "cov"]) == len(rows): y -= 0.4
        if kind == "hmm": y -= 0.25
    axe = col[0][1]
    for r in E.itertuples():
        for p_, q_ in ((r.region1, r.region2), (r.region2, r.region1)):
            pm = (parse(p_)[1] + parse(p_)[2]) / 2; qm = (parse(q_)[1] + parse(q_)[2]) / 2
            if zs <= pm <= ze:
                if zs <= qm <= ze:
                    if pm < qm: arc(axe, pm, qm, 0.8 * (qm - pm) / (ze - zs), LOOPC if r.loop else EDGE, ew(r.n_tfs))
                else:
                    axe.plot([pm, ze if qm > ze else zs], [0, 0.9], color=LOOPC if r.loop else EDGE, lw=ew(r.n_tfs), alpha=0.6)
    axe.set_ylim(0, 1.0); axe.axis("off")
    axe.text(0, 1.04, f"{chrom}:{zs:,}–{ze:,} (hg38)", transform=axe.transAxes, ha="left", va="bottom", fontsize=FS)
    rows, hs, bins, ymax = coverage(zs, ze, 500)
    axe.text(1, 1.04, f"CPM per 500 bp, 0–{ymax:.2g}", transform=axe.transAxes, ha="right", va="bottom", fontsize=FS, color=P.neutral.dark)
    covax = [ax for k, ax in col if k == "cov"]
    for k, (name, hh) in enumerate(zip(rows, hs)):
        ax = covax[k]; c = OTHER if name.startswith("other") else RAW
        ax.fill_between(bins[:-1], 0, np.minimum(hh, ymax), step="post", color=c, lw=0); ax.set_ylim(0, ymax); bare(ax)
        if j == 0: label(ax, name.replace("other", "Other"))
    hm = [ax for k, ax in col if k == "hmm"][0]; draw_states(hm, zs, ze)
    axg = col[-1][1]; packed, n_ = gp[j]
    for r, lane, xm in packed:
        yy = -lane * 3.3; c = P.chromhmm.promoter if r.hub else P.neutral.dark
        gene_body(axg, r, yy, zs, ze, c, h=0.9, lw=0.5)
        if zs <= r.tss <= ze:
            d = (1 if r.strand == "+" else -1) * 0.012 * (ze - zs)
            axg.plot([r.tss, r.tss, r.tss + d], [yy, yy + 0.9, yy + 0.9], color=c, lw=0.5)
        axg.text(xm, yy - 0.6, r.name, ha="center", va="top", fontsize=FS, style="italic", color=c, fontweight="bold" if r.hub else "normal")
    axg.set_ylim(-(nl - 1) * 3.3 - 2.6, 1.3); bare(axg, bottom=True); axg.spines["bottom"].set_color(P.neutral.ink); axg.spines["bottom"].set_linewidth(0.5)
    step = 5000 if ze - zs <= 30000 else 10000
    mm_ticks(axg, zs, ze, step, lambda v: f"{v / 1e6:.3f}")
    shade([ax for _, ax in col], zs, ze)
    if j == 0:
        label(axe, "Shared edges"); label(hm, "ChromHMM"); label(axg, "Genes")
save(fig, "fig3h_hub_chr16_zoom")
