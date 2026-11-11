# m17_fig5_panels Fig 5d draft (SPEC.md Amendment 2): one gene on its locus, bound TFs as circles. Illustration on an already-read benchmark.
# usage: make_gene_tf_panel.py [GENE ...]  (default: top candidate + PGAM1, HSPB1, TXNIP) -> out/gene_example/fig5d_<gene>_<impact|binding>.svg/.png
import subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, FancyArrowPatch
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 5, "pdf.fonttype": 42})
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; D = f"{RG}/m17_fig5_panels/out/gene_example"; GTF = "/work/users/d/y/dyy12/XuLab/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz"
GREY, PROM, ENH = "#BDBDBD", "#2C6FB7", "#E08214"; DOWN, UP = "#009E73", "#CC79A7"  # Okabe-Ito bluish green (down on knockdown = activator), reddish purple (up = repressor)
MM = 1 / 25.4; W = 85
C = pd.read_csv(f"{D}/p4_candidates.tsv", sep="\t"); GT = pd.read_csv(f"{D}/p4_gene_tf.tsv", sep="\t"); E = pd.read_csv(f"{D}/p4_elements.tsv", sep="\t"); L = pd.read_csv(f"{D}/p4_links.tsv", sep="\t").rename(columns={"ABC.Score": "abc"})
genes = sys.argv[1:] or [C.gene[C.figure_top_candidate].iloc[0], "PGAM1", "HSPB1", "TXNIP"]
NOTE = {("TXNIP", "MLX"): "not in confirmed set"}
def model(g, tss):
    out = subprocess.run(f"zcat {GTF} | grep -w 'gene_name \"{g}\";'", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    rows = [l.split("\t") for l in out]; T = [r for r in rows if r[2] == "transcript"]
    if not T: return None
    def tid(r): return r[8].split('transcript_id "')[1].split('"')[0]
    best = min(T, key=lambda r: abs((int(r[3]) if r[6] == "+" else int(r[4])) - tss)); t = tid(best)
    ex = [(int(r[3]), int(r[4])) for r in rows if r[2] == "exon" and tid(r) == t]; return dict(chr=best[0], start=int(best[3]), end=int(best[4]), strand=best[6], exons=ex, tid=t)
def draw(g, enc):
    H = GT[GT.gene == g].set_index("tf"); Lg = L[L.TargetGene == g]; Eg = E[E.gene == g]; tss = int(Lg.TargetGeneTSS.iloc[0]); gm = model(g, tss)
    lo = min(Lg.start.min(), gm["start"] if gm else tss); hi = max(Lg.end.max(), gm["end"] if gm else tss); pad = (hi - lo) * .06 + 500; lo, hi = lo - pad, hi + pad
    X = lambda p: 4 + (p - lo) / (hi - lo) * (W - 8)
    bound = Eg.sort_values("link_score", ascending=False).drop_duplicates("tf")  # strongest linked element per TF
    rows = []
    for r in bound.itertuples():
        h = H.loc[r.tf]; kd = bool(h.has_knockdown) and np.isfinite(h.z_all); x = X((r.element_start + r.element_end) / 2)
        if enc == "impact": rad = .55 if not kd else 0.75 * np.sqrt(abs(h.z_all))
        else: rad = 0.35 + 1.6 * np.sqrt(h.binding_pct)
        rows.append(dict(tf=r.tf, x=x, rad=min(rad, 3.2), kd=kd, z=h.z_all if kd else np.nan, conf=bool(h.confirmed_K10000), pct=h.binding_pct))
    R = pd.DataFrame(rows).sort_values(["conf", "rad"], ascending=False) if rows else pd.DataFrame(columns=["tf", "x", "rad", "kd", "z", "conf", "pct"])
    base = 28.5; placed = []
    for k, r in R.iterrows():  # greedy: lowest height at which the circle clears every placed circle; start at its element
        y = base + r.rad
        for _ in range(400):
            re_ = max(r.rad, .85); hit = [p for p in placed if np.hypot(p[0] - r.x, p[1] - y) < max(p[2], .85) + re_ + .25 or (abs(p[1] - y) < 1.5 and 0 <= r.x - p[0] < 7 and p[2] < 1.4)]
            if not hit: break
            y = max(p[1] + max(np.sqrt(max((max(p[2], .85) + re_ + .3) ** 2 - (p[0] - r.x) ** 2, 0)), 1.6) for p in hit) + .01
        placed.append((r.x, y, r.rad)); R.loc[k, "y"] = y
    top = max([p[1] + p[2] for p in placed] + [base + 6]) + 3; Hh = max(60, top + 7)
    fig = plt.figure(figsize=(W * MM, Hh * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.set_aspect("equal"); ax.axis("off")
    yc = 21.5; ax.plot([X(lo), X(hi)], [yc, yc], color=GREY, lw=2.2, solid_capstyle="butt", zorder=1)
    if gm:
        ax.plot([X(gm["start"]), X(gm["end"])], [yc, yc], color="k", lw=.5, zorder=3)
        for a, b in gm["exons"]: ax.add_patch(Rectangle((X(a), yc - .9), max(X(b) - X(a), .15), 1.8, color="k", lw=0, zorder=3))
        xt = X(gm["start"] if gm["strand"] == "+" else gm["end"]); dx = 2.2 if gm["strand"] == "+" else -2.2
        ax.plot([xt, xt], [yc, yc - 2.6], color="k", lw=.5); ax.add_patch(FancyArrowPatch((xt, yc - 2.6), (xt + dx, yc - 2.6), arrowstyle="-|>", mutation_scale=4, lw=.5, color="k"))
        ax.text(xt + dx * 1.2, yc - 2.6, g, fontsize=6, fontstyle="italic", va="center", ha="left" if dx > 0 else "right")
    xs = X(tss)
    for r in Lg.itertuples():  # elements and arcs to the TSS, line width by ABC score
        xe0, xe1 = X(r.start), X(r.end); col = PROM if r.self else ENH
        ax.add_patch(Rectangle((xe0, yc + 1.4), max(xe1 - xe0, .35), 1.4, color=col, lw=0, zorder=4))
        if not r.self:
            xm = (xe0 + xe1) / 2; w = abs(xm - xs); ax.add_patch(Arc(((xm + xs) / 2, yc + 2.8), w, min(w * .5, 7), theta1=0, theta2=180, lw=.3 + 3 * r.abc, color="#888888", zorder=2))
    for k, r in R.iterrows():
        ax.plot([r.x, r.x], [yc + 2.8, r.y - r.rad], color="#999999", lw=.25, zorder=5)
        if not r.kd:
            ax.add_patch(Circle((r.x, r.y), r.rad, fill=False, ec="#9E9E9E", lw=.4, zorder=6)); a = .4
        else:
            hue = DOWN if r.z < 0 else UP
            a = (1 if r.conf else .3) if enc == "impact" else float(np.clip(.15 + abs(r.z) / 8, .15, 1))
            ax.add_patch(Circle((r.x, r.y), r.rad, fc=hue, alpha=a, ec="none", zorder=6))
            if r.conf: ax.add_patch(Circle((r.x, r.y), r.rad, fill=False, ec="k", lw=.4, zorder=7))
        lab = r.tf + (" *" if (g, r.tf) in NOTE else ""); inside = len(lab) * .95 < 2 * r.rad - .2
        left = (not inside) and r.x + r.rad + .3 + len(lab) * (.95 if r.conf else .75) > W - 1
        ax.text(r.x if inside else (r.x - r.rad - .3 if left else r.x + r.rad + .3), r.y, lab, fontsize=4.5 if r.conf else 3.6, ha="center" if inside else ("right" if left else "left"), va="center", zorder=8,
                color=("white" if (inside and r.kd and a > .6) else ("k" if r.conf else "#555555")), fontweight="bold" if r.conf else "normal")
    # scale bar and legend strip below the locus
    span = hi - lo; sb = 10 ** np.floor(np.log10(span / 4)); sbmm = sb / span * (W - 8)
    ax.plot([X(lo), X(lo) + sbmm], [yc - 4.6, yc - 4.6], color="k", lw=.4); ax.text(X(lo) + sbmm / 2, yc - 5.2, (f"{sb / 1e6:g} Mb" if sb >= 1e6 else f"{sb / 1e3:g} kb"), fontsize=4.2, ha="center", va="top")
    ly = yc - 11; lx = 4
    if enc == "impact":
        for v in (2, 5, 10):
            rr = .75 * np.sqrt(v); ax.add_patch(Circle((lx + rr, ly - 1.4), rr, fill=False, ec="k", lw=.3)); ax.text(lx + 2 * rr + .4, ly - 1.4, f"|z| {v}", fontsize=4.2, va="center"); lx += 2 * rr + 5.5
        lab1 = "area = |z| of the gene on knockdown"
    else:
        for v in (.5, .9, 1.0):
            rr = .35 + 1.6 * np.sqrt(v); ax.add_patch(Circle((lx + rr, ly - 1.4), rr, fill=False, ec="k", lw=.3)); ax.text(lx + 2 * rr + .4, ly - 1.4, f"{v:g}", fontsize=4.2, va="center"); lx += 2 * rr + 3.5
        lab1 = "area = binding percentile; colour strength = |z|"
    ax.text(4, ly + .9, lab1, fontsize=4.2, va="bottom")
    lx2 = 44
    for col, t in ((DOWN, "gene down on knockdown"), (UP, "gene up on knockdown")): ax.add_patch(Circle((lx2 + .7, ly), .7, fc=col, ec="none")); ax.text(lx2 + 1.8, ly, t, fontsize=4.2, va="center"); ly -= 2
    ax.text(lx2, ly - .2, "outlined, bold = confirmed in both halves; faded = not confirmed;\nhollow grey = no knockdown data", fontsize=3.8, va="top")
    ax.add_patch(Rectangle((lx2, ly - 4.4), 1.4, .9, color=PROM, lw=0)); ax.text(lx2 + 1.9, ly - 4, "promoter", fontsize=4.2, va="center")
    ax.add_patch(Rectangle((lx2 + 13, ly - 4.4), 1.4, .9, color=ENH, lw=0)); ax.text(lx2 + 14.9, ly - 4, "ABC-linked enhancer", fontsize=4.2, va="center")
    side = [t for t in H.index[H.confirmed_K10000] if t not in set(R.tf)]; note = ""
    if side: note += "Confirmed, no linked peak: " + ", ".join(f"{t} (z {H.loc[t, 'z_all']:+.1f})" for t in side) + ".  "
    if any((g, t) in NOTE for t in R.tf): note += "* " + "; ".join(f"{t}: {v}" for (gg, t), v in NOTE.items() if gg == g) + "."
    if note: ax.text(4, Hh - 5.2, note, fontsize=4, va="top")
    ax.text(4, Hh - 2, f"{g}  {gm['chr'] if gm else Lg.chr.iloc[0]}:{int(lo):,}-{int(hi):,}  ({len(R)} TFs with a linked q005 peak; {int(H.confirmed_K10000.sum())} confirmed)", fontsize=5, va="top")
    for ext in ("svg", "png"): fig.savefig(f"{D}/fig5d_{g}_{enc}.{ext}", dpi=400)
    plt.close(fig); print(g, enc, "TFs drawn", len(R), "height mm", round(Hh, 1))
for g in genes:
    for enc in ("impact", "binding"): draw(g, enc)
