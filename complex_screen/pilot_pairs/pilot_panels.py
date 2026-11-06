# Pilot pair panels, one SVG per panel (palette C; Arial 7 pt, ticks 6; svg.fonttype none; dense marks rasterized at 600 dpi):
#   <pair>_cobinding.svg  element-level co-binding scatter (Pearson residuals) + layer-1 percentiles (v1 pooled, v2 stratified)
#   <pair>_browser.svg    example co-bound locus: raw 5' ends of both TFs and OmniATAC (CPM, 100 bp bins), elements, genes
#   <pair>_structure.svg  top Boltz-2 model (PyMOL PNG) with ipTM / ipSAE / tier
#   <pair>_motif.svg      motif logos, log2 OR co-bound vs GC-matched background with shuffled-motif null, GC and CpG o/e
import gzip, json, os, re, subprocess, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.font_manager import FontProperties
import matplotlib.transforms as mtrans

X = "/work/users/d/y/dyy12/XuLab"; P = f"{X}/complex_screen/pilot_pairs"; L3 = f"{P}/layer3_v2"; OUT = f"{P}/panels"
PAIRS = [("MAZ", "ZNF143"), ("EZH2", "MTF2"), ("MLX", "SREBF2"), ("CTCF", "STAG2"), ("RBBP7", "SRCAP")]
INK, GREY, NA = "#1A1A1A", "#8C8C8C", "#F2F2F2"
RAW, ATAC = "#D98C1F", "#6A51A3"
CLS = {"promoter": "#B03A2E", "distal_active": "#E8821E", "bivalent": "#55606E"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none",
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "savefig.dpi": 600,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})
os.makedirs(OUT, exist_ok=True)
def clean(ax): ax.spines[["top", "right"]].set_visible(False)

E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
Ea = E[E.cls.isin(["promoter", "distal_active", "bivalent"])].reset_index(drop=True)
L2 = pd.read_csv(f"{P}/layer2/scores.tsv", sep="\t").set_index("pair")
iface = {}

# ---- motif logos ----------------------------------------------------------------------------------------------------------
LETTER_COL = {"A": "#2E8B57", "C": "#1F5FA8", "G": "#E0A21B", "T": "#C0392B"}
FP = FontProperties(family="Arial", weight="bold")
def logo(ax, M, title):
    ic = np.log2(4) + (M * np.log2(np.clip(M, 1e-9, 1))).sum(1)
    for i, (row, h) in enumerate(zip(M, ic)):
        y = 0
        for j in np.argsort(row):
            b = "ACGT"[j]; ht = row[j] * h
            if ht <= 1e-3: continue
            tp = TextPath((0, 0), b, size=1, prop=FP); bb = tp.get_extents()
            tr = mtrans.Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, ht / bb.height).translate(i + 0.05, y)
            ax.add_patch(PathPatch(tr.transform_path(tp), color=LETTER_COL[b], lw=0)); y += ht
    ax.set_xlim(0, len(M)); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2]); ax.set_xticks([])
    ax.set_ylabel("bits", fontsize=6); ax.set_title(title, fontsize=6, pad=2); clean(ax)

def read_meme(path):
    txt = open(path).read(); out = {}
    for blk in txt.split("MOTIF ")[1:]:
        lines = blk.strip().split("\n"); mid = lines[0].split()[0]
        rows = [l for l in lines if l.strip() and l.strip()[0] in "0123456789." and len(l.split()) == 4]
        out[mid] = np.array([[float(x) for x in l.split()] for l in rows])
    return out

# ---- genes (GENCODE v46 basic) for the loci ----------------------------------------------------------------------------------
def genes_in(chrom, s, e):
    cmd = (f"zcat {X}/data/gencode_v46/gencode.v46.basic.annotation.gtf.gz | awk -F'\\t' '$1==\"{chrom}\" && $4<{e} && $5>{s} "
           f"&& ($3==\"exon\" || $3==\"transcript\")' | grep -E 'gene_type \"(protein_coding|lncRNA)\"' | grep 'Ensembl_canonical'")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip().split("\n")
    tx, ex = {}, {}
    for l in out:
        if not l: continue
        c = l.split("\t"); g = re.search(r'gene_name "([^"]+)"', c[8]).group(1)
        if c[2] == "transcript": tx[g] = (int(c[3]), int(c[4]), c[6])
        else: ex.setdefault(g, []).append((int(c[3]), int(c[4])))
    return tx, ex

def frag_cov(tf, chrom, s, e, bins):
    f = f"/vast/som/xujie_lab/TAGATG-293T/bigwig/fragment/TF.{tf}.bed.gz"
    cmd = f"zcat {f} | awk -F'\\t' '$1==\"{chrom}\" && $2>={s} && $2<{e}{{print $2}}'"
    pos = np.array([int(x) for x in subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.split()], int)
    tot = json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{tf}.json"))["reads"]
    return np.histogram(pos, bins)[0] / tot * 1e6

def atac_cov(chrom, s, e, bins):
    f = f"{X}/regulatome/m2_abc/inputs/omniatac_merged.tn5.tagAlign.gz"
    out = subprocess.run(f"tabix {f} {chrom}:{s}-{e}", shell=True, capture_output=True, text=True).stdout.strip().split("\n")
    pos = []
    for l in out:
        if not l: continue
        c = l.split("\t"); pos.append(int(c[1]) if c[5] == "+" else int(c[2]) - 1)
    tot = 36066598
    return np.histogram(np.array(pos, int), bins)[0] / tot * 1e6

for (a, b), SET in [(p_, s_) for p_ in PAIRS for s_ in ("distal", "promoter")]:
    tag0 = f"{a}_{b}"; d = f"{L3}/{tag0}/{SET}"; tag = f"{tag0}_{SET}"
    if not os.path.exists(f"{d}/summary.json"): print(tag, "no layer-3 output"); continue
    S = json.load(open(f"{d}/summary.json")); Z = np.load(f"{d}/scatter.npz", allow_pickle=True)
    # 1) co-binding scatter
    fig = plt.figure(figsize=(2.6, 3.6 if SET == "promoter" else 3.3)); ax = fig.add_axes([0.2, 0.45 if SET == "promoter" else 0.4, 0.72, 0.5])
    sm = Z["setmask"]; xa, yb = np.clip(Z["a"], -3, 20), np.clip(Z["b"], -3, 20); co = Z["cobound"]
    xa, yb = np.where(sm, xa, np.nan), np.where(sm, yb, np.nan)
    ax.hexbin(xa[sm], yb[sm], gridsize=60, bins="log", cmap="Greys", mincnt=1, linewidths=0, rasterized=True)
    ax.scatter(xa[co], yb[co], s=1.2, color=RAW, lw=0, rasterized=True, label=f"co-bound {SET} (n = {S['n_selected']:,})")
    ta, tb = Z["a"][Z["top_a"]].min(), Z["b"][Z["top_b"]].min()
    ax.axvline(ta, color=GREY, lw=0.5, ls="--"); ax.axhline(tb, color=GREY, lw=0.5, ls="--")
    ax.set_xlabel(f"{a} obs/exp (Pearson residual)"); ax.set_ylabel(f"{b} obs/exp (Pearson residual)"); clean(ax)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.2, markerscale=4)
    l1 = S["layer1"]
    def g(src, k): v = l1[src].get(k); return "NA" if v is None else f"{v:.2f}"
    txt = (f"{SET} elements (n = {S['n_set']:,}); per TF top 5% Pearson residual, >= 3 reads, residual > 0 ({S['n_top'][0]:,} / {S['n_top'][1]:,}); "
           f"expected {S['expected']:.0f}, {S['enrichment']:.2f}x, p = {S['p_hyper']:.1e}\n"
           f"pct v1 pooled: Pearson {g('v1_pooled', 'pearson_domain_pct')}, SpQN {g('v1_pooled', 'spqn_chromhmm_pct')}, "
           f"Cicero {g('v1_pooled', 'cicero_excess_z_pct')}\n"
           f"pct v2 stratified: Pearson {g('v2_stratified', 'pearson_domain_pct')}, SpQN {g('v2_stratified', 'spqn_noquies_pct')}, "
           f"Cicero {g('v2_stratified', 'cicero_excess_z_pct')}, compartment {g('v2_stratified', 'compartment_spqn_pct')}\n"
           f"R1 v2 edge: {l1['v2_stratified']['edge_type']}")
    if SET == "promoter":
        txt += ("\nPromoter sets are capped at 1,990 per TF (5%), so expected overlap is ~100 for every pair; any two of MAZ, ZNF143, EZH2, "
                "MTF2 overlap at 1.1-1.35x (e.g. MAZ-EZH2 134 vs MAZ-ZNF143 132)")
    txt = "\n".join(textwrap.fill(l, 78) for l in txt.split("\n"))
    fig.text(0.02, 0.01, txt, fontsize=5, va="bottom", ha="left")
    fig.savefig(f"{OUT}/{tag}_cobinding.svg"); plt.close(fig)

    # 2) browser at the strongest co-bound element (highest min of the two residuals), +/- 20 kb
    if SET != "distal" or not len(co): i = None
    else: i = co[np.argmax(np.minimum(Z["a"][co], Z["b"][co]))]
    el = Ea.iloc[i] if i is not None else None
    if el is not None:
      mid = (el.start + el.end) // 2; s, e = mid - 20000, mid + 20000; bins = np.arange(s, e + 100, 100)
      tracks = [(f"{a} raw", frag_cov(a, el.chrom, s, e, bins), RAW), (f"{b} raw", frag_cov(b, el.chrom, s, e, bins), RAW),
                ("OmniATAC", atac_cov(el.chrom, s, e, bins), ATAC)]
      tx, ex = genes_in(el.chrom, s, e)
      fig, axs = plt.subplots(5, 1, figsize=(3.4, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.35, 0.8], hspace=0.15))
      for ax, (lab, y, col) in zip(axs[:3], tracks):
          ax.bar(bins[:-1], y, width=100, align="edge", color=col, lw=0); ax.set_ylim(0, max(y.max() * 1.05, 1e-9))
          ax.set_yticks([0, round(float(y.max()), 1)]); ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=6); clean(ax)
          ax.tick_params(axis="x", length=0)
      ax = axs[3]
      w = Ea[(Ea.chrom == el.chrom) & (Ea.end > s) & (Ea.start < e)]
      for r in w.itertuples():
          ax.add_patch(Rectangle((r.start, 0.2), r.end - r.start, 0.6, color=CLS.get(r.cls, GREY), lw=0))
      ax.add_patch(Rectangle((el.start, 0.05), el.end - el.start, 0.9, fill=False, ec=INK, lw=0.6))
      ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("elements", rotation=0, ha="right", va="center", fontsize=6)
      ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="x", length=0)
      ax = axs[4]
      for k, (gname, (gs, ge, strand)) in enumerate(sorted(tx.items(), key=lambda x: x[1][0])):
          yy = 0.75 - 0.5 * (k % 2)
          ax.plot([max(gs, s), min(ge, e)], [yy, yy], color=INK, lw=0.5)
          for xs, xe in ex.get(gname, []): ax.add_patch(Rectangle((xs, yy - 0.12), xe - xs, 0.24, color=INK, lw=0))
          ax.text(min(max((gs + ge) / 2, s + 1500), e - 1500), yy + 0.16, f"{gname} ({strand})", fontsize=5, ha="center", va="bottom")
      ax.set_ylim(0, 1.1); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
      ax.set_xlim(s, e); ax.set_xticks([s, mid, e]); ax.set_xticklabels([f"{s:,}", f"{el.chrom}:{mid:,}", f"{e:,}"], fontsize=5)
      axs[0].set_title(f"{tag}: strongest co-bound element {el.eid} ({el.cls}); CPM per 100 bp", fontsize=6, pad=3)
      fig.savefig(f"{OUT}/{tag}_browser.svg", bbox_inches="tight"); plt.close(fig)

    # 3) structure
    png = f"{OUT}/{tag0}_structure.png"
    fig = plt.figure(figsize=(2.6, 2.9)); ax = fig.add_axes([0.02, 0.3, 0.96, 0.68])
    if os.path.exists(png): ax.imshow(plt.imread(png));
    else: ax.text(0.5, 0.5, "model pending", ha="center", va="center", color=GREY)
    ax.axis("off")
    r = L2.loc[tag0] if tag0 in L2.index else None
    def fmt(k): return "NA" if r is None or k not in r or pd.isna(r[k]) else (f"{r[k]:.2f}" if isinstance(r[k], (float, np.floating)) else str(r[k]))
    ifj = f"{OUT}/{tag0}_structure_iface.json"
    ii = json.load(open(ifj)) if os.path.exists(ifj) else {}
    lines = [f"{a} (blue) - {b} (orange)", f"model: {fmt('rendered_model')}",
             f"pilot (screen chains): ipTM {fmt('pilot_iptm_top')}, ipSAE {fmt('pilot_ipsae_max')}, {fmt('pilot_tier')}",
             f"screen: ipTM {fmt('screen_iptm_top')}, ipSAE {fmt('screen_ipsae_max')}"]
    if r is not None and "calib_construct" in r and pd.notna(r.get("calib_construct")):
        lines.append(f"calibration {fmt('calib_construct')}: ipTM {fmt('calib_iptm_top')}, ipSAE {fmt('calib_ipsae_max')}, {fmt('calib_tier')}")
    if ii: lines.append(f"interface residues: {ii['interface_res_A']} / {ii['interface_res_B']} (sticks, 5 A)")
    fig.text(0.03, 0.02, "\n".join(lines), fontsize=5, va="bottom")
    if SET == "distal": fig.savefig(f"{OUT}/{tag0}_structure.svg")
    plt.close(fig)

    # 4) motifs + GC / CpG
    Mt = pd.read_csv(f"{d}/motif_enrichment.tsv", sep="\t") if os.path.getsize(f"{d}/motif_enrichment.tsv") > 5 else pd.DataFrame()
    mats = read_meme(f"{d}/motifs.meme") if os.path.exists(f"{d}/motifs.meme") else {}
    own = Mt[~Mt.motif.str.startswith("CAP_")] if len(Mt) else Mt
    if len(own) and tag == "RBBP7_SRCAP": own = own  # negative pair: pilot motifs as specificity check
    cap = Mt[Mt.motif.str.startswith("CAP_")].sort_values("log2OR", ascending=False).head(4) if len(Mt) else Mt
    show = pd.concat([own, cap]) if len(Mt) else Mt
    nlogo = min(len(own), 3) + min(len(cap), 2) if len(Mt) else 0
    fig = plt.figure(figsize=(4.2, 3.6))
    lg = list(own.head(3).iterrows()) + list(cap.head(max(0, 4 - min(len(own), 3))).iterrows())
    for k, (_, rr) in enumerate(lg):
        lax = fig.add_axes([0.07, 0.86 - k * 0.135, 0.3, 0.085])
        nm = (rr.motif.split("|")[0] + " " + rr.motif.split("|")[1]) if not rr.motif.startswith("CAP_") else "CAP " + rr.motif.split("_")[1] + "_" + rr.motif.split("_")[2]
        logo(lax, mats[rr.motif], nm)
    bx = fig.add_axes([0.6, 0.55, 0.37, 0.4])
    if len(show):
        y = np.arange(len(show))[::-1]
        for yy, (_, rr) in zip(y, show.iterrows()):
            sh = [rr.shuffle_mean, rr.shuffle_max]
            bx.plot([min(sh[0], rr.shuffle_max), rr.shuffle_max], [yy, yy], color=GREY, lw=2, solid_capstyle="butt")
            bx.plot(rr.log2OR, yy, "o", color=INK, ms=3)
        lab = [(m.split("|")[0] + " " + m.split("|")[1]) if not m.startswith("CAP_") else "CAP " + m.split("_")[1] + "_" + m.split("_")[2]
               for m in show.motif]
        bx.set_yticks(y); bx.set_yticklabels(lab, fontsize=5); bx.axvline(0, color=GREY, lw=0.5)
        bx.set_xlabel("log2 OR vs matched background", fontsize=6)
    else:
        bx.text(0.5, 0.5, "no known motif for either TF", ha="center", va="center", fontsize=6, color=GREY, transform=bx.transAxes); bx.set_yticks([])
    clean(bx)
    gx = fig.add_axes([0.6, 0.27, 0.15, 0.16]); cx = fig.add_axes([0.82, 0.27, 0.15, 0.16])
    for ax_, key, lab in ((gx, "gc", "GC"), (cx, "cpg_oe", "CpG o/e")):
        vals = [S[key]["cobound"], S[key]["background"], S[key]["all_in_set"]]
        ax_.bar([0, 1, 2], vals, color=[INK, GREY, NA], ec=INK, lw=0.4, width=0.7)
        ax_.set_xticks([0, 1, 2]); ax_.set_xticklabels(["co", "bg", "all"], fontsize=5); ax_.set_title(lab, fontsize=6, pad=2); clean(ax_)
    sp_ = S.get("spacing", {})
    note = f"{SET}: co-bound {S['n_selected']:,}, bg {S['n_background']:,} (union ATAC, GC x CpG o/e matched); FIMO p < 1e-4; grey = 20 split-half shuffles (mean-max)"
    if sp_.get("chi2_p_distance") is not None:
        note += f"\nspacing ({a}-{b} best hits): chi2 p = {sp_['chi2_p_distance']:.2g}; same strand co {sp_['frac_same_strand']['cobound']:.2f} vs bg {sp_['frac_same_strand']['background']:.2f}"
    if tag0 == "MLX_SREBF2":
        note += ("\nCAP-SELEX: MLX-SREBF2 was tested (SREBF2 prey x MLX bait) and NOT called (0). MLX is called with 23 partners, mostly ETS "
                 "(ELF1/2, ERG, FLI1, GABPA, ETV4; composite) - none bHLH-ZIP; CAP rows here are those MLX/SREBF2 PCMs, not evidence for this pair.")
    note = "\n".join(textwrap.fill(l, 110) for l in note.split("\n"))
    fig.text(0.02, 0.005, note, fontsize=5, va="bottom")
    fig.savefig(f"{OUT}/{tag}_motif.svg"); plt.close(fig)
    print(tag, "panels written", "" if el is None else f"; locus {el.eid} {el.chrom}:{s}-{e}")


# ---- companion panel: CTCF motif calibration on all residual-selected CTCF elements (not co-bound) -------------------------
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.8), gridspec_kw=dict(wspace=0.6, top=0.62, bottom=0.3))
for ax, SET in zip(axs, ("distal", "promoter")):
    d = f"{L3}/CTCF_only/{SET}"; S = json.load(open(f"{d}/summary.json")); M = pd.read_csv(f"{d}/motif_enrichment.tsv", sep="\t").iloc[0]
    sh = pd.read_csv(f"{d}/fimo_cobound.tsv", sep="\t", usecols=[0]).iloc[:, 0]  # not used for values; keeps the panel tied to the run
    ax.plot([M.shuffle_mean, M.shuffle_max], [0, 0], color=GREY, lw=3, solid_capstyle="butt", label="20 shuffles (mean-max)")
    ax.plot(M.log2OR, 0, "o", color=INK, ms=4, label="CTCF MA0139.2")
    ax.axvline(0, color=GREY, lw=0.5); ax.set_yticks([]); clean(ax); ax.spines["left"].set_visible(False)
    ax.set_xlabel("log2 OR vs GC x CpG-matched bg", fontsize=6)
    ax.set_title(f"{SET}: {M.frac_cobound:.1%} vs {M.frac_bg:.1%}\n(n = {S['n_selected']:,} CTCF elements)", fontsize=6, pad=2)
axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.75), ncol=2, fontsize=5)
fig.text(0.01, 0.99, "CTCF calibration: all residual-selected CTCF elements\n(top 5%, >= 3 reads, residual > 0; not co-bound)", fontsize=6, va="top")
fig.savefig(f"{OUT}/CTCF_calibration_motif.svg", bbox_inches="tight"); plt.close(fig)
print("CTCF calibration panel written")
