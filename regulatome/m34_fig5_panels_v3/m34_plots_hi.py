# m34 Amendment 1: panels h and i (threshold sweep) and the high-confidence link table. Descriptive.
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RG_ = "/work/users/d/y/dyy12/XuLab/regulatome"
sys.argv = ["m15.py", "raw"]; src = open(f"{RG_}/m15_gene_to_tf/m15.py").read(); exec(src[:src.index("# ---- binding percentiles")])
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
P = rp.use(); R34 = "/work/users/d/y/dyy12/XuLab/regulatome"; DD = f"{R34}/m34_fig5_panels_v3"; PL = f"{DD}/plots"; S35 = f"{R34}/m35_threshold_sweep/out"
OURS, BULK, INK, DARK, MID, LIGHT, PALE = P.arms.raw, P.neutral.ink, P.neutral.ink, P.neutral.dark, P.neutral.mid, P.neutral.light, P.neutral.pale; SLIDE = (12.9, 5.8)
def style(mode):
    rp.use()
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titlesize": 14, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9, "lines.linewidth": 1.8})
def save(fig, name, mode):
    if mode == "slide": fig.savefig(f"{PL}/slide/{name}.png", dpi=300); fig.savefig(f"{PL}/slide/{name}.svg")
    else: fig.savefig(f"{PL}/paper/{name}.png", dpi=600); fig.savefig(f"{PL}/paper/{name}.svg"); fig.savefig(f"{PL}/paper/{name}.pdf")
    plt.close(fig); print("saved", mode, name, flush=True)
T = pd.read_csv(f"{S35}/per_tf_threshold_sweep.tsv", sep="\t"); Po = pd.read_csv(f"{S35}/pooled_threshold_sweep.tsv", sep="\t"); TH = ["abs2", "abs3", "abs4", "abs5"]; XL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]
def panel_h(mode):
    style(mode); s = mode == "slide"; fs = 12 if s else 6
    fig = plt.figure(figsize=SLIDE if s else (7.0, 2.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28, left=0.08, right=0.98, bottom=0.2, top=0.88)
    ax = fig.add_subplot(gs[0]); g = Po[Po.set == "ours, all 75 TFs"].set_index("threshold").loc[TH]; x = np.arange(4)
    ax.plot(x, g.pooled_pct_recovered, "-o", color=OURS, ms=9 if s else 4, label="found in our Regulome (HEK293T)"); ax.plot(x, g.pooled_reference_pct, "--", color=MID, label="expected from the number of edges")
    for xi, (pc, nt, nl) in enumerate(zip(g.pooled_pct_recovered, g.n_tfs_ge10_responders, g.total_responders)): ax.text(xi, pc + 0.6, f"{nt} TFs\n{nl:,} links", ha="center", va="bottom", fontsize=fs * 0.9, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(XL); ax.set_xlabel("knockdown threshold"); ax.set_ylabel("% of Perturb-seq links found"); ax.set_ylim(0, 30); ax.set_xlim(-0.4, 3.4); ax.set_title("all TFs with a knockdown (pooled)", loc="left", fontsize=fs * 1.05)
    ax.legend(loc="lower left", frameon=False)
    ax2 = fig.add_subplot(gs[1]); o = Po[Po.set == "ours, 10 TFs with bulk ChIP"].set_index("threshold").loc[TH]; b = Po[Po.set.str.startswith("bulk")].set_index("threshold").loc[TH]
    ax2.plot(x, o.pooled_pct_recovered, "-o", color=OURS, ms=9 if s else 4, label="ours"); ax2.plot(x, o.pooled_reference_pct, "--", color=OURS, alpha=0.6, label="ours, expected")
    ax2.plot(x, b.pooled_pct_recovered, "-o", color=BULK, mfc="white", ms=9 if s else 4, label="bulk ChIP"); ax2.plot(x, b.pooled_reference_pct, "--", color=BULK, alpha=0.6, label="bulk, expected")
    for xi, nt in enumerate(o.n_tfs_ge10_responders): ax2.text(xi, 2, f"{nt} TFs", ha="center", fontsize=fs * 0.9, color=DARK)
    ax2.set_xticks(x); ax2.set_xticklabels(XL); ax2.set_xlabel("knockdown threshold"); ax2.set_ylabel("% of links found"); ax2.set_ylim(0, 80); ax2.set_xlim(-0.4, 3.4)
    ax2.set_title("10 TFs with bulk ChIP", loc="left", fontsize=fs * 1.05); ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=fs * 0.9)
    ax2.text(0.0, -0.2, "bulk references: CTCF HEK293T; NFYA, NRF1, E4F1 K562;\nMAZ, ZIC2, ZNF282, ZNF777, ZBTB7A, SETDB1 HEK293", transform=ax2.transAxes, fontsize=fs * 0.8, color=MID, ha="left", va="top")
    save(fig, "Fig5h_threshold_sweep", mode)
def panel_i(mode):
    style(mode); s = mode == "slide"; fs = 12 if s else 6; ours = T[T.source.str.startswith("our")]
    fig = plt.figure(figsize=SLIDE if s else (7.0, 3.4)); gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.62, left=0.07, right=0.99, bottom=0.12, top=0.93)
    NAMED = {"NRF1", "DNMT1", "E4F1", "ZNF143", "CTCF"}
    for row, (th, ttl) in enumerate((("rep2", "Tier-1 links (both cell halves |z| ≥ 2)"), ("abs4", "|z| ≥ 4"))):
        g = ours[ours.threshold == th].copy()
        if th == "abs4": g = g[g.enough_responders]
        g["diff"] = g.pct_recovered - g.reference_pct; g = g.sort_values(["enough_responders", "diff"], ascending=[False, False]).reset_index(drop=True); ax = fig.add_subplot(gs[row]); x = np.arange(len(g))
        for xi, r in g.iterrows():
            c = OURS if r.enough_responders else LIGHT; ax.plot([xi, xi], [r.reference_pct, r.pct_recovered], color=PALE if r.enough_responders else "#F2F2F2", lw=3 if s else 1.5, zorder=1)
            ax.scatter(xi, r.reference_pct, marker="_", s=120 if s else 25, color=MID if r.enough_responders else LIGHT, zorder=2, lw=2 if s else 1); ax.scatter(xi, r.pct_recovered, s=45 if s else 8, color=c, zorder=3)
        top6 = set(g[g.enough_responders].nlargest(6, "diff").TF); lab = [t if (t in NAMED or t in top6 or len(g) <= 20) else "" for t in g.TF]; ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90 if len(g) > 20 else 45, ha="center" if len(g) > 20 else "right", fontsize=fs)
        if len(g) > 20:
            for xi, t in enumerate(g.TF):
                if t in NAMED: ax.get_xticklabels()[xi].set_fontweight("bold")
            ax.tick_params(axis="x", length=0)
        ax.set_xlim(-1, len(g)); ax.set_ylabel("% of links found"); ax.set_title(f"{ttl}: {int(g.enough_responders.sum())} TFs with ≥ 10 links" + (f", grey = fewer" if (~g.enough_responders).any() else ""), loc="left", fontsize=fs * 1.05)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=OURS, label="found in our Regulome"), Line2D([], [], marker="_", ls="", color=MID, markersize=12, mew=2, label="expected from the TF's number of edges")], loc="upper right", ncol=2, frameon=False, fontsize=fs)
    save(fig, "Fig5i_high_confidence_per_tf", mode)
for m in ("paper", "slide"): panel_h(m); panel_i(m)
# high-confidence link table
X34 = "/work/users/d/y/dyy12/XuLab"; TFL = list(TF); gi = {g: k for k, g in enumerate(UG)}; OWN = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
for i, t in enumerate(TFL):
    for o in OWN.get(t, [t]):
        if o in gi: ZA[i, gi[o]] = Z1[i, gi[o]] = Z2[i, gi[o]] = np.nan
Rg = pd.read_csv(f"{R34}/m34_fig5_panels_v3/out/regulome_v1.tsv.gz", sep="\t", usecols=["tf", "gene", "peak_location", "best_abc", "best_element", "tier"]).set_index(["tf", "gene"]); rows = []
for i, t in enumerate(TFL):
    z, z1, z2 = ZA[i], Z1[i], Z2[i]; a = np.abs(np.nan_to_num(z)); rep = np.isfinite(z1) & np.isfinite(z2) & (np.abs(np.nan_to_num(z1)) >= 2) & (np.abs(np.nan_to_num(z2)) >= 2) & (np.sign(z1) == np.sign(z2)) & np.isfinite(z)
    for j in np.flatnonzero(rep | (a >= 4)):
        g = UG[j]; hit = (t, g) in Rg.index; r = Rg.loc[(t, g)] if hit else None
        rows.append(dict(TF=t, gene=g, knockdown_z=z[j], z_half1=z1[j], z_half2=z2[j], tier1_level=bool(rep[j]), abs_z_ge_4=bool(a[j] >= 4), direction="down on knockdown (TF activates)" if z[j] < 0 else "up on knockdown (TF represses)",
                         found_in_regulome="yes" if hit else "no", peak_location=r.peak_location if hit else "", best_element=r.best_element if hit else "", ABC=r.best_abc if hit else np.nan))
H = pd.DataFrame(rows).sort_values(["TF", "knockdown_z"]); H.to_csv(f"{S35}/high_confidence_links.tsv", sep="\t", index=False, float_format="%.4g")
sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps"); H.to_excel(f"{S35}/high_confidence_links.xlsx", index=False)
print("links", len(H), "tier1-level", int(H.tier1_level.sum()), "found", int((H.found_in_regulome == "yes").sum()), "| |z|>=4", int(H.abs_z_ge_4.sum()), "found", int(((H.found_in_regulome == "yes") & H.abs_z_ge_4).sum()))
