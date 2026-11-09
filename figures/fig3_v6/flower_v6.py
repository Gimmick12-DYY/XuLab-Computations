"""Fig 3 v6 panel d (right): the approved SE-hub flower (deck slide 23, results_3d_model/se_sedb/figure_chord_tf/flower_final),
re-run from se_sedb/scripts/42_chord_tf.py (flower part only, same inputs, colours, labels and edge legend).
v6 changes: partner-partner edges are drawn as petals that bulge OUTWARD from the ring (teardrop cubic Bezier leaving each end between
the radial and the bisector direction, tip on the bisector at Rg*(1.08 + 0.18*gap/pi)), like tf_polymer_hicfirst/figure_loopedges_flower_v3/loopedges_schematic; SE edges stay radial
spokes; partner nodes are small solid dots in the type colour (TF pies and their legend removed, user 2026-11-09); original
ring radius (2.05); partner labels pushed slightly outward past the petals; slide title removed.
  python flower_v6.py --sim <dir holding traj182.npz> --conf 9102 --out <dir>"""

import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_sedb"); ap.add_argument("--out", required=True); ap.add_argument("--sim", required=True); ap.add_argument("--conf", type=int, required=True); ap.add_argument("--maxp", type=int, default=150); ap.add_argument("--size", type=int, default=3000); a = ap.parse_args()
R0 = Path(a.root); O = R0 / "figure_chord_tf"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); I = R0 / "model/inputs"; N = 910; L0 = 430000; MAXP = a.maxp
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
TY = dict(promoter="#0072B2", enhancer="#D55E00", other="#BBBBBB"); TN = dict(promoter="promoter–promoter", enhancer="to an enhancer or the SE", other="other"); OTHER, INK, GHOST, NODE, SEC = "#BDBDBD", "#222222", "#D9D9D9", "#6A6A6A", "#262626"
H = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); B = pd.read_csv(I / "beads.tsv", sep="\t").fillna({"tfs": ""}); mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; tb = lambda p: (p - L0) // 2000; H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]; H["kb"] = (H.b1 - H.b2).abs() * 2; EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0"); EN = dict(se_promoter="promoter", se_enhancer="enhancer", se_other="other", pp_promoter="promoter\u2013promoter", pp_enhancer="involving an enhancer", pp_other="other"); H["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(H.cls_a, H.cls_b, H.from_se, H.type)]; se_b = tb((1248039 + 1254347) // 2); H["label_a"] = H.label_a.str.replace("RNU6-2/TMEM259", "TMEM259"); H["label_b"] = H.label_b.str.replace("RNU6-2/TMEM259", "TMEM259"); tfset = lambda b: set(x for x in B.tfs[b].split(",") if x)
Pn = {}
for r in H.itertuples():
    for k, l, b in ((r.a, r.label_a, r.b1), (r.b, r.label_b, r.b2)):
        if str(k) != "SE": Pn[str(k)] = (l, b)
P = pd.DataFrame([dict(key=k, label=v[0], bead=v[1]) for k, v in Pn.items()]); P["keyn"] = P.key.astype(int); P = P.sort_values("keyn").reset_index(drop=True); Tx = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); Tx["bin"] = [mid(x) // 10000 for x in Tx.partner]; CL = dict(zip(Tx.bin, Tx.cls)); EM = pd.read_csv(O / "enhancer_map.tsv", sep="\t").set_index("bin_10kb").name.to_dict(); P["cls"] = [CL[k] for k in P.keyn]; assert Tx.bin.is_unique and set(Tx.bin) == set(P.keyn); OWN = {b_: tb(mid(x)) for b_, x in zip(Tx.bin, Tx.partner)}; P["own_bead"] = [OWN[k] for k in P.keyn]; REG = dict(zip(Tx.bin, Tx.partner)); P["label"] = [EM.get(k, l) for k, l in zip(P.keyn, P.label)]; t_, k_ = divmod(a.conf - 1, 50); X = np.load(Path(a.sim) / f"traj{t_:03d}.npz")["X"][k_, :N].astype(float); H["closed"] = np.linalg.norm(X[H.b1] - X[H.b2], axis=1) < 2.0; H.to_csv(OUT / "hub_edges_in_structure.tsv", sep="\t", index=False)
if True:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon, PathPatch
    from matplotlib.path import Path as MP
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    # network flower (v6: no slide title; ring smaller and centred so the outward petals fit)
    W, H_ = 13.3, 6.2; fig = plt.figure(figsize=(W, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off(); cx, cy, Rg = 4.85, 3.1, 2.05; n = len(P); ang = {str(r.key): np.pi / 2 - 2 * np.pi * (i + 0.5) / n for i, r in P.iterrows()}; pt = lambda k: np.array([cx + Rg * np.cos(ang[k]), cy + Rg * np.sin(ang[k])])
    PIE0 = 0.0; HPMAX = Rg * 0.27
    for r in H.assign(o_=H.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["from_se", "on_loop", "o_"]).itertuples():
        c_ = EC[r.ek]; ls = "-" if r.on_loop == 1 else (0, (2.5, 2)); al = 1 if r.on_loop == 1 else 0.8 if r.from_se else 0.55; lw = 2.6 if (r.from_se and r.on_loop == 1) else 1.2 if r.on_loop == 1 else 1.1 if r.from_se else 0.5
        if r.from_se: k = str(r.b if r.a == "SE" else r.a); q = pt(k); ax.plot([cx, q[0]], [cy, q[1]], color=c_, lw=lw, ls=ls, alpha=al, zorder=3 + r.on_loop)
        else:   # v6: partner-partner edges bulge OUTWARD from the ring as petals (was: inner chords through the centre)
            q0, q1 = pt(str(r.a)), pt(str(r.b)); t0, t1 = ang[str(r.a)], ang[str(r.b)]; dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi; tm = t0 + dt / 2; um = np.array([np.cos(tm), np.sin(tm)]); C0 = np.array([cx, cy])
            u0, u1 = (q0 - C0) / Rg, (q1 - C0) / Rg; d0 = (u0 + um) / np.linalg.norm(u0 + um); d1 = (u1 + um) / np.linalg.norm(u1 + um); m_ = float(np.dot(d0 + d1, um))
            rpk = Rg + Rg * (0.08 + 0.18 * abs(dt) / np.pi); Lc = (rpk - Rg * np.cos(dt / 2)) * 8 / (3 * m_)   # teardrop petal: cubic Bezier whose tip reaches rpk on the bisector
            ax.add_patch(PathPatch(MP([q0, q0 + Lc * d0, q1 + Lc * d1, q1], [MP.MOVETO, MP.CURVE4, MP.CURVE4, MP.CURVE4]), fc="none", ec=c_, lw=lw, ls=ls, alpha=al, zorder=1 + r.on_loop + 0.1 * r.o_))
    from matplotlib.patches import Circle, Wedge
    PT = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; PC = dict(zip(PT, ["#E69F00", "#009E73", "#F0E442", "#CC79A7", "#117733", "#DDCC77", "#44AA99", "#999933", "#AA4499", "#A6761D"])); PIE = 0.075; RING = dict(promoter="#0072B2", enhancer="#D55E00", other="#8C8C8C"); PR = []
    tips = []
    for r in P.itertuples():
        q = pt(str(r.key)); tf_ = set(B.tfs[r.own_bead].split(",")) - {""}; tl_ = [t_ for t_ in PT if t_ in tf_]; no_ = len(tf_ - set(PT)); tot_ = len(tl_) + no_; st_ = 90.0; PR.append(dict(coordinates=REG[r.keyn], bin_10kb=r.keyn, bead=r.own_bead, node_bead_old=r.bead, label=r.label, cls=r.cls, n_tfs=tot_, named=",".join(tl_), n_other=no_, all_tfs=",".join(sorted(tf_))))
        ax.add_patch(Circle(q, PIE, fc=RING[r.cls], ec="none", zorder=6.2))   # v6 (user 2026-11-09): small solid dot in the type colour, no TF pie
        if r.cls != "other": tips.append((q[0], q[1], r.label, r.cls, ang[str(r.key)]))
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=0.34, fc="#FFC20A", ec="#5A4500", lw=1.6, zorder=7)); ax.text(cx, cy, "SE", ha="center", va="center", fontweight="bold", color="#2B2100", zorder=8, fontsize=13)
    for side, sub in ((1, [t for t in tips if t[0] >= cx]), (-1, [t for t in tips if t[0] < cx])):
        sub = sorted(sub, key=lambda t: t[1]); ROUT = Rg + PIE + HPMAX + 0.08; ys_ = [cy + ROUT * np.sin(t[4]) for t in sub]; lx_ = cx + side * (ROUT + 0.3)   # v6: labels pushed slightly out past the petals (ring kept)
        for _ in range(100):
            for i_ in range(1, len(ys_)):
                if ys_[i_] < ys_[i_ - 1] + 0.3: d_ = (ys_[i_ - 1] + 0.3 - ys_[i_]) / 2; ys_[i_] += d_; ys_[i_ - 1] -= d_
        for t, ty in zip(sub, ys_): ax.plot([cx + (Rg + PIE) * np.cos(t[4]), cx + ROUT * np.cos(t[4]), lx_], [cy + (Rg + PIE) * np.sin(t[4]), cy + ROUT * np.sin(t[4]), ty], color="#AAAAAA", lw=0.5, zorder=0); ax.text(lx_ + side * 0.04, ty, t[2], ha="left" if side > 0 else "right", va="center", fontsize=10.5, fontstyle="italic", color="#D55E00" if t[3] == "enhancer" else INK)
    TB = pd.DataFrame(PR); TB.to_csv(OUT / "chord_tf_binding.tsv", sep="\t", index=False); seen = set(",".join(TB.named).split(",")) - {""}
    lx, y = 9.35, 5.6;   # v6: legend shifted right with the labels
    ax.text(lx + 3.0, y, "loop + Cicero", va="center", ha="right", fontsize=11, color="#444444")
    for hd, keys, lw_ in (("From the super-enhancer", ("se_promoter", "se_enhancer", "se_other"), 3.5), ("Between partners", ("pp_promoter", "pp_enhancer", "pp_other"), 2.2)):
        ax.text(lx, y, hd, fontweight="bold", va="center"); y -= 0.29
        for k in keys:
            m = H.ek == k; ax.plot([lx, lx + 0.28], [y, y], color=EC[k], lw=lw_, solid_capstyle="butt"); ax.text(lx + 0.38, y, EN[k], va="center"); ax.text(lx + 3.0, y, f"{int((m & (H.on_loop == 1)).sum())} + {int((m & (H.on_loop == 0)).sum())}", va="center", ha="right"); y -= 0.29
        y -= 0.08
    ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.6); ax.text(lx + 0.38, y, f"on a Hi-C loop ({int((H.on_loop == 1).sum())})", va="center"); y -= 0.29; ax.plot([lx, lx + 0.28], [y, y], color="#555555", lw=1.2, ls=(0, (2.5, 2))); ax.text(lx + 0.38, y, f"Cicero only ({int((H.on_loop == 0).sum())})", va="center"); y -= 0.31
    for x_, c_, n_ in ((lx, "#0072B2", "promoter"), (lx + 1.2, "#D55E00", "enhancer"), (lx + 2.4, "#8C8C8C", "other")): ax.add_patch(Circle((x_ + 0.09, y), 0.075, fc=c_, ec="none")); ax.text(x_ + 0.23, y, n_, va="center")
    y -= 0.12
    y -= 0.2; ax.text(lx, y, "Clockwise in genomic order from chr19:0.51 Mb.", va="top", color="#444444", fontsize=11)
    for e in ("svg", "pdf"): fig.savefig(OUT / f"flower_v6.{e}", dpi=600, transparent=True)
    fig.savefig(OUT / "flower_v6.png", dpi=300, facecolor="white"); fig.savefig(OUT / "flower_v6_check.png", dpi=144, facecolor="white")
