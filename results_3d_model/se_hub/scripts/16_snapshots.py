"""One anchor, alternating loops: 3 typical structures (15_pick_snapshots.py) with the SE candidate at the same position, same scale, red petals for the closed loops, faint rest of the chain, closed partners coloured by dominant TF. Right: flower of all 12 loops, opacity and % = closure across 10,000 structures.
  render : python 16_snapshots.py --sim <traj dir>   (pymol-open-source)
  figure : python 16_snapshots.py --sim <traj dir> --figure"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--figure", action="store_true"); a = ap.parse_args()
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v4_snapshots"; N = 910; SN = pd.read_csv(O / "snapshots.tsv", sep="\t")
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"]; COL = dict(zip(TFS, ["#C48311", "#32A190", "#D9B637", "#C681F1", "#739B64", "#8C486E", "#4DC7E5", "#8AC975", "#65278F", "#712F55"]))
OTHER, INK, RED, SEC, GHOST = "#BDBDBD", "#222222", "#C0392B", "#4D4D4D", "#D0D0D0"
EL = pd.read_csv(Bd / "model/inputs/hub_elements.tsv", sep="\t"); core = int(EL.bead[EL.is_core].iloc[0]); S = pd.read_csv(Bd / "figure_3d_v2/partners_3d.tsv", sep="\t"); Sv = pd.read_csv(Bd / "figure_v2_cand/spokes.tsv", sep="\t").set_index("bead"); S["cls"] = [Sv.cls[b] for b in S.bead]; S["span_kb"] = [Sv.span_kb[b] for b in S.bead]
EC = pd.read_csv(Bd / "model/analysis/edge_closure.tsv", sep="\t"); S["freq"] = [float(EC[(EC[["b1", "b2"]].min(1) == min(core, b)) & (EC[["b1", "b2"]].max(1) == max(core, b))].closure_freq.iloc[0]) for b in S.bead]; S = S.sort_values("bead").reset_index(drop=True); S.to_csv(O / "partners_freq.tsv", sep="\t", index=False)
files = sorted(glob.glob(str(Path(a.sim) / "traj*.npz")))
def load(c):
    t_, k_ = divmod(c - 1, 50); return np.load(files[t_])["X"][k_, :N].astype(float)
if not a.figure:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd, cgo
    def rgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    def octa(c0, r, col):
        V = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]; o = [cgo.COLOR, *col]
        for a_ in V[:2]:
            for b_ in V[2:4]:
                for z_ in V[4:]:
                    o += [cgo.NORMAL, *((a_ + b_ + z_) / np.sqrt(3))] + sum(([cgo.VERTEX, *(c0 + r * v)] for v in (a_, b_, z_)), [])
        return o
    Ys, Ps, Rm = {}, {}, 0
    for c in SN.conf:
        X = load(c); Y = (X - X[core]) * 10.0; cl = [b for b in S.bead if np.linalg.norm(X[b] - X[core]) < 2]; pet = np.zeros(N, bool)
        for b in cl: pet[min(b, core):max(b, core) + 1] = True
        _, _, Vt = np.linalg.svd(Y[pet], full_matrices=False)
        if np.linalg.det(Vt) < 0: Vt[2] *= -1
        Y = Y @ Vt.T; Ys[c] = Y; Ps[c] = (pet, cl); Rm = max(Rm, np.abs(Y[pet][:, :2]).max())
    Rm *= 1.08; w = 3000; HH = {}
    for c in SN.conf:
        Y = Ys[c]; pet, cl = Ps[c]; cmd.reinitialize(); b0, b1 = S.bead.min(), S.bead.max() + 1; Z = gaussian_filter1d(Y, 2.5, axis=0, mode="nearest"); P = Z[b0:b1].copy()
        for b in np.r_[core, S.bead.values]: P[b - b0] = Y[b]
        Pd = np.array([np.interp(np.linspace(0, len(P) - 1, len(P) * 4), np.arange(len(P)), P[:, d]) for d in range(3)]).T; bo = b0 + np.minimum(np.round(np.linspace(0, len(P) - 1, len(Pd))).astype(int), len(P) - 1)
        g, r_ = rgb(GHOST), rgb(RED); gh, pe = [cgo.ALPHA, 0.3], []
        for u, v, b in zip(Pd[:-1], Pd[1:], bo[1:]):
            if pet[b]: pe += [cgo.CYLINDER, *u, *v, 1.5, *r_, *r_, cgo.COLOR, *r_, cgo.SPHERE, *v, 1.5]
            elif abs(u[0]) < Rm and abs(u[1]) < Rm: gh += [cgo.CYLINDER, *u, *v, 0.5, *g, *g]
        cmd.load_cgo(gh, "other"); cmd.load_cgo(pe, "petals"); an = [cgo.COLOR, *rgb(SEC), cgo.SPHERE, 0, 0, 0, 6.0]; oc = [cgo.BEGIN, cgo.TRIANGLES]
        for r in S[S.bead.isin(cl)].itertuples():
            col = rgb(COL.get(r.dominant_tf, OTHER) if isinstance(r.dominant_tf, str) else OTHER)
            if r.cls == "promoter": an += [cgo.COLOR, *col, cgo.SPHERE, *Y[r.bead], 4.2]
            else: oc += octa(Y[r.bead], 6.2, col)
        cmd.load_cgo(an, "anchors"); cmd.load_cgo(oc + [cgo.END], "elements")
        for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.005, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.5, specular=0.1, direct=0.5, reflect=0.25, orthoscopic=1, antialias=2, cgo_lighting=1, transparency_mode=3, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items(): cmd.set(k, v)
        cmd.viewport(w // 4, w // 4); cmd.reset()
        for i, q in enumerate(((Rm, Rm, 0), (-Rm, -Rm, 0), (Rm, -Rm, 0), (-Rm, Rm, 0))): cmd.pseudoatom("frame", pos=list(q), name=f"F{i}")
        cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=0, complete=1); cmd.delete("frame"); view = cmd.get_view(); cmd.png(str(O / f"render_{c}.png"), width=w, height=w, dpi=600, ray=1); cmd.save(str(O / f"render_{c}.pse"))
        R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); idx = np.r_[core, cl]; vc = (Y[idx] - org) @ R_ + cam; hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
        HH[str(c)] = hh; json.dump(dict(R=float(Rm), hh=HH), open(O / "frame.json", "w")); pd.DataFrame(dict(bead=idx, fx=0.5 * (1 + vc[:, 0] / hh), fy=0.5 * (1 + vc[:, 1] / hh))).to_csv(O / f"projection_{c}.tsv", sep="\t", index=False)
else:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon
    from PIL import Image
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    # left
    W, H = 8.05, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.text(0.06, H - 0.2, "One anchor, alternating loops", fontsize=14, fontweight="bold", va="center")
    IW = 2.62; y0 = 2.5; FR = json.load(open(O / "frame.json"))
    for k, c in enumerate(SN.conf):
        x0 = 0.06 + k * (IW + 0.03); ax.text(x0 + IW / 2, H - 0.5, f"Structure {k + 1}: {int(SN.n_closed[k])} loops closed", ha="center", va="center", fontweight="bold")
        im = Image.open(O / f"render_{c}.png").convert("RGBA"); f = FR["R"] / FR["hh"][str(c)]; w0 = im.size[0]; m = int(w0 * (1 - f) / 2); im = im.crop((m, m, w0 - m, w0 - m)); ax.imshow(np.asarray(im), extent=(x0, x0 + IW, y0, y0 + IW), interpolation="lanczos", zorder=0); ax.set_xlim(0, W); ax.set_ylim(0, H)
        PJ = pd.read_csv(O / f"projection_{c}.tsv", sep="\t"); PJ["gx"] = 0.5 + (PJ.fx - 0.5) / f; PJ["gy"] = 0.5 + (PJ.fy - 0.5) / f; ax.text(x0 + PJ.gx[0] * IW, y0 + PJ.gy[0] * IW, "SE", ha="center", va="center", color="white", fontweight="bold", fontsize=12, zorder=4)
        Q = S[S.bead.isin(PJ.bead.iloc[1:])].sort_values("freq", ascending=False)
        for j, r in enumerate(Q.itertuples()):
            yy = 2.34 - j * 0.23; ax.plot(x0 + 0.5, yy, "o" if r.cls == "promoter" else "D", ms=8.5 if r.cls == "promoter" else 7.5, color=COL.get(r.dominant_tf, OTHER) if isinstance(r.dominant_tf, str) else OTHER, mec="#333333", mew=0.5); ax.text(x0 + 0.65, yy, r.label, va="center", fontstyle="italic")
    y = 1.3; ax.plot([0.12, 0.45], [y, y], color=RED, lw=4, solid_capstyle="butt"); ax.text(0.53, y, "closed loop", va="center"); ax.plot(1.8, y, "o", ms=9, color=INK, mec="none"); ax.text(1.93, y, "promoter", va="center"); ax.plot(3.0, y, "D", ms=8, color=INK, mec="none"); ax.text(3.13, y, "other element", va="center"); ax.plot([4.5, 4.83], [y, y], color=GHOST, lw=2); ax.text(4.91, y, "rest of chain", va="center")
    seen = [t for t in TFS if t in set(S.dominant_tf.dropna())]; ax.text(0.1, 0.98, "Colour = strongest TF:", va="center")
    for k, (c_, l) in enumerate([(COL[t], t) for t in seen] + [(OTHER, "none of the 10")]):
        col, row = k % 5, k // 5; xx, yy = (2.05 + col * 1.2, 0.98) if row == 0 and col < 5 else (0.2 + col * 1.2, 0.72); ax.add_patch(Circle((xx, yy), 0.07, color=c_, lw=0)); ax.text(xx + 0.12, yy, l, va="center", fontstyle="italic" if l in TFS else "normal")
    ax.text(0.1, 0.32, "Typical structures (median 4 closed), chosen so that together they cover all 12 partners.", va="center", color="#444444")
    for e in ("svg", "pdf"): fig.savefig(O / f"snapshots.{e}", dpi=600, transparent=True)
    fig.savefig(O / "snapshots.png", dpi=600, facecolor="white"); fig.savefig(O / "snapshots_check.png", dpi=144, facecolor="white"); plt.close(fig)
    # right
    W, H = 5.0, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.text(0.06, H - 0.2, "Across all structures", fontsize=14, fontweight="bold", va="center")
    cx, cy, Rs = W / 2, 3.2, 0.34; n = len(S)
    for i, r in S.iterrows():
        th = np.pi / 2 - 2 * np.pi * i / n; Lp = 0.45 + 0.42 * np.log10(r.span_kb / 40); pol = lambda rr, aa: np.array([cx + rr * np.cos(aa), cy + rr * np.sin(aa)]); P0, P3 = pol(Rs * 0.8, th - 0.2), pol(Rs * 0.8, th + 0.2); P1, P2 = pol(Rs + 1.3 * Lp, th - 0.2), pol(Rs + 1.3 * Lp, th + 0.2); tt = np.linspace(0, 1, 200)[:, None]; Bz = (1 - tt) ** 3 * P0 + 3 * (1 - tt) ** 2 * tt * P1 + 3 * (1 - tt) * tt ** 2 * P2 + tt ** 3 * P3
        ax.fill(Bz[:, 0], Bz[:, 1], color=RED, alpha=float(np.clip(r.freq / 0.65, 0.08, 1)), lw=0, zorder=1); ax.plot(Bz[:, 0], Bz[:, 1], color=RED, lw=1.1, zorder=2, ls="-" if r.cls == "promoter" else (0, (1.2, 1.2)))
        b_ = pol(Rs + 0.17, th); ax.plot(*b_, "o" if r.cls == "promoter" else "D", ms=5.5 if r.cls == "promoter" else 4.8, mfc=INK, mec="white", mew=0.5, zorder=5)
        tip = Bz[np.argmax(np.hypot(Bz[:, 0] - cx, Bz[:, 1] - cy))]; u = (tip - [cx, cy]) / np.hypot(*(tip - [cx, cy])); ha = "left" if u[0] > 0.2 else "right" if u[0] < -0.2 else "center"
        ax.text(tip[0] + 0.07 * u[0], tip[1] + (0.2 * u[1] if abs(u[0]) <= 0.2 else 0.07 * u[1]), f"{r.label}\n{r.freq * 100:.0f}%", ha=ha, va="center", fontsize=12, fontstyle="italic", linespacing=0.95, color=INK)
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=Rs, fc="#F4F4F4", ec=SEC, lw=1.5, zorder=6)); ax.text(cx, cy, "SE\ncand.", ha="center", va="center", fontsize=12, fontweight="bold", color=SEC, linespacing=0.9, zorder=7)
    ax.text(W / 2, 0.82, "Darker petal, % = how often the loop closes\n(of 10,000 structures)", ha="center", va="center", linespacing=1.1)
    y = 0.42; ax.plot(1.15, y, "o", ms=8, color=INK, mec="none"); ax.text(1.27, y, "promoter", va="center"); ax.plot(2.3, y, "D", ms=7, color=INK, mec="none"); ax.text(2.42, y, "other element", va="center")
    ax.text(W / 2, 0.14, "Partners do not contact each other (Hi-C 0.92×)", ha="center", va="center", fontweight="bold", color=RED)
    for e in ("svg", "pdf"): fig.savefig(O / f"flower_all.{e}", dpi=600, transparent=True)
    fig.savefig(O / "flower_all.png", dpi=600, facecolor="white"); fig.savefig(O / "flower_all_check.png", dpi=144, facecolor="white")
