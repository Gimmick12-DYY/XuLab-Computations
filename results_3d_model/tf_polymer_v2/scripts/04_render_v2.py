"""v2 step 4: PyMOL ray traces of the typical hub-flower conformation (full view + zoom on the hub flower) and a gallery.

Chain = spline tube coloured by genomic position (viridis, 29.73 -> 31.55 Mb); clustered ON TFs = translucent spheres
(the hub-flower cluster in a darker blue); free / OFF TFs hidden; 9 hub cores = dark spheres; Cicero hub-component
edges = cylinders in clique colours, drawn whether or not the ends touch.
Full view: centred, PCA-rotated, then turned so the hub-flower centroid faces the camera.
Zoom: only chain segments within --zoom-r sigma of the hub-flower centroid, its TFs and the hub cores inside it.
1 sigma = 10 A. Transparent backgrounds. Writes projected hub-core pixel positions for the vector labels.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--width", type=int, default=3600)
ap.add_argument("--height", type=int, default=3000)
ap.add_argument("--zoom-r", type=float, default=3.5)
ap.add_argument("--gallery-width", type=int, default=1400)
a = ap.parse_args()
Bd = Path(a.base); A = Bd / "analysis"; O = Bd / "render"; O.mkdir(exist_ok=True)
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
CP = pd.read_csv(Bd / "inputs/pairs_cicero.tsv", sep="\t"); HE = CP[CP.set == "hub_edges"].reset_index(drop=True)
HB = B.index[B.hub_label != ""].values
Z = np.load(A / "representative.npz")
S = 10.0
VIR = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]) / 255
def vir(t): return np.array([np.interp(t, np.linspace(0, 1, 5), VIR[:, c]) for c in range(3)])

import pymol
pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, cgo
def hexrgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

def orient(X, focus):
    c = X[:N].mean(0); Y = X - c
    _, _, Vt = np.linalg.svd(Y[:N], full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    h = Y[focus].mean(0); t = np.arctan2(h[1], h[2])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Y @ Rx.T * S

def scene(Xo, L, LT, hc, bead_mask=None, tf_mask=None, tube_r=1.7):
    cmd.reinitialize()
    P = Xo[:N]; T = Xo[N:]
    tck, u_ = splprep(P.T, s=N * 0.15, k=3)
    tt = np.linspace(0, 1, N * 6); fine = np.array(splev(tt, tck)).T; bead_of = np.minimum((tt * (N - 1)).round().astype(int), N - 1)
    tube = []
    for k in range(len(fine) - 1):
        if bead_mask is not None and not (bead_mask[bead_of[k]] and bead_mask[bead_of[k + 1]]): continue
        c1, c2 = vir(bead_of[k] / (N - 1)), vir(bead_of[k + 1] / (N - 1))
        tube += [cgo.CYLINDER, *fine[k], *fine[k + 1], tube_r, *c1, *c2, cgo.COLOR, *c2, cgo.SPHERE, *fine[k + 1], tube_r]
    cmd.load_cgo(tube, "chromatin")
    show = (LT >= 0) if tf_mask is None else tf_mask
    for nm, sel, col, al in (("tf_clusters", show & (LT != hc), "#9DBBE3", 0.45), ("hub_flower", show & (LT == hc), "#4F7FC0", 0.55)):
        sph = [cgo.ALPHA, al, cgo.COLOR, *hexrgb(col)]
        for p in T[sel]: sph += [cgo.SPHERE, *p, 6.5]
        if sel.any(): cmd.load_cgo(sph, nm)
    hb = HB if bead_mask is None else HB[bead_mask[HB]]
    core = [cgo.COLOR, *hexrgb("#262626")]
    for b in hb: core += [cgo.SPHERE, *P[b], 8.5]
    if len(hb): cmd.load_cgo(core, "hub_cores")
    ed = []
    for r in HE.itertuples():
        if bead_mask is not None and not (bead_mask[r.b1] and bead_mask[r.b2]): continue
        c = hexrgb(r.color); ed += [cgo.CYLINDER, *P[r.b1], *P[r.b2], 1.4, *c, *c]
    if ed: cmd.load_cgo(ed, "hub_edges")
    for k, v in dict(ray_trace_mode=0, ambient_occlusion_mode=1, ambient_occlusion_scale=15, ray_shadows=0, ambient=0.4,
                     specular=0.1, direct=0.6, reflect=0.35, orthoscopic=0, depth_cue=0, antialias=2, field_of_view=20,
                     cgo_lighting=1, ray_opaque_background=0, transparency_mode=3).items():
        cmd.set(k, v)
    cmd.bg_color("white")

def shoot(Xo, png, w, h, buffer=1):
    cmd.viewport(int(w / 4), int(h / 4)); cmd.reset(); cmd.zoom("all", buffer=buffer, complete=1)
    view = cmd.get_view(); cmd.png(str(png), width=w, height=h, dpi=600, ray=1)
    R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15])
    vc = (Xo - org) @ R + cam
    hh = -vc[:, 2] * np.tan(np.radians(20.0) / 2); hw = hh * w / h
    return view, w / 2 * (1 + vc[:, 0] / hw), h / 2 * (1 - vc[:, 1] / hh), -vc[:, 2]

X, L, LT, ON, hc = Z["X"], Z["L"], Z["LT"], Z["ON"], int(Z["hub_cluster"])
focus = N + np.where(LT == hc)[0] if (LT == hc).any() else HB
Xo = orient(X, focus)
# full view
scene(Xo, L, LT, hc)
view, px, py, dep = shoot(Xo, O / "representative.png", a.width, a.height)
cmd.save(str(O / "representative.pse"))
hb = B.loc[HB]
pd.DataFrame(dict(bead=HB, label=hb.hub_label.values, cluster=L[HB], px=px[HB], py=py[HB], depth=dep[HB])).to_csv(O / "representative_hub_projection.tsv", sep="\t", index=False)
json.dump(dict(view=list(view), width=a.width, height=a.height, fov=20.0, sigma_A=S), open(O / "representative_view.json", "w"))
# zoom on the hub flower
cen = Xo[focus].mean(0)
bm = np.linalg.norm(Xo[:N] - cen, axis=1) < a.zoom_r * S
tm = (LT == hc)
scene(Xo, L, LT, hc, bead_mask=bm, tf_mask=tm, tube_r=2.0)
_, zx, zy, zd = shoot(Xo, O / "zoom_hub_flower.png", 2800, 2800, buffer=2)
pd.DataFrame(dict(bead=HB, label=hb.hub_label.values, in_zoom=bm[HB], px=zx[HB], py=zy[HB])).to_csv(O / "zoom_hub_projection.tsv", sep="\t", index=False)
seg = np.where(bm)[0]
pd.DataFrame(dict(bead=seg, mb=B.start.values[seg] / 1e6)).to_csv(O / "zoom_beads.tsv", sep="\t", index=False)
with open(O / "representative.pdb", "w") as f:
    for i, p in enumerate(Xo[:N]):
        f.write("ATOM  %5d  CA  BEA A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           C\n" % (i + 1, i % 10000, *p, L[i] + 1))
    for j, p in enumerate(Xo[N:]):
        f.write("HETATM%5d  O   TF  T%4d    %8.3f%8.3f%8.3f  %4.2f%6.2f           O\n" % (N + j + 1, j, *p, float(ON[j]), LT[j] + 1))
print("rendered conf", int(Z["conf"]), "hub cluster", hc, "TFs", int((LT == hc).sum()), "zoom beads", int(bm.sum()))
for g in range(len(Z["gallery"])):
    LTg, hcg = Z["LTG"][g], int(Z["hubclu_g"][g])
    fg = N + np.where(LTg == hcg)[0] if hcg >= 0 and (LTg == hcg).any() else HB
    Xg = orient(Z["XG"][g], fg)
    scene(Xg, Z["LG"][g], LTg, hcg)
    shoot(Xg, O / f"gallery_{g + 1}.png", a.gallery_width, a.gallery_width)
print("gallery", Z["gallery"].tolist())
