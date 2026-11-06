"""Step 4: PyMOL ray trace of the representative conformation (and an optional 6-conformation gallery).

Grey spline tube for the chain, translucent spheres for TF particles in DBSCAN clusters, dark spheres for the 9 hub cores,
Cicero hub-component edges as cylinders in their clique colour (drawn whether or not the two ends touch).
Coordinates are centred and PCA-rotated (PC1 -> x, PC2 -> y); 1 sigma = 10 A in the PDB/PSE.
Transparent background. Writes the projected pixel position of every hub core and edge end for the vector label overlay.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--width", type=int, default=3600)
ap.add_argument("--height", type=int, default=3000)
ap.add_argument("--gallery-width", type=int, default=1300)
a = ap.parse_args()
Bd = Path(a.base); A = Bd / "analysis"; O = Bd / "render"; O.mkdir(exist_ok=True)
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
CP = pd.read_csv(Bd / "inputs/pairs_cicero.tsv", sep="\t"); HE = CP[CP.set == "hub_edges"].reset_index(drop=True)
Z = np.load(A / "representative.npz")
S = 10.0

import pymol
pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, cgo

def hexrgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

HB = B.index[B.hub_label != ""].values
def orient(X):
    """centre; PCA (PC1 -> x); then rotate about x so the hub-core centroid points at the camera (+z)"""
    c = X[:N].mean(0); Y = X - c
    _, _, Vt = np.linalg.svd(Y[:N], full_matrices=False)
    if np.linalg.det(Vt) < 0: Vt[2] *= -1
    Y = Y @ Vt.T
    h = Y[HB].mean(0); t = np.arctan2(h[1], h[2])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Y @ Rx.T * S

def scene(X, L, LT, tag, labels=True):
    cmd.reinitialize()
    P = X[:N]; T = X[N:]
    tck, _ = splprep(P.T, s=N * 0.15, k=3); fine = np.array(splev(np.linspace(0, 1, N * 6), tck)).T
    tube = [cgo.ALPHA, 0.55]
    g = hexrgb("#A0A0A0")
    for p, q in zip(fine[:-1], fine[1:]):
        tube += [cgo.CYLINDER, *p, *q, 1.7, *g, *g]
    for p in fine[1:-1]:
        tube += [cgo.COLOR, *g, cgo.SPHERE, *p, 1.7]
    cmd.load_cgo(tube, "chromatin")
    # TF particles in clusters (translucent), free TFs hidden
    tfc = hexrgb("#7FA7D9")
    sph = [cgo.ALPHA, 0.55, cgo.COLOR, *tfc]
    for p in T[LT >= 0]:
        sph += [cgo.SPHERE, *p, 6.0]
    cmd.load_cgo(sph, "tf_clusters")
    # hub cores
    hb = B.index[B.hub_label != ""].values
    core = [cgo.COLOR, *hexrgb("#262626")]
    for b in hb: core += [cgo.SPHERE, *P[b], 8.5]
    cmd.load_cgo(core, "hub_cores")
    # hub edges
    ed = []
    for r in HE.itertuples():
        c = hexrgb(r.color)
        ed += [cgo.CYLINDER, *P[r.b1], *P[r.b2], 1.4, *c, *c]
    cmd.load_cgo(ed, "hub_edges")
    for k, v in dict(ray_trace_mode=0, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1,
                     ambient_occlusion_scale=15, ray_shadows=0, ambient=0.4, specular=0.1, direct=0.6, reflect=0.35,
                     orthoscopic=0, depth_cue=0, antialias=2, field_of_view=20, cgo_lighting=1, ray_opaque_background=0,
                     cgo_transparency=0.0, transparency_mode=3).items():
        cmd.set(k, v)
    cmd.bg_color("white")

def render(X, L, LT, png, w, h):
    Xo = orient(X)
    scene(Xo, L, LT, png)
    cmd.viewport(int(w / 4), int(h / 4)); cmd.reset(); cmd.zoom("all", buffer=1, complete=1)
    view = cmd.get_view()
    cmd.png(str(png), width=w, height=h, dpi=600, ray=1)
    R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15])
    vc = (Xo - org) @ R + cam
    hh = -vc[:, 2] * np.tan(np.radians(20.0) / 2); hw = hh * w / h
    px = w / 2 * (1 + vc[:, 0] / hw); py = h / 2 * (1 - vc[:, 1] / hh)
    return Xo, view, px, py, -vc[:, 2]

Xo, view, px, py, dep = render(Z["X"], Z["L"], Z["LT"], O / "representative.png", a.width, a.height)
cmd.save(str(O / "representative.pse"))
hb = B[B.hub_label != ""]
pd.DataFrame(dict(bead=hb.bead.values, label=hb.hub_label.values, cluster=Z["L"][hb.bead.values], px=px[hb.bead.values],
                  py=py[hb.bead.values], depth=dep[hb.bead.values])).to_csv(O / "representative_hub_projection.tsv", sep="\t", index=False)
pd.DataFrame(dict(bead=np.arange(N), px=px[:N], py=py[:N], depth=dep[:N])).to_csv(O / "representative_bead_projection.tsv", sep="\t", index=False)
json.dump(dict(view=list(view), width=a.width, height=a.height, fov=20.0, sigma_A=S), open(O / "representative_view.json", "w"))
# PDB of the representative (chain A beads, chain T clustered TFs; B-factor = cluster id + 1, 0 = none)
with open(O / "representative.pdb", "w") as f:
    for i, p in enumerate(Xo[:N]):
        f.write("ATOM  %5d  CA  BEA A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           C\n" % (i + 1, i % 10000, *p, Z["L"][i] + 1))
    for j, p in enumerate(Xo[N:]):
        f.write("HETATM%5d  O   TF  T%4d    %8.3f%8.3f%8.3f  1.00%6.2f           O\n" % (N + j + 1, j, *p, Z["LT"][j] + 1))
print("representative rendered", Z["conf"])
for g in range(len(Z["gallery"])):
    render(Z["XG"][g], Z["LG"][g], Z["LTG"][g], O / f"gallery_{g + 1}.png", a.gallery_width, a.gallery_width)
print("gallery", Z["gallery"].tolist())
