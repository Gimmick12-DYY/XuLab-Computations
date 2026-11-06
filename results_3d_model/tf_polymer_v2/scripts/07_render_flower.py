"""v2 step 7 (Mac, pymol-open-source): rosette render of the picked hub flower, Brackley 2016 style.

TF core = member TFs of the hub cluster (opaque #D55E00). Anchors = beads within anchor_r of them (medium grey spheres;
hub cores large, #222). Petals = chain segments between consecutive anchors pmin..pmax beads apart (leave and re-enter the
core), smooth tubes coloured by petal. Chain between anchors < pmin apart = thin grey "stem". No Cicero chords.
Two versions: flower only, and flower + the rest of the chain as a thin 88%-transparent light-grey ghost.
Core at the origin; camera looks down the smallest-variance axis of the flower (petals spread in the image plane);
orthoscopic; ray_trace_mode 1; ambient occlusion; transparent background.
Writes figure_flower/render_flower{,_ghost}.png, flower_projection.tsv, flower_petals.tsv.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--size", type=int, default=2400)
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_flower"
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
Z = np.load(O / "flower_pick.npz"); X, LT, hc = Z["X"].astype(float), Z["LT"], int(Z["hc"])
PET = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C6BB1", "#7F7F7F"]
S = 10.0

T = X[N:][LT == hc]; cen = T.mean(0)
anc = np.unique(np.concatenate(cKDTree(X[:N]).query_ball_point(T, float(Z["anchor_r"]))).astype(int))
g = np.diff(anc)
petals = [(int(p), int(q)) for p, q, d in zip(anc[:-1], anc[1:], g) if int(Z["pmin"]) <= d <= int(Z["pmax"])]
stems = [(int(p), int(q)) for p, q, d in zip(anc[:-1], anc[1:], g) if d < int(Z["pmin"])]
fl = np.zeros(N, bool); fl[anc] = True
for p, q in petals + stems: fl[p:q + 1] = True
# orient
Y = (X - cen) * S
_, _, Vt = np.linalg.svd(Y[:N][fl], full_matrices=False)
if np.linalg.det(Vt) < 0: Vt[2] *= -1
Y = Y @ Vt.T

def hexrgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
def smooth(idx, s=0.08, m=8):
    P = Y[idx]
    if len(idx) < 4: return P
    tck, _ = splprep(P.T, s=len(idx) * s * S * S / 100, k=3); return np.array(splev(np.linspace(0, 1, len(idx) * m), tck)).T
def tube(pts, r, col, alpha=None):
    c = hexrgb(col) if isinstance(col, str) else col; o = [] if alpha is None else [__import__("pymol").cgo.ALPHA, alpha]
    from pymol import cgo
    for p, q in zip(pts[:-1], pts[1:]): o += [cgo.CYLINDER, *p, *q, r, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *q, r]
    return o

import pymol
pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, cgo
hub = B.index[B.hub_label != ""].values; hub_in = [b for b in hub if b in set(anc)]

def build(ghost):
    cmd.reinitialize()
    if ghost:
        segs, cur = [], []
        for i in range(N):
            if not fl[i]: cur.append(i)
            elif cur: segs.append(cur); cur = []
        if cur: segs.append(cur)
        gh = []
        for sg in segs:
            sg = list(range(max(0, sg[0] - 1), min(N, sg[-1] + 2)))
            gh += tube(smooth(sg, 0.15, 4), 0.9, "#C8C8C8", alpha=0.12)
        cmd.load_cgo(gh, "ghost")
    st = []
    for p, q in stems: st += tube(smooth(list(range(p, q + 1)), 0.05, 8), 1.1, "#9A9A9A")
    if st: cmd.load_cgo(st, "stems")
    for k, (p, q) in enumerate(petals):
        cmd.load_cgo(tube(smooth(list(range(p, q + 1))), 2.2, PET[k % len(PET)]), f"petal{k + 1}")
    core = [cgo.COLOR, *hexrgb("#D55E00")]
    for p in Y[N:][LT == hc]: core += [cgo.SPHERE, *p, 4.8]
    cmd.load_cgo(core, "tf_core")
    an = [cgo.COLOR, *hexrgb("#A6A6A6")]
    for b in anc:
        if b not in hub_in: an += [cgo.SPHERE, *Y[b], 2.4]
    cmd.load_cgo(an, "anchors")
    hc_ = [cgo.COLOR, *hexrgb("#222222")]
    for b in hub_in: hc_ += [cgo.SPHERE, *Y[b], 7.5]
    if hub_in: cmd.load_cgo(hc_, "hub_cores")
    for k, v in dict(ray_trace_mode=1, ray_trace_gain=0.008, ray_trace_color="black", ambient_occlusion_mode=1,
                     ambient_occlusion_scale=15, ray_shadows=0, ambient=0.45, specular=0.15, direct=0.55, reflect=0.3,
                     orthoscopic=1, depth_cue=0, antialias=2, cgo_lighting=1, ray_opaque_background=0, transparency_mode=3).items():
        cmd.set(k, v)
    cmd.bg_color("white")

w = h = a.size
for ghost, nm in ((False, "render_flower"), (True, "render_flower_ghost")):
    build(ghost)
    cmd.viewport(w // 4, h // 4); cmd.reset()
    # frame on the flower (not the ghost), same camera for both versions
    for i, p in enumerate(np.r_[Y[:N][fl], Y[N:][LT == hc]]): cmd.pseudoatom("frame", pos=[float(x) for x in p], name=f"F{i}")
    cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=4, complete=1); cmd.delete("frame")
    view = cmd.get_view(); cmd.png(str(O / f"{nm}.png"), width=w, height=h, dpi=600, ray=1)
    cmd.save(str(O / f"{nm}.pse"))
# projection (orthoscopic): camera space -> pixels; the view's field scale from the clipping/zoom distance
R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15])
vc = (Y - org) @ R + cam
hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2); hw = hh * w / h
px = w / 2 * (1 + vc[:, 0] / hw); py = h / 2 * (1 - vc[:, 1] / hh)
rows = [dict(kind="hub", bead=b, label=B.hub_label[b], px=px[b], py=py[b]) for b in hub_in]
for k, (p, q) in enumerate(petals):
    seg = np.arange(p, q + 1); tip = seg[np.argmax(np.linalg.norm(Y[seg] - Y[N:][LT == hc].mean(0), axis=1))]
    rows.append(dict(kind="petal", bead=int(tip), label=f"P{k + 1}", px=px[tip], py=py[tip]))
pd.DataFrame(rows).to_csv(O / "flower_projection.tsv", sep="\t", index=False)
pd.DataFrame([dict(petal=f"P{k + 1}", bead1=p, bead2=q, start=int(B.start[p]), end=int(B.end[q]), span_kb=(q - p) * 2,
                   anchor1_gene=B.hub_label[p], anchor2_gene=B.hub_label[q], colour=PET[k % len(PET)]) for k, (p, q) in enumerate(petals)]
             ).to_csv(O / "flower_petals.tsv", sep="\t", index=False)
pd.DataFrame(dict(bead=anc, start=B.start.values[anc], n_tfs=B.n_tfs.values[anc], binding=B.bind.values[anc],
                  hub_label=B.hub_label.values[anc])).to_csv(O / "flower_anchors.tsv", sep="\t", index=False)
print(f"conformation {int(Z['conf'])}: TFs {len(T)}, anchors {len(anc)}, petals {[(q - p) * 2 for p, q in petals]} kb, "
      f"stems {len(stems)}, hub cores {[B.hub_label[b] for b in hub_in]}")
