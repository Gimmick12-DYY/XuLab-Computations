"""Hybrid step 10 (Mac): as v2 10_render_hub_flow.py, for --tag selected|typical, plus the restrained Hi-C loops that are
ON in this trajectory and have both anchors among the flower anchors (+-1 bead) drawn as thin dark-grey dashed chords.

v2 notes: (Mac, pymol-open-source): the chr16p11.2 browser hub component rendered as a 3D flower (Brackley 2016 style).

As 07_render_flower.py, except: petals coloured by genomic position (viridis, 29.73-31.55 Mb, same as the mini-track);
hub anchors = the browser-figure elements in the flower (dark, labelled with the browser gene names); the browser
shared edges with both ends in the flower drawn as thin chords in the browser arc colours (#C0392B on a 0.7 loop,
#4A6FA5 not). Writes figure_hub_flow/render_hub_flow{,_ghost}.png and projections.

Original 07 notes:

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
ap.add_argument("--tag", default="selected")
a = ap.parse_args()
Bd = Path(a.base); O = Bd / "figure_hub_flow"
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
Z = np.load(O / f"hub_flow_pick_{a.tag}.npz")
LR = pd.read_csv(Bd / "inputs/loop_restraints.tsv", sep="\t"); LON = Z["loop_on"]
EL = pd.read_csv(O / "hub_elements.tsv", sep="\t"); HE = pd.read_csv(O / "hub_edges_beads.tsv", sep="\t")
ANC_R, PMIN, PMAX = 2.0, 10, 100; X, LT, hc = Z["X"].astype(float), Z["LT"], int(Z["hc"])
PET = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C6BB1", "#7F7F7F"]
S = 10.0

T = X[N:][LT == hc]; cen = T.mean(0)
anc = np.unique(np.concatenate(cKDTree(X[:N]).query_ball_point(T, ANC_R)).astype(int))
g = np.diff(anc)
petals = [(int(p), int(q)) for p, q, d in zip(anc[:-1], anc[1:], g) if PMIN <= d <= PMAX]
stems = [(int(p), int(q)) for p, q, d in zip(anc[:-1], anc[1:], g) if d < PMIN]
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
aset = set(anc.tolist()); hub_in = sorted(set(EL.bead) & aset)
VIR = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]) / 255
def vir(t): return np.array([np.interp(t, np.linspace(0, 1, 5), VIR[:, c]) for c in range(3)])

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
    for p, q in stems: st += tube(smooth(list(range(p, q + 1)), 0.05, 8), 0.7, "#B5B5B5")
    if st: cmd.load_cgo(st, "stems")
    for k, (p, q) in enumerate(petals):
        seg = list(range(p, q + 1)); pts = smooth(seg); bo = np.array(seg)[np.minimum((np.linspace(0, 1, len(pts)) * (len(seg) - 1)).round().astype(int), len(seg) - 1)]
        o = []
        for i in range(len(pts) - 1):
            c1, c2 = vir(bo[i] / (N - 1)), vir(bo[i + 1] / (N - 1))
            o += [cgo.CYLINDER, *pts[i], *pts[i + 1], 2.2, *c1, *c2, cgo.COLOR, *c2, cgo.SPHERE, *pts[i + 1], 2.2]
        cmd.load_cgo(o, f"petal{k + 1}")
    ch = []
    for r in HE.itertuples():
        if r.b1 in aset and r.b2 in aset:
            c = hexrgb("#C0392B" if r._5 == 1 else "#4A6FA5"); ch += [cgo.CYLINDER, *Y[r.b1], *Y[r.b2], 1.3, *c, *c]
    if ch: cmd.load_cgo(ch, "browser_edges")
    near = lambda b: any(abs(b - x) <= 1 for x in aset)
    dl = []
    for b1, b2, on in zip(LR.b1, LR.b2, LON):
        if on and near(b1) and near(b2):
            p0, p1 = Y[b1], Y[b2]; L_ = np.linalg.norm(p1 - p0); n_ = max(2, int(L_ / 5))
            for k in range(0, n_, 2):
                q0 = p0 + (p1 - p0) * k / n_; q1 = p0 + (p1 - p0) * min(k + 1, n_) / n_
                dl += [cgo.CYLINDER, *q0, *q1, 0.8, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33]
    if dl: cmd.load_cgo(dl, "hic_loops")
    core = [cgo.COLOR, *hexrgb("#D55E00")]
    for p in Y[N:][LT == hc]: core += [cgo.SPHERE, *p, 4.8]
    cmd.load_cgo(core, "tf_core")
    an = [cgo.COLOR, *hexrgb("#A6A6A6")]
    for b in anc:
        if b not in hub_in: an += [cgo.SPHERE, *Y[b], 1.5]
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
for ghost, nm in ((False, f"render_hub_flow_{a.tag}"), (True, f"render_hub_flow_{a.tag}_ghost")):
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
lab_of = dict(zip(EL.bead, EL.label))
rows = [dict(kind="hub", bead=b, label=lab_of[b], px=px[b], py=py[b]) for b in hub_in]
for k, (p, q) in enumerate(petals):
    seg = np.arange(p, q + 1); tip = seg[np.argmax(np.linalg.norm(Y[seg] - Y[N:][LT == hc].mean(0), axis=1))]
    rows.append(dict(kind="petal", bead=int(tip), label=f"P{k + 1}", px=px[tip], py=py[tip]))
pd.DataFrame(rows).to_csv(O / f"hub_flow_{a.tag}_projection.tsv", sep="\t", index=False)
lb1, lb2 = LR.b1.values[LON.astype(bool)], LR.b2.values[LON.astype(bool)]
lanch = [bool(np.any((np.abs(lb1 - p) <= 3) & (np.abs(lb2 - q) <= 3))) for p, q in petals]
pd.DataFrame([dict(petal=f"P{k + 1}", bead1=p, bead2=q, start=int(B.start[p]), end=int(B.end[q]), span_kb=(q - p) * 2,
                   anchor1_gene=lab_of.get(p, ""), anchor2_gene=lab_of.get(q, ""), loop_anchored=lanch[k]) for k, (p, q) in enumerate(petals)]
             ).to_csv(O / f"hub_flow_{a.tag}_petals.tsv", sep="\t", index=False)
pd.DataFrame(dict(bead=anc, start=B.start.values[anc], n_tfs=B.n_tfs.values[anc], binding=B.bind.values[anc],
                  hub_label=[lab_of.get(b, "") for b in anc])).to_csv(O / f"hub_flow_{a.tag}_anchors.tsv", sep="\t", index=False)
print(f"conformation {int(Z['conf'])}: TFs {len(T)}, anchors {len(anc)}, petals {[(q - p) * 2 for p, q in petals]} kb, "
      f"loop-anchored {sum(lanch)}, stems {len(stems)}, hub elements {[lab_of[b] for b in hub_in]}")
