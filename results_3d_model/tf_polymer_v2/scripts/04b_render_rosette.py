"""v2 step 4b: rosette view of the hub flower (replaces the 3.5-sigma zoom crop).

Shows only the TF cluster that holds the hub cores (opaque-ish core) and the chromatin that touches it: every bead
assigned to that cluster, plus the whole chain segment between consecutive such beads when they are <= --max-gap beads
apart (a loop that leaves and returns to the core = a petal), plus --flank beads on each side. The rest of the chain is
hidden. Chain coloured by genomic position (viridis); hub cores dark; Cicero hub edges between shown cores in clique colours.
Run with any Python that has PyMOL (pymol-open-source); writes render/zoom_hub_flower.png, zoom_hub_projection.tsv, zoom_beads.tsv.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--max-gap", type=int, default=40)
ap.add_argument("--flank", type=int, default=4)
ap.add_argument("--size", type=int, default=2800)
a = ap.parse_args()
Bd = Path(a.base); A = Bd / "analysis"; O = Bd / "render"
B = pd.read_csv(Bd / "inputs/beads.tsv", sep="\t").fillna({"hub_label": "", "tfs": ""}); N = len(B)
CP = pd.read_csv(Bd / "inputs/pairs_cicero.tsv", sep="\t"); HE = CP[CP.set == "hub_edges"]
Z = np.load(A / "representative.npz"); X, L, LT, hc = Z["X"], Z["L"], Z["LT"], int(Z["hub_cluster"])
S = 10.0
VIR = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]) / 255
def vir(t): return np.array([np.interp(t, np.linspace(0, 1, 5), VIR[:, c]) for c in range(3)])
def hexrgb(h): h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

fb = np.where(L == hc)[0]
show = np.zeros(N, bool); show[fb] = True
for p, q in zip(fb[:-1], fb[1:]):
    if q - p <= a.max_gap: show[p:q + 1] = True
for b in fb: show[max(0, b - a.flank):min(N, b + a.flank + 1)] = True
T = X[N:][LT == hc]; cen = T.mean(0)
Xo = (X - cen) * S
# orient: PCA of the shown chain, largest spread in the image plane
_, _, Vt = np.linalg.svd(Xo[:N][show] - Xo[:N][show].mean(0), full_matrices=False)
if np.linalg.det(Vt) < 0: Vt[2] *= -1
Xo = Xo @ Vt.T

import pymol
pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, cgo
cmd.reinitialize()
segs, cur = [], []
for i in range(N):
    if show[i]: cur.append(i)
    elif cur: segs.append(cur); cur = []
if cur: segs.append(cur)
tube = []
for sg in segs:
    P = Xo[sg]
    if len(sg) < 4:
        pts, bo = P, np.array(sg)
    else:
        tck, _ = splprep(P.T, s=len(sg) * 0.1, k=3); tt = np.linspace(0, 1, len(sg) * 8)
        pts = np.array(splev(tt, tck)).T; bo = np.array(sg)[np.minimum((tt * (len(sg) - 1)).round().astype(int), len(sg) - 1)]
    for k in range(len(pts) - 1):
        c1, c2 = vir(bo[k] / (N - 1)), vir(bo[k + 1] / (N - 1))
        tube += [cgo.CYLINDER, *pts[k], *pts[k + 1], 2.2, *c1, *c2, cgo.COLOR, *c2, cgo.SPHERE, *pts[k + 1], 2.2]
cmd.load_cgo(tube, "petals")
core = [cgo.ALPHA, 0.75, cgo.COLOR, *hexrgb("#4F7FC0")]
for p in Xo[N:][LT == hc]: core += [cgo.SPHERE, *p, 6.5]
cmd.load_cgo(core, "tf_core")
HB = B.index[(B.hub_label != "") & show].values
hcore = [cgo.COLOR, *hexrgb("#262626")]
for b in HB: hcore += [cgo.SPHERE, *Xo[b], 8.5]
if len(HB): cmd.load_cgo(hcore, "hub_cores")
ed = []
for r in HE.itertuples():
    if show[r.b1] and show[r.b2]:
        c = hexrgb(r.color); ed += [cgo.CYLINDER, *Xo[r.b1], *Xo[r.b2], 1.4, *c, *c]
if ed: cmd.load_cgo(ed, "hub_edges")
for k, v in dict(ray_trace_mode=0, ambient_occlusion_mode=1, ambient_occlusion_scale=15, ray_shadows=0, ambient=0.4,
                 specular=0.1, direct=0.6, reflect=0.35, orthoscopic=0, depth_cue=0, antialias=2, field_of_view=20,
                 cgo_lighting=1, ray_opaque_background=0, transparency_mode=3).items():
    cmd.set(k, v)
cmd.bg_color("white")
w = h = a.size
cmd.viewport(w // 4, h // 4); cmd.reset(); cmd.zoom("all", buffer=2, complete=1)
view = cmd.get_view(); cmd.png(str(O / "zoom_hub_flower.png"), width=w, height=h, dpi=600, ray=1)
cmd.save(str(O / "zoom_hub_flower.pse"))
R = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15])
vc = (Xo - org) @ R + cam; hh = -vc[:, 2] * np.tan(np.radians(20.0) / 2); hw = hh * w / h
px = w / 2 * (1 + vc[:, 0] / hw); py = h / 2 * (1 - vc[:, 1] / hh)
hb = B[B.hub_label != ""]
pd.DataFrame(dict(bead=hb.bead.values, label=hb.hub_label.values, in_zoom=show[hb.bead.values], px=px[hb.bead.values],
                  py=py[hb.bead.values])).to_csv(O / "zoom_hub_projection.tsv", sep="\t", index=False)
sb = np.where(show)[0]
pd.DataFrame(dict(bead=sb, mb=B.start.values[sb] / 1e6)).to_csv(O / "zoom_beads.tsv", sep="\t", index=False)
print(f"flower beads {len(fb)}, shown beads {show.sum()} in {len(segs)} segments, petals (gaps<= {a.max_gap}) "
      f"{sum(1 for p, q in zip(fb[:-1], fb[1:]) if 1 < q - p <= a.max_gap)}, TFs {len(T)}, hub cores shown {len(HB)}")
