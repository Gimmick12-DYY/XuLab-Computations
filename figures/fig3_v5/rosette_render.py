"""Fig 3 v5 d1 render: SEdb chr19 SE hub, Hi-C-constrained structure #9102, in the camera and crop of
results_3d_model/se_sedb/scripts/47_rosette_own_beads.py, simplified: grey chromatin tube, gold SE sphere, black promoter
and light-grey other spheres, orange enhancer spheres; no TF satellite spheres, no chords.
Run with the pymol3d env:  /work/users/d/y/dyy12/envs/pymol3d/bin/python rosette_render.py
Writes inputs/rosette9102_v5_render.png and inputs/rosette9102_v5_nodes.tsv (projected pixel position of every element)."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent; INP = HERE / "inputs"
R0 = HERE.parents[1] / "results_3d_model/se_sedb"
CONF, N, L0, S, SIZE = 9102, 910, 430000, 10.0, 3000
files = sorted(glob.glob(str(R0 / "model/sim/eps0/traj*.npz")))
mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2
tb = lambda p: (p - L0) // 2000
H = pd.read_csv(INP / "hub_edges_all.tsv", sep="\t")
H["b1"] = [tb(mid(x)) for x in H.reg_a]; H["b2"] = [tb(mid(x)) for x in H.reg_b]
se_b = tb((1248039 + 1254347) // 2)
Kb = {"SE": se_b}; El = {se_b: ("SE", "se")}
for r in H.itertuples():
    for k_, b_, l_, c_ in ((str(r.a), r.b1, r.label_a, r.cls_a), (str(r.b), r.b2, r.label_b, r.cls_b)):
        if k_ not in Kb: Kb[k_] = int(b_); El[int(b_)] = (l_.replace("RNU6-2/TMEM259", "TMEM259"), c_)
Tx = pd.read_csv(R0 / "edges_exact_se.tsv", sep="\t"); OWN = {str(mid(x) // 10000): tb(mid(x)) for x in Tx.partner}
OLDK = dict(Kb); El0 = dict(El)
Kb = {k_: (b_ if k_ == "SE" else OWN[k_]) for k_, b_ in OLDK.items()}
El = {Kb[k_]: El0[b_] for k_, b_ in OLDK.items()}; assert len(El) == 32
EM = pd.read_csv(INP / "enhancer_map.tsv", sep="\t").set_index("old_label").number.to_dict()
El = {b: ((f"E{EM[l]}" if l in EM else l), c) for b, (l, c) in El.items()}


def load(c):
    t, k = divmod(int(c) - 1, 50); return np.load(files[t])["X"][k, :N].astype(float)


def comp(P, hb):
    pr = cKDTree(P[hb]).query_pairs(2.0, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(hb), len(hb))), directed=False)
    return np.sort(hb[lab == lab[np.where(hb == se_b)[0][0]]])


P = load(CONF)
mem_old = comp(P, np.array(sorted(OLDK.values()))); show_old = np.zeros(N, bool)
for p, q in zip(mem_old[:-1], mem_old[1:]):
    if q - p <= 100: show_old[int(p):int(q) + 1] = True
show_old[mem_old] = True
mem = comp(P, np.array(sorted(El))); loops = [(int(p), int(q)) for p, q in zip(mem[:-1], mem[1:]) if q - p <= 100]
cen = P[mem_old].mean(0); Y = (P - cen) * S; _, _, Vt = np.linalg.svd(Y[show_old] - Y[show_old].mean(0), full_matrices=False)
if np.linalg.det(Vt) < 0: Vt[2] *= -1
Y = Y @ Vt.T
import pymol
pymol.finish_launching(["pymol", "-qc"])
from pymol import cmd, cgo
rgb = lambda h: [int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
cmd.reinitialize()
for k, (p, q) in enumerate(loops):
    seg = np.arange(p, q + 1); pts = Y[seg]; c = rgb("#B4B4B4")
    if len(seg) >= 4: tck, _ = splprep(pts.T, s=len(seg) * 0.08, k=3); pts = np.array(splev(np.linspace(0, 1, len(seg) * 8), tck)).T
    o = []
    for u, v in zip(pts[:-1], pts[1:]): o += [cgo.CYLINDER, *u, *v, 1.4, *c, *c, cgo.COLOR, *c, cgo.SPHERE, *v, 1.4]
    cmd.load_cgo(o, f"loop{k}")
TYPE = dict(promoter="#1A1A1A", enhancer="#D55E00", other="#BDBDBD")
sp = []
for b in mem:
    if b != se_b: sp += [cgo.COLOR, *rgb(TYPE[El[b][1]]), cgo.SPHERE, *Y[b], 4.6]
cmd.load_cgo(sp, "nodes")
for k_, v in dict(ray_trace_mode=1, ray_trace_gain=0.006, ray_trace_color="black", ambient_occlusion_mode=1, ambient_occlusion_scale=15, ambient=0.45,
                  specular=0.1, direct=0.55, reflect=0.3, orthoscopic=1, antialias=2, cgo_lighting=1, ray_opaque_background=0, ray_shadows=0, depth_cue=0).items():
    cmd.set(k_, v)
cmd.viewport(SIZE // 4, SIZE // 4); cmd.reset()
for i, q in enumerate(Y[show_old]): cmd.pseudoatom("frame", pos=[float(x) for x in q], name=f"F{i}")
cmd.hide("everything", "frame"); cmd.zoom("frame", buffer=3, complete=1); cmd.delete("frame"); view = cmd.get_view()
cmd.set_color("segold", rgb("#FFC20A"))
cmd.pseudoatom("se_gold", pos=[float(x) for x in Y[se_b]], vdw=7.0, color="segold"); cmd.show_as("spheres", "se_gold"); cmd.set("sphere_scale", 1.0, "se_gold")
for k_, v in dict(specular=1.0, spec_power=200, spec_reflect=1.5, spec_direct=1.0).items(): cmd.set(k_, v, "se_gold")
cmd.set_view(view); cmd.png(str(INP / "rosette9102_v5_render.png"), width=SIZE, height=SIZE, dpi=600, ray=1)
R_ = np.array(view[:9]).reshape(3, 3); cam = np.array(view[9:12]); org = np.array(view[12:15]); vc = (Y - org) @ R_ + cam
hh = -cam[2] * np.tan(np.radians(cmd.get_setting_float("field_of_view")) / 2)
pd.DataFrame(dict(bead=mem, label=[El[b][0] for b in mem], cls=[El[b][1] for b in mem], px=(SIZE / 2 * (1 + vc[:, 0] / hh))[mem],
                  py=(SIZE / 2 * (1 - vc[:, 1] / hh))[mem])).to_csv(INP / "rosette9102_v5_nodes.tsv", sep="\t", index=False)
print("linked", len(mem), "segments", len(loops), "conf", CONF)
