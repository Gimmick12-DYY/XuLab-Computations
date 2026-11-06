# AF3 full-length MLX-SREBF2 on the E-box oligo, original annotated style: domain colours, pLDDT < 50 hidden unless interface or DNA contact, interface side chains as thin dark sticks, DNA-contacting residues as small spheres. Main view, two close-ups, flat-colour pass for label tips.
import json, numpy as np
from PIL import Image
from pymol import cmd
from scipy.ndimage import binary_erosion
O = "super/dnafl"; hexc = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
Ch = json.load(open("jie_domains.json"))["MLX_SREBF2"]["chains"]; DOM = {}
for ch, c in Ch.items():
    for nm, a, b, col, kind in c["dom"]: DOM[f"{c['name']} {nm}"] = (ch, a, b, col)
cmd.load(f"{O}/model.cif", "m"); cmd.remove("hydro or solvent"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1); cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_nucleic_acid_color", -1); cmd.set("dash_gap", .22); cmd.set("dash_radius", .1); cmd.set("dash_color", "black"); cmd.set("label_size", 0)
def colour():
    cmd.set_color("bA", hexc(Ch["A"]["base"])); cmd.set_color("bB", hexc(Ch["B"]["base"])); cmd.color("bA", "chain A"); cmd.color("bB", "chain B")
    for k, (n, (c, a, b, h)) in enumerate(DOM.items()): cmd.set_color(f"d{k}", hexc(h)); cmd.color(f"d{k}", f"chain {c} and resi {a}-{b}")
    cmd.set_color("gy", hexc("#BDBDBD")); cmd.set_color("go", hexc("#D98C1F")); cmd.color("gy", "chain C+D"); cmd.color("go", "chain C+D and resi 8-13")
def proj(sel, v, W, H):
    xyz = []; cmd.iterate_state(1, sel, "xyz.append((x, y, z))", space={"xyz": xyz}); M = [v[0:3], v[3:6], v[6:9]]; p = np.mean(xyz, axis=0) - np.array(v[12:15]); cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; sc = H / (2 * (-v[11]) * np.tan(np.radians(cmd.get_setting_float("field_of_view") / 2))); return (W / 2 + (cam[0] + v[9]) * sc, H / 2 - (cam[1] + v[10]) * sc)
cmd.select("ifc", "byres ((chain A within 4 of chain B) or (chain B within 4 of chain A))"); cmd.select("dcon", "byres ((chain A+B) within 4 of chain C+D)"); seg = []
for ch in "AB":
    pl = []; cmd.iterate(f"chain {ch} and name CA", "pl.append((int(resi), b))", space={"pl": pl}); run = []
    for r_, b_ in pl + [(10 ** 6, 0)]:
        if b_ > 50: run.append(r_)
        else:
            if len(run) >= 8: seg.append(f"(chain {ch} and resi {run[0]}-{run[-1]})")
            run = []
cmd.select("keep", "chain C+D or " + " or ".join(seg)); cmd.select("ifc", "ifc and keep"); cmd.select("dcon", "dcon and keep"); print("kept CA", cmd.count_atoms("keep and name CA"), "of", cmd.count_atoms("chain A+B and name CA"), "interface", cmd.count_atoms("ifc and name CA"), "DNA-contact", cmd.count_atoms("dcon and name CA"))
r = {"A": [], "B": []}; cmd.iterate("ifc and name CA", "r[chain].append((int(resi), round(b)))", space={"r": r}); print("ifc", r); r2 = {"A": [], "B": []}; cmd.iterate("dcon and name CA", "r2[chain].append(int(resi))", space={"r2": r2}); print("dcon", r2)
colour(); cmd.hide("everything"); cmd.show("cartoon", "keep"); cmd.orient("keep"); cmd.zoom("keep", 0); cmd.viewport(1890, 1620); W, H = 3780, 3240; v = cmd.get_view()
cmd.set("stick_radius", .14); cmd.set("cartoon_side_chain_helper", 0); cmd.show("sticks", "ifc and (sidechain or name CA)"); cmd.set_color("ink", hexc("#1A1A1A")); cmd.color("ink", "ifc and sidechain and elem C+N+O+S"); cmd.set("sphere_scale", .62, "dcon and name CA"); cmd.show("spheres", "dcon and name CA"); cmd.png(f"{O}/over.png", width=W, height=H, dpi=300, ray=1); cmd.hide("sticks"); cmd.hide("spheres"); colour(); out = {"over": {}}
cmd.set("ambient", 1); cmd.set("direct", 0); cmd.set("reflect", 0); cmd.set("specular", 0); cmd.set("antialias", 0); ids = {}; cmd.set_color("idbase", [.5, .5, .5]); cmd.color("idbase", "chain A+B")
for i, (n, (c, a, b, h)) in enumerate(DOM.items(), 1): col = (i * 20, 255 - i * 20, 40 + i * 12); ids[n] = col; cmd.set_color(f"id{i}", [x / 255 for x in col]); cmd.color(f"id{i}", f"chain {c} and resi {a}-{b}")
cmd.set_color("idE", [250 / 255, 10 / 255, 200 / 255]); cmd.set_color("idD", [10 / 255, 250 / 255, 200 / 255]); cmd.color("idD", "chain C+D"); cmd.color("idE", "chain C+D and resi 8-13"); ids["E-box CACGTG"] = (250, 10, 200); cmd.set_view(v); cmd.png(f"{O}/flat.png", width=W, height=H, dpi=300, ray=1); fl = np.asarray(Image.open(f"{O}/flat.png").convert("RGB")).astype(int); sels = {**{n: f"chain {c} and resi {a}-{b}" for n, (c, a, b, h) in DOM.items()}, "E-box CACGTG": "chain C+D and resi 8-13"}
for n, sel in sels.items():
    m = np.abs(fl - np.array(ids[n])).sum(axis=2) <= 8
    if m.sum() < 5: print(n, "not visible", int(m.sum())); continue
    cen = proj(sel, v, W, H)
    for e in (8, 5, 3, 1, 0):
        k = binary_erosion(m, iterations=e) if e else m
        if k.sum() >= 5: break
    ys, xs = np.nonzero(k); j = np.argmin((xs - cen[0]) ** 2 + (ys - cen[1]) ** 2); out["over"][n] = (float(xs[j]), float(ys[j]), int(m.sum())); print(n, out["over"][n], "inset", e)
cmd.set("ambient", .14); cmd.set("direct", .45); cmd.set("reflect", .45); cmd.set("specular", 1); cmd.set("antialias", 2); colour(); cmd.set("stick_radius", .2)
# Close-up a: Glu reading the first C of CACGTG on each strand.
W, H = 2880, 2160; cmd.hide("everything"); cmd.show("cartoon", "(chain A and resi 129-148) or (chain B and resi 330-350)"); cmd.show("cartoon", "chain C+D and resi 5-16"); cmd.set("cartoon_transparency", .55, "m"); cmd.show("sticks", "(chain A and resi 138 or chain B and resi 339) and sidechain"); cmd.set("stick_radius", .34); cmd.show("sticks", "chain C+D and resi 8 and not name P+OP1+OP2"); cmd.set_color("gl", hexc("#1A1A1A")); cmd.color("gl", "(chain A and resi 138 or chain B and resi 339) and elem C"); cmd.color("red", "(chain A and resi 138 or chain B and resi 339) and elem O"); cmd.set("dash_radius", .16); E1 = "chain A and resi 138 and name OE2"; C1 = "chain C and resi 8 and name N4"; E2 = "chain B and resi 339 and name OE2"; C2 = "chain D and resi 8 and name N4"; print("E138-C", round(cmd.get_distance(E1, C1), 2), "E339-C", round(cmd.get_distance(E2, C2), 2)); cmd.distance("d1", E1, C1); cmd.distance("d2", E2, C2); cmd.hide("labels"); cmd.orient("(chain A and resi 138) or (chain B and resi 339) or (chain C+D and resi 8-13)"); cmd.turn("x", 25); cmd.zoom("(chain A and resi 134-142) or (chain B and resi 335-343) or (chain C+D and resi 7-14)", 1); v = cmd.get_view(); cmd.png(f"{O}/c1.png", width=W, height=H, dpi=300, ray=1); out["c1"] = {k: proj(s_, v, W, H) for k, s_ in dict(e1=E1, c1=C1, e2=E2, c2=C2).items()}; cmd.delete("d1"); cmd.delete("d2")
# Close-up b: MLX Y183, SREBF2 Q381.
cmd.set("cartoon_transparency", 0, "m"); cmd.set("dash_radius", .1); cmd.set("stick_radius", .2); colour(); cmd.hide("everything"); cmd.show("cartoon", "(chain A and resi 129-220) or (chain B and resi 330-401)"); S_ = "byres (chain A and resi 183 or chain B and resi 381)"; cmd.show("sticks", S_); cmd.util.cnc(S_); cmd.distance("d3", "chain A and resi 183 and name OH", "chain B and resi 381 and name OE1+NE2", cutoff=3.5, mode=0); cmd.hide("labels"); print("Y183-Q381", [round(cmd.get_distance("chain A and resi 183 and name OH", f"chain B and resi 381 and name {x}"), 2) for x in ("OE1", "NE2")]); cmd.orient("(chain A and resi 175-191) or (chain B and resi 373-389)"); cmd.zoom("(chain A and resi 183) or (chain B and resi 381)", 7); v = cmd.get_view(); cmd.png(f"{O}/c2.png", width=W, height=H, dpi=300, ray=1); out["c2"] = dict(y=proj("chain A and resi 183 and name OH", v, W, H), q=proj("chain B and resi 381 and name CD", v, W, H))
json.dump(out, open(f"{O}/pos.json", "w"))
