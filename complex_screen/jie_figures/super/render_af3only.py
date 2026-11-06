# MLX-SREBF2 AF3 model only: bHLH-ZIP dimer overview, close-ups of the Y183-Q381 contact and of the free basic regions. Headless on the Mac.
import json, math, pymol
from pymol import cmd
pymol.finish_launching(["pymol", "-cq"]); hx = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
cmd.load("mlx_srebf2_model.cif", "mod"); cmd.remove("hydro or solvent"); cmd.remove("not ((chain A and resi 129-220) or (chain B and resi 330-401))"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1); cmd.set("cartoon_fancy_helices", 1); cmd.set("stick_radius", 0.2); cmd.set("dash_gap", 0.25); cmd.set("dash_radius", 0.07); cmd.set("label_size", 0)
for n, h in (("mlx", "#2166AC"), ("srb", "#B2182B"), ("bas", "#6A51A3"), ("ink", "#1A1A1A")): cmd.set_color(n, hx(h))
cmd.color("mlx", "chain A"); cmd.color("srb", "chain B"); cmd.color("bas", "(chain A and resi 129-141) or (chain B and resi 330-342)"); cmd.hide("everything"); cmd.show("cartoon", "all")
def proj(pt, W, H):
    v = cmd.get_view(); M = [v[0:3], v[3:6], v[6:9]]; p = [pt[i] - v[12 + i] for i in range(3)]; cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; h = 2 * (-v[11]) * math.tan(math.radians(cmd.get_setting_float("field_of_view") / 2)); sc = H / h; return (W / 2 + (cam[0] + v[9]) * sc, H / 2 - (cam[1] + v[10]) * sc)
def xyz(sel):
    o_ = []; cmd.iterate_state(1, sel, "o_.append((x, y, z))", space={"o_": o_}); assert o_, sel; return [sum(c[i] for c in o_) / len(o_) for i in range(3)]
J = {}; Y = "chain A and resi 183 and name OH"; Q = "chain B and resi 381 and name NE2"; J["d"] = round(cmd.get_distance(Y, Q), 1); J["names"] = [cmd.get_model("chain A and resi 183 and name CA").atom[0].resn, cmd.get_model("chain B and resi 381 and name CA").atom[0].resn]
W, H = 3120, 3120; cmd.orient("all"); cmd.zoom("all", 2); cmd.viewport(1500, 1500); cmd.png("AF3_over.png", width=W, height=H, dpi=300, ray=1)
J["over"] = {k: proj(xyz(s), W, H) for k, s in dict(MLX="chain A and resi 205 and name CA", SREBF2="chain B and resi 395 and name CA", c1=Y, bas="(chain A and resi 135 and name CA) or (chain B and resi 336 and name CA)").items()}
W2, H2 = 2880, 2160; cmd.show("sticks", "byres (chain A and resi 183 or chain B and resi 381)"); cmd.distance("dd", Y, Q); cmd.set_color("dc", hx("#1A1A1A")); cmd.color("dc", "dd"); cmd.hide("labels", "dd"); cmd.orient("(chain A and resi 175-191) or (chain B and resi 373-389)"); cmd.zoom("(chain A and resi 183) or (chain B and resi 381)", 7); cmd.viewport(1700, 1275); cmd.png("AF3_c1.png", width=W2, height=H2, dpi=300, ray=1); J["p1"] = dict(y=proj(xyz(Y), W2, H2), q=proj(xyz(Q), W2, H2))
cmd.delete("dd"); cmd.hide("sticks"); cmd.orient("(chain A and resi 129-150) or (chain B and resi 330-351)"); cmd.zoom("(chain A and resi 129-160) or (chain B and resi 330-361)", 4); cmd.png("AF3_c2.png", width=W2, height=H2, dpi=300, ray=1); J["p2"] = dict(a=proj(xyz("chain A and resi 135 and name CA"), W2, H2), b=proj(xyz("chain B and resi 336 and name CA"), W2, H2)); json.dump(J, open("af3only.json", "w"), indent=1); print(J["d"], J["names"])
