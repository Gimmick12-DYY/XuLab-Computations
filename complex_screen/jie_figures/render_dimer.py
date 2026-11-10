# Simplified slide render: only the contacting domains, three colours, interface as one translucent surface, no side chains. Writes the image and label anchors in pixels. usage: pymol -cq render_dimer.py -- <config key>
import json, math, sys
from pymol import cmd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/jie_structures"
CFG = {"MLX_SREBF2": dict(file=f"{C}/af3_local/out/mlx_srebf2/mlx_srebf2_model.cif", keep=[("A", 129, 220, "#2166AC"), ("B", 330, 401, "#B2182B")], dna=[("A", 129, 141), ("B", 330, 342)],
    labels={"MLX": ("A", 195), "SREBF2": ("B", 365), "Basic region: DNA contact": ("A", 135), "Basic region ": ("B", 336), "HLH": ("A", 160), "Zipper": ("B", 392), "Interface": ("A", 205)})}
for job in sys.argv[1:]:
    G = CFG[job]; cmd.reinitialize(); cmd.load(G["file"], "m"); cmd.remove("hydro or solvent"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1); cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_tube_radius", 0.35)
    keep = " or ".join(f"(chain {c} and resi {a}-{b})" for c, a, b, _ in G["keep"]); cmd.select("k", keep); cmd.remove("not k"); cmd.hide("everything"); cmd.show("cartoon", "all")
    for i, (c, a, b, col) in enumerate(G["keep"]): cmd.set_color(f"c{i}", [int(col[k:k + 2], 16) / 255 for k in (1, 3, 5)]); cmd.color(f"c{i}", f"chain {c}")
    cmd.set_color("dna", [0x6A / 255, 0x51 / 255, 0xA3 / 255])
    for c, a, b in G["dna"]: cmd.color("dna", f"chain {c} and resi {a}-{b}")
    chs = [c for c, *_ in G["keep"]]; ifc = " or ".join(f"(byres (chain {c} within 5 of ({' or '.join(f'chain {x}' for x in chs if x != c)})))" for c in chs); cmd.select("ifc", ifc); cmd.set_color("ifcgrey", [0.55, 0.55, 0.55]); cmd.show("surface", "ifc"); cmd.set("surface_color", "ifcgrey", "ifc"); cmd.set("transparency", 0.62)
    cmd.orient("all"); cmd.zoom("all", 1); W_, H_ = 3960, 2910; cmd.viewport(1700, 1250); cmd.png(f"{O}/{job}_dimer.png", width=W_, height=H_, dpi=300, ray=1); v = cmd.get_view(); M = [v[0:3], v[3:6], v[6:9]]; pos = {}
    for name, (c, r) in G["labels"].items():
        xy = []; cmd.iterate_state(1, f"chain {c} and resi {r} and name CA", "xy.append((x, y, z))", space={"xy": xy}); p = [xy[0][i] - v[12 + i] for i in range(3)]; cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; h = 2 * (-v[11]) * math.tan(math.radians(cmd.get_setting_float("field_of_view") / 2)); sc = H_ / h; pos[name] = (W_ / 2 + (cam[0] + v[9]) * sc, H_ / 2 - (cam[1] + v[10]) * sc)
    json.dump(pos, open(f"{O}/{job}.dimer_labels.json", "w")); print(job, pos)
