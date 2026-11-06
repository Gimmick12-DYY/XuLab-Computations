# PyMOL render of each pilot pair's top-ranked Boltz-2 model: chain A and B cartoons, interface residues (any atom within
# 5 A of the partner) as sticks; 600 dpi PNG. Run: pymol -cq pilot_render.py -- <pilot_dir>
import json, os, sys
from pymol import cmd
P = sys.argv[1]
SRC = json.load(open(f"{P}/layer2/models.json"))  # pair -> {"cif": path, "a": name, "b": name}
COL = {"A": "0x1F3F8F", "B": "0xD98C1F"}
for pair, m in SRC.items():
    if not m.get("cif") or not os.path.exists(m["cif"]): continue
    cmd.reinitialize(); cmd.load(m["cif"], "mdl")
    cmd.hide("everything"); cmd.show("cartoon"); cmd.set("cartoon_transparency", 0.1)
    cmd.color(COL["A"], "mdl and chain A"); cmd.color(COL["B"], "mdl and chain B")
    cmd.select("ifA", "byres (mdl and chain A) within 5 of (mdl and chain B)")
    cmd.select("ifB", "byres (mdl and chain B) within 5 of (mdl and chain A)")
    cmd.show("sticks", "(ifA or ifB) and not name N+C+O"); cmd.set("stick_radius", 0.18)
    cmd.color("0x6A8FD8", "ifA and elem C"); cmd.color("0xF2B66B", "ifB and elem C")
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("ray_shadows", 0)
    cmd.orient("ifA or ifB"); cmd.zoom("ifA or ifB", 6)
    cmd.png(f"{P}/panels/{pair}_structure.png", width=2400, height=2000, dpi=600, ray=1)
    n_a, n_b = cmd.count_atoms("ifA and name CA"), cmd.count_atoms("ifB and name CA")
    json.dump({"interface_res_A": n_a, "interface_res_B": n_b}, open(f"{P}/panels/{pair}_structure_iface.json", "w"))
