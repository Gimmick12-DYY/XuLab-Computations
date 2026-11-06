# Trimer render plus a flat-colour pass per domain: which pixels of the figure really show each domain, and where a label may land. Same view and same display rules as render_sites.py.
import json, numpy as np
from PIL import Image
from pymol import cmd
Image.MAX_IMAGE_PIXELS = None
J = json.load(open("jie_domains.json"))["MLX_SREBF2"]; O = "super/mlxfix"; hexc = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]; W_, H_ = 3960, 2910; FW, FH = 1980, 1455
cmd.load("super/mlx_srebf2_model.cif", "m"); cmd.remove("hydro or solvent"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1)
chains = list(J["chains"]); sel = []
for ch in chains:
    oth = " or ".join(f"chain {x}" for x in chains if x != ch); cmd.select(f"i{ch}", f"byres (chain {ch} within 5 of ({oth}))"); sel.append(f"i{ch}")
cmd.select("ifc", " or ".join(sel)); dna = " or ".join(f"(chain {ch} and resi {a}-{b})" for ch, D_ in J["chains"].items() for n, a, b, c, kind in D_["dom"] if kind == "dna"); cmd.select("dna", dna); cmd.select("keep", "(b > 50) or ifc or dna"); cmd.hide("everything"); cmd.show("cartoon", "keep"); k = 0
for ch, D_ in J["chains"].items():
    k += 1; cmd.set_color(f"base{k}", hexc(D_["base"])); cmd.color(f"base{k}", f"chain {ch}")
    for j, (n, a, b, c, kind) in enumerate(D_["dom"]): cmd.set_color(f"d{k}_{j}", hexc(c)); cmd.color(f"d{k}_{j}", f"chain {ch} and resi {a}-{b}")
cmd.show("sticks", "ifc and sidechain"); cmd.set("stick_radius", 0.16); cmd.color("0x1A1A1A", "ifc and sidechain and not elem N+O+S"); cmd.show("spheres", "dna and name CA"); cmd.set("sphere_scale", 0.55, "dna")
cmd.orient("keep"); cmd.zoom("keep", 2); cmd.viewport(1700, 1250); v0 = cmd.get_view(); cmd.png(f"{O}/whole.png", width=W_, height=H_, dpi=300, ray=1)
# Flat pass: every domain in a unique ID colour, sticks and spheres included, no shading.
cmd.set("ambient", 1); cmd.set("direct", 0); cmd.set("reflect", 0); cmd.set("specular", 0); cmd.set("antialias", 0); cmd.set("ray_trace_mode", 0); cmd.hide("sticks"); cmd.color("0xFEFEFE", "all"); ids = {}; i = 0
for ch, D_ in J["chains"].items():
    for n, a, b, c, kind in D_["dom"]:
        i += 1; col = (i * 15, 255 - i * 15, 40 + i * 9); ids[f"{ch}:{n}"] = col; cmd.set_color(f"id{i}", [x / 255 for x in col]); cmd.color(f"id{i}", f"chain {ch} and resi {a}-{b}")
cmd.set_view(v0); cmd.png(f"{O}/flat.png", width=FW, height=FH, dpi=300, ray=1); flat = np.asarray(Image.open(f"{O}/flat.png").convert("RGB")).astype(int)
vis = lambda col: (np.abs(flat - np.array(col)).sum(axis=2) <= 6)
M = [v0[0:3], v0[3:6], v0[6:9]]; h = 2 * (-v0[11]) * np.tan(np.radians(cmd.get_setting_float("field_of_view") / 2)); sc = FH / h
out = {}
for ch, D_ in J["chains"].items():
    for n, a, b, c, kind in D_["dom"]:
        key = f"{ch}:{n}"; xyz = []; cmd.iterate_state(1, f"chain {ch} and resi {a}-{b} and keep", "xyz.append((x, y, z))", space={"xyz": xyz}); ntot = cmd.count_atoms(f"chain {ch} and resi {a}-{b}"); mask = vis(ids[key]); npx = int(mask.sum())
        # Unobstructed footprint of the domain alone, to tell hidden from small.
        cmd.hide("everything"); cmd.show("cartoon", f"chain {ch} and resi {a}-{b} and keep"); cmd.show("spheres", f"chain {ch} and resi {a}-{b} and dna and name CA"); cmd.set_view(v0); cmd.png(f"{O}/solo.png", width=FW, height=FH, dpi=300, ray=1); s = np.asarray(Image.open(f"{O}/solo.png").convert("RGB")).astype(int); solo = int((np.abs(s - np.array(ids[key])).sum(axis=2) <= 6).sum())
        cmd.hide("everything"); cmd.show("cartoon", "keep"); cmd.show("spheres", "dna and name CA")
        cen = None
        if xyz:
            p = np.mean(xyz, axis=0) - np.array(v0[12:15]); cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; cen = (FW / 2 + (cam[0] + v0[9]) * sc, FH / 2 - (cam[1] + v0[10]) * sc)
        out[key] = dict(atoms_shown=len(xyz), atoms_all=ntot, px=npx, px_solo=solo, cen=cen)
        if npx and cen:
            ys, xs = np.nonzero(mask); j = np.argmin((xs - cen[0]) ** 2 + (ys - cen[1]) ** 2); out[key]["anchor"] = (float(xs[j]) * 2, float(ys[j]) * 2); out[key]["cen_px_off"] = float(np.hypot(xs[j] - cen[0], ys[j] - cen[1]))
        # Wider check: chain/range of the other atoms under the anchor is shown by px counts only.
        print(key, J["chains"][ch]["name"], n, a, b, "shown atoms", len(xyz), "of", ntot, "visible px", npx, "alone", solo, "frac %.2f" % (npx / max(solo, 1)), "anchor", out[key].get("anchor"))
json.dump(out, open(f"{O}/anchors.json", "w"), indent=0)
