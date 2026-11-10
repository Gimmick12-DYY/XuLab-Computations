# Domain-annotated AF3 renders for the Jie slides, headless PyMOL. Domains from UniProt/InterPro (see DOM). Interface = residues within 5 A of another chain (written to JSON). pLDDT < 50 is hidden unless annotated as interface or DNA site.
# usage: pymol -cq render_sites.py -- <job> [job ...]
import json, os, sys
from pymol import cmd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/jie_structures"
DOM = json.load(open(f"{C}/composite_motif/jie_domains.json"))
hexc = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
for job in sys.argv[1:]:
    J = DOM[job]; d = f"{C}/af3_local/out/{job.lower()}"; cmd.reinitialize(); cmd.load(f"{d}/{job.lower()}_model.cif", "m"); cmd.remove("hydro or solvent"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1); cmd.set("label_size", 22); cmd.set("label_font_id", 7); cmd.set("label_color", "black"); cmd.set("label_outline_color", "white")
    chains = list(J["chains"]); res = {}; sel = []
    for ch in chains:
        oth = " or ".join(f"chain {x}" for x in chains if x != ch); cmd.select(f"i{ch}", f"byres (chain {ch} within 5 of ({oth}))"); r = set(); cmd.iterate(f"i{ch}", "r.add(int(resi))", space={"r": r}); res[ch] = sorted(r); sel.append(f"i{ch}")
        for o in chains:
            if o != ch:
                r2 = set(); cmd.iterate(f"byres (chain {ch} within 5 of chain {o})", "r2.add(int(resi))", space={"r2": r2}); res[f"{ch}_to_{o}"] = sorted(r2)
    json.dump(res, open(f"{O}/{job}.sites.json", "w"))
    cmd.select("ifc", " or ".join(sel)); dna = " or ".join(f"(chain {ch} and resi {a}-{b})" for ch, D_ in J["chains"].items() for n, a, b, c, kind in D_["dom"] if kind == "dna") or "none"; cmd.select("dna", dna); cmd.select("keep", "(b > 50) or ifc or dna"); cmd.hide("everything"); cmd.show("cartoon", "keep"); k = 0
    for ch, D_ in J["chains"].items():
        k += 1; cmd.set_color(f"base{k}", hexc(D_["base"])); cmd.color(f"base{k}", f"chain {ch}")
        for j, (n, a, b, c, kind) in enumerate(D_["dom"]):
            cmd.set_color(f"d{k}_{j}", hexc(c)); cmd.color(f"d{k}_{j}", f"chain {ch} and resi {a}-{b}")

    cmd.show("sticks", "ifc and sidechain"); cmd.set("stick_radius", 0.16); cmd.color("0x1A1A1A", "ifc and sidechain and not elem N+O+S"); cmd.show("spheres", "dna and name CA"); cmd.set("sphere_scale", 0.55, "dna")
    cmd.orient("keep"); cmd.zoom("keep", 2); W_, H_ = 3960, 2910; cmd.viewport(1700, 1250); cmd.png(f"{O}/{job}_sites_whole.png", width=W_, height=H_, dpi=300, ray=1); v = cmd.get_view(); M = [v[0:3], v[3:6], v[6:9]]; pos = {}
    for ch, D_ in J["chains"].items():
        for n, a, b, c, kind in D_["dom"]:
            xy = []; cmd.iterate_state(1, f"chain {ch} and resi {(a + b) // 2} and name CA", "xy.append((x, y, z))", space={"xy": xy})
            if xy:
                p = [xy[0][i] - v[12 + i] for i in range(3)]; cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; h = 2 * (-v[11]) * __import__("math").tan(__import__("math").radians(cmd.get_setting_float("field_of_view") / 2)); sc = H_ / h
                pos[f"{ch}:{n}"] = (W_ / 2 + (cam[0] + v[9]) * sc, H_ / 2 - (cam[1] + v[10]) * sc)
    json.dump(pos, open(f"{O}/{job}.labels.json", "w"))
    cmd.orient("ifc"); cmd.zoom("ifc", 6); cmd.png(f"{O}/{job}_sites_iface.png", width=1700, height=1250, dpi=300, ray=1); print(job, {k_: len(v) for k_, v in res.items()})
