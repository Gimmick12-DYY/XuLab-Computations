# Render AF3 and Boltz full-length top models: chain colours, interface sticks, confident residues only (pLDDT >= 50, interface residues always kept). Run headless with pymol -cq.
import os, sys, json, glob
from pymol import cmd
C = "/work/users/d/y/dyy12/XuLab/complex_screen"; O = f"{C}/jie_structures"; os.makedirs(O, exist_ok=True)
def boltz_top(pid):
    d = f"{C}/fulllength_check/out_seed1/boltz_results_inputs/predictions/{pid}_full"; c = sorted(glob.glob(f"{d}/confidence_{pid}_full_model_*.json")); s = [json.load(open(f))["confidence_score"] for f in c]; k = c[s.index(max(s))].split("_model_")[1][:-5]; return f"{d}/{pid}_full_model_{k}.cif"
J = [("MLX_SREBF2_AF3", f"{C}/af3_local/out/mlx_srebf2/mlx_srebf2_model.cif"), ("MLX_SREBF2_Boltz", boltz_top("MLX__SREBF2")), ("HOXC4_RBBP4_AF3", f"{C}/af3_local/out/hoxc4_rbbp4/hoxc4_rbbp4_model.cif")]
for name, f in J:
    cmd.reinitialize(); cmd.load(f, "m"); cmd.remove("hydro or solvent"); cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("cartoon_fancy_helices", 1)
    cmd.select("ia", "byres (chain A within 5 of chain B)"); cmd.select("ib", "byres (chain B within 5 of chain A)")
    res = {}
    for ch, s in (("A", "ia"), ("B", "ib")):
        r = set(); cmd.iterate(s, "r.add(int(resi))", space={"r": r}); res[ch] = sorted(r)
    json.dump(res, open(f"{O}/{name}.interface.json", "w"))
    cmd.select("keep", "(b > 50) or ia or ib"); cmd.hide("everything"); cmd.show("cartoon", "keep")
    cmd.set_color("chA", [0x21 / 255, 0x66 / 255, 0xAC / 255]); cmd.set_color("chB", [0xB2 / 255, 0x18 / 255, 0x2B / 255]); cmd.color("chA", "chain A"); cmd.color("chB", "chain B")
    cmd.show("sticks", "(ia or ib) and sidechain"); cmd.set("stick_radius", 0.18); util = None
    cmd.orient("ia or ib"); cmd.zoom("ia or ib", 6); cmd.png(f"{O}/{name}_iface.png", width=1400, height=1100, dpi=300, ray=1)
    cmd.hide("sticks"); cmd.orient("keep"); cmd.zoom("keep", 3); cmd.png(f"{O}/{name}_whole.png", width=1400, height=1100, dpi=300, ray=1)
    print(name, f, {k: len(v) for k, v in res.items()})
