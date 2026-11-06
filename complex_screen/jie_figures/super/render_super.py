# Superposition renders in the reference-figure style: experimental structure in grey, AF3 model in one colour; one overview, one close-up per contact. Writes PNGs and 2D anchors for the labels, and the distances. Runs headless on the Mac.
import json, math, pymol
from pymol import cmd
pymol.finish_launching(["pymol", "-cq"]); OUT = {}; hx = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
def setup():
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("ray_shadows", 0); cmd.set("antialias", 2); cmd.set("depth_cue", 0); cmd.set("orthoscopic", 1); cmd.set("cartoon_fancy_helices", 1); cmd.set("stick_radius", 0.16); cmd.set("dash_gap", 0.25); cmd.set("dash_width", 3); cmd.set("dash_radius", 0.07); cmd.set("label_size", 0); cmd.set_color("refgrey", hx("#7A7A7A")); cmd.set_color("ink", hx("#1A1A1A"))
def proj(pt, W, H):
    v = cmd.get_view(); M = [v[0:3], v[3:6], v[6:9]]; p = [pt[i] - v[12 + i] for i in range(3)]; cam = [sum(p[k] * M[k][j] for k in range(3)) for j in range(3)]; h = 2 * (-v[11]) * math.tan(math.radians(cmd.get_setting_float("field_of_view") / 2)); sc = H / h; return (W / 2 + (cam[0] + v[9]) * sc, H / 2 - (cam[1] + v[10]) * sc)
def xyz(sel):
    out_ = []; cmd.iterate_state(1, sel, "out_.append((x, y, z))", space={"out_": out_}); assert out_, sel; return [sum(c[i] for c in out_) / len(out_) for i in range(3)]
def d(a, b): return round(cmd.get_distance(f"({a})", f"({b})"), 1)
def show_pair(job, tag, ra, rb, ma, mb, col, ww, hh, zoom=5, extra=None):
    cmd.hide("everything"); cmd.show("cartoon", "vis"); cmd.set("cartoon_transparency", 0.72, "vis")
    S_ = f"({ra}) or ({rb}) or ({ma}) or ({mb})"; cmd.show("sticks", f"byres ({S_})"); cmd.distance(f"dr{tag}", ra, rb); cmd.distance(f"dm{tag}", ma, mb); cmd.color("ink", f"dr{tag}"); cmd.set_color(f"mc{tag}", hx(col)); cmd.color(f"mc{tag}", f"dm{tag}"); cmd.hide("labels", f"dr{tag}"); cmd.hide("labels", f"dm{tag}")
    if extra: cmd.show("sticks", extra)
    cmd.orient(S_); cmd.zoom(S_, zoom); W, H = ww; cmd.viewport(1700, int(1700 * H / W)); cmd.png(f"{job}_{tag}.png", width=W, height=H, dpi=300, ray=1); pos = {}
    for k, s in (("ra", ra), ("rb", rb), ("ma", ma), ("mb", mb)): pos[k] = proj(xyz(s), W, H)
    cmd.delete(f"dr{tag}"); cmd.delete(f"dm{tag}"); return pos
# MLX-SREBF2 on MYC-MAX DNA
setup(); cmd.load("mlx_srebf2_model.cif", "mod"); cmd.load("1NKP.cif", "ref"); cmd.remove("hydro or solvent"); cmd.remove("ref and not (chain A or chain B or chain F or chain G)"); cmd.remove("mod and not ((chain A and resi 129-220) or (chain B and resi 330-401))"); COL = "#B2182B"
r = cmd.super("mod and name CA", "ref and (chain A or chain B) and name CA", object="sup"); cmd.delete("sup"); cmd.select("vis", "all"); cmd.set_color("mc", hx(COL)); cmd.color("refgrey", "ref"); cmd.color("mc", "mod"); cmd.hide("everything"); cmd.show("cartoon", "polymer.protein"); cmd.show("sticks", "ref and polymer.nucleic"); cmd.set("stick_radius", 0.10, "ref and polymer.nucleic"); cmd.color("refgrey", "ref and polymer.nucleic")
cmd.orient("ref"); cmd.zoom("all", 2); W, H = 3120, 3120; cmd.viewport(1500, 1500); cmd.png("MLX_over.png", width=W, height=H, dpi=300, ray=1)
c1 = dict(ra="ref and chain B and resi 249 and name OH", rb="ref and chain A and resi 954 and name NE2", ma="mod and chain A and resi 183 and name OH", mb="mod and chain B and resi 381 and name NE2"); c2 = dict(ra="ref and chain A and resi 910 and name OE2", rb="ref and chain F and resi 109 and name N6", ma="mod and chain B and resi 339 and name OE2", mb="ref and chain F and resi 109 and name N6")
J = dict(rmsd=round(r[0], 2), n=r[1], c1=dict(ref=d(c1["ra"], c1["rb"]), model=d(c1["ma"], c1["mb"])), c2=dict(ref=d(c2["ra"], c2["rb"]), model=d(c2["ma"], c2["mb"]))); J["over"] = dict(MLX=proj(xyz("mod and chain A and resi 195 and name CA"), W, H), SREBF2=proj(xyz("mod and chain B and resi 392 and name CA"), W, H), MAX=proj(xyz("ref and chain B and resi 230 and name CA"), W, H), MYC=proj(xyz("ref and chain A and resi 935 and name CA"), W, H), c1=proj(xyz(f"{c1['ra']}"), W, H), c2=proj(xyz(f"{c2['ra']}"), W, H), dna=proj(xyz("ref and chain F and resi 109 and name N1"), W, H)); J["W"] = W
J["p1"] = show_pair("MLX", "c1", c1["ra"], c1["rb"], c1["ma"], c1["mb"], COL, (2880, 2160), 2160); J["p2"] = show_pair("MLX", "c2", c2["ra"], c2["rb"], c2["ma"], c2["mb"], COL, (2880, 2160), 2160, zoom=8, extra="ref and chain F and resi 109")
OUT["MLX"] = J; cmd.reinitialize()
# PRC2: local fit on the SUZ12 surface holding PHF19
setup(); cmd.load("ezh2_suz12_mtf2_model.cif", "mod"); cmd.load("6NQ3.cif", "ref"); cmd.remove("hydro or solvent"); cmd.remove("ref and not (chain B or chain C)"); COL = "#2166AC"
cmd.select("site", "ref and chain B and name CA and (byres (ref and chain B within 8 of (ref and chain C)))"); rs = []; cmd.iterate("site", "rs.append(resi)", space={"rs": rs}); r = cmd.super(f"mod and chain B and resi {'+'.join(rs)} and name CA", "site", object="sup"); cmd.delete("sup"); glob = cmd.align("mod and chain B and name CA", "ref and chain B and name CA", cycles=0, transform=0, object="ag"); cmd.delete("ag")
ib = []; cmd.iterate("byres (mod and chain C within 5 of (mod and chain B))", "ib.append(int(resi))", space={"ib": ib}); ia = []; cmd.iterate("byres (mod and chain B within 5 of (mod and chain C))", "ia.append(int(resi))", space={"ia": ia}); ic = []; cmd.iterate("byres (ref and chain B within 5 of (ref and chain C))", "ic.append(int(resi))", space={"ic": ic}); ov = sorted(set(ia) & set(ic))
cmd.remove("mod and chain A"); cmd.remove("mod and b < 50 and not chain A and not (chain C and resi 541-591)"); cmd.select("vis", "all"); cmd.set_color("mc", hx(COL)); cmd.color("refgrey", "ref"); cmd.color("mc", "mod"); cmd.hide("everything"); cmd.show("cartoon", "all")
cmd.orient("ref"); cmd.zoom("all", 2); W, H = 3120, 3120; cmd.viewport(1500, 1500); cmd.png("PRC2_over.png", width=W, height=H, dpi=300, ray=1)
c1 = dict(ra="ref and chain B and resi 95 and name NE2", rb="ref and chain C and resi 542 and name OH", ma="mod and chain B and resi 95 and name NE2", mb="mod and chain C and resi 555 and name OH")
J = dict(rmsd=round(r[0], 2), n=r[1], site_n=len(rs), suz12_all_rmsd=round(glob[0], 2), suz12_all_n=glob[1], c1=dict(ref=d(c1["ra"], c1["rb"]), model=d(c1["ma"], c1["mb"])), model_mtf2_iface=sorted(set(ib)), model_suz12_iface=sorted(set(ia)), ref_suz12_iface_n=len(set(ic)), shared_suz12=ov, mtf2_cterm_in_iface=[x for x in sorted(set(ib)) if 541 <= x <= 591])
J["over"] = dict(MTF2=proj(xyz("mod and chain C and resi 555 and name CA"), W, H), PHF19=proj(xyz("ref and chain C and resi 542 and name CA"), W, H), SUZ12=proj(xyz("ref and chain B and resi 300 and name CA"), W, H), c1=proj(xyz(c1["ra"]), W, H)); J["W"] = W; J["p1"] = show_pair("PRC2", "c1", c1["ra"], c1["rb"], c1["ma"], c1["mb"], COL, (2880, 2160), 2160, zoom=6)
OUT["PRC2"] = J; json.dump(OUT, open("super.json", "w"), indent=1); print(json.dumps({k: {x: v[x] for x in v if x not in ("over", "p1", "p2", "model_mtf2_iface", "model_suz12_iface")} for k, v in OUT.items()})); print("MTF2 iface", OUT["PRC2"]["model_mtf2_iface"][:80])
