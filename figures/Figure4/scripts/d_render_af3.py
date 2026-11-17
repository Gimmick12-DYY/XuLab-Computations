"""Fig 4d step 1 (Longleaf, pymol3d env): render the AF3 5-seed model of each example pair with PyMOL.
Model = top-ranked sample (ranking_scores.csv) of the strong seed (ipSAE >= .3 and ipTM >= .4) with the highest ipSAE, from out_s5.
Interface = residues with any heavy atom within 5 A of the other chain and residue pLDDT (mean atom B) >= 70.
Style: white background, ray-traced cartoon, no fog; residues with pLDDT < 50 and pLDDT >= 50 fragments < 8 residues hidden (interface kept), chain A / chain B in fixed palette-C colours,
interface residues as sticks in the darker shade of their chain. Writes <out>/<PAIR>.png and <out>/structures.tsv (model, per-side counts).
usage: d_render_af3.py out_dir
"""
import os, re, sys, json
import numpy as np
from pymol import cmd

S = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PAIRS = ["EPAS1_HES1", "ADNP_MTF2", "DR1_NFYA"]
COL = {"A": ("0x7A9CCB", "0x2166AC"), "B": ("0xF4A582", "0xB83A2E")}   # cartoon, interface sticks (palette C blue / red family)

def atoms(cif):
    cols, rows, inl = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inl = True; continue
        if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split())
        elif inl and rows and not l.strip(): break
    c = {k: i for i, k in enumerate(cols)}
    R = [r for r in rows if r[c["type_symbol"]] != "H"]
    ch = np.array([r[c["label_asym_id"]] for r in R]); res = np.array([int(r[c["label_seq_id"]]) for r in R])
    xyz = np.array([[float(r[c[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")] for r in R]); b = np.array([float(r[c["B_iso_or_equiv"]]) for r in R])
    return ch, res, xyz, b

rows = []
for p in PAIRS:
    d = f"{S}/out_s5/{p.lower()}"
    seeds = re.findall(r"(\d+):([\d.]+)/([\d.]+) (\w+)", open(f"{S}/scores_s5/{p}.tsv").read().rstrip().split("\t")[-1])
    strong = [(int(s), float(ip), float(ps)) for s, ip, ps, t in seeds if t == "strong"]
    seed = max(strong, key=lambda v: v[2])[0]
    rk = [l.strip().split(",") for l in open(f"{d}/ranking_scores.csv")][1:]
    sample = int(max((r for r in rk if int(r[0]) == seed), key=lambda r: float(r[2]))[1])
    cif = f"{d}/seed-{seed}_sample-{sample}/model.cif"
    ch, res, xyz, b = atoms(cif); iface, plddt = {}, {}
    for c, o in (("A", "B"), ("B", "A")):
        m, n = ch == c, ch == o
        dmin = np.full(m.sum(), np.inf); X2 = xyz[n]
        for i in range(0, m.sum(), 2000): dmin[i:i + 2000] = np.sqrt(((xyz[m][i:i + 2000, None, :] - X2[None]) ** 2).sum(-1)).min(1)
        pl = {r: b[m][res[m] == r].mean() for r in np.unique(res[m])}; plddt[c] = pl
        iface[c] = sorted({int(r) for r in res[m][dmin <= 5.0] if pl[r] >= 70})
    cmd.reinitialize(); cmd.load(cif, "m"); cmd.remove("hydro")
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("orthoscopic", 1)
    cmd.set("cartoon_fancy_helices", 1); cmd.set("cartoon_transparency", 0); cmd.set("ray_shadows", 0); cmd.set("specular", 0.15)
    cmd.set("ambient", 0.45); cmd.set("direct", 0.6); cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_gain", 0.1); cmd.set("ray_trace_color", "0x6E6E6E")
    cmd.hide("everything"); SHOWN = {}
    for c in "AB":
        ok = sorted(r for r, v in plddt[c].items() if v >= 50); keep, run = [], []
        for r in ok + [None]:                         # drop isolated pLDDT >= 50 fragments shorter than 8 residues
            if run and (r is None or r != run[-1] + 1):
                if len(run) >= 8: keep += run
                run = []
            if r is not None: run.append(r)
        keep = sorted(set(keep) | set(iface[c])); SHOWN[c] = len(keep)
        cmd.select(f"show{c}", f"chain {c} and resi " + "+".join(map(str, keep)))
        cmd.show("cartoon", f"show{c}"); cmd.color(COL[c][0], f"chain {c}")
        if iface[c]:
            cmd.select(f"if{c}", f"chain {c} and resi " + "+".join(map(str, iface[c])))
            cmd.show("sticks", f"if{c} and not name N+C+O"); cmd.color(COL[c][1], f"if{c} and not name N+C+O")
    cmd.set("stick_radius", 0.22); cmd.set("cartoon_tube_radius", 0.35)
    cmd.set("depth_cue", 0); cmd.set("fog", 0)
    cmd.orient("ifA or ifB"); cmd.zoom("showA or showB", 1)
    cmd.png(f"{OUT}/{p}.png", width=3600, height=3000, dpi=600, ray=1)
    rows.append(dict(pair=p, seed=seed, sample=sample, model=cif.split("/out_s5/")[1], n_iface_A=len(iface["A"]), n_iface_B=len(iface["B"]),
                     n_shown_A=SHOWN["A"], n_len_A=len(plddt["A"]), n_shown_B=SHOWN["B"], n_len_B=len(plddt["B"]),
                     iface_A=",".join(map(str, iface["A"])), iface_B=",".join(map(str, iface["B"]))))
with open(f"{OUT}/structures.tsv", "w") as f:
    f.write("\t".join(rows[0]) + "\n"); [f.write("\t".join(map(str, r.values())) + "\n") for r in rows]
print(json.dumps([{k: r[k] for k in ("pair", "seed", "sample", "n_iface_A", "n_iface_B", "n_shown_A", "n_len_A", "n_shown_B", "n_len_B")} for r in rows], default=int))
