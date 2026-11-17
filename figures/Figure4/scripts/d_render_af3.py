"""Fig 4d step 1 (Longleaf, pymol3d env), CAP-SELEX-style (Xie et al. 2025 Fig. 4d): for each example pair render
(1) the full complex (cartoon, TF chain A in palette-C blue, partner chain B in grey) and (2) a square close-up of the contact region
(contacting side chains as sticks; the 2-3 key contacts are drawn as vector dashed lines by d_compose.py from projected atom positions). Labels are NOT drawn by PyMOL: their 3D anchors are projected to
pixel coordinates (checked against a rendered marker sphere) and written to JSON, so d_compose.py can place editable text.
Model = top-ranked sample (ranking_scores.csv) of the strong seed (ipSAE >= .3 and ipTM >= .4) with the highest ipSAE (out_s5).
Interface = residues with a heavy atom within 5 A of the other chain and residue pLDDT (mean atom B) >= 70; residues with pLDDT < 50
and pLDDT >= 50 fragments < 8 residues are hidden (interface kept).
Key contacts (both residues confident): salt bridges (K NZ / R NE,NH1,NH2 to D OD1,OD2 / E OE1,OE2, <= 4.0 A) first, then side-chain
hydrogen bonds (N/O to N/O, <= 3.5 A, at least one side-chain atom), shortest first, one per residue pair, max 3; if none, the closest
heavy-atom pairs (also used to fill up to 2 contacts); all drawn contacts lie within 12 A (midpoint) of the first, so the inset shows one contact region. Writes <out>/<PAIR>_full.png, <PAIR>_zoom.png, <PAIR>.json and contacts.tsv. usage: d_render_af3.py out_dir [PAIR ...]  (a pair list re-renders only those pairs and updates out_dir/panels.json)
"""
import os, re, sys, json
import numpy as np
from pymol import cmd

S = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PAIRS = sys.argv[2:] or ["RBBP4_ZNF644", "ADNP_MTF2", "DR1_NFYA"]   # Fig 4 trio (2026-11-17: RBBP4-ZNF644 replaced EPAS1-HES1)
COL = {"A": "0x2166AC", "B": "0xB83A2E"}; CSTICK = {"A": "0x1A4E85", "B": "0x8E2A21"}   # chain B palette-C red (2026-11-17: no grey partner)
AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
POS = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}; NEG = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BB = {"N", "CA", "C", "O"}
FW, FH, ZW = 3000, 2600, 2400

def atoms(cif):
    cols, rows, inl = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inl = True; continue
        if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split())
        elif inl and rows and not l.strip(): break
    c = {k: i for i, k in enumerate(cols)}
    R = [r for r in rows if r[c["type_symbol"]] != "H"]
    return dict(ch=np.array([r[c["label_asym_id"]] for r in R]), res=np.array([int(r[c["label_seq_id"]]) for r in R]), rn=np.array([r[c["label_comp_id"]] for r in R]),
                an=np.array([r[c["label_atom_id"]] for r in R]), el=np.array([r[c["type_symbol"]] for r in R]),
                xyz=np.array([[float(r[c[k]]) for k in ("Cartn_x", "Cartn_y", "Cartn_z")] for r in R]), b=np.array([float(r[c["B_iso_or_equiv"]]) for r in R]))

def project(xyz, W, H):
    """pixel coords of model-space points for the current PyMOL view (orthoscopic)"""
    v = cmd.get_view(); Rm = np.array(v[0:9]).reshape(3, 3); cam = np.array(v[9:12]); org = np.array(v[12:15])
    pc = (np.asarray(xyz) - org) @ Rm + cam
    fov = float(cmd.get("field_of_view")); h = 2 * abs(cam[2]) * np.tan(np.radians(fov) / 2); s = H / h
    return np.column_stack([W / 2 + pc[:, 0] * s, H / 2 - pc[:, 1] * s])

def marker_check(pt, W, H, name):
    """render a lone sphere at pt and return its pixel centroid (validates project())"""
    cmd.pseudoatom("mk", pos=[float(x) for x in pt]); cmd.disable("m"); cmd.show("spheres", "mk"); cmd.set("sphere_scale", 1.5, "mk"); cmd.color("0xFF00FF", "mk")
    cmd.png(f"{OUT}/_marker_{name}.png", width=W, height=H, ray=1)
    cmd.delete("mk"); cmd.enable("m")
    from PIL import Image
    a = np.asarray(Image.open(f"{OUT}/_marker_{name}.png").convert("RGB")).astype(int); m = (a[:, :, 0] > 150) & (a[:, :, 1] < 120) & (a[:, :, 2] > 150)
    os.remove(f"{OUT}/_marker_{name}.png"); y, x = np.nonzero(m); return [float(x.mean()), float(y.mean())] if len(x) else None

def style():
    cmd.bg_color("white"); cmd.set("ray_opaque_background", 1); cmd.set("antialias", 2); cmd.set("orthoscopic", 1); cmd.set("depth_cue", 0); cmd.set("fog", 0)
    cmd.set("cartoon_fancy_helices", 1); cmd.set("ray_shadows", 0); cmd.set("specular", 0.15); cmd.set("ambient", 0.45); cmd.set("direct", 0.6)
    cmd.set("ray_trace_mode", 1); cmd.set("ray_trace_gain", 0.1); cmd.set("ray_trace_color", "0x6E6E6E"); cmd.set("cartoon_tube_radius", 0.35)

rows = []; meta = json.load(open(f"{OUT}/panels.json")) if sys.argv[2:] and os.path.exists(f"{OUT}/panels.json") else {}   # a pair list updates an existing run
for p in PAIRS:
    d = f"{S}/out_s5/{p.lower()}"
    seeds = re.findall(r"(\d+):([\d.]+)/([\d.]+) (\w+)", open(f"{S}/scores_s5/{p}.tsv").read().rstrip().split("\t")[-1])
    seed = max([(int(s), float(ps)) for s, ip, ps, t in seeds if t == "strong"], key=lambda v: v[1])[0]
    rk = [l.strip().split(",") for l in open(f"{d}/ranking_scores.csv")][1:]
    sample = int(max((r for r in rk if int(r[0]) == seed), key=lambda r: float(r[2]))[1])
    cif = f"{d}/seed-{seed}_sample-{sample}/model.cif"; T = atoms(cif)
    pl = {c: {r: T["b"][(T["ch"] == c) & (T["res"] == r)].mean() for r in np.unique(T["res"][T["ch"] == c])} for c in "AB"}
    mA, mB = T["ch"] == "A", T["ch"] == "B"; XA, XB = T["xyz"][mA], T["xyz"][mB]
    Dm = np.sqrt(((XA[:, None, :] - XB[None]) ** 2).sum(-1))
    iface = {"A": sorted({int(r) for r in T["res"][mA][(Dm <= 5).any(1)] if pl["A"][r] >= 70}), "B": sorted({int(r) for r in T["res"][mB][(Dm <= 5).any(0)] if pl["B"][r] >= 70})}
    iA, iB = np.nonzero(Dm <= 4.0); cand = []
    for i, j in zip(iA, iB):
        ra, rb = int(T["res"][mA][i]), int(T["res"][mB][j])
        if pl["A"][ra] < 70 or pl["B"][rb] < 70: continue
        a1, a2 = (T["rn"][mA][i], T["an"][mA][i]), (T["rn"][mB][j], T["an"][mB][j]); dd = float(Dm[i, j])
        kind = None
        if ((a1 in POS and a2 in NEG) or (a1 in NEG and a2 in POS)) and dd <= 4.0: kind = "salt bridge"
        elif T["el"][mA][i] in ("N", "O") and T["el"][mB][j] in ("N", "O") and dd <= 3.5 and not (a1[1] in BB and a2[1] in BB): kind = "H-bond"
        if kind: cand.append((0 if kind == "salt bridge" else 1, dd, kind, ra, rb, i, j))
    if not cand:
        k = np.argsort(Dm, axis=None)[:200]
        for f in k:
            i, j = np.unravel_index(f, Dm.shape); ra, rb = int(T["res"][mA][i]), int(T["res"][mB][j])
            if pl["A"][ra] >= 70 and pl["B"][rb] >= 70: cand.append((2, float(Dm[i, j]), "closest heavy atoms", ra, rb, i, j))
    cand.sort(); chosen, seen = [], set()
    for c in cand:
        if (c[3], c[4]) in seen or c[3] in {x[3] for x in chosen} or c[4] in {x[4] for x in chosen}: continue
        mid = (T["xyz"][mA][c[5]] + T["xyz"][mB][c[6]]) / 2
        if chosen and np.linalg.norm(mid - (T["xyz"][mA][chosen[0][5]] + T["xyz"][mB][chosen[0][6]]) / 2) > 12: continue   # one contact region per inset
        chosen.append(c); seen.add((c[3], c[4]))
        if len(chosen) == 3: break
    if len(chosen) < 2 and chosen:                     # fill to 2 with the closest confident heavy-atom pairs in the same region
        r0 = (T["xyz"][mA][chosen[0][5]] + T["xyz"][mB][chosen[0][6]]) / 2
        for f in np.argsort(Dm, axis=None)[:400]:
            i, j = np.unravel_index(f, Dm.shape); ra, rb = int(T["res"][mA][i]), int(T["res"][mB][j])
            if pl["A"][ra] < 70 or pl["B"][rb] < 70 or ra in {x[3] for x in chosen} or rb in {x[4] for x in chosen}: continue
            if np.linalg.norm((T["xyz"][mA][i] + T["xyz"][mB][j]) / 2 - r0) > 12: continue
            chosen.append((2, float(Dm[i, j]), "closest heavy atoms", ra, rb, i, j))
            if len(chosen) == 2: break
    contacts = [dict(kind=k, resA=f"{AA3[T['rn'][mA][i]]}{ra}", atomA=str(T["an"][mA][i]), resB=f"{AA3[T['rn'][mB][j]]}{rb}", atomB=str(T["an"][mB][j]), dist=round(dd, 2),
                     xyzA=T["xyz"][mA][i].tolist(), xyzB=T["xyz"][mB][j].tolist()) for _, dd, k, ra, rb, i, j in chosen]
    # ---------- full view
    cmd.reinitialize(); cmd.load(cif, "m"); cmd.remove("hydro"); style(); cmd.hide("everything")
    for c in "AB":
        ok = sorted(r for r, v in pl[c].items() if v >= 50); keep, run = [], []
        for r in ok + [None]:
            if run and (r is None or r != run[-1] + 1):
                if len(run) >= 8: keep += run
                run = []
            if r is not None: run.append(r)
        keep = sorted(set(keep) | set(iface[c]))
        cmd.select(f"show{c}", f"chain {c} and resi " + "+".join(map(str, keep))); cmd.show("cartoon", f"show{c}"); cmd.color(COL[c], f"chain {c}")
        cmd.select(f"if{c}", f"chain {c} and resi " + "+".join(map(str, iface[c])))
    cmd.orient("ifA or ifB"); cmd.zoom("showA or showB", 1)
    cmd.png(f"{OUT}/{p}_full.png", width=FW, height=FH, dpi=600, ray=1)
    cen = np.mean([x for c in contacts for x in (c["xyzA"], c["xyzB"])], axis=0)
    near = T["xyz"][np.isin(T["res"], iface["A"]) & mA | np.isin(T["res"], iface["B"]) & mB]
    near = near[np.linalg.norm(near - cen, axis=1) <= 12]
    pc = project([cen], FW, FH)[0]; mk = marker_check(cen, FW, FH, p + "_full")
    rad = float(np.percentile(np.linalg.norm(project(near, FW, FH) - pc, axis=1), 90)) if len(near) else 80.0
    labA = T["xyz"][mA][np.isin(T["res"][mA], [r for r, v in pl["A"].items() if v >= 70])]; labB = T["xyz"][mB][np.isin(T["res"][mB], [r for r, v in pl["B"].items() if v >= 70])]
    full = dict(img=f"{p}_full.png", W=FW, H=FH, circle=pc.tolist(), circle_marker=mk, radius=rad,
                anchorA=project([labA.mean(0)], FW, FH)[0].tolist(), anchorB=project([labB.mean(0)], FW, FH)[0].tolist())
    # ---------- zoom view
    sel_res = {"A": set(), "B": set()}
    for c, oc in (("A", mA), ("B", mB)):
        X = T["xyz"][oc]; R_ = T["res"][oc]; ok_ = (np.linalg.norm(X - cen, axis=1) <= 9) & np.isin(R_, iface[c])
        sel_res[c] = set(int(r) for r in R_[ok_])
    for c in contacts: sel_res["A"].add(int(c["resA"][1:])); sel_res["B"].add(int(c["resB"][1:]))
    cmd.hide("everything"); cmd.show("cartoon", "showA or showB"); cmd.set("cartoon_transparency", 0.55)
    for c in "AB":
        s_ = f"chain {c} and resi " + "+".join(map(str, sorted(sel_res[c])))
        cmd.select(f"z{c}", s_); cmd.show("sticks", f"z{c} and (sidechain or name CA)"); cmd.color(CSTICK[c], f"z{c} and elem C")
        cmd.color("0x2C5AA0", f"z{c} and elem N"); cmd.color("0xC0392B", f"z{c} and elem O"); cmd.color("0xC9A227", f"z{c} and elem S")
    cmd.set("stick_radius", 0.2)
    # contact dashes are drawn by d_compose.py (vector, fixed print width) from the projected atom positions below
    cmd.orient("zA or zB"); cmd.zoom("zA or zB", 2.5)
    cmd.png(f"{OUT}/{p}_zoom.png", width=ZW, height=ZW, dpi=600, ray=1)
    zl = []
    for c in contacts:
        a, b = project([c["xyzA"], c["xyzB"]], ZW, ZW)
        ca = T["xyz"][mA][(T["res"][mA] == int(c["resA"][1:])) & (T["an"][mA] == "CA")][0]; cb = T["xyz"][mB][(T["res"][mB] == int(c["resB"][1:])) & (T["an"][mB] == "CA")][0]
        zl.append(dict(pA=a.tolist(), pB=b.tolist(), kind=c["kind"], mid=((a + b) / 2).tolist(), dist=c["dist"], labA=project([ca], ZW, ZW)[0].tolist(), labB=project([cb], ZW, ZW)[0].tolist(), resA=c["resA"], resB=c["resB"]))
    zmk = marker_check(contacts[0]["xyzA"], ZW, ZW, p + "_zoom") if contacts else None
    zoom = dict(img=f"{p}_zoom.png", W=ZW, H=ZW, labels=zl, marker_check=dict(projected=project([contacts[0]["xyzA"]], ZW, ZW)[0].tolist(), rendered=zmk) if contacts else None)
    meta[p] = dict(seed=seed, sample=sample, model=cif.split("/out_s5/")[1], n_iface_A=len(iface["A"]), n_iface_B=len(iface["B"]), full=full, zoom=zoom,
                   contacts=[{k: v for k, v in c.items() if not k.startswith("xyz")} for c in contacts])
    for c in contacts: rows.append(dict(pair=p, seed=seed, sample=sample, model=meta[p]["model"], kind=c["kind"], resA=c["resA"], atomA=c["atomA"], resB=c["resB"], atomB=c["atomB"], dist_A=c["dist"],
                                        n_iface_A=len(iface["A"]), n_iface_B=len(iface["B"])))
json.dump(meta, open(f"{OUT}/panels.json", "w"), indent=1, default=float)
with open(f"{OUT}/contacts.tsv", "w") as f:                     # all pairs in panels.json
    f.write("pair\tseed\tsample\tmodel\tkind\tresA\tatomA\tresB\tatomB\tdist_A\tn_iface_A\tn_iface_B\n")
    for q, m_ in meta.items():
        for c in m_["contacts"]: f.write("\t".join(map(str, [q, m_["seed"], m_["sample"], m_["model"], c["kind"], c["resA"], c["atomA"], c["resB"], c["atomB"], c["dist"], m_["n_iface_A"], m_["n_iface_B"]])) + "\n")
for p, m in meta.items(): print(p, "seed", m["seed"], "sample", m["sample"], "| circle proj", np.round(m["full"]["circle"], 1), "marker", np.round(m["full"]["circle_marker"], 1) if m["full"]["circle_marker"] else None,
                               "| zoom proj/marker", m["zoom"]["marker_check"], "|", [(c["kind"], c["resA"], c["resB"], c["dist"]) for c in m["contacts"]])
