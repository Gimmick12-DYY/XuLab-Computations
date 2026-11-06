import pymol, json, math
from pymol import cmd
pymol.finish_launching(["pymol","-cq"])
def maps(a_sel, b_sel, name):
    cmd.align(a_sel, b_sel, cycles=0, object=name, transform=0); P = {}
    for pr in cmd.get_raw_alignment(name):
        x = next(q for q in pr if q[0] == "mod"); y = next(q for q in pr if q[0] == "ref"); m = []; n = []; cmd.iterate(f"index {x[1]} and mod", "m.append((chain, resi, resn))", space={"m": m}); cmd.iterate(f"index {y[1]} and ref", "n.append((chain, resi, resn))", space={"n": n}); P[n[0]] = m[0]
    return P
def polar(ref, ca, cb, cut=3.6):
    out = []; a_ = []; cmd.iterate_state(1, f"{ref} and chain {ca} and sidechain and elem N+O", "a_.append((resi, resn, name, x, y, z))", space={"a_": a_}); b_ = []; cmd.iterate_state(1, f"{ref} and chain {cb} and sidechain and elem N+O", "b_.append((resi, resn, name, x, y, z))", space={"b_": b_})
    for a in a_:
        for b in b_:
            d = math.dist(a[3:], b[3:])
            if d < cut: out.append((a[:3], b[:3], round(d, 2)))
    return out
def atomd(s1, s2):
    return round(cmd.get_distance(f"({s1})", f"({s2})"), 2)
# MLX-SREBF2 on MYC-MAX
cmd.load("mlx_srebf2_model.cif", "mod"); cmd.load("1NKP.cif", "ref"); cmd.remove("hydro or solvent")
r = cmd.super("mod and ((chain A and resi 129-220) or (chain B and resi 330-401)) and name CA", "ref and (chain A or chain B) and name CA", object="s1"); print("super", [round(x, 2) if isinstance(x, float) else x for x in r[:5]])
Mx = maps("mod and chain A and resi 129-220 and name CA", "ref and chain B and name CA", "m1"); My = maps("mod and chain B and resi 330-401 and name CA", "ref and chain A and name CA", "m2"); print(len(Mx), len(My), list(Mx.items())[:3])
for a, b, d in polar("ref", "A", "B"):
    ka = next((k for k in My if k[1] == a[0]), None); kb = next((k for k in Mx if k[1] == b[0]), None)
    if ka and kb: print("iface", "MYC", a, "MAX", b, d, "->", My[ka], Mx[kb], "CA-CA ref", atomd(f"ref and chain A and resi {a[0]} and name CA", f"ref and chain B and resi {b[0]} and name CA"), "model", atomd(f"mod and chain {My[ka][0]} and resi {My[ka][1]} and name CA", f"mod and chain {Mx[kb][0]} and resi {Mx[kb][1]} and name CA"))
# basic-region glutamate to E-box base
for chain, M_, nm in (("B", Mx, "MAX"), ("A", My, "MYC")):
    best = []; cmd.iterate_state(1, f"ref and chain {chain} and sidechain and elem O+N", "best.append((resi, resn, name, x, y, z))", space={"best": best}); dna = []; cmd.iterate_state(1, "ref and polymer.nucleic and not (name P+OP1+OP2+O3'+O5'+O4'+C1'+C2'+C3'+C4'+C5') and elem N+O", "dna.append((chain, resi, resn, name, x, y, z))", space={"dna": dna})
    for a in best:
        for d in dna:
            D = math.dist(a[3:], d[4:])
            if D < 3.5:
                k = next((q for q in M_ if q[1] == a[0]), None); mm = M_.get(k) if k else None
                print("DNA", nm, a[:3], d[:4], round(D, 2), "->", mm, "model atom-to-base", (atomd(f"mod and chain {mm[0]} and resi {mm[1]} and name {a[2]}", f"ref and chain {d[0]} and resi {d[1]} and name {d[3]}") if mm and mm[2] == a[1] else "type differs"))
cmd.delete("all")
# PRC2: local fit on the SUZ12 surface that holds PHF19
cmd.load("ezh2_suz12_mtf2_model.cif", "mod"); cmd.load("6NQ3.cif", "ref"); cmd.remove("hydro or solvent")
cmd.select("site", "ref and chain B and name CA and (byres (ref and chain B within 8 of (ref and chain C)))"); n = cmd.count_atoms("site"); P = maps("mod and chain B and name CA", "ref and chain B and name CA", "mS"); rs = []; cmd.iterate("site", "rs.append(resi)", space={"rs": rs}); mres = [P[k][1] for k in P if k[0] == "B" and k[1] in rs]; print("SUZ12 site residues", n, "mapped", len(mres), rs[0], rs[-1])
r = cmd.super(f"mod and chain B and resi {'+'.join(mres)} and name CA", "site", object="s2"); print("local super", [round(x, 2) if isinstance(x, float) else x for x in r[:5]]); r2 = cmd.align("mod and chain C and name CA", "ref and chain C and name CA", cycles=0, transform=0, object="aP"); print("MTF2 vs PHF19 after local fit, no refit", [round(x, 2) if isinstance(x, float) else x for x in r2[:5]])
Q = maps("mod and chain C and name CA", "ref and chain C and name CA", "mP"); D_ = []
for k, v in Q.items():
    D_.append(atomd(f"ref and chain C and resi {k[1]} and name CA", f"mod and chain C and resi {v[1]} and name CA"))
print("CA displacement MTF2 vs PHF19: median", sorted(D_)[len(D_) // 2], "min", min(D_), "max", max(D_), "n", len(D_), "mapping", list(Q.items())[0], list(Q.items())[-1])
for a, b, d in polar("ref", "B", "C"):
    ka = P.get(("B", a[0], a[1])); kb = next((v for k, v in Q.items() if k[1] == b[0]), None)
    if ka and kb: print("iface", "SUZ12", a, "PHF19", b, d, "->", ka, kb, "CA-CA ref", atomd(f"ref and chain B and resi {a[0]} and name CA", f"ref and chain C and resi {b[0]} and name CA"), "model", atomd(f"mod and chain B and resi {ka[1]} and name CA", f"mod and chain C and resi {kb[1]} and name CA"))
