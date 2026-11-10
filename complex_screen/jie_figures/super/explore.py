import pymol, json
from pymol import cmd
pymol.finish_launching(["pymol","-cq"])
def pairs(aln, a, b):
    out=[]
    for x,y in cmd.get_raw_alignment(aln):
        A=x if x[0]==a else y; B=y if y[0]==b else x
        m=[];cmd.iterate(f"index {A[1]} and {a}","m.append((chain,resi,resn))",space={"m":m});n=[];cmd.iterate(f"index {B[1]} and {b}","n.append((chain,resi,resn))",space={"n":n}); out.append((m[0],n[0]))
    return out
# MLX-SREBF2 vs MYC-MAX and SREBP1a
cmd.load("mlx_srebf2_model.cif","mod"); cmd.remove("mod and hydro")
for ref,pm in (("1NKP",[("A","129-220","B"),("B","330-401","A")]),("1AM9",[("A","129-220","A"),("B","330-401","B")])):
    cmd.load(f"{ref}.cif",ref); cmd.remove(f"{ref} and (hydro or solvent)")
    chs=[];cmd.iterate(f"{ref} and name CA","chs.append(chain)",space={"chs":chs}); print(ref,{c:chs.count(c) for c in set(chs)})
    for mc,rng,rc in pm:
        rmsd=cmd.align(f"mod and chain {mc} and resi {rng} and name CA",f"{ref} and chain {rc} and name CA",cycles=0,object=f"a{mc}{ref}",transform=0); print(ref,"per-chain",mc,rc,[round(x,2) if isinstance(x,float) else x for x in rmsd[:5]])
    sel_m=" or ".join(f"(chain {mc} and resi {rng})" for mc,rng,rc in pm); sel_r=" or ".join(f"chain {rc}" for mc,rng,rc in pm)
    r=cmd.super(f"mod and ({sel_m}) and name CA",f"{ref} and ({sel_r}) and name CA",object=f"s{ref}",transform=0); print(ref,"super dimer",[round(x,2) if isinstance(x,float) else x for x in r[:5]])
cmd.delete("all")
# PRC2
cmd.load("ezh2_suz12_mtf2_model.cif","mod"); cmd.load("6NQ3.cif","ref"); cmd.remove("(hydro or solvent)")
r=cmd.align("mod and chain B and name CA","ref and chain B and name CA",object="aS",cycles=5); print("SUZ12 align",[round(x,2) if isinstance(x,float) else x for x in r[:5]])
c=[];cmd.iterate("ref and chain C and name CA","c.append(int(resi))",space={"c":c}); print("PHF19 ref resi",c[0],c[-1],len(c)); c=[];cmd.iterate("ref and chain B and name CA","c.append(int(resi))",space={"c":c}); print("SUZ12 ref resi",c[0],c[-1],len(c))
r=cmd.align("mod and chain C and name CA","ref and chain C and name CA",object="aP",cycles=0,transform=0); print("MTF2-PHF19 (no refit)",[round(x,2) if isinstance(x,float) else x for x in r[:5]]); P=pairs("aP","mod","ref"); print(len(P),P[:3],P[-3:])
cmd.save("prc2_aln.pse")
