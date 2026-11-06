# Score Boltz-2 protein-DNA predictions against the crystal (8BYX, 1NKP): protein CA / DNA C4' RMSD after global superposition,
# protein placement after DNA-only superposition, side-of-DNA check (azimuth of each protein around the DNA axis), ipTM, ipSAE
# (protein-protein, protein-DNA). Writes scores.tsv and the superposed top model (<target>_top_superposed.pdb) + crystal subset.
import glob, itertools, json, sys
import numpy as np, gemmi

D = sys.argv[1]
TARGETS = {  # predicted chain order: proteins then DNA strands; crystal chains are matched by sequence + contacts
    "8BYX_HOXB13_dimer_DNA": dict(pdb="8BYX", prot=["A", "B"], dna=["C", "D"]),
    "1NKP_MYC_MAX_DNA": dict(pdb="1NKP", prot=["A", "B"], dna=["C", "D"]),
}
NUC = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}

def chain_seq(ch):
    p = ch.get_polymer()
    return "".join(NUC[r.name] for r in p) if p and p[0].name in NUC else gemmi.one_letter_code([r.name for r in p])

def coords(ch, nm, crystal):
    out = {}
    for r in ch:
        a = r.find_atom(nm, "*")
        k = r.label_seq if crystal else r.seqid.num
        if a is not None and k is not None: out[k] = np.array(a.pos.tolist())
    return out

def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, qc - pc @ R

def rmsd(A, B): return float(np.sqrt(((A - B) ** 2).sum(1).mean()))

def ipsae(pae, ia, ib, cut=10.0):
    best = 0.0
    for r, c in [(ia, ib), (ib, ia)]:
        sub = pae[np.ix_(r, c)]; ok = sub < cut; n = ok.sum(1)
        d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1)
        best = max(best, float(s.max()))
    return best

def n_contacts(m, ns, a, b):
    """residues of chain a with any atom within 4.5 A of chain b (KD-tree; gemmi NeighborSearch segfaulted here)"""
    from scipy.spatial import cKDTree
    tb = cKDTree(np.array([at.pos.tolist() for r in m[b] for at in r]))
    return sum(1 for r in m[a] if any(tb.query_ball_point(at.pos.tolist(), 4.5) for at in r))

def pairs(pred_ids, xtal_ids, pseq, xseq):
    """all assignments of predicted -> crystal chains with matching sequence (handles homodimers / identical strands)"""
    out = []
    for perm in itertools.permutations(xtal_ids, len(pred_ids)):
        if all(pseq[p] in xseq[x] or xseq[x] in pseq[p] for p, x in zip(pred_ids, perm)): out.append(dict(zip(pred_ids, perm)))
    return out

def centroid(d): return np.mean(list(d.values()), 0)

rows = []
for tgt, cfg in TARGETS.items():
    xs = gemmi.read_structure(f"{D}/pdb/{cfg['pdb']}.cif"); xs.setup_entities(); xm = xs[0]
    ns = None
    xseq = {c.name: chain_seq(c) for c in xm if len(c.get_polymer()) > 3}
    pred_dir = f"{D}/out/boltz_results_inputs/predictions/{tgt}"
    models = sorted(glob.glob(f"{pred_dir}/{tgt}_model_*.cif"), key=lambda f: int(f.split("_model_")[1][:-4]))
    if not models: print(tgt, "no predictions"); continue
    p0 = gemmi.read_structure(models[0])[0]
    pseq = {c.name: chain_seq(c) for c in p0}
    # crystal: the DNA duplex with most inter-strand contacts, then the protein chains contacting it most
    xdna_all = [n for n in xseq if set(xseq[n]) <= set("ACGT")]
    dpair = max(itertools.combinations(xdna_all, 2), key=lambda ab: n_contacts(xm, ns, *ab))
    xprot = sorted([n for n in xseq if n not in xdna_all], key=lambda p: -sum(n_contacts(xm, ns, p, d) for d in dpair))[:len(cfg["prot"])]
    for k, mf in enumerate(models):
        pm = gemmi.read_structure(mf)[0]
        best = None
        for mp in pairs(cfg["prot"], xprot, pseq, xseq):
            for md in pairs(cfg["dna"], list(dpair), pseq, xseq):
                P, X, lab = [], [], []
                for pc, xc in list(mp.items()) + list(md.items()):
                    nm = "CA" if pc in cfg["prot"] else "C4'"
                    a, b = coords(pm[pc], nm, False), coords(xm[xc], nm, True)
                    for i in sorted(set(a) & set(b)): P.append(a[i]); X.append(b[i]); lab.append(pc)
                P, X, lab = np.array(P), np.array(X), np.array(lab)
                R, t = kabsch(P, X); Pg = P @ R + t
                isp = np.isin(lab, cfg["prot"]); isd = ~isp
                Rd, td = kabsch(P[isd], X[isd]); Pd = P @ Rd + td  # DNA-only superposition
                # DNA axis = first principal component of crystal C4' atoms; azimuth of each protein centroid around it
                Xd = X[isd]; c0 = Xd.mean(0); ax = np.linalg.svd(Xd - c0)[2][0]
                def azim(v):
                    w = v - c0; w = w - (w @ ax) * ax; return w / np.linalg.norm(w)
                ang = {pc: float(np.degrees(np.arccos(np.clip(azim(Pd[lab == pc].mean(0)) @ azim(X[lab == pc].mean(0)), -1, 1))))
                       for pc in cfg["prot"]}
                r = dict(target=tgt, model=k, chain_map=",".join(f"{a}>{b}" for a, b in list(mp.items()) + list(md.items())),
                         rmsd_all=rmsd(Pg, X), rmsd_prot_CA=rmsd(Pg[isp], X[isp]), rmsd_dna_C4=rmsd(Pg[isd], X[isd]),
                         rmsd_prot_after_dna_sup=rmsd(Pd[isp], X[isp]),
                         **{f"azimuth_err_{pc}": ang[pc] for pc in cfg["prot"]},
                         **{f"placement_{pc}": rmsd(Pd[lab == pc], X[lab == pc]) for pc in cfg["prot"]},
                         _R=R, _t=t, _map={**mp, **md})
                if best is None or r["rmsd_all"] < best["rmsd_all"]: best = r
        c = json.load(open(f"{pred_dir}/confidence_{tgt}_model_{k}.json"))
        ids = [ch.name for ch in pm]; L = [len(ch) for ch in pm]; off = np.cumsum([0] + L)
        tok = {ch: np.arange(off[i], off[i + 1]) for i, ch in enumerate(ids)}
        pae = np.load(f"{pred_dir}/pae_{tgt}_model_{k}.npz")["pae"]
        pi = {f"{ids[i]}{ids[j]}": c["pair_chains_iptm"][str(i)][str(j)] for i in range(len(ids)) for j in range(len(ids)) if i < j}
        dna_tok = np.concatenate([tok[d] for d in cfg["dna"]])
        best.update(iptm=c["iptm"], confidence=c["confidence_score"], iptm_prot_prot=pi["AB"],
                    iptm_prot_dna=max(pi[f"{p}{d}"] for p in cfg["prot"] for d in cfg["dna"]),
                    ipsae_prot_prot=ipsae(pae, tok["A"], tok["B"]),
                    ipsae_prot_dna_A=ipsae(pae, tok["A"], dna_tok), ipsae_prot_dna_B=ipsae(pae, tok["B"], dna_tok))
        rows.append(best)
        if k == 0:  # superposed top-ranked model + the matched crystal subset, for rendering
            st = gemmi.read_structure(mf)
            for ch in st[0]:
                for res in ch:
                    for at in res:
                        v = np.array(at.pos.tolist()) @ best["_R"] + best["_t"]; at.pos = gemmi.Position(*v)
            st.write_pdb(f"{D}/{tgt}_top_superposed.pdb")
            xc = gemmi.read_structure(f"{D}/pdb/{cfg['pdb']}.cif"); keep = set(best["_map"].values())
            for name in [ch.name for ch in xc[0]]:  # names first: removing while holding Chain refs segfaults
                if name not in keep: xc[0].remove_chain(name)
            xc.remove_ligands_and_waters(); xc.write_pdb(f"{D}/{tgt}_crystal_subset.pdb")
            print(tgt, "crystal chains used:", best["_map"])

import pandas as pd
S = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
S.to_csv(f"{D}/scores.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250)
print(S.round(2).to_string(index=False))
