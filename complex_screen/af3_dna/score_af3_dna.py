# Scores for the AF3 + DNA runs. Per seed: the best-ranked sample. ipTM and ipSAE as in the screen; tier strong = ipSAE >= 0.3 and ipTM >= 0.4, weak = ipSAE >= 0.1.
# Base contacts: distinct (basic-region residue, DNA base) pairs with any heavy atoms within 4 A, base atoms only; basic region = first 13 residues of each construct. Major-groove edge atoms: A N6/N7, G O6/N7, C N4, T O4/C7; minor: A N3, G N2/N3, C O2, T O2.
import csv, glob, json, os, re
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_dna"; MAJ = {"DA": {"N6", "N7"}, "DG": {"O6", "N7"}, "DC": {"N4"}, "DT": {"O4", "C7"}}; MIN = {"DA": {"N3"}, "DG": {"N2", "N3"}, "DC": {"O2"}, "DT": {"O2"}}; BASE = {"N1", "C2", "N3", "C4", "C5", "C6", "C7", "C8", "N9", "N2", "N4", "N6", "N7", "O2", "O4", "O6"}
def ipsae(pae, ra, rb, cut=10.0):
    best = 0.0
    for x, y in ((ra, rb), (rb, ra)):
        sub = pae[np.ix_(x, y)]; ok = sub < cut; n = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8; s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(n, 1); best = max(best, float(s.max()))
    return best
def atoms(f):
    cols, rows, inl = [], [], False
    for l in open(f):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".")[1]); inl = True; continue
        if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split())
        elif inl and rows and l.startswith("#"): break
    d = pd.DataFrame(rows, columns=cols); d = d[d.type_symbol != "H"]; return pd.DataFrame(dict(ch=d.label_asym_id.values, res=d.label_seq_id.astype(int).values, comp=d.label_comp_id.values, atom=d.label_atom_id.str.strip('"').values, x=d.Cartn_x.astype(float).values, y=d.Cartn_y.astype(float).values, z=d.Cartn_z.astype(float).values))
def basic(A, ch):
    P = A[(A.ch == ch) & (A.res <= 13)]; D = A[A.ch.isin(["C", "D"]) & A.atom.isin(BASE)]; dd = np.linalg.norm(P[["x", "y", "z"]].values[:, None] - D[["x", "y", "z"]].values[None], axis=2); i, j = np.nonzero(dd <= 4.0); pairs, maj, mino = set(), set(), set()
    for a, b in zip(i, j):
        r = P.res.values[a]; c = D.ch.values[b]; s = D.res.values[b]; nm = D.atom.values[b]; cp = D.comp.values[b]; pairs.add((r, c, s))
        if nm in MAJ[cp]: maj.add((r, c, s))
        if nm in MIN[cp]: mino.add((r, c, s))
    # Groove from distance to the nearest major vs minor edge atom over all basic-region heavy atoms
    return len(pairs), len(maj), len(mino), sorted({(c, int(s)) for _, c, s in maj})
rows = []
for n in [l.strip() for l in open(f"{O}/runs.txt") if l.strip()]:
    d = f"{O}/out/{n.lower()}"; R = pd.read_csv(f"{d}/ranking_scores.csv")
    for sd, g in R.groupby("seed"):
        b = g.sort_values("ranking_score", ascending=False).iloc[0]; x = f"{d}/seed-{int(sd)}_sample-{int(b['sample'])}"; S = json.load(open(f"{x}/summary_confidences.json")); c = json.load(open(f"{x}/confidences.json")); ch = np.array(c["token_chain_ids"]); pae = np.array(c["pae"]); ix = {k: np.flatnonzero(ch == k) for k in "ABCD"}; ip = S["chain_pair_iptm"]; A = atoms(f"{x}/model.cif"); na = basic(A, "A"); nb = basic(A, "B"); pp = ipsae(pae, ix["A"], ix["B"]); pi = ip[0][1]
        rows.append(dict(run=n, seed=int(sd), sample=int(b["sample"]), ranking_score=round(float(b["ranking_score"]), 3), iptm_overall=S["iptm"], pp_iptm=pi, pp_ipsae=round(pp, 3), tier="strong" if pp >= .3 and pi >= .4 else ("weak" if pp >= .1 else "none"), A_C_iptm=ip[0][2], A_D_iptm=ip[0][3], B_C_iptm=ip[1][2], B_D_iptm=ip[1][3], dna_iptm=ip[2][3], A_base_contacts=na[0], A_major=na[1], A_minor=na[2], B_base_contacts=nb[0], B_major=nb[1], B_minor=nb[2], A_major_groove=int(na[1] > na[2]), B_major_groove=int(nb[1] > nb[2]), has_clash=S["has_clash"]))
T = pd.DataFrame(rows); T.to_csv(f"{O}/scores_per_seed.tsv", sep="\t", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); print(T.drop(columns=["sample", "iptm_overall", "dna_iptm", "has_clash"]).to_string())
num = ["ranking_score", "pp_iptm", "pp_ipsae", "A_C_iptm", "A_D_iptm", "B_C_iptm", "B_D_iptm", "A_base_contacts", "A_major", "A_minor", "B_base_contacts", "B_major", "B_minor"]; o = []
for n, g in T.groupby("run", sort=False):
    r = dict(run=n, tiers="/".join(f"{(g.tier == k).sum()}{k[0]}" for k in ("strong", "weak", "none")), A_major_groove=int(g.A_major_groove.sum()), B_major_groove=int(g.B_major_groove.sum()))
    for k in num: r[k] = f"{g[k].median():.2f} ({g[k].min():.2f}-{g[k].max():.2f})"
    o.append(r)
Sm = pd.DataFrame(o); Sm.to_csv(f"{O}/scores_summary.tsv", sep="\t", index=False); print(Sm.T.to_string()); print("clash", int(T.has_clash.sum()))
