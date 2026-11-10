# Read-only DDIT3 check on the partial AF3 5-seed results. (1) Pair counts by TF among the 97 5-seed pairs and the scored subset, and the expected number
# of DDIT3 pairs among the strong calls under a random draw from the scored set (hypergeometric mean). (2) For each strong DDIT3 pair: per seed, the
# top-ranked sample (ranking_scores.csv); DDIT3 residues with any heavy atom within 5 A of a partner heavy atom and residue pLDDT (mean atom B) >= 70;
# fraction in the bZIP (basic region + leucine zipper, located from the DDIT3 sequence in the model); pairwise Jaccard of the union interface sets across
# partners. Writes rescore_5seed/ddit3_check.tsv. usage: ddit3_check.py
import csv, os
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; OUT = f"{O}/rescore_5seed"
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
def atoms(cif):
    cols, rows, inloop = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inloop = True; continue
        if inloop and (l.startswith("ATOM") or l.startswith("HETATM")): rows.append(l.split())
        elif inloop and rows and not l.strip(): break
    D = pd.DataFrame(rows, columns=cols[:len(rows[0])]); D["x"] = D.Cartn_x.astype(float); D["y"] = D.Cartn_y.astype(float); D["z"] = D.Cartn_z.astype(float); D["b"] = D.B_iso_or_equiv.astype(float); D["res"] = D.label_seq_id.astype(int)
    return D[D.type_symbol != "H"]
P = [l.strip() for l in open(f"{O}/seeds5/all.txt") if l.strip()]; scored = [p for p in P if os.path.exists(f"{O}/scores_s5/{p}.tsv") and os.path.getsize(f"{O}/scores_s5/{p}.tsv")]
def tier(p):
    x = open(f"{O}/scores_s5/{p}.tsv").read().split("\t"); return "strong" if int(x[5]) >= 3 else ("weak" if float(x[2]) >= .1 else "none")
T = {p: tier(p) for p in scored}; strong = [p for p in scored if T[p] == "strong"]; rows = []
for g in ["DDIT3", "MTF2", "RBBP4", "RBBP7", "RBBP4|RBBP7"]:
    has = lambda p: any(x in p.split("_") for x in g.split("|"))
    n97, ns, nst = sum(map(has, P)), sum(map(has, scored)), sum(map(has, strong)); exp = len(strong) * ns / len(scored) if scored else np.nan
    rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="counts", tf=g, n_pairs_97=n97, n_scored=ns, n_strong=nst, n_strong_total=len(strong), n_scored_total=len(scored), expected_strong_random=round(exp, 3)))
IF = {}
for p in [q for q in strong if "DDIT3" in q.split("_")]:
    d = f"{O}/out_s5/{p.lower()}"; rs = pd.read_csv(f"{d}/ranking_scores.csv"); best = rs.loc[rs.groupby("seed").ranking_score.idxmax()]; U = set(); seq = None
    for r in best.itertuples():
        A = atoms(f"{d}/seed-{r.seed}_sample-{r.sample}/model.cif"); ch = A.label_asym_id.unique(); dd = ch[0] if p.split("_")[0] == "DDIT3" else ch[1]; pa = [c for c in ch if c != dd][0]
        Dd, Pa = A[A.label_asym_id == dd], A[A.label_asym_id == pa]
        if seq is None: s = Dd.drop_duplicates("res").sort_values("res"); seq = "".join(AA.get(x, "X") for x in s.label_comp_id)
        X1 = Dd[["x", "y", "z"]].values; X2 = Pa[["x", "y", "z"]].values; close = np.zeros(len(X1), bool)
        for i in range(0, len(X1), 2000): close[i:i + 2000] = (((X1[i:i + 2000, None, :] - X2[None, :, :]) ** 2).sum(-1) <= 25).any(1)
        pl = Dd.groupby("res").b.mean(); res = sorted({int(r_) for r_ in Dd.res.values[close] if pl[r_] >= 70}); U |= set(res)
        rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="interface_seed", pair=p, seed=int(r.seed), sample=int(r.sample), ranking_score=r.ranking_score, n_iface_res=len(res), iface_res=",".join(map(str, res))))
    IF[p] = (U, seq)
# bZIP from sequence: basic region = first window >= 60% R/K starting at or after ~90; zipper = residues to ~160 with L at heptad spacing
seq = next(iter(IF.values()))[1] if IF else ""
if seq:
    bas = [i + 1 for i in range(len(seq) - 20) if sum(c in "RK" for c in seq[i:i + 20]) >= 8]; b0 = min([b for b in bas if b >= 80], default=98)
    Ls = [i + 1 for i, c in enumerate(seq) if c == "L" and i + 1 > b0 + 20]; z1 = max([l for l in Ls if (l - Ls[0]) % 7 == 0], default=160) if Ls else 160
    BZ = set(range(b0, z1 + 1)); rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="bzip_from_sequence", tf="DDIT3", seq_len=len(seq), bzip_start=b0, bzip_end=z1, bzip_seq=seq[b0 - 1:z1]))
    for p, (U, _) in IF.items(): rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="interface_union", pair=p, n_iface_res=len(U), n_in_bzip=len(U & BZ), frac_in_bzip=round(len(U & BZ) / max(len(U), 1), 3), iface_res=",".join(map(str, sorted(U)))))
    ks = list(IF)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = IF[ks[i]][0], IF[ks[j]][0]; rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="jaccard", pair=f"{ks[i]} vs {ks[j]}", jaccard=round(len(a & b) / max(len(a | b), 1), 3), n_shared=len(a & b)))
pd.DataFrame(rows).to_csv(f"{OUT}/ddit3_check.tsv", sep="\t", index=False)
