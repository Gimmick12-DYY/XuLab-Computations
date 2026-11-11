# Read-only docking-site check for HOX partners on the AF3 5-seed outputs (logic as ddit3_check.py). For each pair and each STRONG seed (per-seed
# ipSAE >= .3 and ipTM >= .4, from scores_s5), the top-ranked sample of that seed: HOX residues with any heavy atom within 5 A of a partner heavy atom and
# residue pLDDT (mean atom B) >= 70. Homeodomain = 60 residues located from the model sequence by the conserved helix-3 motif WF.NRR (homeodomain
# positions 48-53), i.e. HD = [m - 47, m + 12] with m the W position; not a UniProt lookup. Also pairwise Jaccard of the per-pair union interface sets
# for the same HOX. Writes rescore_5seed/hox_check.tsv. DESCRIPTIVE. usage: hox_check.py
import re, os
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; OUT = f"{O}/rescore_5seed"
PAIRS = {"HOXB9": ["HOXB9_RBBP7", "HOXB9_XPA", "HOXB9_RBBP4"], "HOXA10": ["HOXA10_XPA", "HOXA10_RBBP4"], "HOXC4": ["HOXC4_ZUP1"]}
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
def atoms(cif):
    cols, rows, inloop = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inloop = True; continue
        if inloop and (l.startswith("ATOM") or l.startswith("HETATM")): rows.append(l.split())
        elif inloop and rows and not l.strip(): break
    D = pd.DataFrame(rows, columns=cols[:len(rows[0])]); D["x"] = D.Cartn_x.astype(float); D["y"] = D.Cartn_y.astype(float); D["z"] = D.Cartn_z.astype(float); D["b"] = D.B_iso_or_equiv.astype(float); D["res"] = D.label_seq_id.astype(int)
    return D[D.type_symbol != "H"]
def seeds_strong(p):
    x = open(f"{O}/scores_s5/{p}.tsv").read().rstrip("\n").split("\t")[-1]
    return [int(s) for s, t in re.findall(r"(\d+):[\d.]+/[\d.]+ (\w+)", x) if t == "strong"]
rows, U, HD = [], {}, {}
for hox, ps in PAIRS.items():
    for p in ps:
        d = f"{O}/out_s5/{p.lower()}"; rs = pd.read_csv(f"{d}/ranking_scores.csv"); best = rs.loc[rs.groupby("seed").ranking_score.idxmax()].set_index("seed"); St = seeds_strong(p); u = set()
        for s in St:
            A = atoms(f"{d}/seed-{s}_sample-{int(best.loc[s, 'sample'])}/model.cif"); ch = list(dict.fromkeys(A.label_asym_id)); hc = ch[0] if p.split("_")[0] == hox else ch[1]; pc = [c for c in ch if c != hc][0]
            H, Pp = A[A.label_asym_id == hc], A[A.label_asym_id == pc]
            if hox not in HD:
                sq = H.drop_duplicates("res").sort_values("res"); seq = "".join(AA.get(c, "X") for c in sq.label_comp_id); m = re.search(r"WF.NRR", seq)
                HD[hox] = (m.start() + 1 - 47, m.start() + 1 + 12, len(seq), seq[m.start() - 47:m.start() + 13]) if m else None
            X1 = H[["x", "y", "z"]].values; X2 = Pp[["x", "y", "z"]].values; close = np.zeros(len(X1), bool)
            for i in range(0, len(X1), 2000): close[i:i + 2000] = (((X1[i:i + 2000, None, :] - X2[None, :, :]) ** 2).sum(-1) <= 25).any(1)
            pl = H.groupby("res").b.mean(); res = sorted({int(r) for r in H.res.values[close] if pl[r] >= 70}); u |= set(res)
            h0, h1 = HD[hox][:2]; nin = sum(h0 <= r <= h1 for r in res)
            rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="seed", hox=hox, pair=p, seed=s, sample=int(best.loc[s, "sample"]), ranking_score=round(best.loc[s, "ranking_score"], 4), n_iface=len(res), n_in_hd=nin,
                             frac_in_hd=round(nin / len(res), 3) if res else np.nan, n_nterm=sum(r < h0 for r in res), n_cterm=sum(r > h1 for r in res), iface_res=",".join(map(str, res))))
        U[p] = u; h0, h1 = HD[hox][:2]; nin = sum(h0 <= r <= h1 for r in u)
        rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="pair_union", hox=hox, pair=p, n_strong_seeds=len(St), strong_seeds=",".join(map(str, St)), n_iface=len(u), n_in_hd=nin, frac_in_hd=round(nin / len(u), 3) if u else np.nan,
                         n_nterm=sum(r < h0 for r in u), n_cterm=sum(r > h1 for r in u), iface_res=",".join(map(str, sorted(u)))))
    h0, h1, n, s = HD[hox]; rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="homeodomain", hox=hox, hd_start=h0, hd_end=h1, seq_len=n, hd_seq=s, hd_source="sequence motif WF.NRR (HD positions 48-53) in the AF3 model sequence"))
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = U[ps[i]], U[ps[j]]; rows.append(dict(DESCRIPTIVE="DESCRIPTIVE", item="jaccard", hox=hox, pair=f"{ps[i]} vs {ps[j]}", jaccard=round(len(a & b) / max(len(a | b), 1), 3), n_shared=len(a & b), shared_res=",".join(map(str, sorted(a & b)))))
pd.DataFrame(rows).to_csv(f"{OUT}/hox_check.tsv", sep="\t", index=False)
