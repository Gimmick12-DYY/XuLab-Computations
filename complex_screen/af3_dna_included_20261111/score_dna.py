# Scores the DNA-included AF3 test exactly as declared (DECLARATION.md). Per seed, the top-ranked sample (ranking_scores.csv). (i) DNA engaged: the
# flagged TF's DBD (UniProt ranges as in rescore_5seed/dbd_face.tsv; model sequences identical to UniProt canonical) has >= 5 residues with pLDDT >= 70
# and a heavy atom within 5 A of a DNA heavy atom (chains C, D), in >= 3/5 seeds. (ii) Partner retained: protein-protein chain_pair_iptm[A][B] of the
# top sample >= .4 and A-B ipSAE (score_one.py formula, PAE cutoff 10 A, max over the seed's samples, as in the 5-seed scoring) >= .3, in >= 3/5 seeds.
# SURVIVES = (i) and (ii); DISPLACED = (i) and not (ii); UNINFORMATIVE = not (i), or not built. Validity: RBBP4-ZIC2 must be SURVIVES, else no pair is
# reclassified. Writes per_seed.tsv and results.tsv. usage: score_dna.py
import glob, json, os, re
import numpy as np, pandas as pd
G = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_dna_included_20261111"; S5 = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"
I = pd.read_csv(f"{G}/inputs.tsv", sep="\t"); DB = pd.read_csv(f"{S5}/rescore_5seed/dbd_face.tsv", sep="\t")
def ipsae(pae, la, lb, cut=10.0):
    best = 0.0
    for (r0, r1), (c0, c1) in [((0, la), (la, la + lb)), ((la, la + lb), (0, la))]:
        sub = pae[r0:r1, c0:c1]; ok = sub < cut; k = ok.sum(1); d0 = 1.24 * np.cbrt(np.maximum(k, 27) - 15) - 1.8
        s = np.where(ok, 1 / (1 + (sub / d0[:, None]) ** 2), 0).sum(1) / np.maximum(k, 1); best = max(best, float(s.max()))
    return best
def atoms(cif):
    cols, rows, inl = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inl = True; continue
        if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split())
        elif inl and rows and not l.strip(): break
    D = pd.DataFrame(rows, columns=cols[:len(rows[0])]); D = D[D.type_symbol != "H"]
    D["xyz"] = list(D[["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values); D["b"] = D.B_iso_or_equiv.astype(float); D["res"] = D.label_seq_id.astype(int); return D
PS, RS = [], []
for r in I.itertuples():
    if r.status != "built": RS.append(dict(pair=r.pair, tf=r.tf, outcome="UNINFORMATIVE", reason=r.status)); continue
    d = f"{G}/out/{r.pair.lower()}_dna"
    if not os.path.exists(f"{d}/ranking_scores.csv"): RS.append(dict(pair=r.pair, tf=r.tf, outcome="NOT_RUN", reason="no AF3 output")); continue
    rng = DB[(DB.pair == r.pair) & (DB.tf == r.tf)].dbd_range.iloc[0]; dbd = set()
    for a, b in re.findall(r":(\d+)-(\d+)", rng): dbd |= set(range(int(a), int(b) + 1))
    tc = "A" if r.pair.split("_")[0] == r.tf else "B"; rk = pd.read_csv(f"{d}/ranking_scores.csv")
    for sd, g in rk.groupby("seed"):
        top = int(g.loc[g.ranking_score.idxmax(), "sample"]); sdir = f"{d}/seed-{sd}_sample-{top}"
        A = atoms(f"{sdir}/model.cif"); T = A[A.label_asym_id == tc]; N = A[A.label_asym_id.isin(["C", "D"])]
        X1 = np.stack(T.xyz.values); X2 = np.stack(N.xyz.values); close = (((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1) <= 25).any(1)
        pl = T.groupby("res").b.mean(); eng = sorted({int(x) for x in T.res.values[close] if x in dbd and pl[x] >= 70})
        sc = json.load(open(f"{sdir}/summary_confidences.json")); ipab = sc["chain_pair_iptm"][0][1]; ips = 0.0
        for f in glob.glob(f"{d}/seed-{sd}_sample-*/confidences.json"):
            c = json.load(open(f)); ch = np.array(c["token_chain_ids"]); ips = max(ips, ipsae(np.array(c["pae"]), int((ch == "A").sum()), int((ch == "B").sum())))
        PS.append(dict(pair=r.pair, tf=r.tf, seed=int(sd), sample=top, n_dbd_dna=len(eng), engaged=len(eng) >= 5, iptm_AB=round(ipab, 3), ipsae_AB=round(ips, 3),
                       retained=(ipab >= .4 and ips >= .3), iptm_TF_DNA=round(max(sc["chain_pair_iptm"][0 if tc == "A" else 1][2:]), 3), dbd_dna_res=",".join(map(str, eng))))
    q = pd.DataFrame([x for x in PS if x["pair"] == r.pair]); e, t = int(q.engaged.sum()), int(q.retained.sum())
    RS.append(dict(pair=r.pair, tf=r.tf, n_seeds=len(q), seeds_engaged=e, seeds_retained=t, outcome="SURVIVES" if e >= 3 and t >= 3 else ("DISPLACED" if e >= 3 else "UNINFORMATIVE"),
                   reason="", median_iptm_AB=q.iptm_AB.median(), median_ipsae_AB=q.ipsae_AB.median(), median_n_dbd_dna=q.n_dbd_dna.median()))
R = pd.DataFrame(RS); ctl = R[R.pair == "RBBP4_ZIC2"].outcome.iloc[0] if (R.pair == "RBBP4_ZIC2").any() else "absent"
R["test_valid"] = ctl == "SURVIVES"; R["DESCRIPTIVE"] = "footnote; ranking stays correlation + AF3 5-seed"
pd.DataFrame(PS).to_csv(f"{G}/per_seed.tsv", sep="\t", index=False); R.to_csv(f"{G}/results.tsv", sep="\t", index=False)
print(R.to_string(index=False)); print("positive control RBBP4-ZIC2:", ctl, "-> test valid" if ctl == "SURVIVES" else "-> test INVALID: no pair is reclassified")
