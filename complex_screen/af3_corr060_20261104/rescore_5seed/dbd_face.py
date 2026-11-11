# DBD-face flag (rescore_5seed/DBD_FACE_RULE.md, declared before computing). For each AF3 5-seed strong pair (>= 3/5 strong seeds) and each DNA-binding TF
# in it: confident interface residues (heavy atom within 5 A of the partner, residue pLDDT >= 70, union over strong seeds, top-ranked sample per seed)
# and the share inside its UniProt-annotated DNA-binding domain(s). DBD features (UniProtKB reviewed human canonical, rest.uniprot.org, cached in
# uniprot_dbd_cache.json): type "DNA binding"; type "Domain" named bZIP/bHLH/Homeobox/HMG/ETS/Fork-head/T-box/MADS/NFYA-HAP2/Nuclear receptor;
# type "Zinc finger" C2H2 or NR C4. Mapped to the AF3 model sequence by exact identity, else by difflib matching blocks. Partners with no DBD
# feature (e.g. RBBP4/7, XPA, MTF2, ZUP1) are reported as non-DBD and never flagged. Writes dbd_face.tsv. DESCRIPTIVE. usage: dbd_face.py
import re, os, json, difflib, urllib.request, urllib.parse
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/af3_corr060_20261104"; OUT = f"{O}/rescore_5seed"; CACHE = f"{OUT}/uniprot_dbd_cache.json"
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
GENE = {"ZUFSP": "ZUP1", "HP1": "CBX3"}
DOM = re.compile(r"bZIP|bHLH|Homeobox|HMG|ETS|Fork-head|T-box|MADS|NFYA|HAP2|Nuclear receptor", re.I)
def atoms(cif):
    cols, rows, inloop = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inloop = True; continue
        if inloop and (l.startswith("ATOM") or l.startswith("HETATM")): rows.append(l.split())
        elif inloop and rows and not l.strip(): break
    D = pd.DataFrame(rows, columns=cols[:len(rows[0])]); D["x"] = D.Cartn_x.astype(float); D["y"] = D.Cartn_y.astype(float); D["z"] = D.Cartn_z.astype(float); D["b"] = D.B_iso_or_equiv.astype(float); D["res"] = D.label_seq_id.astype(int)
    return D[D.type_symbol != "H"]
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def uniprot(g):
    if g not in cache:
        q = urllib.parse.quote(f"gene_exact:{g} AND organism_id:9606 AND reviewed:true")
        r = json.load(urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/search?query={q}&fields=accession,gene_primary,sequence,ft_dna_bind,ft_domain,ft_zn_fing&format=json", timeout=60))["results"]
        r = [x for x in r if x.get("genes") and x["genes"][0].get("geneName", {}).get("value") == g] or r
        x = r[0]; feats = [dict(type=f["type"], desc=f.get("description", ""), start=f["location"]["start"]["value"], end=f["location"]["end"]["value"]) for f in x.get("features", [])]
        cache[g] = dict(acc=x["primaryAccession"], seq=x["sequence"]["value"], feats=feats); json.dump(cache, open(CACHE, "w"), indent=1)
    return cache[g]
def dbd(g):
    u = uniprot(g); F = [f for f in u["feats"] if f["type"] == "DNA binding" or (f["type"] == "Domain" and DOM.search(f["desc"])) or (f["type"] == "Zinc finger" and re.search(r"C2H2|NR C4", f["desc"]))]
    return u, F
def mapper(useq, mseq):
    if useq == mseq: return lambda i: i, "identical"
    m = {}
    for a, b, n in difflib.SequenceMatcher(None, useq, mseq, autojunk=False).get_matching_blocks():
        for k in range(n): m[a + k + 1] = b + k + 1
    return (lambda i: m.get(i)), f"aligned ({len(m)}/{len(useq)} residues matched)"
def seeds_strong(p):
    x = open(f"{O}/scores_s5/{p}.tsv").read().rstrip("\n").split("\t")
    return int(x[5]), [int(s) for s, t in re.findall(r"(\d+):[\d.]+/[\d.]+ (\w+)", x[-1]) if t == "strong"]
P = [l.strip() for l in open(f"{O}/seeds5/all.txt") if l.strip()]
strong = [p for p in P if os.path.exists(f"{O}/scores_s5/{p}.tsv") and os.path.getsize(f"{O}/scores_s5/{p}.tsv") and seeds_strong(p)[0] >= 3]
rows = []
for p in strong:
    a, b = p.split("_"); d = f"{O}/out_s5/{p.lower()}"; rs = pd.read_csv(f"{d}/ranking_scores.csv"); best = rs.loc[rs.groupby("seed").ranking_score.idxmax()].set_index("seed"); St = seeds_strong(p)[1]
    U = {a: set(), b: set()}; SEQ = {}
    for s in St:
        A = atoms(f"{d}/seed-{s}_sample-{int(best.loc[s, 'sample'])}/model.cif"); ch = list(dict.fromkeys(A.label_asym_id))
        for tf, c, oc in ((a, ch[0], ch[1]), (b, ch[1], ch[0])):
            H, Pp = A[A.label_asym_id == c], A[A.label_asym_id == oc]
            if tf not in SEQ: sq = H.drop_duplicates("res").sort_values("res"); SEQ[tf] = "".join(AA.get(x, "X") for x in sq.label_comp_id)
            X1 = H[["x", "y", "z"]].values; X2 = Pp[["x", "y", "z"]].values; close = np.zeros(len(X1), bool)
            for i in range(0, len(X1), 2000): close[i:i + 2000] = (((X1[i:i + 2000, None, :] - X2[None, :, :]) ** 2).sum(-1) <= 25).any(1)
            pl = H.groupby("res").b.mean(); U[tf] |= {int(r) for r in H.res.values[close] if pl[r] >= 70}
    for tf in (a, b):
        g = GENE.get(tf, tf); u, F = dbd(g); f_map, how = mapper(u["seq"], SEQ[tf]); R = set(); rng = []
        for f in F:
            s_, e_ = f_map(f["start"]), f_map(f["end"])
            if s_ and e_: R |= set(range(s_, e_ + 1)); rng.append(f"{f['desc'] or f['type']}:{s_}-{e_}")
        n = len(U[tf]); sh = len(U[tf] & R) / n if n and R else np.nan
        rows.append(dict(pair=p, tf=tf, dbd_source=f"UniProt {u['acc']} ({how})" if F else f"UniProt {u['acc']}: no DNA-binding feature", dbd_range="; ".join(rng) if rng else "none",
                         n_interface=n, share_in_dbd=round(sh, 3) if R else np.nan, flag=("DBD-face" if R and n and sh >= .5 else ("" if R else "non-DBD partner")),
                         n_strong_seeds=len(St), partner=b if tf == a else a))
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/dbd_face.tsv", sep="\t", index=False)
print(len(strong), "strong pairs;", T[T.flag == "DBD-face"].pair.nunique(), "pairs flagged")
