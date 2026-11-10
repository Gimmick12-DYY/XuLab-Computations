# Interface audit (see README.md, declared first). Confident interface residues (CIR) per model; per-seed top models for multi-seed runs.
# usage: audit.py [n_workers]   writes audit.tsv, summary.tsv, moves.tsv in this folder.
import os, sys, glob, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
X = "/work/users/d/y/dyy12/XuLab/complex_screen"; S = f"{X}/af3_corr060_20261104"; OUT = f"{X}/interface_audit_20261110"
def atoms(cif):
    cols, rows, inl = [], [], False
    for l in open(cif):
        if l.startswith("_atom_site."): cols.append(l.strip().split(".", 1)[1]); inl = True; continue
        if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split()[:len(cols)])
        elif inl and rows and (l.startswith("#") or l.startswith("_") or not l.strip()): break
    D = pd.DataFrame(rows, columns=cols); D = D[(D.group_PDB == "ATOM") & (D.type_symbol != "H")]
    D = D.assign(xyz=None); D["b"] = D.B_iso_or_equiv.astype(float); D["res"] = D.label_seq_id.astype(int)
    return D
def cir(cif):
    try: A = atoms(cif)
    except Exception as e: return dict(err=f"parse:{e}")
    ch = list(dict.fromkeys(A.label_asym_id))
    if len(ch) != 2: return dict(err=f"n_chains={len(ch)}")
    out = {}
    P = {c: A[A.label_asym_id == c] for c in ch}; T = {c: cKDTree(P[c][["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values) for c in ch}
    for i, c in enumerate(ch):
        o = ch[1 - i]; d, _ = T[o].query(P[c][["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values, k=1, distance_upper_bound=5.0)
        near = set(P[c].res.values[np.isfinite(d)]); pl = P[c].groupby("res").b.mean()
        out["ab"[i]] = sorted(r for r in near if pl[r] >= 70); out["n_iface_" + "ab"[i]] = len(near)
    return dict(CIR_a=len(out["a"]), CIR_b=len(out["b"]), CIR=len(out["a"]) + len(out["b"]), iface_any=out["n_iface_a"] + out["n_iface_b"], err="")
def seed_models(d):
    """per-seed top-ranked sample cif paths, or [top model] if no seed dirs"""
    rs = f"{d}/ranking_scores.csv"
    if os.path.exists(rs):
        r = pd.read_csv(rs); b = r.loc[r.groupby("seed").ranking_score.idxmax()]
        p = [(int(x.seed), f"{d}/seed-{int(x.seed)}_sample-{int(x.sample)}/model.cif") for x in b.itertuples()]
        p = [q for q in p if os.path.exists(q[1])]
        if p: return p
    t = glob.glob(f"{d}/*_model.cif"); return [(0, t[0])] if t else []
def job(r):
    r = dict(r); ms = seed_models(r["dir"]) if r["tool"] == "AF3" else ([(0, r["dir"])] if os.path.exists(r["dir"]) else [])
    if r["tool"] == "AF3" and r["source"] == "af3_s1_sweep": ms = [m for m in seed_models(r["dir"]) if True]; t = glob.glob(f"{r['dir']}/*_model.cif"); ms = [(0, t[0])] if t else []
    if not ms: r.update(err="no_model"); return r
    res = [(s, cir(p)) for s, p in ms]; ok = [(s, c) for s, c in res if not c.get("err")]
    if not ok: r.update(err=res[0][1]["err"]); return r
    v = np.array([c["CIR"] for _, c in ok])
    r.update(n_models_used=len(ok), CIR_per_seed=";".join(f"{s}:{c['CIR_a']}+{c['CIR_b']}" for s, c in ok), CIR=float(np.median(v)),
             CIR_a=float(np.median([c["CIR_a"] for _, c in ok])), CIR_b=float(np.median([c["CIR_b"] for _, c in ok])),
             iface_any=float(np.median([c["iface_any"] for _, c in ok])), err="")
    return r
lo = lambda p: p.replace("__", "_").lower()
def rows():
    R = []
    for f in glob.glob(f"{S}/scores_s1/*.tsv"):
        x = open(f).read().split("\t"); R.append(dict(tool="AF3", source="af3_s1_sweep", pair=x[0], iptm=float(x[1]), ipsae=float(x[2]), tier=x[4], dir=f"{S}/out_s1/{lo(x[0])}"))
    for f in glob.glob(f"{S}/scores_s5/*.tsv"):
        if not os.path.getsize(f): continue
        x = open(f).read().split("\t"); t = "strong" if int(x[5]) >= 3 else ("weak" if float(x[2]) >= .1 else "none")
        R.append(dict(tool="AF3", source="af3_s5", pair=x[0].replace("_", "__", 1) if "__" not in x[0] else x[0], iptm=float(x[1]), ipsae=float(x[2]), tier=t, n_seeds_strong=int(x[5]), dir=f"{S}/out_s5/{lo(x[0])}"))
    for f, od in [("af3_local/af3_scores.tsv", "af3_local/out"), ("af3_novel_20261103/af3_scores.tsv", "af3_novel_20261103/out"),
                  ("af3_novel_20261103/af3_scores_weak.tsv", "af3_novel_20261103/out"), ("af3_novel_20261103/af3_scores_srcap.tsv", "af3_novel_20261103/out")]:
        for r in pd.read_csv(f"{X}/{f}", sep="\t").itertuples():
            R.append(dict(tool="AF3", source=f.replace("/", ":"), pair=r.pair, iptm=r.iptm_top, ipsae=r.ipsae_max, tier=r.tier, dir=f"{X}/{od}/{lo(r.pair)}"))
    R.append(dict(tool="AF3", source="af3_control", pair="EZH2__SUZ12", iptm=np.nan, ipsae=np.nan, tier="control", dir=f"{S}/out_control/ezh2_suz12"))
    B = pd.read_csv(f"{X}/screen/scores.tsv", sep="\t", keep_default_na=False); B = B[(B["set"] != "homo") & B.hit.isin(["strong", "weak"])]
    for r in B.itertuples(): R.append(dict(tool="Boltz", source="boltz_screen", pair=r.pair_id, iptm=float(r.iptm_top), ipsae=float(r.ipsae_max), tier=r.hit, dir=r.cif))
    D = pd.DataFrame(R); D["pair"] = D.pair.str.replace("_", "__").str.replace("____", "__")
    rng = np.random.default_rng(20261110); D["comparator_sample"] = False
    for t in ("weak", "none"):
        ix = D.index[(D.source == "af3_s1_sweep") & (D.tier == t)]; D.loc[rng.choice(ix, min(100, len(ix)), replace=False), "comparator_sample"] = True
    return D
POS = {"EZH2__SUZ12", "MTF2__RBBP4", "RBBP4__ZIC2"}; NEG = {"RBBP7__SRCAP"}
if __name__ == "__main__":
    D = rows(); print(len(D), D.groupby(["tool", "source", "tier"]).size().to_string(), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p: A = pd.DataFrame(p.map(job, D.to_dict("records"), chunksize=8))
    A["CIR_minchain"] = A[["CIR_a", "CIR_b"]].min(axis=1)  # DESCRIPTIVE only, not used by the declared floor rule
    A["control"] = np.where(A.pair.isin(POS), "positive", np.where(A.pair.isin(NEG), "negative", ""))
    pc = A[(A.control == "positive") & A.CIR.notna()].CIR; p10 = float(np.percentile(pc, 10)) if len(pc) else np.nan
    floor = 10.0 if len(pc) < 5 else min(p10, 10.0)
    A["audit_label"] = np.where((A.tier == "strong") & (A.CIR < floor), "strong_noCIR", A.tier)
    A.drop(columns=["dir"]).assign(DESCRIPTIVE="DESCRIPTIVE").to_csv(f"{OUT}/audit.tsv", sep="\t", index=False)
    S_ = [dict(item="floor", n_pos_obs=len(pc), pos_values=";".join(f"{x:g}" for x in pc), p10=p10, floor=floor)]
    for t, g in A[A.CIR.notna()].groupby("tool"):
        for tr in ("strong", "weak", "none"):
            v = g[g.tier == tr].CIR
            w = g[g.tier == tr].CIR_minchain
            if len(w): S_.append(dict(item="deciles_minchain_DESCRIPTIVE", tool=t, tier=tr, n=len(w), **{f"q{q}": float(np.percentile(w, q)) for q in range(0, 101, 10)}))
            if len(v): S_.append(dict(item="deciles", tool=t, tier=tr, n=len(v), **{f"q{q}": float(np.percentile(v, q)) for q in range(0, 101, 10)}))
        h = g[g.tier.isin(["strong", "weak", "none"]) & g.ipsae.notna()]
        for m in ("ipsae", "iptm"):
            S_.append(dict(item="corr", tool=t, metric=m, n=len(h), pearson=h.CIR.corr(h[m]), spearman=h.CIR.corr(h[m], method="spearman")))
        st = g[g.tier == "strong"]; S_.append(dict(item="moves", tool=t, n_strong_rows=len(st), n_move=int((st.CIR < floor).sum()), n_strong_pairs=st.pair.nunique(),
                                                   n_pairs_move=st[st.CIR < floor].pair.nunique()))
    pd.DataFrame(S_).to_csv(f"{OUT}/summary.tsv", sep="\t", index=False)
    A[A.audit_label == "strong_noCIR"].drop(columns=["dir"]).to_csv(f"{OUT}/moves.tsv", sep="\t", index=False)
    print("floor", floor, "p10", p10, "pos n", len(pc))
