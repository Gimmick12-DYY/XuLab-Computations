# Candidate calls, validation inputs and the gate summary for the composite-motif pair test.
# python summarize.py candidates PAIRS...   -> candidates/<pair>/{candidates.tsv, homer.motifs, meme.meme}
# python summarize.py final PAIRS...        -> gate_summary.tsv (+ candidates_all.tsv)
# Candidate rule (discovery chromosomes): motif present in pair-vs-A-only and pair-vs-B-only HOMER runs (known: same motif; de novo: HOMER-style
# similarity >= 0.6), fold >= 1.5 over each single pool and p <= 1e-5 in both. STREME motifs with SEA (pair vs A-only + B-only) p <= 1e-5 and
# enrichment >= 1.5 are carried to validation as a MEME-side list. Validation (even chromosomes + X, pair vs singles): log2FC > 0.75 and P < 0.01.
import os, sys, re, glob, math
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; FOLD = 1.5; LOGP = -5; SIM = 0.6; NPERM = 50
OWN = {"CTCF-STAG2": ["CTCF(Zf)"], "MAZ-ZNF143": ["Maz(Zf)", "ZNF143|STAF(Zf)"], "NFYA-ZNF644": ["NFY(CCAAT)"], "E2F3-HES1": ["E2F3(E2F)"], "RBBP7-SRCAP": []}
JASPAR_OWN = {"CTCF(Zf)": "MA0139.2", "Maz(Zf)": "MA1522.2", "ZNF143|STAF(Zf)": "MA0088.2", "NFY(CCAAT)": "MA0060.4", "E2F3(E2F)": "MA0469.4"}
def known(f):
    if not os.path.exists(f): return None
    k = pd.read_csv(f, sep="\t"); k.columns = ["name", "cons", "p", "lnp", "q", "nt", "pt", "nb", "pb"] + list(k.columns[9:])
    pt = k.pt.astype(str).str.rstrip("%").astype(float); pb = k.pb.astype(str).str.rstrip("%").astype(float)
    return pd.DataFrame(dict(name=k.name, log10p=k.lnp / math.log(10), fold=(pt + 0.01) / (pb + 0.01), pt=pt, pb=pb)).set_index("name")
def denovo(f):
    out = []
    if not os.path.exists(f): return out
    for blk in open(f).read().split(">")[1:]:
        L = blk.strip().split("\n"); h = L[0].split("\t"); m = np.array([[float(x) for x in l.split()] for l in L[1:]])
        st = h[5]; t = float(re.search(r"T:[\d.]+\(([\d.]+)%", st).group(1)); b = float(re.search(r"B:[\d.]+\(([\d.]+)%", st).group(1))
        out.append(dict(cons=h[0], name=h[1], log10p=float(h[3]) / math.log(10), fold=(t + 0.01) / (b + 0.01), m=m, raw=">" + blk.strip()))
    return out
def sim(a, b):
    best = -1
    for bb in (b, b[::-1, ::-1]):
        for off in range(-(len(bb) - 1), len(a)):
            lo, hi = min(0, off), max(len(a), off + len(bb))
            if min(len(a), off + len(bb)) - max(0, off) < min(len(a), len(bb)) // 2 + 1: continue
            A = np.full((hi - lo, 4), .25); B = A.copy(); A[-lo:-lo + len(a)] = a; B[off - lo:off - lo + len(bb)] = bb
            best = max(best, np.corrcoef(A.ravel(), B.ravel())[0, 1])
    return best
def ok(r): return r.fold >= FOLD and r.log10p <= LOGP
def candidates(pair):
    R = f"{O}/results/{pair}"; C = f"{O}/candidates/{pair}"; os.makedirs(C, exist_ok=True); rows = []; hm = []; mm = []
    kA, kB = known(f"{R}/homer_vsA/known/knownResults.txt"), known(f"{R}/homer_vsB/known/knownResults.txt"); kI = known(f"{R}/homer_vsInput/known/knownResults.txt")
    if kA is not None and kB is not None:
        for n in kA.index.intersection(kB.index):
            a, b = kA.loc[n], kB.loc[n]
            if ok(a) and ok(b):
                rows.append(dict(pair=pair, source="homer_known", motif=n, fold_vsA=a.fold, fold_vsB=b.fold, log10p_vsA=a.log10p, log10p_vsB=b.log10p,
                                 fold_vsInput=kI.loc[n].fold if kI is not None else np.nan, log10p_vsInput=kI.loc[n].log10p if kI is not None else np.nan))
        allk = open("/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs").read().split(">")[1:]
        keep = {r["motif"] for r in rows}; hm += [">" + b.strip() for b in allk if b.split("\t")[1] in keep]
    dA = [m for f in sorted(glob.glob(f"{R}/homer_vsA/L*/homerMotifs.all.motifs")) for m in denovo(f)]
    dB = [m for f in sorted(glob.glob(f"{R}/homer_vsB/L*/homerMotifs.all.motifs")) for m in denovo(f)]
    for x in dA:
        if not ok(pd.Series(x)): continue
        ys = [(sim(x["m"], y["m"]), y) for y in dB if ok(pd.Series(y))]
        if not ys: continue
        s, y = max(ys, key=lambda t: t[0])
        if s >= SIM:
            rows.append(dict(pair=pair, source="homer_denovo", motif=f"{x['cons']}|{x['name'].split('BestGuess:')[-1]}", match_vsB=y["cons"], similarity=round(s, 3),
                             fold_vsA=x["fold"], fold_vsB=y["fold"], log10p_vsA=x["log10p"], log10p_vsB=y["log10p"])); hm.append(x["raw"])
    if os.path.exists(f"{R}/streme/streme.txt"):
        # STREME motifs pass on STREME's own held-out E-value (re-testing them with SEA on the same pools would be circular)
        st = open(f"{R}/streme/streme.txt").read(); hdr = st.split("MOTIF")[0]
        for b in st.split("MOTIF")[1:]:
            m = re.search(r"nsites= *\d+ E= *([\d.e+-]+)", b)
            if m and float(m.group(1)) <= 1e-5: mm.append("MOTIF" + b)
        if mm: open(f"{C}/meme.meme", "w").write(hdr + "".join(mm))
    open(f"{C}/homer.motifs", "w").write("\n".join(hm) + ("\n" if hm else ""))
    T = pd.DataFrame(rows); T.to_csv(f"{C}/candidates.tsv", sep="\t", index=False); return T
def perm_best(pair):
    best = []
    for d in sorted(glob.glob(f"{O}/results/{pair}/perm/*")):
        kA, kB = known(f"{d}/vsA/knownResults.txt"), known(f"{d}/vsB/knownResults.txt")
        if kA is None or kB is None: continue
        j = kA.join(kB, lsuffix="A", rsuffix="B", how="inner"); j = j[(j.foldA >= FOLD) & (j.foldB >= FOLD)]
        best.append(np.max([j.log10pA.values, j.log10pB.values], 0).min() if len(j) else 0.0)
    return np.array(best)
def val(pair):
    out = {}; k = known(f"{O}/results/{pair}/val_homer/knownResults.txt")
    if k is not None:
        for n, r in k.iterrows(): out[n] = (math.log2(r.fold), r.log10p)
    s = f"{O}/results/{pair}/val_sea/sea.tsv"
    if os.path.exists(s):
        v = pd.read_csv(s, sep="\t", comment="#")
        for _, r in v.iterrows(): out[r.ID] = (math.log2((r["TP%"] + 0.01) / (r["FP%"] + 0.01)), r.LOG_PVALUE / math.log(10))
    return out
def centrimo(pair, jid):
    f = f"{O}/results/{pair}/centrimo/site_counts.txt"; t = f"{O}/results/{pair}/centrimo/centrimo.tsv"
    if not os.path.exists(t) or not jid: return np.nan
    c = pd.read_csv(t, sep="\t", comment="#"); c = c[c.motif_id == jid]; return c.log_adj_p_value.min() / math.log(10) if len(c) else np.nan
def spamo(pair, jid):
    f = f"{O}/results/{pair}/spamo_{jid}/spamo.tsv"
    if not jid or not os.path.exists(f): return ""
    s = pd.read_csv(f, sep="\t", comment="#")
    if not len(s): return "none"
    s = s.sort_values("p_value").iloc[0]; return f"{s.secondary_alt or s.secondary_id} gap {s.gap} {s.orientation} p={s.p_value:.1e}"
if __name__ == "__main__":
    stage, pairs = sys.argv[1], sys.argv[2:]
    if stage == "candidates":
        for p in pairs: T = candidates(p); print(p, len(T), "candidates")
    else:
        rows = []; allc = []
        for p in pairs:
            T = pd.read_csv(f"{O}/candidates/{p}/candidates.tsv", sep="\t") if os.path.getsize(f"{O}/candidates/{p}/candidates.tsv") > 1 else pd.DataFrame()
            pb = perm_best(p); V = val(p); allc.append(T)
            short = lambda n: str(n).split("/")[0]
            own = [m for m in OWN.get(p, []) if len(T) and m in set(T[T.source == "homer_known"].motif.map(short))]
            show = T.assign(score=T[["log10p_vsA", "log10p_vsB"]].max(1)).sort_values("score").head(5) if len(T) else T
            if not len(show): rows.append(dict(pair=p, candidate="none", own_motifs=";".join(OWN.get(p, [])), gate_pass=False if OWN.get(p) else np.nan, n_perm=len(pb)))
            for _, r in show.iterrows():
                key = r.motif if r.source == "homer_known" else r.motif.split("|")[0]
                vk = next((k for k in V if k == r.motif or str(k).startswith(key)), None); jid = JASPAR_OWN.get(short(r.motif), "")
                obs = max(r.log10p_vsA, r.log10p_vsB)
                rows.append(dict(pair=p, candidate=r.motif, source=r.source, fold_vsA=round(r.fold_vsA, 2), fold_vsB=round(r.fold_vsB, 2), log10p_vsA=round(r.log10p_vsA, 1), log10p_vsB=round(r.log10p_vsB, 1),
                                 centrimo_log10p=centrimo(p, jid), spamo_top=spamo(p, jid), n_perm=len(pb), perm_p=(1 + np.sum(pb <= obs)) / (1 + len(pb)) if len(pb) else np.nan,
                                 val_log2fc=round(V[vk][0], 2) if vk else np.nan, val_log10p=round(V[vk][1], 1) if vk else np.nan,
                                 validated=bool(vk and V[vk][0] > 0.75 and V[vk][1] < -2), own_motif=r.source == "homer_known" and short(r.motif) in OWN.get(p, []), gate_pass=bool(own) if OWN.get(p) else np.nan))
        G = pd.DataFrame(rows); G.to_csv(f"{O}/gate_summary.tsv", sep="\t", index=False)
        pd.concat([t for t in allc if len(t)] or [pd.DataFrame()]).to_csv(f"{O}/candidates_all.tsv", sep="\t", index=False); print(G.to_string())
