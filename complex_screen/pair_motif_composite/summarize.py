# Candidate calls, validation inputs and the gate summary for the composite-motif pair test.
# python summarize.py candidates PAIRS...   -> candidates/<pair>/{candidates.tsv, homer.motifs, meme.meme}
# python summarize.py final PAIRS...        -> gate_summary.tsv (+ candidates_all.tsv)
# Candidate rule (discovery chromosomes): motif present in pair-vs-A-only and pair-vs-B-only HOMER runs (known: same motif; de novo: HOMER-style
# similarity >= 0.6), fold >= 1.5 over each single pool and p <= 1e-5 in both. STREME motifs with SEA (pair vs A-only + B-only) p <= 1e-5 and
# enrichment >= 1.5 are carried to validation as a MEME-side list. Validation (even chromosomes + X, pair vs singles): log2FC > 0.75 and P < 0.01.
# Two pass levels per pair in gate_summary.tsv:
#   shared    = TF A's own HOMER known motif enriched in the pair pool vs B-only (fold >= 1.5, p <= 1e-5), or B's own motif vs A-only (gate criterion)
#   composite = the candidate rule above (fold >= 1.5 over BOTH single pools, p <= 1e-5 in both; not expected for the gate pairs)
# Each level has its own permutation null (best motif at that level in each permutation). vs-input fold is reported and flagged weak
# where the GC x CpG-matched input pool fell short (>10% short or GC gap > 0.02).
import os, sys, re, glob, math
import numpy as np, pandas as pd
O = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite"; FOLD = 1.5; LOGP = -5; SIM = 0.6; NPERM = 50
OWN = {"CTCF-STAG2": ["CTCF(Zf)"], "MAZ-ZNF143": ["Maz(Zf)", "ZNF143|STAF(Zf)"], "NFYA-ZNF644": ["NFY(CCAAT)"], "E2F3-HES1": ["E2F3(E2F)"], "RBBP7-SRCAP": []}
HOMER_KNOWN = "/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs"
TF_OWN = {"CTCF": ["CTCF(Zf)"], "MAZ": ["Maz(Zf)"], "ZNF143": ["ZNF143|STAF(Zf)"], "NFYA": ["NFY(CCAAT)"], "E2F3": ["E2F3(E2F)"], "SREBF2": ["Srebp2(bHLH)"],
          "DDIT3": ["Chop(bZIP)"], "EPAS1": ["HIF2a(bHLH)"], "RELA": ["NFkB-p65(RHD)", "NFkB-p65-Rel(RHD)"], "RBPJ": ["Rbpj1(?)"], "ETS2": ["ETS(ETS)"]}
def short(n): return str(n).split("/")[0]
def own_of(tf):
    if tf in TF_OWN: return TF_OWN[tf]
    names = {short(b.split("\t")[1]) for b in open(HOMER_KNOWN).read().split(">")[1:]}
    return sorted(n for n in names if re.split(r"[(|]", n)[0].upper() == tf)
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
        a_, b_ = pair.split("-"); ownset = set(own_of(a_) + own_of(b_))
        keep = {r["motif"] for r in rows}; hm += [">" + b.strip() for b in allk if b.split("\t")[1] in keep or short(b.split("\t")[1]) in ownset]
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
def perm_best(pair, ownA=(), ownB=()):
    # per permutation: best composite p (fold >= 1.5 vs both) and best shared p (A own vs B-only, B own vs A-only, fold >= 1.5)
    best = []; sh = []
    for d in sorted(glob.glob(f"{O}/results/{pair}/perm/*")):
        kA, kB = known(f"{d}/vsA/knownResults.txt"), known(f"{d}/vsB/knownResults.txt")
        if kA is None or kB is None: continue
        j = kA.join(kB, lsuffix="A", rsuffix="B", how="inner"); j = j[(j.foldA >= FOLD) & (j.foldB >= FOLD)]
        best.append(np.max([j.log10pA.values, j.log10pB.values], 0).min() if len(j) else 0.0)
        s = [kB.loc[n].log10p for n in kB.index if short(n) in ownA and kB.loc[n].fold >= FOLD] + [kA.loc[n].log10p for n in kA.index if short(n) in ownB and kA.loc[n].fold >= FOLD]
        sh.append(min(s) if s else 0.0)
    return np.array(best), np.array(sh)
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
        PS = pd.read_csv(f"{O}/pool_sizes.tsv", sep="\t").set_index("pair"); rows = []; allc = []
        def look(k, motifs):
            if k is None: return (np.nan, np.nan, "")
            h = [n for n in k.index if short(n) in motifs]
            if not h: return (np.nan, np.nan, "")
            n = min(h, key=lambda n: k.loc[n].log10p); return (round(k.loc[n].fold, 2), round(k.loc[n].log10p, 1), short(n))
        for p in pairs:
            a, b = p.split("-"); R = f"{O}/results/{p}"; oa, ob = own_of(a), own_of(b)
            kA, kB, kI = (known(f"{R}/homer_{c}/known/knownResults.txt") for c in ["vsA", "vsB", "vsInput"])
            f = f"{O}/candidates/{p}/candidates.tsv"; T = pd.read_csv(f, sep="\t") if os.path.getsize(f) > 1 else pd.DataFrame(); allc.append(T)
            pb, ps = perm_best(p, oa, ob); V = val(p); ps_ = PS.loc[p]
            weak = bool(ps_.disc_input_short > 0.1 * 2 * ps_.disc_pair or abs(ps_.disc_gc_pair - ps_.disc_gc_input) > 0.02)
            A_vsB, A_vsA, A_vsI = look(kB, oa), look(kA, oa), look(kI, oa); B_vsA, B_vsB, B_vsI = look(kA, ob), look(kB, ob), look(kI, ob)
            shA = A_vsB[0] >= FOLD and A_vsB[1] <= LOGP; shB = B_vsA[0] >= FOLD and B_vsA[1] <= LOGP
            obs_sh = np.nanmin([x[1] for x, ok_ in [(A_vsB, A_vsB[0] >= FOLD), (B_vsA, B_vsA[0] >= FOLD)] if ok_] or [np.nan])
            vs = lambda m: next((V[k] for k in V if short(k) == m), (np.nan, np.nan))
            r = dict(pair=p, category=ps_.category, n_pair=ps_.disc_pair,
                     own_A=A_vsB[2] or "none", A_fold_vs_partner_only=A_vsB[0], A_log10p_vs_partner_only=A_vsB[1], A_fold_vs_own_only=A_vsA[0], A_fold_vs_input=A_vsI[0],
                     own_B=B_vsA[2] or "none", B_fold_vs_partner_only=B_vsA[0], B_log10p_vs_partner_only=B_vsA[1], B_fold_vs_own_only=B_vsB[0], B_fold_vs_input=B_vsI[0],
                     shared_pass=bool(shA or shB), shared_perm_p=(1 + np.sum(ps <= obs_sh)) / (1 + len(ps)) if len(ps) and not np.isnan(obs_sh) else np.nan,
                     own_A_val_log2fc=round(vs(A_vsB[2])[0], 2), own_B_val_log2fc=round(vs(B_vsA[2])[0], 2), input_weak=weak)
            if len(T):
                c = T.assign(score=T[["log10p_vsA", "log10p_vsB"]].max(1)).sort_values("score").iloc[0]
                key = c.motif if c.source == "homer_known" else c.motif.split("|")[0]
                vk = next((k for k in V if k == c.motif or str(k).startswith(key)), None)
                r.update(composite_n=len(T), composite_best=c.motif if c.source == "homer_denovo" else short(c.motif), composite_source=c.source,
                         comp_fold_vsA=round(c.fold_vsA, 2), comp_fold_vsB=round(c.fold_vsB, 2), comp_log10p=round(c.score, 1), comp_fold_vsInput=round(c.get("fold_vsInput", np.nan), 2),
                         composite_perm_p=(1 + np.sum(pb <= c.score)) / (1 + len(pb)) if len(pb) else np.nan,
                         comp_val_log2fc=round(V[vk][0], 2) if vk else np.nan, comp_val_log10p=round(V[vk][1], 1) if vk else np.nan,
                         composite_validated=bool(vk and V[vk][0] > 0.75 and V[vk][1] < -2))
            else: r.update(composite_n=0, composite_best="none")
            r["composite_pass"] = bool(r["composite_n"] > 0 and r.get("composite_validated", False))
            jid = JASPAR_OWN.get(r["own_A"], "") or JASPAR_OWN.get(r["own_B"], "")
            r.update(centrimo_own_log10p=centrimo(p, jid), spamo_top=" | ".join(x for x in [spamo(p, j) for j in ([JASPAR_OWN.get(r["own_A"])] + [JASPAR_OWN.get(r["own_B"])]) if j] if x))
            r["gate_pass"] = r["shared_pass"] if p in OWN and OWN[p] else np.nan; rows.append(r)
        G = pd.DataFrame(rows); G.to_csv(f"{O}/gate_summary.tsv", sep="\t", index=False)
        pd.concat([t for t in allc if len(t)] or [pd.DataFrame()]).to_csv(f"{O}/candidates_all.tsv", sep="\t", index=False); print(G.to_string())
