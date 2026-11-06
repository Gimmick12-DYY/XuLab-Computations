# Pilot pairs, layers 1 and 3 (CPU). For each pair:
#  co-binding: per TF, element-level obs/exp over the 159,965 active elements from a Poisson GLM (ATAC, leave-one-out pooled
#    panel, width, GC, GC^2, k50 mappability; same model as Gate 1 v2); score = Pearson residual (obs-mu)/sqrt(mu) among
#    elements with >= 3 reads; top N = 5% of active elements (7,998) per TF (same N for both TFs = depth-matched);
#    co-bound = intersection; expected = N_a N_b / n_active.
#  background: union-ATAC active elements, not co-bound, GC-matched (20 GC bins, 3x the co-bound count, seed 1).
#  motifs: JASPAR 2024 for each partner with a known motif; null = 20 split-half shuffled versions per motif (columns
#    shuffled within each half of the motif); FIMO p < 1e-4 (meme 5.5.7); enrichment = log2 OR co-bound vs background.
#  spacing/orientation: for pairs with two motifs, centre distance and relative strand of the best hits in elements with both.
#  MLX-SREBF2: also CAP-SELEX composite/spacing PWMs (Supp. Table 3) involving MLX/MLXIP/MLXIPL/SREBF1/SREBF2.
#  GC and CpG o/e of co-bound vs background vs all active elements.
#  layer 1: v1 pooled and v2 stratified percentiles from r1_tftf.
import json, os, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, hypergeom, chi2_contingency

X = "/work/users/d/y/dyy12/XuLab"; P = f"{X}/complex_screen/pilot_pairs"; L3 = f"{P}/layer3"
PAIRS = [("MAZ", "ZNF143"), ("EZH2", "MTF2"), ("MLX", "SREBF2"), ("CTCF", "STAG2"), ("RBBP7", "SRCAP")]
JASPAR = f"{X}/motif_analysis/cache/motifdb/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt"
MOTIFS = {"MAZ": ["MA1522.2"], "ZNF143": ["MA0088.2"], "CTCF": ["MA0139.2"], "MLX": ["MA0663.1"], "SREBF2": ["MA0596.1", "MA0828.3"]}
CAP_FAM = {"MLX", "MLXIP", "MLXIPL", "SREBF1", "SREBF2"}
FA = f"{X}/downstream/cache/hg38.fa"
TOPF, NSHUF, PTH = 0.05, 20, 1e-4
rng = np.random.default_rng(1)
os.makedirs(L3, exist_ok=True)

# ---- element data -------------------------------------------------------------------------------------------------------
E = pd.read_csv(f"{X}/regulatome/m0_universe/elements_v2.bed", sep="\t", header=None,
                names=["chrom", "start", "end", "eid", "type", "pid", "ptype", "cls"])
act = np.flatnonzero(E.cls.isin(["promoter", "distal_active", "bivalent"]).values)
Ea = E.iloc[act].reset_index(drop=True); nE = len(Ea)
V = f"{X}/regulatome/gate1/v2"
gc = pd.read_csv(f"{V}/gc.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
mp = pd.read_csv(f"{V}/map.tsv", sep="\t", header=None, index_col=0)[1].reindex(Ea.eid).values
atac = np.load(f"{X}/regulatome/m1_binding/v2/counts/atac_counts.npz")["atac"][act].astype(float)
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
raw = {t: np.load(f"{X}/regulatome/m1_binding/v2/counts/{t}.npz")["raw"][act].astype(float) for t in tfs}
reads = {t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["reads"] for t in tfs}
tot = sum(raw.values()); totr = sum(reads.values())
logw = np.log(Ea.end.values - Ea.start.values)

def glm_mu(y, off, Xd):
    b = np.zeros(Xd.shape[1]); b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ b; mu = np.exp(eta); z = eta - off + (y - mu) / mu
        XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(b)), XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-7: b = bn; break
        b = bn
    return np.exp(off + Xd @ b)

def score(t):
    pool = (tot - raw[t]) / (totr - reads[t]) * 1e6
    Xd = np.column_stack([np.ones(nE), np.log1p(atac / atac.sum() * 1e6), np.log1p(pool), logw, gc, gc ** 2, mp])
    mu = glm_mu(raw[t], np.full(nE, np.log(reads[t] / 1e6)), Xd)
    pr = (raw[t] - mu) / np.sqrt(mu)
    return pr, mu

# union3 ATAC overlap for the background
ab = subprocess.run(f"bedtools intersect -u -a <(awk -F'\\t' -v OFS='\\t' '$8==\"promoter\"||$8==\"distal_active\"||$8==\"bivalent\"{{print $1,$2,$3,$4}}' "
                    f"{X}/regulatome/m0_universe/elements_v2.bed) -b {X}/data/HEK293T_ATAC_union3.bed.gz | cut -f4",
                    shell=True, capture_output=True, text=True, executable="/bin/bash", check=True).stdout.split()
in_atac = Ea.eid.isin(set(ab)).values

# ---- motifs ---------------------------------------------------------------------------------------------------------------
def read_meme(path):
    txt = open(path).read(); out = {}
    for blk in txt.split("MOTIF ")[1:]:
        lines = blk.strip().split("\n"); mid = lines[0].split()[0]
        rows = [l for l in lines if l.strip() and l.strip()[0] in "0123456789." and len(l.split()) == 4]
        out[mid] = (lines[0].strip(), np.array([[float(x) for x in l.split()] for l in rows]))
    return out
JM = read_meme(JASPAR)

def cap_pwms():
    s = pd.read_excel(f"{X}/complex_screen/cap_selex/raw/MOESM5.xlsx", header=None)
    out = {}
    for i in range(len(s) - 4):
        if str(s.iloc[i, 0]).strip() == "Base" and isinstance(s.iloc[i, 1], str) and "_" in s.iloc[i, 1]:
            name = s.iloc[i, 1]
            if not (set(name.upper().split("_")[:2]) & CAP_FAM): continue
            M = s.iloc[i + 1:i + 5, 1:].apply(pd.to_numeric, errors="coerce")
            M = M.loc[:, M.notna().all()].values.T.astype(float)
            M = (M + 0.25 * M.sum(1, keepdims=True) / 100 + 1e-3); M = M / M.sum(1, keepdims=True)
            kind = str(s.iloc[i, 7]); key = f"CAP_{name}_{i}"
            out[key] = (f"{key} {kind}", M)
    return out

def shuffle_half(M):
    h = len(M) // 2; a, b = M[:h].copy(), M[h:].copy(); rng.shuffle(a); rng.shuffle(b); return np.vstack([a, b])

def write_meme(motifs, path):
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for k, (_, M) in motifs.items():
            f.write(f"MOTIF {k}\nletter-probability matrix: alength= 4 w= {len(M)}\n")
            f.write("\n".join(" ".join(f"{x:.6f}" for x in r) for r in M) + "\n\n")

def fimo(meme, fasta, out):
    mb = "/nas/longleaf/rhel9/apps/meme/5.5.7"
    cmd = (f"{mb}/libexec/meme-5.5.7/fasta-get-markov -m 0 {fasta} {out}.bg >/dev/null 2>&1 && "
           f"{mb}/bin/fimo --thresh {PTH} --text --bfile {out}.bg {meme} {fasta} > {out} 2>/dev/null")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    return pd.read_csv(out, sep="\t")

def seq_stats(fa):
    seqs, name = {}, None
    for l in open(fa):
        l = l.strip()
        if l.startswith(">"): name = l[1:].split("::")[0]; seqs[name] = []
        else: seqs[name].append(l.upper())
    out = {}
    for k, v in seqs.items():
        s = "".join(v); n = len(s); c, g = s.count("C"), s.count("G"); cg = s.count("CG")
        out[k] = ((c + g) / n, cg * n / (c * g) if c and g else 0.0)
    return out

# ---- layer 1 tables ------------------------------------------------------------------------------------------------------
v1 = pd.read_csv(f"{X}/regulatome/r1_tftf/r1_pairs_provisional.tsv", sep="\t").set_index(["a", "b"])
v2 = pd.read_csv(f"{X}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t").set_index(["a", "b"])

# ======== v2 (2026-10-29): residual-selected sets split into distal (no TSS within 1 kb) and promoter elements; background
# matched jointly on GC and CpG o/e (10 x 10 quantile bins) from union-ATAC elements of the same set; CTCF-only calibration.
L3 = f"{P}/layer3_v2"; os.makedirs(L3, exist_ok=True)
# GC and CpG o/e for every active element (cached)
cache = f"{L3}/element_gc_cpg.npz"
if not os.path.exists(cache):
    Ea[["chrom", "start", "end", "eid"]].to_csv(f"{L3}/active.bed", sep="\t", header=False, index=False)
    subprocess.run(f"bedtools getfasta -fi {FA} -bed {L3}/active.bed -nameOnly -fo {L3}/active.fa", shell=True, check=True)
    st_ = seq_stats(f"{L3}/active.fa")
    np.savez_compressed(cache, gc=np.array([st_[e][0] for e in Ea.eid]), cpg=np.array([st_[e][1] for e in Ea.eid]))
    os.remove(f"{L3}/active.fa")
z_ = np.load(cache); GC2, CPG = z_["gc"], z_["cpg"]
SETS = {"distal": Ea.cls.isin(["distal_active", "bivalent"]).values & (Ea.type != "promoter").values,
        "promoter": (Ea.type == "promoter").values}

def motif_set(units):
    mot = {}
    for t in units:
        for mid in MOTIFS.get(t, []):
            mot[f"{t}|{mid}"] = JM[mid]
            for k in range(NSHUF): mot[f"{t}|{mid}|shuf{k}"] = ("shuffle", shuffle_half(JM[mid][1]))
    if tuple(units) == ("MLX", "SREBF2"):
        caps = cap_pwms(); mot.update(caps)
        for k_, v_ in list(caps.items()):
            for s_ in range(5): mot[f"{k_}|shuf{s_}"] = ("shuffle", shuffle_half(v_[1]))
    if tuple(units) == ("RBBP7", "SRCAP"):  # negative pair has no own motifs: pilot motifs as a specificity check
        for t in ("MAZ", "ZNF143", "CTCF", "MLX", "SREBF2"):
            for mid in MOTIFS[t]:
                mot[f"{t}|{mid}"] = JM[mid]
                for k in range(NSHUF): mot[f"{t}|{mid}|shuf{k}"] = ("shuffle", shuffle_half(JM[mid][1]))
    return mot

SC = {}
summary = []
def analyze(units, setname):
    tag = "_".join(units) + ("_only" if len(units) == 1 else ""); d = f"{L3}/{tag}/{setname}"; os.makedirs(d, exist_ok=True)
    msk = SETS[setname]; ns = int(msk.sum())
    for t in units:
        if t not in SC: SC[t] = score(t)
    top = {}
    for t in units:
        pr = SC[t][0].copy(); pr[(raw[t] < 3) | ~msk | (SC[t][0] <= 0)] = -np.inf  # >= 3 reads and positive residual
        n = int(min(round(TOPF * ns), np.isfinite(pr).sum()))
        top[t] = set(np.argsort(-pr)[:n])
    sel = np.array(sorted(set.intersection(*[top[t] for t in units])))
    nt = [len(top[t]) for t in units]
    exp_n = nt[0] * nt[1] / ns if len(units) == 2 else float(nt[0])
    p_hyper = float(hypergeom.sf(len(sel) - 1, ns, nt[0], nt[1])) if len(units) == 2 else None
    pool = np.flatnonzero(in_atac & msk & ~np.isin(np.arange(nE), sel))
    qg = np.quantile(GC2[msk], np.linspace(0, 1, 11)); qc = np.quantile(CPG[msk], np.linspace(0, 1, 11))
    bgc = np.clip(np.digitize(GC2, qg[1:-1]), 0, 9) * 10 + np.clip(np.digitize(CPG, qc[1:-1]), 0, 9)
    bg = []
    for k in np.unique(bgc[sel]):
        want = 3 * int((bgc[sel] == k).sum()); cand = pool[bgc[pool] == k]
        if len(cand): bg += list(rng.choice(cand, size=min(want, len(cand)), replace=False))
    bg = np.array(sorted(bg))
    for nm, ix in (("cobound", sel), ("background", bg)):
        Ea.iloc[ix][["chrom", "start", "end", "eid"]].to_csv(f"{d}/{nm}.bed", sep="\t", header=False, index=False)
        subprocess.run(f"bedtools getfasta -fi {FA} -bed {d}/{nm}.bed -nameOnly -fo {d}/{nm}.fa", shell=True, check=True)
    mot = motif_set(units); rows, hits = [], {}
    if mot and len(sel):
        write_meme(mot, f"{d}/motifs.meme")
        for nm in ("cobound", "background"):
            hits[nm] = fimo(f"{d}/motifs.meme", f"{d}/{nm}.fa", f"{d}/fimo_{nm}.tsv")
        nco, nbg = len(sel), len(bg); frac = {}
        for k in mot:
            fc = hits["cobound"].loc[hits["cobound"].iloc[:, 0] == k].iloc[:, 2].nunique() if len(hits["cobound"]) else 0
            fb = hits["background"].loc[hits["background"].iloc[:, 0] == k].iloc[:, 2].nunique() if len(hits["background"]) else 0
            frac[k] = (fc, fb, np.log2(((fc + 0.5) / (nco - fc + 0.5)) / ((fb + 0.5) / (nbg - fb + 0.5))))
        for k, (fc, fb, lor) in frac.items():
            if "|shuf" in k: continue
            sh = np.array([frac[x][2] for x in frac if x.startswith(k + "|shuf")])
            rows.append(dict(set=setname, motif=k, desc=mot[k][0], cobound_hit=fc, cobound_n=nco, bg_hit=fb, bg_n=nbg,
                             frac_cobound=fc / max(nco, 1), frac_bg=fb / max(nbg, 1), log2OR=lor,
                             fisher_p=fisher_exact([[fc, nco - fc], [fb, nbg - fb]])[1], shuffle_mean=sh.mean(), shuffle_max=sh.max(),
                             shuffle_pct=float((sh < lor).mean()), beats_all_shuffles=bool(lor > sh.max())))
    M = pd.DataFrame(rows); M.to_csv(f"{d}/motif_enrichment.tsv", sep="\t", index=False, float_format="%.4g")
    spacing = {}
    if len(units) == 2 and all(t in MOTIFS for t in units) and hits:
        a, b = units
        for nm in ("cobound", "background"):
            H = hits[nm]; H.columns = [c.lstrip("#").strip() for c in H.columns]; best = {}
            for t in units:
                best[t] = H[H.motif_id == f"{t}|{MOTIFS[t][0]}"].sort_values("p-value").drop_duplicates("sequence_name").set_index("sequence_name")
            both = best[a].index.intersection(best[b].index)
            ca = (best[a].loc[both, "start"] + best[a].loc[both, "stop"]) / 2; cb = (best[b].loc[both, "start"] + best[b].loc[both, "stop"]) / 2
            spacing[nm] = dict(n_both=int(len(both)), dist=(cb - ca).abs().values.tolist(),
                               same_strand=(best[a].loc[both, "strand"].values == best[b].loc[both, "strand"].values).tolist())
        tab = np.array([np.histogram(spacing[nm]["dist"], [0, 10, 20, 30, 50, 100, 200, 1e9])[0] for nm in ("cobound", "background")])
        spacing["chi2_p_distance"] = float(chi2_contingency(tab + 0.5)[1]) if tab.sum(1).min() > 0 else None
        spacing["frac_same_strand"] = {nm: float(np.mean(spacing[nm]["same_strand"])) if spacing[nm]["n_both"] else None for nm in ("cobound", "background")}
        json.dump({k: spacing[k] for k in ("cobound", "background")}, open(f"{d}/spacing.json", "w"))
    allset = np.flatnonzero(msk)
    med = lambda v, ix: float(np.median(v[ix])) if len(ix) else None
    res = dict(unit=tag, set=setname, n_set=ns, n_top=nt, n_selected=int(len(sel)), expected=exp_n,
               enrichment=(len(sel) / exp_n) if len(units) == 2 and exp_n else None, p_hyper=p_hyper, n_background=int(len(bg)),
               top_fraction=TOPF, min_reads=3, positive_residual=True, background="union-ATAC elements of the same set, matched on GC x CpG o/e deciles, 3x",
               gc={"cobound": med(GC2, sel), "background": med(GC2, bg), "all_in_set": med(GC2, allset)},
               cpg_oe={"cobound": med(CPG, sel), "background": med(CPG, bg), "all_in_set": med(CPG, allset)},
               spacing={k: v for k, v in spacing.items() if k in ("chi2_p_distance", "frac_same_strand")} | {f"n_both_{k}": spacing[k]["n_both"] for k in ("cobound", "background") if k in spacing})
    if len(units) == 2:
        l1 = {}
        a, b = units
        for src, T in (("v1_pooled", v1), ("v2_stratified", v2)):
            k = (a, b) if (a, b) in T.index else (b, a)
            l1[src] = {c: (None if pd.isna(T.loc[k, c]) else float(T.loc[k, c])) for c in T.columns if c.endswith("_pct")}
            l1[src]["view1"] = bool(T.loc[k, "view1"]); l1[src]["edge_type"] = str(T.loc[k, "edge_type"])
        res["layer1"] = l1
        np.savez_compressed(f"{d}/scatter.npz", a=SC[a][0], b=SC[b][0], cobound=sel, top_a=np.array(sorted(top[a])),
                            top_b=np.array(sorted(top[b])), eid=Ea.eid.values, setmask=msk)
    json.dump(res, open(f"{d}/summary.json", "w"), indent=1)
    print(f"== {tag} [{setname}] set {ns:,}; top {nt}; selected {len(sel)} (exp {exp_n:.0f}); bg {len(bg)}; GC {res['gc']}; CpG {res['cpg_oe']}; spacing {res['spacing']}", flush=True)
    if len(M): print(M[~M.motif.str.startswith("CAP_")].round(3).to_string(index=False))
    if len(M) and M.motif.str.startswith("CAP_").any():
        print(M[M.motif.str.startswith("CAP_")].sort_values("log2OR", ascending=False).head(8).round(3).to_string(index=False))
    summary.append(res)

for s_ in ("distal", "promoter"):
    analyze(("CTCF",), s_)  # calibration: all residual-selected CTCF elements
    for a, b in PAIRS: analyze((a, b), s_)
json.dump(summary, open(f"{L3}/all_summary.json", "w"), indent=1)
