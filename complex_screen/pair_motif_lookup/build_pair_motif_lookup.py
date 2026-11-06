#!/usr/bin/env python3
"""Shared-motif lookup for TF pairs from the existing per-TF motif enrichment results.

Reads finished HOMER known / AME / HOMER de novo outputs only (no new enrichment runs).
Sets:
  A_raw_homer      raw lab peaks, HOMER known, -useNewBg             (vast .../bigwig/motif/homer)
  A_raw_ame        raw lab peaks, AME HOCOMOCO v11 core, shuffled    (vast .../bigwig/motif/ame)
  A_raw_denovo     raw lab peaks, HOMER de novo homerResults
  B_imp_peaks_homer / _homer_ocr / _ame / _ame_ocr   imputed MACS3 peaks (motif_analysis/motif)
  C_imp_bins_homer / _ame                            imputed top-20k 1 kb bins (motif_analysis/motif_bins)
  E_imp_denovo_unknown  motif_analysis/motif_imputed (45 TFs without a known motif)
  E_imp_denovo_peaks    motif_analysis/motif/<tf>/homer/homerResults (78 TFs, imputed peaks)
Pass rules: HOMER known p <= 1e-5; AME adj p <= 0.05; de novo motif p <= 1e-5 and
similarity >= 0.6 (HOMER-style: Pearson correlation of aligned frequency matrices, padded
with 0.25, best offset and strand).
Usage (Longleaf, reads the result trees in place):
  python build_pair_motif_lookup.py <out_dir>
Usage (local mirror laid out as inputs/{raw_lab,raw_peaks,motif,motif_bins,motif_imputed,regulatome,complex_screen}):
  python build_pair_motif_lookup.py <out_dir> <mirror_dir>
"""
import glob, math, os, re, sys, itertools
from collections import defaultdict
import numpy as np, pandas as pd

OUT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 2:
    M = sys.argv[2]
    P = dict(raw=f"{M}/raw_lab", raw_peaks=f"{M}/raw_peaks", ma=M, xulab=M, cells=f"{M}/tf_cells.tsv")
else:
    X = "/work/users/d/y/dyy12/XuLab"; V = "/vast/som/xujie_lab/TAGATG-293T/bigwig"   # /vast is read only
    P = dict(raw=f"{V}/motif", raw_peaks=f"{V}/peaks", ma=f"{X}/motif_analysis", xulab=X, cells=f"{HERE}/tf_cells.tsv")
os.makedirs(OUT, exist_ok=True)
HOMER_LOGP = math.log(1e-5)          # HOMER "Log P-value" is natural log
AME_Q, SIM, SPEC_N = 0.05, 0.6, 10
GATES = {("CTCF", "STAG2"): "gate_pos", ("MAZ", "ZNF143"): "gate_pos", ("EZH2", "MTF2"): "gate_pos", ("RBBP7", "SRCAP"): "gate_neg"}

def canon(g):
    g = g.upper()
    return "HP1" if g in ("CBX1", "CBX3", "CBX5", "HP1") else g

def pkey(a, b):
    a, b = sorted((canon(a), canon(b))); return a, b

# ---------------- panel, cells -------------------------------------------------
cells = pd.read_csv(P["cells"], sep="\t").set_index("tf")["n_cells"].to_dict()
PANEL = sorted(cells)
assert len(PANEL) == 79, len(PANEL)

# ---------------- own-motif tokens --------------------------------------------
ALIAS = {"CTCF": {"CTCF", "CTCFL", "BORIS"}, "HP1": {"CBX1", "CBX3", "CBX5", "HP1"}, "NFYA": {"NFYA", "NFY"},
         "ZNF143": {"ZNF143", "STAF"}, "RELA": {"RELA", "P65", "NFKB"}, "SREBF2": {"SREBF2", "SREBP2"}}
def tokens(name):
    s = name.split("/")[0]
    s = re.split(r"[(]", s)[0]
    s = s.split("_HUMAN")[0]
    return {t.upper() for t in re.split(r"[|:,]+", s) if t}
def is_own(tf, motif):
    own = ALIAS.get(tf, {tf}); return bool(own & tokens(motif))

# ---------------- known-motif tables ------------------------------------------
def read_homer(f):
    d = pd.read_csv(f, sep="\t")
    d = d.rename(columns={d.columns[0]: "motif", d.columns[1]: "consensus", d.columns[3]: "logp"})
    d["rank"] = np.arange(1, len(d) + 1); d["p"] = np.exp(d["logp"].clip(lower=-700)); d["p_str"] = d.iloc[:, 2].astype(str)
    d["pass"] = d["logp"] <= HOMER_LOGP
    return d[["motif", "consensus", "p", "p_str", "logp", "rank", "pass"]]
def read_ame(f):
    d = pd.read_csv(f, sep="\t", comment="#")
    d = d[d["motif_ID"].notna()].copy()
    d["motif"] = d["motif_ID"].astype(str); d["p"] = d["adj_p-value"].astype(float); d["p_str"] = d["adj_p-value"].astype(str)
    d["logp"] = np.log(d["p"].clip(lower=1e-300)); d["rank"] = d["rank"].astype(int); d["pass"] = d["p"] <= AME_Q
    return d[["motif", "consensus", "p", "p_str", "logp", "rank", "pass"]]

def gather(pattern, tf_from_path, reader):
    res = {}
    for f in glob.glob(pattern):
        tf = canon(tf_from_path(f))
        if tf in cells:
            try: res[tf] = reader(f)
            except Exception as e: print("skip", f, e)
    return res

KNOWN = {
 "A_raw_homer": gather(f"{P['raw']}/homer/TF.*_motif_NewBg/knownResults.txt", lambda f: re.search(r"TF\.(.+)_motif_NewBg", f).group(1), read_homer),
 "A_raw_ame": gather(f"{P['raw']}/ame/*/ame.tsv", lambda f: f.split("/")[-2], read_ame),
 "B_imp_peaks_homer": gather(f"{P['ma']}/motif/*/homer/knownResults.txt", lambda f: f.split("/")[-3], read_homer),
 "B_imp_peaks_homer_ocr": gather(f"{P['ma']}/motif/*/homer_ocr/knownResults.txt", lambda f: f.split("/")[-3], read_homer),
 "B_imp_peaks_ame": gather(f"{P['ma']}/motif/*/ame/*/ame.tsv", lambda f: f.split("/")[-4], read_ame),
 "B_imp_peaks_ame_ocr": gather(f"{P['ma']}/motif/*/ame_ocr/*/ame.tsv", lambda f: f.split("/")[-4], read_ame),
 "C_imp_bins_homer": gather(f"{P['ma']}/motif_bins/*/homer/knownResults.txt", lambda f: f.split("/")[-3], read_homer),
 "C_imp_bins_ame": gather(f"{P['ma']}/motif_bins/*/ame/*/ame.tsv", lambda f: f.split("/")[-4], read_ame),
}

# ---------------- de novo motifs ----------------------------------------------
def read_motif(f):
    L = open(f).read().strip().splitlines(); h = L[0].split("\t")
    m = np.array([[float(x) for x in l.split()] for l in L[1:]]); m = m / m.sum(1, keepdims=True)
    pm = re.search(r"P:([0-9.eE+-]+)", L[0]); p = float(pm.group(1)) if pm else 1.0
    lp = math.log(p) if p > 0 else float(h[3])
    bg = re.search(r"BestGuess:(.+?)\(([0-9.]+)\)", h[1])
    return dict(id=h[1].split(",")[0], consensus=h[0].lstrip(">"), mat=m, p=p, logp=lp, best=(bg.group(1) if bg else "."), best_score=(float(bg.group(2)) if bg else np.nan),
                rank=int(re.search(r"motif(\d+)\.motif$", f).group(1)))
def gather_denovo(pattern, tf_from_path):
    res = defaultdict(list)
    for f in glob.glob(pattern):
        tf = canon(tf_from_path(f))
        if tf in cells: res[tf].append(read_motif(f))
    return {k: sorted(v, key=lambda r: r["rank"]) for k, v in res.items()}
DENOVO = {
 "A_raw_denovo": gather_denovo(f"{P['raw']}/homer/TF.*_motif_NewBg/homerResults/motif*.motif", lambda f: re.search(r"TF\.(.+)_motif_NewBg", f).group(1)),
 "E_imp_denovo_unknown": gather_denovo(f"{P['ma']}/motif_imputed/*/homer/homerResults/motif*.motif", lambda f: f.split("/")[-4]),
 "E_imp_denovo_peaks": gather_denovo(f"{P['ma']}/motif/*/homer/homerResults/motif*.motif", lambda f: f.split("/")[-4]),
}

def sim_matrix(A, B, min_overlap=6):
    """HOMER-style similarity between motif lists A, B (lists of Lx4 arrays): max over strand/offset of
    Pearson r between the two matrices placed on a common span, uncovered columns padded with 0.25."""
    out = np.full((len(A), len(B)), -1.0)
    byL = lambda M: defaultdict(list, {})
    ga, gb = defaultdict(list), defaultdict(list)
    for i, m in enumerate(A): ga[len(m)].append(i)
    for j, m in enumerate(B): gb[len(m)].append(j)
    for la, ia in ga.items():
        XA = np.stack([A[i] for i in ia])                                   # na x la x 4
        for lb, jb in gb.items():
            XB = np.stack([B[j] for j in jb]); XBr = XB[:, ::-1, ::-1]       # reverse complement (ACGT order)
            best = np.full((len(ia), len(jb)), -1.0)
            for Y in (XB, XBr):
                for off in range(-(lb - min_overlap), la - min_overlap + 1):
                    s0 = min(0, off); s1 = max(la, off + lb); W = s1 - s0
                    PA = np.full((len(ia), W, 4), .25); PA[:, -s0:-s0 + la] = XA
                    PB = np.full((len(jb), W, 4), .25); PB[:, off - s0:off - s0 + lb] = Y
                    a = PA.reshape(len(ia), -1); b = PB.reshape(len(jb), -1)
                    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
                    r = (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-12)
                    best = np.maximum(best, r)
            out[np.ix_(ia, jb)] = best
    return out

# ---------------- panel frequency (known) -------------------------------------
freq_rows, PASS, FREQ = [], {}, {}
for s, D in KNOWN.items():
    PASS[s] = {tf: d[d["pass"]].set_index("motif") for tf, d in D.items()}
    cnt = defaultdict(int); cons = {}
    for tf, d in PASS[s].items():
        for m, r in d.iterrows(): cnt[m] += 1; cons.setdefault(m, r["consensus"])
    FREQ[s] = cnt
    for m, n in cnt.items(): freq_rows.append(dict(set=s, motif=m, consensus=cons[m], n_tfs_passing=n, n_tfs_in_set=len(D)))

# ---------------- de novo: passing motifs, similarity, panel frequency ---------
DN = {}
for s, D in DENOVO.items():
    keep = [(tf, r) for tf in sorted(D) for r in D[tf] if r["logp"] <= HOMER_LOGP]
    mats = [r["mat"] for _, r in keep]
    S = sim_matrix(mats, mats) if mats else np.zeros((0, 0))
    tfv = np.array([tf for tf, _ in keep])
    # panel frequency of a de novo motif = number of TFs (incl. itself) with a passing motif at similarity >= SIM
    nfreq = np.array([len(set(tfv[S[i] >= SIM])) for i in range(len(keep))]) if len(keep) else np.array([])
    DN[s] = dict(keep=keep, S=S, tfv=tfv, nfreq=nfreq, ntf=len(D))
    for i, (tf, r) in enumerate(keep):
        freq_rows.append(dict(set=s, motif=f"{tf}|{r['id']}", consensus=r["consensus"], n_tfs_passing=int(nfreq[i]), n_tfs_in_set=len(D), best_known=r["best"]))
pd.DataFrame(freq_rows).sort_values(["set", "n_tfs_passing"], ascending=[True, False]).to_csv(f"{OUT}/motif_panel_frequency.tsv", sep="\t", index=False, float_format="%.4g")

# ---------------- pair annotations ---------------------------------------------
R1 = pd.read_csv(f"{P['xulab']}/regulatome/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t")
V2 = pd.read_csv(f"{P['xulab']}/regulatome/r1_tftf/v2/r1v2_pairs_provisional.tsv", sep="\t")
ann = {}
for d in (R1,):
    for r in d.itertuples(): ann[pkey(r.a, r.b)] = dict(r1_class=r._3, af3_tier_r1=r.af3_tier, string_physical=r.string_physical, capselex=r.capselex, boltz_trimmed_tier=r.boltz_trimmed_tier)
for r in V2.itertuples():
    ann.setdefault(pkey(r.a, r.b), {}).update(pearson_domain=r.pearson_domain, pearson_domain_pct=r.pearson_domain_pct, spqn_noquies_pct=r.spqn_noquies_pct)
AF = f"{P['xulab']}/complex_screen/af3_novel_20261103"
for f, lab in ((f"{AF}/list.tsv", "af3_novel_list"), (f"{AF}/weak.tsv", "af3_weak_list")):
    for r in pd.read_csv(f, sep="\t").itertuples(): ann.setdefault(pkey(r.a, r.b), {})["af3_input"] = lab
for f in (f"{AF}/af3_scores.tsv", f"{AF}/af3_scores_weak.tsv", f"{AF}/af3_scores_srcap.tsv"):
    for r in pd.read_csv(f, sep="\t").itertuples():
        a, b = r.pair.split("__"); k = pkey(a, b); ann.setdefault(k, {}).update(af3_tier=r.tier, af3_iptm=r.iptm_top, af3_ipsae=r.ipsae_max)

# co-binding: % of the smaller raw peak set overlapping the other TF's raw peaks
def load_peaks(tf):
    f = f"{P['raw_peaks']}/TF.{tf}_peaks.q005.narrowPeak"
    if tf == "HP1" and not os.path.exists(f): f = f"{P['raw_peaks']}/TF.HP1_peaks.q005.narrowPeak"
    d = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2]); return {c: g[[1, 2]].to_numpy() for c, g in d.groupby(0)}
PK = {tf: load_peaks(tf) for tf in PANEL if os.path.exists(f"{P['raw_peaks']}/TF.{tf}_peaks.q005.narrowPeak") and os.path.getsize(f"{P['raw_peaks']}/TF.{tf}_peaks.q005.narrowPeak") > 0}
def frac_overlap(A, B):
    hit = tot = 0
    for c, a in A.items():
        tot += len(a); b = B.get(c)
        if b is None or not len(b): continue
        b = b[np.argsort(b[:, 0])]; mx = np.maximum.accumulate(b[:, 1])
        i = np.searchsorted(b[:, 0], a[:, 1], side="left") - 1   # last B start < A end
        ok = i >= 0; hit += int(np.sum(mx[i[ok]] > a[ok, 0]))
    return hit / tot if tot else np.nan
NPK = {tf: sum(len(v) for v in p.values()) for tf, p in PK.items()}

# ---------------- lookup ------------------------------------------------------
rows, verdict_rows = [], []
pairs = list(itertools.combinations(PANEL, 2))
for a, b in pairs:
    an = ann.get((a, b), {}); grp = GATES.get((a, b), "")
    if a in PK and b in PK:
        s, l = (a, b) if NPK[a] <= NPK[b] else (b, a); cob = 100 * frac_overlap(PK[s], PK[l])
    else: cob = np.nan
    base = dict(pair=f"{a}-{b}", a=a, b=b, cells_a=cells[a], cells_b=cells[b], pct_cobinding=round(cob, 2) if cob == cob else np.nan,
                af3_tier=an.get("af3_tier", an.get("af3_tier_r1", "not_run")), gate=grp)
    v = dict(pair=f"{a}-{b}", a=a, b=b, gate=grp, af3_input=an.get("af3_input", ""), af3_tier=base["af3_tier"], af3_iptm=an.get("af3_iptm", np.nan),
             string_physical=an.get("string_physical", np.nan), capselex=an.get("capselex", ""), r1_class=an.get("r1_class", ""),
             pearson_domain=an.get("pearson_domain", np.nan), pearson_domain_pct=an.get("pearson_domain_pct", np.nan), spqn_noquies_pct=an.get("spqn_noquies_pct", np.nan),
             cells_a=cells[a], cells_b=cells[b], pct_cobinding=base["pct_cobinding"])
    for s in KNOWN:
        P = PASS[s]
        if a not in P or b not in P: v[f"{s}"] = "untestable"; continue
        shared = P[a].index.intersection(P[b].index); nsp = 0; tops = []
        for m in shared:
            ra, rb = P[a].loc[m], P[b].loc[m]; n = FREQ[s][m]; spec = n <= SPEC_N; nsp += spec
            rows.append({**base, "set": s, "motif": m, "consensus": ra["consensus"], "p_a": ra["p_str"], "p_b": rb["p_str"], "rank_a": int(ra["rank"]), "rank_b": int(rb["rank"]),
                         "n_panel_tfs_passing": n, "n_tfs_in_set": len(KNOWN[s]), "is_own_motif_a": is_own(a, m), "is_own_motif_b": is_own(b, m), "pair_specific": spec,
                         "similarity": np.nan, "best_known_match": "."})
            if spec: tops.append((max(ra["logp"], rb["logp"]), m))
        v[f"{s}_n_shared"] = len(shared); v[f"{s}_n_specific"] = nsp
        v[f"{s}_top_specific"] = ";".join(m.split("/")[0] for _, m in sorted(tops)[:3])
        v[s] = "pair-specific shared" if nsp else ("accessibility only" if len(shared) else "nothing shared")
    for s, X in DN.items():
        if a not in DENOVO[s] or b not in DENOVO[s]: v[s] = "untestable"; continue
        ia = np.flatnonzero(X["tfv"] == a); ib = np.flatnonzero(X["tfv"] == b); n_sh = nsp = 0; tops = []
        for i in ia:
            for j in ib:
                sm = X["S"][i, j]
                if sm < SIM: continue
                ra, rb = X["keep"][i][1], X["keep"][j][1]; n = int(max(X["nfreq"][i], X["nfreq"][j])); spec = n <= SPEC_N; n_sh += 1; nsp += spec
                rows.append({**base, "set": s, "motif": f"{ra['id']}~{rb['id']}", "consensus": f"{ra['consensus']}~{rb['consensus']}", "p_a": f"{ra['p']:.0e}", "p_b": f"{rb['p']:.0e}",
                             "rank_a": ra["rank"], "rank_b": rb["rank"], "n_panel_tfs_passing": n, "n_tfs_in_set": X["ntf"], "is_own_motif_a": is_own(a, ra["best"]), "is_own_motif_b": is_own(b, rb["best"]),
                             "pair_specific": spec, "similarity": round(float(sm), 3), "best_known_match": f"{ra['best']}|{rb['best']}"})
                if spec: tops.append((-sm, f"{ra['consensus']}({ra['best'].split('/')[0]})"))
        v[f"{s}_n_shared"] = n_sh; v[f"{s}_n_specific"] = nsp; v[f"{s}_top_specific"] = ";".join(t for _, t in sorted(tops)[:3])
        v[s] = "pair-specific shared" if nsp else ("accessibility only" if n_sh else "nothing shared")
    sets = list(KNOWN) + list(DN)
    v["n_sets_testable"] = sum(v[s] != "untestable" for s in sets)
    v["n_sets_pair_specific"] = sum(v[s] == "pair-specific shared" for s in sets)
    v["sets_pair_specific"] = ",".join(s for s in sets if v[s] == "pair-specific shared")
    v["verdict"] = ("pair-specific shared" if v["n_sets_pair_specific"] else "accessibility only" if any(v[s] == "accessibility only" for s in sets)
                    else "untestable" if v["n_sets_testable"] == 0 else "nothing shared")
    verdict_rows.append(v)

L = pd.DataFrame(rows)
cols = ["pair", "a", "b", "set", "motif", "consensus", "p_a", "p_b", "rank_a", "rank_b", "n_panel_tfs_passing", "n_tfs_in_set", "cells_a", "cells_b", "pct_cobinding",
        "af3_tier", "is_own_motif_a", "is_own_motif_b", "pair_specific", "similarity", "best_known_match", "gate"]
L[cols].to_csv(f"{OUT}/pair_motif_lookup.tsv", sep="\t", index=False)
V = pd.DataFrame(verdict_rows); V.to_csv(f"{OUT}/pair_verdict.tsv", sep="\t", index=False, float_format="%.4g")
print("pairs", len(V), "lookup rows", len(L)); print(V.verdict.value_counts())
