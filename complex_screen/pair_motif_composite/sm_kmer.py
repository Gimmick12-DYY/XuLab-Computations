# Sequence model (2026-11-12, user go): per library, L2 logistic regression on reverse-complement-collapsed 6-mer frequencies, sites vs matched
# background (motif_valid inputs: lab summits top 1,000 in open chromatin, HOT / Alu / artifacts removed, background matched on GC x CpG x
# promoter x ATAC level). One model trained on odd chromosomes (tested on even + X) and one the other way round.
# Gates fixed before results (as motif_valid / motif_centred):
#  G0 (signal): held-out AUROC per library reported.
#  G1 (own motif learned): models' k-mer weights (panel-standardized: minus the mean weight vector over all libraries) correlated with each
#     motif's 6-mer score (best log-odds of the 6-mer inside the PWM); own motif rank <= 10 in BOTH models for > 3 TFs.
#  G2 (anchors): A's odd-chr model scores B's even-chr sites vs B's even-chr background (AUROC); percentile of the real B among the other 78
#     libraries >= 0.95 for >= 2 of CTCF->STAG2, MAZ<->ZNF143, NFYA->ZNF644; negatives CTCF->NFYA, NFYA->CTCF must not pass.
# Pairs (frozen candidates): cross-prediction both ways and weight similarity (cosine of panel-standardized weights), percentile among all pairs.
# Output sequence_model/{auroc.tsv, own_gate.tsv, calibration.tsv, anchor_controls.tsv, pairs.tsv, weights_disc.npy, weights_val.npy, kmers.txt}
import os, re, itertools
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/sequence_model"; V = f"{O}/motif_valid"; os.makedirs(G, exist_ok=True)
K = 6; rc = lambda s: s[::-1].translate(str.maketrans("ACGT", "TGCA")); kmers = sorted({min(k, rc(k)) for k in map("".join, itertools.product("ACGT", repeat=K))}); ki = {k: i for i, k in enumerate(kmers)}
for k in list(ki): ki[rc(k)] = ki[k]
seq = {}; name = None
for l in open(f"{V}/all.fa"):
    if l[0] == ">": name = l[1:].strip()
    else: seq[name] = l.strip().upper()
names = list(seq); ni = {n: i for i, n in enumerate(names)}; rows, cols = [], []
for i, n in enumerate(names):
    s = seq[n]; idx = [ki[s[j:j + K]] for j in range(len(s) - K + 1) if s[j:j + K] in ki]; rows += [i] * len(idx); cols += idx
Xk = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(names), len(kmers))); Xk = sp.csr_matrix(Xk.multiply(1 / np.maximum(Xk.sum(1), 1)))
print(Xk.shape, flush=True)
S = pd.read_csv(f"{V}/sites.tsv", sep="\t"); S["key"] = S.key.str.replace(":", "_").str.replace("-", "_"); tfs = sorted(S.tf.unique())
def xy(t, pt):
    g = S[(S.tf == t) & (S.part == pt)]; f = [ni[k] for k in g.key[g.set == "fg"] if k in ni]; b = [ni[k] for k in g.key[g.set == "bg"] if k in ni]
    return Xk[f + b], np.r_[np.ones(len(f)), np.zeros(len(b))]
W = {"disc": np.zeros((len(tfs), len(kmers))), "val": np.zeros((len(tfs), len(kmers)))}; AU = []; models = {}
for j, t in enumerate(tfs):
    r = dict(tf=t)
    for tr, te in [("disc", "val"), ("val", "disc")]:
        Xa, ya = xy(t, tr); Xb, yb = xy(t, te)
        if ya.sum() < 100 or yb.sum() < 50: continue
        m = LogisticRegression(C=0.1, max_iter=2000).fit(Xa * 100, ya); W[tr][j] = m.coef_[0]; models[(t, tr)] = m
        r[f"auroc_train_{tr}_test_{te}"] = round(roc_auc_score(yb, m.decision_function(Xb * 100)), 3)
    AU.append(r); print(r, flush=True)
AU = pd.DataFrame(AU); AU.to_csv(f"{G}/auroc.tsv", sep="\t", index=False); np.save(f"{G}/weights_disc.npy", W["disc"]); np.save(f"{G}/weights_val.npy", W["val"]); open(f"{G}/kmers.txt", "w").write("\n".join(kmers))
Wp = {p: W[p] - W[p].mean(0, keepdims=True) for p in W}
# motif 6-mer scores
alt, pwm, cur, rr = {}, {}, None, []
for l in list(open(f"{O}/db/known.meme")) + ["MOTIF __end__"]:
    if l.startswith("MOTIF "):
        if cur and rr: pwm[cur] = np.array(rr)
        x = l.split(); cur = x[1]; alt[cur] = x[2] if len(x) > 2 else x[1]; rr = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.": rr.append([float(v) for v in l.split()])
motifs = [m for m in pwm if len(pwm[m]) >= K]; B = {c: i for i, c in enumerate("ACGT")}; KS = np.zeros((len(motifs), len(kmers)))
for a, m in enumerate(motifs):
    L = np.log2((pwm[m] + 1e-3) / 0.25); Lr = L[::-1, ::-1]
    for k, i in ((k, ki[k]) for k in kmers):
        c = [B[x] for x in k]; best = -1e9
        for M_ in (L, Lr):
            for o in range(len(M_) - K + 1): best = max(best, M_[np.arange(o, o + K), c].sum())
        KS[a, i] = best
KSz = (KS - KS.mean(1, keepdims=True)) / KS.std(1, keepdims=True)
COR = {p: (KSz @ ((Wp[p] - Wp[p].mean(1, keepdims=True)) / np.maximum(Wp[p].std(1, keepdims=True), 1e-9)).T) / len(kmers) for p in Wp}  # motifs x TFs
ALIAS = {"HP1": ["CBX3", "CBX1", "CBX5"], "ZUFSP": ["ZUP1"]}
def toks(s): return set(re.split(r"[.@:|/_\- ]+", s.upper()))
own = []; best = {}
for j, t in enumerate(tfs):
    ids = [a for a, m in enumerate(motifs) if any(n in toks(m) | toks(alt[m]) for n in ALIAS.get(t, [t]))]
    if not ids or not W["disc"][j].any(): continue
    rk = {p: pd.Series(COR[p][:, j]).rank(ascending=False, method="min").values for p in COR}; k = ids[int(np.argmin(rk["disc"][ids]))]; best[t] = k
    own.append(dict(tf=t, best_own=motifs[k], rank_disc_model=int(rk["disc"][ids].min()), rank_val_model=int(rk["val"][ids].min()), top_motif_disc=alt[motifs[int(np.argmax(COR["disc"][:, j]))]]))
OW = pd.DataFrame(own); OW["strict"] = (OW.rank_disc_model <= 10) & (OW.rank_val_model <= 10); OW["loose"] = (OW.rank_disc_model <= 25) & (OW.rank_val_model <= 25); OW.to_csv(f"{G}/own_gate.tsv", sep="\t", index=False)
# cross prediction: A's disc model on B's val sites
CP = np.full((len(tfs), len(tfs)), np.nan); XYv = {t: xy(t, "val") for t in tfs}
for i, a in enumerate(tfs):
    if (a, "disc") not in models: continue
    for j, b in enumerate(tfs):
        Xb, yb = XYv[b]
        if yb.sum() >= 50 and (yb == 0).sum() >= 50: CP[i, j] = roc_auc_score(yb, models[(a, "disc")].decision_function(Xb * 100))
pd.DataFrame(CP, index=tfs, columns=tfs).round(3).to_csv(f"{G}/cross_prediction_auroc.tsv", sep="\t")
def pctl(a, b):
    i, j = tfs.index(a), tfs.index(b); v = CP[i]; o = [k for k in range(len(tfs)) if k != i and np.isfinite(v[k])]
    return float(np.mean(v[o] < v[j])) if np.isfinite(v[j]) else np.nan
POS = [("CTCF", "STAG2"), ("MAZ", "ZNF143"), ("ZNF143", "MAZ"), ("NFYA", "ZNF644")]; NEG = [("CTCF", "NFYA"), ("NFYA", "CTCF")]
AC = pd.DataFrame([dict(anchor=a, partner=b, kind="positive" if (a, b) in POS else "negative", cross_auroc=round(CP[tfs.index(a), tfs.index(b)], 3), pct=round(pctl(a, b), 3)) for a, b in POS + NEG]); AC.to_csv(f"{G}/anchor_controls.tsv", sep="\t", index=False)
ps = lambda a, b: bool((AC[(AC.anchor == a) & (AC.partner == b)].pct >= .95).any()); npos = int(ps("CTCF", "STAG2")) + int(ps("MAZ", "ZNF143") or ps("ZNF143", "MAZ")) + int(ps("NFYA", "ZNF644")); nneg = int(ps("CTCF", "NFYA")) + int(ps("NFYA", "CTCF"))
CAL = pd.DataFrame([dict(method="kmer_logreg", median_heldout_auroc=round(AU.filter(like="auroc").stack().median(), 3), n_tf_with_own=len(OW), own_strict=int(OW.strict.sum()), own_loose=int(OW.loose.sum()), own_strict_tfs=",".join(OW.tf[OW.strict]),
                      positives_passed=npos, negatives_passed=nneg, gate1=int(OW.strict.sum()) > 3, gate2=(npos >= 2) and (nneg == 0))]); CAL["usable"] = CAL.gate1 & CAL.gate2; CAL.to_csv(f"{G}/calibration.tsv", sep="\t", index=False)
# pairs
Wn = Wp["disc"] / np.maximum(np.linalg.norm(Wp["disc"], axis=1, keepdims=True), 1e-9); SIM = Wn @ Wn.T; allsim = SIM[np.triu_indices(len(tfs), 1)]
FZ = pd.read_csv(f"{X}/complex_screen/candidates_frozen_v1/candidates_frozen_v1.tsv", sep="\t"); FZ = FZ[FZ.candidate.astype(str) == "True"]; pr = []
for r in FZ.itertuples():
    if r.tf_a not in tfs or r.tf_b not in tfs: continue
    i, j = tfs.index(r.tf_a), tfs.index(r.tf_b); s = SIM[i, j]
    pr.append(dict(pair=r.pair, best_pct=r.best_pct, af3=r.af3_5seed_tier if r.af3_5seed_tier != "not_run" else r.af3_1seed_tier, weight_cosine=round(s, 3), cosine_pct=round(float(np.mean(allsim < s)), 3),
                   cross_A_to_B=round(CP[i, j], 3), pct_A_to_B=round(pctl(r.tf_a, r.tf_b), 3), cross_B_to_A=round(CP[j, i], 3), pct_B_to_A=round(pctl(r.tf_b, r.tf_a), 3)))
PR = pd.DataFrame(pr); PR.to_csv(f"{G}/pairs.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(CAL.to_string(index=False)); print(AC.to_string(index=False)); print(AU.describe().round(3).to_string())
print(OW.sort_values("rank_disc_model").head(20).to_string(index=False))
a = PR[PR.af3.isin(["strong", "weak"])]; print("AF3 pairs:", len(a), "cosine_pct>=0.95:", int((a.cosine_pct >= .95).sum()), "either cross pct>=0.95:", int(((a.pct_A_to_B >= .95) | (a.pct_B_to_A >= .95)).sum()))
print("all candidates:", len(PR), "cosine_pct>=0.95:", int((PR.cosine_pct >= .95).sum()), "either cross pct>=0.95:", int(((PR.pct_A_to_B >= .95) | (PR.pct_B_to_A >= .95)).sum()))
