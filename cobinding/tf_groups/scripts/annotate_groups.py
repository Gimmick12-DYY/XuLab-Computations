"""Biological tests for TF groups. For each group: items (pairs or anchor regions) shared by >=2 members; anchor class
(promoter = GENCODE v46 basic protein-coding TSS +-1 kb, else HEK293T ChromHMM18 majority state); within-group
STRING / CAP-SELEX support; all compared with 1,000 depth-matched random groups (same size, each member swapped for a
non-member of similar item count) and with the members' own non-shared items. Writes group gene lists for GO.
args: indir null_tag [groups_tag]"""
import sys, gzip, re, numpy as np, pandas as pd, scipy.sparse as sp
from collections import defaultdict
from pathlib import Path
ROOT = Path("/work/users/d/y/dyy12/XuLab"); IN = Path(sys.argv[1]); TAG = sys.argv[2]; LEVEL = TAG.split("_")[0]; GTAG = sys.argv[3] if len(sys.argv) > 3 else TAG
rng = np.random.default_rng(7); K = 1000
tfs = (IN / "tfs.txt").read_text().split(); ti = {t: i for i, t in enumerate(tfs)}
A = sp.load_npz(IN / f"inc_{LEVEL}.npz").tocsc()
reg = pd.read_csv(IN / "regions.tsv", sep="\t"); pairs = pd.read_csv(IN / "pairs.tsv", sep="\t")
G = pd.read_csv(IN / f"groups_{GTAG}.tsv", sep="\t"); dep = pd.read_csv(IN / f"null_{TAG}_depth.tsv", sep="\t", header=None, index_col=0)[1]
# ---- region classes
tss = defaultdict(list)
for ln in gzip.open(ROOT / "data/gencode_v46/gencode.v46.basic.annotation.gtf.gz", "rt"):
    if ln[0] == "#": continue
    x = ln.split("\t")
    if x[2] != "gene" or 'gene_type "protein_coding"' not in x[8]: continue
    p = int(x[3]) if x[6] == "+" else int(x[4]); tss[x[0]].append((p, re.search('gene_name "([^"]+)"', x[8]).group(1)))
tss = {c: (np.array([a for a, _ in sorted(v)]), [b for _, b in sorted(v)]) for c, v in tss.items()}
CL = {"TssA": "TSS", "TssFlnk": "TSS", "TssFlnkU": "TSS", "TssFlnkD": "TSS", "EnhA1": "Enh", "EnhA2": "Enh", "EnhG1": "Enh", "EnhG2": "Enh",
      "EnhWk": "Enh", "TssBiv": "Biv", "EnhBiv": "Biv", "Tx": "Tx", "TxWk": "Tx", "ReprPC": "Repr", "ReprPCWk": "Repr", "Het": "Het/ZNF", "ZNF/Rpts": "Het/ZNF", "Quies": "Quies"}
hmm = defaultdict(list)
for ln in gzip.open(ROOT / "data/HEK293T_chromHMM18.bed.gz", "rt"):
    x = ln.split("\t"); hmm[x[0]].append((int(x[1]), int(x[2]), x[3]))
hmm = {c: (np.array([a for a, _, _ in v]), np.array([b for _, b, _ in v]), [s for _, _, s in v]) for c, v in hmm.items()}
cls, gene = [], []
for c, s, e in zip(reg.chrom, reg.start, reg.end):
    g = ""
    if c in tss:
        P, N = tss[c]; lo, hi = np.searchsorted(P, s - 1000), np.searchsorted(P, e + 1000)
        if hi > lo: g = ",".join(sorted(set(N[lo:hi])))
    if g: cls.append("Promoter"); gene.append(g); continue
    gene.append(""); S, E, St = hmm.get(c, (np.array([0]), np.array([0]), ["Quies"]))
    lo = max(np.searchsorted(E, s, side="right"), 0); best, st = 0, "Quies"
    for k in range(lo, min(lo + 20, len(S))):
        if S[k] >= e: break
        ov = min(e, E[k]) - max(s, S[k])
        if ov > best: best, st = ov, St[k]
    cls.append(CL[st])
reg["cls"] = cls; reg["gene"] = gene; reg.to_csv(IN / "regions_annotated.tsv", sep="\t", index=False)
rcls = reg.cls.values
if LEVEL == "pair":
    c1, c2 = rcls[pairs.r1.values], rcls[pairs.r2.values]
    def anchors(items): return np.concatenate([pairs.r1.values[items], pairs.r2.values[items]])
else:
    def anchors(items): return items
# ---- STRING and CAP-SELEX
st = pd.read_csv(ROOT / "cobinding/data_external/string_network.tsv", sep="\t")
mp = pd.read_csv(ROOT / "cobinding/data_external/string_map.tsv", sep="\t"); inv = dict(zip(mp.preferredName, mp["query"]))
string = {}
for a, b, s, e in zip(st.preferredName_A, st.preferredName_B, st.score, st.escore):
    a, b = inv.get(a, a), inv.get(b, b); string[frozenset((a, b))] = (s, e)
cap = pd.read_csv(ROOT / "complex_screen/cap_selex/panel_pairs_capselex.tsv", sep="\t")
capset = {frozenset((a, b)) for a, b, c in zip(cap.tf_a, cap.tf_b, cap.call) if c == 1}
Acsr = A.tocsr()
def shared_items(members):
    idx = [ti[t] for t in members]; cnt = np.asarray(Acsr[idx].sum(0)).ravel(); return np.where(cnt >= 2)[0], np.where(cnt == 1)[0]
def metrics(members):
    sh, own = shared_items(members); a = anchors(sh); n = max(len(a), 1); o = anchors(own)
    m = {"n_shared": len(sh), "prom": (rcls[a] == "Promoter").sum() / n, "enh": (rcls[a] == "Enh").sum() / n,
         "prom_own": (rcls[o] == "Promoter").mean() if len(o) else np.nan, "enh_own": (rcls[o] == "Enh").mean() if len(o) else np.nan}
    if LEVEL == "pair":
        x1, x2 = c1[sh], c2[sh]; pp = ((x1 == "Promoter") & (x2 == "Promoter")).mean() if len(sh) else np.nan
        pe = (((x1 == "Promoter") & (x2 == "Enh")) | ((x2 == "Promoter") & (x1 == "Enh"))).mean() if len(sh) else np.nan
        m.update(PP=pp, PE=pe)
    prs = [frozenset((u, v)) for i, u in enumerate(members) for v in members[i + 1:]]
    m["string04"] = np.mean([string.get(p, (0, 0))[0] >= 0.4 for p in prs]); m["string_exp"] = np.mean([string.get(p, (0, 0))[1] > 0 for p in prs])
    m["capselex"] = sum(p in capset for p in prs)
    return m
elig = G.tf.tolist(); ld = np.log10(dep[elig].astype(float))
lib = pd.read_csv(IN / "library_strata.tsv", sep="\t", header=None, index_col=0)[1] if "strat" in TAG else pd.Series("all", index=elig)
def random_group(members):          # depth- (and, for _strat, library-stratum-) matched: each member -> one of the 3 nearest-depth TFs
    out = []
    for t in members:
        cand = [u for u in elig if u != t and u not in out and lib.get(u) == lib.get(t)] or [u for u in elig if u != t and u not in out]
        cand = sorted(cand, key=lambda u: abs(ld[u] - ld[t]))[:3]; out.append(rng.choice(cand))
    return out
rows, genes_bg = [], set(g for x in reg.gene[reg.cls == "Promoter"] for g in x.split(","))
(IN / f"go_{GTAG}").mkdir(exist_ok=True); (IN / f"go_{GTAG}" / "background.txt").write_text("\n".join(sorted(genes_bg)) + "\n")
for g, sub in G[G.group > 0].groupby("group"):
    mem = sub.tf.tolist(); m = metrics(mem); R = pd.DataFrame([metrics(random_group(mem)) for _ in range(K)])
    row = {"group": g, "size": len(mem), "tfs": ",".join(mem), **m}
    for k in ["n_shared", "prom", "enh", "string04", "string_exp"] + (["PP", "PE"] if LEVEL == "pair" else []):
        row[f"{k}_null_mean"] = R[k].mean(); row[f"{k}_pctile"] = (R[k] < m[k]).mean() + 0.5 * (R[k] == m[k]).mean()
    rows.append(row)
    sh, _ = shared_items(mem); a = anchors(sh)
    gl = sorted(set(x for s in reg.gene.values[a] if s for x in s.split(",")))
    (IN / f"go_{GTAG}" / f"group{g}_genes.txt").write_text("\n".join(gl) + "\n")
pd.DataFrame(rows).to_csv(IN / f"group_tests_{GTAG}.tsv", sep="\t", index=False)
print(pd.DataFrame(rows)[["group", "size", "n_shared", "n_shared_pctile", "prom", "prom_null_mean", "prom_pctile", "enh", "enh_null_mean", "enh_pctile", "string04", "string04_pctile", "capselex"]].round(3).to_string())
print("CAP-SELEX positive panel pairs:", len(capset))
