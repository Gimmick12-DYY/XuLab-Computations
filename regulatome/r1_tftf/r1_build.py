# R1 TF-TF typed edge list (re-runnable; provisional until the Boltz screen finishes; --freeze writes sha256 of the tables)
# View 1 (co-binding): domain-corrected Pearson (enrich), ChromHMM-element SpQN (allS), Cicero shared-pair excess z.
#   Each metric: OLS on library-group pair type + depth (log cells sum, |log cells diff|) -> residual -> depth-matched
#   percentile (null = pairs whose TFs are within +/-4 cell-count ranks of each TF). Edge if >= 2 metrics at >= 0.90.
# View 2 (physical / DNA): Boltz-2 tiers (strong ipSAE >= 0.3 & ipTM >= 0.4; weak ipSAE >= 0.1), CAP-SELEX called pairs.
# Reference annotation only: STRING physical (>= 700), CAP-SELEX tested status. CORUM not available (download endpoints down).
# Types: complex = view1 + Boltz; DNA-guided = view1 + CAP-SELEX call; co-localized = view1 only; physical-only = view2 only.
# Modules: overlapping, seeded from known complexes + STRING physical cliques, extended by view-1 percentiles; Leiden on
#   the remaining TFs (view-1 graph).
import hashlib, itertools, json, sys
import numpy as np, pandas as pd

R = "/work/users/d/y/dyy12/XuLab/regulatome/r1_tftf"
X = "/work/users/d/y/dyy12/XuLab"
FREEZE = "--freeze" in sys.argv
tfs = [l.strip() for l in open(f"{X}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]
SYM = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
to_panel = {g: t for t in tfs for g in SYM.get(t, [t])}
cells = pd.Series({t: json.load(open(f"{X}/regulatome/m1_binding/v2/counts/{t}.json"))["cells"] for t in tfs})
F = pd.read_csv(f"{X}/regulatome/m1_binding/v2/frac_reads_by_class.tsv", sep="\t", index_col=0)
group = (F.all_elements > 0.22).map({True: "high", False: "low"})
pairs = pd.DataFrame(list(itertools.combinations(sorted(tfs), 2)), columns=["a", "b"])

def tfmat(path):
    m = pd.read_csv(path, sep="\t", index_col=0); m.index = m.index.str.upper(); m.columns = m.columns.str.upper()
    return m
def lookup(m, a, b):
    return m.loc[a, b] if a in m.index and b in m.columns else np.nan

# ---- view 1 metrics --------------------------------------------------------------------------------------------------
enrich, allS = tfmat(f"{R}/inputs/enrich.tsv"), tfmat(f"{R}/inputs/allS.tsv")
pairs["pearson_domain"] = [lookup(enrich, a, b) for a, b in zip(pairs.a, pairs.b)]
pairs["spqn_chromhmm"] = [lookup(allS, a, b) for a, b in zip(pairs.a, pairs.b)]
S = pd.read_csv(f"{R}/inputs/fig_tf_x_shared_pairs_min2_matrix.tsv.gz", sep="\t", index_col=0)
S.index = S.index.str.upper()
npairs = pd.read_csv(f"{R}/inputs/pairs_per_tf.tsv", sep="\t", header=None, index_col=0)[1]; npairs.index = npairs.index.str.upper()
U = 375053  # distinct Cicero pairs > 25 kb across 68 TFs
Sv = (S.values > 0).astype(int); ov = pd.DataFrame(Sv @ Sv.T, index=S.index, columns=S.index)
def cic_z(a, b):
    if a not in npairs or b not in npairs or npairs[a] == 0 or npairs[b] == 0: return np.nan
    o = ov.loc[a, b] if a in ov.index and b in ov.index else 0
    e = npairs[a] * npairs[b] / U
    return (o - e) / np.sqrt(e)
pairs["cicero_excess_z"] = [cic_z(a, b) for a, b in zip(pairs.a, pairs.b)]
METRICS = ["pearson_domain", "spqn_chromhmm", "cicero_excess_z"]

pairs["gpair"] = [ "-".join(sorted([group[a], group[b]])) for a, b in zip(pairs.a, pairs.b)]
la, lb = np.log(cells[pairs.a].values), np.log(cells[pairs.b].values)
pairs["depth_sum"], pairs["depth_diff"] = la + lb, np.abs(la - lb)
rank = cells.rank()
for m in METRICS:
    ok = pairs[m].notna()
    D = pd.get_dummies(pairs.loc[ok, "gpair"]).astype(float)
    Xd = np.column_stack([D.values, pairs.loc[ok, ["depth_sum", "depth_diff"]].values])
    beta, *_ = np.linalg.lstsq(Xd, pairs.loc[ok, m].values, rcond=None)
    pairs.loc[ok, f"{m}_resid"] = pairs.loc[ok, m].values - Xd @ beta
    # depth-matched percentile: pairs whose two TFs are each within +/-4 cell-count ranks of a and b
    ra, rb = rank[pairs.a].values, rank[pairs.b].values
    res = pairs[f"{m}_resid"].values; pct = np.full(len(pairs), np.nan)
    for i in np.flatnonzero(ok.values):
        nb = ok.values & (((np.abs(ra - ra[i]) <= 4) & (np.abs(rb - rb[i]) <= 4)) | ((np.abs(ra - rb[i]) <= 4) & (np.abs(rb - ra[i]) <= 4)))
        nb[i] = False
        if nb.sum() >= 10: pct[i] = (res[nb] < res[i]).mean()
    pairs[f"{m}_pct"] = pct
P = pairs[[f"{m}_pct" for m in METRICS]]
pairs["view1_n_metrics"] = P.notna().sum(1)
pairs["view1_n_hi"] = (P >= 0.90).sum(1)
pairs["view1_score"] = P.mean(1)
pairs["view1"] = pairs.view1_n_hi >= 2

# ---- view 2 + references -----------------------------------------------------------------------------------------------
def key(a, b): return tuple(sorted([a, b]))
boltz = {}
try:
    B = pd.read_csv(f"{X}/complex_screen/screen/scores.tsv", sep="\t")
    B = B[(B.status == "ok") & (B.set.isin(["het", "hp1_paralog"]))]
    for r in B.itertuples():
        a, b = to_panel.get(r.chain_a), to_panel.get(r.chain_b)
        if a is None or b is None or a == b: continue
        tier = "strong" if (r.ipsae_max >= 0.3 and r.iptm_top >= 0.4) else ("weak" if r.ipsae_max >= 0.1 else "none")
        k = key(a, b); order = {"none": 0, "weak": 1, "strong": 2}
        prev = boltz.get(k)  # HP1 = best of CBX1/3/5
        if prev is None or order[tier] > order[prev[0]] or (tier == prev[0] and r.ipsae_max > prev[2]):
            boltz[k] = (tier, r.iptm_top, r.ipsae_max, r.pair_id)
except FileNotFoundError:
    B = pd.DataFrame()
pairs["boltz_tier"] = [boltz.get(key(a, b), ("not_run",))[0] for a, b in zip(pairs.a, pairs.b)]
pairs["boltz_iptm"] = [boltz.get(key(a, b), (None, np.nan))[1] for a, b in zip(pairs.a, pairs.b)]
pairs["boltz_ipsae"] = [boltz.get(key(a, b), (None, np.nan, np.nan))[2] for a, b in zip(pairs.a, pairs.b)]
C = pd.read_csv(f"{X}/complex_screen/cap_selex/panel_pairs_capselex.tsv", sep="\t")
C = C[C.call != "motif"]
cap = {}
for r in C.itertuples():
    k = key(r.tf_a, r.tf_b); called = str(r.call).startswith(("1", "2"))
    cap[k] = "called:" + str(r.call) if called else cap.get(k, "tested_not_called")
pairs["capselex"] = [cap.get(key(a, b), "not_tested") for a, b in zip(pairs.a, pairs.b)]
Sp = pd.read_csv(f"{R}/inputs/string_physical_700.tsv", sep="\t")
string = {}
for r in Sp.itertuples():
    a, b = to_panel.get(r.preferredName_A), to_panel.get(r.preferredName_B)
    if a and b and a != b: string[key(a, b)] = max(string.get(key(a, b), 0), r.score)
pairs["string_physical"] = [string.get(key(a, b), np.nan) for a, b in zip(pairs.a, pairs.b)]

def etype(r):
    boltz_hit = r.boltz_tier in ("strong", "weak"); cap_hit = str(r.capselex).startswith("called")
    if r.view1 and boltz_hit: return "complex" + ("" if r.boltz_tier == "strong" else "_weak")
    if r.view1 and cap_hit: return "DNA-guided"
    if r.view1: return "co-localized"
    if r.boltz_tier == "strong" or cap_hit: return "physical-only"
    return "."
pairs["edge_type"] = pairs.apply(etype, axis=1)

# ---- modules --------------------------------------------------------------------------------------------------------------
SEEDS = {"PRC2": ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"], "NuRD": ["MBD3", "RBBP4", "RBBP7"],
         "cohesin_CTCF": ["STAG2", "CTCF"], "MAZ_ZNF143": ["MAZ", "ZNF143"],
         "HP1_ChAHP": ["HP1", "ADNP", "CHAMP1"], "G9a_WIZ": ["WIZ", "ZNF644", "HP1"], "MEF2": ["MEF2A", "MEF2D"],
         "HES1_RBPJ": ["HES1", "RBPJ"]}
v1 = {key(a, b): s for a, b, s in zip(pairs.a, pairs.b, pairs.view1_score)}
mods = []
for name, mem in SEEDS.items():
    ext = []
    for t in tfs:
        if t in mem: continue
        sc = [v1.get(key(t, m), np.nan) for m in mem]
        if np.nanmean(sc) >= 0.90 and np.sum(np.array(sc) >= 0.90) >= max(1, len(mem) / 2): ext.append(t)
    within = [v1.get(key(a, b), np.nan) for a, b in itertools.combinations(mem, 2)]
    mods.append(dict(module=name, source="seed (known complex / STRING physical)", members=",".join(mem),
                     extended_by_cobinding=",".join(ext) or ".", mean_within_view1_pct=np.nanmean(within)))
seeded = set(m for d in mods for m in d["members"].split(",") + ([] if d["extended_by_cobinding"] == "." else d["extended_by_cobinding"].split(",")))
try:
    import igraph as ig, leidenalg as la_
    rest = [t for t in tfs if t not in seeded]
    E = pairs[pairs.view1 & pairs.a.isin(rest) & pairs.b.isin(rest)]
    g = ig.Graph(); g.add_vertices(rest); g.add_edges(list(zip(E.a, E.b))); g.es["weight"] = list(E.view1_score)
    part = la_.find_partition(g, la_.ModularityVertexPartition, weights="weight", seed=1)
    for i, comm in enumerate(part):
        if len(comm) >= 2:
            mem = [rest[j] for j in comm]
            within = [v1.get(key(a, b), np.nan) for a, b in itertools.combinations(mem, 2)]
            mods.append(dict(module=f"leiden_{i + 1}", source="Leiden (view-1 graph, non-seeded TFs)", members=",".join(mem),
                             extended_by_cobinding=".", mean_within_view1_pct=np.nanmean(within)))
except ImportError:
    print("igraph/leidenalg missing: Leiden step skipped")
M = pd.DataFrame(mods)
# a Leiden module made only of high-library TFs reproduces the 25-vs-54 library split, not biology
M["flag"] = ["batch_module (all members in the high-library group)"
             if src.startswith("Leiden") and all(group[t] == "high" for t in mem.split(",")) else "."
             for src, mem in zip(M.source, M.members)]

tag = "frozen" if FREEZE else "provisional"
cols = ["a", "b", "edge_type", "view1", "view1_score", "view1_n_hi", "view1_n_metrics"] + METRICS + [f"{m}_pct" for m in METRICS] + \
       ["gpair", "boltz_tier", "boltz_iptm", "boltz_ipsae", "capselex", "string_physical"]
pairs[cols].to_csv(f"{R}/r1_pairs_{tag}.tsv", sep="\t", index=False, float_format="%.4f")
pairs.loc[pairs.edge_type != ".", cols].to_csv(f"{R}/r1_edges_{tag}.tsv", sep="\t", index=False, float_format="%.4f")
M.to_csv(f"{R}/r1_modules_{tag}.tsv", sep="\t", index=False, float_format="%.3f")
n_boltz = sum(1 for v in boltz.values()); print(f"Boltz pairs available: {n_boltz} of {len(pairs)} panel pairs")
print("edge types:", pairs.edge_type.value_counts().to_dict())
print("view1 edges:", int(pairs.view1.sum()), "| by library pair type:", pairs[pairs.view1].gpair.value_counts().to_dict())
print("STRING-physical pairs recovered by view 1:", int(pairs[pairs.string_physical.notna()].view1.sum()), "of", int(pairs.string_physical.notna().sum()))
chk = [("FOXK1", "MLX"), ("MLX", "MSX2"), ("HOXD10", "SREBF2"), ("MLX", "SREBF2")]
print(pairs[[key(a, b) in [key(*c) for c in chk] for a, b in zip(pairs.a, pairs.b)]][["a", "b", "edge_type", "view1_score"] + [f"{m}_pct" for m in METRICS] + ["boltz_tier", "boltz_ipsae", "capselex"]].round(3).to_string(index=False))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(M.round(3).to_string(index=False))
if FREEZE:
    with open(f"{R}/r1_frozen.sha256", "w") as f:
        for t in ["pairs", "edges", "modules"]:
            p = f"{R}/r1_{t}_frozen.tsv"; f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p}\n")
