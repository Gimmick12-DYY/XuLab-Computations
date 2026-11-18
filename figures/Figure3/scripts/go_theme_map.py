"""Map significant GO BP terms (checks/go_great_regions/per_TF_q005.tsv) to broad themes through GO slim generic (user 2026-11-18).
Slim = BP terms tagged 'subset: goslim_generic' in data/go/go-basic.obo (data-version releases/2026-07-26; no extra download). Plus GO:0006950
response to stress as one extra anchor (not in the slim). Each term is mapped to its most specific slim ancestor via is_a + part_of +
regulates (regulation terms map to the process they regulate) (slim ancestors that are ancestors of another slim ancestor are dropped;
ties -> the slim term with the fewest annotated genes, reported). Slim terms are merged into themes by the fixed table SLIM2THEME below,
written out with the result so the grouping is auditable. Terms without a slim ancestor -> "Other".
Writes per_TF_q005_themes.tsv, slim_to_theme.tsv, theme_ties.tsv.
  python scripts/go_theme_map.py <go_great_regions dir> [input tsv, default per_TF_q005.tsv]   (Longleaf)"""
import gzip, re, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
X = Path("/work/users/d/y/dyy12/XuLab"); D = X / "data/go"; G = Path(sys.argv[1]); IN = sys.argv[2] if len(sys.argv) > 2 else "per_TF_q005.tsv"
TH = {"Development / differentiation": "0048856 0030154 0042060 0022414",
      "Immune / inflammation": "0002376 0006954 0098542",
      "Organ physiology (muscle, heart, kidney, gut)": "0003012 0003013 0003014 0003016 0022600 0050886 0061007",
      "Neural": "0050877",
      "Metabolism": "0005975 0006091 0006520 0006575 0006629 0006766 0006790 0055086 1901135 0009101 0071941 0098754",
      "Cell cycle / DNA / chromatin": "0000278 0000910 0006260 0006281 0006310 0007059 0140013 0140014 0032200 0006325",
      "Protein folding / stress": "0006457 0030163 0006914 0051604 0065003",
      "Gene expression / RNA": "0006351 0006355 0016071 0016073 0031047 0002181 0042254 0006399 0140053",
      "Signalling": "0023052",
      "Cell organization / transport / adhesion": "0007005 0007010 0007018 0007031 0007040 0007155 0007163 0016192 0030198 0034330 0044782 0048870 0055085 0061024 0072659 0006886 0006913 0071554",
      "Other": "0012501 0015979"}
SLIM2THEME = {f"GO:{i}": th for th, ids in TH.items() for i in ids.split()}
par, name, ns, slim, cur = defaultdict(set), {}, {}, set(), None
for ln in open(D / "go-basic.obo"):
    ln = ln.strip()
    if ln == "[Term]": cur = None; continue
    if ln.startswith("id: GO:"): cur = ln[4:]
    elif cur and ln.startswith("name: "): name[cur] = ln[6:]
    elif cur and ln.startswith("namespace: "): ns[cur] = ln[11:]
    elif cur and ln == "subset: goslim_generic": slim.add(cur)
    elif cur and ln.startswith("is_a: "): par[cur].add(ln[6:16])
    elif cur and ln.startswith("relationship: part_of "): par[cur].add(ln[22:32])
    elif cur and re.match(r"relationship: (positively_|negatively_)?regulates GO:", ln): par[cur].add(re.search(r"GO:\d{7}", ln).group(0))
slim = {t for t in slim if ns.get(t) == "biological_process"} | {"GO:0006950"}   # + response to stress (not in GO slim generic): stress terms
SLIM2THEME["GO:0006950"] = "Protein folding / stress"
missing = slim - set(SLIM2THEME); assert not missing, f"slim terms without a theme: {missing}"
anc = {}; sys.setrecursionlimit(10000)
def up(t):
    if t not in anc:
        s = {t}
        for p in par.get(t, ()): s |= up(p)
        anc[t] = s
    return anc[t]
ng = defaultdict(set)
for ln in gzip.open(D / "goa_human.gaf.gz", "rt"):
    if ln[0] == "!": continue
    x = ln.split("\t")
    if "NOT" in x[3] or x[8] != "P": continue
    for t in up(x[4]) & slim: ng[t].add(x[2])
S = pd.read_csv(G / IN, sep="\t"); ties = []
def m(t):
    sa = up(t) & slim
    if not sa: return "", "Other"
    spec = [a for a in sa if not any(a != b and a in up(b) for b in sa)]
    spec.sort(key=lambda a: len(ng[a]))
    if len(spec) > 1: ties.append(dict(term=t, name=name.get(t, ""), candidates=";".join(f"{a} {name[a]} ({len(ng[a])} genes)" for a in spec), chosen=spec[0]))
    return spec[0], SLIM2THEME[spec[0]]
mp = {t: m(t) for t in S.term.unique()}
S["slim"] = S.term.map(lambda t: mp[t][0]); S["slim_name"] = S.slim.map(lambda t: name.get(t, "")); S["theme"] = S.term.map(lambda t: mp[t][1])
S.to_csv(G / IN.replace(".tsv", "_themes.tsv"), sep="\t", index=False)
pd.DataFrame([(t, name[t], SLIM2THEME[t], len(ng[t])) for t in sorted(slim)], columns=["slim_term", "slim_name", "theme", "annotated_genes"]).to_csv(G / "slim_to_theme.tsv", sep="\t", index=False)
pd.DataFrame(ties).drop_duplicates("term").to_csv(G / "theme_ties.tsv", sep="\t", index=False)
print("slim BP terms", len(slim), "; ties", len({d["term"] for d in ties}))
print(S.groupby(["group", "theme"]).size().unstack(0).fillna(0).astype(int))
