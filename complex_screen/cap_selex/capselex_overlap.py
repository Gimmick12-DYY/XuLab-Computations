# Overlap of our 79-TF panel with the CAP-SELEX screen (Xie et al. 2025): constructs (S1), interaction matrix (S2),
# composite / spacing motifs (S4, CAP-SELEX rows). Writes panel_in_capselex.tsv and panel_pairs_capselex.tsv
import itertools, re, sys
import pandas as pd
raw, panel_file, out = sys.argv[1:4]
ALIAS = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1", "ZUFSP"]}
panel = [l.strip().upper() for l in open(panel_file) if l.strip()]
sym = {g: p for p in panel for g in ALIAS.get(p, [p])}

s1 = pd.read_excel(f"{raw}/MOESM3.xlsx", header=None)
hdr = [i for i in range(len(s1)) if str(s1.iloc[i, 0]).strip().startswith("Ensembl")]
cons = set()
for h in hdr:
    for i in range(h + 1, len(s1)):
        v = s1.iloc[i, 1]
        if pd.isna(v) or str(s1.iloc[i, 0]).startswith(("Section", "SELECTION")): break
        cons.add(str(v).strip().upper())
s2 = pd.read_excel(f"{raw}/MOESM4.xlsx", header=None)
r0 = next(i for i in range(len(s2)) if str(s2.iloc[i, 0]).startswith("Prey"))
M = s2.iloc[r0 + 1:].copy(); M.columns = ["prey"] + [str(c).strip().upper() for c in s2.iloc[r0, 1:]]
M = M[M.prey.notna()]; M["prey"] = M.prey.astype(str).str.strip().str.upper(); M = M.set_index("prey")
M = M[[c for c in M.columns if c != "NAN" and not c.startswith("UNNAMED")]]
baits, preys = set(M.columns), set(M.index)
s4 = pd.read_excel(f"{raw}/MOESM6.xlsx", header=None)
r4 = next(i for i in range(5, len(s4)) if str(s4.iloc[i, 0]) == "ID")
S4 = s4.iloc[r4 + 1:, :4]; S4.columns = ["ID", "family", "study", "experiment"]
cap = S4[S4.experiment.astype(str).str.contains("CAP")].copy()
cap[["tf1", "tf2"]] = cap.ID.str.split("_", n=2, expand=True)[[0, 1]].apply(lambda s: s.str.upper())
ht = set(S4[S4.experiment.astype(str).str.contains("HT-SELEX|Methyl")].ID.str.split("_").str[0].str.upper())

rows = []
for p in panel:
    gs = ALIAS.get(p, [p])
    rows.append(dict(panel_tf=p, symbols=",".join(gs), construct=any(g in cons for g in gs),
                     as_bait=any(g in baits for g in gs), as_prey=any(g in preys for g in gs),
                     n_tested_partners=int(sum(M[g].notna().sum() if g in baits else 0 for g in gs) + sum(M.loc[g].notna().sum() if g in preys else 0 for g in gs)),
                     n_interactions=int(sum(M[g].astype(str).str.match(r"^[12]").sum() if g in baits else 0 for g in gs) + sum(M.loc[g].astype(str).str.match(r"^[12]").sum() if g in preys else 0 for g in gs)),
                     n_cap_motifs=int(((cap.tf1.isin(gs)) | (cap.tf2.isin(gs))).sum()), ht_selex_motif=any(g in ht for g in gs)))
P = pd.DataFrame(rows); P.to_csv(f"{out}/panel_in_capselex.tsv", sep="\t", index=False)
pr = []
for a, b in itertools.combinations(panel, 2):
    for ga in ALIAS.get(a, [a]):
        for gb in ALIAS.get(b, [b]):
            for x, y in [(ga, gb), (gb, ga)]:
                if y in M.columns and x in M.index and pd.notna(M.loc[x, y]):
                    pr.append(dict(tf_a=a, tf_b=b, prey=x, bait=y, call=M.loc[x, y]))
    mot = cap[((cap.tf1.isin(ALIAS.get(a, [a]))) & (cap.tf2.isin(ALIAS.get(b, [b])))) | ((cap.tf2.isin(ALIAS.get(a, [a]))) & (cap.tf1.isin(ALIAS.get(b, [b]))))]
    if len(mot): pr.append(dict(tf_a=a, tf_b=b, prey=".", bait=".", call="motif", motifs=";".join(mot.ID)))
R = pd.DataFrame(pr)
R.to_csv(f"{out}/panel_pairs_capselex.tsv", sep="\t", index=False)
pd.set_option("display.width", 220)
print(f"CAP-SELEX: {len(cons)} construct symbols, matrix {M.shape[0]} preys x {M.shape[1]} baits, {len(cap)} CAP-SELEX motifs")
print(f"panel TFs with a construct: {P.construct.sum()}/79; in the interaction matrix: {(P.as_bait | P.as_prey).sum()}/79; with any interaction call: {(P.n_interactions > 0).sum()}; with a CAP-SELEX motif: {(P.n_cap_motifs > 0).sum()}")
print(P[(P.as_bait | P.as_prey | P.construct)].to_string(index=False))
print("\npanel-panel pairs tested:", 0 if R.empty else len(R[R.call != "motif"]))
print(R.to_string(index=False) if len(R) else "none")
