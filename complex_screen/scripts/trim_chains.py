# Trim each panel protein to its ordered segments (AFDB pLDDT >= 50), always keeping DNA-binding domains,
# annotated domains/motifs and known interaction motifs; cap at ~1000 aa. Writes chains.tsv + chains.fasta.
import json, sys, re
import numpy as np
import pandas as pd

seq_dir, out = sys.argv[1:3]
PLDDT, GAP_FILL, MIN_SEG, PAD, CAP, LINKER = 50, 30, 15, 3, 1000, "GSGSGSGSGS"
FORCE_TYPES = {"Domain", "Zinc finger", "DNA binding", "Motif", "Coiled coil"}
REGION_KEEP = re.compile(r"interact|binding|zipper|required for|homodimer|PxVxL|LxCxE", re.I)

# Manual decisions (uniprot coordinates, 1-based inclusive) with reasons
FORCE = {  # extra regions kept regardless of pLDDT
    "CTCF": [(180, 265, "N-term YDF motif (~aa 226) binds SA2-SCC1; pilot construct")],
    "ADNP": [(740, 900, "homeobox + PxVxL (aa ~820) HP1-binding motif; pilot construct")],
    "WIZ": [(1440, 1651, "C-terminal ZF region reported to bind G9a/GLP; pilot construct")],
}
MANUAL = {  # whole-construct domain selection, overrides the automatic trimming
    "SRCAP": ([(110, 240), (590, 1010), (1894, 2213), (2852, 2874), (2931, 2953), (2999, 3021)],
              "no AFDB model (3230 aa); kept HSA (124-196) + Snf2 ATPase lobe 1 (Helicase ATP-binding 630-795) + lobe 2 "
              "(1894-2213, Helicase C 2044-2197, bounded by UniProt disordered regions) + 3 AT-hooks (DNA binding); dropped the "
              "lobe insertion (1011-1893, RUVBL/ZNHIT1 scaffold) and disordered N/C termini"),
    "DNMT1": ([(646, 1600)],
              "1616 aa; kept CXXC (DNA binding, 646-692) + autoinhibitory linker + BAH1 + BAH2 + MTase (to 1600) as one contiguous "
              "catalytic core; dropped N-term 1-645 (DMAP1-binding, PCNA PIP box 163-174, RFTS 400-533) to meet the cap"),
    "STAG2": ([(80, 1060)], "1231 aa; SA2 HEAT core 80-1060 (pilot construct; contains the CES pocket bound by the CTCF YDF motif and "
                            "the SCC1-binding surface); dropped disordered N-term and C-term 1061-1231"),
    "SETDB1": ([(15, 111), (147, 435), (557, 880), (1150, 1291)],
               "1291 aa; kept N-term coiled coil (ATF7IP binding), triple Tudor, MBD, pre-SET and the SET domain halves; dropped the "
               "disordered SET insertion 881-1149 (AFDB pLDDT ~32; UniProt disordered 868-1160) that the SET annotation had forced in"),
    "ZBTB40": ([(1, 119), (250, 683), (760, 1165)],
               "1239 aa; kept BTB (24-87), the ordered middle region 250-683 (pLDDT ~75, no annotation) and all 12 C2H2 ZFs (807-1158); "
               "dropped low-pLDDT linkers and the C-terminal tail 1166-1239 (pLDDT <50) to meet the cap"),
    "SREBF2": ([(1, 480)], "1141 aa; nuclear (cleaved) form 1-480 with TAD + bHLH-ZIP; transmembrane/SCAP-binding part is not nuclear"),
}

def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]: j += 1
            out.append([i, j]); i = j + 1
        else: i += 1
    return out

def features(d):
    keep = []
    for x in d.get("features", []):
        t, desc = x["type"], x.get("description", "")
        s, e = x["location"]["start"].get("value"), x["location"]["end"].get("value")
        if s is None or e is None: continue
        if t in FORCE_TYPES or (t == "Region" and REGION_KEEP.search(desc) and e - s + 1 <= 60):
            keep.append((s, e, f"{t}:{desc}" if desc else t))
    return keep

prot = pd.concat([pd.read_csv(f"{seq_dir}/proteins.tsv", sep="\t")])
rows, fa = [], []
for _, r in prot.iterrows():
    g = r.gene
    d = json.load(open(f"{seq_dir}/uniprot/{g}.json"))
    seq = d["sequence"]["value"]; L = len(seq)
    feats = features(d)
    if g in MANUAL:
        segs, note = MANUAL[g]; segs = [list(x) for x in segs]; method = "manual"
    else:
        af = json.load(open(f"{seq_dir}/afdb/{g}.json"))
        pl = np.array(af["plddt"], float)
        mask = pl >= PLDDT
        for s, e, _ in feats + [(a, b, c) for a, b, c in FORCE.get(g, [])]:
            mask[max(0, s - 1 - PAD):min(L, e + PAD)] = True
        # fill short gaps between kept runs, then drop tiny isolated ordered islands that carry no forced feature
        rr = runs(mask)
        for (a0, a1), (b0, b1) in zip(rr, rr[1:]):
            if b0 - a1 - 1 <= GAP_FILL: mask[a1 + 1:b0] = True
        forced = np.zeros(L, bool)
        for s, e, _ in feats + list(FORCE.get(g, [])): forced[s - 1:e] = True
        segs = [[a + 1, b + 1] for a, b in runs(mask) if b - a + 1 >= MIN_SEG or forced[a:b + 1].any()]
        note = "auto: AFDB pLDDT>=50, gaps<=30 filled, UniProt domains/ZF/DNA-binding/motifs forced"
        if g in FORCE: note += "; forced " + "; ".join(f"{a}-{b} ({c})" for a, b, c in FORCE[g])
        method = "auto"
    chain = LINKER.join(seq[s - 1:e] for s, e in segs)
    kept = sum(e - s + 1 for s, e in segs)
    # features fully dropped (sanity check: DNA-binding/ZF/Domain must survive)
    lost = [f"{c} {s}-{e}" for s, e, c in feats if not any(a <= s and e <= b for a, b in segs)]
    rows.append(dict(chain_id=g, panel_name=r.panel_name, uniprot=r.uniprot, full_length=L, kept_residues=kept,
                     chain_length=len(chain), n_segments=len(segs), segments=",".join(f"{s}-{e}" for s, e in segs),
                     method=method, over_cap=len(chain) > CAP, dropped_features="; ".join(lost) or ".", note=note))
    fa.append(f">{g}\n{chain}\n")

df = pd.DataFrame(rows)
df.to_csv(f"{out}/chains.tsv", sep="\t", index=False)
open(f"{out}/chains.fasta", "w").write("".join(fa))
print(df[["chain_id", "full_length", "chain_length", "n_segments", "method", "over_cap"]].to_string())
print("total residues", df.chain_length.sum(), "over cap:", list(df.chain_id[df.over_cap]))
print(df[df.dropped_features != "."][["chain_id", "dropped_features"]].to_string())
