# Peak re-calling gate summary: own-motif rank (HOMER known library) per TF and variant, discovery and held-out.
# Own motif = HOMER motifs whose factor name matches the TF (case-insensitive, also across '|' / ':' / '-' parts), plus family entries for
# TFs whose exact factor is not in the library. Pass (per TF) = rank <= 10 on both disc and val; loose = rank <= 25 on both.
# Output recall/gate_own_motif.tsv, recall/gate_summary.tsv
import os, re, glob
import numpy as np, pandas as pd
import os
R = "/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite/recall"; HD = os.environ.get("HOMERDIR", "homer"); SUF = os.environ.get("SUFFIX", "")
FAM = {"MLX": r"^(Usf[12]|Max|c-Myc|n-Myc|CLOCK|NPAS2|MNT|bHLHE40)\(", "TBX3": r"^Tbx[235]\(", "HES1": r"^(HEY[12]?|Hes1?|HEY)\(", "SREBF2": r"^SREBP?[12]?\(", "E2F3": r"^E2F[0-9]*\(",
       "HOXA10": r"^Hox[abcd](9|10|11|13)\(", "HOXB9": r"^Hox[abcd](9|10|11|13)\(", "HOXC9": r"^Hox[abcd](9|10|11|13)\(", "HOXD10": r"^Hox[abcd](9|10|11|13)\(", "HOXA5": r"^Hox[abcd][1-8]\(",
       "HOXB6": r"^Hox[abcd][1-8]\(", "HOXB8": r"^Hox[abcd][1-8]\(", "HOXC4": r"^Hox[abcd][1-8]\(", "MSX2": r"^Msx[12]?\(", "EPAS1": r"^(HIF2a|HIF-1a|HIF1b|NPAS)", "RELA": r"^(p65|NFkB-p65)",
       "DDIT3": r"^(CHOP|Chop|Atf4|CEBP:AP1)", "ETS2": r"^(ETS|Ets1|ETS1|GABPA|ELF|Fli1|ERG)\(", "MEF2A": r"^Mef2[abcd]\(", "MEF2D": r"^Mef2[abcd]\(", "FOXC1": r"^Fox[a-z][0-9]*\(", "FOXJ3": r"^Fox[a-z][0-9]*\(",
       "FOXK1": r"^FOXK[12]\(", "SOX4": r"^Sox[0-9]+\(", "SOX12": r"^Sox[0-9]+\(", "NEUROD4": r"^(NeuroD1|NeuroG2|Atoh1|Olig2|Ascl1)\(", "RBPJ": r"^RBPJ", "ZIC2": r"^Zic[0-9]?\(", "NR1D2": r"^(Rev-Erb|RORgt|RORa|Erra)",
       "ZBTB7A": r"^ZBTB7", "NFYA": r"^NFY\(", "NRF1": r"^NRF1?\(", "ZNF143": r"ZNF143", "MAZ": r"^Maz\(", "CTCF": r"^CTCF\(", "E4F1": r"^E4F1?\(", "ZNF768": r"^ZNF768", "ZNF282": r"^ZNF282"}
rows = []
for f in sorted(glob.glob(f"{R}/{HD}/*/*/*/knownResults.txt")):
    v, t, pt = f.split("/")[-4:-1]; k = pd.read_csv(f, sep="\t"); k.columns = ["name", "cons", "p", "logp", "q", "nt", "pt", "nb", "pb"]
    fac = k.name.str.split("(").str[0]; parts = fac.str.upper().str.split(r"[|:/\-]")
    m = parts.apply(lambda L: t in L)
    if t in FAM: m |= k.name.str.contains(FAM[t], regex=True)
    if not m.any(): continue
    j = int(np.flatnonzero(m.values)[0]); b = k.iloc[j]; fo = float(str(b.pt).rstrip("%")) / max(float(str(b.pb).rstrip("%")), .01)
    rows.append(dict(variant=v, tf=t, part=pt, own=b["name"].split("/")[0], rank=j + 1, fold=round(fo, 2), log10p=round(b.logp / np.log(10), 1), top=k.name.iloc[0].split("/")[0]))
D = pd.DataFrame(rows); W = D.pivot_table(index=["tf", "variant"], columns="part", values="rank").reset_index()
W = W.merge(D[D.part == "disc"][["tf", "variant", "own", "fold", "log10p", "top"]], on=["tf", "variant"], how="left"); W.to_csv(f"{R}/gate_own_motif{SUF}.tsv", sep="\t", index=False)
S = W.groupby("variant").apply(lambda g: pd.Series(dict(n_tf=len(g), strict_le10=int(((g.disc <= 10) & (g.val <= 10)).sum()), loose_le25=int(((g.disc <= 25) & (g.val <= 25)).sum()),
     median_rank_disc=g.disc.median(), tfs_strict=",".join(sorted(g.tf[(g.disc <= 10) & (g.val <= 10)]))))).reset_index()
S.to_csv(f"{R}/gate_summary{SUF}.tsv", sep="\t", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200); print(S.to_string(index=False))
P = W.pivot_table(index="tf", columns="variant", values="disc"); P2 = W.pivot_table(index="tf", columns="variant", values="val"); print((P.astype("Int64").astype(str) + "/" + P2.astype("Int64").astype(str)).to_string())
