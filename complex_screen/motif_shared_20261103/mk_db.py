# Motif set: JASPAR 2024 CORE vertebrates non-redundant plus, for panel TFs without a JASPAR motif, their Codebook or HOCOMOCO H12CORE motif.
import re
X = "/work/users/d/y/dyy12/XuLab"; D = f"{X}/motif_analysis/cache/motifdb"; O = f"{X}/complex_screen/motif_shared_20261103"; tfs = [t.strip().upper() for t in open(f"{O}/tfs.txt")]; ALT = {"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}
J = open(f"{D}/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt").read(); head, *bj = J.split("MOTIF "); names = {b.split()[0]: b.split("\n")[0].split()[1] if len(b.split("\n")[0].split()) > 1 else "" for b in bj}
def parts(n): return [x.upper() for x in re.split(r"::|\(", n)]
own = {t: [m for m, n in names.items() if any(a in parts(n) for a in [t] + ALT.get(t, []))] for t in tfs}; extra = []; src = {}
def blocks(f): return ["MOTIF " + b for b in open(f).read().split("MOTIF ")[1:]]
CB = blocks(f"{D}/codebook_top1.meme"); H = blocks(f"{D}/H12CORE_meme_format.meme")
for t in tfs:
    if own[t]: src[t] = "JASPAR"; continue
    al = [t] + ALT.get(t, []); hit = [b for b in CB if any(re.search(rf"\b{a}\b", b.split("\n")[0], re.I) for a in al)][:1]; s = "Codebook"
    if not hit:
        h2 = [re.sub(r"^ZNF", "ZN", a) for a in al]; hit = [b for b in H if b.split("\n")[0].split()[1].split(".")[0].upper() in h2][:1]; s = "H12CORE"
    if hit:
        q = hit[0].split("\n"); ids = q[0].split(); q[0] = f"MOTIF x_{t}_{ids[1].replace('|', '_')} {t}"; extra.append("\n".join(q)); own[t] = [q[0].split()[1]]; src[t] = s
    else: src[t] = "none"
open(f"{O}/db/db.meme", "w").write(J.rstrip() + "\n\n" + "\n\n".join(x.rstrip() for x in extra) + "\n"); import json; json.dump(dict(own=own, src=src), open(f"{O}/db/own.json", "w"), indent=0); from collections import Counter; print(Counter(src.values()), [t for t in tfs if src[t] == "none"], "extras", len(extra))
