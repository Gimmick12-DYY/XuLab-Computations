# Gapped motif part for one pair: SpaMo (MEME 5.5.7) with the primary motif = best own DB motif of A (from motif_valid own_gate fisher; else B),
# secondary = the 2,528-motif DB, on co-bound sites (+/- 150 bp) and on the A-only control sites, discovery and held-out chromosomes separately.
import sys, os, subprocess, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; G = f"{O}/composite_gapped"; pair = sys.argv[1]; R = f"{G}/results/{pair}"
P = pd.read_csv(f"{G}/pairs.tsv", sep="\t").set_index("pair"); a, b = P.at[pair, "tf_a"], P.at[pair, "tf_b"]
OW = pd.read_csv(f"{O}/motif_valid/own_gate.tsv", sep="\t"); OW = OW[OW.method == "fisher"].set_index("tf")
prim = [(t, OW.at[t, "best_own"]) for t in (a, b) if t in OW.index]
if not prim: open(f"{R}/spamo_none.txt", "w").write("no own motif for either TF\n"); sys.exit(0)
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash")
for t, m in prim:
    for pt in ["disc", "val"]:
        for st, bed in [("cob", "cob_300.bed"), ("aonly", "aonly.bed")]:
            f = f"{G}/pools/{pair}/{pt}/{bed}"; fa = f"{R}/{t}_{pt}_{st}.fa"
            if os.path.getsize(f) == 0: continue
            sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -fi {X}/downstream/cache/hg38.fa -bed {f} -fo {fa}")
            out = f"{R}/spamo_{t}_{pt}_{st}"
            sh(f"module load meme/5.5.7 >/dev/null 2>&1; spamo --oc {out} --verbosity 1 --text --primary {m} --margin 100 --bgfile {O}/reversed/bins.bg {fa} {O}/db/known.meme {O}/db/known.meme > {out}.tsv 2>/dev/null || spamo --oc {out} --verbosity 1 --primary {m} --margin 100 --bgfile {O}/reversed/bins.bg {fa} {O}/db/known.meme {O}/db/known.meme")
            if os.path.exists(f"{out}/spamo.tsv"): os.replace(f"{out}/spamo.tsv", f"{out}.tsv")
            for x in ["spamo.html", "spamo.xml"]:
                if os.path.exists(f"{out}/{x}"): os.remove(f"{out}/{x}")
            os.remove(fa)
print("spamo done", pair, prim)
