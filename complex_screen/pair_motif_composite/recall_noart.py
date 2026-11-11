# Artifact-dropped copy of the gate pools (2026-11-11, Yuyang: drop the Codebook artifact list). As the 2026-10-26 audit
# (motif_analysis/denovo_macs2/audit_artifact_drop.sh): remove sites overlapping an Alu (RepeatMasker, HOMER hg38.repeats) or with a FIMO
# hit p < 1e-5 to one of the 18 Codebook non-endogenous artifact motifs (ACGACG self-annealing, GGAGGG, Alu-fragment, CAC-repeat,
# GGAA-repeat, GACA/GAC), from foreground AND background. Output recall/pools_noart/<variant>/<TF>/<part>/{fg,bg}.bed, recall/noart_counts.tsv
import os, glob, subprocess
import pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; R = f"{X}/complex_screen/pair_motif_composite/recall"; AU = f"{X}/motif_analysis/denovo_macs2/out/audit"
def sh(c): r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash"); assert r.returncode == 0, (c, r.stderr[-2000:]); return r.stdout
fs = sorted(glob.glob(f"{R}/pools/*/*/*/*.bed")); keys = set()
for f in fs:
    for l in open(f): x = l.split("\t"); keys.add((x[0], int(x[1]), int(x[2])))
os.makedirs(f"{R}/noart_tmp", exist_ok=True); u = f"{R}/noart_tmp/unique.bed"
with open(u, "w") as fo:
    for c, s, e in sorted(keys): fo.write(f"{c}\t{s}\t{e}\t{c}:{s}-{e}\n")
alu = set(sh(f"module load bedtools >/dev/null 2>&1; bedtools intersect -u -a {u} -b {AU}/alu.bed | cut -f4").split())
sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -nameOnly -fi {X}/downstream/cache/hg38.fa -bed {u} -fo {R}/noart_tmp/unique.fa")
art = set(sh(f"module load meme/5.5.7 >/dev/null 2>&1; fimo --text --thresh 1e-5 --verbosity 1 {AU}/artdrop/artifacts_drop.meme {R}/noart_tmp/unique.fa | awk 'NR>1{{print $3}}' | sort -u").split())
bad = alu | art; print(len(keys), "unique sites;", len(alu), "Alu;", len(art), "artifact-motif;", len(bad), "dropped", flush=True); rows = []
for f in fs:
    v, t, pt, nm = f.split("/")[-4:]; L = [l for l in open(f)]; keep = [l for l in L if "{}:{}-{}".format(*l.split("\t")[:3]) not in bad]
    d = f"{R}/pools_noart/{v}/{t}/{pt}"; os.makedirs(d, exist_ok=True); open(f"{d}/{nm}", "w").write("".join(keep)); rows.append(dict(variant=v, tf=t, part=pt, set=nm[:-4], n=len(L), kept=len(keep)))
C = pd.DataFrame(rows); C.to_csv(f"{R}/noart_counts.tsv", sep="\t", index=False); C["frac_dropped"] = 1 - C.kept / C.n.clip(lower=1); print(C.groupby(["variant", "set"]).frac_dropped.median().round(3).to_string())
