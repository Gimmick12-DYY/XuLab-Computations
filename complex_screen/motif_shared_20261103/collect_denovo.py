# Pools de novo motifs (this run's STREME, E <= 0.05; top 5 per TF and run from earlier STREME and HOMER runs on imputed, pmat and MACS2 peaks) into one MEME file; motif ids carry run and TF.
import glob, json, os, re
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/motif_shared_20261103"; M = f"{X}/motif_analysis"; tfs = [t.strip() for t in open(f"{O}/tfs.txt")]; H = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"; out = []; n = {}
def streme(f, run, t, cap=99):
    for b in open(f).read().split("MOTIF ")[1:cap + 1]:
        L = b.split("\n"); m = re.search(r"E= *([0-9.e+-]+)", L[1]); w = int(re.search(r"w= *(\d+)", L[1]).group(1))
        if m and float(m.group(1)) > .05: continue
        out.append(f"MOTIF {run}|{t}|{L[0].split()[0]}\nletter-probability matrix: alength= 4 w= {w}\n" + "\n".join(L[2:2 + w]) + "\n"); n[run] = n.get(run, 0) + 1
def homer(f, run, t, cap=99):
    bs = [b.strip().split("\n") for b in open(f).read().split(">")[1:]]; bs = sorted(bs, key=lambda L: float(L[0].split("\t")[3]))[:cap]
    for L in bs:
        h = L[0].split("\t")
        if float(h[3]) > -23: continue
        out.append(f"MOTIF {run}|{t}|{h[0]}\nletter-probability matrix: alength= 4 w= {len(L) - 1}\n" + "\n".join(L[1:]) + "\n"); n[run] = n.get(run, 0) + 1
for t in tfs:
    streme(f"{O}/streme/{t}/streme.txt", "new", t)
    for run, base in (("motif_imputed", f"{M}/motif_imputed/{t}"), ("motif_pmat", f"{M}/motif_pmat/{t}"), ("denovo_macs2", f"{M}/denovo_macs2/out/denovo/{t.upper()}")):
        for f in glob.glob(f"{base}/streme/streme.txt") + glob.glob(f"{base}/*/streme/streme.txt"): streme(f, run, t, 5)
        for f in glob.glob(f"{base}/homer/homerMotifs.all.motifs") + glob.glob(f"{base}/*/homer/homerMotifs.all.motifs"): homer(f, run + "_homer", t, 5)
open(f"{O}/db/denovo_top5.meme", "w").write(H + "\n".join(out)); print(n, len(out)); print("macs2 layout", sorted(os.listdir(f"{M}/denovo_macs2"))[:6])
