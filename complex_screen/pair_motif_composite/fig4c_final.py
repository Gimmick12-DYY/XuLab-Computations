# Fig 4c (user spec 2026-11-17): rows RBBP4-ZNF644, ADNP-MTF2, DR1-NFYA; 3 shared motifs per pair; logos built from our own motif instances
# (FIMO p <= 1e-4 hits of the source motif in both TFs' foreground sites, both chromosome halves, aligned on the motif strand), labelled by the
# instance consensus only; one line per logo = largest of the four p-values (2 TFs x discovery / held-out). 183 mm wide, Arial 7 pt, palette C.
# Motif choice: the pair's specific shared motifs (pair_shared_motifs_p005: p <= 0.05 in both TFs on both halves, not generic, not Tn5
# background), ranked by the largest of the four p-values; if a pair has fewer than 3, the next best non-Tn5-background HOMER shared motif is used and
# marked with a dagger (enriched in > 10 of 79 TFs). Output fig4/fig4c_motifs.{svg,png}, fig4/fig4c_motifs_values.tsv
import os, re, sys, math, subprocess
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.textpath import TextPath; from matplotlib.patches import PathPatch; from matplotlib.transforms import Affine2D
from scipy.stats import norm
X = "/work/users/d/y/dyy12/XuLab"; O = f"{X}/complex_screen/pair_motif_composite"; OUT = f"{O}/fig4"; os.makedirs(OUT, exist_ok=True)
for f in os.listdir("/work/users/d/y/dyy12/fonts/arial"): fm.fontManager.addfont(f"/work/users/d/y/dyy12/fonts/arial/{f}")
sys.path.insert(0, f"{X}/figures_paletteC"); import regulome_palette as rp; Pc = rp.use()
plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none"})
FPB = fm.FontProperties(fname="/work/users/d/y/dyy12/fonts/arial/Arial Bold.ttf")
PAIRS = ["RBBP4-ZNF644", "ADNP-MTF2", "DR1-NFYA"]
def sh(c, inp=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash", input=inp); assert r.returncode == 0, (c, r.stderr[-800:]); return r.stdout
hom = {}; cur = None; rows = []
for l in list(open("/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs")) + [">end\tend\t0"]:
    if l.startswith(">"):
        if cur and cur not in hom: hom[cur] = np.array(rows)
        cur = l.split("\t")[1].split("/")[0]; rows = []
    elif l.strip(): rows.append([float(x) for x in l.split()])
db, alt, cur, rows = {}, {}, None, []
for l in list(open(f"{O}/db/known.meme")) + ["MOTIF __end__"]:
    if l.startswith("MOTIF "):
        if cur and rows: db[cur] = np.array(rows)
        x = l.split(); cur = x[1]; alt[cur] = x[2] if len(x) > 2 else x[1]; rows = []
    elif cur and len(l.split()) == 4 and l.strip()[0] in "0123456789.": rows.append([float(v) for v in l.split()])
name2id = {}
for k, v in alt.items(): name2id.setdefault(v, k); name2id.setdefault(k, k)
H = {}
def homer_lp(t, pt, m):
    if (t, pt) not in H:
        k = pd.read_csv(f"{O}/recall/homer/raw_lab/{t}/{pt}/knownResults.txt", sep="\t"); s = pd.Series((k.iloc[:, 3] / math.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0]); H[(t, pt)] = s[~s.index.duplicated()]
    return float(H[(t, pt)].get(m, np.nan))
Fd = pd.read_csv(f"{O}/motif_valid/scores_fisher_disc.tsv", sep="\t", index_col=0); Fv = pd.read_csv(f"{O}/motif_valid/scores_fisher_val.tsv", sep="\t", index_col=0)
def db_lp(t, pt, mid):
    z = (Fd if pt == "disc" else Fv).at[mid, t]; return float(np.log10(max(norm.sf(z), 1e-300)))
def lps(src, m, a, b): return {(t, pt): (homer_lp(t, pt, m) if src == "H" else db_lp(t, pt, name2id[m])) for t in (a, b) for pt in ("disc", "val")}
T = pd.read_csv(f"{O}/motif_tables/pair_shared_motifs_p005.tsv", sep="\t").set_index("pair")
def parse(cell):
    out = []
    for x in str(cell).split(";"):
        x = x.strip()
        if not x or x == "nan": continue
        m = re.match(r"^(.*?)(?: \[(.*)\])?$", x); out.append((m.group(1).strip(), m.group(2) or ""))
    return out
sel = {}
for p in PAIRS:
    a, b = p.split("-"); r = T.loc[p]; cand = []
    for nm, tag in parse(r.shared_homer_specific): cand.append(("H", nm, ""))
    for nm, tag in parse(r.shared_db_specific): cand.append(("D", nm, ""))
    cand = [c for c in cand if (c[0] == "H" and c[1] in hom) or (c[0] == "D" and name2id.get(c[1]) in db)]
    cand.sort(key=lambda c: max(lps(c[0], c[1], a, b).values()))
    if len(cand) < 3:
        have = [c[1] for c in cand]
        extra = [("H", nm, tag) for nm, tag in parse(r.shared_homer_all) if "Tn5" not in tag and nm in hom and nm not in have]
        # database extras are not used: their generic hits are SP/KLF GC-box words that the Codebook list calls endogenous background
        extra.sort(key=lambda c: max(lps(c[0], c[1], a, b).values())); cand += extra[:3 - len(cand)]
    sel[p] = cand[:3]
FG = {}
def fg_fasta(t, src):
    key = (t, src)
    if key in FG: return FG[key]
    base = f"{O}/recall/pools/raw_lab/{t}" if src == "H" else f"{O}/recall/pools_noart/raw_lab/{t}"
    bed = "".join(open(f"{base}/{pt}/fg.bed").read() for pt in ("disc", "val")); fn = f"{OUT}/tmp_{t}_{src}.fa"
    sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -fi {X}/downstream/cache/hg38.fa -bed - -fo {fn}", inp=bed); FG[key] = fn; return fn
def meme_of(M):
    M = np.asarray(M, float); M = M / M.sum(1, keepdims=True)
    return "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\nMOTIF q\nletter-probability matrix: alength= 4 w= %d\n" % len(M) + "\n".join(" ".join(f"{v:.4f}" for v in r) for r in M) + "\n"
def instances(src, m, a, b):
    M = hom[m] if src == "H" else db[name2id[m]]; mf = f"{OUT}/tmp_motif.meme"; open(mf, "w").write(meme_of(M)); seqs = []
    for t in (a, b):
        out = sh(f"module load meme/5.5.7 >/dev/null 2>&1; fimo --thresh 1e-4 --text --verbosity 1 {mf} {fg_fasta(t, src)}")
        d = pd.read_csv(pd.io.common.StringIO(out), sep="\t"); d.columns = [c.strip("#").strip() for c in d.columns]
        if not len(d): continue
        d = d.sort_values("p-value").drop_duplicates("sequence_name"); seqs += [s.upper() for s in d.matched_sequence if isinstance(s, str) and set(s.upper()) <= set("ACGT")]
    C = np.zeros((len(M), 4)); idx = {c: i for i, c in enumerate("ACGT")}
    for s in seqs:
        if len(s) != len(M): continue
        for i, ch in enumerate(s): C[i, idx[ch]] += 1
    return C, len(seqs)
IUPAC = {frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("GC"): "S", frozenset("AT"): "W", frozenset("GT"): "K", frozenset("AC"): "M"}
def trim_ic(P):
    ic = 2 + (P * np.log2(np.maximum(P, 1e-9))).sum(1); keep = np.flatnonzero(ic >= .5)
    return (keep[0], keep[-1] + 1) if len(keep) else (0, len(P))
def consensus(P):
    out = []
    for r in P:
        o = np.argsort(r)[::-1]
        if r[o[0]] >= .6: out.append("ACGT"[o[0]])
        elif r[o[0]] + r[o[1]] >= .75: out.append(IUPAC[frozenset("ACGT"[o[0]] + "ACGT"[o[1]])])
        else: out.append("N")
    return "".join(out)
COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}
def logo(ax, P):
    ic = 2 + (P * np.log2(np.maximum(P, 1e-9))).sum(1)
    for i, (row, h) in enumerate(zip(P, ic)):
        y = 0
        for k in np.argsort(row):
            hh = row[k] * h
            if hh < .01: continue
            tp = TextPath((0, 0), "ACGT"[k], size=1, prop=FPB); bb = tp.get_extents(); tr = Affine2D().translate(-bb.x0, -bb.y0).scale(.9 / bb.width, hh / bb.height).translate(i + .05, y)
            ax.add_patch(PathPatch(tr.transform_path(tp), fc=COL["ACGT"[k]], lw=0)); y += hh
    ax.set_xlim(0, len(P)); ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_yticks([0, 1, 2]); ax.spines["bottom"].set_visible(False); ax.set_ylabel("bits", labelpad=1)
mm = 1 / 25.4; fig = plt.figure(figsize=(183 * mm, 100 * mm)); gs = fig.add_gridspec(3, 3, left=.13, right=.99, top=.93, bottom=.06, hspace=1.0, wspace=.32); vals = []
for i, p in enumerate(PAIRS):
    a, b = p.split("-")
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        if j == 0: ax.text(-.36, .5, p, transform=ax.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
        if j >= len(sel[p]): ax.axis("off"); continue
        src, m, tag = sel[p][j]; C, n = instances(src, m, a, b); Pm = (C + .25) / (C + .25).sum(1, keepdims=True); s0, s1 = trim_ic(Pm); Pm = Pm[s0:s1]
        lp = lps(src, m, a, b); pmax = 10 ** max(lp.values()); cons = consensus(Pm); dag = "†" if tag else ""
        logo(ax, Pm); ax.set_title(cons + dag, pad=2)
        ax.text(.5, -.13, f"p ≤ {pmax:.2g}", transform=ax.transAxes, ha="center", va="top")
        vals.append(dict(pair=p, consensus=cons, source_motif=m if src == "H" else alt.get(name2id[m], m), source="HOMER" if src == "H" else "JASPAR/HOCOMOCO/Codebook", generic_flag=tag,
                         n_instances=n, max_p_of_4=pmax, **{f"p_{t}_{pt}": 10 ** lp[(t, pt)] for t in (a, b) for pt in ("disc", "val")}))
fig.savefig(f"{OUT}/fig4c_motifs.svg"); fig.savefig(f"{OUT}/fig4c_motifs.png", dpi=600)
# Illustrator reads the CSS font shorthand "700 7px" as a size; write explicit properties instead (same fix as Fig 3, commit 715c1a5)
sv = open(f"{OUT}/fig4c_motifs.svg").read()
sv = re.sub(r"font: (?:(\d{3}) )?([\d.]+)px '([^']+)'", lambda m: f"font-family: '{m.group(3)}'; font-size: {m.group(2)}px" + (f"; font-weight: {m.group(1)}" if m.group(1) else ""), sv)
open(f"{OUT}/fig4c_motifs.svg", "w").write(sv)
V = pd.DataFrame(vals); V.to_csv(f"{OUT}/fig4c_motifs_values.tsv", sep="\t", index=False); pd.set_option("display.width", 250); print(V.to_string(index=False))
for f in os.listdir(OUT):
    if f.startswith("tmp_"): os.remove(f"{OUT}/{f}")
