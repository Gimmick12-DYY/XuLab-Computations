"""Figure 4c, de novo version (source data). For each example pair, the shared de novo words: HOMER de novo motifs found separately in each TF's
sites with no source motif (complex_screen/pair_motif_composite/per_tf_raw: top 5,000 lab MACS2 summits in open chromatin, odd chromosomes,
matched open-chromatin background, lengths 8/10/12; kept if fold >= 1.5 and p <= 1e-10), compared all TF A motifs x all TF B motifs with Tomtom
(Pearson, min overlap 5). Two selections of up to three A-B word pairs (lowest Tomtom q <= 0.05, each motif used once): "all", and "specific" = both words in de novo families found in <= 10 of 79 TFs (family = Tomtom q <= 0.05 to a per_tf_raw family representative). Each word is also annotated with
how many of the 79 TFs have a de novo motif in the same family (per_tf_raw/denovo_families.tsv), as a generic flag (> 10 TFs).

Writes source_data/c_denovo_words.tsv and source_data/c_denovo_words_matrices.tsv, and appends both to MANIFEST.tsv.
  python prepare_denovo_words.py
"""
import hashlib
import io
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "source_data"
PM = Path("/work/users/d/y/dyy12/XuLab/complex_screen/pair_motif_composite")
DENOVO = PM / "per_tf_raw/results"
PAIRS = ["RBBP4-ZNF644", "ADNP-MTF2", "DR1-NFYA"]


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stderr[-800:]}")
    return r.stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def meme(motifs):
    s = "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
    for name, M in motifs:
        M = np.asarray(M, float)
        M = M / M.sum(1, keepdims=True)
        s += f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {len(M)}\n" + "\n".join(" ".join(f"{v:.4f}" for v in r) for r in M) + "\n\n"
    return s


def denovo(tf):
    out = []
    for f in sorted(DENOVO.glob(f"{tf}/L*/homerMotifs.all.motifs")):
        for blk in f.read_text().split(">")[1:]:
            lines = blk.strip().split("\n")
            h = lines[0].split("\t")
            t = float(re.search(r"T:[\d.]+\(([\d.]+)%", h[5]).group(1))
            b = float(re.search(r"B:[\d.]+\(([\d.]+)%", h[5]).group(1))
            lp, fold = float(h[3]) / math.log(10), (t + 0.01) / (b + 0.01)
            if fold >= 1.5 and lp <= -10:
                out.append(dict(id=f"{tf}_{len(out)}", cons=h[0], log10p=lp, fold=fold, pct_sites=t, pct_bg=b,
                                M=np.array([[float(x) for x in l.split()] for l in lines[1:] if l.strip()])))
    return out


def family_size(cons_list):
    """n TFs whose de novo motifs fall in the same family as a consensus (exact consensus lookup in the family table, else NaN)."""
    F = pd.read_csv(PM / "per_tf_raw/denovo_families.tsv", sep="\t")
    return F


def main():
    tmp = SRC / "_tmp"
    tmp.mkdir(exist_ok=True)
    rows, mats = [], []
    FAM = pd.read_csv(PM / "per_tf_raw/denovo_families.tsv", sep="\t")
    for pair in PAIRS:
        a, b = pair.split("-")
        A, B = denovo(a), denovo(b)
        (tmp / "a.meme").write_text(meme([(m["id"], m["M"]) for m in A]))
        (tmp / "b.meme").write_text(meme([(m["id"], m["M"]) for m in B]))
        txt = sh(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 1 -evalue {tmp}/a.meme {tmp}/b.meme")
        d = pd.read_csv(io.StringIO(txt), sep="\t", comment="#").sort_values(["q-value", "p-value"])
        fam = {}
        for who, L in (("a", A), ("b", B)):
            tt = sh(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.05 {tmp}/{who}.meme {PM}/per_tf_raw/denovo_cmp/reps.meme")
            t2 = pd.read_csv(io.StringIO(tt), sep="\t", comment="#")
            if len(t2):
                t2 = t2.sort_values("q-value").drop_duplicates("Query_ID")
                for q, tg in zip(t2.Query_ID, t2.Target_ID): fam[q] = int(FAM.set_index("family").n_tf.get(int(tg.replace("fam", "")), 0))
        for sel in ("all", "specific"):
          used_a, used_b, k = set(), set(), 0
          Ai, Bi = {m["id"]: m for m in A}, {m["id"]: m for m in B}
          for r in d.itertuples():
              if r.Query_ID in used_a or r.Target_ID in used_b:
                  continue
              if r._6 > 0.05:
                  break
              if sel == "specific" and (fam.get(r.Query_ID, 0) > 10 or fam.get(r.Target_ID, 0) > 10):
                  continue
              ma, mb = Ai[r.Query_ID], Bi[r.Target_ID]
              # drop near-duplicate words already shown (same consensus core in A)
              used_a.add(r.Query_ID)
              used_b.add(r.Target_ID)
              k += 1
              rows.append(dict(selection=sel, pair=pair, rank=k, family_TFs_A=fam.get(r.Query_ID, 0), family_TFs_B=fam.get(r.Target_ID, 0), tf_A=a, tf_B=b, word_A=ma["cons"], word_B=mb["cons"], tomtom_q=r._6, tomtom_p=r._4, tomtom_E=r._5,
                               offset=int(r.Optimal_offset), orientation=r.Orientation,
                               A_log10p=round(ma["log10p"], 1), A_fold=round(ma["fold"], 2), A_pct_sites=ma["pct_sites"], A_pct_bg=ma["pct_bg"],
                               B_log10p=round(mb["log10p"], 1), B_fold=round(mb["fold"], 2), B_pct_sites=mb["pct_sites"], B_pct_bg=mb["pct_bg"],
                               n_A=len(A), n_B=len(B)))
              for who, m in (("A", ma), ("B", mb)):
                  for i, v in enumerate(m["M"]):
                      mats.append(dict(selection=sel, pair=pair, rank=k, tf=who, position=i + 1, A=v[0], C=v[1], G=v[2], T=v[3]))
              if k == 3:
                  break
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    R = pd.DataFrame(rows)
    R.to_csv(SRC / "c_denovo_words.tsv", sep="\t", index=False)
    pd.DataFrame(mats).to_csv(SRC / "c_denovo_words_matrices.tsv", sep="\t", index=False)
    man = SRC / "MANIFEST.tsv"
    lines = [l for l in man.read_text().splitlines() if not l.startswith(("c_denovo_words.tsv", "c_denovo_words_matrices.tsv"))]
    for f in ("c_denovo_words.tsv", "c_denovo_words_matrices.tsv"):
        lines.append(f"{f}\tderived\tfigure4/prepare_denovo_words.py\t{sha256(SRC / f)}")
    man.write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(R[["selection", "pair", "rank", "family_TFs_A", "family_TFs_B", "word_A", "word_B", "tomtom_q", "A_log10p", "A_fold", "B_log10p", "B_fold", "n_A", "n_B"]].to_string(index=False))


if __name__ == "__main__":
    main()
