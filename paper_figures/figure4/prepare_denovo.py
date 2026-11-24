"""Figure 4c de novo comparison (source data). For each shared motif of panel c (source_data/c_shared_motifs.tsv), find each TF's own de novo motif
that matches it, then compare TF A's and TF B's de novo motifs directly.

De novo motifs: HOMER de novo (lengths 8, 10, 12) run separately in each TF's sites, with no source motif (complex_screen/pair_motif_composite/
per_tf_raw: top 5,000 lab MACS2 summits in open chromatin, odd chromosomes, matched open-chromatin background); motifs with fold >= 1.5 and
p <= 1e-10 kept. For each TF, the best de novo match to the shared motif = lowest Tomtom q-value (Pearson, min overlap 5). A vs B similarity =
Tomtom of TF A's matched de novo motif against TF B's (p and q against a single target are reported as Tomtom gives them; E-value too).

Writes source_data/c_denovo_pairs.tsv (one row per shared motif: matched de novo motif per TF, their HOMER p and fold, Tomtom q to the shared
motif, Tomtom A-vs-B p / E / q, offset, orientation) and source_data/c_denovo_matrices.tsv (probability matrices of the matched de novo motifs),
and appends both files to MANIFEST.tsv.
  python prepare_denovo.py
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
HOMER_LIB = Path("/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs")


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


def homer_denovo(tf):
    out = []
    for f in sorted(DENOVO.glob(f"{tf}/L*/homerMotifs.all.motifs")):
        for blk in f.read_text().split(">")[1:]:
            lines = blk.strip().split("\n")
            h = lines[0].split("\t")
            t = float(re.search(r"T:[\d.]+\(([\d.]+)%", h[5]).group(1))
            b = float(re.search(r"B:[\d.]+\(([\d.]+)%", h[5]).group(1))
            log10p = float(h[3]) / math.log(10)
            fold = (t + 0.01) / (b + 0.01)
            if fold >= 1.5 and log10p <= -10:
                out.append(dict(id=f"{tf}_{f.parent.name}_{len(out)}", cons=h[0], log10p=log10p, fold=fold,
                                M=np.array([[float(x) for x in l.split()] for l in lines[1:] if l.strip()])))
    return out


def tomtom(qfile, tfile):
    txt = sh(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 1 -evalue {qfile} {tfile}")
    d = pd.read_csv(io.StringIO(txt), sep="\t", comment="#")
    return d


def main():
    S = pd.read_csv(SRC / "c_shared_motifs.tsv", sep="\t")
    hom = {}
    cur, rows = None, []
    for line in list(open(HOMER_LIB)) + [">end\tend\t0"]:
        if line.startswith(">"):
            if cur and cur not in hom:
                hom[cur] = np.array(rows)
            cur, rows = line.split("\t")[1].split("/")[0], []
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    db, alt, cur, rows = {}, {}, None, []
    for line in list(open(PM / "db/known.meme")) + ["MOTIF __end__"]:
        if line.startswith("MOTIF "):
            if cur and rows:
                db[cur] = np.array(rows)
            x = line.split()
            cur, rows = x[1], []
            alt[cur] = x[2] if len(x) > 2 else x[1]
        elif cur and len(line.split()) == 4 and line.strip()[0] in "0123456789.":
            rows.append([float(v) for v in line.split()])
    byname = {}
    for k, v in alt.items():
        byname.setdefault(v, k)
        byname.setdefault(k, k)
    tmp = SRC / "_tmp"
    tmp.mkdir(exist_ok=True)
    dn = {}
    for tf in sorted(set(S.tf_A) | set(S.tf_B)):
        dn[tf] = homer_denovo(tf)
        (tmp / f"{tf}.meme").write_text(meme([(m["id"], m["M"]) for m in dn[tf]]))
    out, mats = [], []
    for r in S.itertuples():
        src_M = hom[r.source_motif] if r.source == "HOMER" else db[byname[r.source_motif]]
        (tmp / "q.meme").write_text(meme([("shared", src_M)]))
        row = dict(pair=r.pair, rank=r.rank, consensus=r.consensus, tf_A=r.tf_A, tf_B=r.tf_B)
        best = {}
        for who, tf in (("A", r.tf_A), ("B", r.tf_B)):
            d = tomtom(tmp / "q.meme", tmp / f"{tf}.meme")
            if len(d):
                b = d.sort_values(["q-value", "p-value"]).iloc[0]
                m = next(x for x in dn[tf] if x["id"] == b.Target_ID)
                best[who] = m
                row.update({f"denovo_{who}": m["cons"], f"denovo_{who}_log10p": round(m["log10p"], 1), f"denovo_{who}_fold": round(m["fold"], 2),
                            f"denovo_{who}_q_vs_shared": float(b["q-value"]), f"denovo_{who}_matched": bool(b["q-value"] <= 0.05)})
                for i, v in enumerate(m["M"]):
                    mats.append(dict(pair=r.pair, rank=r.rank, tf=who, position=i + 1, A=v[0], C=v[1], G=v[2], T=v[3]))
            else:
                row[f"denovo_{who}_matched"] = False
        if "A" in best and "B" in best:
            (tmp / "a.meme").write_text(meme([("A", best["A"]["M"])]))
            (tmp / "b.meme").write_text(meme([("B", best["B"]["M"])]))
            d = tomtom(tmp / "a.meme", tmp / "b.meme")
            if len(d):
                b = d.iloc[0]
                row.update(tomtom_AB_p=float(b["p-value"]), tomtom_AB_E=float(b["E-value"]), tomtom_AB_q=float(b["q-value"]),
                           tomtom_AB_offset=int(b["Optimal_offset"]), tomtom_AB_orientation=b["Orientation"])
        out.append(row)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    O = pd.DataFrame(out)
    O.to_csv(SRC / "c_denovo_pairs.tsv", sep="\t", index=False)
    pd.DataFrame(mats).to_csv(SRC / "c_denovo_matrices.tsv", sep="\t", index=False)
    man = SRC / "MANIFEST.tsv"
    lines = [l for l in man.read_text().splitlines() if not l.startswith(("c_denovo_pairs.tsv", "c_denovo_matrices.tsv", "input:denovo_"))]
    for f in ("c_denovo_pairs.tsv", "c_denovo_matrices.tsv"):
        lines.append(f"{f}\tderived\tfigure4/prepare_denovo.py\t{sha256(SRC / f)}")
    for tf in sorted(dn):
        for f in sorted(DENOVO.glob(f"{tf}/L*/homerMotifs.all.motifs")):
            lines.append(f"input:denovo_{tf}\t{f}\tcomplex_screen/pair_motif_composite/run_tf.sbatch (HOMER de novo)\t{sha256(f)}")
    man.write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 250)
    print(O.to_string(index=False))


if __name__ == "__main__":
    main()
