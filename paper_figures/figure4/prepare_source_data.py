"""Figure 4 source data (panel c, shared enriched motifs). Run on Longleaf after the motif analyses in complex_screen/pair_motif_composite.

Panel c shows, for the three example pairs, three shared motifs each. A motif is shared when it is enriched (p <= 0.05) in BOTH TFs' single-cell
binding sites on BOTH chromosome halves (odd = discovery, even + X = held-out); "specific" excludes motifs enriched in > 10 of 79 TFs (generic)
and HOMER motifs enriched in >= 5 of 10 pooled-Tn5 control libraries. Motifs are ranked by the largest of the four p-values; a pair with fewer
than three specific motifs is filled with the next HOMER motif that is not Tn5 background, flagged generic (dagger on the panel).
Logos are built from our own instances: FIMO (p <= 1e-4) matches of the source motif in both TFs' foreground sites, both halves, best match per
site, counted per position; then trimmed to the informative core (positions with IC >= 0.5, at most one weak position inside).

Writes source_data/c_shared_motifs.tsv (one row per logo), source_data/c_logo_counts.tsv (per logo, per site set both / A / B, and position: A/C/G/T counts; same core trim for all three) and
source_data/MANIFEST.tsv (file, origin, made_by, sha256), including sha256 of every upstream input.
  python prepare_source_data.py
"""
import hashlib
import io
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
OUT = HERE / "source_data"
X = Path("/work/users/d/y/dyy12/XuLab")
PM = X / "complex_screen/pair_motif_composite"
HOMER_LIB = Path("/nas/longleaf/rhel9/apps/homer/5.1/data/knownTFs/vertebrates/known.motifs")
GENOME = X / "downstream/cache/hg38.fa"
PAIRS = ["RBBP4-ZNF644", "ADNP-MTF2", "DR1-NFYA"]
INPUTS = {
    "pair_table": PM / "motif_tables/pair_shared_motifs_p005.tsv",
    "db_motifs": PM / "db/known.meme",
    "fisher_disc": PM / "motif_valid/scores_fisher_disc.tsv",
    "fisher_val": PM / "motif_valid/scores_fisher_val.tsv",
    "homer_library": HOMER_LIB,
}


def sh(cmd, inp=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", input=inp)
    if r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stderr[-800:]}")
    return r.stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_homer():
    hom, cur, rows = {}, None, []
    for line in list(open(HOMER_LIB)) + [">end\tend\t0"]:
        if line.startswith(">"):
            if cur and cur not in hom:
                hom[cur] = np.array(rows)
            cur, rows = line.split("\t")[1].split("/")[0], []
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    return hom


def read_meme():
    db, alt, cur, rows = {}, {}, None, []
    for line in list(open(INPUTS["db_motifs"])) + ["MOTIF __end__"]:
        if line.startswith("MOTIF "):
            if cur and rows:
                db[cur] = np.array(rows)
            x = line.split()
            cur, rows = x[1], []
            alt[cur] = x[2] if len(x) > 2 else x[1]
        elif cur and len(line.split()) == 4 and line.strip()[0] in "0123456789.":
            rows.append([float(v) for v in line.split()])
    return db, alt


def parse(cell):
    out = []
    for x in str(cell).split(";"):
        x = x.strip()
        if x and x != "nan":
            m = re.match(r"^(.*?)(?: \[(.*)\])?$", x)
            out.append((m.group(1).strip(), m.group(2) or ""))
    return out


def trim(P):
    ic = 2 + (P * np.log2(np.maximum(P, 1e-9))).sum(1)
    keep = np.flatnonzero(ic >= 0.5)
    if not len(keep):
        return 0, len(P)
    blocks, st = [], keep[0]
    for a, b in zip(keep[:-1], keep[1:]):
        if b - a > 2:
            blocks.append((st, a + 1))
            st = b
    blocks.append((st, keep[-1] + 1))
    return max(blocks, key=lambda bl: ic[bl[0]:bl[1]].sum())


IUPAC = {frozenset("AG"): "R", frozenset("CT"): "Y", frozenset("GC"): "S", frozenset("AT"): "W", frozenset("GT"): "K", frozenset("AC"): "M"}


def consensus(P):
    s = []
    for r in P:
        o = np.argsort(r)[::-1]
        if r[o[0]] >= 0.6:
            s.append("ACGT"[o[0]])
        elif r[o[0]] + r[o[1]] >= 0.75:
            s.append(IUPAC[frozenset("ACGT"[o[0]] + "ACGT"[o[1]])])
        else:
            s.append("N")
    return "".join(s)


def main():
    OUT.mkdir(exist_ok=True)
    hom = read_homer()
    db, alt = read_meme()
    name2id = {}
    for k, v in alt.items():
        name2id.setdefault(v, k)
        name2id.setdefault(k, k)
    fisher = {pt: pd.read_csv(INPUTS[f"fisher_{pt}"], sep="\t", index_col=0) for pt in ("disc", "val")}
    homer_cache = {}

    def log10p(src, m, t, pt):
        if src == "H":
            if (t, pt) not in homer_cache:
                k = pd.read_csv(PM / f"recall/homer/raw_lab/{t}/{pt}/knownResults.txt", sep="\t")
                s = pd.Series((k.iloc[:, 3] / math.log(10)).values, index=k.iloc[:, 0].str.split("/").str[0])
                homer_cache[(t, pt)] = s[~s.index.duplicated()]
            return float(homer_cache[(t, pt)].get(m, np.nan))
        return float(np.log10(max(norm.sf(fisher[pt].at[name2id[m], t]), 1e-300)))

    T = pd.read_csv(INPUTS["pair_table"], sep="\t").set_index("pair")
    rows, counts, used_inputs = [], [], set()
    for pair in PAIRS:
        a, b = pair.split("-")
        r = T.loc[pair]
        cand = [("H", n, "") for n, _ in parse(r.shared_homer_specific)] + [("D", n, "") for n, _ in parse(r.shared_db_specific)]
        cand = [c for c in cand if (c[0] == "H" and c[1] in hom) or (c[0] == "D" and name2id.get(c[1]) in db)]
        worst = lambda c: max(log10p(c[0], c[1], t, pt) for t in (a, b) for pt in ("disc", "val"))  # noqa: E731
        cand.sort(key=worst)
        if len(cand) < 3:
            have = {c[1] for c in cand}
            extra = [("H", n, tag) for n, tag in parse(r.shared_homer_all) if "Tn5" not in tag and n in hom and n not in have]
            extra.sort(key=worst)
            cand += extra[:3 - len(cand)]
        for rank, (src, m, tag) in enumerate(cand[:3], 1):
            M = hom[m] if src == "H" else db[name2id[m]]
            M = M / M.sum(1, keepdims=True)
            meme = ("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
                    f"MOTIF q\nletter-probability matrix: alength= 4 w= {len(M)}\n" + "\n".join(" ".join(f"{v:.4f}" for v in x) for x in M) + "\n")
            mf = OUT / "_tmp_motif.meme"
            mf.write_text(meme)
            C = np.zeros((len(M), 4))
            CT = {}
            n_inst = 0
            for t in (a, b):
                pool = PM / ("recall/pools/raw_lab" if src == "H" else "recall/pools_noart/raw_lab") / t
                beds = [pool / pt / "fg.bed" for pt in ("disc", "val")]
                used_inputs.update(beds)
                fa = OUT / "_tmp.fa"
                sh(f"module load bedtools >/dev/null 2>&1; bedtools getfasta -fi {GENOME} -bed - -fo {fa}", inp="".join(p.read_text() for p in beds))
                d = pd.read_csv(io.StringIO(sh(f"module load meme/5.5.7 >/dev/null 2>&1; fimo --thresh 1e-4 --text --verbosity 1 {mf} {fa}")), sep="\t")
                d.columns = [c.strip("#").strip() for c in d.columns]
                CT[t] = np.zeros((len(M), 4))
                if len(d):
                    d = d.sort_values("p-value").drop_duplicates("sequence_name")
                    for s_ in d.matched_sequence:
                        s_ = str(s_).upper()
                        if len(s_) == len(M) and set(s_) <= set("ACGT"):
                            for i, ch in enumerate(s_):
                                C[i, "ACGT".index(ch)] += 1
                                CT[t][i, "ACGT".index(ch)] += 1
                            n_inst += 1
                fa.unlink()
            mf.unlink()
            P = (C + 0.25) / (C + 0.25).sum(1, keepdims=True)
            s0, s1 = trim(P)
            lp = {(t, pt): log10p(src, m, t, pt) for t in (a, b) for pt in ("disc", "val")}
            rows.append(dict(pair=pair, rank=rank, consensus=consensus(P[s0:s1]), generic=bool(tag), generic_note=tag,
                             source="HOMER" if src == "H" else "JASPAR/HOCOMOCO v12/Codebook",
                             source_motif=m if src == "H" else alt.get(name2id[m], m), n_instances=n_inst,
                             n_instances_A=int(CT[a].sum(1).max()), n_instances_B=int(CT[b].sum(1).max()), p_max_of_4=10 ** max(lp.values()),
                             **{f"p_{who}_{pt}": 10 ** lp[(t, pt)] for who, t in (("A", a), ("B", b)) for pt in ("disc", "val")}, tf_A=a, tf_B=b))
            for i in range(s0, s1):
                counts.append(dict(pair=pair, rank=rank, sites="both", position=i - s0 + 1, A=C[i, 0], C=C[i, 1], G=C[i, 2], T=C[i, 3]))
                for who, t in (("A", a), ("B", b)):
                    counts.append(dict(pair=pair, rank=rank, sites=who, position=i - s0 + 1, A=CT[t][i, 0], C=CT[t][i, 1], G=CT[t][i, 2], T=CT[t][i, 3]))
    pd.DataFrame(rows).to_csv(OUT / "c_shared_motifs.tsv", sep="\t", index=False)
    pd.DataFrame(counts).to_csv(OUT / "c_logo_counts.tsv", sep="\t", index=False)
    man = [("c_shared_motifs.tsv", "derived", "figure4/prepare_source_data.py"), ("c_logo_counts.tsv", "derived", "figure4/prepare_source_data.py")]
    with open(OUT / "MANIFEST.tsv", "w") as fh:
        fh.write("file\torigin\tmade_by\tsha256\n")
        for f, o, by in man:
            fh.write(f"{f}\t{o}\t{by}\t{sha256(OUT / f)}\n")
        for k, p in INPUTS.items():
            fh.write(f"input:{k}\t{p}\tupstream\t{sha256(p)}\n")
        for p in sorted(used_inputs):
            fh.write(f"input:fg_sites\t{p}\tcomplex_screen/pair_motif_composite/recall_pools.py\t{sha256(p)}\n")
    print(pd.DataFrame(rows)[["pair", "rank", "consensus", "source_motif", "n_instances", "p_max_of_4", "generic"]].to_string(index=False))


if __name__ == "__main__":
    main()
