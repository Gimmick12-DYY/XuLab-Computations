# m45 Amendment 2 (SPEC.md): bulk ChIP (HEK293T / HEK293 only) vs our HITAG, per TF, padding 0 and 10 kb, matched gene shuffle.
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; M = f"{R}/m45_stage1_summary"
s = open(f"{M}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])
s40 = open(f"{R}/m40_overlap_reference/m40_perm.py").read(); a = s40.index("D33 = "); exec(s40[a:s40.index("T25 = pd.read_csv", a)])  # CH, BULK: A = padding 0, G = A + 10 kb
O = f"{M}/out"; rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1); ti = {t: i for i, t in enumerate(TFS)}
REF = {"CTCF": "HEK293T, GSE103651 (ChIP-nexus)", "ZNF282": "HEK293T, GSM2466515 (HA-tag ChIP-exo)", "MAZ": "HEK293, ENCODE ENCFF529XTF", "ZIC2": "HEK293, ENCODE ENCFF359GIY",
       "ZNF777": "HEK293, ENCODE ENCFF735CAY", "ZBTB7A": "HEK293, ENCODE ENCFF114DGV", "SETDB1": "HEK293, ENCODE reads ENCFF000ZTR (cell line from file name)"}
TF7 = [t for t, _ in CH if t in REF]; assert len(TF7) == 7 and all(t in ti for t in TF7)
def one(bl, i, c):
    s_ = STRAT[OK[i]]; b = bl[OK[i]]; p_ = (AZ[i] >= c)[OK[i]]; dr = np.zeros(NP)
    for st in np.unique(s_):
        m = s_ == st; n = int(m.sum()); nb = int(b[m].sum()); k = int(p_[m].sum())
        if k and nb: dr += rng.hypergeometric(nb, n - nb, k, size=NP)
    n = int(p_.sum()); o = int((b & p_).sum()); dd = 100 * dr / max(n, 1)
    return {"genes in list": int(b.sum()), "in both": o, "recall (%)": 100 * o / n if n else np.nan, "shuffle mean (%)": dd.mean() if n else np.nan, "shuffle 2.5-97.5 (%)": f"{np.percentile(dd,2.5):.1f}-{np.percentile(dd,97.5):.1f}" if n else "", "p (one-sided)": pval(100 * o / n, dd) if n else np.nan}
rows = []
for pad, bk in (("0", "A"), ("10 kb", "G")):
    for t in TF7:
        i = ti[t]
        for c in (2, 3, 4, 5):
            n = int((OK[i] & (AZ[i] >= c)).sum()); base = {"padding": pad, "TF": t, "bulk ChIP reference": REF[t], "cutoff": f"|z| >= {c}", "Perturb-seq genes": n, "n < 10": n < 10}
            rows.append({**base, **{f"bulk ChIP: {k}": v for k, v in one(BULK[bk][t], i, c).items()}, **{f"our HITAG: {k}": v for k, v in one(LISTS[pad][t], i, c).items()}})
T = pd.DataFrame(rows); T.to_csv(f"{O}/11_bulk_chip_hek_per_tf.tsv", sep="\t", index=False, float_format="%.4g"); T.to_excel(f"{O}/11_bulk_chip_hek_per_tf.xlsx", index=False)
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 22)
print(T.drop(columns=["bulk ChIP reference"]).round(2).to_string(index=False)); print("done")
