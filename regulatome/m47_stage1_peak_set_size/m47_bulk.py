# m47 Amendment 2: HEK293T / HEK293 bulk ChIP (top 5,000 / top 10,000 / full) beside our HITAG (top 5,000 / top 10,000), per TF, Stage 1 setting.
import numpy as np, pandas as pd
R = "/work/users/d/y/dyy12/XuLab/regulatome"; X = "/work/users/d/y/dyy12/XuLab"; M45 = f"{R}/m45_stage1_summary"
s = open(f"{M45}/m45_tables.py").read(); exec(s[:s.index("VERS = {")])
O = f"{R}/m47_stage1_peak_set_size/out"  # set after the exec, which defines its own O
D = f"{X}/data"
REF = {"CTCF": ("CTCF_HEK293T_GSE103651_peaks.bed", "HEK293T, GSE103651 ChIP-nexus"), "ZNF282": ("ZNF282_HEK293_peaks.bed", "HEK293T, GSM2466515 HA-tag ChIP-exo"), "MAZ": ("MAZ_HEK293_peaks.bed", "HEK293, ENCODE ENCFF529XTF"),
       "ZIC2": ("ZIC2_HEK293_peaks.bed", "HEK293, ENCODE ENCFF359GIY"), "ZNF777": ("ZNF777_HEK293_peaks.bed", "HEK293, ENCODE ENCFF735CAY"), "ZBTB7A": ("ZBTB7A_HEK293_peaks.bed", "HEK293, ENCODE ENCFF114DGV"),
       "SETDB1": ("SETDB1_HEK293_peaks.bed", "HEK293, ENCODE reads ENCFF000ZTR (cell line from file name)")}
ti = {t: i for i, t in enumerate(TFS)}; print("TFs with a usable top-25% knockdown:", [t for t in REF if t in ti], flush=True)
def lists(path, ns):
    P = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 6, 9], names=["c", "s", "sig", "off"]); P["rk"] = P.sig.rank(ascending=False, method="first"); out = {}
    for nm, n in ns:
        x = P if n is None else P[P.rk <= n]; c5, p5 = x.c.values.astype(str), (x.s + x.off).values; A, _ = rule_abc(c5, p5)
        out[nm] = ({p: (A | window(c5, p5, w) if w else A) for p, w in PADS.items()}, len(x))
    return out, len(P)
rng = np.random.default_rng(20261118); NP = 1000; pval = lambda o, d: (1 + int((np.asarray(d) >= o).sum())) / (len(d) + 1); rows = []
for t, (f, cell) in REF.items():
    if t not in ti: continue
    i = ti[t]; bl, nb_total = lists(f"{D}/{f}", [("bulk ChIP top 5,000", 5000), ("bulk ChIP top 10,000", 10000), ("bulk ChIP full list", None)])
    ol, no_total = lists(f"{V}/macs2/TF.{t}/TF.{t}_peaks.narrowPeak", [("our HITAG top 5,000", 5000), ("our HITAG top 10,000", 10000)])
    for nm, (L, npk) in {**bl, **ol}.items():
        for p in PADS:
            b = L[p][OK[i]]; s_ = STRAT[OK[i]]; ours = int(b.sum())
            for c in (2, 3, 4, 5):
                p_ = (AZ[i] >= c)[OK[i]]; n = int(p_.sum()); ob = int((b & p_).sum()); dr = np.zeros(NP)
                for st in np.unique(s_):
                    m = s_ == st; nn = int(m.sum()); k_ = int(b[m].sum()); kk = int(p_[m].sum())
                    if kk and k_: dr += rng.hypergeometric(k_, nn - k_, kk, size=NP)
                sm = dr.mean()
                rows.append({"TF": t, "list": nm, "reference / cell line": cell if nm.startswith("bulk") else "our HITAG, HEK293T", "peaks used": npk, "peaks in file": nb_total if nm.startswith("bulk") else no_total, "padding": p, "cutoff": f"|z| >= {c}",
                             "linked genes": ours, "Perturb-seq genes": n, "n < 10": n < 10, "in both": ob, "recall (%)": 100 * ob / n if n else np.nan, "shuffle mean recall (%)": 100 * sm / n if n else np.nan, "recall gap (points)": 100 * (ob - sm) / n if n else np.nan,
                             "precision observed (%)": 100 * ob / ours if ours else np.nan, "precision shuffle (%)": 100 * sm / ours if ours else np.nan, "observed / shuffle": ob / sm if sm else np.nan, "p (one-sided)": pval(ob, dr) if n else np.nan})
    print(t, flush=True)
T = pd.DataFrame(rows); T.to_csv(f"{O}/4_bulk_vs_ours_per_tf.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(f"{O}/stage1_bulk_vs_ours_per_tf.xlsx") as w:
    for p in PADS: T[T.padding == p].to_excel(w, sheet_name=f"padding {p}", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 600)
print(T[T.padding == "10 kb"][["TF", "list", "peaks used", "cutoff", "linked genes", "Perturb-seq genes", "in both", "recall (%)", "recall gap (points)", "precision observed (%)", "observed / shuffle", "p (one-sided)"]].round(3).to_string(index=False)); print("done")
