# pair_summary.tsv for a candidate run (BASE): strict / relaxed shared calls (gate_levels.tsv), tier, best validated composite word
# with its Tomtom match against db/known.meme, and the random-pair pass rates from the GC-matched gate run as the empirical null.
import os, sys, re, math, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import summarize as S
O = os.path.dirname(os.path.abspath(__file__)); B = os.environ["BASE"]
L = pd.read_csv(f"{B}/gate_levels.tsv", sep="\t"); PS = pd.read_csv(f"{B}/pool_sizes.tsv", sep="\t").set_index("pair")
R = pd.read_csv(f"{O}/gc_matched/gate_levels.tsv", sep="\t"); R = R[R.category.str.startswith("random")]
rate = dict(random_strict_rate=f"{int(R.strict_pass.sum())}/{len(R)}", random_relaxed_rate=f"{int(R.relaxed_pass.sum())}/{len(R)}", random_composite_rate=f"{int((R.composite_n_validated > 0).sum())}/{len(R)}")
def tomtom(raw, tag):
    lines = raw.strip().split("\n"); m = [l.split() for l in lines[1:] if l.strip()]
    d = f"{B}/tomtom_best/{tag}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/q.meme", "w").write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
                                   f"MOTIF {tag}\nletter-probability matrix: alength= 4 w= {len(m)}\n" + "\n".join(" ".join(r) for r in m) + "\n")
    subprocess.run(f"module load meme/5.5.7 >/dev/null 2>&1; tomtom -no-ssc -verbosity 1 -text -min-overlap 5 -dist pearson -thresh 0.5 {d}/q.meme {O}/db/known.meme > {d}/tomtom.tsv", shell=True, executable="/bin/bash")
    t = pd.read_csv(f"{d}/tomtom.tsv", sep="\t", comment="#")
    if not len(t): return "none"
    t = t.sort_values("q-value").iloc[0]; return f"{t['Target_ID']} q={t['q-value']:.2g}"
rows = []
for _, l in L.iterrows():
    p = l.pair; c = f"{B}/candidates/{p}"; best = ("none", "", np.nan, np.nan)
    if os.path.exists(f"{c}/candidates.tsv") and os.path.getsize(f"{c}/candidates.tsv") > 1:
        T = pd.read_csv(f"{c}/candidates.tsv", sep="\t"); V = S.val(p); VC = {(k.split("-", 1)[1] if re.match(r"^\d+-", str(k)) else k): v for k, v in V.items()}
        ok = []
        for _, x in T.iterrows():
            key = x.motif if x.source == "homer_known" else x.motif.split("|")[0]
            if key in VC and VC[key][0] > 0.75 and VC[key][1] < -2: ok.append((VC[key][1], key, VC[key][0]))
        if ok:
            vp, key, fc = min(ok); blocks = open(f"{c}/homer.motifs").read().split(">")[1:]
            raw = next((">" + b for b in blocks if b.split("\t")[0] == key or b.split("\t")[1] == key), None)
            best = (key, tomtom(raw, p) if raw else "", round(fc, 2), round(vp, 1))
    rows.append(dict(pair=p, tier=PS.loc[p, "category"], n_pair=l.n_pair, own_A=l.own_A, A_fold_vs_partner_only=l.A_fold, A_log10p=l.A_log10p, A_val_log2fc=l.A_val_log2fc, A_val_log10p=l.A_val_log10p,
                     own_B=l.own_B, B_fold_vs_partner_only=l.B_fold, B_log10p=l.B_log10p, B_val_log2fc=l.B_val_log2fc, B_val_log10p=l.B_val_log10p, shared_perm_p=l.shared_perm_p,
                     strict_pass=l.strict_pass, relaxed_pass=l.relaxed_pass, composite_validated=l.composite_n_validated, best_composite_word=best[0], best_composite_val_log2fc=best[2],
                     best_composite_val_log10p=best[3], tomtom_match=best[1], **rate))
Z = pd.DataFrame(rows); Z.to_csv(f"{B}/pair_summary.tsv", sep="\t", index=False); pd.set_option("display.width", 300); print(Z.to_string())
