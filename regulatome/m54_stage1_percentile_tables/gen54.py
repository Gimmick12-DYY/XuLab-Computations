# generate m54 scripts from the m52 ones: within-TF percentile of |score| (m53, full precision) in place of the score cutoff; labels "top X% of genes".
import os, re
SRC = "/private/tmp/claude-501/-Users-dyy-Desktop-Group-Meeting/ad0f2564-9221-41b3-a63a-8d93e5231160/scratchpad/m52/out"
R = "/work/users/d/y/dyy12/XuLab/regulatome"; N = f"{R}/m54_stage1_percentile_tables"
OLDINJ_START = 'SC_ = pd.read_csv('; OLDINJ_END = 'and "non-changing" |z| < 1 as |score| < 0.1\n'
NEWINJ = ('_F = np.load("' + R + '/m53_stage1_percentile/out/paper_score_full_precision.npz", allow_pickle=True); SC_ = pd.DataFrame(_F["score"], index=_F["tfs"].astype(str), columns=_F["genes"].astype(str)).reindex(index=TFS, columns=UG).values.astype(float)\n'
          'assert (np.isfinite(SC_) == np.isfinite(Z)).all(), "score matrix does not cover exactly the z pairs"\n'
          'from scipy.stats import rankdata\n'
          'PA_ = np.full(SC_.shape, np.nan)\n'
          'for i_ in range(len(TFS)):\n'
          '    ok_ = np.isfinite(SC_[i_]); PA_[i_, ok_] = 100 * (rankdata(np.abs(SC_[i_, ok_])) - 0.5) / ok_.sum()\n'
          '# m54: within-TF percentile of |score| mapped so that the m48 cutoffs read as top 10 / 5 / 2 / 1 % (|Z| >= 2 / 3 / 4 / 5) and "non-changing" (|Z| < 1) = bottom 50 %\n'
          'Z = np.sign(SC_) * np.interp(np.nan_to_num(PA_) - 1e-9, [0, 50, 90, 95, 98, 99, 100], [0, 1, 2, 3, 4, 5, 6]); Z[~np.isfinite(SC_)] = np.nan\n'
          'CUTL = {2: 10, 3: 5, 4: 2, 5: 1}; PE = {0: 0, 1: 50, 2: 90, 3: 95, 4: 98, 5: 99}\n'
          'assert all(int((np.abs(Z[i_][np.isfinite(Z[i_])]) >= 2).sum()) in (1122, 1123) for i_ in range(len(TFS)))\n')
def common(s):
    s = s.replace('f"|score| >= {c / 10:g}"', 'f"top {CUTL[c]}% of genes"').replace("m52_stage1_paper_score", "m54_stage1_percentile_tables").replace("m52_tables5k.py", "m54_tables5k.py").replace("m52_tables.py", "m54_tables.py")
    return s
def write(name, s, head):
    s = head + "\n" + s; compile(s, name, "exec"); open(f"out/{name}", "w").write(s)
os.makedirs("out", exist_ok=True)
for f in ("m52_tables.py", "m52_tables5k.py"):
    s = open(f"{SRC}/{f}").read(); a = s.index(OLDINJ_START); b = s.index(OLDINJ_END) + len(OLDINJ_END); s = s[:a] + NEWINJ + s[b:]
    s = s.replace('"|score| bin": f"{lo_ / 10:g}-{hi_ / 10:g}" if np.isfinite(hi_) else f">= {lo_ / 10:g}"', '"percentile bin": f"{PE[lo_]}-{PE[hi_]}" if np.isfinite(hi_) else f"{PE[lo_]}-100"'); assert '"percentile bin"' in s
    s = common(s); s = s.split("\n", 1)[1]; assert "f\"|score|" not in s and "\"|score| bin\"" not in s, f
    write(f.replace("m52", "m54"), s, f"# m54: generated from {f} (m52) by gen54.py: responders = top X % of |score| within the TF (m53 full-precision score).")
for f in ("m52_shuffle_sets.py", "m52_tflabel.py", "m52_47.py", "m52_47_bulk.py", "m52_47_paired.py", "m52_47_coord.py", "m52_47_precision.py", "m52_compare.py"):
    s = common(open(f"{SRC}/{f}").read()).split("\n", 1)[1]; s = s.replace('"(|z| version m40 rule G was 24.74 / 28.26 / 32.95 / 37.46; not expected to match with the score)"', '"(|z| version m40 rule G was 24.74 / 28.26 / 32.95 / 37.46; not expected to match)"')
    assert "|score|" not in s, f; write(f.replace("m52", "m54"), s, f"# m54: generated from {f} (m52) by gen54.py.")
s = open(f"{SRC}/m52_bulk_side.py").read().split("\n", 1)[1].replace("m52_stage1_paper_score", "m54_stage1_percentile_tables")
s = s.replace('("|score| >= 0.2", "|score| >= 0.3", "|score| >= 0.4", "|score| >= 0.5")', '("top 10% of genes", "top 5% of genes", "top 2% of genes", "top 1% of genes")').replace('"|score| ≥ 0.3"', '"top 5% of genes"'); assert "|score|" not in s
write("m54_bulk_side.py", s, "# m54: generated from m52_bulk_side.py by gen54.py (panel at top 5 % per TF).")
s = open(f"{SRC}/m52_padding.py").read().split("\n", 1)[1].replace("m52_stage1_paper_score", "m54_stage1_percentile_tables")
s = s.replace("    for c in (2, 3, 4, 5):  # |score| x 10", "    for c in (2, 3, 4, 5):  # top 10 / 5 / 2 / 1 %").replace('x = g.loc[f"|score| >= {c / 10:g}"]; r[f"|score| ≥ {c / 10:g}"]', 'x = g.loc[f"top {CUTL[c]}% of genes"]; r[f"top {CUTL[c]}%"]')
s = s.replace('"cutoff": f"|score| >= {c / 10:g}"', '"cutoff": f"top {CUTL[c]}% of genes"').replace("(|score| >= cutoff,", "(top X % of |score| within the TF,")
s = s.replace("import pandas as pd\n", "import pandas as pd\nCUTL = {2: 10, 3: 5, 4: 2, 5: 1}\n", 1); assert "|score|" not in s.replace("of |score| within", ""), s[:300]
write("m54_padding.py", s, "# m54: generated from m52_padding.py by gen54.py.")
s = open(f"{SRC}/m52_plots.py").read().split("\n", 1)[1].replace("m52_stage1_paper_score", "m54_stage1_percentile_tables")
rep = [('CUT = ["|score| >= 0.2", "|score| >= 0.3", "|score| >= 0.4", "|score| >= 0.5"]; CL = ["0.2", "0.3", "0.4", "0.5"]', 'CUT = ["top 10% of genes", "top 5% of genes", "top 2% of genes", "top 1% of genes"]; CL = ["10", "5", "2", "1"]'),
       ('BINS = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", ">= 0.5"]; BL = ["0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "≥ 0.5"]', 'BINS = ["0-50", "50-90", "90-95", "95-98", "98-99", "99-100"]; BL = ["0–50", "50–90", "90–95", "95–98", "98–99", "99–100"]'),
       ('set_index("|score| bin")', 'set_index("percentile bin")'), ('"Perturb-seq response, |score| ≥ (Perturb-seq genes, TFs left)"', '"Perturb-seq response, top % of genes per TF (Perturb-seq genes, TFs left)"'),
       ('"Perturb-seq response, |score| ≥"', '"Perturb-seq response, top % of genes per TF"'), ('"Perturb-seq response, |score| bin"', '"Perturb-seq response, within-TF percentile of |score|"'),
       ('"gene changed if |score| ≥ cutoff\\n(own gene removed)"', '"gene changed if in the top %\\nof genes per TF (own gene removed)"'), ('"|score| ≥ 0.2 · 0.3 · 0.4 · 0.5"', '"top 10 · 5 · 2 · 1 % per TF"')]
for a, b in rep: assert a in s, a[:50]; s = s.replace(a, b)
assert "|score| ≥" not in s and "|score| >=" not in s; write("m54_plots.py", s, "# m54: generated from m52_plots.py by gen54.py (percentile cutoffs).")
print(sorted(os.listdir("out")))
