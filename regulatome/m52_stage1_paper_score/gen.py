# generate m52 scripts from the m48 / m47 sources: paper per-gene score (x 10) in place of z, labels rewritten, new output folders.
import re, os
R = "/work/users/d/y/dyy12/XuLab/regulatome"; N = f"{R}/m52_stage1_paper_score"
SCORE = f"{R}/m51_edistance/out/paper_score_tf_by_gene.tsv.gz"
INJ = ('SC_ = pd.read_csv("' + SCORE + '", sep="\\t", index_col=0).reindex(index=TFS, columns=UG).values.astype(float)\n'
       'assert (np.isfinite(SC_) == np.isfinite(Z)).all(), "score matrix does not cover exactly the z pairs"\n'
       'Z = 10 * SC_  # m52: paper per-gene score x 10, so the m48 cutoffs 2 / 3 / 4 / 5 read as |score| 0.2 / 0.3 / 0.4 / 0.5 and "non-changing" |z| < 1 as |score| < 0.1\n')
def common(s):
    s = s.replace('f"|z| >= {c}"', 'f"|score| >= {c / 10:g}"').replace("m48_stage1_top10k", "m52_stage1_paper_score")
    return s
def write(name, s, head):
    open(f"out/{name}", "w").write(head + "\n" + s); compile(head + "\n" + s, name, "exec")
os.makedirs("out", exist_ok=True)
# tables (top 10,000) and the top 5,000 version
t = open("src/m48_tables.py").read(); t = common(t)
a = "OK = np.isfinite(Z); AZ = np.abs(np.nan_to_num(Z))"; assert a in t; t = t.replace(a, INJ + a)
t = t.replace('"|z| bin": f"{lo_}-{hi_}" if np.isfinite(hi_) else f">= {lo_}"', '"|score| bin": f"{lo_ / 10:g}-{hi_ / 10:g}" if np.isfinite(hi_) else f">= {lo_ / 10:g}"'); assert '"|score| bin"' in t
write("m52_tables.py", t, "# m52 (SPEC.md): m48 Stage 1 tables with the paper per-gene score (m51) as the Perturb-seq response; generated from m48_tables.py by gen.py.")
t5 = t.replace("NPEAK = 10000", "NPEAK = 5000").replace('O = f"{R}/m52_stage1_paper_score/out"', 'O = f"{R}/m52_stage1_paper_score/out5k"'); assert "NPEAK = 5000" in t5 and "out5k" in t5
write("m52_tables5k.py", t5, "# m52: the same tables with the top 5,000 peaks per TF (m45 setting), output out5k.")
# shuffle sets (+ broad only), TF label shuffle
s = common(open("src/m48_shuffle_sets.py").read()); a = '"without both, top-10k grouping": EX3 | BROAD10}'; assert a in s
s = s.replace(a, '"without both, top-10k grouping": EX3 | BROAD10, "broad TFs only (8)": set(TFS) - BROAD}'); s = s.replace('exec(s[:s.index("VERS = {")])', 'exec(s[:s.index("VERS = {")])').replace('s = open(f"{M}/m48_tables.py")', 's = open(f"{M}/m52_tables.py")')
assert 'm52_tables.py' in s; write("m52_shuffle_sets.py", s, "# m52: generated from m48_shuffle_sets.py (score responders; broad-only set added).")
s = common(open("src/m48_tflabel.py").read()).replace('s = open(f"{M}/m48_tables.py")', 's = open(f"{M}/m52_tables.py")'); assert 'm52_tables.py' in s; write("m52_tflabel.py", s, "# m52: generated from m48_tflabel.py.")
# m47 family: top 5,000 prefix, output out47
for f in ("m47", "m47_bulk", "m47_paired", "m47_coord", "m47_precision"):
    s = common(open(f"src/{f}.py").read()); s = s.replace('s = open(f"{M45}/m45_tables.py")', f's = open("{N}/m52_tables5k.py")'); assert "m52_tables5k.py" in s, f
    s = s.replace('O = f"{R}/m47_stage1_peak_set_size/out"', f'O = "{N}/out47"'); assert 'out47' in s, f
    s = s.replace('"(m40 rule G: 24.74 / 28.26 / 32.95 / 37.46)"', '"(|z| version m40 rule G was 24.74 / 28.26 / 32.95 / 37.46; not expected to match with the score)"')
    write(f.replace("m47", "m52_47") + ".py", s, f"# m52: generated from {f}.py (score responders; prefix m52_tables5k.py; output out47).")
# bulk side table + S6
s = common(open("src/m48_bulk_side.py").read()).replace('M47 = f"{R}/m47_stage1_peak_set_size/out"', f'M47 = "{N}/out47"')
s = s.replace('("|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5")', '("|score| >= 0.2", "|score| >= 0.3", "|score| >= 0.4", "|score| >= 0.5")').replace('"|z| ≥ 3"', '"|score| ≥ 0.3"'); assert "|z|" not in s and "out47" in s
write("m52_bulk_side.py", s, "# m52: generated from m48_bulk_side.py.")
# padding table
s = common(open("src/m48_padding.py").read()).replace('I = f"{R}/m47_stage1_peak_set_size/out"', f'I = "{N}/out47"').replace('r[f"|z| ≥ {c}"]', 'r[f"|score| ≥ {c / 10:g}"]').replace("(|z| >= cutoff,", "(|score| >= cutoff,")
s = s.replace("for c in (2, 3, 4, 5):", "for c in (2, 3, 4, 5):  # |score| x 10")
assert "|z|" not in s; write("m52_padding.py", s, "# m52: generated from m48_padding.py (padding table 0 / 10 / 50 kb, top 10,000, from out47 table 1).")
# compare workbook
s = open("src/m48_compare.py").read().replace('A = f"{R}/m45_stage1_summary/out"; B = f"{R}/m48_stage1_top10k/out"', f'A = "{N}/out5k"; B = "{N}/out"')
s = s.replace('for n in ("8_tf_label_shuffle", "regulome_v4_summary"): sheets[n] = rd(B, n)', 'for n in ("8_tf_label_shuffle",): sheets[n] = rd(B, n)').replace('sheets["9_counts_per_cutoff"] = rd(A, "9_counts_per_cutoff")', 'sheets["9_counts_per_cutoff"] = rd(B, "9_counts_per_cutoff")')
s = s.replace('f"{R}/m47_stage1_peak_set_size/out/3_precision.tsv"', f'"{N}/out47/3_precision.tsv"').replace('"precision (m47)"', '"precision"'); assert "m45_stage1" not in s and "m47_stage1" not in s and "m48_stage1" not in s
write("m52_compare.py", s, "# m52: generated from m48_compare.py (top 5,000 = out5k beside top 10,000 = out).")
# plots S1-S5
s = common(open("src/m48_plots.py").read())
rep = [('CUT = ["|z| >= 2", "|z| >= 3", "|z| >= 4", "|z| >= 5"]; CL = ["|z| ≥ 2", "|z| ≥ 3", "|z| ≥ 4", "|z| ≥ 5"]', 'CUT = ["|score| >= 0.2", "|score| >= 0.3", "|score| >= 0.4", "|score| >= 0.5"]; CL = ["0.2", "0.3", "0.4", "0.5"]'),
       ('BINS = ["0-1", "1-2", "2-3", "3-4", "4-5", ">= 5"]; BL = ["0–1", "1–2", "2–3", "3–4", "4–5", "≥ 5"]', 'BINS = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", ">= 0.5"]; BL = ["0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", "≥ 0.5"]'),
       ('set_index("|z| bin")', 'set_index("|score| bin")'), ('"Perturb-seq cutoff (Perturb-seq genes, TFs left)"', '"Perturb-seq response, |score| ≥ (Perturb-seq genes, TFs left)"'), ('"Perturb-seq cutoff"', '"Perturb-seq response, |score| ≥"'),
       ('"knockdown response, |z| bin"', '"Perturb-seq response, |score| bin"'), ('"top-25% sgRNA cells\\nz per TF × gene"', '"top-25% sgRNA cells\\nper-gene score per TF × gene"'),
       ('"gene changed if |z| ≥ cutoff\\n(own gene removed)"', '"gene changed if |score| ≥ cutoff\\n(own gene removed)"'), ('dial(35, yp - 7.4, "|z| ≥ 2 · 3 · 4 · 5")', 'dial(35, yp - 7.4, "|score| ≥ 0.2 · 0.3 · 0.4 · 0.5")'),
       ('    ax.text(1, 56, "Two dials only (highlighted); everything else fixed", fontsize=sf, color=DARK, va="center")\n', '')]
for a, b in rep: assert a in s, a; s = s.replace(a, b)
assert "|z|" not in s; write("m52_plots.py", s, "# m52: generated from m48_plots.py (score cutoffs; S1 text changed from |z| to |score|).")
print("ok", sorted(os.listdir("out")))
