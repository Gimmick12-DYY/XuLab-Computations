# Triple-support table (correlation x AF3 x motif) from existing results only; no new motif runs, no GPU.
# Universe: pairs with AF3 strong/weak (1-seed sweep scores_s1 + earlier AF3 runs) and best-of-five pct >= 0.60, plus reference positives.
# Writes complex_screen/triple_support_20261109/triple_support.tsv and set_background.tsv. usage: build_triple.py
import glob, os
import numpy as np, pandas as pd
X = "/work/users/d/y/dyy12/XuLab"; C = f"{X}/complex_screen"; O = f"{C}/af3_corr060_20261104"; OUT = f"{C}/triple_support_20261109"; os.makedirs(OUT, exist_ok=True)
assert not os.path.exists(f"{OUT}/triple_support.tsv"), "output exists"
ALIAS = {"CBX1": "HP1", "CBX3": "HP1", "CBX5": "HP1", "ZUP1": "ZUFSP"}; pn = lambda t: ALIAS.get(t.upper(), t.upper()); key = lambda a, b: tuple(sorted((pn(a), pn(b))))
REF = [key(*p) for p in (("EZH2", "MTF2"), ("MAZ", "ZNF143"), ("CTCF", "STAG2"), ("MLX", "SREBF2"), ("RBBP4", "RBBP7"))]
rk = {"strong": 2, "weak": 1, "none": 0}
# --- correlation
MATS = {"all": "pct_all", "noquies": "pct_noquies", "tss": "pct_tss", "enh": "pct_enhancer_biv", "rep": "pct_repressive"}
def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
M = {k: rd(f"{O}/align_corr_20261109/inputs/{v}.tsv") for k, v in MATS.items()}
R = pd.read_csv(f"{X}/regulatome/r1_tftf/v2_final_v3/r1_pairs_frozen_v3.tsv", sep="\t", keep_default_na=False, na_values=[""]); RD = {key(a, b): r for (a, b), (_, r) in zip(zip(R.a, R.b), R.iterrows())}
# --- AF3: 1-seed (sweep, earlier 1-seed) and 5-seed (earlier) kept apart
a1, a5 = {}, {}
for f in glob.glob(f"{O}/scores_s1/*.tsv"):
    x = open(f).read().rstrip("\n").split("\t"); n = os.path.basename(f)[:-4]; u, v = n.split("_", 1); a1[key(u, v)] = dict(tier=x[4], ipsae=float(x[2]), iptm=float(x[1]), src="sweep_s1", run_name=n)
for f in ("af3_local/af3_scores.tsv", "af3_novel_20261103/af3_scores.tsv", "af3_novel_20261103/af3_scores_weak.tsv", "af3_novel_20261103/af3_scores_srcap.tsv"):
    for r in pd.read_csv(f"{C}/{f}", sep="\t").itertuples():
        u, v = r.pair.split("__"); k = key(u, v); d = dict(tier=r.tier, ipsae=float(r.ipsae_max), iptm=float(r.iptm_top), src=f.split("/")[0], run_name=r.pair, seeds_strong=int(r.seeds_strong))
        tgt = a5 if int(r.n_models) >= 25 else a1
        if k not in tgt or rk[d["tier"]] > rk[tgt[k]["tier"]]: tgt[k] = d
def af3(k):
    s5 = a5.get(k); s1 = a1.get(k); best = s5 if s5 else s1; return best, s1, s5
# --- Boltz trimmed (het rows) and full-length check
Sc = pd.read_csv(f"{C}/screen/scores.tsv", sep="\t", keep_default_na=False); Sc = Sc[Sc["set"] == "het"]; BZ = {key(*p.split("__")): {"strong": "strong", "weak": "weak"}.get(h, "none") for p, h in zip(Sc.pair_id, Sc.hit)}
FL = pd.read_csv(f"{C}/fulllength_check/fulllength_scores.tsv", sep="\t", keep_default_na=False); FLt = {key(*p.split("__")): t for p, t in zip(FL.pair, FL.full_tier)}
# --- motif m1 / m2 from TF1000cells_motifs.xlsx
xl = pd.ExcelFile(f"{X}/motif_analysis/TF1000cells_motifs.xlsx")
H = set(xl.parse("HOCOMOCOv11_full_annotation_HUM")["Transcription factor"].astype(str).str.upper())
KM = xl.parse("known_motifs"); KM = KM[KM.TF.notna()]; KM["TF"] = KM.TF.astype(str).str.upper(); known_via = dict(zip(KM.TF, KM["Known via"].astype(str)))
F = xl.parse("TF1000cells_known_motifs (Full)", header=None).iloc[3:]
cols = ["raw_ame_peaks", "raw_homer_peaks", "raw_ame_bins", "raw_homer_bins", "imp_ame_peaks", "imp_homer_peaks", "imp_ame_bins", "imp_homer_bins", "ocr_ame_peaks", "ocr_homer_peaks", "ocr_ame_bins", "ocr_homer_bins"]
OWN = {}
def num(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan
for r in F.itertuples(index=False):
    t = str(r[0]).strip().upper()
    if not t or t == "NAN" or " " in t: continue
    v = {c: num(r[2 + i]) for i, c in enumerate(cols)}
    ok = {c: (v[c] <= .05 if "ame" in c else v[c] <= 1e-5) if not np.isnan(v[c]) else False for c in cols}
    OWN[t] = dict(raw_best_p=np.nanmin([v[c] for c in cols[:4]] + [np.inf]), raw_pass=any(ok[c] for c in cols[:4]), imp_best_p=np.nanmin([v[c] for c in cols[4:8]] + [np.inf]), imp_pass=any(ok[c] for c in cols[4:8]), ocr_best_p=np.nanmin([v[c] for c in cols[8:]] + [np.inf]), ocr_pass=any(ok[c] for c in cols[8:]))
hoc = lambda t: any(s in H for s in ({"HP1": ["CBX1", "CBX3", "CBX5"], "ZUFSP": ["ZUP1"]}.get(t, [t])))
def m12(t):
    o = OWN.get(t); return dict(hocomoco=hoc(t), known_via=known_via.get(t, "NA"), dna_motif=known_via.get(t, "none") not in ("none", "NA"),
                                own_raw_p=o["raw_best_p"] if o else np.nan, own_imp_p=o["imp_best_p"] if o else np.nan, own_ocr_p=o["ocr_best_p"] if o else np.nan,
                                own_pass=(o["raw_pass"] or o["imp_pass"] or o["ocr_pass"]) if o else False, own_pass_sets=",".join(s for s in ("raw", "imp", "ocr") if o and o[f"{s}_pass"]) or ".")
# --- m3 pair-specific shared motif (pair_verdict.tsv)
V = pd.read_csv(f"{C}/pair_motif_lookup/pair_verdict.tsv", sep="\t", keep_default_na=False); VD = {key(a, b): r for (a, b), (_, r) in zip(zip(V.a, V.b), V.iterrows())}
SETS = ["A_raw_homer", "A_raw_ame", "A_raw_denovo", "B_imp_peaks_homer", "B_imp_peaks_ame", "B_imp_peaks_homer_ocr", "B_imp_peaks_ame_ocr", "C_imp_bins_homer", "C_imp_bins_ame", "E_imp_denovo_unknown", "E_imp_denovo_peaks"]
bg = pd.DataFrame([dict(set=s, n_pairs=len(V), n_pair_specific=int((V[s] == "pair-specific shared").sum()), frac_pair_specific=round((V[s] == "pair-specific shared").mean(), 4),
                        n_testable=int((V[s] != "untestable").sum()), frac_of_testable=round((V[s] == "pair-specific shared").sum() / max((V[s] != "untestable").sum(), 1), 4)) for s in SETS])
bg.to_csv(f"{OUT}/set_background.tsv", sep="\t", index=False)
def m3(k):
    if k not in VD: return dict(m3_raw_sets="NA", m3_imp_sets="NA", m3_raw_motifs="NA")
    r = VD[k]; ps = [s for s in SETS if r[s] == "pair-specific shared"]; raw = [s for s in ps if s.startswith("A_raw")]
    return dict(m3_raw_sets=",".join(raw) or ".", m3_imp_sets=",".join(s for s in ps if not s.startswith("A_raw")) or ".", m3_raw_motifs=" | ".join(f"{s}:{r[s + '_top_specific']}" for s in raw if r.get(s + "_top_specific", "")) or ".")
# --- universe
best = lambda k: max(float(m.loc[k[0], k[1]]) for m in M.values()) if k[0] in M["all"].index and k[1] in M["all"].index else np.nan
allk = set(a1) | set(a5)
U = sorted({k for k in allk if rk[af3(k)[0]["tier"]] >= 1 and best(k) >= .60} | set(REF))
rows = []
for k in U:
    a, b = k; ok = a in M["all"].index and b in M["all"].index; pc = {m: (round(float(M[m].loc[a, b]), 4) if ok else np.nan) for m in MATS}; bp = max(pc.values()) if ok else np.nan
    bst, s1, s5 = af3(k); r1 = RD.get(k); ma, mb = m12(a), m12(b)
    row = dict(pair=f"{a}-{b}", reference=k in REF, best_pct=bp, best_matrix=max(pc, key=pc.get) if ok else "NA", **{f"pct_{m}": v for m, v in pc.items()}, n_mat_ge080=sum(v >= .8 for v in pc.values()) if ok else np.nan,
               cobind_depth_matched_pct=r1.view1_score if r1 is not None else np.nan, cicero_excess_z=r1.cicero_excess_z if r1 is not None else np.nan,
               af3_tier=bst["tier"] if bst else "not_run", af3_ipsae=bst["ipsae"] if bst else np.nan, af3_iptm=bst["iptm"] if bst else np.nan, af3_seeds=(5 if s5 else 1) if bst else 0, af3_source=bst["src"] if bst else ".",
               af3_s1_tier=s1["tier"] if s1 else ".", af3_s5_tier=s5["tier"] if s5 else "NA", af3_s5_seeds_strong=s5["seeds_strong"] if s5 else np.nan,
               boltz_trimmed=BZ.get(k, "not_screened"), boltz_fulllength=FLt.get(k, "NA"), hub_pair=(r1.hub_partner != ".") if r1 is not None else False, rbbp47=bool({"RBBP4", "RBBP7"} & set(k)),
               **{f"{x}_{c}": v for x, mm in (("a", ma), ("b", mb)) for c, v in mm.items()}, **m3(k),
               m4_capselex=r1.capselex if r1 is not None else "NA", m5_string_physical=r1.string_physical if r1 is not None and pd.notna(r1.string_physical) else ".", m5_r1_class=r1["class"] if r1 is not None else "NA")
    rows.append(row)
T = pd.DataFrame(rows)
pair_motif = (T.m3_raw_sets.ne(".") & T.m3_raw_sets.ne("NA")) | T.m4_capselex.astype(str).str.startswith("called")
dnaA, dnaB = T.a_dna_motif, T.b_dna_motif; passA, passB = T.a_own_pass, T.b_own_pass; hit = T.af3_tier.isin(["strong", "weak"]) & (T.best_pct >= .60)
t1 = (T.af3_tier == "strong") & (T.n_mat_ge080 >= 3) & pair_motif
t2 = hit & dnaA & dnaB & passA & passB
t3 = hit & ((dnaA & passA & ~dnaB) | (dnaB & passB & ~dnaA))
T["tier"] = np.select([t1, t2, t3, hit], ["T1", "T2", "T3", "AF3_only"], "reference_only")
T["pair_level_motif"] = pair_motif
T = T.sort_values(["tier", "af3_ipsae"], ascending=[True, False]); T.to_csv(f"{OUT}/triple_support.tsv", sep="\t", index=False, float_format="%.4g")
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
print("universe", len(T), "(reference rows", int(T.reference.sum()), ")"); print(T.groupby(["tier"]).size().to_string()); print(pd.crosstab(T.tier, T.af3_tier).to_string())
print(bg.to_string(index=False))
show = ["pair", "tier", "reference", "af3_tier", "af3_ipsae", "af3_iptm", "af3_seeds", "af3_s5_tier", "best_pct", "best_matrix", "n_mat_ge080", "cobind_depth_matched_pct", "cicero_excess_z", "boltz_trimmed", "boltz_fulllength", "hub_pair", "rbbp47",
        "a_dna_motif", "a_own_pass_sets", "b_dna_motif", "b_own_pass_sets", "m3_raw_sets", "m3_imp_sets", "m4_capselex", "m5_string_physical", "m5_r1_class"]
print(T[T.tier.isin(["T1", "T2", "T3"]) | T.reference][show].to_string(index=False))
print("pairs with pair-level motif (any tier):"); print(T[T.pair_level_motif][["pair", "tier", "af3_tier", "n_mat_ge080", "m3_raw_sets", "m3_raw_motifs", "m4_capselex"]].to_string(index=False))
tf = sorted({t for k in U for t in k}); print("TF motif status:"); print(pd.DataFrame([dict(tf=t, **m12(t)) for t in tf]).to_string(index=False))
