# m46 Amendment 4: collect the all_universe bulk benchmark (same parsing as downstream/build_slide_all_universe.py) for raw + masks.
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/downstream"); import build_slide_all_universe as B
X = Path("/work/users/d/y/dyy12/XuLab"); BD = X / "regulatome/m46_stage1_imputed/work/bench"; EB = X / "downstream/eval/bins"; O = X / "regulatome/m46_stage1_imputed/out"
REF = {"ctcf": "HEK293 (ENCODE ENCFF139DCW, CTCF_HEK293 peaks)", "ctcf_maj": "HEK293 + HEK293T consensus (majority 2 of 3)", "maz": "HEK293 (ENCODE ENCFF529XTF)", "zic2": "HEK293 (ENCODE ENCFF359GIY)",
       "zbtb7a": "HEK293 (ENCODE ENCFF114DGV)", "znf777": "HEK293 (ENCODE ENCFF735CAY)", "znf282": "HEK293T (GSM2466515)", "nfya": "K562 (ENCODE ENCFF765GYT); not HEK, reported separately"}
ORDER = ["raw", "Mpeak", "M0", "Mdrop5", "Mzero", "M5", "M50"]; rows = []
for tf, mode in [("ctcf", "all_universe"), ("ctcf", "maj_zeroneg"), ("maz", "all_universe"), ("zic2", "all_universe"), ("zbtb7a", "all_universe"), ("znf777", "all_universe"), ("znf282", "all_universe"), ("nfya", "all_universe")]:
    tag = f"{tf}_{mode}"; cmp_ = B.load_json(BD / f"compare_pos_neg_{tag}" / f"compare_pos_neg_{tag}.json"); binj = B.load_json(BD / f"bin_coverage_{tag}" / f"bin_coverage_{tag}.json")
    pt = BD / f"peak_coverage_{tag}" / f"peak_coverage_{tag}_table.txt"; peak = B.parse_peak_table(pt); umi = {}
    if pt.is_file():
        for line in pt.read_text().splitlines()[2:]:
            p = re.split(r"\s{2,}", line.strip())
            if len(p) > 2 and not line.startswith("-"): umi[p[0]] = p[2]
    au = {r["label"]: float(r.get("auroc_full", r.get("auroc"))) for r in cmp_.get("inputs", []) if isinstance(r, dict) and r.get("label")}
    bc = {r["label"]: r for r in binj.get("inputs", []) if isinstance(r, dict) and r.get("label")}
    lab = pd.read_csv(EB / f"bins_{tf}_{mode}" / "pos_neg_bins.tsv", sep="\t", usecols=["label"]).label.value_counts()
    for p in ORDER:
        b = bc.get(p, {}); pk = peak.get(p, {})
        rows.append({"TF": tf.upper(), "positive set": "CTCF consensus (maj 2/3), zero-read negatives" if mode == "maj_zeroneg" else "all_universe", "bulk reference": REF["ctcf_maj" if mode == "maj_zeroneg" else tf],
                     "Pipeline": "unified (Mpeak, current)" if p == "Mpeak" else p, "n_pos": int(lab.get("pos", 0)), "n_neg": int(lab.get("neg", 0)), "AUROC": au.get(p),
                     "Bin.pos %": 100 * float((b.get("pos") or {}).get("coverage", "nan")), "Bin.neg %": 100 * float((b.get("neg") or {}).get("coverage", "nan")),
                     "Peak.pos %": float(pk["pos_in_peaks_pct"]) if pk.get("pos_in_peaks_pct") else None, "Peak.neg %": float(pk["neg_in_peaks_pct"]) if pk.get("neg_in_peaks_pct") else None, "Mean frags/cell": umi.get(p, "")})
T = pd.DataFrame(rows); T.to_csv(O / "18_bulk_benchmark_masks.tsv", sep="\t", index=False, float_format="%.4g")
with pd.ExcelWriter(O / "18_bulk_benchmark_masks.xlsx") as w:
    T[T.TF != "NFYA"].to_excel(w, sheet_name="HEK293T and HEK293 references", index=False); T[T.TF == "NFYA"].to_excel(w, sheet_name="NFYA (K562 reference)", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); print(T.drop(columns=["bulk reference", "positive set"]).round(4).to_string(index=False)); print("done")
