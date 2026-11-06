import glob, json, pandas as pd
own = json.load(open("db/own.json"))["own"]; rows = []
for f in sorted(glob.glob("diag/*/*/sea/sea.tsv")):
    p = f.split("/"); d = pd.read_csv(f, sep="\t", comment="#"); d = d[d.ID.notna()]; t = p[2]; s = d[d.ID.isin(own.get(t.upper(), []))]; rows.append(dict(arm=p[1], tf=t, n_sig=int((d.QVALUE < .05).sum()), own_best_q=s.QVALUE.min(), own_rank=int(s.RANK.min()) if len(s) else -1, enr=round(s.sort_values("QVALUE").ENR_RATIO.iloc[0], 2) if len(s) else None, top3=", ".join(d.ALT_ID.fillna(d.ID).head(3))))
    if t in ("mlx", "srebf2"): rows[-1]["ebox_sre"] = "; ".join(f"{r.ID} q={r.QVALUE:.2g} r={r.ENR_RATIO}" for r in d[d.ID.isin(["MA0663.1", "MA0828.3", "MA0596.1"])].itertuples())
T = pd.DataFrame(rows); pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 80); print(T.to_string(index=False)); T.to_csv("diag/diag_all_summary.tsv", sep="\t", index=False)
