# All novel correlation-only candidates (best-of-five percentile >= 0.60, same exclusions as the slide 1 table), minus pairs already run in AF3, high to low percentile. HP1 = the CBX paralog with the best Boltz score. Writes list.tsv
import pandas as pd, sys
src = open("summary_table_data.py").read(); head = src[:src.index("N = []\n")].replace('C.to_csv("data/summary_current_values.tsv", sep="\\t", index=False)', "pass").replace("print(", "(lambda *a, **k: None)(")
g = {}; exec(head, g); S, tfs, key, KN, strg, cap, stag, CUR = [g[x] for x in "S tfs key KN strg cap stag CUR".split()]; pn = g["pn"]
nov = {key(*p) for nm, (c, ps) in CUR.items() if c == "Novel" for p in ps}; A = sys.argv[1]
done = set()
for f in ("complex_screen/af3_local/af3_scores.tsv", "complex_screen/af3_novel_20261103/af3_scores.tsv", "complex_screen/af3_novel_20261103/af3_scores_weak.tsv", "complex_screen/af3_novel_20261103/af3_scores_srcap.tsv"):
    for p in pd.read_csv(f"{A}/{f}", sep="\t").pair: x, y = p.split("__"); done.add(key(pn(x), pn(y)))
Sc = pd.read_csv(f"{A}/complex_screen/screen/scores.tsv", sep="\t", keep_default_na=False); Sc = Sc[Sc["set"] != "homo"]; hp = {}
for r in Sc.itertuples():
    x, y = r.pair_id.split("__")
    for u, v in ((x, y), (y, x)):
        if u in ("CBX1", "CBX3", "CBX5"):
            v = pn(v)
            k = (float(r.ipsae_max), float(r.iptm_top))
            if v not in hp or k > hp[v][0]: hp[v] = (k, u)
R = []; n = 0
for i, a in enumerate(tfs):
    for b in tfs[i + 1:]:
        k = key(a, b)
        if (k in KN and k not in nov) or k in strg or k in cap or k in stag or "NEUROD4" in k: continue
        v = {s: float(m.loc[a, b]) for s, m in S.items()}; s = max(v, key=v.get)
        if v[s] < .60: continue
        n += 1
        if k in done: continue
        u, w = k
        if u == "HP1": u = hp[w][1]
        if w == "HP1": w = hp[u][1]
        R.append(dict(a=u.replace("ZUFSP", "ZUP1"), b=w.replace("ZUFSP", "ZUP1"), panel=f"{k[0]}-{k[1]}", value=round(v[s], 4), set=s))
T = pd.DataFrame(R).sort_values("value", ascending=False); T.to_csv(sys.argv[2], sep="\t", index=False); print("eligible >=.60", n, "already run", n - len(T), "to run", len(T)); print(T.head(3).to_string()); print(T.panel[T.panel.str.contains("HP1")].shape[0], T[T.panel.str.contains("HP1")].a.value_counts().to_dict(), T[T.panel.str.contains("HP1")].b.value_counts().to_dict())
