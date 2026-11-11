# m7_step2: write job lists. usage: make_jobs.py <raw|imp> <expr_tune|expr_final|kd_tune|kd_final>
import sys, os
import pandas as pd
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2")
GRID_N = {"enet": 15, "lgb": 12, "mlp": 12}; MODELS = ("enet", "lgb", "mlp"); P = ["pseudo%d" % d for d in range(5)]; M = ["mis%d" % d for d in range(5)]; kind, which = sys.argv[1:3]; O = "/work/users/d/y/dyy12/XuLab/regulatome/m7_step2/jobs"; os.makedirs(O, exist_ok=True)
C = ["target", "arm", "centring", "model", "mode", "cond", "split", "rung", "kind", "hp", "phase"]; rows = []
def add(*a): rows.append(dict(zip(C, a)))
if which == "expr_tune":
    for m in MODELS:
        for r in (1, 2, 3, 4):
            for h in range(GRID_N[m]): add("expr", "", "", m, "expr_only", "real", "gene", r, kind, h, "tune")
elif which == "expr_final":
    for m in MODELS:
        for r in (1, 2, 3, 4): add("expr", "", "", m, "expr_only", "real", "gene", r, kind, "frozen", "final")
        for r in (3, 4):
            for c in P: add("expr", "", "", m, "expr_only", c, "gene", r, kind, "frozen", "final")
elif which == "kd_tune":
    for m in MODELS:
        for mo in ("kd_only", "both"):
            for c in ("real", "base"):
                for h in range(GRID_N[m]): add("kd", "arm2", "train", m, mo, c, "gene", 0, kind, h, "tune")
elif which == "kd_final":
    combos = [("arm2", "train", c) for c in ["real", "base"] + P + M] + [("arm2", "raw", c) for c in ("real", "base")] + [("arm1", "train", c) for c in ("real", "base")] + [("arm1", "raw", c) for c in ("real", "base")]
    if kind == "imp": combos = [("arm2", "train", c) for c in ["real", "base"] + P + M] + [("arm1", "train", c) for c in ("real", "base")]
    for sp in ["gene"] + ["fold%d" % k for k in range(5)]:
        for arm, ce, c in combos:
            for m in MODELS:
                for mo in ("kd_only", "both"): add("kd", arm, ce, m, mo, c, sp, 0, kind, "frozen", "final")
J = pd.DataFrame(rows, columns=C); L = os.environ.get("M7_LINK", "powerlaw"); f = f"{O}/{(L + chr(95)) if L != 'powerlaw' else ''}{kind}_{which}.tsv"; J.to_csv(f, sep="\t", index=False); print(f, len(J))
