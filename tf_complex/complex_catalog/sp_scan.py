# Known complexes scanned in Spearman (primary) and Pearson TF x TF matrices.
# Input: sp_mats.json from spearman_pearson_matrices.py (Longleaf tf_complex/results_spearman_pearson/).
# Output: sp_scan.json (per matrix x complex: mean r, mean percentile, positions in the dendrogram order).
import json, sys
import numpy as np
exec(open("matfig.py").read().split('if __name__=="__main__":')[0])
exec(open("known_def.py").read())
VER = sys.argv[1] if len(sys.argv) > 1 else "spqn"      # "raw" or "spqn"
J = json.load(open("sp_mats.json"))
TFS = J["tfs"]
SP_MATS = [("all", "ChromHMM: all elements"), ("noquies", "ChromHMM: quiescent removed"),
           ("tss", "ChromHMM: TSS only"), ("enhancer", "ChromHMM: enhancer only"),
           ("wang18", "ChromHMM-18 states (Dr. Wang)")]


def mat(mk, metric):
    m = J["mats"][mk]
    # Dr. Wang's 18-state matrix has no depth confound (corr with cell count ~0.04), so it stays raw as in the original
    return np.array(m[metric + ("_spqn" if VER == "spqn" and mk != "wang18" else "")])


def pct(R):
    n = len(R); off = ~np.eye(n, dtype=bool)
    v = np.sort(R[off])
    P = np.searchsorted(v, R, side="right") / len(v)
    np.fill_diagonal(P, 1.0)
    return P


def scan():
    out = {"ver": VER, "tfs": TFS, "mats": {}}
    for mk, _ in SP_MATS:
        out["mats"][mk] = {}
        for metric in ("spearman", "pearson"):
            R = mat(mk, metric); P = pct(R); o = order_of(R)
            Lo = [TFS[i] for i in o]
            res = {"order": Lo, "complex": {}}
            for k, name, mem in KNOWN:
                idx = [TFS.index(t) for t in mem]
                pr = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]
                pos = sorted(Lo.index(t) for t in mem)
                # partner rank: for each member, rank of its best fellow member among all 78 partners
                nn = []
                for a in idx:
                    others = np.argsort(-np.where(np.arange(len(TFS)) == a, -9, R[a]))
                    nn.append(min(list(others).index(b) + 1 for b in idx if b != a))
                res["complex"][k] = {
                    "r": float(np.mean([R[a, b] for a, b in pr])),
                    "pct": float(np.mean([P[a, b] for a, b in pr])),
                    "pairs": {f"{TFS[a]}-{TFS[b]}": [round(float(R[a, b]), 4), round(float(P[a, b]), 4)] for a, b in pr},
                    "pos": pos, "span": pos[-1] - pos[0], "adj": pos[-1] - pos[0] == len(pos) - 1,
                    "best_partner_rank": nn,
                }
            out["mats"][mk][metric] = res
    return out


if __name__ == "__main__":
    S = scan()
    json.dump(S, open(f"sp_scan_{VER}.json", "w"))
    for mk, mt in SP_MATS:
        print("==", mt)
        for k, name, mem in KNOWN:
            s = S["mats"][mk]["spearman"]["complex"][k]; p = S["mats"][mk]["pearson"]["complex"][k]
            print(f"  {name:28s} S r={s['r']:+.3f} pct={s['pct']:.3f} span={s['span']:2d} nn={s['best_partner_rank']}"
                  f" | P r={p['r']:+.3f} pct={p['pct']:.3f} span={p['span']:2d}")
